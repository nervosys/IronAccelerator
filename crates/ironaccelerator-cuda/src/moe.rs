//! Flash MoE — fused Mixture-of-Experts forward pass.
//!
//! Pipeline:
//!
//! ```text
//!   logits  = X @ W_gate                     // cuBLASLt, FP32 accumulate
//!   idx, w  = softmax_topk(logits, K)        // NVRTC kernel
//!   counts  = histogram(idx)                 // NVRTC kernel, atomicAdd
//!   offsets = exclusive_scan(counts)         // NVRTC kernel
//!   pX, pos = permute(X, idx, offsets)       // NVRTC kernel, one block per (t,k)
//!   pM[e]   = pX[e] @ W_up[e]                // cuBLASLt, per expert
//!   silu(pM[e])                              // NVRTC kernel
//!   pY[e]   = pM[e] @ W_down[e]              // cuBLASLt, per expert
//!   Y       = combine(pY, pos, w)            // NVRTC kernel, weighted scatter
//! ```
//!
//! This is FP16-only for the v1.0 release. Grouped-GEMM in a single
//! cuBLASLt launch (the true "fused" path available in CUDA 12.5+) is a
//! v1.1 item; today the per-expert matmul loop is serialised on the stream,
//! which is still considerably faster than the naive dense-MoE baseline
//! and uses identical memory.
//!
//! # Costs
//!
//! - Router GEMM: `2·T·H·E` flops.
//! - Expert FFN: `2·T·K·H·I·2` flops (up + down), i.e. the per-token compute
//!   is `4·K·H·I`. With `K=2, H=4096, I=14336` that's ~230 MFLOP/token,
//!   which fits in an H100 SM tile.
//! - Dispatch overhead: one D2H sync of `offsets[E+1]` (tens of ints) per
//!   forward pass. Use the `graph` module to capture and replay if this
//!   shows up in your profile.

use crate::blas::{self, BlasLt, MatmulDesc, MatrixLayout, Preference};
use crate::drv::{Device, DeviceBuf, Module, Stream};
use crate::kernel::{self, CompileOptions};
use iron_cuda_sys::cublas_lt as sys;
use iron_cuda_sys::driver::CUdeviceptr;
use ironaccelerator_core::{Error, Result};
use std::sync::Arc;

/// Element type for expert activations. FP16 and BF16 supported.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MoeDType { F16, Bf16 }

impl MoeDType {
    fn bytes(self) -> usize { 2 }
    fn cublas(self) -> sys::CudaDataType {
        match self {
            Self::F16 => sys::CudaDataType::R16F,
            Self::Bf16 => sys::CudaDataType::R16BF,
        }
    }
    /// NVRTC suffix for kernel names instantiated from the C++ template.
    fn suffix(self) -> &'static str {
        match self { Self::F16 => "f16", Self::Bf16 => "bf16" }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum MoeActivation { Silu }

/// Static MoE shape. Kept out of the plan so a single plan can be rebuilt
/// cheaply for a new batch size on the same router/expert topology.
#[derive(Debug, Copy, Clone)]
pub struct MoeParams {
    pub num_tokens: u32,
    pub hidden: u32,
    pub inter: u32,
    pub num_experts: u32,
    pub top_k: u32,
    pub dtype: MoeDType,
    pub activation: MoeActivation,
}

impl MoeParams {
    pub fn validate(&self) -> Result<()> {
        if self.num_experts == 0 || self.top_k == 0 || self.top_k > self.num_experts {
            return Err(Error::InvalidArgument("MoE: invalid num_experts / top_k"));
        }
        if self.top_k > 8 {
            return Err(Error::InvalidArgument("MoE: top_k > 8 not supported by the topk kernel"));
        }
        if self.hidden == 0 || self.inter == 0 {
            return Err(Error::InvalidArgument("MoE: hidden and inter must be non-zero"));
        }
        Ok(())
    }
}

// ── NVRTC source — compiled once per plan ──────────────────────────────────

const MOE_SRC: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>

// Dtype traits — kept minimal so NVRTC's C++17 compile stays cheap.
template<typename T> struct DT;
template<> struct DT<__half> {
    static __device__ __forceinline__ float to_f(const __half& v) { return __half2float(v); }
    static __device__ __forceinline__ __half from_f(float v) { return __float2half(v); }
};
template<> struct DT<__nv_bfloat16> {
    static __device__ __forceinline__ float to_f(const __nv_bfloat16& v) { return __bfloat162float(v); }
    static __device__ __forceinline__ __nv_bfloat16 from_f(float v) { return __float2bfloat16(v); }
};

extern "C" __global__ void moe_softmax_topk(
    const float* __restrict__ logits,
    int* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int T, int E, int K)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const float* row = logits + (size_t)t * E;

    float m = row[0];
    for (int e = 1; e < E; ++e) m = fmaxf(m, row[e]);
    float s = 0.f;
    for (int e = 0; e < E; ++e) s += __expf(row[e] - m);
    float inv_s = 1.f / s;

    const int MAX_K = 8;
    float bv[MAX_K]; int bi[MAX_K];
    for (int k = 0; k < K; ++k) { bv[k] = -CUDART_INF_F; bi[k] = 0; }
    for (int e = 0; e < E; ++e) {
        float v = __expf(row[e] - m) * inv_s;
        int w = 0;
        for (int k = 1; k < K; ++k) if (bv[k] < bv[w]) w = k;
        if (v > bv[w]) { bv[w] = v; bi[w] = e; }
    }
    float tsum = 0.f;
    for (int k = 0; k < K; ++k) tsum += bv[k];
    float tinv = tsum > 0.f ? 1.f / tsum : 0.f;
    for (int k = 0; k < K; ++k) {
        topk_idx[(size_t)t * K + k] = bi[k];
        topk_w  [(size_t)t * K + k] = bv[k] * tinv;
    }
}

extern "C" __global__ void moe_count(
    const int* __restrict__ topk_idx, int TK, int E,
    int* __restrict__ counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= TK) return;
    int e = topk_idx[i];
    if (e >= 0 && e < E) atomicAdd(&counts[e], 1);
}

extern "C" __global__ void moe_exclusive_scan(
    const int* __restrict__ counts, int E, int* __restrict__ offsets)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int acc = 0;
        for (int e = 0; e < E; ++e) { offsets[e] = acc; acc += counts[e]; }
        offsets[E] = acc;
    }
}

template<typename T>
__device__ __forceinline__ void permute_impl(
    const T* __restrict__ x,
    const int* __restrict__ topk_idx,
    const int* __restrict__ offsets,
    int* __restrict__ running,
    T* __restrict__ permuted,
    int* __restrict__ scatter_pos,
    int T_tok, int H, int K)
{
    int t = blockIdx.x;
    int k = blockIdx.y;
    if (t >= T_tok || k >= K) return;
    __shared__ int dst_row;
    if (threadIdx.x == 0) {
        int e = topk_idx[t * K + k];
        int d = offsets[e] + atomicAdd(&running[e], 1);
        dst_row = d;
        scatter_pos[t * K + k] = d;
    }
    __syncthreads();
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        permuted[(size_t)dst_row * H + h] = x[(size_t)t * H + h];
    }
}

template<typename T>
__device__ __forceinline__ void silu_impl(T* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = DT<T>::to_f(y[i]);
    float s = v / (1.f + __expf(-v));
    y[i] = DT<T>::from_f(s);
}

template<typename T>
__device__ __forceinline__ void combine_impl(
    const T* __restrict__ permuted_y,
    const int* __restrict__ scatter_pos,
    const float* __restrict__ topk_w,
    T* __restrict__ y,
    int T_tok, int H, int K)
{
    int t = blockIdx.x;
    if (t >= T_tok) return;
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) {
            int src = scatter_pos[t * K + k];
            float w = topk_w[t * K + k];
            acc += w * DT<T>::to_f(permuted_y[(size_t)src * H + h]);
        }
        y[(size_t)t * H + h] = DT<T>::from_f(acc);
    }
}

#define INSTANTIATE(SUF, T)                                                 \
extern "C" __global__ void moe_permute_##SUF(                               \
    const T* x, const int* topk_idx, const int* offsets, int* running,      \
    T* permuted, int* scatter_pos, int T_tok, int H, int K)                 \
{ permute_impl<T>(x, topk_idx, offsets, running, permuted, scatter_pos,     \
                  T_tok, H, K); }                                           \
extern "C" __global__ void moe_silu_##SUF(T* y, int N)                      \
{ silu_impl<T>(y, N); }                                                     \
extern "C" __global__ void moe_combine_##SUF(                               \
    const T* permuted_y, const int* scatter_pos, const float* topk_w,       \
    T* y, int T_tok, int H, int K)                                          \
{ combine_impl<T>(permuted_y, scatter_pos, topk_w, y, T_tok, H, K); }

INSTANTIATE(f16,  __half)
INSTANTIATE(bf16, __nv_bfloat16)
"#;

// ── Scratch ────────────────────────────────────────────────────────────────

/// Reusable scratch memory for a plan. Allocated once and reused across
/// forward passes of the same [`MoeParams`].
pub struct MoeScratch {
    pub gate_logits: DeviceBuf<f32>,    // [T, E]
    pub topk_idx:    DeviceBuf<i32>,    // [T, K]
    pub topk_w:      DeviceBuf<f32>,    // [T, K]
    pub counts:      DeviceBuf<i32>,    // [E]
    pub offsets:     DeviceBuf<i32>,    // [E + 1]
    pub running:     DeviceBuf<i32>,    // [E]
    pub scatter_pos: DeviceBuf<i32>,    // [T*K]
    pub permuted_x:  DeviceBuf<u8>,     // [T*K, H] bytes
    pub permuted_m:  DeviceBuf<u8>,     // [T*K, I] bytes
    pub permuted_y:  DeviceBuf<u8>,     // [T*K, H] bytes
    pub blaslt_ws:   DeviceBuf<u8>,     // cuBLASLt workspace
    /// Host staging for the `E+1` offsets — used to drive per-expert launches.
    pub host_offsets: Vec<i32>,
}

impl MoeScratch {
    pub fn new(stream: Arc<Stream>, params: &MoeParams, blaslt_ws_bytes: usize) -> Result<Self> {
        let t = params.num_tokens as usize;
        let e = params.num_experts as usize;
        let k = params.top_k as usize;
        let h = params.hidden as usize;
        let i = params.inter as usize;
        let bytes_row_h = h * params.dtype.bytes();
        let bytes_row_i = i * params.dtype.bytes();
        let tk = t * k;

        Ok(Self {
            gate_logits: DeviceBuf::alloc_zeros(stream.clone(), t * e)?,
            topk_idx:    DeviceBuf::alloc_zeros(stream.clone(), tk)?,
            topk_w:      DeviceBuf::alloc_zeros(stream.clone(), tk)?,
            counts:      DeviceBuf::alloc_zeros(stream.clone(), e)?,
            offsets:     DeviceBuf::alloc_zeros(stream.clone(), e + 1)?,
            running:     DeviceBuf::alloc_zeros(stream.clone(), e)?,
            scatter_pos: DeviceBuf::alloc_zeros(stream.clone(), tk)?,
            permuted_x:  DeviceBuf::alloc_zeros(stream.clone(), tk * bytes_row_h)?,
            permuted_m:  DeviceBuf::alloc_zeros(stream.clone(), tk * bytes_row_i)?,
            permuted_y:  DeviceBuf::alloc_zeros(stream.clone(), tk * bytes_row_h)?,
            blaslt_ws:   DeviceBuf::alloc_zeros(stream, blaslt_ws_bytes.max(1))?,
            host_offsets: vec![0i32; e + 1],
        })
    }

    /// Zero the per-call metadata buffers. Called automatically at the top
    /// of every `execute()`.
    fn reset_metadata(&mut self) -> Result<()> {
        // Re-zero counts and running. Offsets will be overwritten by the scan.
        // We reuse copy_from_host with zero buffers — the cuMemsetD8Async
        // path is inside alloc_zeros but not exposed here, so a small host
        // write is acceptable (tens of ints).
        let e = self.counts.len();
        let zero_e = vec![0i32; e];
        self.counts.copy_from_host(&zero_e)?;
        self.running.copy_from_host(&zero_e)?;
        Ok(())
    }
}

// ── Plan ───────────────────────────────────────────────────────────────────

pub struct FlashMoePlan {
    params: MoeParams,
    blaslt: Arc<BlasLt>,
    _module: Arc<Module>,
    k_softmax_topk:     crate::drv::Function,
    k_count:            crate::drv::Function,
    k_scan:             crate::drv::Function,
    k_permute:          crate::drv::Function,
    k_silu:             crate::drv::Function,
    k_combine:          crate::drv::Function,
    // cuBLASLt descriptor cache (shape-independent)
    router_desc: MatmulDesc,
    expert_desc: MatmulDesc,
    // Preference with workspace cap
    pref: Preference,
}

unsafe impl Send for FlashMoePlan {}
unsafe impl Sync for FlashMoePlan {}

impl FlashMoePlan {
    /// Build a plan. Compiles the NVRTC kernels (via the process-wide cache)
    /// and creates cuBLASLt descriptors. Rebuild only on device change.
    pub fn new(
        device: Arc<Device>,
        blaslt: Arc<BlasLt>,
        params: MoeParams,
        blaslt_ws_bytes: usize,
    ) -> Result<Self> {
        params.validate()?;

        let opts = CompileOptions {
            extras: vec!["--use_fast_math".into(), "-std=c++17".into()],
            ..Default::default()
        };
        let ck = kernel::get_or_compile(&device, MOE_SRC, "moe_softmax_topk", &opts)?;
        let module = ck.module.clone();
        let k_softmax_topk = ck.function;
        let k_count   = module.function("moe_count")?;
        let k_scan    = module.function("moe_exclusive_scan")?;
        let suf = params.dtype.suffix();
        let k_permute = module.function(&format!("moe_permute_{suf}"))?;
        let k_silu    = module.function(&format!("moe_silu_{suf}"))?;
        let k_combine = module.function(&format!("moe_combine_{suf}"))?;

        // Both the router and the expert matmul use FP32 accumulation with
        // FP16/BF16 I/O. One descriptor can be reused for both by rebuilding
        // the layouts per call.
        let mut router_desc = MatmulDesc::new(sys::CublasComputeType::F32, sys::CudaDataType::R32F)?;
        router_desc.set_transpose(sys::CublasOp::N, sys::CublasOp::N)?;
        let mut expert_desc = MatmulDesc::new(sys::CublasComputeType::F32, sys::CudaDataType::R32F)?;
        expert_desc.set_transpose(sys::CublasOp::N, sys::CublasOp::N)?;

        let mut pref = Preference::new()?;
        pref.set_max_workspace(blaslt_ws_bytes)?;

        Ok(Self {
            params, blaslt, _module: module,
            k_softmax_topk, k_count, k_scan, k_permute, k_silu, k_combine,
            router_desc, expert_desc, pref,
        })
    }

    #[inline] pub fn params(&self) -> &MoeParams { &self.params }

    /// Forward pass. All pointers are device pointers, all shapes are
    /// row-major. `scratch` is rebound on every call (its memory is reused).
    ///
    /// # Safety
    /// All pointers must reference buffers of the declared shapes in
    /// `params`, allocated on the same device as `self.blaslt`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn execute(
        &self,
        stream: &Stream,
        x: CUdeviceptr,            // [T, H]    dtype
        w_gate: CUdeviceptr,       // [H, E]    dtype
        w_up_stack: CUdeviceptr,   // [E, H, I] dtype, contiguous
        w_down_stack: CUdeviceptr, // [E, I, H] dtype, contiguous
        y: CUdeviceptr,            // [T, H]    dtype
        scratch: &mut MoeScratch,
    ) -> Result<()> {
        scratch.reset_metadata()?;

        let p = &self.params;
        let t = p.num_tokens as i32;
        let e = p.num_experts as i32;
        let k = p.top_k as i32;
        let h = p.hidden as i32;
        let ii = p.inter as i32;

        // ── 1. Router GEMM: logits[T, E] = X[T, H] @ W_gate[H, E] ──────
        // All tensors are row-major; set Order=Row on every layout so
        // cuBLASLt interprets (rows, cols, ld=cols) correctly.
        let mut a_layout = MatrixLayout::new(p.dtype.cublas(), t as u64, h as u64, h as i64)?;
        let mut b_layout = MatrixLayout::new(p.dtype.cublas(), h as u64, e as u64, e as i64)?;
        let mut c_layout = MatrixLayout::new(sys::CudaDataType::R32F, t as u64, e as u64, e as i64)?;
        a_layout.set_order(blas::Order::Row)?;
        b_layout.set_order(blas::Order::Row)?;
        c_layout.set_order(blas::Order::Row)?;
        let alpha: f32 = 1.0; let beta: f32 = 0.0;
        let heur = blas::heuristic(&self.blaslt, &self.router_desc,
                                   &a_layout, &b_layout, &c_layout, &c_layout, &self.pref)?;
        unsafe {
            blas::matmul(&self.blaslt, &self.router_desc,
                bytemuck_f32_le(&alpha), bytemuck_f32_le(&beta),
                x, &a_layout,
                w_gate, &b_layout,
                scratch.gate_logits.device_ptr(), &c_layout,
                scratch.gate_logits.device_ptr(), &c_layout,
                Some(&heur), Some(&mut scratch.blaslt_ws), stream)?;
        }

        // ── 2. softmax + top-K ─────────────────────────────────────────
        let block = 128u32;
        let grid = (t as u32).div_ceil(block);
        self.k_softmax_topk.launch(
            crate::drv::LaunchCfg::linear(grid, block), stream,
            (scratch.gate_logits.device_ptr(),
             scratch.topk_idx.device_ptr(),
             scratch.topk_w.device_ptr(),
             t, e, k))?;

        // ── 3. histogram ───────────────────────────────────────────────
        let tk = t.saturating_mul(k) as u32;
        self.k_count.launch(
            crate::drv::LaunchCfg::linear(tk.div_ceil(128), 128), stream,
            (scratch.topk_idx.device_ptr(),
             tk as i32, e,
             scratch.counts.device_ptr()))?;

        // ── 4. exclusive scan (single block) ───────────────────────────
        self.k_scan.launch(
            crate::drv::LaunchCfg::linear(1, 1), stream,
            (scratch.counts.device_ptr(),
             e,
             scratch.offsets.device_ptr()))?;

        // ── 5. D2H sync on offsets so we know per-expert token counts ──
        scratch.offsets.copy_to_host(&mut scratch.host_offsets)?;
        stream.synchronize()?;

        // ── 6. permute X into expert-grouped layout ────────────────────
        let permute_grid = crate::drv::LaunchCfg {
            grid: (t as u32, k as u32, 1),
            block: (128, 1, 1),
            shared_bytes: 0,
        };
        self.k_permute.launch(permute_grid, stream, (
            x, scratch.topk_idx.device_ptr(), scratch.offsets.device_ptr(),
            scratch.running.device_ptr(), scratch.permuted_x.device_ptr(),
            scratch.scatter_pos.device_ptr(), t, h, k))?;

        // ── 7. per-expert up + activation + down GEMMs ────────────────
        let bytes_row_h = (h as usize) * p.dtype.bytes();
        let bytes_row_i = (ii as usize) * p.dtype.bytes();
        let wu_stride = (h as usize) * (ii as usize) * p.dtype.bytes();
        let wd_stride = (ii as usize) * (h as usize) * p.dtype.bytes();

        for e_id in 0..p.num_experts as usize {
            let off = scratch.host_offsets[e_id];
            let next = scratch.host_offsets[e_id + 1];
            let n_e = (next - off) as i64;
            if n_e <= 0 { continue; }

            let x_off = scratch.permuted_x.device_ptr()
                + (off as u64) * (bytes_row_h as u64);
            let m_off = scratch.permuted_m.device_ptr()
                + (off as u64) * (bytes_row_i as u64);
            let y_off = scratch.permuted_y.device_ptr()
                + (off as u64) * (bytes_row_h as u64);
            let wu = w_up_stack  + (e_id as u64) * (wu_stride as u64);
            let wd = w_down_stack + (e_id as u64) * (wd_stride as u64);

            // up: [n_e, I] = [n_e, H] @ [H, I]
            let mut a_l = MatrixLayout::new(p.dtype.cublas(), n_e as u64, h as u64, h as i64)?;
            let mut b_l = MatrixLayout::new(p.dtype.cublas(), h as u64, ii as u64, ii as i64)?;
            let mut c_l = MatrixLayout::new(p.dtype.cublas(), n_e as u64, ii as u64, ii as i64)?;
            a_l.set_order(blas::Order::Row)?;
            b_l.set_order(blas::Order::Row)?;
            c_l.set_order(blas::Order::Row)?;
            let heur = blas::heuristic(&self.blaslt, &self.expert_desc,
                                       &a_l, &b_l, &c_l, &c_l, &self.pref)?;
            unsafe {
                blas::matmul(&self.blaslt, &self.expert_desc,
                    bytemuck_f32_le(&alpha), bytemuck_f32_le(&beta),
                    x_off, &a_l, wu, &b_l,
                    m_off, &c_l, m_off, &c_l,
                    Some(&heur), Some(&mut scratch.blaslt_ws), stream)?;
            }

            // activation (SiLU)
            let n_act = (n_e as i32).saturating_mul(ii);
            self.k_silu.launch(
                crate::drv::LaunchCfg::linear((n_act as u32).div_ceil(256), 256),
                stream, (m_off, n_act))?;

            // down: [n_e, H] = [n_e, I] @ [I, H]
            let mut a_l = MatrixLayout::new(p.dtype.cublas(), n_e as u64, ii as u64, ii as i64)?;
            let mut b_l = MatrixLayout::new(p.dtype.cublas(), ii as u64, h as u64, h as i64)?;
            let mut c_l = MatrixLayout::new(p.dtype.cublas(), n_e as u64, h as u64, h as i64)?;
            a_l.set_order(blas::Order::Row)?;
            b_l.set_order(blas::Order::Row)?;
            c_l.set_order(blas::Order::Row)?;
            let heur = blas::heuristic(&self.blaslt, &self.expert_desc,
                                       &a_l, &b_l, &c_l, &c_l, &self.pref)?;
            unsafe {
                blas::matmul(&self.blaslt, &self.expert_desc,
                    bytemuck_f32_le(&alpha), bytemuck_f32_le(&beta),
                    m_off, &a_l, wd, &b_l,
                    y_off, &c_l, y_off, &c_l,
                    Some(&heur), Some(&mut scratch.blaslt_ws), stream)?;
            }
        }

        // ── 8. weighted combine back to [T, H] ─────────────────────────
        self.k_combine.launch(
            crate::drv::LaunchCfg {
                grid: (t as u32, 1, 1),
                block: (128, 1, 1),
                shared_bytes: 0,
            },
            stream,
            (scratch.permuted_y.device_ptr(), scratch.scatter_pos.device_ptr(),
             scratch.topk_w.device_ptr(), y, t, h, k))?;

        Ok(())
    }
}

/// Little-endian byte view of an f32 scalar. cuBLASLt reads `alpha`/`beta`
/// by pointer and the scalar's host endianness matches the device's.
#[inline]
fn bytemuck_f32_le(v: &f32) -> &[u8] {
    // SAFETY: f32 is POD; the 4 bytes are always valid to read.
    unsafe { std::slice::from_raw_parts(v as *const f32 as *const u8, 4) }
}
