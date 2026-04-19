//! FlashAttention-3 via cuDNN v9 backend descriptors.
//!
//! Composes the canonical SDPA pattern as a raw `cudnnBackend*` operation
//! graph:
//!
//! ```text
//!     S      = MATMUL(Q, K^T)         // [B,H,Sq,Sk]
//!     S'     = MUL(S, softmax_scale)  // pointwise
//!     M      = REDUCE_MAX(S', dim=-1) // [B,H,Sq,1]
//!     T      = SUB(S', M)             // broadcast
//!     E      = EXP(T)
//!     Z      = REDUCE_SUM(E, dim=-1)  // [B,H,Sq,1]
//!     P      = DIV(E, Z)
//!     O      = MATMUL(P, V)           // [B,H,Sq,D]
//! ```
//!
//! cuDNN 9's `CUDNN_HEUR_MODE_A` recognises this pattern and, on Hopper with
//! BF16 / FP16 / FP8 inputs, routes execution to the fused FlashAttention-3
//! kernels. On earlier arches the same graph falls through to FA-2 or a
//! generic composed kernel — the planner correctly degrades.
//!
//! # Layout
//!
//! Tensors are **BHSD** (batch, heads, seq, head_dim). `K` is presented with
//! strides swapping the last two dims so cuDNN sees it as `[D, Sk]` for the
//! first matmul (the implicit-transpose trick).
//!
//! # Non-goals (v1.0)
//!
//! - Explicit causal / attention-bias masks — to be added once the baseline
//!   graph is verified on Hopper.
//! - Dropout — same rationale.
//! - Backward pass — forward only.

use crate::attention::AttentionParams;
use crate::cudnn::{BackendDescr, CudnnDType, CudnnHandle};
use crate::drv::Stream;
use iron_cuda_sys::cudnn as sys;
use ironaccelerator_core::{DType, Error, Result};
use std::ffi::c_void;
use std::sync::Arc;

// ── UIDs ────────────────────────────────────────────────────────────────────
// Stable within a plan; used for variant-pack pointer binding.

const UID_Q: i64 = 1;
const UID_K: i64 = 2;
const UID_V: i64 = 3;
const UID_O: i64 = 4;
// virtuals — intermediates the engine materialises or fuses away
const UID_S: i64 = 100;
const UID_S_SCALED: i64 = 101;
const UID_ROW_MAX: i64 = 102;
const UID_SHIFTED: i64 = 103;
const UID_EXP: i64 = 104;
const UID_ROW_SUM: i64 = 105;
const UID_P: i64 = 106;

const BYTE_ALIGN: i64 = 16;

fn dtype_to_cudnn(d: DType) -> Result<CudnnDType> {
    Ok(match d {
        DType::F16 => CudnnDType::Half,
        DType::Bf16 => CudnnDType::Bfloat16,
        DType::F32 => CudnnDType::Float,
        DType::F8E4M3 => CudnnDType::Fp8E4M3,
        DType::F8E5M2 => CudnnDType::Fp8E5M2,
        _ => return Err(Error::Unsupported("flash_attention: unsupported dtype")),
    })
}

// ── Tensor descriptor ───────────────────────────────────────────────────────

fn tensor_desc(
    uid: i64,
    dtype: CudnnDType,
    dims: &[i64],
    strides: &[i64],
    is_virtual: bool,
) -> Result<BackendDescr> {
    debug_assert_eq!(dims.len(), strides.len());
    let mut d = BackendDescr::new(sys::CUDNN_BACKEND_TENSOR_DESCRIPTOR)?;
    d.set_i64(sys::CUDNN_ATTR_TENSOR_UNIQUE_ID, &[uid])?;
    d.set_data_type(sys::CUDNN_ATTR_TENSOR_DATA_TYPE, &[dtype])?;
    d.set_i64(sys::CUDNN_ATTR_TENSOR_DIMENSIONS, dims)?;
    d.set_i64(sys::CUDNN_ATTR_TENSOR_STRIDES, strides)?;
    d.set_i64(sys::CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, &[BYTE_ALIGN])?;
    d.set_bool(sys::CUDNN_ATTR_TENSOR_IS_VIRTUAL, &[is_virtual as u8])?;
    d.finalize()?;
    Ok(d)
}

fn bhsd(b: i64, h: i64, s: i64, d: i64) -> ([i64; 4], [i64; 4]) {
    let dims = [b, h, s, d];
    let strides = [h * s * d, s * d, d, 1];
    (dims, strides)
}

/// K presented as `[B, H, D, Sk]` via swapped last-two strides — i.e. the
/// buffer is still physically `[B, H, Sk, D]` BHSD, but cuDNN reads it as
/// its transpose so the first matmul computes `Q @ K^T`.
fn bhsd_transposed(b: i64, h: i64, s_k: i64, d: i64) -> ([i64; 4], [i64; 4]) {
    let dims = [b, h, d, s_k];
    let strides = [h * s_k * d, s_k * d, 1, d];
    (dims, strides)
}

// ── MATMUL op ───────────────────────────────────────────────────────────────

fn matmul_desc(compute: CudnnDType) -> Result<BackendDescr> {
    let mut d = BackendDescr::new(sys::CUDNN_BACKEND_MATMUL_DESCRIPTOR)?;
    d.set_data_type(sys::CUDNN_ATTR_MATMUL_COMP_TYPE, &[compute])?;
    d.finalize()?;
    Ok(d)
}

fn matmul_op(a: &BackendDescr, b: &BackendDescr, c: &BackendDescr, mm: &BackendDescr)
    -> Result<BackendDescr>
{
    let mut d = BackendDescr::new(sys::CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)?;
    d.set_descriptors(sys::CUDNN_ATTR_OPERATION_MATMUL_ADESC, &[a.raw()])?;
    d.set_descriptors(sys::CUDNN_ATTR_OPERATION_MATMUL_BDESC, &[b.raw()])?;
    d.set_descriptors(sys::CUDNN_ATTR_OPERATION_MATMUL_CDESC, &[c.raw()])?;
    d.set_descriptors(sys::CUDNN_ATTR_OPERATION_MATMUL_DESC, &[mm.raw()])?;
    d.finalize()?;
    Ok(d)
}

// ── Pointwise op ────────────────────────────────────────────────────────────

fn pointwise_desc(mode: i64, math_prec: CudnnDType) -> Result<BackendDescr> {
    let mut d = BackendDescr::new(sys::CUDNN_BACKEND_POINTWISE_DESCRIPTOR)?;
    d.set_i64(sys::CUDNN_ATTR_POINTWISE_MODE, &[mode])?;
    d.set_data_type(sys::CUDNN_ATTR_POINTWISE_MATH_PREC, &[math_prec])?;
    d.finalize()?;
    Ok(d)
}

/// Unary pointwise (EXP, IDENTITY): y = f(x).
fn pointwise_op_unary(pw: &BackendDescr, x: &BackendDescr, y: &BackendDescr)
    -> Result<BackendDescr>
{
    let mut d = BackendDescr::new(sys::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)?;
    d.set_descriptors(sys::CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, &[pw.raw()])?;
    d.set_descriptors(sys::CUDNN_ATTR_OPERATION_POINTWISE_XDESC, &[x.raw()])?;
    d.set_descriptors(sys::CUDNN_ATTR_OPERATION_POINTWISE_YDESC, &[y.raw()])?;
    d.finalize()?;
    Ok(d)
}

/// Binary pointwise (ADD, MUL, SUB, DIV): y = f(x, b).
fn pointwise_op_binary(
    pw: &BackendDescr,
    x: &BackendDescr, b: &BackendDescr, y: &BackendDescr,
) -> Result<BackendDescr> {
    let mut d = BackendDescr::new(sys::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)?;
    d.set_descriptors(sys::CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, &[pw.raw()])?;
    d.set_descriptors(sys::CUDNN_ATTR_OPERATION_POINTWISE_XDESC, &[x.raw()])?;
    d.set_descriptors(sys::CUDNN_ATTR_OPERATION_POINTWISE_BDESC, &[b.raw()])?;
    d.set_descriptors(sys::CUDNN_ATTR_OPERATION_POINTWISE_YDESC, &[y.raw()])?;
    d.finalize()?;
    Ok(d)
}

/// Scalar-MUL pointwise using ALPHA1: y = alpha * x. Cheaper than a
/// by-value tensor for the softmax scale factor.
fn pointwise_op_scale(
    pw: &BackendDescr,
    x: &BackendDescr, y: &BackendDescr, alpha: f64,
) -> Result<BackendDescr> {
    let mut d = BackendDescr::new(sys::CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)?;
    d.set_descriptors(sys::CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, &[pw.raw()])?;
    d.set_descriptors(sys::CUDNN_ATTR_OPERATION_POINTWISE_XDESC, &[x.raw()])?;
    d.set_descriptors(sys::CUDNN_ATTR_OPERATION_POINTWISE_YDESC, &[y.raw()])?;
    d.set_f64(sys::CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, &[alpha])?;
    d.finalize()?;
    Ok(d)
}

// ── Reduction op ────────────────────────────────────────────────────────────

fn reduction_desc(op: i64, compute: CudnnDType) -> Result<BackendDescr> {
    let mut d = BackendDescr::new(sys::CUDNN_BACKEND_REDUCTION_DESCRIPTOR)?;
    d.set_i64(sys::CUDNN_ATTR_REDUCTION_OPERATOR, &[op])?;
    d.set_data_type(sys::CUDNN_ATTR_REDUCTION_COMP_TYPE, &[compute])?;
    d.finalize()?;
    Ok(d)
}

fn reduction_op(r: &BackendDescr, x: &BackendDescr, y: &BackendDescr)
    -> Result<BackendDescr>
{
    let mut d = BackendDescr::new(sys::CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)?;
    d.set_descriptors(sys::CUDNN_ATTR_OPERATION_REDUCTION_DESC, &[r.raw()])?;
    d.set_descriptors(sys::CUDNN_ATTR_OPERATION_REDUCTION_XDESC, &[x.raw()])?;
    d.set_descriptors(sys::CUDNN_ATTR_OPERATION_REDUCTION_YDESC, &[y.raw()])?;
    d.finalize()?;
    Ok(d)
}

// ── Plan ────────────────────────────────────────────────────────────────────

/// A compiled FA-3 execution plan for a specific `AttentionParams` shape on
/// a specific cuDNN handle (i.e. a specific device). Reuse across calls of
/// the same shape is the whole point — building the plan is not cheap.
pub struct FlashAttention3Plan {
    handle: Arc<CudnnHandle>,
    plan: BackendDescr,
    params: AttentionParams,
    workspace_size: i64,
    // Held to keep the descriptor graph alive for the lifetime of the plan —
    // cuDNN does not clone them into the ExecutionPlan.
    _retained: Vec<BackendDescr>,
}

unsafe impl Send for FlashAttention3Plan {}
unsafe impl Sync for FlashAttention3Plan {}

impl FlashAttention3Plan {
    pub fn new(handle: Arc<CudnnHandle>, params: AttentionParams) -> Result<Self> {
        let dtype = dtype_to_cudnn(params.dtype)?;
        // Accumulate in F32 for all supported dtypes (BF16/FP16/FP8).
        let compute = CudnnDType::Float;

        let b = params.batch as i64;
        let h = params.heads as i64;
        let sq = params.seq_q as i64;
        let sk = params.seq_k as i64;
        let dh = params.head_dim as i64;

        // ── tensors ──────────────────────────────────────────────────────
        let (q_dims, q_str) = bhsd(b, h, sq, dh);
        let t_q = tensor_desc(UID_Q, dtype, &q_dims, &q_str, false)?;
        let (k_dims, k_str) = bhsd_transposed(b, h, sk, dh);
        let t_k = tensor_desc(UID_K, dtype, &k_dims, &k_str, false)?;
        let (v_dims, v_str) = bhsd(b, h, sk, dh);
        let t_v = tensor_desc(UID_V, dtype, &v_dims, &v_str, false)?;
        let (o_dims, o_str) = bhsd(b, h, sq, dh);
        let t_o = tensor_desc(UID_O, dtype, &o_dims, &o_str, false)?;

        // intermediates
        let s_dims = [b, h, sq, sk];
        let s_str = [h * sq * sk, sq * sk, sk, 1];
        let t_s = tensor_desc(UID_S, compute, &s_dims, &s_str, true)?;
        let t_s_scaled = tensor_desc(UID_S_SCALED, compute, &s_dims, &s_str, true)?;

        let red_dims = [b, h, sq, 1];
        let red_str = [h * sq, sq, 1, 1];
        let t_row_max = tensor_desc(UID_ROW_MAX, compute, &red_dims, &red_str, true)?;
        let t_row_sum = tensor_desc(UID_ROW_SUM, compute, &red_dims, &red_str, true)?;

        let t_shifted = tensor_desc(UID_SHIFTED, compute, &s_dims, &s_str, true)?;
        let t_exp = tensor_desc(UID_EXP, compute, &s_dims, &s_str, true)?;
        let t_p = tensor_desc(UID_P, dtype, &s_dims, &s_str, true)?;

        // ── ops ──────────────────────────────────────────────────────────
        // 1. S = Q @ K^T
        let mm_desc = matmul_desc(compute)?;
        let op_qk = matmul_op(&t_q, &t_k, &t_s, &mm_desc)?;

        // 2. S' = S * softmax_scale
        let pw_mul = pointwise_desc(sys::CUDNN_POINTWISE_IDENTITY, compute)?;
        let op_scale = pointwise_op_scale(&pw_mul, &t_s, &t_s_scaled, params.softmax_scale as f64)?;

        // 3. row_max = reduce_max(S', axis=-1)
        let red_max = reduction_desc(sys::CUDNN_REDUCE_TENSOR_MAX, compute)?;
        let op_max = reduction_op(&red_max, &t_s_scaled, &t_row_max)?;

        // 4. shifted = S' - row_max (broadcast)
        let pw_sub = pointwise_desc(sys::CUDNN_POINTWISE_SUB, compute)?;
        let op_sub = pointwise_op_binary(&pw_sub, &t_s_scaled, &t_row_max, &t_shifted)?;

        // 5. exp = exp(shifted)
        let pw_exp = pointwise_desc(sys::CUDNN_POINTWISE_EXP, compute)?;
        let op_exp = pointwise_op_unary(&pw_exp, &t_shifted, &t_exp)?;

        // 6. row_sum = reduce_sum(exp, axis=-1)
        let red_sum = reduction_desc(sys::CUDNN_REDUCE_TENSOR_ADD, compute)?;
        let op_sum = reduction_op(&red_sum, &t_exp, &t_row_sum)?;

        // 7. P = exp / row_sum (broadcast)  — cast down to input dtype
        let pw_div = pointwise_desc(sys::CUDNN_POINTWISE_DIV, dtype)?;
        let op_div = pointwise_op_binary(&pw_div, &t_exp, &t_row_sum, &t_p)?;

        // 8. O = P @ V
        let op_pv = matmul_op(&t_p, &t_v, &t_o, &mm_desc)?;

        // ── operation graph ──────────────────────────────────────────────
        let ops = [
            op_qk.raw(), op_scale.raw(), op_max.raw(), op_sub.raw(),
            op_exp.raw(), op_sum.raw(), op_div.raw(), op_pv.raw(),
        ];
        let mut op_graph = BackendDescr::new(sys::CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR)?;
        op_graph.set_handle(sys::CUDNN_ATTR_OPERATIONGRAPH_HANDLE, &[handle.raw()])?;
        op_graph.set_descriptors(sys::CUDNN_ATTR_OPERATIONGRAPH_OPS, &ops)?;
        op_graph.finalize()?;

        // ── heuristics → engine → engine config → execution plan ─────────
        let mut heur = BackendDescr::new(sys::CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR)?;
        heur.set_descriptors(sys::CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, &[op_graph.raw()])?;
        heur.set_i64(sys::CUDNN_ATTR_ENGINEHEUR_MODE, &[sys::CUDNN_HEUR_MODE_A])?;
        heur.finalize()?;

        // Pre-create an empty ENGINECFG descriptor; GetAttribute populates
        // it in place (the handle is stable, only its internal state is
        // filled). Mark it finalized afterwards so downstream wrappers
        // accept it.
        let cfg = BackendDescr::new(sys::CUDNN_BACKEND_ENGINECFG_DESCRIPTOR)?;
        let mut cfg_raw = cfg.raw();
        let n = unsafe {
            heur.get_attribute(
                sys::CUDNN_ATTR_ENGINEHEUR_RESULTS,
                crate::cudnn::AttrType::BackendDescriptor,
                1, &mut cfg_raw)?
        };
        if n < 1 {
            return Err(Error::Unsupported("flash_attention: no heuristic engine found"));
        }
        // Re-wrap so `finalized = true` flows through to BackendDescr::execute.
        let cfg_handle = cfg.raw();
        std::mem::forget(cfg);
        let cfg = crate::cudnn::adopt_descriptor(
            cfg_handle, sys::CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, true);

        let mut plan = BackendDescr::new(sys::CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR)?;
        plan.set_handle(sys::CUDNN_ATTR_EXECUTION_PLAN_HANDLE, &[handle.raw()])?;
        plan.set_descriptors(sys::CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, &[cfg.raw()])?;
        plan.finalize()?;

        // Workspace size is an output attribute of the finalized plan.
        let ws = plan.get_i64(sys::CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE)?;

        Ok(Self {
            handle,
            plan,
            params,
            workspace_size: ws,
            _retained: vec![
                t_q, t_k, t_v, t_o,
                t_s, t_s_scaled, t_row_max, t_row_sum, t_shifted, t_exp, t_p,
                mm_desc, pw_mul, pw_sub, pw_exp, pw_div, red_max, red_sum,
                op_qk, op_scale, op_max, op_sub, op_exp, op_sum, op_div, op_pv,
                op_graph, heur, cfg,
            ],
        })
    }

    /// Bytes of device workspace the plan requires. Callers allocate this
    /// once per plan and reuse it across executions.
    #[inline] pub fn workspace_size(&self) -> usize { self.workspace_size.max(0) as usize }

    #[inline] pub fn params(&self) -> &AttentionParams { &self.params }

    /// Execute the plan. All pointers are device pointers. `workspace` must
    /// be at least [`Self::workspace_size`] bytes — pass null if the plan
    /// needs none.
    ///
    /// # Safety
    /// Device pointers must reference buffers of the shapes declared in
    /// `params` and remain live for the duration of the stream's execution.
    pub unsafe fn execute(
        &self,
        stream: &Stream,
        q: *const c_void, k: *const c_void, v: *const c_void,
        o: *mut c_void,
        workspace: *mut c_void,
    ) -> Result<()> {
        self.handle.set_stream(stream)?;

        let uids: [i64; 4] = [UID_Q, UID_K, UID_V, UID_O];
        let ptrs: [*mut c_void; 4] = [
            q as *mut c_void, k as *mut c_void, v as *mut c_void, o,
        ];

        let mut vp = BackendDescr::new(sys::CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR)?;
        vp.set_i64(sys::CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, &uids)?;
        unsafe {
            vp.set_attribute(
                sys::CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                crate::cudnn::AttrType::VoidPtr, &ptrs)?;
            vp.set_attribute(
                sys::CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                crate::cudnn::AttrType::VoidPtr, &[workspace])?;
        }
        vp.finalize()?;

        BackendDescr::execute(&self.handle, &self.plan, &vp)
    }
}

