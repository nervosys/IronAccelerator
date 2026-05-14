//! Hardware sub-graph. Curated set of representative parts per family.
//! Add entries here when a new SKU needs explicit treatment; generic SKUs
//! are still planned for via the family-level `prefers` edges.
//!
//! Every backend family has at least one entry so an autonomous agent
//! querying the ontology gets a complete picture of what IronAccelerator
//! targets.

use crate::{HardwareNode, Id, Ontology};
use ironaccelerator_core::capability::CapabilityFlags as C;
use ironaccelerator_core::{BackendKind, ComputeTier};

#[allow(clippy::too_many_arguments)]
fn h(
    id: &str,
    backend: BackendKind,
    vendor: &str,
    family: &str,
    arch: &str,
    year: u16,
    tier: ComputeTier,
    caps: CapabilityFlags,
    fp16: Option<f32>,
    fp8: Option<f32>,
    bw: Option<f32>,
    mem_gib: Option<f32>,
    tags: &[&str],
    notes: &str,
) -> HardwareNode {
    HardwareNode {
        id: Id(id.into()),
        backend,
        vendor: vendor.into(),
        family: family.into(),
        arch: arch.into(),
        launch_year: year,
        compute_tier: tier,
        capabilities: caps,
        fp16_tflops: fp16,
        fp8_tflops: fp8,
        mem_bandwidth_gbs: bw,
        device_memory_gib: mem_gib,
        tags: tags.iter().map(|s| s.to_string()).collect(),
        notes: notes.into(),
    }
}

// Shortcuts for the most common capability combinations.
type CapabilityFlags = C;
const CUDA_BASE: C = C::FP32
    .union(C::FP16)
    .union(C::BF16)
    .union(C::TENSOR_CORES)
    .union(C::MULTI_STREAM)
    .union(C::GRAPHS)
    .union(C::ASYNC_ALLOC)
    .union(C::COOPERATIVE_LAUNCH)
    .union(C::NCCL)
    .union(C::PEER_ACCESS)
    .union(C::HOST_PINNED);
const CUDA_HOPPER: C = CUDA_BASE
    .union(C::FP8_E4M3)
    .union(C::FP8_E5M2)
    .union(C::TRANSFORMER_ENGINE)
    .union(C::SPARSE_2_4)
    .union(C::TF32)
    .union(C::HBM)
    .union(C::NVLINK)
    .union(C::DYNAMIC_PARALLELISM)
    .union(C::FLASH_ATTN);
const CUDA_BLACKWELL: C = CUDA_HOPPER.union(C::FP4);
const CUDA_AMPERE: C = CUDA_BASE
    .union(C::TF32)
    .union(C::SPARSE_2_4)
    .union(C::HBM)
    .union(C::NVLINK)
    .union(C::DYNAMIC_PARALLELISM)
    .union(C::FLASH_ATTN);
const CUDA_ADA: C = CUDA_BASE
    .union(C::TF32)
    .union(C::FP8_E4M3)
    .union(C::FP8_E5M2)
    .union(C::FLASH_ATTN);
const ROCM_BASE: C = C::FP32
    .union(C::FP16)
    .union(C::BF16)
    .union(C::WMMA)
    .union(C::MULTI_STREAM)
    .union(C::ASYNC_ALLOC)
    .union(C::RCCL)
    .union(C::PEER_ACCESS)
    .union(C::HOST_PINNED);
const ROCM_CDNA3: C = ROCM_BASE
    .union(C::FP8_E4M3)
    .union(C::FP8_E5M2)
    .union(C::INFINITY_FABRIC)
    .union(C::HBM)
    .union(C::FLASH_ATTN);
const ROCM_CDNA4: C = ROCM_CDNA3.union(C::FP4);
const METAL_BASE: C = C::FP32
    .union(C::FP16)
    .union(C::BF16)
    .union(C::UNIFIED_MEMORY)
    .union(C::MULTI_STREAM)
    .union(C::ANE);
const QNN_BASE: C = C::FP16.union(C::INT8).union(C::HVX).union(C::HMX);
const CPU_BASE: C = C::FP64
    .union(C::FP32)
    .union(C::FP16)
    .union(C::BF16)
    .union(C::INT8)
    .union(C::MULTI_STREAM);
const VK_BASE: C = C::FP32.union(C::FP16).union(C::MULTI_STREAM);
const WEBGPU_BASE: C = C::FP32.union(C::FP16);
const TPU_BASE: C = C::FP32
    .union(C::BF16)
    .union(C::INT8)
    .union(C::HBM)
    .union(C::TENSOR_CORES);
const LZ_BASE: C = C::FP32
    .union(C::FP16)
    .union(C::BF16)
    .union(C::INT8)
    .union(C::MULTI_STREAM);
const NEURON_BASE: C = C::FP32
    .union(C::FP16)
    .union(C::BF16)
    .union(C::INT8)
    .union(C::HBM);

pub fn populate(o: &mut Ontology) {
    use ComputeTier::*;
    let nodes = [
        // ── NVIDIA ────────────────────────────────────────────────────────
        h(
            "hardware.nvidia.gb200",
            BackendKind::Cuda,
            "nvidia",
            "blackwell",
            "sm_100",
            2024,
            Datacenter,
            CUDA_BLACKWELL,
            Some(2500.0),
            Some(5000.0),
            Some(8000.0),
            Some(192.0),
            &["datacenter", "fp4", "transformer-engine", "nvlink", "hbm3e"],
            "Grace-Blackwell superchip; 5th-gen Tensor Cores w/ FP4.",
        ),
        h(
            "hardware.nvidia.h100",
            BackendKind::Cuda,
            "nvidia",
            "hopper",
            "sm_90a",
            2022,
            Datacenter,
            CUDA_HOPPER,
            Some(989.0),
            Some(1979.0),
            Some(3350.0),
            Some(80.0),
            &["datacenter", "fp8", "transformer-engine", "nvlink", "hbm3"],
            "Reference Hopper SKU; FP8 TE recipe is the optimal LLM path.",
        ),
        h(
            "hardware.nvidia.h200",
            BackendKind::Cuda,
            "nvidia",
            "hopper",
            "sm_90a",
            2024,
            Datacenter,
            CUDA_HOPPER,
            Some(989.0),
            Some(1979.0),
            Some(4800.0),
            Some(141.0),
            &["datacenter", "fp8", "transformer-engine", "nvlink", "hbm3e"],
            "Hopper refresh; 141 GB HBM3e, 4.8 TB/s — more KV-cache headroom than H100.",
        ),
        h(
            "hardware.nvidia.b100",
            BackendKind::Cuda,
            "nvidia",
            "blackwell",
            "sm_100",
            2024,
            Datacenter,
            CUDA_BLACKWELL,
            Some(1800.0),
            Some(3500.0),
            Some(8000.0),
            Some(192.0),
            &[
                "datacenter",
                "fp4",
                "fp8",
                "transformer-engine",
                "nvlink",
                "hbm3e",
            ],
            "Blackwell datacenter part; FP4 TE extends FP8 recipe to 2x density.",
        ),
        h(
            "hardware.nvidia.l40s",
            BackendKind::Cuda,
            "nvidia",
            "ada",
            "sm_89",
            2023,
            Workstation,
            CUDA_ADA,
            Some(362.0),
            Some(733.0),
            Some(864.0),
            Some(48.0),
            &["workstation", "fp8", "graphics"],
            "Ada workstation; great FP8 inference for sub-70B models.",
        ),
        h(
            "hardware.nvidia.a100",
            BackendKind::Cuda,
            "nvidia",
            "ampere",
            "sm_80",
            2020,
            Datacenter,
            CUDA_AMPERE,
            Some(312.0),
            None,
            Some(2039.0),
            Some(80.0),
            &["datacenter", "tf32", "sparse-2-4", "nvlink", "hbm2e"],
            "Ampere; 2:4 sparsity, no FP8.",
        ),
        h(
            "hardware.nvidia.rtx5090",
            BackendKind::Cuda,
            "nvidia",
            "blackwell",
            "sm_120",
            2025,
            Consumer,
            CUDA_BLACKWELL,
            Some(838.0),
            Some(1676.0),
            Some(1792.0),
            Some(32.0),
            &["consumer", "fp4", "fp8"],
            "Consumer Blackwell.",
        ),
        h(
            "hardware.nvidia.rtx4090",
            BackendKind::Cuda,
            "nvidia",
            "ada",
            "sm_89",
            2022,
            Consumer,
            CUDA_ADA,
            Some(330.0),
            Some(660.0),
            Some(1008.0),
            Some(24.0),
            &["consumer", "fp8"],
            "Top consumer Ada part.",
        ),
        h(
            "hardware.nvidia.rtx3090ti",
            BackendKind::Cuda,
            "nvidia",
            "ampere",
            "sm_86",
            2022,
            Consumer,
            CUDA_AMPERE,
            Some(160.0),
            None,
            Some(1008.0),
            Some(24.0),
            &["consumer", "tf32", "sparse-2-4"],
            "Consumer Ampere reference — the IronAccelerator smoke-test rig.",
        ),
        // ── AMD ─────────────────────────────────────────────────────────--
        h(
            "hardware.amd.mi300x",
            BackendKind::Rocm,
            "amd",
            "cdna3",
            "gfx942",
            2023,
            Datacenter,
            ROCM_CDNA3,
            Some(1307.0),
            Some(2614.0),
            Some(5300.0),
            Some(192.0),
            &["datacenter", "fp8", "infinity-fabric", "hbm3"],
            "192 GB HBM3 — fits 70B in FP16 on a single device.",
        ),
        h(
            "hardware.amd.mi325x",
            BackendKind::Rocm,
            "amd",
            "cdna3",
            "gfx942",
            2024,
            Datacenter,
            ROCM_CDNA3,
            Some(1307.0),
            Some(2614.0),
            Some(6000.0),
            Some(256.0),
            &["datacenter", "fp8", "hbm3e"],
            "Refresh of MI300X with HBM3e.",
        ),
        h(
            "hardware.amd.rx7900xtx",
            BackendKind::Rocm,
            "amd",
            "rdna3",
            "gfx1100",
            2022,
            Consumer,
            ROCM_BASE,
            Some(123.0),
            None,
            Some(960.0),
            Some(24.0),
            &["consumer"],
            "Consumer RDNA3; ROCm support is good but no FP8.",
        ),
        h(
            "hardware.amd.mi355x",
            BackendKind::Rocm,
            "amd",
            "cdna4",
            "gfx950",
            2025,
            Datacenter,
            ROCM_CDNA4,
            Some(2300.0),
            Some(4600.0),
            Some(8000.0),
            Some(288.0),
            &["datacenter", "fp4", "fp8", "hbm3e"],
            "CDNA4 with FP4 support.",
        ),
        h(
            "hardware.amd.mi250x",
            BackendKind::Rocm,
            "amd",
            "cdna2",
            "gfx90a",
            2021,
            Datacenter,
            ROCM_BASE.union(C::INFINITY_FABRIC).union(C::HBM),
            Some(383.0),
            None,
            Some(3276.0),
            Some(128.0),
            &["datacenter", "infinity-fabric", "hbm2e"],
            "CDNA2; first AMD MI part with strong tensor throughput.",
        ),
        // ── APPLE ─────────────────────────────────────────────────────────
        h(
            "hardware.apple.m3-max",
            BackendKind::Metal,
            "apple",
            "m-series",
            "Apple9",
            2023,
            Consumer,
            METAL_BASE,
            Some(28.0),
            None,
            Some(400.0),
            Some(128.0),
            &["consumer", "unified-memory", "ane"],
            "Up to 128 GB unified memory; ANE bridge via CoreML.",
        ),
        h(
            "hardware.apple.m4-max",
            BackendKind::Metal,
            "apple",
            "m-series",
            "Apple10",
            2024,
            Consumer,
            METAL_BASE,
            Some(38.0),
            None,
            Some(546.0),
            Some(128.0),
            &["consumer", "unified-memory", "ane"],
            "M4 Max.",
        ),
        h(
            "hardware.apple.m2-ultra",
            BackendKind::Metal,
            "apple",
            "m-series",
            "Apple8",
            2023,
            Workstation,
            METAL_BASE,
            Some(54.0),
            None,
            Some(800.0),
            Some(192.0),
            &["workstation", "unified-memory"],
            "192 GB unified — useful for LLMs.",
        ),
        h(
            "hardware.apple.m4-pro",
            BackendKind::Metal,
            "apple",
            "m-series",
            "Apple10",
            2024,
            Consumer,
            METAL_BASE,
            Some(17.0),
            None,
            Some(273.0),
            Some(64.0),
            &["consumer", "unified-memory", "ane"],
            "M4 Pro.",
        ),
        h(
            "hardware.apple.m1-max",
            BackendKind::Metal,
            "apple",
            "m-series",
            "Apple7",
            2021,
            Consumer,
            METAL_BASE,
            Some(21.0),
            None,
            Some(400.0),
            Some(64.0),
            &["consumer", "unified-memory", "ane"],
            "M1 Max. Useful baseline for Apple silicon perf trends.",
        ),
        h(
            "hardware.apple.m3-ultra",
            BackendKind::Metal,
            "apple",
            "m-series",
            "Apple9",
            2025,
            Workstation,
            METAL_BASE,
            Some(56.0),
            None,
            Some(819.0),
            Some(512.0),
            &["workstation", "unified-memory"],
            "M3 Ultra; up to 512 GB unified.",
        ),
        // ── QUALCOMM ──────────────────────────────────────────────────────
        h(
            "hardware.qualcomm.snapdragon-x-elite",
            BackendKind::QualcommNpu,
            "qualcomm",
            "hexagon",
            "v75",
            2024,
            Mobile,
            QNN_BASE,
            Some(45.0),
            None,
            None,
            None,
            &["mobile", "npu", "hmx", "hvx"],
            "Hexagon NPU w/ HMX; 45 TOPS INT8 sustained.",
        ),
        h(
            "hardware.qualcomm.snapdragon-8gen3",
            BackendKind::QualcommNpu,
            "qualcomm",
            "hexagon",
            "v73",
            2023,
            Mobile,
            QNN_BASE,
            Some(20.0),
            None,
            None,
            None,
            &["mobile", "npu"],
            "Mobile SoC NPU.",
        ),
        h(
            "hardware.qualcomm.snapdragon-8gen4",
            BackendKind::QualcommNpu,
            "qualcomm",
            "hexagon",
            "v75",
            2024,
            Mobile,
            QNN_BASE,
            Some(45.0),
            None,
            None,
            None,
            &["mobile", "npu", "hmx"],
            "Mobile SoC NPU with HMX.",
        ),
        h(
            "hardware.qualcomm.cloud-ai-100-ultra",
            BackendKind::QualcommNpu,
            "qualcomm",
            "ai-100",
            "ai100u",
            2024,
            Datacenter,
            QNN_BASE,
            Some(870.0),
            None,
            None,
            Some(128.0),
            &["datacenter", "npu", "low-power"],
            "Datacenter inference accelerator.",
        ),
        h(
            "hardware.qualcomm.cloud-ai-100",
            BackendKind::QualcommNpu,
            "qualcomm",
            "ai-100",
            "ai100",
            2021,
            Datacenter,
            QNN_BASE,
            Some(400.0),
            None,
            None,
            Some(32.0),
            &["datacenter", "npu", "low-power"],
            "Original Cloud AI 100.",
        ),
        // ── Intel Gaudi ───────────────────────────────────────────────────
        h(
            "hardware.intel.gaudi3",
            BackendKind::Cpu,
            "intel",
            "gaudi",
            "gaudi3",
            2024,
            Datacenter,
            C::FP32
                .union(C::BF16)
                .union(C::FP8_E4M3)
                .union(C::FP8_E5M2)
                .union(C::HBM)
                .union(C::TENSOR_CORES),
            Some(1835.0),
            Some(1835.0),
            Some(3700.0),
            Some(128.0),
            &["datacenter", "fp8", "ethernet-fabric"],
            "Intel Gaudi 3; FP8 + on-die 200 GbE NICs. Dedicated backend TODO.",
        ),
        // ── Intel via Level Zero ──────────────────────────────────────────
        h(
            "hardware.intel.arc-b580",
            BackendKind::LevelZero,
            "intel",
            "battlemage",
            "xe2",
            2024,
            Consumer,
            LZ_BASE,
            Some(70.0),
            None,
            Some(456.0),
            Some(12.0),
            &["consumer", "xmx"],
            "Consumer Xe2 dGPU via Level Zero.",
        ),
        h(
            "hardware.intel.max-1550",
            BackendKind::LevelZero,
            "intel",
            "ponte-vecchio",
            "xe-hpc",
            2023,
            Datacenter,
            LZ_BASE.union(C::HBM).union(C::TENSOR_CORES),
            Some(839.0),
            None,
            Some(3200.0),
            Some(128.0),
            &["datacenter", "hbm", "xe-link"],
            "Ponte Vecchio data-center dGPU.",
        ),
        h(
            "hardware.intel.meteor-lake-npu",
            BackendKind::LevelZero,
            "intel",
            "meteor-lake",
            "npu3.0",
            2023,
            Mobile,
            LZ_BASE,
            Some(11.0),
            None,
            None,
            None,
            &["mobile", "npu"],
            "Integrated NPU in Core Ultra; Level-Zero-exposed.",
        ),
        // ── Cross-vendor Vulkan ───────────────────────────────────────────
        h(
            "hardware.vulkan.generic",
            BackendKind::Vulkan,
            "any",
            "vulkan",
            "1.3",
            2022,
            Consumer,
            VK_BASE,
            None,
            None,
            None,
            None,
            &["cross-vendor", "portable"],
            "Cross-vendor Vulkan Compute 1.3 target. Fallback when no \
             first-class backend is available.",
        ),
        h(
            "hardware.opengl.generic",
            BackendKind::OpenGl,
            "any",
            "opengl",
            "4.3",
            2012,
            Consumer,
            C::FP32.union(C::FP16),
            None,
            None,
            None,
            None,
            &["cross-vendor", "legacy"],
            "OpenGL 4.3 compute shaders — embedded/legacy GPU fallback.",
        ),
        // ── WebGPU (browser + native wgpu) ────────────────────────────────
        h(
            "hardware.webgpu.browser",
            BackendKind::WebGpu,
            "any",
            "webgpu",
            "1.0",
            2023,
            Consumer,
            WEBGPU_BASE,
            None,
            None,
            None,
            None,
            &["wasm", "browser", "cross-vendor"],
            "WebGPU 1.0 target; driving path for Rust→WASM accelerator code.",
        ),
        // ── Google TPU ────────────────────────────────────────────────────
        h(
            "hardware.google.tpu-v5p",
            BackendKind::Tpu,
            "google",
            "tpu",
            "v5p",
            2023,
            Datacenter,
            TPU_BASE,
            Some(459.0),
            None,
            Some(2765.0),
            Some(95.0),
            &["datacenter", "mxu", "ici"],
            "TPU v5p via PJRT; MXU (matrix unit) is the primary engine.",
        ),
        h(
            "hardware.google.tpu-v6e",
            BackendKind::Tpu,
            "google",
            "tpu",
            "v6e",
            2024,
            Datacenter,
            TPU_BASE.union(C::FP8_E4M3).union(C::FP8_E5M2),
            Some(918.0),
            Some(1836.0),
            Some(1640.0),
            Some(32.0),
            &["datacenter", "inference", "fp8"],
            "Trillium (TPU v6e); inference-focused, FP8 support.",
        ),
        // ── AWS Neuron ────────────────────────────────────────────────────
        h(
            "hardware.aws.trainium2",
            BackendKind::Neuron,
            "aws",
            "trainium",
            "trn2",
            2024,
            Datacenter,
            NEURON_BASE.union(C::FP8_E4M3),
            Some(840.0),
            Some(1280.0),
            Some(2900.0),
            Some(96.0),
            &["datacenter", "train", "neuron-cores-v3"],
            "Trainium 2 via Neuron Runtime; 96 GB HBM3.",
        ),
        h(
            "hardware.aws.inferentia2",
            BackendKind::Neuron,
            "aws",
            "inferentia",
            "inf2",
            2023,
            Datacenter,
            NEURON_BASE,
            Some(190.0),
            None,
            Some(820.0),
            Some(32.0),
            &["datacenter", "inference", "neuron-cores-v2"],
            "Inferentia 2; inference-focused Neuron part.",
        ),
        // ── CPU SIMD baseline (reference path) ────────────────────────────
        h(
            "hardware.cpu.x86_64-avx512",
            BackendKind::Cpu,
            "generic",
            "cpu",
            "avx512",
            2017,
            Baseline,
            CPU_BASE,
            Some(2.0),
            None,
            Some(80.0),
            None,
            &["baseline", "simd"],
            "AVX-512 reference path.",
        ),
        h(
            "hardware.cpu.aarch64-sve2",
            BackendKind::Cpu,
            "generic",
            "cpu",
            "sve2",
            2019,
            Baseline,
            CPU_BASE,
            Some(1.0),
            None,
            Some(60.0),
            None,
            &["baseline", "simd"],
            "ARM SVE2 reference path.",
        ),
    ];

    for n in nodes {
        o.hardware.insert(n.id.clone(), n);
    }
}
