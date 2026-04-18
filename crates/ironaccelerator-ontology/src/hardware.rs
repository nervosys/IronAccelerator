//! Hardware sub-graph. Curated set of representative parts per family.
//! Add entries here when a new SKU needs explicit treatment; generic SKUs
//! are still planned for via the family-level `prefers` edges.

use crate::{HardwareNode, Id, Ontology};
use ironaccelerator_core::BackendKind;

fn h(
    id: &str,
    backend: BackendKind,
    vendor: &str,
    family: &str,
    arch: &str,
    year: u16,
    fp16: Option<f32>,
    fp8: Option<f32>,
    bw: Option<f32>,
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
        fp16_tflops: fp16,
        fp8_tflops: fp8,
        mem_bandwidth_gbs: bw,
        tags: tags.iter().map(|s| s.to_string()).collect(),
        notes: notes.into(),
    }
}

pub fn populate(o: &mut Ontology) {
    let nodes = [
        // ── NVIDIA ────────────────────────────────────────────────────────
        h("hardware.nvidia.gb200", BackendKind::Cuda, "nvidia", "blackwell", "sm_100",
            2024, Some(2500.0), Some(5000.0), Some(8000.0),
            &["datacenter", "fp4", "transformer-engine", "nvlink", "hbm3e"],
            "Grace-Blackwell superchip; 5th-gen Tensor Cores w/ FP4."),
        h("hardware.nvidia.h100", BackendKind::Cuda, "nvidia", "hopper", "sm_90a",
            2022, Some(989.0), Some(1979.0), Some(3350.0),
            &["datacenter", "fp8", "transformer-engine", "nvlink", "hbm3"],
            "Reference Hopper SKU; FP8 TE recipe is the optimal LLM path."),
        h("hardware.nvidia.l40s", BackendKind::Cuda, "nvidia", "ada", "sm_89",
            2023, Some(362.0), Some(733.0), Some(864.0),
            &["workstation", "fp8", "graphics"],
            "Ada workstation; great FP8 inference for sub-70B models."),
        h("hardware.nvidia.a100", BackendKind::Cuda, "nvidia", "ampere", "sm_80",
            2020, Some(312.0), None, Some(2039.0),
            &["datacenter", "tf32", "sparse-2-4", "nvlink", "hbm2e"],
            "Ampere; 2:4 sparsity, no FP8."),
        h("hardware.nvidia.rtx5090", BackendKind::Cuda, "nvidia", "blackwell", "sm_120",
            2025, Some(838.0), Some(1676.0), Some(1792.0),
            &["consumer", "fp4", "fp8"], "Consumer Blackwell."),
        h("hardware.nvidia.rtx4090", BackendKind::Cuda, "nvidia", "ada", "sm_89",
            2022, Some(330.0), Some(660.0), Some(1008.0),
            &["consumer", "fp8"], "Top consumer Ada part."),

        // ── AMD ─────────────────────────────────────────────────────────--
        h("hardware.amd.mi300x", BackendKind::Rocm, "amd", "cdna3", "gfx942",
            2023, Some(1307.0), Some(2614.0), Some(5300.0),
            &["datacenter", "fp8", "infinity-fabric", "hbm3"],
            "192 GB HBM3 — fits 70B in FP16 on a single device."),
        h("hardware.amd.mi325x", BackendKind::Rocm, "amd", "cdna3", "gfx942",
            2024, Some(1307.0), Some(2614.0), Some(6000.0),
            &["datacenter", "fp8", "hbm3e"], "Refresh of MI300X with HBM3e."),
        h("hardware.amd.rx7900xtx", BackendKind::Rocm, "amd", "rdna3", "gfx1100",
            2022, Some(123.0), None, Some(960.0),
            &["consumer"], "Consumer RDNA3; ROCm support is good but no FP8."),

        // ── APPLE ─────────────────────────────────────────────────────────
        h("hardware.apple.m3-max", BackendKind::Metal, "apple", "m-series", "Apple9",
            2023, Some(28.0), None, Some(400.0),
            &["consumer", "unified-memory", "ane"],
            "Up to 128 GB unified memory; ANE bridge via CoreML."),
        h("hardware.apple.m4-max", BackendKind::Metal, "apple", "m-series", "Apple10",
            2024, Some(38.0), None, Some(546.0),
            &["consumer", "unified-memory", "ane"], "M4 Max."),
        h("hardware.apple.m2-ultra", BackendKind::Metal, "apple", "m-series", "Apple8",
            2023, Some(54.0), None, Some(800.0),
            &["workstation", "unified-memory"], "192 GB unified — useful for LLMs."),

        // ── QUALCOMM ──────────────────────────────────────────────────────
        h("hardware.qualcomm.snapdragon-x-elite", BackendKind::QualcommNpu, "qualcomm", "hexagon", "v75",
            2024, Some(45.0), None, None,
            &["mobile", "npu", "hmx", "hvx"],
            "Hexagon NPU w/ HMX; 45 TOPS INT8 sustained."),
        h("hardware.qualcomm.snapdragon-8gen3", BackendKind::QualcommNpu, "qualcomm", "hexagon", "v73",
            2023, Some(20.0), None, None,
            &["mobile", "npu"], "Mobile SoC NPU."),
        h("hardware.qualcomm.cloud-ai-100-ultra", BackendKind::QualcommNpu, "qualcomm", "ai-100", "ai100u",
            2024, Some(870.0), None, None,
            &["datacenter", "npu", "low-power"], "Datacenter inference accelerator."),

        // ── Extra NVIDIA SKUs ─────────────────────────────────────────────
        h("hardware.nvidia.b100", BackendKind::Cuda, "nvidia", "blackwell", "sm_100",
            2024, Some(1800.0), Some(3500.0), Some(8000.0),
            &["datacenter", "fp4", "fp8", "transformer-engine", "hbm3e"],
            "Datacenter Blackwell B100; sibling of GB200."),
        h("hardware.nvidia.h200", BackendKind::Cuda, "nvidia", "hopper", "sm_90a",
            2023, Some(989.0), Some(1979.0), Some(4800.0),
            &["datacenter", "fp8", "transformer-engine", "nvlink", "hbm3e"],
            "Refresh of H100 with 141 GB HBM3e — much higher bandwidth."),
        h("hardware.nvidia.rtx6000-ada", BackendKind::Cuda, "nvidia", "ada", "sm_89",
            2022, Some(364.0), Some(728.0), Some(960.0),
            &["workstation", "fp8"], "Top Ada workstation; 48 GB."),

        // ── Extra AMD SKUs ────────────────────────────────────────────────
        h("hardware.amd.mi355x", BackendKind::Rocm, "amd", "cdna4", "gfx950",
            2025, Some(2300.0), Some(4600.0), Some(8000.0),
            &["datacenter", "fp4", "fp8", "hbm3e"], "CDNA4 with FP4 support."),
        h("hardware.amd.mi250x", BackendKind::Rocm, "amd", "cdna2", "gfx90a",
            2021, Some(383.0), None, Some(3276.0),
            &["datacenter", "infinity-fabric", "hbm2e"],
            "CDNA2; first AMD MI part with strong tensor throughput."),

        // ── Extra Apple SKUs ──────────────────────────────────────────────
        h("hardware.apple.m4-pro", BackendKind::Metal, "apple", "m-series", "Apple10",
            2024, Some(17.0), None, Some(273.0),
            &["consumer", "unified-memory", "ane"], "M4 Pro."),
        h("hardware.apple.m1-max", BackendKind::Metal, "apple", "m-series", "Apple7",
            2021, Some(21.0), None, Some(400.0),
            &["consumer", "unified-memory", "ane"],
            "M1 Max. Useful baseline for Apple silicon perf trends."),
        h("hardware.apple.m3-ultra", BackendKind::Metal, "apple", "m-series", "Apple9",
            2025, Some(56.0), None, Some(819.0),
            &["workstation", "unified-memory"], "M3 Ultra; up to 512 GB unified."),

        // ── Extra Qualcomm SKUs ───────────────────────────────────────────
        h("hardware.qualcomm.snapdragon-8gen4", BackendKind::QualcommNpu, "qualcomm", "hexagon", "v75",
            2024, Some(45.0), None, None,
            &["mobile", "npu", "hmx"], "Mobile SoC NPU with HMX."),
        h("hardware.qualcomm.cloud-ai-100", BackendKind::QualcommNpu, "qualcomm", "ai-100", "ai100",
            2021, Some(400.0), None, None,
            &["datacenter", "npu", "low-power"], "Original Cloud AI 100."),

        // ── Intel (Gaudi via QNN-style external runtime; surfaced under Cpu
        //   backend for now until a dedicated backend lands) ───────────────
        h("hardware.intel.gaudi3", BackendKind::Cpu, "intel", "gaudi", "gaudi3",
            2024, Some(1835.0), Some(1835.0), Some(3700.0),
            &["datacenter", "fp8", "ethernet-fabric"],
            "Intel Gaudi 3; FP8 + on-die 200 GbE NICs. Dedicated backend TODO."),

        // ── CPU SIMD baseline (for the reference path) ────────────────────
        h("hardware.cpu.x86_64-avx512", BackendKind::Cpu, "generic", "cpu", "avx512",
            2017, Some(2.0), None, Some(80.0),
            &["baseline", "simd"], "AVX-512 reference path."),
        h("hardware.cpu.aarch64-sve2", BackendKind::Cpu, "generic", "cpu", "sve2",
            2019, Some(1.0), None, Some(60.0),
            &["baseline", "simd"], "ARM SVE2 reference path."),
    ];

    for n in nodes {
        o.hardware.insert(n.id.clone(), n);
    }
}
