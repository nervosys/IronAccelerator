//! Demonstrates the agentic-discovery flow: ask the ontology for
//! recommendations, then dispatch via the runtime.

use ironaccelerator::prelude::*;
use ironaccelerator::ontology::Ontology;

fn main() {
    let o = Ontology::global();

    println!("== Recommendations: FlashAttention on H100 ==");
    for rec in o.recommend(WorkloadKind::FlashAttention, "hardware.nvidia.h100") {
        println!("  [{:>4.1}] {} — {}", rec.score, rec.id, rec.rationale);
    }

    println!("\n== Recommendations: GEMM on MI300X ==");
    for rec in o.recommend(WorkloadKind::Gemm, "hardware.amd.mi300x") {
        println!("  [{:>4.1}] {} — {}", rec.score, rec.id, rec.rationale);
    }

    println!("\n== Recommendations: Attention on Snapdragon X Elite ==");
    for rec in o.recommend(WorkloadKind::Attention, "hardware.qualcomm.snapdragon-x-elite") {
        println!("  [{:>4.1}] {} — {}", rec.score, rec.id, rec.rationale);
    }

    println!("\n== Live runtime planning ==");
    let runtime = ironaccelerator::init();
    let wl = Workload::gemm(8192, 8192, 8192, DType::F8E4M3);
    match runtime.plan(&wl) {
        Ok(plan) => println!("  selected: {plan:?}"),
        Err(e)   => println!("  no live device found ({e})"),
    }
}
