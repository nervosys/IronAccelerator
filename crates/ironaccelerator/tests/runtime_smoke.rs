//! Umbrella smoke test: construct a `Runtime`, enumerate devices across
//! every compiled-in backend, and run the planner against a reference
//! GEMM. On CI hosts without real accelerators the planner legitimately
//! returns `BackendUnavailable` — we accept that, we just don't accept
//! panics or malformed plans.

use ironaccelerator::Runtime;
use ironaccelerator_core::{DType, Error, Workload};

#[test]
fn runtime_constructs_and_enumerates() {
    let rt = Runtime::new();
    let devices = rt.devices();
    for d in &devices {
        assert!(!d.name.is_empty(), "device name must be non-empty");
    }
}

#[test]
fn planner_handles_reference_gemm() {
    let rt = Runtime::new();
    let wl = Workload::gemm(1024, 1024, 1024, DType::F32);
    match rt.plan(&wl) {
        Ok(plan) => {
            assert!(plan.score > 0.0, "winning plan must have positive score");
        }
        Err(Error::BackendUnavailable(_)) => {
            // No vendor runtime on this host — acceptable in CI.
        }
        Err(other) => panic!("unexpected planner error: {other:?}"),
    }
}
