//! Umbrella smoke test: construct a `Runtime` and survey devices across every
//! compiled-in backend. On CI hosts without real accelerators the survey
//! legitimately comes back empty — we accept that, we just don't accept panics
//! or malformed descriptors.

use ironaccelerator::Runtime;
use ironaccelerator_core::CapabilityFlags;

#[test]
fn runtime_constructs_and_enumerates() {
    let rt = Runtime::new();
    let devices = rt.devices();
    for d in &devices {
        assert!(!d.name.is_empty(), "device name must be non-empty");
    }
}

#[test]
fn capability_filter_is_a_subset_of_the_full_survey() {
    let rt = Runtime::new();
    let all = rt.devices();
    let fp32 = rt.devices_with(CapabilityFlags::FP32);
    assert!(fp32.len() <= all.len());
    for d in &fp32 {
        assert!(d.capability.flags.contains(CapabilityFlags::FP32));
    }
}

#[test]
fn available_backends_are_registered_and_queryable() {
    let rt = Runtime::new();
    for kind in rt.available_backends() {
        assert!(rt.registry().get(kind).is_some());
        // Ordinal 0 may or may not exist; we only require a non-panicking
        // typed answer from a backend that claims to be available.
        assert!(rt.capabilities(kind, 0).is_some());
    }
}
