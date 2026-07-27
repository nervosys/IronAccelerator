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

/// Every backend must answer `capabilities(ordinal)` consistently with the
/// descriptor `enumerate()` produced for that same ordinal.
///
/// Several backends used to ignore the ordinal entirely and return a constant,
/// so `Runtime::capabilities(Vulkan, 0)` could disagree with
/// `devices()[0].capability.flags` on the same machine. This pins that shut.
#[test]
fn per_device_capabilities_match_enumerated_descriptors() {
    let rt = Runtime::new();
    for b in rt.registry().available() {
        for d in b.enumerate().unwrap_or_default() {
            let live = b
                .capabilities(d.id.ordinal)
                .unwrap_or_else(|e| panic!("{:?} ordinal {}: {e}", b.kind(), d.id.ordinal));
            assert_eq!(
                live,
                d.capability.flags,
                "{:?} ordinal {} disagrees between capabilities() and enumerate()",
                b.kind(),
                d.id.ordinal
            );
        }
    }
}

/// An ordinal past the end must be a typed error, never a plausible answer.
#[test]
fn out_of_range_ordinals_are_rejected() {
    let rt = Runtime::new();
    for b in rt.registry().available() {
        let n = b.enumerate().map(|d| d.len()).unwrap_or(0) as u32;
        assert!(
            b.capabilities(n.max(1) + 1_000).is_err(),
            "{:?} answered for an ordinal it never enumerated",
            b.kind()
        );
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
