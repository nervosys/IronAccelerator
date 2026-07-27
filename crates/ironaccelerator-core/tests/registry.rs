//! Backend-registry integration test using a synthetic in-test backend.

use ironaccelerator_core::*;

struct DummyBackend;

impl Backend for DummyBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }
    fn is_available(&self) -> bool {
        true
    }
    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>> {
        Ok(Vec::new())
    }
    fn capabilities(&self, _: u32) -> Result<CapabilityFlags> {
        Ok(CapabilityFlags::FP32)
    }
}

static DUMMY: DummyBackend = DummyBackend;

#[test]
fn registry_round_trip() {
    let mut reg = BackendRegistry::new();
    reg.register(&DUMMY);
    reg.register(&DUMMY); // idempotent
    assert!(reg.get(BackendKind::Cpu).is_some());
    assert_eq!(reg.iter().count(), 1);
    assert_eq!(reg.available().count(), 1);
}

#[test]
fn registry_reports_capabilities_per_device() {
    let mut reg = BackendRegistry::new();
    reg.register(&DUMMY);
    let b = reg.get(BackendKind::Cpu).expect("registered");
    assert_eq!(b.capabilities(0).unwrap(), CapabilityFlags::FP32);
    assert!(reg.describe_all().is_empty());
}

#[test]
fn launch_dims_linear() {
    let d = LaunchDims::linear(1000, 256);
    assert_eq!(d.grid, (4, 1, 1));
    assert_eq!(d.block, (256, 1, 1));
    assert!(d.elements() >= 1000);
}

#[test]
fn dtype_classifications() {
    assert_eq!(DType::F32.class(), NumericClass::Float);
    assert_eq!(DType::I32.class(), NumericClass::Integer);
    assert_eq!(DType::Bool.class(), NumericClass::Boolean);
    assert_eq!(DType::QuantBlock.class(), NumericClass::Quantized);
    assert_eq!(DType::F8E4M3.bits(), 8);
    assert_eq!(DType::F4.bits(), 4);
    assert_eq!(DType::QuantBlock.bits(), 0);
}
