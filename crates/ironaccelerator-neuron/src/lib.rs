//! # `ironaccelerator-neuron`
//!
//! AWS Neuron backend — covers **Trainium** (trn1 / trn2) and
//! **Inferentia** (inf1 / inf2). The Neuron Runtime exposes a C API
//! through `libnrt.so`; we dynamically load it and call `nrt_init` +
//! `nrt_get_total_nc_count` for device enumeration. NeuronCores are the
//! scheduling unit Neuron exposes, so each NeuronCore becomes one
//! `DeviceDescriptor`.
//!
//! Instance family is read from AWS metadata-style env hints where
//! available (`NEURON_RT_VISIBLE_CORES`, `AWS_NEURON_VISIBLE_CORES`,
//! instance-type file under `/sys/devices/virtual/dmi/id/product_name`
//! when root is reachable); generation is inferred from the runtime
//! version string.

#![allow(clippy::missing_safety_doc)]

pub mod backend;
pub mod drv;
pub mod runtime;

pub use backend::{NeuronBackend, NEURON_BACKEND};

pub fn register(reg: &mut ironaccelerator_core::BackendRegistry) {
    reg.register(&NEURON_BACKEND);
}
