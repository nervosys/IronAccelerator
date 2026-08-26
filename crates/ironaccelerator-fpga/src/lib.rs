//! # `ironaccelerator-fpga`
//!
//! FPGA backend — AMD/Xilinx **Alveo** and **Versal** accelerator cards via the
//! Xilinx Runtime (XRT). The runtime is loaded dynamically from `libxrt_core`;
//! nothing links XRT at build time, so this crate compiles on any host and a
//! machine without XRT reports zero devices.
//!
//! ## Scope: probe / enumerate only
//!
//! This is deliberately a discovery scaffold, at the maturity of the TPU and
//! Neuron backends. FPGA kernels are **pre-synthesised bitstreams** (`.xclbin`)
//! built offline by Vitis — there is no runtime-compile path analogous to
//! NVRTC, and the execution model (load a bitstream, bind memory banks, run
//! compute units) is a workload concern that layers *above* the driver line
//! this project stops at. What belongs here is discovering the cards and
//! reporting them into the common device survey; that is what this crate does.
//!
//! See [`drv`] for the XRT loader and [`backend`] for the [`Backend`] impl.
//!
//! [`Backend`]: ironaccelerator_core::Backend

#![allow(clippy::missing_safety_doc)]

pub mod backend;
pub mod drv;

pub use backend::{FpgaBackend, FPGA_BACKEND};

pub fn register(reg: &mut ironaccelerator_core::BackendRegistry) {
    reg.register(&FPGA_BACKEND);
}
