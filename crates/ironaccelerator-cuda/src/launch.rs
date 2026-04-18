//! IronAccelerator launch helpers over [`crate::drv::Function::launch`].
//!
//! The safe driver layer already exposes a compile-time typed launch path via
//! the [`LaunchArgs`](crate::drv::LaunchArgs) trait. This module adds shorthand
//! constructors for 1-D / 2-D geometries and a pass-through for the fully
//! general case.

use crate::drv::{Function, LaunchArgs, LaunchCfg, Stream};
use ironaccelerator_core::{kernel::LaunchDims, Result};

/// 1-D launch over `n` elements with `block` threads per block.
#[inline(always)]
pub fn launch_1d<A: LaunchArgs>(
    stream: &Stream, func: &Function, n: u32, block: u32, args: A,
) -> Result<()> {
    func.launch(LaunchCfg::for_elements(n, block), stream, args).map_err(Into::into)
}

/// 2-D launch over `(rows, cols)` with `(by, bx)` block dims.
#[inline(always)]
pub fn launch_2d<A: LaunchArgs>(
    stream: &Stream, func: &Function,
    rows: u32, cols: u32, by: u32, bx: u32, args: A,
) -> Result<()> {
    let cfg = LaunchCfg {
        grid: (cols.div_ceil(bx), rows.div_ceil(by), 1),
        block: (bx, by, 1),
        shared_bytes: 0,
    };
    func.launch(cfg, stream, args).map_err(Into::into)
}

/// Take a fully-described [`LaunchDims`] and dispatch.
#[inline(always)]
pub fn launch_dims<A: LaunchArgs>(
    stream: &Stream, func: &Function, dims: LaunchDims, args: A,
) -> Result<()> {
    let cfg = LaunchCfg {
        grid: dims.grid,
        block: dims.block,
        shared_bytes: dims.shared_bytes,
    };
    func.launch(cfg, stream, args).map_err(Into::into)
}

/// Lowest-level launch — pass a fully-formed [`LaunchCfg`].
#[inline(always)]
pub fn raw_launch<A: LaunchArgs>(
    stream: &Stream, func: &Function, cfg: LaunchCfg, args: A,
) -> Result<()> {
    func.launch(cfg, stream, args).map_err(Into::into)
}
