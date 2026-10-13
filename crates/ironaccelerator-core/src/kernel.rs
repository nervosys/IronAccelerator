//! Kernel launch description shared across backends.

/// Grid and block geometry for a kernel launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LaunchDims {
    /// Grid dimensions in blocks (x, y, z).
    pub grid: (u32, u32, u32),
    /// Threads per block (x, y, z).
    pub block: (u32, u32, u32),
    /// Dynamic shared / threadgroup memory in bytes.
    pub shared_bytes: u32,
}

impl LaunchDims {
    /// Geometry for `threads` total threads in blocks of `block` threads, with a 1-D grid rounded up.
    pub const fn linear(threads: u32, block: u32) -> Self {
        let grid = threads.div_ceil(block);
        Self {
            grid: (grid, 1, 1),
            block: (block, 1, 1),
            shared_bytes: 0,
        }
    }

    /// Total thread count (grid volume times block volume).
    pub const fn elements(self) -> u64 {
        let g = self.grid.0 as u64 * self.grid.1 as u64 * self.grid.2 as u64;
        let b = self.block.0 as u64 * self.block.1 as u64 * self.block.2 as u64;
        g * b
    }
}

/// A pre-translated kernel ready to be launched. Backends extend this with
/// the actual function pointer / pipeline state.
pub trait KernelLaunch: Send + Sync {
    /// Returns the kernel's name.
    fn name(&self) -> &str;
    /// Suggested occupancy for the launch geometry, or `0` when the backend has no estimate.
    fn occupancy_hint(&self) -> u32 {
        0
    }
}
