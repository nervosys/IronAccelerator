//! Kernel launch description shared across backends.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LaunchDims {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    /// Dynamic shared / threadgroup memory in bytes.
    pub shared_bytes: u32,
}

impl LaunchDims {
    pub const fn linear(threads: u32, block: u32) -> Self {
        let grid = threads.div_ceil(block);
        Self { grid: (grid, 1, 1), block: (block, 1, 1), shared_bytes: 0 }
    }

    pub const fn elements(self) -> u64 {
        let g = self.grid.0 as u64 * self.grid.1 as u64 * self.grid.2 as u64;
        let b = self.block.0 as u64 * self.block.1 as u64 * self.block.2 as u64;
        g * b
    }
}

/// A pre-translated kernel ready to be launched. Backends extend this with
/// the actual function pointer / pipeline state.
pub trait KernelLaunch: Send + Sync {
    fn name(&self) -> &str;
    fn occupancy_hint(&self) -> u32 {
        0
    }
}
