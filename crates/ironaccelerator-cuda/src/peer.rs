//! Peer access management between CUDA devices — NVLink / PCIe.

use crate::drv::Device;
use ironaccelerator_core::Result;
use std::sync::Arc;

/// `true` if `src` can directly read `dst`'s memory.
pub fn can_access(src: &Arc<Device>, dst: &Arc<Device>) -> Result<bool> {
    src.can_access_peer(dst).map_err(Into::into)
}

/// Enable P2P access from `src` to `peer`. Idempotent.
pub fn enable(src: &Arc<Device>, peer: &Arc<Device>) -> Result<()> {
    src.enable_peer_access(peer).map_err(Into::into)
}

pub fn enable_bidirectional(a: &Arc<Device>, b: &Arc<Device>) -> Result<()> {
    enable(a, b)?;
    enable(b, a)
}

/// `N×N` connectivity matrix.
pub fn topology(devs: &[Arc<Device>]) -> Result<Vec<Vec<bool>>> {
    let n = devs.len();
    let mut m = vec![vec![false; n]; n];
    for i in 0..n {
        for j in 0..n {
            m[i][j] = if i == j {
                true
            } else {
                can_access(&devs[i], &devs[j])?
            };
        }
    }
    Ok(m)
}
