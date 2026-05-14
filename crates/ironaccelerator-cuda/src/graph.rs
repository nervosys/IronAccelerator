//! CUDA Graphs — thin adapter over [`crate::drv::CapturedGraph`] and
//! [`crate::drv::GraphExec`]. Captures a sequence of kernel launches on a
//! stream, then replays them with a single `cuGraphLaunch`.

use crate::drv::{CapturedGraph, Device, GraphExec, Stream};
use ironaccelerator_core::Result;
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub enum CaptureMode {
    Global,
    ThreadLocal,
    Relaxed,
}

pub struct Capture<'a> {
    stream: &'a Arc<Stream>,
    active: bool,
}

impl<'a> Capture<'a> {
    pub fn begin(stream: &'a Arc<Stream>, _mode: CaptureMode) -> Result<Self> {
        stream.begin_capture()?;
        Ok(Self {
            stream,
            active: true,
        })
    }

    pub fn end(mut self) -> Result<Graph> {
        let cap = self.stream.end_capture()?;
        self.active = false;
        Ok(Graph { inner: Some(cap) })
    }
}

impl<'a> Drop for Capture<'a> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.stream.end_capture();
        }
    }
}

pub struct Graph {
    inner: Option<CapturedGraph>,
}

impl Graph {
    pub fn instantiate(mut self, device: Arc<Device>) -> Result<ExecGraph> {
        let cap = self.inner.take().expect("graph consumed");
        let exec = GraphExec::new(cap, device)?;
        Ok(ExecGraph { inner: exec })
    }
}

pub struct ExecGraph {
    inner: GraphExec,
}

impl ExecGraph {
    #[inline]
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        self.inner.launch(stream)?;
        Ok(())
    }
}

/// Convenience: capture the work enqueued by `f` into a ready [`ExecGraph`].
pub fn capture<F>(
    stream: &Arc<Stream>,
    device: Arc<Device>,
    mode: CaptureMode,
    f: F,
) -> Result<ExecGraph>
where
    F: FnOnce(&Arc<Stream>) -> Result<()>,
{
    let cap = Capture::begin(stream, mode)?;
    f(stream)?;
    let g = cap.end()?;
    g.instantiate(device)
}
