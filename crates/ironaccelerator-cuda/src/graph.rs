//! CUDA Graphs — thin adapter over [`crate::drv::CapturedGraph`] and
//! [`crate::drv::GraphExec`]. Captures a sequence of kernel launches on a
//! stream, then replays them with a single `cuGraphLaunch`.

use crate::drv::{CapturedGraph, Device, GraphExec, Stream};
use crate::sys::driver::CUstreamCaptureMode;
use ironaccelerator_core::Result;
use std::sync::Arc;

/// Stream-capture mode passed to [`Capture::begin`] and [`capture`].
#[derive(Debug, Clone, Copy)]
pub enum CaptureMode {
    /// Mirrors `CU_STREAM_CAPTURE_MODE_GLOBAL`.
    Global,
    /// Mirrors `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`.
    ThreadLocal,
    /// Mirrors `CU_STREAM_CAPTURE_MODE_RELAXED`.
    Relaxed,
}

impl From<CaptureMode> for CUstreamCaptureMode {
    fn from(mode: CaptureMode) -> Self {
        match mode {
            CaptureMode::Global => CUstreamCaptureMode::Global,
            CaptureMode::ThreadLocal => CUstreamCaptureMode::ThreadLocal,
            CaptureMode::Relaxed => CUstreamCaptureMode::Relaxed,
        }
    }
}

/// RAII guard for an in-progress stream capture. Capture is ended on drop if
/// [`Capture::end`] was not called.
pub struct Capture<'a> {
    stream: &'a Arc<Stream>,
    active: bool,
}

impl<'a> Capture<'a> {
    /// Begin capturing work enqueued on `stream` in `mode`. Wraps
    /// `cuStreamBeginCapture_v2`.
    pub fn begin(stream: &'a Arc<Stream>, mode: CaptureMode) -> Result<Self> {
        stream.begin_capture_mode(mode.into())?;
        Ok(Self {
            stream,
            active: true,
        })
    }

    /// End capture and return the captured graph. Wraps `cuStreamEndCapture`.
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

/// A captured CUDA graph that has not been instantiated yet.
pub struct Graph {
    inner: Option<CapturedGraph>,
}

impl Graph {
    /// Instantiate into an executable graph on `device`. Consumes `self`.
    pub fn instantiate(mut self, device: Arc<Device>) -> Result<ExecGraph> {
        let cap = self.inner.take().expect("graph consumed");
        let exec = GraphExec::new(cap, device)?;
        Ok(ExecGraph { inner: exec })
    }
}

/// An instantiated, launchable CUDA graph.
pub struct ExecGraph {
    inner: GraphExec,
}

impl ExecGraph {
    /// Launch the graph on `stream`. Wraps `cuGraphLaunch`.
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
