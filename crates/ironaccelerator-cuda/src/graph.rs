//! CUDA Graphs — thin adapter over [`crate::drv::CapturedGraph`] and
//! [`crate::drv::GraphExec`]. Captures a sequence of kernel launches on the
//! session's stream, then replays them with a single `cuGraphLaunch`.

use crate::drv::{CapturedGraph, GraphExec};
use crate::Session;
use ironaccelerator_core::Result;

#[derive(Debug, Clone, Copy)]
pub enum CaptureMode { Global, ThreadLocal, Relaxed }

pub struct Capture<'a> {
    session: &'a Session,
    active: bool,
}

impl<'a> Capture<'a> {
    pub fn begin(session: &'a Session, _mode: CaptureMode) -> Result<Self> {
        session.stream().begin_capture()?;
        Ok(Self { session, active: true })
    }

    pub fn end(mut self) -> Result<Graph> {
        let cap = self.session.stream().end_capture()?;
        self.active = false;
        Ok(Graph { inner: Some(cap) })
    }
}

impl<'a> Drop for Capture<'a> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.session.stream().end_capture();
        }
    }
}

pub struct Graph { inner: Option<CapturedGraph> }

impl Graph {
    pub fn instantiate(mut self, session: &Session) -> Result<ExecGraph> {
        let cap = self.inner.take().expect("graph consumed");
        let exec = GraphExec::new(cap, session.device().clone())?;
        Ok(ExecGraph { inner: exec })
    }
}

pub struct ExecGraph { inner: GraphExec }

impl ExecGraph {
    #[inline]
    pub fn launch(&self, session: &Session) -> Result<()> {
        self.inner.launch(session.stream())?;
        session.metrics().record_launch();
        Ok(())
    }
}

/// Convenience: capture the work enqueued by `f` into a ready [`ExecGraph`].
pub fn capture<F>(session: &Session, mode: CaptureMode, f: F) -> Result<ExecGraph>
where F: FnOnce(&Session) -> Result<()>,
{
    let cap = Capture::begin(session, mode)?;
    f(session)?;
    let g = cap.end()?;
    g.instantiate(session)
}
