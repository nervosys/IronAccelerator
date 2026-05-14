//! Lightweight CUDA timer primitive.
//!
//! A [`Timer`] pairs a start/stop [`TimingEvent`] for in-band kernel timing.
//! Use [`crate::drv::Event`] directly when you only need a fence.

use crate::drv::{Stream, TimingEvent};
use ironaccelerator_core::Result;
use std::sync::Arc;

pub struct Timer {
    start: TimingEvent,
    stop: TimingEvent,
}

impl Timer {
    pub fn new(stream: &Arc<Stream>) -> Result<Self> {
        let device = stream.device().clone();
        let start = TimingEvent::new(device.clone())?;
        let stop = TimingEvent::new(device)?;
        Ok(Self { start, stop })
    }

    #[inline]
    pub fn begin(&self, stream: &Arc<Stream>) -> Result<()> {
        self.start.record(stream).map_err(Into::into)
    }

    #[inline]
    pub fn end(&self, stream: &Arc<Stream>) -> Result<()> {
        self.stop.record(stream).map_err(Into::into)
    }

    /// Block host until `stop` has fired, then return elapsed milliseconds.
    pub fn elapsed_ms(&self) -> Result<f32> {
        self.stop.synchronize()?;
        TimingEvent::elapsed_ms(&self.start, &self.stop).map_err(Into::into)
    }

    pub fn time<R, F: FnOnce() -> Result<R>>(stream: &Arc<Stream>, f: F) -> Result<(R, f32)> {
        let t = Self::new(stream)?;
        t.begin(stream)?;
        let r = f()?;
        t.end(stream)?;
        Ok((r, t.elapsed_ms()?))
    }
}
