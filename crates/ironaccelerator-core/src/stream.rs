//! Streams and events. A stream is the unit of asynchronous submission;
//! every backend has one (CUDA streams, HIP streams, Metal command queues,
//! QNN graph executors).

use crate::error::Result;

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

pub trait Stream: Send + Sync {
    /// Submit a no-op barrier and return when GPU has caught up.
    fn synchronize(&self) -> Result<()>;
    /// Whether all previously enqueued work has completed.
    fn is_idle(&self) -> bool;
    /// Record an event on this stream.
    fn record(&self) -> Result<Box<dyn Event>>;
}

pub trait Event: Send + Sync {
    /// Block until the recorded point is reached.
    fn wait(&self) -> Result<()>;
    /// Whether the event has fired.
    fn is_complete(&self) -> bool;
    /// Elapsed time in milliseconds between this event and `start`.
    fn elapsed_ms(&self, start: &dyn Event) -> Result<f32>;
}
