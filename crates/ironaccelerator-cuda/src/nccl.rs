//! NCCL safe wrapper.
//!
//! Single-process multi-GPU groups use [`CommGroup::from_sessions`], which
//! performs the classic `ncclGroupStart` / per-rank `ncclCommInitRank` /
//! `ncclGroupEnd` dance. Multi-process init goes through
//! [`CommGroup::from_rank`] with a shared [`Id`].

use crate::drv::{Device, DeviceBuf, Repr};
use crate::Session;
use iron_cuda_sys::nccl as sys;
use ironaccelerator_core::{Error, Result};
use std::any::TypeId;
use std::ffi::c_int;
use std::sync::Arc;

pub use sys::NcclUniqueId as Id;

fn fns() -> Result<&'static sys::NcclFns> {
    sys::fns().map_err(|e| Error::Other(Box::leak(
        format!("nccl not available: {e}").into_boxed_str())))
}

fn check(_op: &'static str, s: sys::NcclResult) -> Result<()> {
    if s.is_ok() { Ok(()) } else {
        Err(Error::Backend {
            backend: ironaccelerator_core::BackendKind::Cuda,
            code: (s as u32) as i64,
        })
    }
}

fn dtype_of<T: Repr>() -> Result<sys::NcclDataType> {
    let t = TypeId::of::<T>();
    Ok(if t == TypeId::of::<i8>()  { sys::NcclDataType::Int8 }
       else if t == TypeId::of::<u8>()  { sys::NcclDataType::Uint8 }
       else if t == TypeId::of::<i32>() { sys::NcclDataType::Int32 }
       else if t == TypeId::of::<u32>() { sys::NcclDataType::Uint32 }
       else if t == TypeId::of::<i64>() { sys::NcclDataType::Int64 }
       else if t == TypeId::of::<u64>() { sys::NcclDataType::Uint64 }
       else if t == TypeId::of::<f32>() { sys::NcclDataType::Float32 }
       else if t == TypeId::of::<f64>() { sys::NcclDataType::Float64 }
       else { return Err(Error::Other("nccl: unsupported element type (use i8/u8/i32/u32/i64/u64/f32/f64)")); })
}

#[derive(Copy, Clone, Debug)]
pub enum ReduceOp { Sum, Prod, Max, Min, Avg }

impl ReduceOp {
    fn to_sys(self) -> sys::NcclRedOp {
        match self {
            ReduceOp::Sum  => sys::NcclRedOp::Sum,
            ReduceOp::Prod => sys::NcclRedOp::Prod,
            ReduceOp::Max  => sys::NcclRedOp::Max,
            ReduceOp::Min  => sys::NcclRedOp::Min,
            ReduceOp::Avg  => sys::NcclRedOp::Avg,
        }
    }
}

/// One rank's communicator handle plus the device it's bound to.
pub struct RankComm {
    comm: sys::NcclComm,
    device: Arc<Device>,
}

unsafe impl Send for RankComm {}
unsafe impl Sync for RankComm {}

impl Drop for RankComm {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe { let _ = (f.ncclCommDestroy)(self.comm); }
        }
    }
}

pub struct CommGroup {
    comms: Vec<RankComm>,
}

impl CommGroup {
    /// Generate a fresh NCCL unique ID (one per training job).
    pub fn new_id() -> Result<Id> {
        let f = fns()?;
        let mut id = Id::default();
        unsafe { check("ncclGetUniqueId", (f.ncclGetUniqueId)(&mut id))?; }
        Ok(id)
    }

    /// Single-process multi-GPU init. Creates one communicator per session,
    /// numbered in argument order.
    pub fn from_sessions(sessions: &[&Session]) -> Result<Self> {
        let f = fns()?;
        let n = sessions.len() as c_int;
        if n == 0 {
            return Err(Error::Other("nccl::from_sessions: empty session list"));
        }
        let id = Self::new_id()?;
        let mut comms: Vec<RankComm> = Vec::with_capacity(sessions.len());
        unsafe { check("ncclGroupStart", (f.ncclGroupStart)())?; }
        for (rank, s) in sessions.iter().enumerate() {
            s.device().bind()?;
            let mut c = sys::NcclComm::default();
            unsafe {
                check("ncclCommInitRank",
                      (f.ncclCommInitRank)(&mut c, n, id, rank as c_int))?;
            }
            comms.push(RankComm { comm: c, device: s.device().clone() });
        }
        unsafe { check("ncclGroupEnd", (f.ncclGroupEnd)())?; }
        Ok(Self { comms })
    }

    /// Multi-process init: one rank per process, using a shared `id`.
    pub fn from_rank(session: &Session, rank: usize, world_size: usize, id: Id) -> Result<Self> {
        let f = fns()?;
        session.device().bind()?;
        let mut c = sys::NcclComm::default();
        unsafe {
            check("ncclCommInitRank",
                  (f.ncclCommInitRank)(&mut c, world_size as c_int, id, rank as c_int))?;
        }
        Ok(Self { comms: vec![RankComm { comm: c, device: session.device().clone() }] })
    }

    #[inline] pub fn ranks(&self) -> usize { self.comms.len() }

    fn pick(&self, rank: usize) -> Result<&RankComm> {
        self.comms.get(rank).ok_or(Error::Other("nccl: rank out of range"))
    }

    pub fn all_reduce<T: Repr>(
        &self, rank: usize, send: &DeviceBuf<T>, recv: &mut DeviceBuf<T>, op: ReduceOp,
        stream: &crate::drv::Stream,
    ) -> Result<()> {
        let f = fns()?;
        let rc = self.pick(rank)?;
        rc.device.bind()?;
        unsafe {
            check("ncclAllReduce", (f.ncclAllReduce)(
                send.device_ptr() as *const _, recv.device_ptr() as *mut _,
                send.len(), dtype_of::<T>()?, op.to_sys(), rc.comm, stream.raw(),
            ))
        }
    }

    pub fn all_reduce_in_place<T: Repr>(
        &self, rank: usize, buf: &mut DeviceBuf<T>, op: ReduceOp, stream: &crate::drv::Stream,
    ) -> Result<()> {
        let f = fns()?;
        let rc = self.pick(rank)?;
        rc.device.bind()?;
        unsafe {
            check("ncclAllReduce", (f.ncclAllReduce)(
                buf.device_ptr() as *const _, buf.device_ptr() as *mut _,
                buf.len(), dtype_of::<T>()?, op.to_sys(), rc.comm, stream.raw(),
            ))
        }
    }

    pub fn all_gather<T: Repr>(
        &self, rank: usize, send: &DeviceBuf<T>, recv: &mut DeviceBuf<T>,
        stream: &crate::drv::Stream,
    ) -> Result<()> {
        let f = fns()?;
        let rc = self.pick(rank)?;
        rc.device.bind()?;
        unsafe {
            check("ncclAllGather", (f.ncclAllGather)(
                send.device_ptr() as *const _, recv.device_ptr() as *mut _,
                send.len(), dtype_of::<T>()?, rc.comm, stream.raw(),
            ))
        }
    }

    pub fn broadcast<T: Repr>(
        &self, rank: usize, root: i32, send: &DeviceBuf<T>, recv: &mut DeviceBuf<T>,
        stream: &crate::drv::Stream,
    ) -> Result<()> {
        let f = fns()?;
        let rc = self.pick(rank)?;
        rc.device.bind()?;
        unsafe {
            check("ncclBroadcast", (f.ncclBroadcast)(
                send.device_ptr() as *const _, recv.device_ptr() as *mut _,
                send.len(), dtype_of::<T>()?, root, rc.comm, stream.raw(),
            ))
        }
    }

    pub fn reduce_scatter<T: Repr>(
        &self, rank: usize, send: &DeviceBuf<T>, recv: &mut DeviceBuf<T>, op: ReduceOp,
        stream: &crate::drv::Stream,
    ) -> Result<()> {
        let f = fns()?;
        let rc = self.pick(rank)?;
        rc.device.bind()?;
        unsafe {
            check("ncclReduceScatter", (f.ncclReduceScatter)(
                send.device_ptr() as *const _, recv.device_ptr() as *mut _,
                recv.len(), dtype_of::<T>()?, op.to_sys(), rc.comm, stream.raw(),
            ))
        }
    }

    /// Wrap a closure in `ncclGroupStart`/`ncclGroupEnd` — required when
    /// issuing multiple collectives across ranks from a single thread.
    pub fn group<F: FnOnce() -> Result<()>>(f: F) -> Result<()> {
        let n = fns()?;
        unsafe { check("ncclGroupStart", (n.ncclGroupStart)())?; }
        let r = f();
        unsafe { check("ncclGroupEnd", (n.ncclGroupEnd)())?; }
        r
    }
}

pub fn world_from_all_devices() -> Result<(Vec<Arc<Session>>, CommGroup)> {
    let n = Device::count()?;
    let sessions: Vec<Arc<Session>> = (0..n)
        .map(|ord| Session::new(ord).map(Arc::new))
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<&Session> = sessions.iter().map(|s| s.as_ref()).collect();
    let group = CommGroup::from_sessions(&refs)?;
    Ok((sessions, group))
}
