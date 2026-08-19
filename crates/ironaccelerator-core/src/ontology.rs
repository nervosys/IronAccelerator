//! What this crate can be asked to do, and what each of those costs.
//!
//! IronAccelerator is a library, not a service. Nothing here is enforced: an
//! operation declared as consuming device memory will not be refused if the
//! caller has no authority to consume it, because there is no caller identity
//! at this layer to refuse. This document exists so that whatever *does* have
//! one -- an agent framework, a scheduler, a gateway -- can decide before
//! calling rather than discover afterwards.
//!
//! # Why this is hand-maintained
//!
//! The CLI tools in this stack derive their ontologies from clap, which knows
//! its own commands, so a drift test can walk the real surface and fail when
//! something is missing. Rust has no equivalent reflection over traits: there
//! is no way to enumerate a trait's methods at run time and compare them with
//! this list. The tests below therefore check internal consistency and the
//! claims that matter, not coverage. Adding a trait method without adding it
//! here will not fail a build, and that is a real limitation rather than an
//! oversight -- it is stated so a reader knows what this document is worth.
//!
//! # Guard-rails
//!
//! This crate prioritises throughput over guard-rails and says so in its own
//! preamble. That makes the `unchecked` flag below the most useful field for
//! an agent: it marks where the safety a caller would expect has been
//! deliberately traded away, and where the caller is the one holding it.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(feature = "std")]
use std::vec::Vec;

/// One thing a caller can ask this crate to do.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Operation {
    /// Stable identifier, `surface.method`.
    pub id: &'static str,
    /// The trait it belongs to.
    pub surface: &'static str,
    /// What it does, in one line.
    pub doc: &'static str,
    pub effects: Effects,
}

/// What invoking an operation costs.
///
/// Deliberately not a single `read_only` boolean. On a device, the authority
/// to use it and the cost of using it are the same resource, so the useful
/// question is not "does this mutate" but "what share of the device does this
/// take, and for how long".
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Effects {
    /// Takes device memory that no other caller can use until it is freed.
    pub allocates: bool,
    /// Occupies the device for as long as the submitted work runs.
    pub executes: bool,
    /// Compiles caller-supplied code for the device. Closer to loading a
    /// plugin than to allocating memory: the bytes become instructions.
    pub compiles: bool,
    /// Moves bytes between host and device. Cost scales with the data.
    pub transfers: bool,
    /// Blocks the calling thread until the device catches up.
    pub blocks: bool,
    /// Hands out a raw pointer, or otherwise drops a check the caller must
    /// then perform. Safe to call, dangerous to misuse.
    pub unchecked: bool,
}

impl Effects {
    const NONE: Self = Self {
        allocates: false,
        executes: false,
        compiles: false,
        transfers: false,
        blocks: false,
        unchecked: false,
    };

    /// Reads state that already exists and takes no share of the device.
    pub const fn discovery() -> Self {
        Self::NONE
    }

    /// True when the operation takes some share of the device.
    pub const fn consumes(&self) -> bool {
        self.allocates || self.executes || self.compiles || self.transfers
    }
}

const fn op(
    id: &'static str,
    surface: &'static str,
    doc: &'static str,
    effects: Effects,
) -> Operation {
    Operation {
        id,
        surface,
        doc,
        effects,
    }
}

/// Every operation this crate's trait surface offers.
pub fn operations() -> Vec<Operation> {
    vec![
        // -- discovery: reads what is already true -------------------------
        op(
            "backend.kind",
            "Backend",
            "Static identity of a compiled-in backend.",
            Effects::discovery(),
        ),
        op(
            "backend.is_available",
            "Backend",
            "Whether the backend's runtime libraries were found on this host.",
            Effects::discovery(),
        ),
        op(
            "backend.enumerate",
            "Backend",
            "Enumerate every visible device.",
            Effects::discovery(),
        ),
        op(
            "backend.capabilities",
            "Backend",
            "Coarse capability flags for one device, in the common space.",
            Effects::discovery(),
        ),
        op(
            "device.descriptor",
            "Device",
            "Static description of a device.",
            Effects::discovery(),
        ),
        op(
            "device.id",
            "Device",
            "The device's identifier.",
            Effects::discovery(),
        ),
        op(
            "kernel.name",
            "KernelLaunch",
            "The kernel's name.",
            Effects::discovery(),
        ),
        op(
            "kernel.occupancy_hint",
            "KernelLaunch",
            "Suggested occupancy for launch geometry.",
            Effects::discovery(),
        ),
        op(
            "allocation.kind",
            "Allocation",
            "Which memory space an allocation lives in.",
            Effects::discovery(),
        ),
        op(
            "allocation.len_bytes",
            "Allocation",
            "Size of an allocation.",
            Effects::discovery(),
        ),
        op(
            "memory.kind",
            "MemoryPool",
            "Which memory space a pool serves.",
            Effects::discovery(),
        ),
        op(
            "compute.buffer_len",
            "ComputeDevice",
            "Size of a device buffer.",
            Effects::discovery(),
        ),
        op(
            "stream.is_idle",
            "Stream",
            "Whether a stream has outstanding work.",
            Effects::discovery(),
        ),
        op(
            "stream.record",
            "Stream",
            "Record an event on a stream.",
            Effects::discovery(),
        ),
        op(
            "event.is_complete",
            "Event",
            "Whether a recorded event has fired.",
            Effects::discovery(),
        ),
        op(
            "event.elapsed_ms",
            "Event",
            "Time between two recorded events.",
            Effects::discovery(),
        ),
        // -- consumption: takes a share of the device ----------------------
        op(
            "memory.alloc",
            "MemoryPool",
            "Allocate a device buffer.",
            Effects {
                allocates: true,
                ..Effects::NONE
            },
        ),
        op(
            "compute.device_buffer",
            "ComputeDevice",
            "Allocate a device buffer through the compute surface.",
            Effects {
                allocates: true,
                ..Effects::NONE
            },
        ),
        op(
            "compute.upload",
            "ComputeDevice",
            "Copy host data into a device buffer.",
            Effects {
                allocates: true,
                transfers: true,
                ..Effects::NONE
            },
        ),
        op(
            "compute.download",
            "ComputeDevice",
            "Copy device data back to the host.",
            Effects {
                transfers: true,
                blocks: true,
                ..Effects::NONE
            },
        ),
        op(
            "compute.pipeline",
            "ComputeDevice",
            "Build a compute pipeline from caller-supplied code.",
            Effects {
                compiles: true,
                ..Effects::NONE
            },
        ),
        op(
            "compute.dispatch",
            "ComputeDevice",
            "Dispatch a compute pipeline.",
            Effects {
                executes: true,
                ..Effects::NONE
            },
        ),
        op(
            "kernel.launch",
            "KernelLaunch",
            "Launch a kernel on a stream.",
            Effects {
                executes: true,
                ..Effects::NONE
            },
        ),
        // -- lifecycle and synchronisation ---------------------------------
        op(
            "memory.trim",
            "MemoryPool",
            "Return unused pooled memory to the driver.",
            Effects {
                blocks: true,
                ..Effects::NONE
            },
        ),
        op(
            "stream.synchronize",
            "Stream",
            "Block until a stream's queued work completes.",
            Effects {
                blocks: true,
                ..Effects::NONE
            },
        ),
        op(
            "event.wait",
            "Event",
            "Block until an event completes.",
            Effects {
                blocks: true,
                ..Effects::NONE
            },
        ),
        // -- the traded-away guard-rails -----------------------------------
        op(
            "allocation.as_ptr",
            "Allocation",
            "Raw device pointer. Backends that virtualise pointers return a \
             handle cast to a raw pointer, so it is not always dereferenceable \
             on the host.",
            Effects {
                unchecked: true,
                ..Effects::NONE
            },
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique() {
        let ops = operations();
        let mut seen = std::collections::BTreeSet::new();
        for o in &ops {
            assert!(seen.insert(o.id), "duplicate operation id: {}", o.id);
        }
        assert_eq!(seen.len(), ops.len());
    }

    #[test]
    fn ids_are_surface_qualified() {
        for o in operations() {
            assert!(
                o.id.contains('.'),
                "`{}` is not qualified by its surface",
                o.id
            );
        }
    }

    /// The operations that take a share of the device declare it. Pinned by
    /// name because this is the claim a scheduler acts on.
    #[test]
    fn consuming_operations_declare_consumption() {
        let ops = operations();
        for id in [
            "memory.alloc",
            "compute.device_buffer",
            "compute.upload",
            "compute.download",
            "compute.pipeline",
            "compute.dispatch",
            "kernel.launch",
        ] {
            let o = ops
                .iter()
                .find(|o| o.id == id)
                .unwrap_or_else(|| panic!("`{id}` is missing from the ontology"));
            assert!(
                o.effects.consumes(),
                "`{id}` takes device resources but declares none"
            );
        }
    }

    /// Discovery never claims to consume.
    ///
    /// The guard against the opposite failure: declaring everything as
    /// consuming would satisfy the test above while making the field useless,
    /// since a scheduler that cannot tell enumeration from dispatch gains
    /// nothing from either.
    #[test]
    fn discovery_operations_consume_nothing() {
        let ops = operations();
        for id in [
            "backend.kind",
            "backend.is_available",
            "backend.enumerate",
            "backend.capabilities",
            "device.descriptor",
            "compute.buffer_len",
        ] {
            let o = ops.iter().find(|o| o.id == id).expect("declared");
            assert!(
                !o.effects.consumes(),
                "`{id}` only reads but declares consumption"
            );
        }
    }

    /// Compiling caller-supplied code is declared distinctly from executing
    /// it. A caller gating only on execution would let arbitrary program text
    /// reach the device unexamined.
    #[test]
    fn pipeline_declares_compilation_not_merely_execution() {
        let ops = operations();
        let pipeline = ops
            .iter()
            .find(|o| o.id == "compute.pipeline")
            .expect("declared");
        assert!(pipeline.effects.compiles);
        assert!(
            !pipeline.effects.executes,
            "building a pipeline does not itself run it; dispatch does"
        );

        let dispatch = ops
            .iter()
            .find(|o| o.id == "compute.dispatch")
            .expect("declared");
        assert!(dispatch.effects.executes);
        assert!(!dispatch.effects.compiles);
    }

    /// The one operation that hands out a raw pointer says so, and it stays
    /// the only one without someone noticing.
    #[test]
    fn raw_pointer_access_is_flagged_unchecked() {
        let ops = operations();
        let unchecked: Vec<&str> = ops
            .iter()
            .filter(|o| o.effects.unchecked)
            .map(|o| o.id)
            .collect();
        assert_eq!(
            unchecked,
            vec!["allocation.as_ptr"],
            "the set of unchecked operations changed; each addition trades \
             away a guard-rail and should be reviewed as one"
        );
    }
}
