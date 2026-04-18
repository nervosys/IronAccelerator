//! # IronAccelerator Ontology
//!
//! A machine-readable, queryable knowledge base describing every accelerator
//! family, workload class, kernel strategy, and the cross-cutting relations
//! between them. Designed to be consumed by **agents** (LLM tool-use loops,
//! auto-tuners, schedulers) so they can answer questions like:
//!
//! > *"On Hopper, for an FP8 GEMM with M=N=K=8192, what strategies exist,
//! >  and which one minimises HBM traffic during prefill?"*
//!
//! ## Entities
//!
//! | Entity         | Type        | Examples                                          |
//! |----------------|-------------|---------------------------------------------------|
//! | [`HardwareNode`] | accelerator | `nvidia.h100`, `amd.mi300x`, `apple.m3-max`     |
//! | [`WorkloadClass`] | workload   | `gemm`, `flash-attention`, `paged-attention`     |
//! | [`StrategyClass`] | algorithm  | `cublaslt-epilogue`, `cutlass-tile`, `triton-jit`|
//! | [`Optimization`]  | technique  | `kv-cache-paging`, `2:4-sparsity`, `fp8-recipe`  |
//! | [`Edge`]          | relation   | `supports`, `prefers`, `requires`                |
//!
//! Every entity has a stable, dotted ID so agents can address them durably
//! across runs and across IronAccelerator versions.
//!
//! ## Discovery
//!
//! ```no_run
//! use ironaccelerator_ontology::Ontology;
//! let o = Ontology::global();
//! let plan = o.recommend(
//!     ironaccelerator_core::WorkloadKind::FlashAttention,
//!     "nvidia.h100",
//! );
//! for s in plan { println!("{} -> {}", s.id, s.rationale); }
//! ```

use core::fmt;
use ironaccelerator_core::{BackendKind, WorkloadKind};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod edges;
pub mod hardware;
pub mod optimizations;
pub mod query;
pub mod strategies;
pub mod workloads;

pub use query::{Explanation, FilterSpec, RankBy};

/// Stable dotted identifier — `vendor.family.model` for hardware,
/// `family.kind` for workloads, `library.kernel` for strategies.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Id(pub String);

impl<S: Into<String>> From<S> for Id {
    fn from(s: S) -> Self { Self(s.into()) }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.0) }
}

/// One node in the hardware sub-graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareNode {
    pub id: Id,
    pub backend: BackendKind,
    pub vendor: String,
    pub family: String,
    pub arch: String,
    pub launch_year: u16,
    pub fp16_tflops: Option<f32>,
    pub fp8_tflops: Option<f32>,
    pub mem_bandwidth_gbs: Option<f32>,
    pub tags: Vec<String>,
    pub notes: String,
}

/// A class of workloads (algorithm-level).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadClass {
    pub id: Id,
    pub kind: WorkloadKind,
    pub description: String,
    /// Roofline-style hint: `"compute"`, `"memory"`, `"latency"`.
    pub bound_by: String,
    pub tags: Vec<String>,
}

/// An implementation strategy (a *how*).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyClass {
    pub id: Id,
    pub library: String,
    pub backends: Vec<BackendKind>,
    pub min_arch: HashMap<String, String>, // backend.name() -> min arch
    pub workloads: Vec<Id>,                // applicable WorkloadClass ids
    pub optimizations: Vec<Id>,            // referenced Optimization ids
    pub description: String,
    pub rationale: String,
    pub jit: bool,
    pub open_source: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Optimization {
    pub id: Id,
    pub description: String,
    pub savings: String, // e.g. "~2x bandwidth", "~30% memory"
    pub requires: Vec<String>, // capability strings
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub from: Id,
    pub to: Id,
    pub relation: Relation,
    pub weight: f32,
    pub note: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Relation {
    Supports,
    Prefers,
    Requires,
    Replaces,
    OptimizedFor,
    BoundBy,
}

/// A single recommendation row returned to an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub id: Id,
    pub strategy: StrategyClass,
    pub rationale: String,
    pub score: f32,
}

/// The full ontology graph. Indexed for O(1) lookup by id.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Ontology {
    pub hardware: HashMap<Id, HardwareNode>,
    pub workloads: HashMap<Id, WorkloadClass>,
    pub strategies: HashMap<Id, StrategyClass>,
    pub optimizations: HashMap<Id, Optimization>,
    pub edges: Vec<Edge>,
}

impl Ontology {
    /// The compiled-in default ontology.
    pub fn global() -> &'static Ontology {
        &GLOBAL
    }

    pub fn build_default() -> Ontology {
        let mut o = Ontology::default();
        hardware::populate(&mut o);
        workloads::populate(&mut o);
        strategies::populate(&mut o);
        optimizations::populate(&mut o);
        edges::populate(&mut o);
        o
    }

    /// Return strategies applicable to a workload class on a hardware id,
    /// ranked by a simple weight = backend match * arch match * edge weight.
    pub fn recommend(&self, kind: WorkloadKind, hardware_id: &str) -> Vec<Recommendation> {
        let hw = match self.hardware.get(&Id(hardware_id.to_string())) {
            Some(h) => h,
            None => return Vec::new(),
        };

        let mut out = Vec::new();
        for s in self.strategies.values() {
            if !s.backends.contains(&hw.backend) {
                continue;
            }
            if !s.workloads.iter().any(|wid| {
                self.workloads
                    .get(wid)
                    .map(|w| w.kind == kind)
                    .unwrap_or(false)
            }) {
                continue;
            }

            let edge_score = self
                .edges
                .iter()
                .filter(|e| e.from == s.id && e.to == hw.id)
                .map(|e| e.weight)
                .sum::<f32>()
                .max(0.5);

            out.push(Recommendation {
                id: s.id.clone(),
                strategy: s.clone(),
                rationale: s.rationale.clone(),
                score: edge_score,
            });
        }
        out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(core::cmp::Ordering::Equal));
        out
    }

    /// Serialise the entire graph as JSON — the canonical format consumed by
    /// external agent tooling.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("ontology serialises cleanly")
    }

    /// List every entity id under a given namespace prefix
    /// (e.g. `"strategy."`, `"hardware.nvidia."`). Useful for agent menus.
    pub fn ids_with_prefix(&self, prefix: &str) -> Vec<Id> {
        let mut v = Vec::new();
        v.extend(self.hardware.keys().filter(|i| i.0.starts_with(prefix)).cloned());
        v.extend(self.workloads.keys().filter(|i| i.0.starts_with(prefix)).cloned());
        v.extend(self.strategies.keys().filter(|i| i.0.starts_with(prefix)).cloned());
        v.extend(self.optimizations.keys().filter(|i| i.0.starts_with(prefix)).cloned());
        v
    }

    /// Hardware filtered by tag (e.g. `"datacenter"`, `"unified-memory"`).
    pub fn hardware_by_tag<'a>(&'a self, tag: &'a str) -> impl Iterator<Item = &'a HardwareNode> + 'a {
        self.hardware.values().filter(move |h| h.tags.iter().any(|t| t == tag))
    }

    /// Strategies that target a given backend.
    pub fn strategies_for_backend<'a>(&'a self, backend: BackendKind)
        -> impl Iterator<Item = &'a StrategyClass> + 'a
    {
        self.strategies.values().filter(move |s| s.backends.contains(&backend))
    }

    /// Strategies that implement a given workload kind.
    pub fn strategies_for_workload<'a>(&'a self, kind: WorkloadKind)
        -> impl Iterator<Item = &'a StrategyClass> + 'a
    {
        self.strategies.values().filter(move |s| {
            s.workloads.iter().any(|wid| {
                self.workloads.get(wid).map(|w| w.kind == kind).unwrap_or(false)
            })
        })
    }

    /// Optimisations referenced by a strategy id, resolved through the index.
    pub fn optimizations_of<'a>(&'a self, strategy: &Id)
        -> impl Iterator<Item = &'a Optimization> + 'a
    {
        let opt_ids: Vec<Id> = self.strategies.get(strategy)
            .map(|s| s.optimizations.clone()).unwrap_or_default();
        opt_ids.into_iter().filter_map(move |id| self.optimizations.get(&id))
    }

    /// All edges originating at `from` (any relation).
    pub fn edges_from<'a>(&'a self, from: &'a Id) -> impl Iterator<Item = &'a Edge> + 'a {
        self.edges.iter().filter(move |e| &e.from == from)
    }

    /// All edges incident to `to` (any relation).
    pub fn edges_to<'a>(&'a self, to: &'a Id) -> impl Iterator<Item = &'a Edge> + 'a {
        self.edges.iter().filter(move |e| &e.to == to)
    }

    /// Internal consistency: every edge endpoint and every cross-reference
    /// resolves to a known entity. Returns the list of dangling references.
    pub fn validate(&self) -> Vec<String> {
        let mut errs = Vec::new();
        let known = |id: &Id| -> bool {
            self.hardware.contains_key(id)
                || self.workloads.contains_key(id)
                || self.strategies.contains_key(id)
                || self.optimizations.contains_key(id)
        };
        for e in &self.edges {
            if !known(&e.from) { errs.push(format!("dangling edge.from: {}", e.from)); }
            if !known(&e.to)   { errs.push(format!("dangling edge.to: {}",   e.to)); }
        }
        for s in self.strategies.values() {
            for w in &s.workloads {
                if !self.workloads.contains_key(w) {
                    errs.push(format!("strategy {} references unknown workload {}", s.id, w));
                }
            }
            for o in &s.optimizations {
                if !self.optimizations.contains_key(o) {
                    errs.push(format!("strategy {} references unknown optimization {}", s.id, o));
                }
            }
        }
        errs
    }

    /// Counts of each entity class — useful for agent debugging.
    pub fn stats(&self) -> Stats {
        Stats {
            hardware: self.hardware.len(),
            workloads: self.workloads.len(),
            strategies: self.strategies.len(),
            optimizations: self.optimizations.len(),
            edges: self.edges.len(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Stats {
    pub hardware: usize,
    pub workloads: usize,
    pub strategies: usize,
    pub optimizations: usize,
    pub edges: usize,
}

static GLOBAL: Lazy<Ontology> = Lazy::new(Ontology::build_default);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_ontology_is_internally_consistent() {
        let errs = Ontology::global().validate();
        assert!(errs.is_empty(), "dangling refs: {errs:#?}");
    }

    #[test]
    fn stats_are_nonzero() {
        let s = Ontology::global().stats();
        assert!(s.hardware > 10);
        assert!(s.workloads > 5);
        assert!(s.strategies > 10);
        assert!(s.optimizations > 10);
        assert!(s.edges > 20);
    }

    #[test]
    fn hardware_by_tag_finds_datacenter_parts() {
        let o = Ontology::global();
        let n = o.hardware_by_tag("datacenter").count();
        assert!(n >= 3, "expected several datacenter parts, got {n}");
    }

    #[test]
    fn strategies_for_backend_cuda_is_rich() {
        let o = Ontology::global();
        let n = o.strategies_for_backend(BackendKind::Cuda).count();
        assert!(n >= 8, "expected many CUDA strategies, got {n}");
    }

    #[test]
    fn recommend_hopper_flash_attention_prefers_v3() {
        let o = Ontology::global();
        let recs = o.recommend(WorkloadKind::FlashAttention, "hardware.nvidia.h100");
        assert!(!recs.is_empty());
        assert!(recs[0].id.0.contains("flashattn.v3"),
            "expected FA-v3 to rank first on H100, got {:?}", recs[0].id);
    }
}
