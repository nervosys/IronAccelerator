//! Structured queries against the ontology.
//!
//! Where [`Ontology::recommend`](crate::Ontology::recommend) returns a flat
//! ranked list, this module exposes:
//!
//! - [`FilterSpec`] — declarative filter (backend / workload / capability /
//!   open-source-only / no-JIT) matching the same shape as
//!   [`StrategyHint`](ironaccelerator_core::StrategyHint).
//! - [`Ontology::query`] — apply a filter and return matching strategies.
//! - [`Ontology::explain`] — walk strategy → workload → optimisations → edges
//!   for a hardware id, producing an agent-readable [`Explanation`].

use crate::{Edge, HardwareNode, Id, Ontology, Optimization, StrategyClass, WorkloadClass};
use ironaccelerator_core::{BackendKind, WorkloadKind};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FilterSpec {
    pub backend: Option<BackendKind>,
    pub workload: Option<WorkloadKind>,
    pub hardware_id: Option<String>,
    pub forbid_jit: bool,
    pub require_open_source: bool,
    /// Substring required to appear in `tags` of either the strategy's
    /// workloads or the hardware node.
    pub tag_contains: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub enum RankBy {
    #[default]
    EdgeWeight,
    /// Rank by `(edge_weight + 0.1 * #optimizations)`. Encourages strategies
    /// that bundle many optimisations.
    OptimisationCount,
    /// Stable alphabetical for reproducible test output.
    Name,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Explanation {
    pub strategy: StrategyClass,
    pub workloads: Vec<WorkloadClass>,
    pub optimizations: Vec<Optimization>,
    pub matched_hardware: Option<HardwareNode>,
    pub edges: Vec<Edge>,
    /// Pre-rendered narrative ready to be returned to an LLM tool call.
    pub narrative: String,
}

impl Ontology {
    /// Apply a [`FilterSpec`] and return matching strategy ids ranked by
    /// `rank_by`.
    pub fn query(&self, filter: &FilterSpec, rank_by: RankBy) -> Vec<Id> {
        let mut hits: Vec<(&StrategyClass, f32)> = self
            .strategies
            .values()
            .filter(|s| {
                if filter.forbid_jit && s.jit { return false; }
                if filter.require_open_source && !s.open_source { return false; }
                if let Some(b) = filter.backend {
                    if !s.backends.contains(&b) { return false; }
                }
                if let Some(k) = filter.workload {
                    let ok = s.workloads.iter().any(|wid| {
                        self.workloads.get(wid).map(|w| w.kind == k).unwrap_or(false)
                    });
                    if !ok { return false; }
                }
                if let Some(tag) = &filter.tag_contains {
                    let strategy_has = s.workloads.iter().any(|wid| {
                        self.workloads.get(wid)
                            .map(|w| w.tags.iter().any(|t| t.contains(tag.as_str())))
                            .unwrap_or(false)
                    });
                    let hw_has = filter.hardware_id.as_ref()
                        .and_then(|h| self.hardware.get(&Id(h.clone())))
                        .map(|h| h.tags.iter().any(|t| t.contains(tag.as_str())))
                        .unwrap_or(false);
                    if !strategy_has && !hw_has { return false; }
                }
                true
            })
            .map(|s| {
                let score = match rank_by {
                    RankBy::Name => 0.0,
                    RankBy::EdgeWeight => filter
                        .hardware_id
                        .as_ref()
                        .map(|h| self.weight_between(&s.id, &Id(h.clone())))
                        .unwrap_or(1.0),
                    RankBy::OptimisationCount => {
                        let w = filter.hardware_id.as_ref()
                            .map(|h| self.weight_between(&s.id, &Id(h.clone())))
                            .unwrap_or(1.0);
                        w + 0.1 * s.optimizations.len() as f32
                    }
                };
                (s, score)
            })
            .collect();

        match rank_by {
            RankBy::Name => hits.sort_by(|a, b| a.0.id.0.cmp(&b.0.id.0)),
            _ => hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal)),
        }

        hits.into_iter().map(|(s, _)| s.id.clone()).collect()
    }

    /// Sum of edge weights from `strategy` to `hardware`.
    fn weight_between(&self, strategy: &Id, hardware: &Id) -> f32 {
        self.edges.iter()
            .filter(|e| &e.from == strategy && &e.to == hardware)
            .map(|e| e.weight).sum::<f32>()
            .max(0.5)
    }

    /// Build a full [`Explanation`] for `(strategy, hardware)`.
    pub fn explain(&self, strategy: &str, hardware: Option<&str>) -> Option<Explanation> {
        let s = self.strategies.get(&Id(strategy.into()))?.clone();
        let workloads: Vec<WorkloadClass> = s.workloads.iter()
            .filter_map(|w| self.workloads.get(w).cloned()).collect();
        let optimizations: Vec<Optimization> = s.optimizations.iter()
            .filter_map(|o| self.optimizations.get(o).cloned()).collect();
        let matched_hardware = hardware
            .and_then(|h| self.hardware.get(&Id(h.into())).cloned());
        let edges: Vec<Edge> = match &matched_hardware {
            Some(h) => self.edges.iter()
                .filter(|e| e.from == s.id && e.to == h.id)
                .cloned().collect(),
            None => self.edges.iter()
                .filter(|e| e.from == s.id).cloned().collect(),
        };

        let mut n = String::new();
        n.push_str(&format!("Strategy `{}` ({}) — {}\n", s.id, s.library, s.description));
        n.push_str(&format!("Why: {}\n", s.rationale));
        if !workloads.is_empty() {
            n.push_str("Applies to: ");
            n.push_str(&workloads.iter().map(|w| w.id.0.as_str())
                .collect::<Vec<_>>().join(", "));
            n.push('\n');
        }
        if !optimizations.is_empty() {
            n.push_str("Optimisations:\n");
            for o in &optimizations {
                n.push_str(&format!("  - {}: {} ({})\n", o.id, o.description, o.savings));
            }
        }
        if let Some(hw) = &matched_hardware {
            n.push_str(&format!("On {}: {}\n", hw.id, hw.notes));
            for e in &edges {
                n.push_str(&format!("  · {:?} (w={:.1}) {}\n", e.relation, e.weight, e.note));
            }
        }
        if s.jit { n.push_str("Note: requires runtime JIT compilation.\n"); }
        if !s.open_source { n.push_str("Note: closed-source vendor library.\n"); }

        Some(Explanation { strategy: s, workloads, optimizations, matched_hardware, edges, narrative: n })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_by_backend_and_workload() {
        let o = Ontology::global();
        let ids = o.query(&FilterSpec {
            backend: Some(BackendKind::Cuda),
            workload: Some(WorkloadKind::FlashAttention),
            ..Default::default()
        }, RankBy::Name);
        assert!(ids.iter().any(|i| i.0 == "strategy.flashattn.v3"));
        assert!(ids.iter().any(|i| i.0 == "strategy.flashattn.v2"));
        // Should not include AMD-only strategies for a CUDA filter.
        assert!(!ids.iter().any(|i| i.0 == "strategy.hipblaslt.fp8"));
    }

    #[test]
    fn forbid_jit_filters_triton() {
        let o = Ontology::global();
        let ids = o.query(&FilterSpec {
            backend: Some(BackendKind::Cuda),
            workload: Some(WorkloadKind::Gemm),
            forbid_jit: true,
            ..Default::default()
        }, RankBy::Name);
        assert!(!ids.iter().any(|i| i.0 == "strategy.triton.gemm"));
        assert!(ids.iter().any(|i| i.0 == "strategy.cublaslt.bf16"));
    }

    #[test]
    fn require_open_source_filters_cublas_family() {
        let o = Ontology::global();
        let ids = o.query(&FilterSpec {
            backend: Some(BackendKind::Cuda),
            workload: Some(WorkloadKind::Gemm),
            require_open_source: true,
            ..Default::default()
        }, RankBy::Name);
        assert!(!ids.iter().any(|i| i.0.starts_with("strategy.cublaslt")));
        assert!(ids.iter().any(|i| i.0 == "strategy.cutlass.tile"));
    }

    #[test]
    fn explain_produces_narrative() {
        let o = Ontology::global();
        let ex = o.explain("strategy.flashattn.v3", Some("hardware.nvidia.h100")).unwrap();
        assert!(!ex.narrative.is_empty());
        assert!(ex.narrative.contains("FlashAttention-3"));
        assert!(ex.matched_hardware.is_some());
        assert!(!ex.edges.is_empty());
    }

    #[test]
    fn ranking_prefers_high_edge_weight() {
        let o = Ontology::global();
        let ids = o.query(&FilterSpec {
            backend: Some(BackendKind::Cuda),
            workload: Some(WorkloadKind::FlashAttention),
            hardware_id: Some("hardware.nvidia.h100".into()),
            ..Default::default()
        }, RankBy::EdgeWeight);
        assert_eq!(ids.first().map(|i| i.0.as_str()), Some("strategy.flashattn.v3"));
    }
}
