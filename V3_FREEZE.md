# V3 Frozen Candidate Specification: E2-Lite (Lexical Only)

**Date**: 2026-09-22  
**Status**: 100% FROZEN CANDIDATE FOR HOLDOUT-2 CONFIRMATION  
**Candidate Name**: `E2-Lite` (Slot-Conditioned Lexical Targeted Descent)  
**Upstream Frozen Components**:
- Global FIB Vector Retrieval (Top-5)
- Control Plane Shadow Candidate Plane (Top-20 RIB)
- Document Prefix Resolution & Scoring
- Lane Detection (`FAST_PATH`, `TEMPORAL_BASIS`, `COMPOSITE_EVIDENCE`)
- Hierarchical Parent Lift / Graph Resolution Topology
- Admission Gate: `E1 Composer` (Coverage-Preserving Evidence Composition)
- Downstream Synthesis: B0 Raw Answer Prompt

---

## 1. Minimal Sufficient Architecture Rationale

In Stage B.1 Channel Ablation on Dev-216 ($N=216$):
- **Semantic Channel Contribution**: `SEMANTIC_UNIQUE_GOLD = 0`. The dense vector channel within the target document failed to retrieve any gold chunk that was not already captured by the lexical channel.
- **Lexical Channel Superiority**: `LEXICAL_UNIQUE_GOLD = 4`. The lexical channel successfully located 4 gold chunk instances (in Q078 and Q089) that the semantic channel completely missed.
- **Dilution in Hybrid**: In `E2-Hybrid`, RRF rank fusion with the semantic channel diluted the top lexical candidates, causing a loss of 2 complete chains compared to pure lexical descent.
- **Minimalism**: Removing the semantic channel and RRF simplifies the architecture, speeds up descent, reduces dependencies, and strictly improves all retrieval metrics:
  - Candidate Pool Gold Recall: $6.02\% \to 6.94\%$
  - Final Gold Chunk Recall: $53.32\% \to 53.63\%$
  - Chain Completion: $74 \to 76$ complete chains ($35.19\%$)
  - Net Retrieval Rescues vs B0: $+2 \to +4$
  - Useful Evidence Evictions: **0** (100% safe)
  - End-to-End Accuracy: $74.07\% \to 74.54\%$ ($161/216$, $+2.32\text{pp}$ vs B0)
  - Multi-Hop Accuracy: $69.74\% \to 70.39\%$ ($107/152$, $+3.28\text{pp}$ vs B0)
  - Same-Evidence Flips: **0**

Under the core governing principle **"Keep only what earns its complexity"**, `E2-Lite (Lexical Only)` is formally selected and frozen.

---

## 2. Integrity Hashes & Commit Signatures

```yaml
git_commit: "4e5f40679086a5f403e0af42d120dcea79effc89"
frozen_candidate: "E2-Lite (Lexical Only)"
components:
  router:
    path: "src/routing/e2_descent_router.py"
    sha256: "946e59c96acc9c572f6edf95a99851b6d65e66c3b2e8f51b91270bf240e7fec5"
  descent_module:
    path: "src/composition/descent.py"
    sha256: "001497c30cc07d51995771834d2d3e0c98738ddf4517aadfafab6943d8b7fa0c"
  composer:
    path: "src/composition/composer.py"
    sha256: "224ef0e88364a969a7c919a4195d4a26ab7f86373a33d1c0625c7f12fa09291d"
  slots:
    path: "src/composition/slots.py"
    sha256: "e6d5b960c924e7d350d9cc8bb9eba8aaf070e3b4419f4ec0dbd8648efde9953f"
  coverage:
    path: "src/composition/coverage.py"
    sha256: "e75d4af5cae647c61fcfebea2b2ad3d657dca634c9f1273e3e47944fbdd3b7c5"
  prompt:
    path: "src/common/prompt.py"
    sha256: "cc9c8ad349c642275b233db5f9e823751735377c80e30f62cd3bf7c16d023961"
```

---

## 3. Operational Configuration & Retrieval Parameters

```yaml
retrieval:
  top_k_fib: 5
  shadow_rib_top_k: 20
  lane_detection:
    fast_path_threshold: 0.85
    lanes: ["FAST_PATH", "TEMPORAL_BASIS", "COMPOSITE_EVIDENCE"]
  descent:
    channel_mode: "lexical_only"
    fts_top_k: 3
    candidate_cap_per_doc: 3
    heading_bonus:
      structural_category: 0.010
      keyword_match: 0.005
  composition:
    max_evidence_chunks: 5
    max_replacements: 2
    coverage_threshold: 0.40
    eviction_penalty_prevention: true
  synthesis:
    prompt_mode: "RAW_B0"
    max_evidence_tokens: 4000
```

---

## 4. Holdout-2 Readiness Gate Checklist

- [x] **Minimal Architecture**: Semantic channel and RRF deleted; pure generic lexical descent retained.
- [x] **Statistical Funnel Standardized**: Denominators reconciled; zero discrepancy between A0 and E2.
- [x] **Composer Stability**: Useful Evidence Eviction strictly $= 0$; Admission Rate $= 100\%$.
- [x] **Zero Benchmark-Specific Heuristics**: Zero statute name strings, zero qid lookup tables.
- [x] **Prompt Frozen**: 100% original B0 answer prompt.
- [x] **Physical Isolation**: Holdout-1 remained sealed throughout; Holdout-2 has zero code overlap.

**FINAL GATE VERDICT**: **`READY FOR HOLDOUT-2`**
