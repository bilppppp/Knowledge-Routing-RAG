# Mechanism Holdout-3 Pre-Registration Protocol

**Pre-Registration Date**: 2026-09-22  
**Target Repository**: `Knowledge-Routing-RAG`  
**Status**: `PRE_REGISTRATION_FROZEN`  
**Investigation Goal**: Determine whether Graph Relationships / Knowledge Routing provides an independent causal increment beyond dense recall expansion, metadata/title resolution, local lexical retrieval, and coverage-aware composition across varying degrees of statutory explicitness (Explicit, Partial, Implicit).

---

## 1. System Cryptographic Hashes

All code components are frozen prior to running Mechanism Holdout-3 confirmation. No code modifications, hyperparameter adjustments, or prompt changes are permitted during or after execution.

```text
================================================================================
COMPONENT SHA256 HASH RECORD:
--------------------------------------------------------------------------------
Corpus Manifest (data/manifests/corpus_manifest.json):
  bb21f21fb3f4785c23ac82e18d926e5185252d498d1f0a1da26dad8325add47f

Chunks Store (data/chunks.jsonl):
  d3c054be36cd5c805b9d6f0612aa2771990aa3bd890551ca72579d9d3f32513c

Graph SQLite Store (data/knowledge_lsdb.sqlite):
  a1f5881acf21d0db2b3803f1a16f1341e01324164bec16448bf018ab4446ebb3

E2-Descent Router System (src/routing/e2_descent_router.py):
  946e59c96acc9c572f6edf95a99851b6d65e66c3b2e8f51b91270bf240e7fec5

C7 Clean Router & Alias Table (src/routing/c7_clean_router.py):
  4892ce9e1251d5e9641ead47ad771179ef2268a151f5a559411223f3f80c0c0a

E1 Coverage-Aware Composer (src/composition/composer.py):
  224ef0e88364a969a7c919a4195d4a26ab7f86373a33d1c0625c7f12fa09291d

Slots Module (src/composition/slots.py):
  e6d5b960c924e7d350d9cc8bb9eba8aaf070e3b4419f4ec0dbd8648efde9953f

Targeted Descent Module (src/composition/descent.py):
  001497c30cc07d51995771834d2d3e0c98738ddf4517aadfafab6943d8b7fa0c

Common Prompt (src/common/prompt.py):
  cc9c8ad349c642275b233db5f9e823751735377c80e30f62cd3bf7c16d023961

Search Service (src/services/search.py):
  aaf449a539f68d68d328a5b6e652d8a80235c26c697a11ca88c6c06c5cd83a8e

LLM Service (src/services/llm.py):
  f01554a078e8abe23d609fa909d6819e2b9ebe44dde0c72433e704c9342a668a

Evaluator Metrics (src/evaluation/metrics.py):
  6d455ffcf3ff5dcc3bbbe173bd0c18cb2385da05bd8670aeb492d037c763ebde

Mechanism Holdout-3 Gold Dataset (benchmark/holdout3/gold.jsonl):
  a8c9cc2329a8853901ca12c4b8ed8f8c322e196c0a9d572b67d33bfc9ae66051
================================================================================
```

---

## 2. Experimental Systems Specification

All 6 systems share:
- Target Corpus: `D100` (Corrected Frozen Corpus)
- Embedding Model: `bge-large-zh-v1.5` (Cosine similarity, 1024-dim)
- FTS Index: SQLite FTS5 (`chunks_fts`)
- Evidence Budget: Maximum 5 chunks ($k=5$), Maximum 4,000 evidence tokens
- Downstream Synthesis: Raw B0 Prompt (`SYSTEM_PROMPT` in `src/common/prompt.py`)
- Random Seeds: 101, 202, 303 for 3 independent paired generation and judge runs
- **Strict No-Shortcut Mandate**: Every system for every seed undergoes fresh, independent LLM generation and evaluation. No answer or judge copying is permitted even when evidence chunk IDs match.

### Systems Under Test

1. **S0 — Dense Top-5 (Historical Baseline B0)**:
   - Query -> Dense Vector Search -> Top-5 Chunks -> Generator.
2. **S1 — Dense Top-20 + Composer**:
   - Query -> Dense Vector Search -> Top-20 Chunks.
   - Initial 5 as base; remaining 15 as candidate pool.
   - E1 Coverage-Aware Composer selects final 5 chunks -> Generator.
3. **S2 — Metadata / Title Routing + Local BM25**:
   - Question -> Deterministic statutory entity/title extraction via frozen alias table.
   - Resolved documents -> Document-local FTS/BM25 retrieval (Top-3 per doc).
   - If no title matches (e.g. Q-I), fallback to global FTS.
   - E1 Coverage-Aware Composer selects final 5 chunks -> Generator.
   - **Zero graph edges, zero neighbor traversal, zero relation lookups.**
4. **S3 — V3-NoGraph (Matched Ablation)**:
   - Frozen V3 pipeline with all graph-based candidate expansions completely disabled (`allow_graph=False`).
   - Shadow candidate plane and E2-Lite lexical targeted descent remain active.
   - E1 Coverage-Aware Composer selects final 5 chunks -> Generator.
5. **S4 — V3-TrueGraph (Frozen V3)**:
   - Frozen V3 pipeline with real Knowledge LSDB graph routing (`REFERENCES`, `BASED_ON`, `AMENDS`, `SUPERSEDES`).
6. **S5 — V3-ShuffledGraph (Control Baseline)**:
   - Frozen V3 pipeline utilizing a degree-matched, relation-preserving shuffled graph (`shuffled_lsdb`).
   - Random seed: 42. In-degree and out-degree distributions strictly matched to TrueGraph, with zero self-loops.

---

## 3. Pre-Registered Primary Causal Contrasts

1. **C1 — Wider Dense Recall**:
   $$\Delta_{\text{dense}} = S_1 - S_0$$
   *Question*: How much of historical V3 gains can be explained simply by wider dense candidate pooling ($k=20$) and coverage composition?
2. **C2 — Structured Metadata Baseline**:
   $$\Delta_{\text{meta}} = S_2 - S_0$$
   *Question*: How much gain is achieved purely by statutory title resolution + local BM25 without any graph?
3. **C3 — Graph Increment (Core Mechanism Test)**:
   $$\Delta_{\text{graph}} = S_4 - S_3$$
   *Question*: With shadow candidate plane, metadata descent, and composer held strictly identical, do real graph edges provide an independent accuracy increment?
4. **C4 — Graph Semantics**:
   $$\Delta_{\text{semantics}} = S_4 - S_5$$
   *Question*: Does true semantic relation traversal outperform a topology-matched random expansion?

---

## 4. Pre-Registered Verdict Decision Rules

- **`M-A — GRAPH INCREMENTAL VALUE CONFIRMED`**:
  - $S_4 > S_3$ and $S_4 > S_5$ (with statistical significance or non-overlapping bootstrap 95% CIs).
  - True Graph Causal Rescues > Graph Causal Regressions.
  - Increment is concentrated in **Partial (Q-P)** or **Implicit (Q-I)** strata (not restricted solely to Explicit).
- **`M-B — STRUCTURED RETRIEVAL CONFIRMED, GRAPH NOT NECESSARY`**:
  - $S_2 \approx S_4$ or $S_3 \approx S_4$ ($\Delta_{\text{graph}} \le 0$ or bootstrap CI covers 0).
  - Strong structured non-graph baselines ($S_1$ or $S_2$) significantly outperform $S_0$.
  - Knowledge Routing repositioned as structured/hierarchical retrieval rather than graph routing.
- **`M-C — STRONG BASELINES EXPLAIN THE HISTORICAL GAIN`**:
  - $S_1$ or $S_2$ equals or exceeds $S_4$.
  - Historical $+5.60\text{pp}$ gain is fully explained by baseline strength differences.
