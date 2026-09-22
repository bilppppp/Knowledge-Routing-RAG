# Release Notes: Knowledge-Routing-RAG — Experimental V3 Final

**Tag**: `v3-research-final`  
**Date**: 2026-09-22  
**Repository**: [https://github.com/bilppppp/Knowledge-Routing-RAG](https://github.com/bilppppp/Knowledge-Routing-RAG)  
**Status**: Experimental Research Finalized & Frozen  

---

## What This Release Contains

This release provides the frozen code, dataset benchmarks, execution traces, statistical analyses, and documentation for the `Knowledge-Routing-RAG` project:

1. **Frozen Candidate Architecture (V3-Frozen / `E2-Lite`)**:
   - Dense vector entrance (`Qdrant`, Cosine Top-5).
   - Fast-Path gating for single-document queries ($T \ge 0.85$).
   - Decontaminated Control Plane Shadow Candidate Plane (Top-20 prefixes) with parent lift and route-prefix resolution.
   - Slot-conditioned lexical targeted descent (`BM25` + heading bonuses).
   - Coverage-preserving evidence composition (`E1 Composer`) with zero useful evidence eviction.
   - Fixed context budget: $\le 5$ chunks ($\le 4000$ tokens), passed to unmodified B0 prompt.

2. **Complete Research Evidence Trail**:
   - All evaluation logs, majorities, and reports across 9 chronological phases:
     - Phase V1 (Hypothesis Failure: K1–K4)
     - Phase V2 (Candidate Search: C1–C7)
     - Decontamination Audit (Removal of benchmark-specific heuristics)
     - Holdout-1 Confirmation (Navigation gain confirmed, end-to-end not confirmed)
     - Phase V3 (Composition & Descent Minimality Ablations)
     - Holdout-2 (Primary Independent Confirmation)
     - Phase C (Scale Robustness Characterization)
     - Phase D (Hub Stress Characterization)
     - Phase E (Cost, Latency & Systems Audit)

3. **Pre-Registered Independent Datasets**:
   - `Holdout-2` ($N=250$, unseen questions on $D_{100}$, with 3-run majority and blind review pack).
   - `ScaleSet-1` ($N=80$, evaluated on $D_{20}$, $D_{50}$, $D_{100}$).
   - `HubSet-1` ($N=100$, stratified across 5 graph degree tiers).
   - `Holdout-1` ($N=200$, clean confirmation set).
   - `Dev-216` ($N=216$ runs, clearly disclosed as an Optimization / Development Set).

---

## Confirmed Results

On the primary independent evaluation set (**Holdout-2**, $N=250$):
- **Majority Answer Accuracy**: Vector RAG $67.60\%$ $\to$ V3-Frozen $73.20\%$ ($\Delta = \mathbf{+5.60\text{pp}}$, Exact McNemar $p = 0.0043$).
- **95% Paired Bootstrap Confidence Interval**: `[+2.00pp, +9.20pp]`.
- **Transitions**: 18 Stable Rescues vs. 4 Stable Regressions (Net Stable Rescue = **+14**).
- **Gold Document Recall**: $+9.60\text{pp}$ ($88.93\% \to 98.53\%$).
- **Multi-Hop Reasoning**: $+8.97\text{pp}$ on 2-hop+ queries; 0 regressions on 1-hop simple queries.
- **Engineering Value**: Local P50 latency overhead $+0.58\text{ ms}$; mean local overhead $+3.63\text{ ms}$ ($0.10\%$ of end-to-end request time); 0 additional online LLM calls; 0 additional query embeddings; $+1.96\%$ mean input tokens.

---

## Known Limitations & Non-Confirmed Claims

- **Scale Robustness**: **NOT CONFIRMED**. V3 degraded by $+2.50\text{pp}$ from $D_{20}$ to $D_{100}$ while B0 was flat ($+0.00\text{pp}$), yielding $\text{RA} = -2.50\text{pp}$ (95% CI `[-6.25pp, 0.00pp]`). V3 is not claimed to be more robust to corpus expansion than Vector RAG.
- **Hub Robustness**: **NOT CONFIRMED**. In high-degree nodes ($>20$), candidate pool P95 expanded to 27.1 chunks, and evidence-chain completion dropped from $60\%$ to $42.5\%$.
- **Graph Flooding Suppression**: **NOT TESTABLE**. Legacy unconstrained graph baseline was unavailable.
- **Corpus & Domain**: Findings are specific to 100 Chinese medical/healthcare administrative regulations with statutory cross-references. Results should not be generalized to unstructured text without evaluation.
- **Prototype Status**: This is a research artifact designed to establish retrieval principles, not a production-hardened microservice.

---

## Reproduction

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for environment configuration, hashes, and reproduction scripts across 3 verification levels:
- Level 1: `python scripts/release_smoke_test.py`
- Level 2: `python scripts/run_holdout2_experiment.py --retrieval-only`
- Level 3: `python scripts/run_holdout2_experiment.py` (Full 3-run generation)

---

## Definitive Final Report

The comprehensive scientific narrative, ablation analysis, and methodological disclosures are available in:
- [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md)
