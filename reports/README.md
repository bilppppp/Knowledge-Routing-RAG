# Research Evidence & Reports Index

This directory contains all experimental outputs, statistical tests, blind adjudication packs, integrity audit findings, and causal mechanism analyses produced across the entire research lifecycle of `Knowledge-Routing-RAG`.

> [!NOTE]
> All primary artifacts are preserved in their original form to maintain an unbroken audit trail: from initial network-routing failure, through heuristic search and historical full-stack confirmation, to the Gate I data audit and the final Holdout-3 causal mechanism resolution.

---

## Chronological Research Phase Index

| Stage | Phase Name | Purpose | Verdict | Report Link |
|:---|:---|:---|:---|:---|
| **01** | **V1 — Initial Hypothesis** | Test original network-routing-inspired Knowledge Routing (K1–K4 routers) against standard Vector RAG on Dev benchmark. | **`NO-GO — HYPOTHESIS FAILED`**<br>Autonomous routing lagged Vector RAG; unconstrained graph traversal caused heavy candidate drift. | [`preflight_report.md`](preflight_report.md)<br>[`v1_historical_report.md`](v1_historical_report.md)<br>[`v1_correction_audit.md`](v1_historical_report.md) |
| **02** | **V2 — Directed Search** | Explore candidate structures (C1–C7) to separate routing plane from generation context. | **`DIRECTED SEARCH IDENTIFIED MECHANISMS`**<br>Identified shadow candidate plane, parent lift, and local lexical descent as effective heuristics. | [`v2_search_log.md`](v2_search_log.md)<br>[`v2_candidate_tree.md`](v2_candidate_tree.md) |
| **03** | **Decontamination** | Rigorous audit of heuristic rules in C7 to eliminate benchmark-specific prompts and hardcoded statute mappings. | **`DECONTAMINATION COMPLETE`**<br>Removed all hardcoded lookup tables and statute title regexes; established generic `C7-Clean`. | [`decontamination_rule_audit.md`](decontamination_rule_audit.md)<br>[`c7_decontamination_ablation.md`](c7_decontamination_ablation.md) |
| **04** | **Holdout-1** | Independent confirmation of C7-Clean on unseen $N=200$ holdout set under sealed conditions. | **`NAVIGATION GAIN CONFIRMED, ANSWER GAIN NOT CONFIRMED`**<br>Doc recall $+8.08\text{pp}$, but answer accuracy $-0.50\text{pp}$ due to useful evidence eviction. | [`clean_architecture_confirmation.md`](clean_architecture_confirmation.md)<br>[`confirmation_dataset_audit.md`](confirmation_dataset_audit.md) |
| **05** | **V3 Development** | Resolve useful evidence eviction (E1 Composer) and chunk missing (E2-Lite Lexical Descent). | **`MINIMAL SUFFICIENT V3 FROZEN`**<br>E1 resolved evictions in dev ablation; E2 proved semantic channel unneeded; frozen as `E2-Lite`. | [`v3_e1_evidence_composition.md`](v3_e1_evidence_composition.md)<br>[`v3_e2_targeted_descent_analysis.md`](v3_e2_targeted_descent_analysis.md)<br>[`v3_b1_e2_minimality_audit.md`](v3_b1_e2_minimality_audit.md) |
| **06** | **Holdout-2** | Historical pre-registered 3-run evaluation of full V3 stack vs Dense Top-5 baseline ($N=250$, old corpus snapshot). | **`HISTORICAL FULL-STACK CONFIRMATION`**<br>Full V3 (73.20%) > Dense Top-5 (67.60%, $+5.60\text{pp}$, $p=0.0043$). Confirms full stack, but does NOT confirm graph edges were the cause. | [`../HOLDOUT2_PREREGISTRATION.md`](../HOLDOUT2_PREREGISTRATION.md)<br>[`v3_holdout2_confirmation.md`](v3_holdout2_confirmation.md)<br>[`holdout2_dataset_audit.md`](holdout2_dataset_audit.md) |
| **07** | **Scale Robustness** | Test performance degradation across expanding corpora ($D_{20} \subset D_{50} \subset D_{100}$, $N=80$). | **`SCALE ROBUSTNESS NOT CONFIRMED`** (Verdict S-C)<br>V3 degraded $+2.50\text{pp}$ vs B0 $+0.00\text{pp}$ ($\text{RA} = -2.50\text{pp}$); claim prohibited. | [`../SCALE1_PREREGISTRATION.md`](../SCALE1_PREREGISTRATION.md)<br>[`v3_scale_robustness_confirmation.md`](v3_scale_robustness_confirmation.md)<br>[`scale1_dataset_audit.md`](scale1_dataset_audit.md) |
| **08** | **Hub Stress** | Test graph degree sensitivity across 5 graph degree tiers (<5 to >50, $N=100$). | **`HUB STABILITY NOT CONFIRMED / FLOODING NOT TESTABLE`** (Verdict H2-C)<br>High-degree candidate P95 rose to 27.1; chain completion dropped $17.5\text{pp}$. | [`../HUB1_PREREGISTRATION.md`](../HUB1_PREREGISTRATION.md)<br>[`v3_hub_stress_confirmation.md`](v3_hub_stress_confirmation.md)<br>[`hub1_dataset_audit.md`](hub1_dataset_audit.md) |
| **09** | **Engineering Cost** | Benchmark latency, token overhead, and operational complexity on Holdout-2. | **`ENGINEERING VALUE CONFIRMED`** (Verdict E-A)<br>Local P50 overhead $+0.58\text{ ms}$, 0 extra online LLM calls; graph removal in S3 further simplifies operational profile. | [`v3_cost_complexity_audit.md`](v3_cost_complexity_audit.md)<br>[`phase_e_cost_audit.json`](phase_e_cost_audit.json) |
| **10** | **Gate I Integrity Audit** | Full-corpus metadata audit (100 docs), chunk provenance audit (2862 chunks), benchmark decontamination, and runner audit. | **`I-B — CORRECTABLE INTEGRITY ISSUES`**<br>Discovered doc034/doc086 title misplacements (corrected in freeze); exposed Holdout-2 runner same-evidence shortcut and 94.87% explicitness bias. | [`integrity/GATE_I_DATA_EVAL_AUDIT_REPORT.md`](integrity/GATE_I_DATA_EVAL_AUDIT_REPORT.md)<br>[`../CORPUS_CORRECTED_FREEZE.md`](../CORPUS_CORRECTED_FREEZE.md) |
| **11** | **Mechanism Holdout-3** | Definitive causal ablation separating graph edges from metadata, lexical, and composition baselines on Corrected Corpus ($N=240$). | **`M-B — STRUCTURED RETRIEVAL CONFIRMED, GRAPH NOT NECESSARY`**<br>$S_2$ (+6.25pp) & $S_3$ (+6.67pp) explain gains over $S_0$; $S_4$ TrueGraph underperformed $S_3$ by $-5.42\text{pp}$ ($p=0.0059$); True Graph Rescues = 0. | [`MECHANISM_HOLDOUT3_REPORT.md`](MECHANISM_HOLDOUT3_REPORT.md)<br>[`../MECHANISM_HOLDOUT3_PREREGISTRATION.md`](../MECHANISM_HOLDOUT3_PREREGISTRATION.md) |
| **12** | **Final Report** | Authoritative standalone synthesis covering the full research arc, confirmed mechanisms, rejected claims, and finalized architecture. | **`FINAL RESEARCH REPORT COMPLETE`**<br>Recommends graph-free V3-NoGraph structured retrieval; fully documents historical failures and scientific boundaries. | [`FINAL_REPORT.md`](FINAL_REPORT.md) |

---

## Primary Data Artifacts

- **Mechanism Holdout-3 Evidence**:
  - Deterministic Retrieval Traces (6 Systems, $N=240$): [`holdout3_retrieval.json`](holdout3_retrieval.json)
  - Independent End-to-End Generation & Judging ($N=240$): [`holdout3_run1.json`](holdout3_run1.json)
  - Formal Mechanism Analysis Report: [`MECHANISM_HOLDOUT3_REPORT.md`](MECHANISM_HOLDOUT3_REPORT.md)
- **Holdout-2 Historical Evidence**:
  - Raw 3 Runs: [`holdout2_run1.json`](holdout2_run1.json), [`holdout2_run2.json`](holdout2_run2.json), [`holdout2_run3.json`](holdout2_run3.json)
  - Majority Aggregation & Retrieval: [`holdout2_majority.json`](holdout2_majority.json), [`holdout2_retrieval.json`](holdout2_retrieval.json)
  - Blind Review Pack & Decisions: [`holdout2_blind_pack.json`](holdout2_blind_pack.json), [`holdout2_blind_decisions.json`](holdout2_blind_decisions.json)
- **Scale & Hub Stress Evidence**:
  - Scale Runs & Majority: [`scale1_majority.json`](scale1_majority.json), [`scale1_degradation.json`](scale1_degradation.json)
  - Hub Runs & Majority: [`hub1_majority.json`](hub1_majority.json), [`hub1_hub_stress_analysis.json`](hub1_hub_stress_analysis.json)
