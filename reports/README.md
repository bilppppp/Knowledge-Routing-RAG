# Research Evidence & Reports Index

This directory contains all raw experimental outputs, statistical tests, blind adjudication packs, and analysis reports produced across the 9 distinct phases of the `Knowledge-Routing-RAG` project.

> [!NOTE]
> All primary artifacts are preserved in their original form to maintain an unbroken audit trail from initial hypothesis failure to final confirmation and release.

---

## Chronological Research Phase Index

| Stage | Phase Name | Primary Purpose | Definitive Verdict | Key Report Links |
|:---|:---|:---|:---|:---|
| **01** | **Exploration** (V1 & V2) | Initial hypothesis testing (K1–K4) & directed candidate search (C1–C7) on Dev benchmark. | **`HYPOTHESIS FAILED → DIRECTED SEARCH FOUND MECHANISMS`**<br>Early unconstrained routing failed; C4–C7 identified shadow candidate plane and parent lift. | [`preflight_report.md`](preflight_report.md)<br>[`v1_correction_audit.md`](v1_correction_audit.md)<br>[`v2_search_log.md`](v2_search_log.md)<br>[`v1_historical_report.md`](v1_historical_report.md) *(historical V1/V2)* |
| **02** | **Decontamination** | Rigorous audit of heuristic rules in C7 to eliminate benchmark-specific prompts and lookup tables. | **`DECONTAMINATION COMPLETE`**<br>Removed all hardcoded statute tables and regex matching; established generic `C7-Clean`. | [`decontamination_rule_audit.md`](decontamination_rule_audit.md)<br>[`c7_decontamination_ablation.md`](c7_decontamination_ablation.md) |
| **03** | **Holdout-1 Confirmation** | Independent test of C7-Clean on unseen $N=200$ holdout set under sealed conditions. | **`NAVIGATION GAIN CONFIRMED, END-TO-END NOT CONFIRMED`**<br>Doc recall $+8.08\text{pp}$, but answer accuracy $-0.50\text{pp}$ due to useful evidence eviction. | [`clean_architecture_confirmation.md`](clean_architecture_confirmation.md)<br>[`confirmation_dataset_audit.md`](confirmation_dataset_audit.md) |
| **04** | **V3 Development & Minimality** | Resolving evidence eviction (E1 Composer) and target chunk missing (E2-Lite Lexical Descent). | **`MINIMAL SUFFICIENT V3 FROZEN`**<br>E1 eliminated evictions; E2 ablation proved semantic channel & RRF unneeded; frozen as `E2-Lite`. | [`v3_a0_descent_oracle_audit.md`](v3_a0_descent_oracle_audit.md)<br>[`v3_e1_evidence_composition.md`](v3_e1_evidence_composition.md)<br>[`v3_e2_targeted_descent_analysis.md`](v3_e2_targeted_descent_analysis.md)<br>[`v3_b1_e2_minimality_audit.md`](v3_b1_e2_minimality_audit.md)<br>[`../V3_FREEZE.md`](../V3_FREEZE.md) |
| **05** | **Holdout-2 Confirmation** | Pre-registered independent 3-run evaluation of Frozen V3 vs B0 Vector RAG ($N=250$). | **`END-TO-END ACCURACY GAIN CONFIRMED`** (Verdict A)<br>Accuracy $+5.60\text{pp}$ ($67.6\% \to 73.2\%$, $p=0.0043$), Net Rescue $+14$, Multi-hop $+8.97\text{pp}$. | [`../HOLDOUT2_PREREGISTRATION.md`](../HOLDOUT2_PREREGISTRATION.md)<br>[`v3_holdout2_confirmation.md`](v3_holdout2_confirmation.md)<br>[`holdout2_dataset_audit.md`](holdout2_dataset_audit.md) |
| **06** | **Phase C: Scale Robustness** | Testing performance degradation across expanding corpora ($D_{20} \subset D_{50} \subset D_{100}$, $N=80$). | **`SCALE ROBUSTNESS NOT CONFIRMED`** (Verdict S-C)<br>V3 degraded $+2.50\text{pp}$ vs B0 $+0.00\text{pp}$ ($\text{RA} = -2.50\text{pp}$); claim prohibited. | [`../SCALE1_PREREGISTRATION.md`](../SCALE1_PREREGISTRATION.md)<br>[`v3_scale_robustness_confirmation.md`](v3_scale_robustness_confirmation.md)<br>[`scale1_dataset_audit.md`](scale1_dataset_audit.md) |
| **07** | **Phase D: Hub Stress** | Testing degree sensitivity across 5 graph degree tiers (<5 to >50, $N=100$). | **`V3 HUB STABILITY NOT CONFIRMED`** (Verdict H2-C)<br>Candidate P95 rose to 27.1; chain completion dropped $17.5\text{pp}$; legacy baseline unavailable. | [`../HUB1_PREREGISTRATION.md`](../HUB1_PREREGISTRATION.md)<br>[`v3_hub_stress_confirmation.md`](v3_hub_stress_confirmation.md)<br>[`hub1_dataset_audit.md`](hub1_dataset_audit.md) |
| **08** | **Phase E: Cost & Complexity** | Benchmarking latency, token overhead, and engineering complexity on Holdout-2. | **`ENGINEERING VALUE CONFIRMED`** (Verdict E-A)<br>Local P50 overhead $+0.58\text{ ms}$, 0 extra online LLM calls, 0 query embeddings, token overhead $+1.96\%$. | [`v3_cost_complexity_audit.md`](v3_cost_complexity_audit.md) |
| **09** | **Final Synthesized Report** | Comprehensive scientific report integrating all research findings, mechanisms, and limits. | **`FINAL RESEARCH REPORT COMPLETE`**<br>Authoritative standalone narrative synthesizing the full research arc. | [`FINAL_REPORT.md`](FINAL_REPORT.md)<br>[`final_repository_audit.md`](final_repository_audit.md) |

---

## Key Data Artifacts & Blind Review Records

- **Holdout-2 Evidence**:
  - Raw 3 Runs: [`holdout2_run1.json`](holdout2_run1.json), [`holdout2_run2.json`](holdout2_run2.json), [`holdout2_run3.json`](holdout2_run3.json)
  - Majority Aggregation: [`holdout2_majority.json`](holdout2_majority.json)
  - Retrieval Metrics: [`holdout2_retrieval.json`](holdout2_retrieval.json)
  - Blind Review Pack & Decisions: [`holdout2_blind_pack.json`](holdout2_blind_pack.json), [`holdout2_blind_decisions.json`](holdout2_blind_decisions.json)
  - Causal Attribution: [`holdout2_effect_attribution.json`](holdout2_effect_attribution.json)
- **Phase C Scale Evidence**:
  - Runs & Majority: [`scale1_run1.json`](scale1_run1.json), [`scale1_run2.json`](scale1_run2.json), [`scale1_run3.json`](scale1_run3.json), [`scale1_majority.json`](scale1_majority.json)
  - Degradation Analysis: [`scale1_degradation.json`](scale1_degradation.json)
- **Phase D Hub Evidence**:
  - Runs & Majority: [`hub1_run1.json`](hub1_run1.json), [`hub1_run2.json`](hub1_run2.json), [`hub1_run3.json`](hub1_run3.json), [`hub1_majority.json`](hub1_majority.json)
  - Hub Stress Analysis: [`hub1_hub_stress_analysis.json`](hub1_hub_stress_analysis.json)
- **Phase E Cost Evidence**:
  - Latency & Token Audit: [`phase_e_cost_audit.json`](phase_e_cost_audit.json)
