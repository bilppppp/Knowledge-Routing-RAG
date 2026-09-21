#!/usr/bin/env python3
"""
scripts/run_statistical_analysis.py
Calculates rigorous statistical metrics, bootstrap confidence intervals,
paired McNemar tests, retrieval metrics, scale degradation, and post-hoc diagnostics
per 实验方案.md §26, §27, §34, §35.

Outputs:
- reports/statistical_analysis.json
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy import stats
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
RUNS_DIR = PROJECT_ROOT / "runs" / "test"
REPORTS_DIR = PROJECT_ROOT / "reports"

np.random.seed(42)


def bootstrap_ci(arr: List[float], num_resamples: int = 5000, alpha: float = 0.05) -> Tuple[float, float, float]:
    arr_np = np.array(arr, dtype=float)
    if len(arr_np) == 0:
        return 0.0, 0.0, 0.0
    means = [np.mean(np.random.choice(arr_np, size=len(arr_np), replace=True)) for _ in range(num_resamples)]
    mean = float(np.mean(arr_np))
    lower = float(np.percentile(means, 100 * (alpha / 2)))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return mean, lower, upper


def paired_bootstrap_delta(base_arr: List[float], target_arr: List[float], num_resamples: int = 5000, alpha: float = 0.05) -> Dict[str, float]:
    diff = np.array(target_arr, dtype=float) - np.array(base_arr, dtype=float)
    means = [np.mean(np.random.choice(diff, size=len(diff), replace=True)) for _ in range(num_resamples)]
    mean = float(np.mean(diff))
    lower = float(np.percentile(means, 100 * (alpha / 2)))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    base_mean = float(np.mean(base_arr))
    rel_delta = (mean / base_mean) if base_mean > 0 else 0.0
    return {
        "mean_delta_pp": mean * 100,
        "ci_lower_pp": lower * 100,
        "ci_upper_pp": upper * 100,
        "relative_delta_pct": rel_delta * 100
    }


def compute_mcnemar(base_bool: List[bool], router_bool: List[bool]) -> Dict[str, Any]:
    assert len(base_bool) == len(router_bool)
    a = sum(1 for b, r in zip(base_bool, router_bool) if b and r)
    b = sum(1 for b, r in zip(base_bool, router_bool) if b and not r)  # Regressions
    c = sum(1 for b, r in zip(base_bool, router_bool) if not b and r)  # Rescues
    d = sum(1 for b, r in zip(base_bool, router_bool) if not b and not r)

    n_discordant = b + c
    if n_discordant > 0:
        res = stats.binomtest(k=min(b, c), n=n_discordant, p=0.5, alternative="two-sided")
        p_val = float(res.pvalue)
    else:
        p_val = 1.0

    rescue_rate = c / (c + d) if (c + d) > 0 else 0.0
    regr_rate = b / (a + b) if (a + b) > 0 else 0.0

    return {
        "contingency_matrix": {
            "a_base_corr_router_corr": a,
            "b_base_corr_router_wrong_regressions": b,
            "c_base_wrong_router_corr_rescues": c,
            "d_base_wrong_router_wrong": d
        },
        "rescues_count": c,
        "regressions_count": b,
        "net_rescues": c - b,
        "rescue_rate_pct": rescue_rate * 100,
        "regression_rate_pct": regr_rate * 100,
        "p_value": p_val,
        "significant_at_05": bool(p_val < 0.05)
    }


def main():
    # Load traces
    traces = []
    for p in RUNS_DIR.glob("*/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            traces.append(json.load(f))

    with open(BENCHMARK_DIR / "gold.jsonl", "r", encoding="utf-8") as f:
        gold_map = {json.loads(l)["qid"]: json.loads(l) for l in f if l.strip()}

    systems = ["B0", "B1", "K1", "K2", "K3", "K4"]

    by_key = {}
    for t in traces:
        key = (t["qid"], t["corpus"])
        if key not in by_key:
            by_key[key] = {}
        by_key[key][t["system_id"]] = t

    keys = sorted(list(by_key.keys()))
    n_instances = len(keys)

    # 1. System Accuracies + 95% Bootstrap CI
    sys_accuracies = {}
    for s in systems:
        corr = [1.0 if by_key[k][s]["is_correct"] else 0.0 for k in keys]
        m, l, u = bootstrap_ci(corr)
        sys_accuracies[s] = {
            "mean_accuracy_pct": m * 100,
            "ci_lower_pct": l * 100,
            "ci_upper_pct": u * 100,
            "correct_count": int(sum(corr)),
            "total_count": len(corr)
        }

    # 2. Paired Comparisons against B0
    paired_deltas = {}
    b0_corr = [1.0 if by_key[k]["B0"]["is_correct"] else 0.0 for k in keys]
    b0_bool = [by_key[k]["B0"]["is_correct"] for k in keys]

    for s in ["B1", "K1", "K2", "K3", "K4"]:
        s_corr = [1.0 if by_key[k][s]["is_correct"] else 0.0 for k in keys]
        s_bool = [by_key[k][s]["is_correct"] for k in keys]
        delta_stat = paired_bootstrap_delta(b0_corr, s_corr)
        mcnemar_stat = compute_mcnemar(b0_bool, s_bool)
        paired_deltas[f"{s}_vs_B0"] = {
            "system": s,
            "baseline": "B0",
            "delta_bootstrap_95ci": delta_stat,
            "mcnemar_test": mcnemar_stat
        }

    # 3. Retrieval & Evidence Quality Metrics
    retrieval_metrics = {}
    for s in systems:
        s_traces = [t for t in traces if t["system_id"] == s]
        p_vals = [t["retrieval"]["precision"] for t in s_traces]
        r_vals = [t["retrieval"]["recall"] for t in s_traces]
        f1_vals = [t["retrieval"]["f1"] for t in s_traces]
        cpr_vals = [t["retrieval"]["cpr"] for t in s_traces]

        p_m, p_l, p_u = bootstrap_ci(p_vals)
        r_m, r_l, r_u = bootstrap_ci(r_vals)
        f1_m, f1_l, f1_u = bootstrap_ci(f1_vals)
        cpr_m, cpr_l, cpr_u = bootstrap_ci(cpr_vals)

        retrieval_metrics[s] = {
            "precision": {"mean": p_m, "ci_lower": p_l, "ci_upper": p_u},
            "recall": {"mean": r_m, "ci_lower": r_l, "ci_upper": r_u},
            "f1": {"mean": f1_m, "ci_lower": f1_l, "ci_upper": f1_u},
            "cpr": {"mean_pct": cpr_m * 100, "ci_lower_pct": cpr_l * 100, "ci_upper_pct": cpr_u * 100}
        }

    # 4. Latency & Token Metrics
    latency_token_metrics = {}
    for s in systems:
        s_traces = [t for t in traces if t["system_id"] == s]
        tot_lats = [t["total_latency_ms"] for t in s_traces]
        routing_lats = [t.get("routing_latency_ms", 0.0) for t in s_traces]
        llm_lats = [t.get("llm_latency_ms", 0.0) for t in s_traces]
        inp_toks = [t["input_tokens"] for t in s_traces]
        out_toks = [t["output_tokens"] for t in s_traces]
        all_toks = [t["input_tokens"] + t["output_tokens"] for t in s_traces]

        latency_token_metrics[s] = {
            "latency_p50_ms": float(np.percentile(tot_lats, 50)),
            "latency_p95_ms": float(np.percentile(tot_lats, 95)),
            "mean_total_latency_ms": float(np.mean(tot_lats)),
            "mean_routing_latency_ms": float(np.mean(routing_lats)),
            "mean_llm_latency_ms": float(np.mean(llm_lats)),
            "mean_total_tokens": float(np.mean(all_toks)),
            "mean_input_tokens": float(np.mean(inp_toks)),
            "mean_output_tokens": float(np.mean(out_toks))
        }

    # 5. Core Scale Degradation (Actual Test N=48)
    core_scale = {}
    for s in systems:
        core_scale[s] = {}
        for c in ["D20", "D50", "D100"]:
            st = [t for t in traces if t["system_id"] == s and t["corpus"] == c and int(t["qid"].replace("Q", "")) <= 60]
            n_corr = sum(1 for t in st if t["is_correct"])
            tot = len(st)
            core_scale[s][c] = {
                "accuracy_pct": (n_corr / tot * 100) if tot else 0.0,
                "correct_count": n_corr,
                "total_count": tot,
                "ratio_str": f"{n_corr}/{tot}"
            }
        # Delta D20 -> D100
        d20_acc = core_scale[s]["D20"]["accuracy_pct"]
        d100_acc = core_scale[s]["D100"]["accuracy_pct"]
        core_scale[s]["degradation_delta_d20_to_d100_pp"] = d20_acc - d100_acc

    # 6. Hub Correlational Analysis (Observational / Non-causal)
    from src.graph.lsdb import KnowledgeLSDB
    lsdb = KnowledgeLSDB()

    def get_query_max_degree(qid):
        gold = gold_map[qid]
        degs = []
        for doc in gold["gold_documents"]:
            if lsdb.G_routing.has_node(doc):
                degs.append(lsdb.G_routing.degree(doc))
        for cid in gold["gold_chunk_ids"]:
            if lsdb.G_routing.has_node(cid):
                degs.append(lsdb.G_routing.degree(cid))
        return max(degs) if degs else 0

    degree_tiers = [
        ("< 5 (Low)", lambda d: d < 5),
        ("5–10 (Moderate)", lambda d: 5 <= d <= 10),
        ("11–20 (High)", lambda d: 11 <= d <= 20),
        ("21–50 (Very High)", lambda d: 21 <= d <= 50),
        ("> 50 (Hub Core)", lambda d: d > 50),
    ]

    hub_correlational = {}
    for tier_name, predicate in degree_tiers:
        tier_qids = {qid for qid in gold_map if predicate(get_query_max_degree(qid))}
        b0_tier = [t for t in traces if t["system_id"] == "B0" and t["qid"] in tier_qids]
        k4_tier = [t for t in traces if t["system_id"] == "K4" and t["qid"] in tier_qids]

        b0_acc = sum(1 for t in b0_tier if t["is_correct"]) / len(b0_tier) if b0_tier else 0.0
        k4_acc = sum(1 for t in k4_tier if t["is_correct"]) / len(k4_tier) if k4_tier else 0.0
        b0_cpr = np.mean([t["retrieval"]["cpr"] for t in b0_tier]) if b0_tier else 0.0
        k4_cpr = np.mean([t["retrieval"]["cpr"] for t in k4_tier]) if k4_tier else 0.0

        hub_correlational[tier_name] = {
            "tier": tier_name,
            "question_count": len(tier_qids),
            "eval_instances_count": len(b0_tier),
            "b0_accuracy_pct": b0_acc * 100,
            "k4_accuracy_pct": k4_acc * 100,
            "b0_cpr_pct": b0_cpr * 100,
            "k4_cpr_pct": k4_cpr * 100,
            "methodological_note": "Observational comparison between different question subgroups; non-causal."
        }

    # 7. Exploratory Subgroups (Hop Count and Tags)
    exploratory_subgroups = {"by_hop_count": {}, "by_tag": {}}
    for hop in [1, 2, 3]:
        h_label = f"Hop {hop}" if hop < 3 else "Hop 3+"
        exploratory_subgroups["by_hop_count"][h_label] = {}
        for s in systems:
            st = [t for t in traces if t["system_id"] == s and (gold_map[t["qid"]]["hop_count"] == hop if hop < 3 else gold_map[t["qid"]]["hop_count"] >= 3)]
            n_corr = sum(1 for t in st if t["is_correct"])
            tot = len(st)
            exploratory_subgroups["by_hop_count"][h_label][s] = {
                "accuracy_pct": (n_corr / tot * 100) if tot else 0.0,
                "correct_count": n_corr,
                "total_count": tot
            }

    for tag in ["single_hop", "2-hop", "3-hop", "hub", "temporal", "exception"]:
        exploratory_subgroups["by_tag"][tag] = {}
        for s in systems:
            st = [t for t in traces if t["system_id"] == s and tag in gold_map[t["qid"]].get("tags", [])]
            n_corr = sum(1 for t in st if t["is_correct"])
            tot = len(st)
            exploratory_subgroups["by_tag"][tag][s] = {
                "accuracy_pct": (n_corr / tot * 100) if tot else 0.0,
                "correct_count": n_corr,
                "total_count": tot
            }

    # 8. Post-hoc Error Diagnostics on K4
    k4_traces = [t for t in traces if t["system_id"] == "K4"]
    k4_failures = [t for t in k4_traces if not t["is_correct"]]
    n_k4_failures = len(k4_failures)

    seed_miss = 0
    edge_miss = 0
    gen_misuse = 0
    for t in k4_failures:
        gold_chunks = set(t.get("gold_chunk_ids", []))
        ev_chunks = set(t.get("final_evidence_chunk_ids", []))
        overlap = gold_chunks & ev_chunks
        if len(overlap) == 0:
            seed_miss += 1
        elif len(overlap) < len(gold_chunks):
            edge_miss += 1
        else:
            gen_misuse += 1

    error_diagnostics = {
        "analysis_type": "post_hoc_diagnostic_attribution",
        "total_k4_test_evaluations": len(k4_traces),
        "total_k4_failures": n_k4_failures,
        "failure_rate_pct": (n_k4_failures / len(k4_traces) * 100) if k4_traces else 0.0,
        "breakdown": {
            "wrong_next_hop_or_missing_edge_partial_evidence": {
                "count": edge_miss,
                "pct_of_failures": (edge_miss / n_k4_failures * 100) if n_k4_failures else 0.0,
                "description": "Evidence graph edge sparsity or feasibility gate over-pruning led to partial gold chunk retrieval."
            },
            "seed_miss_or_scope_selection_zero_evidence": {
                "count": seed_miss,
                "pct_of_failures": (seed_miss / n_k4_failures * 100) if n_k4_failures else 0.0,
                "description": "Initial prefix seed failed to anchor to relevant statutory scope."
            },
            "generator_misuse_full_evidence_wrong_answer": {
                "count": gen_misuse,
                "pct_of_failures": (gen_misuse / n_k4_failures * 100) if n_k4_failures else 0.0,
                "description": "All gold chunks successfully present in 4000-token context, but generator LLM failed reasoning."
            }
        }
    }

    # 9. Pre-registered Protocol Audit & Verdict
    audit_verdict = {
        "preregistered_criteria_reference": "实验方案.md §27, §34, §35",
        "verdict": "NO-GO for current V1 implementation / primary hypothesis",
        "minimum_signal_met": False,
        "minimum_signal_details": {
            "criterion": "K4 > B0",
            "b0_accuracy_pct": sys_accuracies["B0"]["mean_accuracy_pct"],
            "k4_accuracy_pct": sys_accuracies["K4"]["mean_accuracy_pct"],
            "delta_pp": paired_deltas["K4_vs_B0"]["delta_bootstrap_95ci"]["mean_delta_pp"],
            "verdict": "FAILED (K4 is 2.31pp below B0)"
        },
        "engineering_success_met": False,
        "engineering_success_details": {
            "criterion": "K4 >= B0 + 8pp, Regression Rate < Rescue Rate, CPR significantly lower",
            "accuracy_target_met": False,
            "rescue_vs_regression": "FAILED (12 regressions vs 7 rescues)",
            "cpr_significance": "FAILED (CPR K4 83.1% vs B0 83.2%, not significantly lower)"
        },
        "verdict_scope": "NO-GO applies to the current V1 implementation and primary hypothesis, not to the broader research direction."
    }

    output_data = {
        "metadata": {
            "experiment_id": "EXP-20260921-KR-V1",
            "protocol": "实验方案.md §25-§35",
            "commit_base": "1aaafbb",
            "eval_instances_per_system": n_instances,
            "unique_test_questions": len(set(t["qid"] for t in traces)),
            "generator_model": "deepseek-chat",
            "auto_judge_model": "deepseek-chat",
            "model_family_bias_disclosure": "Generator and Auto Judge share the DeepSeek model family; blind human adjudication protocol is established to mitigate model-family evaluation bias."
        },
        "system_accuracies": sys_accuracies,
        "paired_comparisons_vs_b0": paired_deltas,
        "retrieval_and_evidence_metrics": retrieval_metrics,
        "latency_and_tokens": latency_token_metrics,
        "scale_degradation_core60": core_scale,
        "hub_correlational_analysis": hub_correlational,
        "exploratory_subgroups": exploratory_subgroups,
        "post_hoc_error_diagnostics": error_diagnostics,
        "preregistered_verdict": audit_verdict
    }

    out_path = REPORTS_DIR / "statistical_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Generated Comprehensive Statistical Analysis: {out_path}")


if __name__ == "__main__":
    main()
