#!/usr/bin/env python3
"""
scripts/build_blind_adjudication.py
Implements Protocol-Mandated Blind Adjudication Workflow per 实验方案.md §25.

Generates:
1. reports/blind_review_pack.json: Anonymized pair comparisons (Output A / Output B)
2. reports/blind_review_mapping.json: Secret system mappings (B0/K2/K4)
3. reports/blind_review_decisions.json: Adjudication template (or preserves existing human reviews)
4. reports/final_adjudicated_results.json: Combined results tracking final_is_correct and judgment_source
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
RUNS_DIR = PROJECT_ROOT / "runs" / "test"
REPORTS_DIR = PROJECT_ROOT / "reports"

CONFIDENCE_THRESHOLD = 0.8  # Preset threshold per protocol


def load_traces() -> Dict[str, Any]:
    traces = []
    for p in RUNS_DIR.glob("*/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            traces.append(json.load(f))
    return traces


def load_gold() -> Dict[str, Any]:
    gold_map = {}
    with open(BENCHMARK_DIR / "gold.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gold_map[item["qid"]] = item
    return gold_map


def build_blind_workflow():
    traces = load_traces()
    gold_map = load_gold()

    # Index traces by (qid, corpus, system_id)
    by_key: Dict[tuple, Dict[str, Any]] = {}
    for t in traces:
        key = (t["qid"], t["corpus"])
        if key not in by_key:
            by_key[key] = {}
        by_key[key][t["system_id"]] = t

    # Identify candidate pairs for review:
    # 1. B0 vs K2 disagreements
    # 2. B0 vs K4 disagreements
    # 3. Any case with judge_confidence < CONFIDENCE_THRESHOLD
    review_pairs = []

    for key, smap in sorted(by_key.items()):
        qid, corpus = key
        b0_trace = smap.get("B0")
        k2_trace = smap.get("K2")
        k4_trace = smap.get("K4")

        # Check B0 vs K2
        if b0_trace and k2_trace:
            disagree = (b0_trace["is_correct"] != k2_trace["is_correct"])
            low_conf = (b0_trace.get("judge_confidence", 1.0) < CONFIDENCE_THRESHOLD or
                        k2_trace.get("judge_confidence", 1.0) < CONFIDENCE_THRESHOLD)
            if disagree or low_conf:
                review_pairs.append({
                    "qid": qid,
                    "corpus": corpus,
                    "comparison_type": "B0_vs_K2",
                    "sys1": "B0",
                    "sys2": "K2",
                    "trace1": b0_trace,
                    "trace2": k2_trace,
                    "reason": "disagreement" if disagree else "low_confidence"
                })

        # Check B0 vs K4
        if b0_trace and k4_trace:
            disagree = (b0_trace["is_correct"] != k4_trace["is_correct"])
            low_conf = (b0_trace.get("judge_confidence", 1.0) < CONFIDENCE_THRESHOLD or
                        k4_trace.get("judge_confidence", 1.0) < CONFIDENCE_THRESHOLD)
            if disagree or low_conf:
                review_pairs.append({
                    "qid": qid,
                    "corpus": corpus,
                    "comparison_type": "B0_vs_K4",
                    "sys1": "B0",
                    "sys2": "K4",
                    "trace1": b0_trace,
                    "trace2": k4_trace,
                    "reason": "disagreement" if disagree else "low_confidence"
                })

    # Fixed seed for deterministic randomization
    rng = random.Random(42)

    blind_pack = []
    mapping = {}
    review_decisions_template = []

    # Check if decisions file already exists with completed adjudications
    decisions_path = REPORTS_DIR / "blind_review_decisions.json"
    existing_decisions = {}
    if decisions_path.exists():
        try:
            with open(decisions_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for item in saved:
                    if item.get("adjudicated", False):
                        existing_decisions[item["review_id"]] = item
        except Exception:
            pass

    for idx, item in enumerate(review_pairs, start=1):
        review_id = f"REV-{idx:03d}"
        qid = item["qid"]
        corpus = item["corpus"]
        gold = gold_map[qid]

        trace1 = item["trace1"]
        trace2 = item["trace2"]

        # Randomize order (Output A vs Output B)
        if rng.random() < 0.5:
            sys_a, trace_a = item["sys1"], trace1
            sys_b, trace_b = item["sys2"], trace2
        else:
            sys_a, trace_a = item["sys2"], trace2
            sys_b, trace_b = item["sys1"], trace1

        # Build Blind Pack item (Strictly NO system/algorithm names)
        blind_pack.append({
            "review_id": review_id,
            "qid": qid,
            "corpus": corpus,
            "question": gold["question"],
            "gold_answer": gold["gold_answer"],
            "gold_spans": gold.get("gold_spans", []),
            "output_a": trace_a["generated_answer"],
            "output_b": trace_b["generated_answer"],
            "mapping_hidden": True
        })

        # Build secret mapping
        mapping[review_id] = {
            "review_id": review_id,
            "qid": qid,
            "corpus": corpus,
            "comparison_type": item["comparison_type"],
            "trigger_reason": item["reason"],
            "output_a_system": sys_a,
            "output_b_system": sys_b,
            "auto_judge_initial": {
                "A": trace_a["is_correct"],
                "B": trace_b["is_correct"]
            }
        }

        # Build decisions template or preserve existing adjudication
        if review_id in existing_decisions:
            review_decisions_template.append(existing_decisions[review_id])
        else:
            review_decisions_template.append({
                "review_id": review_id,
                "qid": qid,
                "corpus": corpus,
                "adjudicated": False,
                "winner_or_correctness": {
                    "A": None,
                    "B": None
                },
                "notes": ""
            })

    # Save pack, mapping, and decisions
    pack_path = REPORTS_DIR / "blind_review_pack.json"
    mapping_path = REPORTS_DIR / "blind_review_mapping.json"

    with open(pack_path, "w", encoding="utf-8") as f:
        json.dump(blind_pack, f, ensure_ascii=False, indent=2)

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    with open(decisions_path, "w", encoding="utf-8") as f:
        json.dump(review_decisions_template, f, ensure_ascii=False, indent=2)

    # Compile final_adjudicated_results.json
    # Track human adjudicated overwrites vs auto judge
    human_overwrites: Dict[tuple, Dict[str, bool]] = {}
    adjudicated_count = 0

    for dec in review_decisions_template:
        if dec.get("adjudicated", False):
            rev_id = dec["review_id"]
            m = mapping[rev_id]
            key = (m["qid"], m["corpus"])
            if key not in human_overwrites:
                human_overwrites[key] = {}
            
            w = dec.get("winner_or_correctness", {})
            if w.get("A") is not None:
                human_overwrites[key][m["output_a_system"]] = bool(w["A"])
            if w.get("B") is not None:
                human_overwrites[key][m["output_b_system"]] = bool(w["B"])
            adjudicated_count += 1

    disputed_keys = {(item["qid"], item["corpus"]) for item in review_pairs}

    results_by_system: Dict[str, List[Dict[str, Any]]] = {}
    all_systems = ["B0", "B1", "K1", "K2", "K3", "K4"]
    for s in all_systems:
        results_by_system[s] = []

    for key, smap in sorted(by_key.items()):
        qid, corpus = key
        for s in all_systems:
            t = smap.get(s)
            if not t:
                continue

            auto_corr = bool(t["is_correct"])
            auto_conf = float(t.get("judge_confidence", 1.0))

            # Determine human adjudicated result if present
            human_corr = None
            if key in human_overwrites and s in human_overwrites[key]:
                human_corr = human_overwrites[key][s]

            if human_corr is not None:
                final_corr = human_corr
                src = "human_adjudicated"
            elif key in disputed_keys:
                final_corr = auto_corr
                src = "provisional_auto_pending_human_adjudication"
            else:
                final_corr = auto_corr
                src = "auto_high_confidence"

            results_by_system[s].append({
                "qid": qid,
                "corpus": corpus,
                "auto_judge_is_correct": auto_corr,
                "auto_judge_confidence": auto_conf,
                "human_adjudicated_is_correct": human_corr,
                "final_is_correct": final_corr,
                "judgment_source": src
            })

    adjudication_status = "COMPLETED" if adjudicated_count == len(review_pairs) and len(review_pairs) > 0 else "PENDING_HUMAN_ADJUDICATION"

    final_results = {
        "status": adjudication_status,
        "protocol_reference": "实验方案.md §25",
        "adjudication_summary": {
            "total_test_instances_per_system": len(by_key),
            "total_disputed_comparison_pairs": len(review_pairs),
            "unique_disputed_questions": len(disputed_keys),
            "adjudicated_pairs_count": adjudicated_count,
            "pending_pairs_count": len(review_pairs) - adjudicated_count,
            "provisional_unadjudicated_count": len(disputed_keys) if adjudicated_count == 0 else (len(disputed_keys) - len(human_overwrites))
        },
        "systems": results_by_system
    }

    final_results_path = REPORTS_DIR / "final_adjudicated_results.json"
    with open(final_results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"Generated Blind Review Pack: {pack_path} ({len(blind_pack)} items)")
    print(f"Generated Secret Mapping: {mapping_path}")
    print(f"Generated Review Decisions Template: {decisions_path}")
    print(f"Generated Final Adjudicated Results: {final_results_path} (Status: {adjudication_status})")


if __name__ == "__main__":
    build_blind_workflow()
