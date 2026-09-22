#!/usr/bin/env python3
"""
scripts/run_v3_e1_evaluation.py
End-to-End Evaluation Runner for Candidate E1 (Coverage-Preserving Evidence Composition).
Reference: V3 Architecture Specification (Sections IX, XVI, XVII, XVIII, XXVIII-XLIV).

Compares:
  - D0: B0 Baseline
  - D1: C7-Clean-Raw (Position replacement)
  - E1: Coverage-Preserving Composition (E1 Composer)

Across the Dev Benchmark (N = 216 tasks across D20, D50, D100).
Prompt: 100% Frozen B0 Raw Answer Prompt.
Generates full JSON summary and prepares data for reports/v3_e1_evidence_composition.md.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.search import SearchService
from src.services.llm import LLMService
from src.graph.lsdb import KnowledgeLSDB
from src.evaluation.metrics import Evaluator
from src.routing.e1_composer_router import E1ComposerRouterSystem

BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
RUNS_TEST_DIR = PROJECT_ROOT / "runs" / "test"
RUNS_ABLATION_DIR = PROJECT_ROOT / "runs" / "ablation"
RUNS_V3_DIR = PROJECT_ROOT / "runs" / "v3"
REPORTS_DIR = PROJECT_ROOT / "reports"

RUNS_V3_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_b0_traces() -> Dict[Tuple[str, str], Dict[str, Any]]:
    b0_map = {}
    for p in RUNS_TEST_DIR.glob("B0_*/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            b0_map[(d["qid"], d["corpus"])] = d
    return b0_map


def load_d1_traces() -> Dict[Tuple[str, str], Dict[str, Any]]:
    d1_map = {}
    for p in (RUNS_ABLATION_DIR / "A2").glob("*/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            d1_map[(d["qid"], d["corpus"])] = d
    return d1_map


def load_benchmark():
    with open(BENCHMARK_DIR / "split.json", "r", encoding="utf-8") as f:
        split_data = json.load(f)
    test_qids = split_data.get("test_qids", split_data.get("test", []))

    gold_map = {}
    with open(BENCHMARK_DIR / "gold.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gold_map[item["qid"]] = item

    return test_qids, gold_map


def run_single_e1_task(
    task_key: Tuple[str, str],
    gold: Dict[str, Any],
    system_instance: E1ComposerRouterSystem,
    evaluator: Evaluator,
    b0_traces: Dict[Tuple[str, str], Dict[str, Any]],
    out_dir: Path,
    fresh_mode: bool = False
) -> Dict[str, Any]:
    qid, corpus = task_key
    question = gold["question"]

    corpus_dir = out_dir / f"E1_{corpus}"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    trace_file = corpus_dir / f"{qid}.json"

    if trace_file.exists():
        with open(trace_file, "r", encoding="utf-8") as f:
            return json.load(f)

    try:
        trace = system_instance.run(qid=qid, question=question, corpus=corpus)
    except Exception as e:
        print(f"  [ERROR] E1 on {qid} ({corpus}): {e}")
        return None

    b0_item = b0_traces.get(task_key)
    can_pass_through = (
        not fresh_mode
        and b0_item is not None
        and trace.final_evidence_chunk_ids == b0_item["final_evidence_chunk_ids"]
        and trace.generated_answer == b0_item["generated_answer"]
    )

    if can_pass_through:
        ret_metrics = b0_item["retrieval"]
        judge_result = {
            "is_correct": b0_item["is_correct"],
            "confidence": b0_item["judge_confidence"],
            "reasoning": b0_item["judge_reasoning"]
        }
        is_reused_b0 = True
    else:
        ret_metrics = evaluator.evaluate_retrieval(
            retrieved_ids=trace.final_evidence_chunk_ids,
            gold_ids=gold["gold_chunk_ids"]
        )
        judge_result = evaluator.judge_answer(
            question=question,
            gold_answer=gold["gold_answer"],
            gold_spans=gold["gold_spans"],
            generated_answer=trace.generated_answer
        )
        is_reused_b0 = False

    record = {
        "qid": qid,
        "system_id": "E1-Composer",
        "corpus": corpus,
        "question": question,
        "hop_count": gold["hop_count"],
        "tags": gold["tags"],
        "gold_chunk_ids": gold["gold_chunk_ids"],
        "gold_documents": gold.get("gold_documents", []),
        "gold_answer": gold["gold_answer"],
        "generated_answer": trace.generated_answer,
        "final_evidence_chunk_ids": trace.final_evidence_chunk_ids,
        "evidence_tokens": trace.evidence_tokens,
        "is_correct": judge_result["is_correct"],
        "judge_confidence": judge_result["confidence"],
        "judge_reasoning": judge_result["reasoning"],
        "retrieval": ret_metrics,
        "is_reused_b0_output": is_reused_b0,
        "hub_encounters": trace.hub_encounters,
        "hub_expansions_prevented": trace.hub_expansions_prevented,
        "feasibility_rejects": trace.feasibility_rejects,
        "active_fallbacks": trace.active_fallbacks,
        "retrieval_latency_ms": trace.retrieval_latency_ms,
        "routing_latency_ms": trace.routing_latency_ms,
        "llm_latency_ms": trace.llm_latency_ms,
        "total_latency_ms": trace.total_latency_ms,
        "input_tokens": trace.input_tokens,
        "output_tokens": trace.output_tokens,
        "llm_calls": trace.llm_calls + (0 if can_pass_through else 1),
        "evidence_complete": set(gold["gold_chunk_ids"]).issubset(set(trace.final_evidence_chunk_ids)),
        "metadata": trace.metadata
    }

    with open(trace_file, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return record


def run_e1_evaluation(concurrency: int = 5, fresh_mode: bool = False) -> Dict[str, Any]:
    mode_str = "FULL FRESH RUN" if fresh_mode else "STANDARD PASS-THROUGH RUN"
    print(f"\n=======================================================")
    print(f" Executing E1 Evaluation [{mode_str}] (N = 216)")
    print(f"=======================================================")

    test_qids, gold_map = load_benchmark()
    b0_traces = load_b0_traces()
    d1_traces = load_d1_traces()
    tasks = sorted(list(b0_traces.keys()))

    search_service = SearchService()
    llm_service = LLMService()
    lsdb = KnowledgeLSDB()
    evaluator = Evaluator(llm_service=llm_service)

    b0_cache = {} if fresh_mode else b0_traces
    system_instance = E1ComposerRouterSystem(
        search_service=search_service,
        llm_service=llm_service,
        lsdb=lsdb,
        b0_traces=b0_cache,
        top_k=5,
        shadow_top_k=20,
        max_hops=2,
        max_evidence_tokens=4000,
        generation_mode="RAW"
    )

    out_name = "E1_fresh" if fresh_mode else "E1"
    out_dir = RUNS_V3_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    completed_records = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_task = {
            executor.submit(
                run_single_e1_task,
                task_key=t,
                gold=gold_map[t[0]],
                system_instance=system_instance,
                evaluator=evaluator,
                b0_traces=b0_traces,
                out_dir=out_dir,
                fresh_mode=fresh_mode
            ): t for t in tasks
        }
        count = 0
        for fut in as_completed(future_to_task):
            rec = fut.result()
            if rec:
                completed_records.append(rec)
            count += 1
            if count % 25 == 0 or count == len(tasks):
                print(f"  [E1] Progress: {count}/{len(tasks)} ({(count/len(tasks)*100):.1f}%) completed...")

    elapsed = time.time() - t_start
    print(f"Finished {len(completed_records)} tasks in {elapsed:.1f}s.")

    # Detailed Comparative Analysis
    e1_map = {(r["qid"], r["corpus"]): r for r in completed_records}
    common_keys = sorted(list(set(b0_traces.keys()) & set(d1_traces.keys()) & set(e1_map.keys())))
    N = len(common_keys)

    # 1. Answer Accuracy
    b0_correct = [b0_traces[k]["is_correct"] for k in common_keys]
    d1_correct = [d1_traces[k]["is_correct"] for k in common_keys]
    e1_correct = [e1_map[k]["is_correct"] for k in common_keys]

    b0_acc = sum(b0_correct) / N * 100
    d1_acc = sum(d1_correct) / N * 100
    e1_acc = sum(e1_correct) / N * 100

    # Rescues and Regressions vs B0
    e1_rescues_b0 = [f"{k[0]}_{k[1]}" for k, b, e in zip(common_keys, b0_correct, e1_correct) if not b and e]
    e1_regressions_b0 = [f"{k[0]}_{k[1]}" for k, b, e in zip(common_keys, b0_correct, e1_correct) if b and not e]
    e1_net_rescue_b0 = len(e1_rescues_b0) - len(e1_regressions_b0)

    # Rescues and Regressions vs D1
    e1_rescues_d1 = [f"{k[0]}_{k[1]}" for k, d, e in zip(common_keys, d1_correct, e1_correct) if not d and e]
    e1_regressions_d1 = [f"{k[0]}_{k[1]}" for k, d, e in zip(common_keys, d1_correct, e1_correct) if d and not e]
    e1_net_rescue_d1 = len(e1_rescues_d1) - len(e1_regressions_d1)

    # Hop breakdown
    hops = [gold_map[k[0]]["hop_count"] for k in common_keys]
    hop1_keys = [k for k, h in zip(common_keys, hops) if h == 1]
    hop_multi_keys = [k for k, h in zip(common_keys, hops) if h > 1]

    e1_hop1_acc = sum(1 for k in hop1_keys if e1_map[k]["is_correct"]) / len(hop1_keys) * 100
    b0_hop1_acc = sum(1 for k in hop1_keys if b0_traces[k]["is_correct"]) / len(hop1_keys) * 100
    d1_hop1_acc = sum(1 for k in hop1_keys if d1_traces[k]["is_correct"]) / len(hop1_keys) * 100

    e1_multi_acc = sum(1 for k in hop_multi_keys if e1_map[k]["is_correct"]) / len(hop_multi_keys) * 100
    b0_multi_acc = sum(1 for k in hop_multi_keys if b0_traces[k]["is_correct"]) / len(hop_multi_keys) * 100
    d1_multi_acc = sum(1 for k in hop_multi_keys if d1_traces[k]["is_correct"]) / len(hop_multi_keys) * 100

    # McNemar test
    mcnemar_e1_b0 = Evaluator.mcnemar_exact_test(b0_correct, e1_correct)
    mcnemar_e1_d1 = Evaluator.mcnemar_exact_test(d1_correct, e1_correct)

    results_summary = {
        "N": N,
        "fresh_mode": fresh_mode,
        "execution_time_s": round(elapsed, 2),
        "accuracy": {
            "d0_b0": round(b0_acc, 2),
            "d0_correct": sum(b0_correct),
            "d1_clean_raw": round(d1_acc, 2),
            "d1_correct": sum(d1_correct),
            "e1_composer": round(e1_acc, 2),
            "e1_correct": sum(e1_correct),
            "delta_e1_vs_b0": round(e1_acc - b0_acc, 2),
            "delta_e1_vs_d1": round(e1_acc - d1_acc, 2)
        },
        "answer_transitions_vs_b0": {
            "rescues_count": len(e1_rescues_b0),
            "regressions_count": len(e1_regressions_b0),
            "net_rescue": e1_net_rescue_b0,
            "rescues": e1_rescues_b0,
            "regressions": e1_regressions_b0
        },
        "answer_transitions_vs_d1": {
            "rescues_count": len(e1_rescues_d1),
            "regressions_count": len(e1_regressions_d1),
            "net_rescue": e1_net_rescue_d1,
            "rescues": e1_rescues_d1,
            "regressions": e1_regressions_d1
        },
        "hop_breakdown": {
            "hop1": {
                "count": len(hop1_keys),
                "d0_b0_acc": round(b0_hop1_acc, 2),
                "d1_clean_acc": round(d1_hop1_acc, 2),
                "e1_composer_acc": round(e1_hop1_acc, 2)
            },
            "multi_hop": {
                "count": len(hop_multi_keys),
                "d0_b0_acc": round(b0_multi_acc, 2),
                "d1_clean_acc": round(d1_multi_acc, 2),
                "e1_composer_acc": round(e1_multi_acc, 2)
            }
        },
        "statistical_tests": {
            "mcnemar_e1_vs_b0_p": mcnemar_e1_b0["p_value"],
            "mcnemar_e1_vs_d1_p": mcnemar_e1_d1["p_value"]
        }
    }

    print("\n-------------------------------------------------------")
    print(" E1 Answer Evaluation Results Summary:")
    print("-------------------------------------------------------")
    print(f"B0 Baseline Accuracy : {b0_acc:.2f}% ({sum(b0_correct)}/{N})")
    print(f"D1 Clean-Raw Accuracy: {d1_acc:.2f}% ({sum(d1_correct)}/{N})")
    print(f"E1 Composer Accuracy : {e1_acc:.2f}% ({sum(e1_correct)}/{N}) (Δ vs B0: {e1_acc-b0_acc:+.2f}pp, Δ vs D1: {e1_acc-d1_acc:+.2f}pp)")
    print(f"Rescues vs B0        : {len(e1_rescues_b0)} | Regressions: {len(e1_regressions_b0)} | Net: {e1_net_rescue_b0:+d}")
    print(f"Rescues vs D1        : {len(e1_rescues_d1)} | Regressions: {len(e1_regressions_d1)} | Net: {e1_net_rescue_d1:+d}")
    print(f"1-Hop Accuracy       : B0 = {b0_hop1_acc:.2f}% | D1 = {d1_hop1_acc:.2f}% | E1 = {e1_hop1_acc:.2f}%")
    print(f"Multi-Hop Accuracy   : B0 = {b0_multi_acc:.2f}% | D1 = {d1_multi_acc:.2f}% | E1 = {e1_multi_acc:.2f}%")
    print(f"McNemar p (E1 vs B0) : {mcnemar_e1_b0['p_value']:.4f}")
    print(f"McNemar p (E1 vs D1) : {mcnemar_e1_d1['p_value']:.4f}")

    out_file = REPORTS_DIR / f"v3_e1_answer_{'fresh' if fresh_mode else 'summary'}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved answer summary to {out_file}")

    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=5, help="Worker concurrency")
    parser.add_argument("--fresh", action="store_true", help="Force fresh generation for all instances")
    args = parser.parse_args()

    run_e1_evaluation(concurrency=args.concurrency, fresh_mode=args.fresh)
