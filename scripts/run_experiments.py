#!/usr/bin/env python3
"""
Unified Experiment Runner for Knowledge-Routing-RAG.
Reference: 实验方案.md Section 8, 9, 21, 22, 23, 24, 25, 26

Modes:
  --mode dev   : Runs exclusively on Dev set (24 questions) to tune/verify parameters.
  --mode test  : Runs on Test set (96 questions) with strictly FROZEN parameters.
  --mode hub   : Runs Hub Stress Test analysis across degree tiers.

Systems:
  B0 (Vector RAG)
  B1 (Hybrid Vector+FTS)
  K1 (RIFT Routing)
  K2 (EIGRP Feasibility)
  K3 (SRv6 Segment Plan)
  K4 (Full Router + Evidence Gap)

Outputs:
  - Per-query traces saved in runs/{mode}/{system}_{corpus}/{qid}.json
  - Aggregated metrics and evaluation summaries in reports/
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.services.embedding import EmbeddingService
from src.services.search import SearchService
from src.services.llm import LLMService
from src.graph.lsdb import KnowledgeLSDB
from src.retrieval.vector_rag import VectorRAGSystem
from src.retrieval.hybrid_rag import HybridRAGSystem
from src.routing.k1_rift import RIFTRouterSystem
from src.routing.k2_eigrp import EIGRPRouterSystem
from src.routing.k3_srv6 import SRv6RouterSystem
from src.routing.k4_router import KnowledgeRouterSystem
from src.evaluation.metrics import Evaluator

BENCHMARK_DIR = BASE_DIR / "benchmark"
RUNS_DIR = BASE_DIR / "runs"
REPORTS_DIR = BASE_DIR / "reports"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    with open(BENCHMARK_DIR / "split.json", "r", encoding="utf-8") as f:
        split_data = json.load(f)
    
    gold_map = {}
    with open(BENCHMARK_DIR / "gold.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gold_map[item["qid"]] = item

    return split_data, gold_map

def run_experiment(
    mode: str = "dev",
    systems_to_run: List[str] = None,
    max_questions: int = None
):
    print(f"\n=======================================================")
    print(f" Starting Experiment Run: Mode={mode.upper()}")
    print(f"=======================================================")

    split_data, gold_map = load_data()
    qids = split_data["dev_qids"] if mode == "dev" else split_data["test_qids"]
    if max_questions:
        qids = qids[:max_questions]
    print(f"Selected {len(qids)} questions for mode '{mode}'.")

    # Initialize shared services
    embedding_service = EmbeddingService()
    search_service = SearchService(embedding_service=embedding_service)
    llm_service = LLMService()
    lsdb = KnowledgeLSDB()
    evaluator = Evaluator(llm_service=llm_service)

    # Initialize all systems
    systems: Dict[str, Any] = {
        "B0": VectorRAGSystem(search_service, llm_service),
        "B1": HybridRAGSystem(search_service, llm_service),
        "K1": RIFTRouterSystem(search_service, llm_service, lsdb),
        "K2": EIGRPRouterSystem(search_service, llm_service, lsdb),
        "K3": SRv6RouterSystem(search_service, llm_service, lsdb),
        "K4": KnowledgeRouterSystem(search_service, llm_service, lsdb),
    }

    if systems_to_run:
        systems = {k: v for k, v in systems.items() if k in systems_to_run}

    print(f"Active Systems: {list(systems.keys())}")

    # Determine corpus runs for each question:
    # Core-60 queries run across D20, D50, D100
    # Build task list
    tasks = []
    for qid in qids:
        gold = gold_map[qid]
        for corpus in gold["answerable_in"]:
            for sys_id in systems.keys():
                tasks.append((qid, corpus, sys_id))

    print(f"Total evaluation tasks: {len(tasks)}")
    all_results = []
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    lock = threading.Lock()

    def process_task(task_tuple):
        qid, corpus, sys_id = task_tuple
        gold = gold_map[qid]
        question = gold["question"]
        sys_instance = systems[sys_id]

        trace_dir = RUNS_DIR / mode / f"{sys_id}_{corpus}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file = trace_dir / f"{qid}.json"

        if trace_file.exists():
            with open(trace_file, "r", encoding="utf-8") as tf:
                cached_record = json.load(tf)
            return cached_record

        try:
            trace = sys_instance.run(qid=qid, question=question, corpus=corpus)
        except Exception as e:
            print(f"  [ERROR] {sys_id} on {qid} ({corpus}): {e}")
            return None

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

        record = {
            "qid": qid,
            "system_id": sys_id,
            "corpus": corpus,
            "question": question,
            "hop_count": gold["hop_count"],
            "tags": gold["tags"],
            "gold_chunk_ids": gold["gold_chunk_ids"],
            "gold_answer": gold["gold_answer"],
            "generated_answer": trace.generated_answer,
            "final_evidence_chunk_ids": trace.final_evidence_chunk_ids,
            "evidence_tokens": trace.evidence_tokens,
            "is_correct": judge_result["is_correct"],
            "judge_confidence": judge_result["confidence"],
            "judge_reasoning": judge_result["reasoning"],
            "retrieval": ret_metrics,
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
            "llm_calls": trace.llm_calls
        }

        with open(trace_file, "w", encoding="utf-8") as tf:
            json.dump(record, tf, ensure_ascii=False, indent=2)

        status_icon = "✓" if record["is_correct"] else "✗"
        with lock:
            print(f"  {sys_id:<2} [{corpus:<4}] {qid} {status_icon} (F1:{ret_metrics['f1']:.2f}, CPR:{ret_metrics['cpr']:.2f}, Lat:{trace.total_latency_ms:.0f}ms)")
        return record

    max_workers = int(os.getenv("DEEPSEEK_MAX_CONCURRENCY", "5"))
    print(f"Executing with ThreadPoolExecutor (concurrency = {max_workers})...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_task, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                all_results.append(res)

    # Aggregate & Save summary report
    summary = aggregate_results(all_results, mode=mode)
    report_file = REPORTS_DIR / f"{mode}_summary.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved summary report to: {report_file}")
    return summary

def aggregate_results(records: List[Dict[str, Any]], mode: str = "dev") -> Dict[str, Any]:
    from collections import defaultdict
    import numpy as np

    grouped = defaultdict(list)
    for r in records:
        key = (r["system_id"], r["corpus"])
        grouped[key].append(r)

    summary: Dict[str, Any] = {"mode": mode, "systems": {}}

    for (sys_id, corpus), items in grouped.items():
        n = len(items)
        if n == 0:
            continue
        acc = sum(1 for x in items if x["is_correct"]) / n
        prec = float(np.mean([x["retrieval"]["precision"] for x in items]))
        rec = float(np.mean([x["retrieval"]["recall"] for x in items]))
        f1 = float(np.mean([x["retrieval"]["f1"] for x in items]))
        cpr = float(np.mean([x["retrieval"]["cpr"] for x in items]))
        lat_p50 = float(np.percentile([x["total_latency_ms"] for x in items], 50))
        lat_p95 = float(np.percentile([x["total_latency_ms"] for x in items], 95))
        tokens_mean = float(np.mean([x["input_tokens"] + x["output_tokens"] for x in items]))

        entry_name = f"{sys_id}_{corpus}"
        summary["systems"][entry_name] = {
            "system_id": sys_id,
            "corpus": corpus,
            "count": n,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "cpr": cpr,
            "latency_p50_ms": lat_p50,
            "latency_p95_ms": lat_p95,
            "tokens_mean": tokens_mean
        }

    # Print summary table
    print("\n=========================================================================================")
    print(f" EXPERIMENT SUMMARY ({mode.upper()})")
    print("=========================================================================================")
    print(f"{'System':<10} | {'Corpus':<8} | {'N':<4} | {'Accuracy':<10} | {'Prec':<7} | {'Recall':<7} | {'CPR':<7} | {'P50 (ms)':<9} | {'P95 (ms)'}")
    print("-" * 89)
    for k, v in sorted(summary["systems"].items()):
        print(f"{v['system_id']:<10} | {v['corpus']:<8} | {v['count']:<4} | {v['accuracy']*100:>6.1f}%   | {v['precision']:>5.3f} | {v['recall']:>5.3f} | {v['cpr']:>5.3f} | {v['latency_p50_ms']:>8.0f} | {v['latency_p95_ms']:>8.0f}")
    print("=" * 89)

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dev", "test"], default="dev")
    parser.add_argument("--systems", type=str, default="B0,B1,K1,K2,K3,K4")
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()

    sys_list = [s.strip() for s in args.systems.split(",") if s.strip()]
    run_experiment(mode=args.mode, systems_to_run=sys_list, max_questions=args.max)
