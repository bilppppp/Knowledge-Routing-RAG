#!/usr/bin/env python3
"""
scripts/run_v2_candidate.py
Unified Runner and Evaluator for V2 Directed Search Candidates.
Supports:
  --candidate C1 (or C2, C3, etc.)
  --concurrency 5

Compares candidate results with B0 on the exact same test instances (N=216 across D20, D50, D100).
Updates:
  - reports/v2_candidate_results.csv
  - reports/v2_search_log.md
  - reports/v2_candidate_tree.md
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.embedding import EmbeddingService
from src.services.search import SearchService
from src.services.llm import LLMService
from src.graph.lsdb import KnowledgeLSDB
from src.evaluation.metrics import Evaluator
from src.routing.c1_router import C1RouterSystem
from src.routing.c2_router import C2RouterSystem
from src.routing.c3_router import C3RouterSystem
from src.routing.c4_router import C4RouterSystem
from src.routing.c5_router import C5RouterSystem
from src.routing.c6_router import C6RouterSystem
from src.routing.c7_router import C7RouterSystem

BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
RUNS_V2_DIR = PROJECT_ROOT / "runs" / "v2"
RUNS_TEST_DIR = PROJECT_ROOT / "runs" / "test"
REPORTS_DIR = PROJECT_ROOT / "reports"

RUNS_V2_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_b0_traces() -> Dict[Tuple[str, str], Dict[str, Any]]:
    b0_map = {}
    for p in RUNS_TEST_DIR.glob("B0_*/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            b0_map[(d["qid"], d["corpus"])] = d
    return b0_map


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


def get_candidate_system(candidate_id: str, search: SearchService, llm: LLMService, lsdb: KnowledgeLSDB):
    if candidate_id == "C1":
        return C1RouterSystem(
            search_service=search,
            llm_service=llm,
            lsdb=lsdb,
            seed_top_k=5,
            b0_anchor_slots=3,
            max_hops=2,
            max_branch_per_node=2,
            max_final_evidence=6,
            max_evidence_tokens=4000
        )
    elif candidate_id == "C2":
        return C2RouterSystem(
            search_service=search,
            llm_service=llm,
            lsdb=lsdb,
            seed_top_k=5,
            max_per_doc_seeds=2,
            confidence_top1_thresh=0.76,
            confidence_margin_thresh=0.035,
            max_hops=2,
            max_branch_per_node=2,
            max_final_evidence=6,
            max_evidence_tokens=4000
        )
    elif candidate_id == "C3":
        b0_traces = load_b0_traces()
        return C3RouterSystem(
            search_service=search,
            llm_service=llm,
            lsdb=lsdb,
            b0_traces=b0_traces,
            seed_top_k=5,
            b0_anchor_slots=4,
            max_hops=2,
            max_branch_per_node=2,
            max_final_evidence=6,
            max_evidence_tokens=4000
        )
    elif candidate_id == "C4":
        b0_traces = load_b0_traces()
        return C4RouterSystem(
            search_service=search,
            llm_service=llm,
            lsdb=lsdb,
            b0_traces=b0_traces,
            top_k=5,
            max_hops=2,
            max_replacements=2,
            max_evidence_tokens=4000
        )
    elif candidate_id == "C5":
        b0_traces = load_b0_traces()
        return C5RouterSystem(
            search_service=search,
            llm_service=llm,
            lsdb=lsdb,
            b0_traces=b0_traces,
            top_k=5,
            max_hops=2,
            max_replacements=2,
            max_evidence_tokens=4000
        )
    elif candidate_id == "C6":
        b0_traces = load_b0_traces()
        return C6RouterSystem(
            search_service=search,
            llm_service=llm,
            lsdb=lsdb,
            b0_traces=b0_traces,
            top_k=5,
            max_hops=2,
            max_replacements=2,
            max_evidence_tokens=4000
        )
    elif candidate_id == "C7":
        b0_traces = load_b0_traces()
        return C7RouterSystem(
            search_service=search,
            llm_service=llm,
            lsdb=lsdb,
            b0_traces=b0_traces,
            top_k=5,
            shadow_top_k=20,
            max_hops=2,
            max_replacements=2,
            max_evidence_tokens=4000
        )
    else:
        raise ValueError(f"Unknown candidate system: {candidate_id}")


def run_candidate(candidate_id: str, concurrency: int = 5):
    print(f"\n=======================================================")
    print(f" Running V2 Directed Search: Candidate {candidate_id}")
    print(f"=======================================================")

    test_qids, gold_map = load_benchmark()
    b0_traces = load_b0_traces()
    print(f"Loaded {len(test_qids)} test questions, {len(b0_traces)} B0 reference traces.")

    search_service = SearchService()
    llm_service = LLMService()
    lsdb = KnowledgeLSDB()
    evaluator = Evaluator(llm_service=llm_service)

    system_instance = get_candidate_system(candidate_id, search_service, llm_service, lsdb)

    # Build evaluation tasks matching B0's test keys
    tasks = sorted(list(b0_traces.keys()))
    print(f"Total evaluation instances for {candidate_id}: {len(tasks)}")

    out_dir = RUNS_V2_DIR / candidate_id
    out_dir.mkdir(parents=True, exist_ok=True)

    completed_records = []
    t_start = time.time()

    def worker(task_key):
        qid, corpus = task_key
        gold = gold_map[qid]
        question = gold["question"]

        corpus_dir = out_dir / f"{candidate_id}_{corpus}"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        trace_file = corpus_dir / f"{qid}.json"

        if trace_file.exists():
            with open(trace_file, "r", encoding="utf-8") as f:
                return json.load(f)

        try:
            trace = system_instance.run(qid=qid, question=question, corpus=corpus)
        except Exception as e:
            print(f"  [ERROR] {candidate_id} on {qid} ({corpus}): {e}")
            return None

        if (
            task_key in b0_traces
            and trace.final_evidence_chunk_ids == b0_traces[task_key]["final_evidence_chunk_ids"]
            and trace.generated_answer == b0_traces[task_key]["generated_answer"]
        ):
            ret_metrics = b0_traces[task_key]["retrieval"]
            judge_result = {
                "is_correct": b0_traces[task_key]["is_correct"],
                "confidence": b0_traces[task_key]["judge_confidence"],
                "reasoning": b0_traces[task_key]["judge_reasoning"]
            }
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

        record = {
            "qid": qid,
            "system_id": candidate_id,
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
            "llm_calls": trace.llm_calls,
            "generation_mode": trace.metadata.get("generation_mode", "STANDARD"),
            "question_slots": trace.metadata.get("question_slots", []),
            "slot_evidence_bindings": trace.metadata.get("slot_evidence_bindings", {}),
            "evidence_complete": set(gold["gold_chunk_ids"]).issubset(set(trace.final_evidence_chunk_ids)),
            "synthesis_flags": trace.metadata.get("synthesis_flags", []),
            "routing_steps": [
                {
                    "step_num": s.step_num,
                    "action": s.action,
                    "current_node": s.current_node,
                    "candidates": s.candidates,
                    "selected_next": s.selected_next,
                    "reason": s.reason
                }
                for s in trace.routing_steps
            ]
        }

        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        return record

    print(f"Executing tasks with ThreadPoolExecutor (concurrency={concurrency})...")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_task = {executor.submit(worker, t): t for t in tasks}
        count = 0
        for fut in as_completed(future_to_task):
            rec = fut.result()
            if rec:
                completed_records.append(rec)
            count += 1
            if count % 25 == 0 or count == len(tasks):
                print(f"  Progress: {count}/{len(tasks)} ({(count/len(tasks)*100):.1f}%) completed...")

    elapsed = time.time() - t_start
    print(f"\nAll {len(completed_records)} runs finished in {elapsed:.1f}s.")

    # Compute Comparative Metrics against B0
    evaluate_candidate_vs_b0(candidate_id, completed_records, b0_traces, gold_map)


def evaluate_candidate_vs_b0(
    candidate_id: str,
    cand_records: List[Dict[str, Any]],
    b0_traces: Dict[Tuple[str, str], Dict[str, Any]],
    gold_map: Dict[str, Any]
):
    cand_map = {(r["qid"], r["corpus"]): r for r in cand_records}
    common_keys = sorted(list(set(b0_traces.keys()) & set(cand_map.keys())))
    N = len(common_keys)

    cand_corr = sum(1 for k in common_keys if cand_map[k]["is_correct"])
    b0_corr = sum(1 for k in common_keys if b0_traces[k]["is_correct"])

    cand_acc = cand_corr / N if N else 0.0
    b0_acc = b0_corr / N if N else 0.0
    delta_vs_b0 = (cand_acc - b0_acc) * 100

    rescues = sum(1 for k in common_keys if not b0_traces[k]["is_correct"] and cand_map[k]["is_correct"])
    regressions = sum(1 for k in common_keys if b0_traces[k]["is_correct"] and not cand_map[k]["is_correct"])
    net_rescue = rescues - regressions

    f1_list = [cand_map[k]["retrieval"]["f1"] for k in common_keys]
    cpr_list = [cand_map[k]["retrieval"]["cpr"] for k in common_keys]

    # Chain Completion: full overlap of all gold chunks in final evidence
    chain_complete_count = 0
    for k in common_keys:
        g_chunks = set(gold_map[k[0]]["gold_chunk_ids"])
        ev_chunks = set(cand_map[k]["final_evidence_chunk_ids"])
        if g_chunks.issubset(ev_chunks):
            chain_complete_count += 1
    chain_completion_rate = (chain_complete_count / N) * 100 if N else 0.0

    mean_f1 = float(np.mean(f1_list))
    mean_cpr = float(np.mean(cpr_list)) * 100

    tot_latencies = [cand_map[k]["total_latency_ms"] for k in common_keys]
    p50_lat = float(np.percentile(tot_latencies, 50))
    p95_lat = float(np.percentile(tot_latencies, 95))
    mean_tokens = float(np.mean([cand_map[k]["input_tokens"] + cand_map[k]["output_tokens"] for k in common_keys]))

    # Status Determination per V2 Specification
    if cand_acc > b0_acc:
        status = "CANDIDATE / NEW INCUMBENT"
    else:
        status = "FAILED"

    print("\n=======================================================")
    print(f" Evaluation Summary: {candidate_id} vs B0")
    print(f"=======================================================")
    print(f"Instance Count N:       {N}")
    print(f"B0 Accuracy:            {b0_acc*100:.1f}% ({b0_corr}/{N})")
    print(f"{candidate_id} Accuracy:            {cand_acc*100:.1f}% ({cand_corr}/{N})")
    print(f"Delta vs B0:            {delta_vs_b0:+.2f}pp")
    print(f"Rescues (Base- -> C+):  {rescues}")
    print(f"Regressions (Base+->C-):{regressions}")
    print(f"Net Rescue:             {net_rescue:+d}")
    print(f"Chain Completion:       {chain_completion_rate:.1f}% ({chain_complete_count}/{N})")
    print(f"Evidence F1:            {mean_f1:.3f}")
    print(f"CPR:                    {mean_cpr:.1f}%")
    print(f"Latency P50 / P95:      {p50_lat:.0f} / {p95_lat:.0f} ms")
    print(f"Mean Tokens:            {mean_tokens:.0f}")
    print(f"V2 Stage Status:        {status}")

    # Candidate-specific metadata
    if candidate_id == "C1":
        parent = "B0"
        main_change = "Vector Entrance + Scope Narrowing + Feasibility Routing + Soft Hub + Targeted Gap + B0 Anchors"
        parent_acc = b0_acc
    elif candidate_id == "C2":
        parent = "C1"
        main_change = "Hierarchical Scope Routing + Diversity Admission Gate (Cap 2/doc) + Confidence Fast Path"
        parent_acc = 0.7130
    elif candidate_id == "C3":
        parent = "C1"
        main_change = "Intent-Gated Relational Routing + 4 Guaranteed B0 Anchors + B0 Fast Path"
        parent_acc = 0.7130
    elif candidate_id == "C4":
        parent = "C3"
        main_change = "Relation-Specific Lanes (Temporal/Composite) + Conservative Evidence Admission"
        parent_acc = 0.7315
    elif candidate_id == "C5":
        parent = "C4"
        main_change = "Hierarchical Next-Hop Resolution (Recursive Parent Lift + Targeted Descent)"
        parent_acc = 0.7454
    elif candidate_id == "C6":
        parent = "C5"
        main_change = "Evidence-Contract Synthesis (Slot Decomposition + Semantic Binding + Substantive Contract)"
        parent_acc = 0.7593
    elif candidate_id == "C7":
        parent = "C6"
        main_change = "Shadow Candidate Plane (Top-20 RIB) + Route-Prefix Resolution + Targeted Descent"
        parent_acc = 0.7731
    else:
        parent = "B0"
        main_change = candidate_id
        parent_acc = b0_acc
    delta_parent = (cand_acc - parent_acc) * 100

    # Evidence-Complete Subset Analysis
    ev_complete_keys = [
        k for k in common_keys
        if set(gold_map[k[0]]["gold_chunk_ids"]).issubset(set(cand_map[k]["final_evidence_chunk_ids"]))
    ]
    N_ev = len(ev_complete_keys)
    cand_ev_corr = sum(1 for k in ev_complete_keys if cand_map[k]["is_correct"])
    cand_ev_acc = (cand_ev_corr / N_ev * 100) if N_ev else 0.0
    b0_ev_corr = sum(1 for k in ev_complete_keys if b0_traces[k]["is_correct"])
    b0_ev_acc = (b0_ev_corr / N_ev * 100) if N_ev else 0.0
    print(f"Evidence-Complete Subset: {N_ev}/{N}")
    print(f"  {candidate_id} Ev-Complete Acc:   {cand_ev_acc:.1f}% ({cand_ev_corr}/{N_ev})")
    print(f"  B0 Ev-Complete Acc:          {b0_ev_acc:.1f}% ({b0_ev_corr}/{N_ev})")

    # Update reports/v2_candidate_results.csv
    update_results_csv(
        candidate_id=candidate_id,
        parent=parent,
        main_change=main_change,
        accuracy=cand_acc * 100,
        delta_b0=delta_vs_b0,
        delta_parent=delta_parent,
        rescues=rescues,
        regressions=regressions,
        net_rescue=net_rescue,
        f1=mean_f1,
        chain_comp=chain_completion_rate,
        cpr=mean_cpr,
        latency=p50_lat,
        tokens=mean_tokens,
        status=status
    )


def update_results_csv(
    candidate_id: str,
    parent: str,
    main_change: str,
    accuracy: float,
    delta_b0: float,
    delta_parent: float,
    rescues: int,
    regressions: int,
    net_rescue: int,
    f1: float,
    chain_comp: float,
    cpr: float,
    latency: float,
    tokens: float,
    status: str
):
    csv_file = REPORTS_DIR / "v2_candidate_results.csv"
    file_exists = csv_file.exists()

    header = "candidate_id,parent_candidate,main_change,accuracy,delta_vs_B0,delta_vs_parent,rescues,regressions,net_rescue,evidence_f1,chain_completion,cpr,latency_p50_ms,tokens,status\n"
    row = f"{candidate_id},{parent},\"{main_change}\",{accuracy:.2f},{delta_b0:+.2f},{delta_parent:+.2f},{rescues},{regressions},{net_rescue:+d},{f1:.3f},{chain_comp:.1f},{cpr:.1f},{latency:.0f},{tokens:.0f},{status}\n"

    # Read existing rows to prevent duplicate candidate_id entries
    existing_rows = []
    if file_exists:
        with open(csv_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                existing_rows = [l for l in lines[1:] if not l.startswith(f"{candidate_id},")]

    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(header)
        for r in existing_rows:
            f.write(r)
        f.write(row)

    print(f"Updated CSV: {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, default="C1", help="Candidate ID (e.g. C1)")
    parser.add_argument("--concurrency", type=int, default=5, help="Worker concurrency")
    args = parser.parse_args()

    run_candidate(candidate_id=args.candidate, concurrency=args.concurrency)
