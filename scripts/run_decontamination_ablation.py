#!/usr/bin/env python3
"""
scripts/run_decontamination_ablation.py
Decontamination / Ablation Audit Runner for Candidate C7.

Evaluates:
  - A0: B0 Baseline (N=216)
  - A1: C7-Raw (C7 Architecture + B0 Prompt)
  - A2: C7-Clean-Raw (C7-Clean Architecture + B0 Prompt)
  - A3: C7-Clean-Generic (C7-Clean Architecture + Generic Contract)
  - A3.5: C7-Spec-Generic (C7 Specialized Retrieval + Generic Contract)
  - A4: Current C7 (Commit d202f0b Reference)

Features:
  - Supports Standard Pass-Through mode and Full Fresh Run mode (--fresh)
  - Tracks REUSED_B0_OUTPUT vs FRESH_GENERATION counts
  - Tracks SAME_EVIDENCE_FLIP instances (both wrong->correct and correct->wrong)
  - Comprehensive metrics: Accuracy, Rescues, Regressions, F1, Chain Comp, Ev-Complete Acc, McNemar
  - Generates full structured JSON summary for downstream reporting
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

from src.services.embedding import EmbeddingService
from src.services.search import SearchService
from src.services.llm import LLMService
from src.graph.lsdb import KnowledgeLSDB
from src.evaluation.metrics import Evaluator
from src.common.prompt import SYSTEM_PROMPT, format_user_prompt, pack_evidence_context
from src.routing.c7_router import C7RouterSystem
from src.routing.c7_clean_router import C7CleanRouterSystem

BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
RUNS_TEST_DIR = PROJECT_ROOT / "runs" / "test"
RUNS_V2_DIR = PROJECT_ROOT / "runs" / "v2"
RUNS_ABLATION_DIR = PROJECT_ROOT / "runs" / "ablation"
REPORTS_DIR = PROJECT_ROOT / "reports"

RUNS_ABLATION_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_b0_traces() -> Dict[Tuple[str, str], Dict[str, Any]]:
    b0_map = {}
    for p in RUNS_TEST_DIR.glob("B0_*/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            b0_map[(d["qid"], d["corpus"])] = d
    return b0_map


def load_current_c7_traces() -> Dict[Tuple[str, str], Dict[str, Any]]:
    c7_map = {}
    for p in (RUNS_V2_DIR / "C7").glob("*/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            c7_map[(d["qid"], d["corpus"])] = d
    return c7_map


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


def get_ablation_system(
    system_id: str,
    search: SearchService,
    llm: LLMService,
    lsdb: KnowledgeLSDB,
    b0_traces: Dict[Tuple[str, str], Dict[str, Any]],
    fresh_mode: bool = False
):
    b0_cache = {} if fresh_mode else b0_traces

    if system_id == "A1":
        # C7-Raw: C7 Architecture + B0 Prompt
        return C7RouterSystem(
            search_service=search,
            llm_service=llm,
            lsdb=lsdb,
            b0_traces=b0_cache,
            top_k=5,
            shadow_top_k=20,
            max_hops=2,
            max_replacements=2,
            max_evidence_tokens=4000,
            generation_mode="RAW"
        )
    elif system_id == "A2":
        # C7-Clean-Raw: C7-Clean Architecture + B0 Prompt
        return C7CleanRouterSystem(
            search_service=search,
            llm_service=llm,
            lsdb=lsdb,
            b0_traces=b0_cache,
            top_k=5,
            shadow_top_k=20,
            max_hops=2,
            max_replacements=2,
            max_evidence_tokens=4000,
            generation_mode="RAW"
        )
    elif system_id == "A3":
        # C7-Clean-Generic: C7-Clean Architecture + Generic Contract
        return C7CleanRouterSystem(
            search_service=search,
            llm_service=llm,
            lsdb=lsdb,
            b0_traces=b0_cache,
            top_k=5,
            shadow_top_k=20,
            max_hops=2,
            max_replacements=2,
            max_evidence_tokens=4000,
            generation_mode="GENERIC_CONTRACT"
        )
    elif system_id == "A3.5":
        # C7-Spec-Generic: C7 Specialized Retrieval + Generic Contract
        return C7RouterSystem(
            search_service=search,
            llm_service=llm,
            lsdb=lsdb,
            b0_traces=b0_cache,
            top_k=5,
            shadow_top_k=20,
            max_hops=2,
            max_replacements=2,
            max_evidence_tokens=4000,
            generation_mode="GENERIC_CONTRACT"
        )
    elif system_id == "A0_FRESH":
        # Fresh baseline B0 (calls generator & judge fresh on B0 evidence)
        return None
    else:
        raise ValueError(f"Unknown ablation system_id: {system_id}")


def run_single_task(
    system_id: str,
    task_key: Tuple[str, str],
    gold: Dict[str, Any],
    system_instance: Any,
    evaluator: Evaluator,
    b0_traces: Dict[Tuple[str, str], Dict[str, Any]],
    out_dir: Path,
    fresh_mode: bool = False
) -> Dict[str, Any]:
    qid, corpus = task_key
    question = gold["question"]

    corpus_dir = out_dir / f"{system_id}_{corpus}"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    trace_file = corpus_dir / f"{qid}.json"

    if trace_file.exists():
        with open(trace_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Special handling for A0_FRESH
    if system_id == "A0_FRESH":
        b0_rec = b0_traces[task_key]
        evidence_cids = b0_rec["final_evidence_chunk_ids"]
        # Re-fetch evidence items from search/lsdb
        lsdb = evaluator.llm_service  # placeholder
        # Build prompt using B0 prompt
        # We can reconstruct evidence text from b0_rec
        ev_items = []
        # Reconstruct EvidenceItems from b0_rec or LSDB
        from src.common.models import EvidenceItem
        from src.graph.lsdb import KnowledgeLSDB
        lsdb_inst = KnowledgeLSDB()
        for cid in evidence_cids:
            chk = lsdb_inst.get_chunk_evidence(cid)
            if chk:
                ev_items.append(EvidenceItem(
                    chunk_id=chk["chunk_id"],
                    doc_id=chk["doc_id"],
                    title=chk["title"],
                    heading_path=chk["heading_path"],
                    text=chk["text"],
                    score=1.0
                ))
        ev_context = pack_evidence_context(ev_items, max_tokens=4000)
        u_prompt = format_user_prompt(question, ev_context)
        gen_ans, usage, lat_ms = evaluator.llm_service.generate(prompt=u_prompt, system_prompt=SYSTEM_PROMPT, seed=42)

        judge_res = evaluator.judge_answer(
            question=question,
            gold_answer=gold["gold_answer"],
            gold_spans=gold["gold_spans"],
            generated_answer=gen_ans
        )
        ret_metrics = evaluator.evaluate_retrieval(
            retrieved_ids=evidence_cids,
            gold_ids=gold["gold_chunk_ids"]
        )

        record = {
            "qid": qid,
            "system_id": system_id,
            "corpus": corpus,
            "question": question,
            "hop_count": gold["hop_count"],
            "tags": gold["tags"],
            "gold_chunk_ids": gold["gold_chunk_ids"],
            "gold_answer": gold["gold_answer"],
            "generated_answer": gen_ans,
            "final_evidence_chunk_ids": evidence_cids,
            "evidence_tokens": len(ev_context) // 2,
            "is_correct": judge_res["is_correct"],
            "judge_confidence": judge_res["confidence"],
            "judge_reasoning": judge_res["reasoning"],
            "retrieval": ret_metrics,
            "is_reused_b0_output": False,
            "total_latency_ms": lat_ms,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "llm_calls": 2,
            "metadata": {"fresh_run": True}
        }
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        return record

    try:
        trace = system_instance.run(qid=qid, question=question, corpus=corpus)
    except Exception as e:
        print(f"  [ERROR] {system_id} on {qid} ({corpus}): {e}")
        return None

    # Decision on judging: Pass-through vs Fresh Judge
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
        "system_id": system_id,
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


def run_ablation_system(
    system_id: str,
    concurrency: int = 5,
    fresh_mode: bool = False
) -> List[Dict[str, Any]]:
    mode_str = "FULL FRESH RUN (No Pass-Through)" if fresh_mode else "STANDARD PASS-THROUGH RUN"
    print(f"\n=======================================================")
    print(f" Executing Ablation: {system_id} [{mode_str}]")
    print(f"=======================================================")

    test_qids, gold_map = load_benchmark()
    b0_traces = load_b0_traces()
    tasks = sorted(list(b0_traces.keys()))

    search_service = SearchService()
    llm_service = LLMService()
    lsdb = KnowledgeLSDB()
    evaluator = Evaluator(llm_service=llm_service)

    system_instance = get_ablation_system(system_id, search_service, llm_service, lsdb, b0_traces, fresh_mode)

    sub_name = f"{system_id}_fresh" if fresh_mode else system_id
    out_dir = RUNS_ABLATION_DIR / sub_name
    out_dir.mkdir(parents=True, exist_ok=True)

    completed_records = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_task = {
            executor.submit(
                run_single_task,
                system_id=system_id,
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
                print(f"  [{system_id}] Progress: {count}/{len(tasks)} ({(count/len(tasks)*100):.1f}%) completed...")

    elapsed = time.time() - t_start
    print(f"Finished {len(completed_records)} instances in {elapsed:.1f}s.")
    return completed_records


def analyze_system_vs_baseline(
    cand_records: List[Dict[str, Any]],
    baseline_records: Dict[Tuple[str, str], Dict[str, Any]],
    gold_map: Dict[str, Any],
    label: str = ""
) -> Dict[str, Any]:
    cand_map = {(r["qid"], r["corpus"]): r for r in cand_records}
    common_keys = sorted(list(set(baseline_records.keys()) & set(cand_map.keys())))
    N = len(common_keys)

    cand_corr = sum(1 for k in common_keys if cand_map[k]["is_correct"])
    base_corr = sum(1 for k in common_keys if baseline_records[k]["is_correct"])

    cand_acc = cand_corr / N if N else 0.0
    base_acc = base_corr / N if N else 0.0
    delta_vs_base = (cand_acc - base_acc) * 100

    rescues = [k for k in common_keys if not baseline_records[k]["is_correct"] and cand_map[k]["is_correct"]]
    regressions = [k for k in common_keys if baseline_records[k]["is_correct"] and not cand_map[k]["is_correct"]]
    net_rescue = len(rescues) - len(regressions)

    f1_list = [cand_map[k]["retrieval"]["f1"] for k in common_keys]
    cpr_list = [cand_map[k]["retrieval"]["cpr"] for k in common_keys]
    mean_f1 = float(np.mean(f1_list))
    mean_cpr = float(np.mean(cpr_list)) * 100

    chain_complete_count = 0
    for k in common_keys:
        g_chunks = set(gold_map[k[0]]["gold_chunk_ids"])
        ev_chunks = set(cand_map[k]["final_evidence_chunk_ids"])
        if g_chunks.issubset(ev_chunks):
            chain_complete_count += 1
    chain_completion_rate = (chain_complete_count / N) * 100 if N else 0.0

    ev_complete_keys = [
        k for k in common_keys
        if set(gold_map[k[0]]["gold_chunk_ids"]).issubset(set(cand_map[k]["final_evidence_chunk_ids"]))
    ]
    N_ev = len(ev_complete_keys)
    cand_ev_corr = sum(1 for k in ev_complete_keys if cand_map[k]["is_correct"])
    cand_ev_acc = (cand_ev_corr / N_ev * 100) if N_ev else 0.0

    # SAME_EVIDENCE_FLIP analysis
    same_ev_flips = []
    same_ev_wrong_to_correct = []
    same_ev_correct_to_wrong = []
    reused_b0_count = sum(1 for k in common_keys if cand_map[k].get("is_reused_b0_output", False))
    fresh_gen_count = N - reused_b0_count

    for k in common_keys:
        b_ev = baseline_records[k]["final_evidence_chunk_ids"]
        c_ev = cand_map[k]["final_evidence_chunk_ids"]
        b_c = baseline_records[k]["is_correct"]
        c_c = cand_map[k]["is_correct"]
        if b_ev == c_ev and b_c != c_c:
            same_ev_flips.append((k, b_c, c_c))
            if not b_c and c_c:
                same_ev_wrong_to_correct.append(k)
            else:
                same_ev_correct_to_wrong.append(k)

    # McNemar test
    mcnemar = Evaluator.mcnemar_exact_test(
        baseline_correct=[baseline_records[k]["is_correct"] for k in common_keys],
        router_correct=[cand_map[k]["is_correct"] for k in common_keys]
    )

    summary = {
        "label": label,
        "N": N,
        "accuracy": round(cand_acc * 100, 2),
        "correct": cand_corr,
        "baseline_accuracy": round(base_acc * 100, 2),
        "baseline_correct": base_corr,
        "delta_vs_baseline": round(delta_vs_base, 2),
        "rescues_count": len(rescues),
        "regressions_count": len(regressions),
        "net_rescue": net_rescue,
        "rescues": [f"{k[0]}_{k[1]}" for k in rescues],
        "regressions": [f"{k[0]}_{k[1]}" for k in regressions],
        "evidence_f1": round(mean_f1, 3),
        "chain_completion": round(chain_completion_rate, 1),
        "evidence_complete_acc": round(cand_ev_acc, 1),
        "evidence_complete_count": f"{cand_ev_corr}/{N_ev}",
        "reused_b0_count": reused_b0_count,
        "fresh_gen_count": fresh_gen_count,
        "same_ev_flips_count": len(same_ev_flips),
        "same_ev_wrong_to_correct": [f"{k[0]}_{k[1]}" for k in same_ev_wrong_to_correct],
        "same_ev_correct_to_wrong": [f"{k[0]}_{k[1]}" for k in same_ev_correct_to_wrong],
        "mcnemar_p_value": mcnemar["p_value"]
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", default=["A1", "A2", "A3", "A3.5"], help="Systems to evaluate")
    parser.add_argument("--concurrency", type=int, default=5, help="Worker concurrency")
    parser.add_argument("--fresh", action="store_true", help="Run full fresh execution without pass-through")
    args = parser.parse_args()

    for sys_id in args.systems:
        run_ablation_system(system_id=sys_id, concurrency=args.concurrency, fresh_mode=args.fresh)
