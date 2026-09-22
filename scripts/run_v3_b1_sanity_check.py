#!/usr/bin/env python3
"""
scripts/run_v3_b1_sanity_check.py
Stage B.1: Answer Sanity Check for Frozen Minimal Candidate (E2-Lite Lexical Only).

Evaluates end-to-end answers on Dev-216:
  - Uses original B0 raw prompt (100% frozen).
  - Tasks with evidence identical to E2/E1 reuse previous answers/judgments (zero judge variance, 0 same-evidence flips).
  - Tasks with different evidence generate fresh answers and are evaluated by LLM judge.
  - Outputs summary metrics and saves execution traces.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.search import SearchService
from src.services.llm import LLMService
from src.evaluation.metrics import Evaluator
from src.graph.lsdb import KnowledgeLSDB
from src.common.prompt import SYSTEM_PROMPT, pack_evidence_context, format_user_prompt
from src.common.models import EvidenceItem
from scripts.evaluate_v3_b1_ablation import load_previous_traces, run_descent_for_channel_mode

BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
RUNS_V3_E2_DIR = PROJECT_ROOT / "runs" / "v3" / "E2"
RUNS_V3_E2_LITE_DIR = PROJECT_ROOT / "runs" / "v3" / "E2_Lite"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_benchmark() -> Dict[str, Any]:
    gold_map = {}
    with open(BENCHMARK_DIR / "gold.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gold_map[item["qid"]] = item
    return gold_map


def load_e2_traces() -> Dict[Tuple[str, str], Dict[str, Any]]:
    e2_map = {}
    for p in RUNS_V3_E2_DIR.glob("*/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            e2_map[(d["qid"], d["corpus"])] = d
    return e2_map


def main():
    print("=" * 70)
    print("Stage B.1: Answer Sanity Check for Frozen Candidate (E2-Lite Lexical)")
    print("=" * 70)

    gold_map = load_benchmark()
    b0_traces, d1_traces, e1_traces = load_previous_traces()
    e2_traces = load_e2_traces()

    tasks = sorted(list(b0_traces.keys()))
    print(f"Total tasks: {len(tasks)}")

    search_service = SearchService()
    llm_service = LLMService()
    lsdb = KnowledgeLSDB()
    evaluator = Evaluator(llm_service=llm_service)

    # 1. Run retrieval for E2-Lite Lexical Only
    print("\nExecuting retrieval-only for E2-Lite (Lexical Only)...")
    res_l = run_descent_for_channel_mode("lexical_only", tasks, gold_map, b0_traces, search_service, lsdb)

    # 2. Compare evidence sets against E2-Hybrid and E1
    e2_lite_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    tasks_to_generate = []
    same_evidence_reused = 0

    for task_key in tasks:
        qid, corpus = task_key
        gold = gold_map[qid]
        l_trace = res_l["traces"][task_key]
        l_cids = l_trace["final_cids"]

        e2_trace = e2_traces.get(task_key)
        e2_cids = e2_trace["final_evidence_chunk_ids"] if e2_trace else []

        out_file = RUNS_V3_E2_LITE_DIR / f"E2_Lite_{corpus}" / f"{qid}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if e2_trace and l_cids == e2_cids:
            # Evidence is 100% identical to E2-Hybrid: reuse answer and judgment
            record = dict(e2_trace)
            record["system_id"] = "E2-Lite-Lexical"
            record["is_reused_e2_output"] = True
            record["metadata"]["descent_channel"] = "lexical_only_exact_match"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            e2_lite_records[task_key] = record
            same_evidence_reused += 1
        else:
            # Evidence differs: queue for fresh generation
            tasks_to_generate.append((task_key, l_cids, gold, l_trace))

    print(f"Same-evidence tasks reused (0 flips guaranteed): {same_evidence_reused}")
    print(f"Tasks requiring fresh answer generation and LLM judge: {len(tasks_to_generate)}")

    def process_fresh_task(item):
        task_key, l_cids, gold, l_trace = item
        qid, corpus = task_key
        question = gold["question"]

        evidence_items = []
        for cid in l_cids:
            cdata = lsdb.get_chunk_evidence(cid)
            if cdata:
                evidence_items.append(EvidenceItem(
                    chunk_id=cdata["chunk_id"], doc_id=cdata["doc_id"], title=cdata["title"],
                    heading_path=cdata["heading_path"], text=cdata["text"], score=0.95
                ))

        evidence_context = pack_evidence_context(evidence_items, max_tokens=4000)
        user_prompt = format_user_prompt(question, evidence_context)
        ans_content, usage, llm_latency_ms = llm_service.generate(prompt=user_prompt, system_prompt=SYSTEM_PROMPT)

        judge_res = evaluator.judge_answer(
            question=question,
            gold_answer=gold["gold_answer"],
            gold_spans=gold["gold_spans"],
            generated_answer=ans_content
        )

        ret_metrics = evaluator.evaluate_retrieval(
            retrieved_ids=l_cids,
            gold_ids=gold["gold_chunk_ids"]
        )

        out_rec = {
            "qid": qid,
            "system_id": "E2-Lite-Lexical",
            "corpus": corpus,
            "question": question,
            "hop_count": gold["hop_count"],
            "tags": gold["tags"],
            "gold_chunk_ids": gold["gold_chunk_ids"],
            "gold_documents": gold.get("gold_documents", []),
            "gold_answer": gold["gold_answer"],
            "generated_answer": ans_content,
            "final_evidence_chunk_ids": l_cids,
            "evidence_tokens": len(evidence_context) // 2,
            "is_correct": judge_res["is_correct"],
            "judge_confidence": judge_res["confidence"],
            "judge_reasoning": judge_res["reasoning"],
            "retrieval": ret_metrics,
            "is_reused_b0_output": False,
            "is_reused_e1_output": False,
            "is_reused_e2_output": False,
            "evidence_complete": set(gold["gold_chunk_ids"]).issubset(set(l_cids)),
            "metadata": {
                "lane": l_trace["lane"],
                "candidate_pool_chunk_ids": l_trace["candidate_pool_cids"],
                "descent_channel": "lexical_only_fresh"
            }
        }

        out_file = RUNS_V3_E2_LITE_DIR / f"E2_Lite_{corpus}" / f"{qid}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out_rec, f, ensure_ascii=False, indent=2)

        return task_key, out_rec

    if tasks_to_generate:
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_map = {executor.submit(process_fresh_task, item): item[0] for item in tasks_to_generate}
            for fut in as_completed(future_map):
                t_k = future_map[fut]
                try:
                    _, rec = fut.result()
                    e2_lite_records[t_k] = rec
                except Exception as e:
                    print(f"Error processing {t_k}: {e}")

    # 3. Accuracy and breakdown calculation
    total_correct = sum(1 for rec in e2_lite_records.values() if rec["is_correct"])
    acc = round(total_correct / len(tasks) * 100, 2)

    # Compare with B0
    b0_correct_count = sum(1 for b in b0_traces.values() if b["is_correct"])
    b0_acc = round(b0_correct_count / len(tasks) * 100, 2)

    rescues_b0, regressions_b0 = 0, 0
    for k in tasks:
        b_corr = b0_traces[k]["is_correct"]
        l_corr = e2_lite_records[k]["is_correct"]
        if l_corr and not b_corr:
            rescues_b0 += 1
        elif not l_corr and b_corr:
            regressions_b0 += 1

    # 1-hop vs Multi-hop
    one_hop_tasks = [k for k in tasks if gold_map[k[0]]["hop_count"] == 1]
    multi_hop_tasks = [k for k in tasks if gold_map[k[0]]["hop_count"] > 1]

    one_hop_b0_correct = sum(1 for k in one_hop_tasks if b0_traces[k]["is_correct"])
    one_hop_l_correct = sum(1 for k in one_hop_tasks if e2_lite_records[k]["is_correct"])
    one_hop_acc = round(one_hop_l_correct / len(one_hop_tasks) * 100, 2)

    multi_b0_correct = sum(1 for k in multi_hop_tasks if b0_traces[k]["is_correct"])
    multi_l_correct = sum(1 for k in multi_hop_tasks if e2_lite_records[k]["is_correct"])
    multi_acc = round(multi_l_correct / len(multi_hop_tasks) * 100, 2)

    print("\n" + "=" * 50)
    print("SANITY CHECK RESULTS:")
    print(f"  Total Accuracy:    {acc}% ({total_correct}/{len(tasks)}) (vs B0: {b0_acc}%)")
    print(f"  Rescues vs B0:     {rescues_b0}")
    print(f"  Regressions vs B0: {regressions_b0}")
    print(f"  Net Rescue vs B0:  +{rescues_b0 - regressions_b0}")
    print(f"  1-Hop Accuracy:    {one_hop_acc}% ({one_hop_l_correct}/{len(one_hop_tasks)})")
    print(f"  Multi-Hop Accuracy:{multi_acc}% ({multi_l_correct}/{len(multi_hop_tasks)})")
    print(f"  Same-Evidence Flips: 0")
    print("=" * 50)

    summary_data = {
        "N": len(tasks),
        "system_id": "E2-Lite-Lexical",
        "accuracy": acc,
        "correct_count": total_correct,
        "b0_accuracy": b0_acc,
        "b0_correct_count": b0_correct_count,
        "rescues_vs_b0": rescues_b0,
        "regressions_vs_b0": regressions_b0,
        "net_rescue_vs_b0": rescues_b0 - regressions_b0,
        "1_hop": {
            "N": len(one_hop_tasks),
            "accuracy": one_hop_acc,
            "correct_count": one_hop_l_correct,
            "b0_accuracy": round(one_hop_b0_correct / len(one_hop_tasks) * 100, 2)
        },
        "multi_hop": {
            "N": len(multi_hop_tasks),
            "accuracy": multi_acc,
            "correct_count": multi_l_correct,
            "b0_accuracy": round(multi_b0_correct / len(multi_hop_tasks) * 100, 2)
        },
        "same_evidence_flips": 0
    }

    summary_path = REPORTS_DIR / "v3_b1_sanity_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Saved sanity summary to {summary_path}")


if __name__ == "__main__":
    main()
