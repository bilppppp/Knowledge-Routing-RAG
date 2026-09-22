#!/usr/bin/env python3
"""
scripts/run_confirmation_evaluation.py
Execution Runner for Independent Holdout Confirmation Experiment.
Reference: Section XII-XXIII, XXXIII-XXXV of Confirmation Specification.

Executes:
  - Phase 1: 3 Independent Fresh Generation & Evaluation Runs:
             Run 1 (seed=101), Run 2 (seed=202), Run 3 (seed=303).
             Zero answer pass-through, zero cache reuse, fresh judge calls.
             Per-run metrics: Accuracy, Delta, Rescues, Regressions, McNemar test, Bootstrap CI.
  - Phase 2: Deterministic Evidence Retrieval & Retrieval-Only Analysis across D100 (H0 vs H1/H2)
             Metrics: Gold Doc/Chunk Recall, Precision, Recall, F1, CPR, Chain Completion, Rescues/Regressions.
  - Phase 3: Majority Voting & Stability Analysis:
             3-run majority accuracy, stable rescues, stable regressions, net stable rescue.
             Stratified analysis: 1-hop (Simple) vs. Multi-hop (Hard), tag-level analysis.
  - Phase 4: Blind Adjudication Pack:
             Anonymized review pack for discordant cases (H0 majority vs H1 majority).

Generates:
  - reports/confirmation_retrieval_analysis.json
  - reports/confirmation_run1.json
  - reports/confirmation_run2.json
  - reports/confirmation_run3.json
  - reports/confirmation_majority_results.json
  - reports/confirmation_blind_review_pack.json
  - reports/confirmation_blind_review_mapping.json
  - reports/confirmation_blind_review_decisions.json
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Any, Tuple, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.embedding import EmbeddingService
from src.services.search import SearchService
from src.services.llm import LLMService
from src.graph.lsdb import KnowledgeLSDB
from src.evaluation.metrics import Evaluator
from src.retrieval.vector_rag import VectorRAGSystem
from src.routing.c7_clean_router import C7CleanRouterSystem

CONFIRMATION_BENCHMARK_DIR = PROJECT_ROOT / "benchmark" / "confirmation"
RUNS_CONFIRMATION_DIR = PROJECT_ROOT / "runs" / "confirmation"
REPORTS_DIR = PROJECT_ROOT / "reports"

RUNS_CONFIRMATION_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_confirmation_benchmark():
    questions_file = CONFIRMATION_BENCHMARK_DIR / "questions.jsonl"
    gold_file = CONFIRMATION_BENCHMARK_DIR / "gold.jsonl"

    questions = []
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    gold_map = {}
    with open(gold_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gold_map[item["qid"]] = item

    return questions, gold_map


def get_seeded_llm_service(seed: int) -> LLMService:
    llm = LLMService()
    orig_generate = llm.generate

    def seeded_generate(*args, **kwargs):
        if "seed" not in kwargs or kwargs["seed"] == 42:
            kwargs["seed"] = seed
        return orig_generate(*args, **kwargs)

    llm.generate = seeded_generate
    return llm


class ConfirmationExperimentRunner:
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.search_service = SearchService()
        self.lsdb = KnowledgeLSDB()
        self.questions, self.gold_map = load_confirmation_benchmark()

    def run_single_fresh_run(self, run_num: int, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Runs a single fresh run across H0, H1, H2 with fixed seed.
        """
        print(f"\n=======================================================")
        print(f"  RUN {run_num}: FRESH GENERATION & JUDGE (seed={seed}, N={len(self.questions)})")
        print("=======================================================")

        run_dir = RUNS_CONFIRMATION_DIR / f"run{run_num}"
        (run_dir / "H0").mkdir(parents=True, exist_ok=True)
        (run_dir / "H1").mkdir(parents=True, exist_ok=True)
        (run_dir / "H2").mkdir(parents=True, exist_ok=True)

        seeded_llm = get_seeded_llm_service(seed=seed)
        evaluator = Evaluator(llm_service=seeded_llm)

        h0_system = VectorRAGSystem(
            search_service=self.search_service,
            llm_service=seeded_llm,
            top_k=5,
            max_evidence_tokens=4000
        )

        h1_system = C7CleanRouterSystem(
            search_service=self.search_service,
            llm_service=seeded_llm,
            lsdb=self.lsdb,
            b0_traces={},
            top_k=5,
            shadow_top_k=20,
            max_hops=2,
            max_replacements=2,
            max_evidence_tokens=4000,
            generation_mode="RAW"
        )

        h2_system = C7CleanRouterSystem(
            search_service=self.search_service,
            llm_service=seeded_llm,
            lsdb=self.lsdb,
            b0_traces={},
            top_k=5,
            shadow_top_k=20,
            max_hops=2,
            max_replacements=2,
            max_evidence_tokens=4000,
            generation_mode="GENERIC_CONTRACT"
        )

        results_h0: Dict[str, Dict[str, Any]] = {}
        results_h1: Dict[str, Dict[str, Any]] = {}
        results_h2: Dict[str, Dict[str, Any]] = {}

        def process_question(q_item):
            qid = q_item["qid"]
            question = q_item["question"]
            gold = self.gold_map[qid]
            gold_answer = gold["gold_answer"]
            gold_spans = gold["gold_spans"]

            # 1. Execute H0
            trace_h0 = h0_system.run(qid=qid, question=question, corpus="D100")
            judge_h0 = evaluator.judge_answer(
                question=question,
                gold_answer=gold_answer,
                gold_spans=gold_spans,
                generated_answer=trace_h0.generated_answer
            )

            # 2. Execute H1
            trace_h1 = h1_system.run(qid=qid, question=question, corpus="D100")
            judge_h1 = evaluator.judge_answer(
                question=question,
                gold_answer=gold_answer,
                gold_spans=gold_spans,
                generated_answer=trace_h1.generated_answer
            )

            # 3. Execute H2
            trace_h2 = h2_system.run(qid=qid, question=question, corpus="D100")
            judge_h2 = evaluator.judge_answer(
                question=question,
                gold_answer=gold_answer,
                gold_spans=gold_spans,
                generated_answer=trace_h2.generated_answer
            )

            rec_h0 = {
                "qid": qid, "system_id": "H0_B0_Fresh", "seed": seed,
                "generated_answer": trace_h0.generated_answer,
                "evidence_chunk_ids": trace_h0.final_evidence_chunk_ids,
                "is_correct": judge_h0["is_correct"],
                "confidence": judge_h0["confidence"],
                "reasoning": judge_h0["reasoning"]
            }
            rec_h1 = {
                "qid": qid, "system_id": "H1_Clean_Raw_Fresh", "seed": seed,
                "generated_answer": trace_h1.generated_answer,
                "evidence_chunk_ids": trace_h1.final_evidence_chunk_ids,
                "routing_steps": [s.model_dump() if hasattr(s, "model_dump") else s.dict() if hasattr(s, "dict") else str(s) for s in trace_h1.routing_steps],
                "is_correct": judge_h1["is_correct"],
                "confidence": judge_h1["confidence"],
                "reasoning": judge_h1["reasoning"]
            }
            rec_h2 = {
                "qid": qid, "system_id": "H2_Clean_Generic_Fresh", "seed": seed,
                "generated_answer": trace_h2.generated_answer,
                "evidence_chunk_ids": trace_h2.final_evidence_chunk_ids,
                "is_correct": judge_h2["is_correct"],
                "confidence": judge_h2["confidence"],
                "reasoning": judge_h2["reasoning"]
            }

            with open(run_dir / "H0" / f"{qid}.json", "w", encoding="utf-8") as f:
                json.dump(rec_h0, f, ensure_ascii=False)
            with open(run_dir / "H1" / f"{qid}.json", "w", encoding="utf-8") as f:
                json.dump(rec_h1, f, ensure_ascii=False)
            with open(run_dir / "H2" / f"{qid}.json", "w", encoding="utf-8") as f:
                json.dump(rec_h2, f, ensure_ascii=False)

            return qid, rec_h0, rec_h1, rec_h2

        print(f"Running H0, H1, H2 on {len(self.questions)} questions with {self.max_workers} workers...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_question, q) for q in self.questions]
            done_cnt = 0
            for f in as_completed(futures):
                qid, rh0, rh1, rh2 = f.result()
                results_h0[qid] = rh0
                results_h1[qid] = rh1
                results_h2[qid] = rh2
                done_cnt += 1
                if done_cnt % 50 == 0 or done_cnt == len(self.questions):
                    print(f"  Run {run_num} progress: {done_cnt}/{len(self.questions)}")

        h0_corr = [results_h0[q["qid"]]["is_correct"] for q in self.questions]
        h1_corr = [results_h1[q["qid"]]["is_correct"] for q in self.questions]
        h2_corr = [results_h2[q["qid"]]["is_correct"] for q in self.questions]

        h0_acc = float(np.mean(h0_corr))
        h1_acc = float(np.mean(h1_corr))
        h2_acc = float(np.mean(h2_corr))

        mcnemar_h1_h0 = evaluator.mcnemar_exact_test(h0_corr, h1_corr)
        mcnemar_h2_h1 = evaluator.mcnemar_exact_test(h1_corr, h2_corr)
        mcnemar_h2_h0 = evaluator.mcnemar_exact_test(h0_corr, h2_corr)

        deltas = [1.0 if r else 0.0 - (1.0 if b else 0.0) for b, r in zip(h0_corr, h1_corr)]
        _, ci_low, ci_high = evaluator.bootstrap_ci(deltas)

        run_summary = {
            "run_num": run_num,
            "seed": seed,
            "h0_accuracy": h0_acc,
            "h0_correct_count": sum(h0_corr),
            "h1_accuracy": h1_acc,
            "h1_correct_count": sum(h1_corr),
            "h2_accuracy": h2_acc,
            "h2_correct_count": sum(h2_corr),
            "delta_h1_vs_h0_pp": (h1_acc - h0_acc) * 100.0,
            "delta_h2_vs_h1_pp": (h2_acc - h1_acc) * 100.0,
            "delta_h2_vs_h0_pp": (h2_acc - h0_acc) * 100.0,
            "h1_rescues": mcnemar_h1_h0["rescues"],
            "h1_regressions": mcnemar_h1_h0["regressions"],
            "h1_net_rescue": mcnemar_h1_h0["rescues"] - mcnemar_h1_h0["regressions"],
            "mcnemar_h1_h0_p_value": mcnemar_h1_h0["p_value"],
            "delta_95_ci": [ci_low * 100.0, ci_high * 100.0],
            "per_question_h0": {q["qid"]: results_h0[q["qid"]]["is_correct"] for q in self.questions},
            "per_question_h1": {q["qid"]: results_h1[q["qid"]]["is_correct"] for q in self.questions},
            "per_question_h2": {q["qid"]: results_h2[q["qid"]]["is_correct"] for q in self.questions}
        }

        print(f"\n--- Run {run_num} Results (seed={seed}) ---")
        print(f"H0 (B0 Fresh)          : {h0_acc*100:.2f}% ({sum(h0_corr)}/200)")
        print(f"H1 (Clean-Raw Fresh)   : {h1_acc*100:.2f}% ({sum(h1_corr)}/200) [Δ = {run_summary['delta_h1_vs_h0_pp']:+.2f}pp]")
        print(f"H2 (Clean-Generic Fresh: {h2_acc*100:.2f}% ({sum(h2_corr)}/200) [Δ = {run_summary['delta_h2_vs_h0_pp']:+.2f}pp]")
        print(f"H1 Rescues / Regress   : Rescues={run_summary['h1_rescues']}, Regressions={run_summary['h1_regressions']}, Net={run_summary['h1_net_rescue']:+d}")
        print(f"McNemar p-value (H1-H0): {run_summary['mcnemar_h1_h0_p_value']:.4f}")

        with open(REPORTS_DIR / f"confirmation_run{run_num}.json", "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2, ensure_ascii=False)

        return run_summary, results_h0, results_h1, results_h2

    def run_retrieval_analysis(self, results_h0: Dict[str, Any], results_h1: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 2: Deterministic Evidence Retrieval Analysis using traces from Run 1.
        """
        print("\n=======================================================")
        print(f"  PHASE 2: RETRIEVAL-ONLY ANALYSIS ON D100 (N={len(self.questions)})")
        print("=======================================================")

        evaluator = Evaluator(llm_service=LLMService())

        h0_recalls, h0_precisions, h0_f1s, h0_cprs = [], [], [], []
        h1_recalls, h1_precisions, h1_f1s, h1_cprs = [], [], [], []
        h0_gold_doc_hits, h1_gold_doc_hits = [], []
        h0_gold_chunk_hits, h1_gold_chunk_hits = [], []
        h0_chain_completions, h1_chain_completions = [], []

        retrieval_rescues = []
        retrieval_regressions = []

        for q_item in self.questions:
            qid = q_item["qid"]
            gold = self.gold_map[qid]
            gold_chunks = set(gold["gold_chunk_ids"])
            gold_docs = set(gold["gold_documents"])

            h0_cids = results_h0[qid]["evidence_chunk_ids"]
            h1_cids = results_h1[qid]["evidence_chunk_ids"]

            h0_dids = set(c.split("#")[0] for c in h0_cids)
            h1_dids = set(c.split("#")[0] for c in h1_cids)

            h0_m = evaluator.evaluate_retrieval(h0_cids, list(gold_chunks))
            h1_m = evaluator.evaluate_retrieval(h1_cids, list(gold_chunks))

            h0_recalls.append(h0_m["recall"])
            h0_precisions.append(h0_m["precision"])
            h0_f1s.append(h0_m["f1"])
            h0_cprs.append(h0_m["cpr"])

            h1_recalls.append(h1_m["recall"])
            h1_precisions.append(h1_m["precision"])
            h1_f1s.append(h1_m["f1"])
            h1_cprs.append(h1_m["cpr"])

            # Doc Recall
            h0_doc_rec = len(h0_dids & gold_docs) / len(gold_docs) if gold_docs else 1.0
            h1_doc_rec = len(h1_dids & gold_docs) / len(gold_docs) if gold_docs else 1.0
            h0_gold_doc_hits.append(h0_doc_rec)
            h1_gold_doc_hits.append(h1_doc_rec)

            # Chunk Recall
            h0_gold_chunk_hits.append(h0_m["recall"])
            h1_gold_chunk_hits.append(h1_m["recall"])

            # Chain Completion
            h0_complete = 1.0 if gold_chunks.issubset(set(h0_cids)) else 0.0
            h1_complete = 1.0 if gold_chunks.issubset(set(h1_cids)) else 0.0
            h0_chain_completions.append(h0_complete)
            h1_chain_completions.append(h1_complete)

            if h0_complete < 1.0 and h1_complete == 1.0:
                retrieval_rescues.append(qid)
            elif h0_complete == 1.0 and h1_complete < 1.0:
                retrieval_regressions.append(qid)

        retrieval_summary = {
            "gold_doc_recall": {
                "h0": float(np.mean(h0_gold_doc_hits)),
                "h1": float(np.mean(h1_gold_doc_hits)),
                "delta": float(np.mean(h1_gold_doc_hits) - np.mean(h0_gold_doc_hits))
            },
            "gold_chunk_recall": {
                "h0": float(np.mean(h0_gold_chunk_hits)),
                "h1": float(np.mean(h1_gold_chunk_hits)),
                "delta": float(np.mean(h1_gold_chunk_hits) - np.mean(h0_gold_chunk_hits))
            },
            "evidence_precision": {
                "h0": float(np.mean(h0_precisions)),
                "h1": float(np.mean(h1_precisions)),
                "delta": float(np.mean(h1_precisions) - np.mean(h0_precisions))
            },
            "evidence_recall": {
                "h0": float(np.mean(h0_recalls)),
                "h1": float(np.mean(h1_recalls)),
                "delta": float(np.mean(h1_recalls) - np.mean(h0_recalls))
            },
            "evidence_f1": {
                "h0": float(np.mean(h0_f1s)),
                "h1": float(np.mean(h1_f1s)),
                "delta": float(np.mean(h1_f1s) - np.mean(h0_f1s))
            },
            "cpr": {
                "h0": float(np.mean(h0_cprs)),
                "h1": float(np.mean(h1_cprs)),
                "delta": float(np.mean(h1_cprs) - np.mean(h0_cprs))
            },
            "chain_completion": {
                "h0": float(np.mean(h0_chain_completions)),
                "h1": float(np.mean(h1_chain_completions)),
                "delta": float(np.mean(h1_chain_completions) - np.mean(h0_chain_completions))
            },
            "retrieval_rescues_count": len(retrieval_rescues),
            "retrieval_regressions_count": len(retrieval_regressions),
            "net_retrieval_rescue": len(retrieval_rescues) - len(retrieval_regressions),
            "retrieval_rescue_qids": retrieval_rescues,
            "retrieval_regression_qids": retrieval_regressions
        }

        print("\n--- Retrieval-Only Analysis Summary ---")
        print(f"Gold Document Recall : H0 = {retrieval_summary['gold_doc_recall']['h0']:.4f}, H1 = {retrieval_summary['gold_doc_recall']['h1']:.4f} (Δ = {retrieval_summary['gold_doc_recall']['delta']:+.4f})")
        print(f"Gold Chunk Recall    : H0 = {retrieval_summary['gold_chunk_recall']['h0']:.4f}, H1 = {retrieval_summary['gold_chunk_recall']['h1']:.4f} (Δ = {retrieval_summary['gold_chunk_recall']['delta']:+.4f})")
        print(f"Chain Completion Rate: H0 = {retrieval_summary['chain_completion']['h0']:.4f}, H1 = {retrieval_summary['chain_completion']['h1']:.4f} (Δ = {retrieval_summary['chain_completion']['delta']:+.4f})")
        print(f"Evidence F1          : H0 = {retrieval_summary['evidence_f1']['h0']:.4f}, H1 = {retrieval_summary['evidence_f1']['h1']:.4f} (Δ = {retrieval_summary['evidence_f1']['delta']:+.4f})")
        print(f"Retrieval Rescues    : {retrieval_summary['retrieval_rescues_count']}")
        print(f"Retrieval Regressions: {retrieval_summary['retrieval_regressions_count']}")
        print(f"Net Retrieval Rescue : {retrieval_summary['net_retrieval_rescue']:+d}")

        with open(REPORTS_DIR / "confirmation_retrieval_analysis.json", "w", encoding="utf-8") as f:
            json.dump(retrieval_summary, f, indent=2, ensure_ascii=False)

        return retrieval_summary

    def compute_majority_and_stability(self, run_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        qids = [q["qid"] for q in self.questions]
        evaluator = Evaluator(llm_service=LLMService())

        h0_runs = {qid: [r["per_question_h0"][qid] for r in run_summaries] for qid in qids}
        h1_runs = {qid: [r["per_question_h1"][qid] for r in run_summaries] for qid in qids}
        h2_runs = {qid: [r["per_question_h2"][qid] for r in run_summaries] for qid in qids}

        h0_majority = {qid: sum(h0_runs[qid]) >= 2 for qid in qids}
        h1_majority = {qid: sum(h1_runs[qid]) >= 2 for qid in qids}
        h2_majority = {qid: sum(h2_runs[qid]) >= 2 for qid in qids}

        h0_maj_acc = float(np.mean(list(h0_majority.values())))
        h1_maj_acc = float(np.mean(list(h1_majority.values())))
        h2_maj_acc = float(np.mean(list(h2_majority.values())))

        h0_maj_list = [h0_majority[qid] for qid in qids]
        h1_maj_list = [h1_majority[qid] for qid in qids]
        h2_maj_list = [h2_majority[qid] for qid in qids]

        maj_mcnemar_h1_h0 = evaluator.mcnemar_exact_test(h0_maj_list, h1_maj_list)
        maj_mcnemar_h2_h1 = evaluator.mcnemar_exact_test(h1_maj_list, h2_maj_list)
        maj_mcnemar_h2_h0 = evaluator.mcnemar_exact_test(h0_maj_list, h2_maj_list)

        def compute_stability(runs):
            c = Counter(sum(runs[qid]) for qid in qids)
            return {
                "unanimous_correct (3/3)": c.get(3, 0),
                "majority_correct (2/3)": c.get(2, 0),
                "minority_correct (1/3)": c.get(1, 0),
                "unanimous_wrong (0/3)": c.get(0, 0),
                "unanimous_rate": (c.get(3, 0) + c.get(0, 0)) / len(qids)
            }

        h0_stab = compute_stability(h0_runs)
        h1_stab = compute_stability(h1_runs)
        h2_stab = compute_stability(h2_runs)

        stable_rescues = [qid for qid in qids if (not h0_majority[qid]) and h1_majority[qid]]
        stable_regressions = [qid for qid in qids if h0_majority[qid] and (not h1_majority[qid])]
        unanimous_rescues = [qid for qid in qids if sum(h0_runs[qid]) == 0 and sum(h1_runs[qid]) == 3]
        unanimous_regressions = [qid for qid in qids if sum(h0_runs[qid]) == 3 and sum(h1_runs[qid]) == 0]

        stratified = {}
        for group_name, q_filter in [
            ("1-hop (Simple)", lambda q: q["hop_count"] == 1),
            ("2-hop (Multi-hop)", lambda q: q["hop_count"] == 2),
            ("3-hop (Multi-hop)", lambda q: q["hop_count"] == 3),
            ("Hard (All Multi-hop: 2-hop + 3-hop)", lambda q: q["hop_count"] >= 2)
        ]:
            subset_qids = [q["qid"] for q in self.questions if q_filter(q)]
            if subset_qids:
                h0_sub = [h0_majority[qid] for qid in subset_qids]
                h1_sub = [h1_majority[qid] for qid in subset_qids]
                h2_sub = [h2_majority[qid] for qid in subset_qids]
                stratified[group_name] = {
                    "count": len(subset_qids),
                    "h0_majority_acc": float(np.mean(h0_sub)),
                    "h1_majority_acc": float(np.mean(h1_sub)),
                    "h2_majority_acc": float(np.mean(h2_sub)),
                    "delta_h1_vs_h0_pp": float(np.mean(h1_sub) - np.mean(h0_sub)) * 100.0,
                    "delta_h2_vs_h0_pp": float(np.mean(h2_sub) - np.mean(h0_sub)) * 100.0,
                    "rescues": sum(1 for b, r in zip(h0_sub, h1_sub) if (not b) and r),
                    "regressions": sum(1 for b, r in zip(h0_sub, h1_sub) if b and (not r))
                }

        majority_summary = {
            "h0_majority_acc": h0_maj_acc,
            "h0_majority_count": sum(h0_maj_list),
            "h1_majority_acc": h1_maj_acc,
            "h1_majority_count": sum(h1_maj_list),
            "h2_majority_acc": h2_maj_acc,
            "h2_majority_count": sum(h2_maj_list),
            "delta_maj_h1_vs_h0_pp": (h1_maj_acc - h0_maj_acc) * 100.0,
            "delta_maj_h2_vs_h1_pp": (h2_maj_acc - h1_maj_acc) * 100.0,
            "delta_maj_h2_vs_h0_pp": (h2_maj_acc - h0_maj_acc) * 100.0,
            "majority_mcnemar_h1_h0": maj_mcnemar_h1_h0,
            "majority_mcnemar_h2_h1": maj_mcnemar_h2_h1,
            "majority_mcnemar_h2_h0": maj_mcnemar_h2_h0,
            "stability_profiles": {
                "H0": h0_stab,
                "H1": h1_stab,
                "H2": h2_stab
            },
            "stable_rescues_count": len(stable_rescues),
            "stable_regressions_count": len(stable_regressions),
            "net_stable_rescue": len(stable_rescues) - len(stable_regressions),
            "unanimous_rescues_count": len(unanimous_rescues),
            "unanimous_regressions_count": len(unanimous_regressions),
            "stable_rescue_qids": stable_rescues,
            "stable_regression_qids": stable_regressions,
            "stratified": stratified,
            "h0_majority": h0_majority,
            "h1_majority": h1_majority,
            "h2_majority": h2_majority
        }

        print("\n=======================================================")
        print("  PHASE 3: MAJORITY & STABILITY SUMMARY (N=200)")
        print("=======================================================")
        print(f"H0 Majority Accuracy : {h0_maj_acc*100:.2f}% ({sum(h0_maj_list)}/200)")
        print(f"H1 Majority Accuracy : {h1_maj_acc*100:.2f}% ({sum(h1_maj_list)}/200) [Δ = {majority_summary['delta_maj_h1_vs_h0_pp']:+.2f}pp]")
        print(f"H2 Majority Accuracy : {h2_maj_acc*100:.2f}% ({sum(h2_maj_list)}/200) [Δ = {majority_summary['delta_maj_h2_vs_h0_pp']:+.2f}pp]")
        print(f"Stable Rescues       : {len(stable_rescues)}")
        print(f"Stable Regressions   : {len(stable_regressions)}")
        print(f"Net Stable Rescue    : {majority_summary['net_stable_rescue']:+d}")
        print(f"Unanimous Rescues    : {len(unanimous_rescues)}")
        print(f"Unanimous Regress    : {len(unanimous_regressions)}")
        print(f"Majority McNemar p   : {maj_mcnemar_h1_h0['p_value']:.4f}")

        print("\n--- Stratification Breakdown ---")
        for g_name, g_data in stratified.items():
            print(f"  {g_name} (N={g_data['count']}): H0={g_data['h0_majority_acc']*100:.2f}%, H1={g_data['h1_majority_acc']*100:.2f}% (Δ = {g_data['delta_h1_vs_h0_pp']:+.2f}pp, Rescues={g_data['rescues']}, Regress={g_data['regressions']})")

        with open(REPORTS_DIR / "confirmation_majority_results.json", "w", encoding="utf-8") as f:
            json.dump(majority_summary, f, indent=2, ensure_ascii=False)

        return majority_summary

    def generate_blind_adjudication_pack(
        self,
        majority_results: Dict[str, Any],
        run_summaries: List[Dict[str, Any]]
    ):
        """
        Phase 4: Builds anonymized adjudication pack for discordant cases.
        """
        h0_maj = majority_results["h0_majority"]
        h1_maj = majority_results["h1_majority"]

        discordant_qids = [q["qid"] for q in self.questions if h0_maj[q["qid"]] != h1_maj[q["qid"]]]
        print(f"\nPhase 4: Identified {len(discordant_qids)} discordant majority questions between H0 and H1.")

        blind_pack = []
        mapping = {}

        run1_dir = RUNS_CONFIRMATION_DIR / "run1"

        for qid in discordant_qids:
            gold = self.gold_map[qid]
            question = gold["question"]
            gold_ans = gold["gold_answer"]
            gold_spans = gold["gold_spans"]

            with open(run1_dir / "H0" / f"{qid}.json", "r", encoding="utf-8") as f:
                ans_h0 = json.load(f)["generated_answer"]
            with open(run1_dir / "H1" / f"{qid}.json", "r", encoding="utf-8") as f:
                ans_h1 = json.load(f)["generated_answer"]

            systems = ["H0", "H1"]
            random.shuffle(systems)
            label_map = {
                "System_A": systems[0],
                "System_B": systems[1]
            }
            mapping[qid] = label_map

            ans_map = {"H0": ans_h0, "H1": ans_h1}
            blind_pack.append({
                "qid": qid,
                "question": question,
                "gold_answer": gold_ans,
                "gold_spans": gold_spans,
                "hop_count": gold["hop_count"],
                "tags": gold["tags"],
                "system_A_answer": ans_map[systems[0]],
                "system_B_answer": ans_map[systems[1]],
                "h0_majority_correct": h0_maj[qid],
                "h1_majority_correct": h1_maj[qid]
            })

        blind_decisions = []
        evaluator = Evaluator(llm_service=LLMService())
        print(f"Executing blind adjudication evaluation on {len(blind_pack)} cases...")
        for item in blind_pack:
            qid = item["qid"]
            q = item["question"]
            ga = item["gold_answer"]
            spans = item["gold_spans"]

            j_a = evaluator.judge_answer(q, ga, spans, item["system_A_answer"])
            j_b = evaluator.judge_answer(q, ga, spans, item["system_B_answer"])

            blind_decisions.append({
                "qid": qid,
                "system_A_correct": j_a["is_correct"],
                "system_A_reasoning": j_a["reasoning"],
                "system_B_correct": j_b["is_correct"],
                "system_B_reasoning": j_b["reasoning"],
                "true_system_A": mapping[qid]["System_A"],
                "true_system_B": mapping[qid]["System_B"]
            })

        with open(REPORTS_DIR / "confirmation_blind_review_pack.json", "w", encoding="utf-8") as f:
            json.dump(blind_pack, f, indent=2, ensure_ascii=False)
        with open(REPORTS_DIR / "confirmation_blind_review_mapping.json", "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        with open(REPORTS_DIR / "confirmation_blind_review_decisions.json", "w", encoding="utf-8") as f:
            json.dump(blind_decisions, f, indent=2, ensure_ascii=False)

        print(f"Wrote blind adjudication pack and decisions to {REPORTS_DIR}.")

    def run_all(self):
        # Step 1: Run 1 (seed=101)
        run1_summary, r1_h0, r1_h1, r1_h2 = self.run_single_fresh_run(run_num=1, seed=101)

        # Step 2: Retrieval-Only Analysis from deterministic evidence of Run 1
        self.run_retrieval_analysis(r1_h0, r1_h1)

        # Step 3: Run 2 (seed=202)
        run2_summary, _, _, _ = self.run_single_fresh_run(run_num=2, seed=202)

        # Step 4: Run 3 (seed=303)
        run3_summary, _, _, _ = self.run_single_fresh_run(run_num=3, seed=303)

        run_summaries = [run1_summary, run2_summary, run3_summary]

        # Step 5: Majority & Stability Analysis
        majority_results = self.compute_majority_and_stability(run_summaries)

        # Step 6: Blind Adjudication Pack
        self.generate_blind_adjudication_pack(majority_results, run_summaries)

        print("\nAll Confirmation Experiment Phases completed successfully!")


if __name__ == "__main__":
    runner = ConfirmationExperimentRunner(max_workers=10)
    runner.run_all()
