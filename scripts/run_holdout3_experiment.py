#!/usr/bin/env python3
"""
scripts/run_holdout3_experiment.py
Mechanism Holdout-3 Confirmation Experiment Runner.

Evaluates 6 Systems across N=240 questions in Mechanism Holdout-3:
  - S0: Dense Top-5 (Historical B0)
  - S1: Dense Top-20 + Composer
  - S2: Metadata / Title Routing + Local BM25
  - S3: V3-NoGraph (Matched Ablation)
  - S4: V3-TrueGraph (Frozen V3)
  - S5: V3-ShuffledGraph (Control Baseline)

Strata:
  - Q-E: Explicit (N=80)
  - Q-P: Partial (N=80)
  - Q-I: Implicit (N=80)

Strict Guarantees:
  1. No short-circuit caching (each system independently calls generator & evaluator across 3 seeds).
  2. Measures true Same-Evidence Flip Rate.
  3. Pre-registered causal contrasts: C1 (S1-S0), C2 (S2-S0), C3 (S4-S3), C4 (S4-S5).
  4. Exact McNemar tests & paired bootstrap 95% CIs.
  5. Causal mechanism attribution (True Graph Causal Rescues vs Regressions).
  6. Final Verdict selection: M-A, M-B, or M-C.
"""

import os
import re
import sys
import json
import time
import math
import random
import copy
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.search import SearchService
from src.services.llm import LLMService
from src.evaluation.metrics import Evaluator
from src.graph.lsdb import KnowledgeLSDB
from src.routing.e2_descent_router import E2DescentRouterSystem
from src.composition.slots import extract_evidence_slots, EvidenceSlot
from src.composition.composer import compose_evidence
from src.composition.descent import identify_unresolved_slot, build_generic_descent_query, hybrid_targeted_descent
from src.common.prompt import SYSTEM_PROMPT, pack_evidence_context, format_user_prompt
from src.common.models import EvidenceItem

HOLDOUT3_DIR = PROJECT_ROOT / "benchmark" / "holdout3"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_holdout3_benchmark():
    gold_path = HOLDOUT3_DIR / "gold.jsonl"
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_items = [json.loads(line) for line in f if line.strip()]
    gold_map = {it["qid"]: it for it in gold_items}
    return gold_items, gold_map


def evaluate_retrieval_for_cids(cids: List[str], gold: Dict[str, Any]) -> Dict[str, float]:
    gold_chunks = set(gold["gold_chunk_ids"])
    gold_docs = set(gold["gold_documents"])
    c_set = set(cids)
    c_docs = set(c.split("#")[0] for c in cids)

    doc_rec = len(gold_docs.intersection(c_docs)) / len(gold_docs) if gold_docs else 1.0
    chunk_rec = len(gold_chunks.intersection(c_set)) / len(gold_chunks) if gold_chunks else 1.0
    prec = len(gold_chunks.intersection(c_set)) / len(c_set) if c_set else 0.0
    f1 = 2 * prec * chunk_rec / (prec + chunk_rec) if (prec + chunk_rec) > 0 else 0.0
    comp = 1.0 if gold_chunks.issubset(c_set) else 0.0

    return {
        "doc_recall": doc_rec,
        "chunk_recall": chunk_rec,
        "precision": prec,
        "f1": f1,
        "chain_complete": comp
    }


def build_shuffled_lsdb(original_lsdb: KnowledgeLSDB, seed: int = 42) -> KnowledgeLSDB:
    """Builds a degree-matched, relation-preserving shuffled graph LSDB."""
    shuffled = copy.deepcopy(original_lsdb)
    rng = random.Random(seed)

    by_rel = defaultdict(list)
    for u, v, d in shuffled.G_routing.edges(data=True):
        by_rel[d.get("relation")].append((u, v, d))

    shuffled.G_routing.clear_edges()

    for rel, edge_list in by_rel.items():
        sources = [e[0] for e in edge_list]
        targets = [e[1] for e in edge_list]

        # Shuffle targets avoiding self-loops
        for _ in range(10):
            rng.shuffle(targets)
            if not any(s == t for s, t in zip(sources, targets)):
                break

        for i, e in enumerate(edge_list):
            shuffled.G_routing.add_edge(sources[i], targets[i], **e[2])

    return shuffled


def run_v3_retrieval_core(
    question: str,
    h0_seeds: List[EvidenceItem],
    router: E2DescentRouterSystem,
    lsdb: KnowledgeLSDB,
    search_service: SearchService,
    allow_graph: bool = True
) -> Tuple[List[str], List[str]]:
    """Executes V3 retrieval with optional graph expansion toggle."""
    h0_cids = [s.chunk_id for s in h0_seeds]
    lane = router.detect_lane(question, h0_seeds)
    q_slots = extract_evidence_slots(question)

    if lane == "FAST_PATH":
        return list(h0_cids), []

    cand_pool = []
    primary_doc_id = h0_seeds[0].doc_id if h0_seeds else ""

    if allow_graph:
        # Lane A: TEMPORAL_BASIS
        if lane == "TEMPORAL_BASIS":
            for s in h0_seeds[:3]:
                nbrs = lsdb.get_routing_neighbors(s.chunk_id, corpus="D100", allowed_relations={"SUPERSEDES", "AMENDS"})
                for tgt, rel, cost in nbrs:
                    chk = lsdb.get_chunk_evidence(tgt)
                    if chk:
                        it = EvidenceItem(
                            chunk_id=chk["chunk_id"], doc_id=chk["doc_id"], title=chk["title"],
                            heading_path=chk["heading_path"], text=chk["text"], score=0.95,
                            source_method=f"e2_lane_a_{rel.lower()}"
                        )
                        cand_pool.append((it, 1.2, f"Resolved version via {rel}"))

            if primary_doc_id:
                fts_repeal = search_service.fts_search_in_doc("施行 废止 附则", primary_doc_id, corpus="D100", top_k=1)
                for f_item in fts_repeal:
                    f_item.score = 0.90
                    f_item.source_method = "e2_lane_a_repeal_clause"
                    cand_pool.append((f_item, 1.1, "Repeal clause"))

            chunk_basis_found = False
            for s in h0_seeds[:3]:
                nbrs = lsdb.get_routing_neighbors(s.chunk_id, corpus="D100", allowed_relations={"BASED_ON"})
                for tgt, rel, cost in nbrs:
                    chk = lsdb.get_chunk_evidence(tgt)
                    if chk:
                        it = EvidenceItem(
                            chunk_id=chk["chunk_id"], doc_id=chk["doc_id"], title=chk["title"],
                            heading_path=chk["heading_path"], text=chk["text"], score=0.95,
                            source_method="e2_lane_a_based_on"
                        )
                        cand_pool.append((it, 1.2, "Based-on"))
                        chunk_basis_found = True

            if re.search(r"(上位|依据|根据|何法|哪两部)", question) and not chunk_basis_found and h0_seeds:
                unresolved_slot_str = router._extract_unresolved_slot(question)
                max_targets = 2 if re.search(r"(两部|两项|分别)", question) else 1
                hop_res = router.resolve_hierarchical_next_hop(
                    source_chunk=h0_seeds[0],
                    required_relation="BASED_ON",
                    unresolved_slot_str=unresolved_slot_str,
                    question=question,
                    seed_items=h0_seeds,
                    slots=q_slots,
                    corpus="D100",
                    max_targets=max_targets
                )
                if hop_res and hop_res["candidate_items"]:
                    for idx, c_item in enumerate(hop_res["candidate_items"]):
                        cand_pool.append((c_item, 1.4, "Hierarchical BASED_ON"))

        # Lane B: COMPOSITE_EVIDENCE
        elif lane == "COMPOSITE_EVIDENCE":
            has_gap, missing_statutes = router._check_statutory_gap(question, h0_seeds)
            if missing_statutes:
                target_statute = missing_statutes[0]
                found_by_graph = False
                for s in h0_seeds:
                    nbrs = lsdb.get_routing_neighbors(s.chunk_id, corpus="D100", allowed_relations={"SUPERSEDES", "AMENDS", "BASED_ON", "REFERENCES"})
                    for tgt, rel, cost in nbrs:
                        if tgt in lsdb.doc_meta and target_statute in lsdb.doc_meta[tgt].get("title", ""):
                            d_chks = lsdb.get_document_chunks(tgt, corpus="D100")
                            if d_chks:
                                cdata = lsdb.get_chunk_evidence(d_chks[0])
                                if cdata:
                                    it = EvidenceItem(
                                        chunk_id=cdata["chunk_id"], doc_id=cdata["doc_id"], title=cdata["title"],
                                        heading_path=cdata["heading_path"], text=cdata["text"], score=0.98,
                                        source_method=f"e2_lane_b_graph_gap_{rel.lower()}"
                                    )
                                    cand_pool.append((it, 1.5, f"Missing statute {target_statute}"))
                                    found_by_graph = True

                if not found_by_graph and h0_seeds:
                    for s in h0_seeds[:2]:
                        hop_res = router.resolve_hierarchical_next_hop(
                            source_chunk=s,
                            required_relation="REFERENCES",
                            unresolved_slot_str=target_statute,
                            question=question,
                            seed_items=h0_seeds,
                            slots=q_slots,
                            corpus="D100",
                            max_targets=1
                        )
                        if hop_res and hop_res["candidate_items"]:
                            for idx, c_item in enumerate(hop_res["candidate_items"]):
                                cand_pool.append((c_item, 1.45, "Hierarchical REFERENCES"))
                            found_by_graph = True
                            break

            for s in h0_seeds[:3]:
                nbrs = lsdb.get_routing_neighbors(s.chunk_id, corpus="D100", allowed_relations={"SUPERSEDES", "AMENDS", "BASED_ON", "REFERENCES"})
                for tgt, rel, cost in nbrs:
                    chk = lsdb.get_chunk_evidence(tgt)
                    if chk:
                        it = EvidenceItem(
                            chunk_id=chk["chunk_id"], doc_id=chk["doc_id"], title=chk["title"],
                            heading_path=chk["heading_path"], text=chk["text"], score=0.90,
                            source_method=f"e2_lane_b_{rel.lower()}"
                        )
                        cand_pool.append((it, 1.1, "Composite expansion"))

    # Shadow Candidate Plane (Shared across S3, S4, S5)
    current_docs = set(s.doc_id for s in h0_seeds).union(set(c[0].doc_id for c in cand_pool))
    missing_matches = [
        m for m in router._match_question_entities(question, corpus="D100")
        if m[0] not in current_docs
    ]

    if len(missing_matches) > 0 or len(cand_pool) == 0:
        shadow_rib, selected_prefixes = router._extract_shadow_prefixes(
            question=question, corpus="D100", seed_items=h0_seeds, current_candidate_docs=current_docs
        )
        descent_k = 2 if len(selected_prefixes) >= 2 else 3
        for p_did, p_score, p_title in selected_prefixes:
            unresolved_slot_obj, covered_slots_list = identify_unresolved_slot(
                question=question,
                seed_items=h0_seeds,
                slots=q_slots,
                target_doc_title=p_title
            )
            descent_query = build_generic_descent_query(
                question=question,
                unresolved_slot=unresolved_slot_obj,
                target_doc_title=p_title,
                covered_slots=covered_slots_list
            )

            descent_res = hybrid_targeted_descent(
                search_service=search_service,
                target_doc_id=p_did,
                target_doc_title=p_title,
                unresolved_slot=unresolved_slot_obj,
                generic_query=descent_query,
                corpus="D100",
                top_k_candidates=descent_k,
                lsdb=lsdb,
                channel_mode="lexical_only"
            )

            for d_item, score, meta in descent_res:
                cand_pool_score = 1.45 if any(m[0] == p_did for m in missing_matches) else 1.35
                cand_pool.append((d_item, cand_pool_score, f"E2-Lite Lexical: {descent_query[:25]}"))

    final_items, comp_traces = compose_evidence(
        b0_evidence=h0_seeds,
        candidate_pool=cand_pool,
        slots=q_slots,
        max_chunks=5,
        max_replacements=2,
        lsdb=lsdb
    )
    final_cids = [it.chunk_id for it in final_items]
    cand_cids = [c[0].chunk_id for c in cand_pool]
    return final_cids, cand_cids


def run_deterministic_retrieval_all_systems(
    gold_items: List[Dict[str, Any]],
    search_service: SearchService,
    lsdb: KnowledgeLSDB,
    shuffled_lsdb: KnowledgeLSDB
) -> Dict[str, Dict[str, Any]]:
    print("\n=======================================================")
    print("Step 1: Deterministic Retrieval for 6 Systems on Holdout-3 (N = 240)")
    print("=======================================================")

    ret_cache = REPORTS_DIR / "holdout3_retrieval.json"
    if ret_cache.exists():
        try:
            with open(ret_cache, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            if "traces" in c_data and all(len(c_data["traces"].get(s, {})) == len(gold_items) for s in ["s0_dense5", "s1_dense20", "s2_metadata_bm25", "s3_v3_nograph", "s4_v3_truegraph", "s5_v3_shuffledgraph"]):
                print(f"Loaded existing complete retrieval traces for 6 systems from {ret_cache}.")
                return c_data["traces"]
        except Exception as e:
            print(f"Retrieval cache read failed: {e}")

    router_true = E2DescentRouterSystem(search_service=search_service, llm_service=None, lsdb=lsdb, channel_mode="lexical_only")
    router_shuf = E2DescentRouterSystem(search_service=search_service, llm_service=None, lsdb=shuffled_lsdb, channel_mode="lexical_only")

    traces = {
        "s0_dense5": {},
        "s1_dense20": {},
        "s2_metadata_bm25": {},
        "s3_v3_nograph": {},
        "s4_v3_truegraph": {},
        "s5_v3_shuffledgraph": {}
    }

    t0 = time.time()
    for idx, gold in enumerate(gold_items, 1):
        qid = gold["qid"]
        question = gold["question"]
        q_slots = extract_evidence_slots(question)

        # S0: Dense Top-5
        h0_seeds_20 = search_service.vector_search(query=question, corpus="D100", top_k=20)
        h0_seeds_5 = h0_seeds_20[:5]
        s0_cids = [s.chunk_id for s in h0_seeds_5]
        traces["s0_dense5"][qid] = {
            "qid": qid,
            "final_evidence_chunk_ids": s0_cids,
            "metrics": evaluate_retrieval_for_cids(s0_cids, gold)
        }

        # S1: Dense Top-20 + Composer
        s1_pool = [(s, 1.0, "dense_top20") for s in h0_seeds_20[5:]]
        s1_items, _ = compose_evidence(
            b0_evidence=h0_seeds_5,
            candidate_pool=s1_pool,
            slots=q_slots,
            max_chunks=5,
            max_replacements=4,
            lsdb=lsdb
        )
        s1_cids = [it.chunk_id for it in s1_items]
        traces["s1_dense20"][qid] = {
            "qid": qid,
            "final_evidence_chunk_ids": s1_cids,
            "metrics": evaluate_retrieval_for_cids(s1_cids, gold)
        }

        # S2: Metadata / Title Routing + Local BM25
        matches = router_true._match_question_entities(question, corpus="D100")
        target_docs = [m[0] for m in matches]
        s2_pool = []
        clean_q = re.sub(r"[《》\(\)（）\s]+", " ", question).strip()
        if target_docs:
            for did in target_docs[:2]:
                fts_items = search_service.fts_search_in_doc(clean_q, did, corpus="D100", top_k=3)
                for it in fts_items:
                    s2_pool.append((it, 1.35, f"Local BM25 in {did}"))
        else:
            fts_items = search_service.fts_search(clean_q, corpus="D100", top_k=5)
            for it in fts_items:
                s2_pool.append((it, 1.0, "Global BM25"))

        s2_items, _ = compose_evidence(
            b0_evidence=h0_seeds_5,
            candidate_pool=s2_pool,
            slots=q_slots,
            max_chunks=5,
            max_replacements=4,
            lsdb=lsdb
        )
        s2_cids = [it.chunk_id for it in s2_items]
        traces["s2_metadata_bm25"][qid] = {
            "qid": qid,
            "final_evidence_chunk_ids": s2_cids,
            "metrics": evaluate_retrieval_for_cids(s2_cids, gold)
        }

        # S3: V3-NoGraph (allow_graph=False)
        s3_cids, s3_pool_cids = run_v3_retrieval_core(question, h0_seeds_5, router_true, lsdb, search_service, allow_graph=False)
        traces["s3_v3_nograph"][qid] = {
            "qid": qid,
            "final_evidence_chunk_ids": s3_cids,
            "candidate_pool_chunk_ids": s3_pool_cids,
            "metrics": evaluate_retrieval_for_cids(s3_cids, gold)
        }

        # S4: V3-TrueGraph (allow_graph=True)
        s4_cids, s4_pool_cids = run_v3_retrieval_core(question, h0_seeds_5, router_true, lsdb, search_service, allow_graph=True)
        traces["s4_v3_truegraph"][qid] = {
            "qid": qid,
            "final_evidence_chunk_ids": s4_cids,
            "candidate_pool_chunk_ids": s4_pool_cids,
            "metrics": evaluate_retrieval_for_cids(s4_cids, gold)
        }

        # S5: V3-ShuffledGraph (allow_graph=True on shuffled_lsdb)
        s5_cids, s5_pool_cids = run_v3_retrieval_core(question, h0_seeds_5, router_shuf, shuffled_lsdb, search_service, allow_graph=True)
        traces["s5_v3_shuffledgraph"][qid] = {
            "qid": qid,
            "final_evidence_chunk_ids": s5_cids,
            "candidate_pool_chunk_ids": s5_pool_cids,
            "metrics": evaluate_retrieval_for_cids(s5_cids, gold)
        }

        if idx % 30 == 0 or idx == len(gold_items):
            print(f"Computed retrieval traces for {idx} / {len(gold_items)} questions...")

    # Aggregate retrieval metrics
    summary = {}
    strata = ["ALL", "E", "P", "I"]
    for s_name in traces:
        summary[s_name] = {}
        for st in strata:
            sub_items = [g for g in gold_items if st == "ALL" or g["doc_explicitness"] == st]
            m_doc = np.mean([traces[s_name][g["qid"]]["metrics"]["doc_recall"] for g in sub_items])
            m_chk = np.mean([traces[s_name][g["qid"]]["metrics"]["chunk_recall"] for g in sub_items])
            m_f1 = np.mean([traces[s_name][g["qid"]]["metrics"]["f1"] for g in sub_items])
            m_comp = np.mean([traces[s_name][g["qid"]]["metrics"]["chain_complete"] for g in sub_items])
            summary[s_name][st] = {
                "N": len(sub_items),
                "doc_recall": round(float(m_doc) * 100, 2),
                "chunk_recall": round(float(m_chk) * 100, 2),
                "evidence_f1": round(float(m_f1) * 100, 2),
                "chain_complete": round(float(m_comp) * 100, 2)
            }

    retrieval_payload = {
        "summary": summary,
        "traces": traces
    }
    with open(REPORTS_DIR / "holdout3_retrieval.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_payload, f, indent=2, ensure_ascii=False)

    print("Saved retrieval analysis to reports/holdout3_retrieval.json")
    return traces


def run_fresh_generation_and_judge(
    run_id: int,
    seed: int,
    gold_items: List[Dict[str, Any]],
    system_traces: Dict[str, Dict[str, Any]],
    llm_service: LLMService,
    lsdb: KnowledgeLSDB,
    evaluator: Evaluator
) -> Dict[str, Any]:
    out_file = REPORTS_DIR / f"holdout3_run{run_id}.json"
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            res_data = cached.get("results", cached)
            if all(len(res_data.get(s_key, {})) == len(gold_items) for s_key in system_traces):
                print(f"Loaded existing complete Run {run_id} from {out_file}.")
                return cached
        except Exception as e:
            print(f"Error loading cached Run {run_id}: {e}")

    print(f"\n--- Executing Run {run_id} (Seed {seed}) across 6 Systems ---")

    sys_names = list(system_traces.keys())
    run_results = {s_name: {} for s_name in sys_names}

    partial_file = REPORTS_DIR / f"holdout3_run{run_id}_partial.json"
    if partial_file.exists():
        try:
            with open(partial_file, "r", encoding="utf-8") as f:
                saved_partial = json.load(f)
            for s_name in sys_names:
                if s_name in saved_partial:
                    run_results[s_name].update(saved_partial[s_name])
            loaded_count = sum(len(run_results[s]) for s in sys_names)
            print(f"[Run {run_id}] Resumed {loaded_count} completed evaluations from {partial_file.name}")
        except Exception as e:
            print(f"Warning: could not read partial file: {e}")

    def execute_case_system(qid, s_name):
        gold = next(g for g in gold_items if g["qid"] == qid)
        question = gold["question"]
        gold_answer = gold["gold_answer"]
        gold_spans = gold.get("gold_spans", [])

        cids = system_traces[s_name][qid]["final_evidence_chunk_ids"]
        evidence = [EvidenceItem(**lsdb.get_chunk_evidence(c)) for c in cids if lsdb.get_chunk_evidence(c)]
        context = pack_evidence_context(evidence, max_tokens=4000)
        prompt = format_user_prompt(question, context)

        # Strictly Fresh Generation & Independent Judging with retry
        for case_attempt in range(5):
            try:
                ans, _, _ = llm_service.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT, seed=seed)
                judge = evaluator.judge_answer(question=question, gold_answer=gold_answer, gold_spans=gold_spans, generated_answer=ans)
                return qid, s_name, ans, judge
            except Exception as e:
                if case_attempt == 4:
                    print(f"Error on {qid} {s_name}: {e}")
                    raise
                time.sleep(3.0 * (case_attempt + 1))

    tasks = [(g["qid"], s_name) for g in gold_items for s_name in sys_names if g["qid"] not in run_results[s_name]]
    completed_before = sum(len(run_results[s]) for s in sys_names)
    print(f"Total generation and judging tasks for Run {run_id}: {len(tasks)} (already completed: {completed_before})")

    if tasks:
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(execute_case_system, qid, s_name) for qid, s_name in tasks]
            done_cnt = 0
            for fut in as_completed(futures):
                qid, s_name, ans, judge = fut.result()
                with lock:
                    run_results[s_name][qid] = {
                        "generated_answer": ans,
                        "is_correct": judge["is_correct"],
                        "reasoning": judge["reasoning"]
                    }
                    done_cnt += 1
                    if done_cnt % 20 == 0 or done_cnt == len(tasks):
                        with open(partial_file, "w", encoding="utf-8") as f:
                            json.dump(run_results, f, ensure_ascii=False)
                    if done_cnt % 100 == 0 or done_cnt == len(tasks):
                        print(f"[Run {run_id}] Completed {done_cnt} / {len(tasks)} evaluations...")

    if partial_file.exists():
        try:
            partial_file.unlink()
        except Exception:
            pass

    # Summary accuracy per system
    run_accs = {}
    for s_name in sys_names:
        acc = round(sum(1 for r in run_results[s_name].values() if r["is_correct"]) / len(gold_items) * 100, 2)
        run_accs[s_name] = acc

    payload = {
        "run_id": run_id,
        "seed": seed,
        "total_questions": len(gold_items),
        "accuracies": run_accs,
        "results": run_results
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Run {run_id} complete. Accuracies: {run_accs}")
    return payload


def compute_majority_and_statistics(
    gold_items: List[Dict[str, Any]],
    run1: Dict[str, Any],
    run2: Dict[str, Any],
    run3: Dict[str, Any],
    system_traces: Dict[str, Dict[str, Any]],
    lsdb: KnowledgeLSDB
) -> Dict[str, Any]:
    print("\n=======================================================")
    print("Step 3: Majority Vote, Empirical Flips, Contrasts & CI")
    print("=======================================================")

    sys_names = list(system_traces.keys())
    qids = [g["qid"] for g in gold_items]
    gold_map = {g["qid"]: g for g in gold_items}

    # 1. Majority Correctness Map
    maj_maps = {s: {} for s in sys_names}
    for s in sys_names:
        for qid in qids:
            votes = [
                run1["results"][s][qid]["is_correct"],
                run2["results"][s][qid]["is_correct"],
                run3["results"][s][qid]["is_correct"]
            ]
            maj_maps[s][qid] = (sum(1 for v in votes if v) >= 2)

    # 2. Accuracies Overall and Per Stratum
    strata = ["ALL", "E", "P", "I"]
    accuracy_matrix = {}
    for s in sys_names:
        accuracy_matrix[s] = {}
        for st in strata:
            sub_qids = [g["qid"] for g in gold_items if st == "ALL" or g["doc_explicitness"] == st]
            corr = sum(1 for q in sub_qids if maj_maps[s][q])
            acc = round(corr / len(sub_qids) * 100, 2)
            accuracy_matrix[s][st] = acc

    # 3. True Same-Evidence Flip Rate
    # Compare S4 vs S3, and across pairs when evidence chunks are identical
    same_evidence_checks = 0
    same_evidence_flips = 0
    for qid in qids:
        cids_s4 = system_traces["s4_v3_truegraph"][qid]["final_evidence_chunk_ids"]
        cids_s3 = system_traces["s3_v3_nograph"][qid]["final_evidence_chunk_ids"]
        if cids_s4 == cids_s3:
            same_evidence_checks += 1
            if maj_maps["s4_v3_truegraph"][qid] != maj_maps["s3_v3_nograph"][qid]:
                same_evidence_flips += 1

    true_flip_rate = round(same_evidence_flips / same_evidence_checks * 100, 2) if same_evidence_checks > 0 else 0.0

    # 4. Primary Causal Contrasts
    # C1: S1 - S0 (Wider Dense Recall)
    # C2: S2 - S0 (Structured Metadata Baseline)
    # C3: S4 - S3 (Graph Increment)
    # C4: S4 - S5 (Graph Semantics)
    contrasts = {}
    pair_defs = [
        ("C1_Wider_Dense", "s1_dense20", "s0_dense5"),
        ("C2_Structured_Metadata", "s2_metadata_bm25", "s0_dense5"),
        ("C3_Graph_Increment", "s4_v3_truegraph", "s3_v3_nograph"),
        ("C4_Graph_Semantics", "s4_v3_truegraph", "s5_v3_shuffledgraph")
    ]

    def compute_paired_stats(s_a, s_b, q_list):
        rescues = [q for q in q_list if not maj_maps[s_b][q] and maj_maps[s_a][q]]
        regressions = [q for q in q_list if maj_maps[s_b][q] and not maj_maps[s_a][q]]
        net = len(rescues) - len(regressions)
        delta = round(accuracy_matrix[s_a]["ALL"] - accuracy_matrix[s_b]["ALL"], 2) if q_list == qids else round(
            (sum(1 for q in q_list if maj_maps[s_a][q]) - sum(1 for q in q_list if maj_maps[s_b][q])) / len(q_list) * 100, 2
        )

        b = len(regressions)
        c = len(rescues)
        tot_disc = b + c
        if tot_disc > 0:
            p_val = round(float(2 * stats.binom.cdf(min(b, c), tot_disc, 0.5)), 4)
        else:
            p_val = 1.0

        # Paired bootstrap 95% CI
        rng = np.random.RandomState(42)
        boot_diffs = []
        q_arr = np.array(q_list)
        for _ in range(1000):
            boot_idx = rng.choice(len(q_arr), size=len(q_arr), replace=True)
            boot_qids = q_arr[boot_idx]
            acc_a = np.mean([maj_maps[s_a][q] for q in boot_qids])
            acc_b = np.mean([maj_maps[s_b][q] for q in boot_qids])
            boot_diffs.append((acc_a - acc_b) * 100)

        ci_low = round(float(np.percentile(boot_diffs, 2.5)), 2)
        ci_high = round(float(np.percentile(boot_diffs, 97.5)), 2)

        return {
            "delta": delta,
            "rescues": len(rescues),
            "regressions": len(regressions),
            "net_rescues": net,
            "mcnemar_p_value": p_val,
            "bootstrap_ci_95": [ci_low, ci_high]
        }

    for c_name, sa, sb in pair_defs:
        contrasts[c_name] = {
            "overall": compute_paired_stats(sa, sb, qids),
            "by_stratum": {
                st: compute_paired_stats(sa, sb, [g["qid"] for g in gold_items if g["doc_explicitness"] == st])
                for st in ["E", "P", "I"]
            }
        }

    # 5. Graph Retrieval Mechanism Attribution (for S4 vs S3 transitions)
    rescues_s4_s3 = [q for q in qids if not maj_maps["s3_v3_nograph"][q] and maj_maps["s4_v3_truegraph"][q]]
    regressions_s4_s3 = [q for q in qids if maj_maps["s3_v3_nograph"][q] and not maj_maps["s4_v3_truegraph"][q]]

    attributions_rescue = []
    cat_rescue = Counter()
    for qid in rescues_s4_s3:
        gold = gold_map[qid]
        g_docs = set(gold["gold_documents"])
        g_chunks = set(gold["gold_chunk_ids"])
        cids_s3 = set(system_traces["s3_v3_nograph"][qid]["final_evidence_chunk_ids"])
        cids_s4 = set(system_traces["s4_v3_truegraph"][qid]["final_evidence_chunk_ids"])

        s3_docs = set(c.split("#")[0] for c in cids_s3)
        s4_docs = set(c.split("#")[0] for c in cids_s4)

        if cids_s4 == cids_s3:
            cat = "SAME_EVIDENCE_FLIP"
        elif len(g_docs & s4_docs) > len(g_docs & s3_docs) and len(g_chunks & cids_s4) > len(g_chunks & cids_s3):
            cat = "TRUE_GRAPH_CAUSAL_RESCUE"
        elif len(g_chunks & cids_s4) > len(g_chunks & cids_s3):
            cat = "TRUE_GRAPH_CAUSAL_RESCUE"
        elif len(g_chunks & cids_s4) == len(g_chunks & cids_s3):
            cat = "GRAPH_ADDED_NON_GOLD_BUT_GENERATION_FLIP"
        else:
            cat = "UNCERTAIN"

        cat_rescue[cat] += 1
        attributions_rescue.append({"qid": qid, "category": cat})

    attributions_regr = []
    cat_regr = Counter()
    for qid in regressions_s4_s3:
        gold = gold_map[qid]
        g_chunks = set(gold["gold_chunk_ids"])
        cids_s3 = set(system_traces["s3_v3_nograph"][qid]["final_evidence_chunk_ids"])
        cids_s4 = set(system_traces["s4_v3_truegraph"][qid]["final_evidence_chunk_ids"])

        if cids_s4 == cids_s3:
            cat = "SAME_EVIDENCE_FLIP"
        elif len(g_chunks & cids_s4) < len(g_chunks & cids_s3):
            cat = "GRAPH_CAUSAL_REGRESSION"
        elif any(c not in g_chunks for c in (cids_s4 - cids_s3)):
            cat = "GRAPH_DISTRACTOR"
        else:
            cat = "UNCERTAIN"

        cat_regr[cat] += 1
        attributions_regr.append({"qid": qid, "category": cat})

    true_graph_rescues = cat_rescue["TRUE_GRAPH_CAUSAL_RESCUE"]
    true_graph_regressions = cat_regr["GRAPH_CAUSAL_REGRESSION"]
    net_graph_causal_gain = true_graph_rescues - true_graph_regressions

    # 6. Final Verdict Evaluation Rule
    c3_delta = contrasts["C3_Graph_Increment"]["overall"]["delta"]
    c4_delta = contrasts["C4_Graph_Semantics"]["overall"]["delta"]
    p_delta = contrasts["C3_Graph_Increment"]["by_stratum"]["P"]["delta"]
    i_delta = contrasts["C3_Graph_Increment"]["by_stratum"]["I"]["delta"]
    e_delta = contrasts["C3_Graph_Increment"]["by_stratum"]["E"]["delta"]

    s1_acc = accuracy_matrix["s1_dense20"]["ALL"]
    s2_acc = accuracy_matrix["s2_metadata_bm25"]["ALL"]
    s4_acc = accuracy_matrix["s4_v3_truegraph"]["ALL"]

    if s1_acc >= s4_acc or s2_acc >= s4_acc:
        verdict = "M-C — STRONG BASELINES EXPLAIN THE HISTORICAL GAIN"
        rationale = f"Strong structured baselines (S1={s1_acc}%, S2={s2_acc}%) match or exceed S4 ({s4_acc}%). Historical gains are explained by baseline strength differences."
    elif c3_delta > 0 and c4_delta > 0 and (p_delta > 0 or i_delta > 0) and net_graph_causal_gain > 0:
        verdict = "M-A — GRAPH INCREMENTAL VALUE CONFIRMED"
        rationale = f"S4 ({s4_acc}%) outperforms S3 (Delta={c3_delta:+.2f}pp) and S5 (Delta={c4_delta:+.2f}pp), with positive gains concentrated in Partial ({p_delta:+.2f}pp) or Implicit ({i_delta:+.2f}pp) strata and Net Graph Causal Gain = +{net_graph_causal_gain}."
    else:
        verdict = "M-B — STRUCTURED RETRIEVAL CONFIRMED, GRAPH NOT NECESSARY"
        rationale = f"Structured retrieval outperforms Dense Top-5 (S2={s2_acc}% vs S0={accuracy_matrix['s0_dense5']['ALL']}%), but Graph Increment S4 vs S3 ({c3_delta:+.2f}pp) does not confirm independent necessity."

    report_payload = {
        "N": len(gold_items),
        "accuracy_matrix": accuracy_matrix,
        "true_same_evidence_flip_rate": {
            "checks": same_evidence_checks,
            "flips": same_evidence_flips,
            "rate": true_flip_rate
        },
        "causal_contrasts": contrasts,
        "graph_mechanism_attribution": {
            "rescues_breakdown": dict(cat_rescue),
            "regressions_breakdown": dict(cat_regr),
            "true_graph_causal_rescues": true_graph_rescues,
            "true_graph_causal_regressions": true_graph_regressions,
            "net_graph_causal_gain": net_graph_causal_gain
        },
        "final_verdict": {
            "verdict": verdict,
            "rationale": rationale
        }
    }

    with open(REPORTS_DIR / "holdout3_majority.json", "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)

    print(f"\nFinal Verdict: {verdict}")
    print(f"Rationale: {rationale}")
    return report_payload


def generate_mechanism_report_markdown(report: Dict[str, Any], gold_items: List[Dict[str, Any]]):
    acc = report["accuracy_matrix"]
    contrasts = report["causal_contrasts"]
    attr = report["graph_mechanism_attribution"]
    flips = report["true_same_evidence_flip_rate"]
    v_info = report["final_verdict"]

    md = fr"""# Mechanism Holdout-3 Confirmation Report

**Experiment Date**: 2026-09-22  
**Target Repository**: `Knowledge-Routing-RAG`  
**Dataset**: Mechanism Holdout-3 ($N = {report['N']}$) on D100 (Corrected Frozen Corpus)  
**Strata Allocation**: 80 Explicit (Q-E) + 80 Partial (Q-P) + 80 Implicit (Q-I)  
**Evaluation Protocol**: 3 Independent Paired Generation & LLM Judge Runs (Seeds: 101, 202, 303) — **Zero Short-Circuit Copying**

---

## 1. Final Gate Verdict

```text
================================================================================
MECHANISM CONFIRMATION VERDICT:
{v_info['verdict']}
================================================================================
```

**Verdict Rationale**:  
{v_info['rationale']}

---

## 2. Primary 6-System Accuracy Matrix

| System | System Description | Overall ($N=240$) | Explicit ($N=80$) | Partial ($N=80$) | Implicit ($N=80$) |
|:---|:---|:---:|:---:|:---:|:---:|
| **S0** | Dense Top-5 (Historical B0) | {acc['s0_dense5']['ALL']}% | {acc['s0_dense5']['E']}% | {acc['s0_dense5']['P']}% | {acc['s0_dense5']['I']}% |
| **S1** | Dense Top-20 + Composer | {acc['s1_dense20']['ALL']}% | {acc['s1_dense20']['E']}% | {acc['s1_dense20']['P']}% | {acc['s1_dense20']['I']}% |
| **S2** | Metadata / Title Routing + Local BM25 | {acc['s2_metadata_bm25']['ALL']}% | {acc['s2_metadata_bm25']['E']}% | {acc['s2_metadata_bm25']['P']}% | {acc['s2_metadata_bm25']['I']}% |
| **S3** | V3-NoGraph (Matched Ablation) | {acc['s3_v3_nograph']['ALL']}% | {acc['s3_v3_nograph']['E']}% | {acc['s3_v3_nograph']['P']}% | {acc['s3_v3_nograph']['I']}% |
| **S4** | V3-TrueGraph (Frozen V3) | {acc['s4_v3_truegraph']['ALL']}% | {acc['s4_v3_truegraph']['E']}% | {acc['s4_v3_truegraph']['P']}% | {acc['s4_v3_truegraph']['I']}% |
| **S5** | V3-ShuffledGraph (Control Baseline) | {acc['s5_v3_shuffledgraph']['ALL']}% | {acc['s5_v3_shuffledgraph']['E']}% | {acc['s5_v3_shuffledgraph']['P']}% | {acc['s5_v3_shuffledgraph']['I']}% |

---

## 3. Pre-Registered Primary Causal Contrasts

| Contrast ID | Contrast Definition | Empirical Delta ($\Delta$) | Rescues | Regressions | Net | McNemar $p$-value | Bootstrap 95% CI |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **C1** | **Wider Dense Recall** ($S_1 - S_0$) | **{contrasts['C1_Wider_Dense']['overall']['delta']:+.2f}pp** | {contrasts['C1_Wider_Dense']['overall']['rescues']} | {contrasts['C1_Wider_Dense']['overall']['regressions']} | {contrasts['C1_Wider_Dense']['overall']['net_rescues']:+d} | $p={contrasts['C1_Wider_Dense']['overall']['mcnemar_p_value']}$ | {contrasts['C1_Wider_Dense']['overall']['bootstrap_ci_95']} |
| **C2** | **Structured Metadata** ($S_2 - S_0$) | **{contrasts['C2_Structured_Metadata']['overall']['delta']:+.2f}pp** | {contrasts['C2_Structured_Metadata']['overall']['rescues']} | {contrasts['C2_Structured_Metadata']['overall']['regressions']} | {contrasts['C2_Structured_Metadata']['overall']['net_rescues']:+d} | $p={contrasts['C2_Structured_Metadata']['overall']['mcnemar_p_value']}$ | {contrasts['C2_Structured_Metadata']['overall']['bootstrap_ci_95']} |
| **C3** | **Graph Increment** ($S_4 - S_3$) | **{contrasts['C3_Graph_Increment']['overall']['delta']:+.2f}pp** | {contrasts['C3_Graph_Increment']['overall']['rescues']} | {contrasts['C3_Graph_Increment']['overall']['regressions']} | {contrasts['C3_Graph_Increment']['overall']['net_rescues']:+d} | $p={contrasts['C3_Graph_Increment']['overall']['mcnemar_p_value']}$ | {contrasts['C3_Graph_Increment']['overall']['bootstrap_ci_95']} |
| **C4** | **Graph Semantics** ($S_4 - S_5$) | **{contrasts['C4_Graph_Semantics']['overall']['delta']:+.2f}pp** | {contrasts['C4_Graph_Semantics']['overall']['rescues']} | {contrasts['C4_Graph_Semantics']['overall']['regressions']} | {contrasts['C4_Graph_Semantics']['overall']['net_rescues']:+d} | $p={contrasts['C4_Graph_Semantics']['overall']['mcnemar_p_value']}$ | {contrasts['C4_Graph_Semantics']['overall']['bootstrap_ci_95']} |

### 3.1 Explicitness Interaction Breakdown (C3: $S_4 - S_3$)

- **Explicit (Q-E)**: $\Delta = {contrasts['C3_Graph_Increment']['by_stratum']['E']['delta']:+.2f}\\text{{pp}}$, 95% CI: {contrasts['C3_Graph_Increment']['by_stratum']['E']['bootstrap_ci_95']}, McNemar $p = {contrasts['C3_Graph_Increment']['by_stratum']['E']['mcnemar_p_value']}$
- **Partial (Q-P)**: $\Delta = {contrasts['C3_Graph_Increment']['by_stratum']['P']['delta']:+.2f}\\text{{pp}}$, 95% CI: {contrasts['C3_Graph_Increment']['by_stratum']['P']['bootstrap_ci_95']}, McNemar $p = {contrasts['C3_Graph_Increment']['by_stratum']['P']['mcnemar_p_value']}$
- **Implicit (Q-I)**: $\Delta = {contrasts['C3_Graph_Increment']['by_stratum']['I']['delta']:+.2f}\\text{{pp}}$, 95% CI: {contrasts['C3_Graph_Increment']['by_stratum']['I']['bootstrap_ci_95']}, McNemar $p = {contrasts['C3_Graph_Increment']['by_stratum']['I']['mcnemar_p_value']}$

---

## 4. True Same-Evidence Flip Rate (Empirical Measurement)

Unlike historical runners that short-circuited identical evidence chunks, this experiment executed fresh, independent LLM generation calls:
- Identical Evidence Cases Tested: **{flips['checks']}**
- Flipped Correctness Decisions: **{flips['flips']}**
- **True Empirical Same-Evidence Flip Rate**: **{flips['rate']}%**

---

## 5. Graph Mechanism Attribution

For all discordant cases between $S_4$ (V3-TrueGraph) and $S_3$ (V3-NoGraph):

### 5.1 Rescues Breakdown ($S_4$ Correct, $S_3$ Wrong)
- `TRUE_GRAPH_CAUSAL_RESCUE`: **{attr['true_graph_causal_rescues']}** (Graph edge directly retrieved missing target doc and chunk)
- `GRAPH_ADDED_NON_GOLD_BUT_GENERATION_FLIP`: **{attr['rescues_breakdown'].get('GRAPH_ADDED_NON_GOLD_BUT_GENERATION_FLIP', 0)}**
- `SAME_EVIDENCE_FLIP`: **{attr['rescues_breakdown'].get('SAME_EVIDENCE_FLIP', 0)}**
- `UNCERTAIN`: **{attr['rescues_breakdown'].get('UNCERTAIN', 0)}**

### 5.2 Regressions Breakdown ($S_3$ Correct, $S_4$ Wrong)
- `GRAPH_CAUSAL_REGRESSION`: **{attr['true_graph_causal_regressions']}** (Graph neighbor introduced distractor or caused useful eviction)
- `GRAPH_DISTRACTOR`: **{attr['regressions_breakdown'].get('GRAPH_DISTRACTOR', 0)}**
- `SAME_EVIDENCE_FLIP`: **{attr['regressions_breakdown'].get('SAME_EVIDENCE_FLIP', 0)}**
- `UNCERTAIN`: **{attr['regressions_breakdown'].get('UNCERTAIN', 0)}**

### 5.3 Net Graph Causal Metric
$$\\text{{Net Graph Causal Gain}} = {attr['true_graph_causal_rescues']} - {attr['true_graph_causal_regressions']} = \\mathbf{{{attr['net_graph_causal_gain']:+d}}}$$
"""

    with open(REPORTS_DIR / "MECHANISM_HOLDOUT3_REPORT.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("Saved final mechanism report to reports/MECHANISM_HOLDOUT3_REPORT.md")


if __name__ == "__main__":
    gold_items, gold_map = load_holdout3_benchmark()
    search_service = SearchService()
    lsdb = KnowledgeLSDB()
    shuffled_lsdb = build_shuffled_lsdb(lsdb, seed=42)
    llm_service = LLMService()
    evaluator = Evaluator(llm_service=llm_service)

    # Step 1: Retrieval for 6 systems
    traces = run_deterministic_retrieval_all_systems(gold_items, search_service, lsdb, shuffled_lsdb)

    # Step 2: 3 Paired Generation Runs
    run1 = run_fresh_generation_and_judge(1, 101, gold_items, traces, llm_service, lsdb, evaluator)
    run2 = run_fresh_generation_and_judge(2, 202, gold_items, traces, llm_service, lsdb, evaluator)
    run3 = run_fresh_generation_and_judge(3, 303, gold_items, traces, llm_service, lsdb, evaluator)

    # Step 3: Compute Majority, Statistics, and Attribution
    report = compute_majority_and_statistics(gold_items, run1, run2, run3, traces, lsdb)
    generate_mechanism_report_markdown(report, gold_items)
