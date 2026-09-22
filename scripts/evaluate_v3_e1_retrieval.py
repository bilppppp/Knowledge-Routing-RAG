#!/usr/bin/env python3
"""
scripts/evaluate_v3_e1_retrieval.py
Retrieval-Only Evaluation and Comparative Analysis of:
  - D0: B0 Baseline (Vector Top-5)
  - D1: C7-Clean-Raw (Position-based slot 4/5 replacement)
  - E1: Coverage-Preserving Composition (E1 Composer)

Evaluates on the Optimization / Development Benchmark (N = 216 tasks across D20, D50, D100).
Zero LLM generation cost; pure retrieval, coverage, and evidence composition verification.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.search import SearchService
from src.services.llm import LLMService
from src.graph.lsdb import KnowledgeLSDB
from src.evaluation.metrics import Evaluator
from src.routing.c7_clean_router import C7CleanRouterSystem
from src.routing.e1_composer_router import E1ComposerRouterSystem
from src.composition.slots import extract_evidence_slots
from src.composition.composer import compose_evidence, compute_set_slot_coverage, compute_set_redundancy

BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
RUNS_TEST_DIR = PROJECT_ROOT / "runs" / "test"
RUNS_ABLATION_DIR = PROJECT_ROOT / "runs" / "ablation"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_dev_benchmark() -> Tuple[List[str], Dict[str, Any]]:
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


def run_e1_retrieval_suite():
    print("=================================================================")
    print(" V3 Stage A Candidate E1: Retrieval-Only Evaluation (N = 216)")
    print("=================================================================")

    test_qids, gold_map = load_dev_benchmark()
    b0_traces = load_b0_traces()
    d1_traces = load_d1_traces()

    tasks = sorted(list(b0_traces.keys()))
    print(f"Loaded {len(tasks)} benchmark tasks across D20, D50, D100.")

    search_service = SearchService()
    llm_service = LLMService()
    lsdb = KnowledgeLSDB()
    evaluator = Evaluator(llm_service=llm_service)

    # Instantiate E1 router
    e1_system = E1ComposerRouterSystem(
        search_service=search_service,
        llm_service=llm_service,
        lsdb=lsdb,
        b0_traces=b0_traces,
        top_k=5,
        shadow_top_k=20,
        max_hops=2,
        max_evidence_tokens=4000,
        generation_mode="RAW"
    )

    e1_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    t_start = time.time()

    # Metrics collectors
    d0_doc_hits, d1_doc_hits, e1_doc_hits = [], [], []
    d0_chunk_hits, d1_chunk_hits, e1_chunk_hits = [], [], []
    d0_f1s, d1_f1s, e1_f1s = [], [], []
    d0_chains, d1_chains, e1_chains = [], [], []

    d1_ret_rescues, d1_ret_regressions = [], []
    e1_ret_rescues, e1_ret_regressions = [], []

    # Evidence eviction tracking
    d1_evictions: List[Dict[str, Any]] = []
    e1_evictions: List[Dict[str, Any]] = []

    # Coverage metrics
    total_slots_count = 0
    covered_slots_count = 0
    unique_slots_total = 0
    unique_slots_preserved = 0
    routed_accepted_count = 0
    routed_rejected_count = 0
    locked_rejections_count = 0
    redundancy_scores = []
    marginal_deltas = []

    for idx, (qid, corpus) in enumerate(tasks, 1):
        gold = gold_map[qid]
        question = gold["question"]
        gold_chunks = set(gold["gold_chunk_ids"])
        gold_docs = set(gold["gold_documents"])

        # D0 (B0)
        d0_cids = b0_traces[(qid, corpus)]["final_evidence_chunk_ids"]
        # D1 (C7-Clean-Raw)
        d1_cids = d1_traces[(qid, corpus)]["final_evidence_chunk_ids"]

        # E1 Retrieval & Composition Execution
        seed_items = search_service.vector_search(query=question, corpus=corpus, top_k=5)
        lane = e1_system.detect_lane(question, seed_items)

        if lane == "FAST_PATH":
            e1_cids = [s.chunk_id for s in seed_items]
            comp_traces = []
            q_slots = extract_evidence_slots(question)
        else:
            # Reconstruct candidate pool exactly as in Clean Router
            # Upstream candidate discovery is frozen
            cand_pool: List[Tuple[Any, float, str]] = []
            primary_doc_id = seed_items[0].doc_id if seed_items else ""

            if lane == "TEMPORAL_BASIS":
                for s in seed_items[:3]:
                    nbrs = lsdb.get_routing_neighbors(s.chunk_id, corpus=corpus, allowed_relations={"SUPERSEDES", "AMENDS"})
                    for tgt, rel, cost in nbrs:
                        chk = lsdb.get_chunk_evidence(tgt)
                        if chk:
                            from src.common.models import EvidenceItem
                            cand_pool.append((EvidenceItem(
                                chunk_id=chk["chunk_id"], doc_id=chk["doc_id"], title=chk["title"],
                                heading_path=chk["heading_path"], text=chk["text"], score=0.95,
                                source_method=f"clean_lane_a_{rel.lower()}"
                            ), 1.2, f"Resolved version via {rel}"))

                if primary_doc_id:
                    fts_repeal = search_service.fts_search_in_doc("施行 废止 附则", primary_doc_id, corpus=corpus, top_k=1)
                    for f_item in fts_repeal:
                        f_item.score = 0.90
                        f_item.source_method = "clean_lane_a_repeal_clause"
                        cand_pool.append((f_item, 1.1, "Repeal/effective date clause"))

                chunk_basis_found = False
                for s in seed_items[:3]:
                    nbrs = lsdb.get_routing_neighbors(s.chunk_id, corpus=corpus, allowed_relations={"BASED_ON"})
                    for tgt, rel, cost in nbrs:
                        chk = lsdb.get_chunk_evidence(tgt)
                        if chk:
                            from src.common.models import EvidenceItem
                            cand_pool.append((EvidenceItem(
                                chunk_id=chk["chunk_id"], doc_id=chk["doc_id"], title=chk["title"],
                                heading_path=chk["heading_path"], text=chk["text"], score=0.95,
                                source_method="clean_lane_a_based_on"
                            ), 1.2, "Followed legislative basis"))
                            chunk_basis_found = True

                if any(k in question for k in ["上位", "依据", "根据", "何法", "哪两部"]) and not chunk_basis_found and seed_items:
                    unresolved_slot = e1_system._extract_unresolved_slot(question)
                    max_targets = 2 if any(k in question for k in ["两部", "两项", "分别"]) else 1
                    hop_res = e1_system.resolve_hierarchical_next_hop(
                        source_chunk=seed_items[0],
                        required_relation="BASED_ON",
                        unresolved_slot=unresolved_slot,
                        corpus=corpus,
                        max_targets=max_targets
                    )
                    if hop_res and hop_res["candidate_items"]:
                        for c_item in hop_res["candidate_items"]:
                            cand_pool.append((c_item, 1.4, "Hierarchical resolution"))

            elif lane == "COMPOSITE_EVIDENCE":
                has_gap, missing_statutes = e1_system._check_statutory_gap(question, seed_items)
                if missing_statutes:
                    target_statute = missing_statutes[0]
                    found_by_graph = False
                    for s in seed_items:
                        nbrs = lsdb.get_routing_neighbors(s.chunk_id, corpus=corpus, allowed_relations={"SUPERSEDES", "AMENDS", "BASED_ON", "REFERENCES"})
                        for tgt, rel, cost in nbrs:
                            if tgt in lsdb.doc_meta and target_statute in lsdb.doc_meta[tgt].get("title", ""):
                                d_chks = lsdb.get_document_chunks(tgt, corpus=corpus)
                                if d_chks:
                                    cdata = lsdb.get_chunk_evidence(d_chks[0])
                                    if cdata:
                                        from src.common.models import EvidenceItem
                                        cand_pool.append((EvidenceItem(
                                            chunk_id=cdata["chunk_id"], doc_id=cdata["doc_id"], title=cdata["title"],
                                            heading_path=cdata["heading_path"], text=cdata["text"], score=0.98,
                                            source_method=f"clean_lane_b_graph_gap_{rel.lower()}"
                                        ), 1.5, "Resolved statute via graph"))
                                        found_by_graph = True

                    if not found_by_graph and seed_items:
                        for s in seed_items[:2]:
                            hop_res = e1_system.resolve_hierarchical_next_hop(
                                source_chunk=s,
                                required_relation="REFERENCES",
                                unresolved_slot=target_statute,
                                corpus=corpus,
                                max_targets=1
                            )
                            if hop_res and hop_res["candidate_items"]:
                                for c_item in hop_res["candidate_items"]:
                                    cand_pool.append((c_item, 1.45, "Hierarchical resolution"))
                                break

                for s in seed_items[:3]:
                    nbrs = lsdb.get_routing_neighbors(s.chunk_id, corpus=corpus, allowed_relations={"SUPERSEDES", "AMENDS", "BASED_ON", "REFERENCES"})
                    for tgt, rel, cost in nbrs:
                        chk = lsdb.get_chunk_evidence(tgt)
                        if chk:
                            from src.common.models import EvidenceItem
                            cand_pool.append((EvidenceItem(
                                chunk_id=chk["chunk_id"], doc_id=chk["doc_id"], title=chk["title"],
                                heading_path=chk["heading_path"], text=chk["text"], score=0.90,
                                source_method=f"clean_lane_b_{rel.lower()}"
                            ), 1.1, f"Composite expansion via {rel}"))

            # Shadow Candidate Plane
            current_docs = set(s.doc_id for s in seed_items).union(set(c[0].doc_id for c in cand_pool))
            missing_matches = [
                m for m in e1_system._match_question_entities(question, corpus=corpus)
                if m[0] not in current_docs
            ]

            if len(missing_matches) > 0 or len(cand_pool) == 0:
                shadow_rib, selected_prefixes = e1_system._extract_shadow_prefixes(
                    question=question, corpus=corpus, seed_items=seed_items, current_candidate_docs=current_docs
                )
                descent_k = 1 if len(selected_prefixes) >= 2 else 2
                for p_did, p_score, p_title in selected_prefixes:
                    descent_query = e1_system._build_generic_descent_query(question, p_title, seed_items)
                    descent_items = search_service.fts_search_in_doc(query=descent_query, doc_id=p_did, corpus=corpus, top_k=descent_k)
                    if not descent_items:
                        c1_data = lsdb.get_chunk_evidence(f"{p_did}#c001")
                        if c1_data:
                            from src.common.models import EvidenceItem
                            descent_items = [EvidenceItem(
                                chunk_id=c1_data["chunk_id"], doc_id=c1_data["doc_id"], title=c1_data["title"],
                                heading_path=c1_data["heading_path"], text=c1_data["text"], score=0.92,
                                source_method="clean_shadow_fallback_c001"
                            )]
                    for d_item in descent_items:
                        d_item.score = 0.95
                        d_item.source_method = "clean_shadow_prefix_descent"
                        cand_pool.append((d_item, 1.45, f"Shadow descent: {p_title[:20]}"))

            q_slots = extract_evidence_slots(question)
            final_items, comp_traces = compose_evidence(
                b0_evidence=seed_items,
                candidate_pool=cand_pool,
                slots=q_slots,
                max_chunks=5,
                lsdb=lsdb
            )
            e1_cids = [it.chunk_id for it in final_items]

        # Accounting for candidates and coverage
        b0_cov = compute_set_slot_coverage(seed_items, q_slots)
        e1_items_reconstructed = []
        for cid in e1_cids:
            chk = lsdb.get_chunk_evidence(cid)
            if chk:
                from src.common.models import EvidenceItem
                e1_items_reconstructed.append(EvidenceItem(
                    chunk_id=chk["chunk_id"], doc_id=chk["doc_id"], title=chk["title"],
                    heading_path=chk["heading_path"], text=chk["text"], score=1.0
                ))
        e1_cov = compute_set_slot_coverage(e1_items_reconstructed, q_slots)

        total_slots_count += len(q_slots)
        covered_slots_count += len(e1_cov["covered_slots"])

        # Track unique slots preserved
        b0_unique_map = b0_cov["unique_coverage_map"]
        for cid, u_slots in b0_unique_map.items():
            if u_slots:
                unique_slots_total += len(u_slots)
                # Check if still covered in E1
                for s in u_slots:
                    if s in e1_cov["covered_slots"]:
                        unique_slots_preserved += 1

        for tr in comp_traces:
            if tr["action"] in {"REPLACE", "ADD"}:
                routed_accepted_count += 1
                marginal_deltas.append(tr.get("marginal_delta", 0.0))
            elif tr["action"] == "REJECT":
                routed_rejected_count += 1
                if "UNIQUE_COVERAGE_LOCK" in tr.get("reason", ""):
                    locked_rejections_count += 1

        red_score = compute_set_redundancy(e1_items_reconstructed, q_slots)
        redundancy_scores.append(red_score)

        # Retrieval Metric Calculations
        # 1. Document Recall
        d0_dids = set(c.split("#")[0] for c in d0_cids)
        d1_dids = set(c.split("#")[0] for c in d1_cids)
        e1_dids = set(c.split("#")[0] for c in e1_cids)

        d0_doc_hits.append(len(d0_dids & gold_docs) / len(gold_docs) if gold_docs else 1.0)
        d1_doc_hits.append(len(d1_dids & gold_docs) / len(gold_docs) if gold_docs else 1.0)
        e1_doc_hits.append(len(e1_dids & gold_docs) / len(gold_docs) if gold_docs else 1.0)

        # 2. Chunk Recall & F1
        d0_m = evaluator.evaluate_retrieval(d0_cids, list(gold_chunks))
        d1_m = evaluator.evaluate_retrieval(d1_cids, list(gold_chunks))
        e1_m = evaluator.evaluate_retrieval(e1_cids, list(gold_chunks))

        d0_chunk_hits.append(d0_m["recall"])
        d1_chunk_hits.append(d1_m["recall"])
        e1_chunk_hits.append(e1_m["recall"])

        d0_f1s.append(d0_m["f1"])
        d1_f1s.append(d1_m["f1"])
        e1_f1s.append(e1_m["f1"])

        # 3. Chain Completion
        d0_comp = 1.0 if gold_chunks.issubset(set(d0_cids)) else 0.0
        d1_comp = 1.0 if gold_chunks.issubset(set(d1_cids)) else 0.0
        e1_comp = 1.0 if gold_chunks.issubset(set(e1_cids)) else 0.0

        d0_chains.append(d0_comp)
        d1_chains.append(d1_comp)
        e1_chains.append(e1_comp)

        # 4. Retrieval Rescues & Regressions vs D0
        if d0_comp < 1.0 and d1_comp == 1.0:
            d1_ret_rescues.append(f"{qid}_{corpus}")
        elif d0_comp == 1.0 and d1_comp < 1.0:
            d1_ret_regressions.append(f"{qid}_{corpus}")

        if d0_comp < 1.0 and e1_comp == 1.0:
            e1_ret_rescues.append(f"{qid}_{corpus}")
        elif d0_comp == 1.0 and e1_comp < 1.0:
            e1_ret_regressions.append(f"{qid}_{corpus}")

        # 5. Useful Evidence Displacement / Eviction
        # Defined as: B0 chunk that was supporting Gold was evicted, and resulting set lacks it
        d0_gold_in_set = set(d0_cids) & gold_chunks
        d1_gold_in_set = set(d1_cids) & gold_chunks
        e1_gold_in_set = set(e1_cids) & gold_chunks

        d1_lost = d0_gold_in_set - d1_gold_in_set
        if d1_lost:
            d1_evictions.append({
                "task": f"{qid}_{corpus}",
                "lost_gold": list(d1_lost),
                "evicted_cids": list(set(d0_cids) - set(d1_cids)),
                "admitted_cids": list(set(d1_cids) - set(d0_cids))
            })

        e1_lost = d0_gold_in_set - e1_gold_in_set
        if e1_lost:
            e1_evictions.append({
                "task": f"{qid}_{corpus}",
                "lost_gold": list(e1_lost),
                "evicted_cids": list(set(d0_cids) - set(e1_cids)),
                "admitted_cids": list(set(e1_cids) - set(d0_cids)),
                "traces": comp_traces
            })

        e1_results[(qid, corpus)] = {
            "qid": qid,
            "corpus": corpus,
            "final_cids": e1_cids,
            "lane": lane,
            "traces": comp_traces
        }

    elapsed = time.time() - t_start

    # Summary dictionary
    summary = {
        "N": len(tasks),
        "execution_time_s": round(elapsed, 2),
        "retrieval_metrics": {
            "gold_doc_recall": {
                "d0": float(np.mean(d0_doc_hits)),
                "d1": float(np.mean(d1_doc_hits)),
                "e1": float(np.mean(e1_doc_hits)),
                "delta_d1_vs_d0": float(np.mean(d1_doc_hits) - np.mean(d0_doc_hits)),
                "delta_e1_vs_d0": float(np.mean(e1_doc_hits) - np.mean(d0_doc_hits)),
                "delta_e1_vs_d1": float(np.mean(e1_doc_hits) - np.mean(d1_doc_hits))
            },
            "gold_chunk_recall": {
                "d0": float(np.mean(d0_chunk_hits)),
                "d1": float(np.mean(d1_chunk_hits)),
                "e1": float(np.mean(e1_chunk_hits)),
                "delta_d1_vs_d0": float(np.mean(d1_chunk_hits) - np.mean(d0_chunk_hits)),
                "delta_e1_vs_d0": float(np.mean(e1_chunk_hits) - np.mean(d0_chunk_hits)),
                "delta_e1_vs_d1": float(np.mean(e1_chunk_hits) - np.mean(d1_chunk_hits))
            },
            "evidence_f1": {
                "d0": float(np.mean(d0_f1s)),
                "d1": float(np.mean(d1_f1s)),
                "e1": float(np.mean(e1_f1s)),
                "delta_d1_vs_d0": float(np.mean(d1_f1s) - np.mean(d0_f1s)),
                "delta_e1_vs_d0": float(np.mean(e1_f1s) - np.mean(d0_f1s)),
                "delta_e1_vs_d1": float(np.mean(e1_f1s) - np.mean(d1_f1s))
            },
            "chain_completion": {
                "d0": float(np.mean(d0_chains)),
                "d1": float(np.mean(d1_chains)),
                "e1": float(np.mean(e1_chains)),
                "delta_d1_vs_d0": float(np.mean(d1_chains) - np.mean(d0_chains)),
                "delta_e1_vs_d0": float(np.mean(e1_chains) - np.mean(d0_chains)),
                "delta_e1_vs_d1": float(np.mean(e1_chains) - np.mean(d1_chains))
            },
            "rescues_and_regressions": {
                "d1_rescues": len(d1_ret_rescues),
                "d1_regressions": len(d1_ret_regressions),
                "d1_net_rescue": len(d1_ret_rescues) - len(d1_ret_regressions),
                "d1_rescue_keys": d1_ret_rescues,
                "d1_regression_keys": d1_ret_regressions,
                "e1_rescues": len(e1_ret_rescues),
                "e1_regressions": len(e1_ret_regressions),
                "e1_net_rescue": len(e1_ret_rescues) - len(e1_ret_regressions),
                "e1_rescue_keys": e1_ret_rescues,
                "e1_regression_keys": e1_ret_regressions
            }
        },
        "evidence_displacement": {
            "d1_useful_evidence_evicted_count": len(d1_evictions),
            "d1_evicted_instances": d1_evictions,
            "e1_useful_evidence_evicted_count": len(e1_evictions),
            "e1_evicted_instances": e1_evictions
        },
        "coverage_metrics": {
            "total_slots": total_slots_count,
            "covered_slots": covered_slots_count,
            "slot_coverage_rate": covered_slots_count / max(1, total_slots_count),
            "avg_covered_slots_per_q": covered_slots_count / len(tasks),
            "unique_slots_total": unique_slots_total,
            "unique_slots_preserved": unique_slots_preserved,
            "unique_slot_preservation_rate": unique_slots_preserved / max(1, unique_slots_total),
            "routed_candidates_accepted": routed_accepted_count,
            "routed_candidates_rejected": routed_rejected_count,
            "locked_rejections_count": locked_rejections_count,
            "avg_marginal_gain": float(np.mean(marginal_deltas)) if marginal_deltas else 0.0,
            "avg_set_redundancy": float(np.mean(redundancy_scores))
        }
    }

    # Print Report to console
    print("\n-----------------------------------------------------------------")
    print(" Retrieval Metrics Comparison (D0 Baseline vs D1 Clean-Raw vs E1 Composer):")
    print("-----------------------------------------------------------------")
    print(f"Gold Document Recall : D0 = {summary['retrieval_metrics']['gold_doc_recall']['d0']*100:.2f}% | D1 = {summary['retrieval_metrics']['gold_doc_recall']['d1']*100:.2f}% | E1 = {summary['retrieval_metrics']['gold_doc_recall']['e1']*100:.2f}% (Δ vs D1: {summary['retrieval_metrics']['gold_doc_recall']['delta_e1_vs_d1']*100:+.2f}pp)")
    print(f"Gold Chunk Recall    : D0 = {summary['retrieval_metrics']['gold_chunk_recall']['d0']*100:.2f}% | D1 = {summary['retrieval_metrics']['gold_chunk_recall']['d1']*100:.2f}% | E1 = {summary['retrieval_metrics']['gold_chunk_recall']['e1']*100:.2f}% (Δ vs D1: {summary['retrieval_metrics']['gold_chunk_recall']['delta_e1_vs_d1']*100:+.2f}pp)")
    print(f"Chain Completion Rate: D0 = {summary['retrieval_metrics']['chain_completion']['d0']*100:.2f}% | D1 = {summary['retrieval_metrics']['chain_completion']['d1']*100:.2f}% | E1 = {summary['retrieval_metrics']['chain_completion']['e1']*100:.2f}% (Δ vs D1: {summary['retrieval_metrics']['chain_completion']['delta_e1_vs_d1']*100:+.2f}pp)")
    print(f"Evidence F1          : D0 = {summary['retrieval_metrics']['evidence_f1']['d0']*100:.2f}% | D1 = {summary['retrieval_metrics']['evidence_f1']['d1']*100:.2f}% | E1 = {summary['retrieval_metrics']['evidence_f1']['e1']*100:.2f}% (Δ vs D1: {summary['retrieval_metrics']['evidence_f1']['delta_e1_vs_d1']*100:+.2f}pp)")
    print(f"Retrieval Rescues    : D1 = {summary['retrieval_metrics']['rescues_and_regressions']['d1_rescues']} | E1 = {summary['retrieval_metrics']['rescues_and_regressions']['e1_rescues']}")
    print(f"Retrieval Regressions: D1 = {summary['retrieval_metrics']['rescues_and_regressions']['d1_regressions']} | E1 = {summary['retrieval_metrics']['rescues_and_regressions']['e1_regressions']}")
    print(f"Net Retrieval Rescue : D1 = {summary['retrieval_metrics']['rescues_and_regressions']['d1_net_rescue']:+d} | E1 = {summary['retrieval_metrics']['rescues_and_regressions']['e1_net_rescue']:+d}")

    print("\n-----------------------------------------------------------------")
    print(" Evidence Displacement & Eviction Analysis:")
    print("-----------------------------------------------------------------")
    print(f"D1 Useful Evidence Evictions (Lost Gold): {summary['evidence_displacement']['d1_useful_evidence_evicted_count']}")
    for ev in summary['evidence_displacement']['d1_evicted_instances']:
        print(f"  D1 Evicted: {ev['task']} | Lost Gold: {ev['lost_gold']} | Evicted B0: {ev['evicted_cids']} | Admitted: {ev['admitted_cids']}")

    print(f"E1 Useful Evidence Evictions (Lost Gold): {summary['evidence_displacement']['e1_useful_evidence_evicted_count']}")
    for ev in summary['evidence_displacement']['e1_evicted_instances']:
        print(f"  E1 Evicted: {ev['task']} | Lost Gold: {ev['lost_gold']} | Evicted B0: {ev['evicted_cids']} | Admitted: {ev['admitted_cids']}")

    print("\n-----------------------------------------------------------------")
    print(" Coverage-Aware Decision Metrics:")
    print("-----------------------------------------------------------------")
    print(f"Slot Coverage Rate            : {summary['coverage_metrics']['slot_coverage_rate']*100:.2f}% ({summary['coverage_metrics']['covered_slots']}/{summary['coverage_metrics']['total_slots']})")
    print(f"Avg Covered Slots / Question  : {summary['coverage_metrics']['avg_covered_slots_per_q']:.2f}")
    print(f"Unique Slot Preservation Rate : {summary['coverage_metrics']['unique_slot_preservation_rate']*100:.2f}% ({summary['coverage_metrics']['unique_slots_preserved']}/{summary['coverage_metrics']['unique_slots_total']})")
    print(f"Routed Candidates Accepted    : {summary['coverage_metrics']['routed_candidates_accepted']}")
    print(f"Routed Candidates Rejected    : {summary['coverage_metrics']['routed_candidates_rejected']}")
    print(f"  Rejected via Unique Lock    : {summary['coverage_metrics']['locked_rejections_count']}")
    print(f"Avg Marginal Gain of Admits   : {summary['coverage_metrics']['avg_marginal_gain']:.4f}")
    print(f"Avg Set Redundancy Score      : {summary['coverage_metrics']['avg_set_redundancy']:.4f}")

    out_file = REPORTS_DIR / "v3_e1_retrieval_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed retrieval analysis to {out_file}")

    return summary, e1_results


if __name__ == "__main__":
    run_e1_retrieval_suite()
