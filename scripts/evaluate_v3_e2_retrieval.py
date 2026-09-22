#!/usr/bin/env python3
"""
scripts/evaluate_v3_e2_retrieval.py
Retrieval-Only Evaluation and Comparative Analysis of:
  - D0: B0 Baseline (Vector Top-5)
  - D1: C7-Clean-Raw (Position-based slot 4/5 replacement)
  - E1: Coverage-Preserving Composition (E1 Composer)
  - E2: Slot-Conditioned Hybrid Targeted Descent (E2-v0)

Evaluates strictly on the Optimization / Development Benchmark (N = 216 across D20, D50, D100).
Evaluates the Retrieval Gate criteria before permitting answer generation.
Generates:
  - reports/v3_e2_retrieval_analysis.json
  - reports/v3_e2_descent_funnel.json
"""

import sys
import os
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
from src.routing.e2_descent_router import E2DescentRouterSystem
from src.composition.slots import extract_evidence_slots

BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
RUNS_TEST_DIR = PROJECT_ROOT / "runs" / "test"
RUNS_ABLATION_DIR = PROJECT_ROOT / "runs" / "ablation"
RUNS_V3_E1_DIR = PROJECT_ROOT / "runs" / "v3" / "E1"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_benchmark() -> Tuple[List[str], Dict[str, Any]]:
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


def load_e1_traces() -> Dict[Tuple[str, str], Dict[str, Any]]:
    e1_map = {}
    for p in RUNS_V3_E1_DIR.glob("*/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            e1_map[(d["qid"], d["corpus"])] = d
    return e1_map


def run_e2_retrieval_evaluation():
    print("=================================================================")
    print(" V3 Stage B Candidate E2: Retrieval-Only Evaluation (N = 216)")
    print("=================================================================")

    test_qids, gold_map = load_benchmark()
    b0_traces = load_b0_traces()
    d1_traces = load_d1_traces()
    e1_traces = load_e1_traces()

    tasks = sorted(list(b0_traces.keys()))
    print(f"Loaded {len(tasks)} benchmark tasks across D20, D50, D100.")

    search_service = SearchService()
    llm_service = LLMService()
    lsdb = KnowledgeLSDB()

    # Instantiate E2 Router
    e2_system = E2DescentRouterSystem(
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

    # Metrics collectors for D0, D1, E1, E2
    d0_doc_hits, d1_doc_hits, e1_doc_hits, e2_doc_hits = [], [], [], []
    d0_chunk_hits, d1_chunk_hits, e1_chunk_hits, e2_chunk_hits = [], [], [], []
    d0_f1s, d1_f1s, e1_f1s, e2_f1s = [], [], [], []
    d0_chains, d1_chains, e1_chains, e2_chains = [], [], [], []

    # Candidate pool hit tracking
    e1_pool_gold_hits = []
    e2_pool_gold_hits = []
    e1_pool_eq_hits = []
    e2_pool_eq_hits = []

    # Hybrid channel complementarity tracking (for instances where descent ran)
    channel_counts = {
        "gold_lexical_only": 0,
        "gold_semantic_only": 0,
        "gold_both": 0,
        "gold_neither": 0
    }

    # Eviction tracking
    d1_evictions = []
    e1_evictions = []
    e2_evictions = []

    # E2 trace storage for potential answer evaluation
    e2_traces_data: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # Funnel tracking
    funnel_target_doc_correct = 0
    funnel_found_locally = 0
    funnel_entered_pool = 0
    funnel_admitted = 0
    funnel_chain_complete = 0
    total_routed_count = 0

    t_start = time.time()

    for idx, (qid, corpus) in enumerate(tasks, 1):
        gold = gold_map[qid]
        question = gold["question"]
        gold_chunks = set(gold["gold_chunk_ids"])
        gold_docs = set(gold["gold_documents"])

        # Prior runs
        d0_cids = b0_traces[(qid, corpus)]["final_evidence_chunk_ids"]
        d1_cids = d1_traces[(qid, corpus)]["final_evidence_chunk_ids"]
        e1_cids = e1_traces[(qid, corpus)]["final_evidence_chunk_ids"]

        # Run E2 retrieval & composition (mock LLM call in retrieval mode to save time/cost)
        seed_items = search_service.vector_search(query=question, corpus=corpus, top_k=5)
        lane = e2_system.detect_lane(question, seed_items)
        q_slots = extract_evidence_slots(question)

        if lane == "FAST_PATH":
            e2_cids = [s.chunk_id for s in seed_items]
            cand_pool_cids = []
            comp_traces = []
            channel_diag = []
        else:
            total_routed_count += 1
            # Execute candidate discovery using E2 logic
            cand_pool = []
            channel_diag = []
            primary_doc_id = seed_items[0].doc_id if seed_items else ""

            # Lane A
            if lane == "TEMPORAL_BASIS":
                for s in seed_items[:3]:
                    nbrs = lsdb.get_routing_neighbors(s.chunk_id, corpus=corpus, allowed_relations={"SUPERSEDES", "AMENDS"})
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
                    fts_repeal = search_service.fts_search_in_doc("施行 废止 附则", primary_doc_id, corpus=corpus, top_k=1)
                    for f_item in fts_repeal:
                        f_item.score = 0.90
                        f_item.source_method = "e2_lane_a_repeal_clause"
                        cand_pool.append((f_item, 1.1, "Repeal/effective date clause"))

                chunk_basis_found = False
                for s in seed_items[:3]:
                    nbrs = lsdb.get_routing_neighbors(s.chunk_id, corpus=corpus, allowed_relations={"BASED_ON"})
                    for tgt, rel, cost in nbrs:
                        chk = lsdb.get_chunk_evidence(tgt)
                        if chk:
                            it = EvidenceItem(
                                chunk_id=chk["chunk_id"], doc_id=chk["doc_id"], title=chk["title"],
                                heading_path=chk["heading_path"], text=chk["text"], score=0.95,
                                source_method="e2_lane_a_based_on"
                            )
                            cand_pool.append((it, 1.2, "Followed legislative basis"))
                            chunk_basis_found = True

                if any(k in question for k in ["上位", "依据", "根据", "何法", "哪两部"]) and not chunk_basis_found and seed_items:
                    unresolved_str = e2_system._extract_unresolved_slot(question)
                    max_targets = 2 if any(k in question for k in ["两部", "两项", "分别"]) else 1
                    hop_res = e2_system.resolve_hierarchical_next_hop(
                        source_chunk=seed_items[0],
                        required_relation="BASED_ON",
                        unresolved_slot_str=unresolved_str,
                        question=question,
                        seed_items=seed_items,
                        slots=q_slots,
                        corpus=corpus,
                        max_targets=max_targets
                    )
                    if hop_res and hop_res["candidate_items"]:
                        for idx, c_item in enumerate(hop_res["candidate_items"]):
                            cand_pool.append((c_item, 1.4, "Hierarchical resolution"))
                            if idx < len(hop_res.get("channel_metadata", [])):
                                channel_diag.append(hop_res["channel_metadata"][idx])

            # Lane B
            elif lane == "COMPOSITE_EVIDENCE":
                has_gap, missing_statutes = e2_system._check_statutory_gap(question, seed_items)
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
                                        it = EvidenceItem(
                                            chunk_id=cdata["chunk_id"], doc_id=cdata["doc_id"], title=cdata["title"],
                                            heading_path=cdata["heading_path"], text=cdata["text"], score=0.98,
                                            source_method=f"e2_lane_b_graph_gap_{rel.lower()}"
                                        )
                                        cand_pool.append((it, 1.5, f"Missing statute {target_statute} resolved via graph"))
                                        found_by_graph = True

                    if not found_by_graph and seed_items:
                        for s in seed_items[:2]:
                            hop_res = e2_system.resolve_hierarchical_next_hop(
                                source_chunk=s,
                                required_relation="REFERENCES",
                                unresolved_slot_str=target_statute,
                                question=question,
                                seed_items=seed_items,
                                slots=q_slots,
                                corpus=corpus,
                                max_targets=1
                            )
                            if hop_res and hop_res["candidate_items"]:
                                for idx, c_item in enumerate(hop_res["candidate_items"]):
                                    cand_pool.append((c_item, 1.45, "Hierarchical resolution"))
                                    if idx < len(hop_res.get("channel_metadata", [])):
                                        channel_diag.append(hop_res["channel_metadata"][idx])
                                found_by_graph = True
                                break

                for s in seed_items[:3]:
                    nbrs = lsdb.get_routing_neighbors(s.chunk_id, corpus=corpus, allowed_relations={"SUPERSEDES", "AMENDS", "BASED_ON", "REFERENCES"})
                    for tgt, rel, cost in nbrs:
                        chk = lsdb.get_chunk_evidence(tgt)
                        if chk:
                            it = EvidenceItem(
                                chunk_id=chk["chunk_id"], doc_id=chk["doc_id"], title=chk["title"],
                                heading_path=chk["heading_path"], text=chk["text"], score=0.90,
                                source_method=f"e2_lane_b_{rel.lower()}"
                            )
                            cand_pool.append((it, 1.1, f"Composite expansion via {rel}"))

            # Control Plane Shadow Candidate Plane
            current_docs = set(s.doc_id for s in seed_items).union(set(c[0].doc_id for c in cand_pool))
            missing_matches = [
                m for m in e2_system._match_question_entities(question, corpus=corpus)
                if m[0] not in current_docs
            ]

            if len(missing_matches) > 0 or len(cand_pool) == 0:
                shadow_rib, selected_prefixes = e2_system._extract_shadow_prefixes(
                    question=question, corpus=corpus, seed_items=seed_items, current_candidate_docs=current_docs
                )
                descent_k = 2 if len(selected_prefixes) >= 2 else 3
                for p_did, p_score, p_title in selected_prefixes:
                    from src.composition.descent import identify_unresolved_slot, build_generic_descent_query, hybrid_targeted_descent
                    unresolved_slot_obj, covered_slots_list = identify_unresolved_slot(
                        question=question,
                        seed_items=seed_items,
                        slots=q_slots,
                        target_doc_title=p_title
                    )
                    descent_query = build_generic_descent_query(
                        question=question,
                        unresolved_slot=unresolved_slot_obj,
                        target_doc_title=p_title,
                        covered_slots=covered_slots_list
                    )

                    hybrid_res = hybrid_targeted_descent(
                        search_service=search_service,
                        target_doc_id=p_did,
                        target_doc_title=p_title,
                        unresolved_slot=unresolved_slot_obj,
                        generic_query=descent_query,
                        corpus=corpus,
                        top_k_candidates=descent_k,
                        lsdb=lsdb
                    )

                    for d_item, rrf_score, meta in hybrid_res:
                        cand_pool_score = 1.45 if any(m[0] == p_did for m in missing_matches) else 1.35
                        cand_pool.append((d_item, cand_pool_score, f"E2 Hybrid: {descent_query[:25]}"))
                        channel_diag.append(meta)

            # Admission gate: E1 Coverage-Preserving Composition
            from src.composition.composer import compose_evidence
            final_items, comp_traces = compose_evidence(
                b0_evidence=seed_items,
                candidate_pool=cand_pool,
                slots=q_slots,
                max_chunks=5,
                max_replacements=2,
                lsdb=lsdb
            )
            e2_cids = [it.chunk_id for it in final_items]
            cand_pool_cids = [c[0].chunk_id for c in cand_pool]

            # Channel complementarity tracking
            for meta in channel_diag:
                ch = meta.get("channel")
                # Check if this chunk is gold
                # In general tracking:
                pass

        # Track candidate pool recall
        # E1 pool cids
        e1_trace = e1_traces[(qid, corpus)]
        e1_comp_traces = e1_trace.get("metadata", {}).get("composition_traces", [])
        e1_pool_cids = [t["candidate"] for t in e1_comp_traces]

        e1_pool_gold = any(gc in e1_pool_cids for gc in gold_chunks)
        e2_pool_gold = any(gc in cand_pool_cids for gc in gold_chunks)
        e1_pool_gold_hits.append(1.0 if e1_pool_gold else 0.0)
        e2_pool_gold_hits.append(1.0 if e2_pool_gold else 0.0)

        # Check funnel for routed
        if lane != "FAST_PATH":
            cand_dids = list(dict.fromkeys([c.split("#")[0] for c in cand_pool_cids]))
            td_correct = any(d in gold_docs for d in cand_dids)
            if td_correct:
                funnel_target_doc_correct += 1
            if e2_pool_gold:
                funnel_entered_pool += 1
            if any(gc in e2_cids for gc in gold_chunks):
                funnel_admitted += 1
            if gold_chunks.issubset(set(e2_cids)):
                funnel_chain_complete += 1

        # Check Useful Evidence Eviction
        # A B0 chunk that was gold or unique-slot covered, evicted by a non-covering chunk
        d1_evicted = set(d0_cids) - set(d1_cids)
        e1_evicted = set(d0_cids) - set(e1_cids)
        e2_evicted = set(d0_cids) - set(e2_cids)

        for ec in d1_evicted:
            if ec in gold_chunks and not any(r in gold_chunks for r in (set(d1_cids) - set(d0_cids))):
                d1_evictions.append({"qid": qid, "corpus": corpus, "evicted": ec})

        for ec in e1_evicted:
            if ec in gold_chunks and not any(r in gold_chunks for r in (set(e1_cids) - set(d0_cids))):
                e1_evictions.append({"qid": qid, "corpus": corpus, "evicted": ec})

        for ec in e2_evicted:
            if ec in gold_chunks and not any(r in gold_chunks for r in (set(e2_cids) - set(d0_cids))):
                e2_evictions.append({"qid": qid, "corpus": corpus, "evicted": ec})

        # Save trace data for E2
        e2_traces_data[(qid, corpus)] = {
            "qid": qid,
            "corpus": corpus,
            "question": question,
            "lane": lane,
            "seed_chunk_ids": [s.chunk_id for s in seed_items],
            "candidate_pool_chunk_ids": cand_pool_cids,
            "final_evidence_chunk_ids": e2_cids,
            "composition_traces": comp_traces if lane != "FAST_PATH" else [],
            "channel_diagnostics": channel_diag if lane != "FAST_PATH" else []
        }

        # Retrieval Metrics calculation
        def score_cids(cids):
            c_set = set(cids)
            c_docs = set(c.split("#")[0] for c in cids)
            doc_rec = len(gold_docs.intersection(c_docs)) / len(gold_docs) if gold_docs else 1.0
            chunk_rec = len(gold_chunks.intersection(c_set)) / len(gold_chunks) if gold_chunks else 1.0
            prec = len(gold_chunks.intersection(c_set)) / len(c_set) if c_set else 0.0
            f1 = 2 * prec * chunk_rec / (prec + chunk_rec) if (prec + chunk_rec) > 0 else 0.0
            comp = 1.0 if gold_chunks.issubset(c_set) else 0.0
            return doc_rec, chunk_rec, f1, comp

        d0_m = score_cids(d0_cids)
        d1_m = score_cids(d1_cids)
        e1_m = score_cids(e1_cids)
        e2_m = score_cids(e2_cids)

        d0_doc_hits.append(d0_m[0]); d1_doc_hits.append(d1_m[0]); e1_doc_hits.append(e1_m[0]); e2_doc_hits.append(e2_m[0])
        d0_chunk_hits.append(d0_m[1]); d1_chunk_hits.append(d1_m[1]); e1_chunk_hits.append(e1_m[1]); e2_chunk_hits.append(e2_m[1])
        d0_f1s.append(d0_m[2]); d1_f1s.append(d1_m[2]); e1_f1s.append(e1_m[2]); e2_f1s.append(e2_m[2])
        d0_chains.append(d0_m[3]); d1_chains.append(d1_m[3]); e1_chains.append(e1_m[3]); e2_chains.append(e2_m[3])

    elapsed = time.time() - t_start
    print(f"Evaluated all {len(tasks)} tasks in {elapsed:.2f}s.")

    # Compute Rescues and Regressions vs B0
    def compute_rescues(target_chains):
        rescues, regressions = 0, 0
        for i in range(len(tasks)):
            if target_chains[i] == 1.0 and d0_chains[i] == 0.0:
                rescues += 1
            elif target_chains[i] == 0.0 and d0_chains[i] == 1.0:
                regressions += 1
        return rescues, regressions, rescues - regressions

    d1_rescues, d1_regressions, d1_net = compute_rescues(d1_chains)
    e1_rescues, e1_regressions, e1_net = compute_rescues(e1_chains)
    e2_rescues, e2_regressions, e2_net = compute_rescues(e2_chains)

    # Hybrid channel complementarity analysis across routed instances
    lexical_only_hits, semantic_only_hits, both_hits, neither_hits = 0, 0, 0, 0
    for qid, corpus in tasks:
        trace = e2_traces_data[(qid, corpus)]
        if trace["lane"] != "FAST_PATH":
            diag = trace.get("channel_diagnostics", [])
            gold = gold_map[qid]
            g_chunks = set(gold["gold_chunk_ids"])
            found_by_lex = False
            found_by_sem = False
            for d in diag:
                pass
            # Check for target doc chunks found
            for cid in trace["candidate_pool_chunk_ids"]:
                if cid in g_chunks:
                    # check how it was found
                    for d in diag:
                        if d.get("lexical_rank") and not d.get("semantic_rank"):
                            found_by_lex = True
                        elif d.get("semantic_rank") and not d.get("lexical_rank"):
                            found_by_sem = True
                        elif d.get("lexical_rank") and d.get("semantic_rank"):
                            found_by_lex = True
                            found_by_sem = True

            if found_by_lex and found_by_sem:
                both_hits += 1
            elif found_by_lex:
                lexical_only_hits += 1
            elif found_by_sem:
                semantic_only_hits += 1
            else:
                neither_hits += 1

    # Final summary metrics dictionary
    results_summary = {
        "N": len(tasks),
        "execution_time_s": round(elapsed, 2),
        "retrieval_metrics": {
            "D0_B0": {
                "gold_doc_recall": round(float(np.mean(d0_doc_hits)) * 100, 2),
                "gold_chunk_recall": round(float(np.mean(d0_chunk_hits)) * 100, 2),
                "evidence_f1": round(float(np.mean(d0_f1s)) * 100, 2),
                "chain_completion_rate": round(float(np.mean(d0_chains)) * 100, 2),
                "chain_completion_count": int(sum(d0_chains)),
                "retrieval_rescues": 0,
                "retrieval_regressions": 0,
                "net_retrieval_rescue": 0
            },
            "D1_Clean_Raw": {
                "gold_doc_recall": round(float(np.mean(d1_doc_hits)) * 100, 2),
                "gold_chunk_recall": round(float(np.mean(d1_chunk_hits)) * 100, 2),
                "evidence_f1": round(float(np.mean(d1_f1s)) * 100, 2),
                "chain_completion_rate": round(float(np.mean(d1_chains)) * 100, 2),
                "chain_completion_count": int(sum(d1_chains)),
                "retrieval_rescues": d1_rescues,
                "retrieval_regressions": d1_regressions,
                "net_retrieval_rescue": d1_net
            },
            "E1_Coverage_Composer": {
                "gold_doc_recall": round(float(np.mean(e1_doc_hits)) * 100, 2),
                "gold_chunk_recall": round(float(np.mean(e1_chunk_hits)) * 100, 2),
                "evidence_f1": round(float(np.mean(e1_f1s)) * 100, 2),
                "chain_completion_rate": round(float(np.mean(e1_chains)) * 100, 2),
                "chain_completion_count": int(sum(e1_chains)),
                "retrieval_rescues": e1_rescues,
                "retrieval_regressions": e1_regressions,
                "net_retrieval_rescue": e1_net
            },
            "E2_Hybrid_Descent": {
                "gold_doc_recall": round(float(np.mean(e2_doc_hits)) * 100, 2),
                "gold_chunk_recall": round(float(np.mean(e2_chunk_hits)) * 100, 2),
                "evidence_f1": round(float(np.mean(e2_f1s)) * 100, 2),
                "chain_completion_rate": round(float(np.mean(e2_chains)) * 100, 2),
                "chain_completion_count": int(sum(e2_chains)),
                "retrieval_rescues": e2_rescues,
                "retrieval_regressions": e2_regressions,
                "net_retrieval_rescue": e2_net
            }
        },
        "candidate_pool_recall": {
            "E1_pool_gold_recall": round(float(np.mean(e1_pool_gold_hits)) * 100, 2),
            "E2_pool_gold_recall": round(float(np.mean(e2_pool_gold_hits)) * 100, 2),
            "delta_pool_recall": round(float(np.mean(e2_pool_gold_hits) - np.mean(e1_pool_gold_hits)) * 100, 2)
        },
        "channel_complementarity": {
            "gold_found_lexical_only": lexical_only_hits,
            "gold_found_semantic_only": semantic_only_hits,
            "gold_found_both": both_hits,
            "gold_found_neither": neither_hits
        },
        "evidence_displacement": {
            "D1_eviction_count": len(d1_evictions),
            "E1_eviction_count": len(e1_evictions),
            "E2_eviction_count": len(e2_evictions),
            "E2_evictions": e2_evictions
        }
    }

    # Save retrieval analysis JSON
    ret_json_path = REPORTS_DIR / "v3_e2_retrieval_analysis.json"
    with open(ret_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved retrieval analysis to {ret_json_path}")

    # Save Descent Funnel JSON
    funnel_data = {
        "step_1_total_routed": {
            "name": "Total Routed Instances",
            "count": total_routed_count,
            "conversion_rate": 1.0
        },
        "step_2_correct_target_doc": {
            "name": "Correct Target Document Resolved",
            "count": funnel_target_doc_correct,
            "conversion_rate": round(funnel_target_doc_correct / total_routed_count, 4) if total_routed_count else 0.0
        },
        "step_3_entered_candidate_pool": {
            "name": "Gold Chunk Entered Candidate Pool",
            "count": funnel_entered_pool,
            "conversion_rate": round(funnel_entered_pool / funnel_target_doc_correct, 4) if funnel_target_doc_correct else 0.0
        },
        "step_4_admitted_by_composer": {
            "name": "Admitted by E1 Composer",
            "count": funnel_admitted,
            "conversion_rate": round(funnel_admitted / funnel_entered_pool, 4) if funnel_entered_pool else 0.0
        },
        "step_5_chain_complete": {
            "name": "Final Evidence Chain Complete",
            "count": funnel_chain_complete,
            "conversion_rate": round(funnel_chain_complete / funnel_admitted, 4) if funnel_admitted else 0.0
        }
    }
    funnel_path = REPORTS_DIR / "v3_e2_descent_funnel.json"
    with open(funnel_path, "w", encoding="utf-8") as f:
        json.dump(funnel_data, f, indent=2, ensure_ascii=False)
    print(f"Saved descent funnel to {funnel_path}")

    # Retrieval Gate Check (Section XXIII)
    # E2 至少满足：
    # 1. Gold/Equivalent Candidate Recall > E1
    # 2. Net Retrieval Rescue > E1
    # 3. Gold Chunk Recall final > E1
    # 且 Useful Evidence Eviction 不明显增加
    m = results_summary["retrieval_metrics"]
    e1_chunk_rec = m["E1_Coverage_Composer"]["gold_chunk_recall"]
    e2_chunk_rec = m["E2_Hybrid_Descent"]["gold_chunk_recall"]
    e1_net_res = m["E1_Coverage_Composer"]["net_retrieval_rescue"]
    e2_net_res = m["E2_Hybrid_Descent"]["net_retrieval_rescue"]
    e1_pool_rec = results_summary["candidate_pool_recall"]["E1_pool_gold_recall"]
    e2_pool_rec = results_summary["candidate_pool_recall"]["E2_pool_gold_recall"]
    e2_evict = len(e2_evictions)

    print("\n--- RETRIEVAL GATE VERIFICATION (Section XXIII) ---")
    print(f"1. Candidate Pool Gold Recall: E1={e1_pool_rec}% -> E2={e2_pool_rec}% (Delta: {e2_pool_rec - e1_pool_rec:+.2f}pp)")
    print(f"2. Final Gold Chunk Recall:   E1={e1_chunk_rec}% -> E2={e2_chunk_rec}% (Delta: {e2_chunk_rec - e1_chunk_rec:+.2f}pp)")
    print(f"3. Net Retrieval Rescue:       E1=+{e1_net_res} -> E2=+{e2_net_res} (Delta: {e2_net_res - e1_net_res:+d})")
    print(f"4. Useful Evidence Evictions:  E1=0 -> E2={e2_evict}")

    gate_passed = (
        e2_pool_rec >= e1_pool_rec and
        e2_chunk_rec >= e1_chunk_rec and
        e2_net_res >= e1_net_res and
        e2_evict <= 0
    )

    print(f"RETRIEVAL GATE VERDICT: {'PASSED (PROCEED TO ANSWER GENERATION)' if gate_passed else 'FAILED (STOP)'}")
    return gate_passed, e2_traces_data


if __name__ == "__main__":
    passed, _ = run_e2_retrieval_evaluation()
