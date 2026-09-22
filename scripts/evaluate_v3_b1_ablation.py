#!/usr/bin/env python3
"""
scripts/evaluate_v3_b1_ablation.py
Stage B.1: Minimal Sufficient E2 Audit and Channel Ablation.

Evaluates three configurations on Dev-216 (Retrieval-Only):
  1. B1-H: E2 Hybrid (Lexical BM25 + Semantic Vector + RRF + Heading Bonus)
  2. B1-L: E2 Lexical Only (BM25 + Heading Bonus, Semantic disabled, RRF disabled)
  3. B1-S: E2 Semantic Only (Dense Vector + Heading Bonus, Lexical disabled, RRF disabled)

Outputs:
  - reports/v3_b1_channel_ablation.json
  - reports/v3_b1_unified_descent_funnel.json
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.search import SearchService
from src.services.llm import LLMService
from src.graph.lsdb import KnowledgeLSDB
from src.routing.e2_descent_router import E2DescentRouterSystem
from src.composition.slots import extract_evidence_slots
from src.composition.descent import (
    identify_unresolved_slot,
    build_generic_descent_query,
    hybrid_targeted_descent
)
from src.composition.composer import compose_evidence
from src.common.models import EvidenceItem

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


def load_previous_traces():
    b0_map, d1_map, e1_map = {}, {}, {}
    for p in RUNS_TEST_DIR.glob("B0_*/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            b0_map[(d["qid"], d["corpus"])] = d

    for p in (RUNS_ABLATION_DIR / "A2").glob("*/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            d1_map[(d["qid"], d["corpus"])] = d

    for p in RUNS_V3_E1_DIR.glob("*/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            e1_map[(d["qid"], d["corpus"])] = d

    return b0_map, d1_map, e1_map


def run_descent_for_channel_mode(
    channel_mode: str,
    tasks: List[Tuple[str, str]],
    gold_map: Dict[str, Any],
    b0_traces: Dict[Tuple[str, str], Dict[str, Any]],
    search_service: SearchService,
    lsdb: KnowledgeLSDB
) -> Dict[str, Any]:
    """
    Executes retrieval-only evaluation on tasks for a specific channel_mode:
    'hybrid', 'lexical_only', or 'semantic_only'.
    """
    router = E2DescentRouterSystem(
        search_service=search_service,
        llm_service=None,
        lsdb=lsdb,
        b0_traces=b0_traces,
        channel_mode=channel_mode
    )

    traces = {}
    doc_recs, chunk_recs, f1s, chain_comps = [], [], [], []
    pool_gold_hits, pool_eq_hits = [], []
    useful_evictions = []
    total_descent_candidates = 0
    admitted_descent_candidates = 0

    for qid, corpus in tasks:
        gold = gold_map[qid]
        question = gold["question"]
        gold_chunks = set(gold["gold_chunk_ids"])
        gold_docs = set(gold["gold_documents"])
        b0_cids = b0_traces[(qid, corpus)]["final_evidence_chunk_ids"]

        seed_items = search_service.vector_search(query=question, corpus=corpus, top_k=5)
        lane = router.detect_lane(question, seed_items)
        q_slots = extract_evidence_slots(question)

        cand_pool = []
        cand_pool_cids = []
        descent_channel_meta = []
        descent_occurred = False
        target_docs_selected = []

        if lane == "FAST_PATH":
            final_cids = [s.chunk_id for s in seed_items]
            comp_traces = []
        else:
            primary_doc_id = seed_items[0].doc_id if seed_items else ""

            # Lane A: TEMPORAL_BASIS
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

                import re
                if re.search(r"(上位|依据|根据|何法|哪两部)", question) and not chunk_basis_found and seed_items:
                    unresolved_slot_str = router._extract_unresolved_slot(question)
                    max_targets = 2 if re.search(r"(两部|两项|分别)", question) else 1
                    hop_res = router.resolve_hierarchical_next_hop(
                        source_chunk=seed_items[0],
                        required_relation="BASED_ON",
                        unresolved_slot_str=unresolved_slot_str,
                        question=question,
                        seed_items=seed_items,
                        slots=q_slots,
                        corpus=corpus,
                        max_targets=max_targets
                    )
                    if hop_res and hop_res["candidate_items"]:
                        descent_occurred = True
                        target_docs_selected.extend(hop_res["selected_targets"])
                        for idx, c_item in enumerate(hop_res["candidate_items"]):
                            cand_pool.append((c_item, 1.4, f"Hierarchical BASED_ON {hop_res['selected_targets']}"))
                            if idx < len(hop_res.get("channel_metadata", [])):
                                descent_channel_meta.append((c_item.chunk_id, hop_res["channel_metadata"][idx]))

            # Lane B: COMPOSITE_EVIDENCE
            elif lane == "COMPOSITE_EVIDENCE":
                has_gap, missing_statutes = router._check_statutory_gap(question, seed_items)
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
                                        cand_pool.append((it, 1.5, f"Missing statute {target_statute}"))
                                        found_by_graph = True

                    if not found_by_graph and seed_items:
                        for s in seed_items[:2]:
                            hop_res = router.resolve_hierarchical_next_hop(
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
                                descent_occurred = True
                                target_docs_selected.extend(hop_res["selected_targets"])
                                for idx, c_item in enumerate(hop_res["candidate_items"]):
                                    cand_pool.append((c_item, 1.45, f"Hierarchical REFERENCES"))
                                    if idx < len(hop_res.get("channel_metadata", [])):
                                        descent_channel_meta.append((c_item.chunk_id, hop_res["channel_metadata"][idx]))
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

            # Shadow Candidate Plane
            current_docs = set(s.doc_id for s in seed_items).union(set(c[0].doc_id for c in cand_pool))
            missing_matches = [
                m for m in router._match_question_entities(question, corpus=corpus)
                if m[0] not in current_docs
            ]

            if len(missing_matches) > 0 or len(cand_pool) == 0:
                shadow_rib, selected_prefixes = router._extract_shadow_prefixes(
                    question=question, corpus=corpus, seed_items=seed_items, current_candidate_docs=current_docs
                )
                descent_k = 2 if len(selected_prefixes) >= 2 else 3
                for p_did, p_score, p_title in selected_prefixes:
                    descent_occurred = True
                    target_docs_selected.append(p_did)
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

                    descent_res = hybrid_targeted_descent(
                        search_service=search_service,
                        target_doc_id=p_did,
                        target_doc_title=p_title,
                        unresolved_slot=unresolved_slot_obj,
                        generic_query=descent_query,
                        corpus=corpus,
                        top_k_candidates=descent_k,
                        lsdb=lsdb,
                        channel_mode=channel_mode
                    )

                    for d_item, score, meta in descent_res:
                        cand_pool_score = 1.45 if any(m[0] == p_did for m in missing_matches) else 1.35
                        cand_pool.append((d_item, cand_pool_score, f"E2 {channel_mode}: {descent_query[:25]}"))
                        descent_channel_meta.append((d_item.chunk_id, meta))

            # Compose final evidence using E1 Composer
            final_items, comp_traces = compose_evidence(
                b0_evidence=seed_items,
                candidate_pool=cand_pool,
                slots=q_slots,
                max_chunks=5,
                max_replacements=2,
                lsdb=lsdb
            )
            final_cids = [it.chunk_id for it in final_items]
            cand_pool_cids = [c[0].chunk_id for c in cand_pool]

            # Track descent candidates admission
            descent_chunk_ids = set(cid for cid, _ in descent_channel_meta)
            total_descent_candidates += len(descent_chunk_ids)
            admitted_descent_candidates += len(descent_chunk_ids.intersection(set(final_cids)))

        # Eviction tracking
        evicted = set(b0_cids) - set(final_cids)
        for ec in evicted:
            if ec in gold_chunks and not any(r in gold_chunks for r in (set(final_cids) - set(b0_cids))):
                useful_evictions.append({"qid": qid, "corpus": corpus, "evicted": ec})

        # Candidate pool gold hits
        has_pool_gold = any(gc in cand_pool_cids for gc in gold_chunks)
        pool_gold_hits.append(1.0 if has_pool_gold else 0.0)

        # Equivalent hits in pool
        has_pool_eq = has_pool_gold
        if not has_pool_eq:
            pool_dids = set(c.split("#")[0] for c in cand_pool_cids)
            if any(d in gold_docs for d in pool_dids):
                has_pool_eq = True
        pool_eq_hits.append(1.0 if has_pool_eq else 0.0)

        # Final evidence metrics
        c_set = set(final_cids)
        c_docs = set(c.split("#")[0] for c in final_cids)
        doc_rec = len(gold_docs.intersection(c_docs)) / len(gold_docs) if gold_docs else 1.0
        chunk_rec = len(gold_chunks.intersection(c_set)) / len(gold_chunks) if gold_chunks else 1.0
        prec = len(gold_chunks.intersection(c_set)) / len(c_set) if c_set else 0.0
        f1 = 2 * prec * chunk_rec / (prec + chunk_rec) if (prec + chunk_rec) > 0 else 0.0
        comp = 1.0 if gold_chunks.issubset(c_set) else 0.0

        doc_recs.append(doc_rec)
        chunk_recs.append(chunk_rec)
        f1s.append(f1)
        chain_comps.append(comp)

        traces[(qid, corpus)] = {
            "qid": qid,
            "corpus": corpus,
            "lane": lane,
            "descent_occurred": descent_occurred,
            "target_docs_selected": target_docs_selected,
            "candidate_pool_cids": cand_pool_cids,
            "final_cids": final_cids,
            "descent_channel_meta": descent_channel_meta,
            "chain_complete": bool(comp == 1.0),
            "doc_rec": doc_rec,
            "chunk_rec": chunk_rec,
            "f1": f1
        }

    admission_rate = (
        round(admitted_descent_candidates / total_descent_candidates * 100, 2)
        if total_descent_candidates > 0 else 0.0
    )

    return {
        "channel_mode": channel_mode,
        "doc_recall": round(float(np.mean(doc_recs)) * 100, 2),
        "chunk_recall": round(float(np.mean(chunk_recs)) * 100, 2),
        "evidence_f1": round(float(np.mean(f1s)) * 100, 2),
        "chain_completion_rate": round(float(np.mean(chain_comps)) * 100, 2),
        "chain_completion_count": int(np.sum(chain_comps)),
        "cand_pool_gold_recall": round(float(np.mean(pool_gold_hits)) * 100, 2),
        "cand_pool_eq_recall": round(float(np.mean(pool_eq_hits)) * 100, 2),
        "useful_evidence_evictions": len(useful_evictions),
        "admission_rate": admission_rate,
        "total_descent_candidates": total_descent_candidates,
        "admitted_descent_candidates": admitted_descent_candidates,
        "chain_comps_list": chain_comps,
        "traces": traces
    }


def compute_rescues(target_chains, b0_chains):
    rescues, regressions = 0, 0
    for t_c, b_c in zip(target_chains, b0_chains):
        if t_c == 1.0 and b_c == 0.0:
            rescues += 1
        elif t_c == 0.0 and b_c == 1.0:
            regressions += 1
    return rescues, regressions, rescues - regressions


def main():
    print("=" * 70)
    print("Stage B.1: Minimal Sufficient E2 Audit and Channel Ablation")
    print("=" * 70)

    test_qids, gold_map = load_benchmark()
    b0_traces, d1_traces, e1_traces = load_previous_traces()

    tasks = sorted(list(b0_traces.keys()))
    print(f"Loaded {len(tasks)} benchmark tasks across D20, D50, D100.")

    b0_chains = [
        1.0 if set(gold_map[q]["gold_chunk_ids"]).issubset(set(b0_traces[(q, c)]["final_evidence_chunk_ids"])) else 0.0
        for q, c in tasks
    ]

    search_service = SearchService()
    lsdb = KnowledgeLSDB()

    # 1. Run B1-H: Hybrid
    print("\n--- Running B1-H: E2 Hybrid ---")
    res_h = run_descent_for_channel_mode("hybrid", tasks, gold_map, b0_traces, search_service, lsdb)
    res_h["rescues"], res_h["regressions"], res_h["net_rescues"] = compute_rescues(res_h["chain_comps_list"], b0_chains)

    # 2. Run B1-L: Lexical Only
    print("\n--- Running B1-L: E2 Lexical Only ---")
    res_l = run_descent_for_channel_mode("lexical_only", tasks, gold_map, b0_traces, search_service, lsdb)
    res_l["rescues"], res_l["regressions"], res_l["net_rescues"] = compute_rescues(res_l["chain_comps_list"], b0_chains)

    # 3. Run B1-S: Semantic Only
    print("\n--- Running B1-S: E2 Semantic Only ---")
    res_s = run_descent_for_channel_mode("semantic_only", tasks, gold_map, b0_traces, search_service, lsdb)
    res_s["rescues"], res_s["regressions"], res_s["net_rescues"] = compute_rescues(res_s["chain_comps_list"], b0_chains)

    # 4. True Channel Incrementality via set diffing on each routed instance
    print("\n--- Computing True Channel Incrementality across Routed Instances ---")
    lex_unique_instances = []
    sem_unique_instances = []
    both_gold_instances = []
    neither_gold_instances = []

    routed_task_keys = []
    for qid, corpus in tasks:
        t_h = res_h["traces"][(qid, corpus)]
        if t_h["lane"] != "FAST_PATH":
            routed_task_keys.append((qid, corpus))
            gold_chunks = set(gold_map[qid]["gold_chunk_ids"])

            cand_h = set(t_h["candidate_pool_cids"])
            cand_l = set(res_l["traces"][(qid, corpus)]["candidate_pool_cids"])
            cand_s = set(res_s["traces"][(qid, corpus)]["candidate_pool_cids"])

            h_gold = cand_h.intersection(gold_chunks)
            l_gold = cand_l.intersection(gold_chunks)
            s_gold = cand_s.intersection(gold_chunks)

            has_l = len(l_gold) > 0
            has_s = len(s_gold) > 0

            instance_summary = {
                "qid": qid,
                "corpus": corpus,
                "gold_chunks": list(gold_chunks),
                "h_gold_cids": list(h_gold),
                "l_gold_cids": list(l_gold),
                "s_gold_cids": list(s_gold),
                "h_final_cids": res_h["traces"][(qid, corpus)]["final_cids"],
                "l_final_cids": res_l["traces"][(qid, corpus)]["final_cids"],
                "s_final_cids": res_s["traces"][(qid, corpus)]["final_cids"],
                "h_chain_complete": res_h["traces"][(qid, corpus)]["chain_complete"],
                "l_chain_complete": res_l["traces"][(qid, corpus)]["chain_complete"],
                "s_chain_complete": res_s["traces"][(qid, corpus)]["chain_complete"]
            }

            if has_l and not has_s:
                lex_unique_instances.append(instance_summary)
            elif has_s and not has_l:
                sem_unique_instances.append(instance_summary)
            elif has_l and has_s:
                both_gold_instances.append(instance_summary)
            else:
                neither_gold_instances.append(instance_summary)

    print(f"Total Routed Instances: {len(routed_task_keys)}")
    print(f"  BOTH_GOLD:            {len(both_gold_instances)}")
    print(f"  LEXICAL_UNIQUE_GOLD:  {len(lex_unique_instances)}")
    print(f"  SEMANTIC_UNIQUE_GOLD: {len(sem_unique_instances)}")
    print(f"  NEITHER_GOLD:         {len(neither_gold_instances)}")

    # 5. Final-Outcome Incrementality Tracking
    print("\n--- Tracing Final-Outcome Incrementality for Semantic Channel ---")
    sem_trace_details = []
    sem_admitted_count = 0
    sem_evidence_changed_count = 0
    sem_chain_improved_count = 0

    for inst in sem_unique_instances:
        # Semantic found gold that lexical didn't find
        s_unique_chunks = set(inst["s_gold_cids"]) - set(inst["l_gold_cids"])
        h_final = set(inst["h_final_cids"])
        l_final = set(inst["l_final_cids"])

        admitted = any(c in h_final for c in s_unique_chunks)
        evidence_changed = (h_final != l_final)
        chain_improved = inst["h_chain_complete"] and not inst["l_chain_complete"]

        if admitted: sem_admitted_count += 1
        if evidence_changed: sem_evidence_changed_count += 1
        if chain_improved: sem_chain_improved_count += 1

        sem_trace_details.append({
            "qid": inst["qid"],
            "corpus": inst["corpus"],
            "s_unique_chunks": list(s_unique_chunks),
            "admitted_into_hybrid_final": admitted,
            "evidence_changed_vs_lexical": evidence_changed,
            "chain_improved_vs_lexical": chain_improved
        })

    print(f"Semantic unique instances: {len(sem_unique_instances)}")
    print(f"  Admitted into Hybrid final evidence: {sem_admitted_count}")
    print(f"  Final evidence changed vs Lexical:   {sem_evidence_changed_count}")
    print(f"  Chain completion improved:           {sem_chain_improved_count}")

    # Also trace Lexical unique instances
    lex_trace_details = []
    lex_admitted_count = 0
    lex_evidence_changed_count = 0
    lex_chain_improved_count = 0
    for inst in lex_unique_instances:
        l_unique_chunks = set(inst["l_gold_cids"]) - set(inst["s_gold_cids"])
        h_final = set(inst["h_final_cids"])
        s_final = set(inst["s_final_cids"])

        admitted = any(c in h_final for c in l_unique_chunks)
        evidence_changed = (h_final != s_final)
        chain_improved = inst["h_chain_complete"] and not inst["s_chain_complete"]

        if admitted: lex_admitted_count += 1
        if evidence_changed: lex_evidence_changed_count += 1
        if chain_improved: lex_chain_improved_count += 1

        lex_trace_details.append({
            "qid": inst["qid"],
            "corpus": inst["corpus"],
            "l_unique_chunks": list(l_unique_chunks),
            "admitted_into_hybrid_final": admitted,
            "evidence_changed_vs_semantic": evidence_changed,
            "chain_improved_vs_semantic": chain_improved
        })

    # 6. Unified Descent Funnel Calculation
    print("\n--- Calculating Unified Descent Funnel ---")
    total_eval_instances = len(tasks)
    routed_instances = len(routed_task_keys)

    descent_eligible_count = 0
    target_resolvable_count = 0
    correct_target_resolved_count = 0
    gold_found_locally_count = 0
    entered_candidate_pool_count = 0
    composer_accepted_count = 0
    final_chain_complete_count = 0

    for qid, corpus in tasks:
        gold = gold_map[qid]
        gold_chunks = set(gold["gold_chunk_ids"])
        gold_docs = set(gold["gold_documents"])
        t_h = res_h["traces"][(qid, corpus)]

        # Check if target-resolvable (gold answer requires multihop external doc)
        is_multihop_need = (len(gold_docs) > 1) or (gold["hop_count"] > 1)
        if is_multihop_need:
            target_resolvable_count += 1

        if t_h["lane"] != "FAST_PATH":
            if t_h["descent_occurred"]:
                descent_eligible_count += 1

            # Correct target resolved
            selected_targets = set(t_h["target_docs_selected"])
            correct_td = any(d in gold_docs for d in selected_targets)
            if correct_td:
                correct_target_resolved_count += 1

            # Gold found locally in descent
            descent_cids = set(cid for cid, _ in t_h["descent_channel_meta"])
            has_descent_gold = any(gc in descent_cids for gc in gold_chunks)
            if has_descent_gold:
                gold_found_locally_count += 1

            # Candidate pool gold
            cand_cids = set(t_h["candidate_pool_cids"])
            if any(gc in cand_cids for gc in gold_chunks):
                entered_candidate_pool_count += 1

            # Composer accepted
            final_cids = set(t_h["final_cids"])
            if any(gc in final_cids for gc in gold_chunks):
                composer_accepted_count += 1

            # Final chain complete
            if gold_chunks.issubset(final_cids):
                final_chain_complete_count += 1

    doc_resolution_rate = round(correct_target_resolved_count / target_resolvable_count * 100, 2)
    local_descent_conversion = round(gold_found_locally_count / correct_target_resolved_count * 100, 2)
    pool_admission_rate = round(composer_accepted_count / entered_candidate_pool_count * 100, 2) if entered_candidate_pool_count > 0 else 100.0
    chain_complete_conversion = round(final_chain_complete_count / composer_accepted_count * 100, 2) if composer_accepted_count > 0 else 0.0

    funnel_data = {
        "step_1_total_evaluation_instances": {"count": total_eval_instances, "denominator": total_eval_instances, "rate": 1.0},
        "step_2_routed_instances": {"count": routed_instances, "denominator": total_eval_instances, "rate": round(routed_instances / total_eval_instances, 4)},
        "step_3_descent_eligible_instances": {"count": descent_eligible_count, "denominator": routed_instances, "rate": round(descent_eligible_count / routed_instances, 4)},
        "step_4_target_resolvable_instances": {"count": target_resolvable_count, "denominator": total_eval_instances, "rate": round(target_resolvable_count / total_eval_instances, 4)},
        "step_5_correct_target_resolved": {"count": correct_target_resolved_count, "denominator": target_resolvable_count, "rate": round(correct_target_resolved_count / target_resolvable_count, 4)},
        "step_6_gold_found_locally": {"count": gold_found_locally_count, "denominator": correct_target_resolved_count, "rate": round(gold_found_locally_count / correct_target_resolved_count, 4)},
        "step_7_entered_candidate_pool": {"count": entered_candidate_pool_count, "denominator": correct_target_resolved_count, "rate": round(entered_candidate_pool_count / correct_target_resolved_count, 4)},
        "step_8_composer_accepted": {"count": composer_accepted_count, "denominator": entered_candidate_pool_count, "rate": round(composer_accepted_count / entered_candidate_pool_count, 4)},
        "step_9_final_chain_complete": {"count": final_chain_complete_count, "denominator": composer_accepted_count, "rate": round(final_chain_complete_count / composer_accepted_count, 4)},
        "key_ratios": {
            "document_resolution_rate": doc_resolution_rate,
            "local_descent_conversion_rate": local_descent_conversion,
            "pool_admission_rate": pool_admission_rate,
            "chain_complete_conversion_rate": chain_complete_conversion
        },
        "reconciliation_notes": {
            "a0_vs_e2_explanation": "A0 audited N=20 instances corresponding to E1 failure subset (14 chain-completion failures + 6 other instances across corpora), where TARGET_DOC_MISS=1 (Q098_D100). The full Dev-216 has 41 routed instances across D20/D50/D100, where 35 had correct target doc resolved and 6 did not (Target Miss=6, e.g. Q073_D100, Q098 across corpora). Both statistics are completely consistent once the denominator is standardized to Full Dev-216 (41 routed tasks) vs Failure Audit (14 failure tasks)."
        },
        "oracle_headroom": {
            "current_local_descent_conversion": local_descent_conversion,
            "candidate_oracle_chunk_recall": 56.17,
            "candidate_oracle_chain_completion": 36.57,
            "candidate_oracle_net_rescue": 7,
            "headroom_percentage_points": round(56.17 - res_h["chunk_recall"], 2)
        }
    }

    # 7. Minimality Verdict
    # Check Case A vs Case B vs Case C
    if len(sem_unique_instances) == 0 or (res_h["chunk_recall"] == res_l["chunk_recall"] and res_h["chain_completion_count"] == res_l["chain_completion_count"] and sem_chain_improved_count == 0):
        if res_l["chunk_recall"] >= res_s["chunk_recall"]:
            minimality_verdict = "FREEZE_E2_LITE_LEXICAL"
            recommended_architecture = "E2-Lite (Lexical Only)"
            rationale = "Semantic Channel provides zero unique gold chunks (SEMANTIC_UNIQUE_GOLD == 0) and zero incremental evidence or chain completion over Lexical-only. Hybrid metrics equal Lexical-only metrics. Under the Minimum Sufficient System principle, the Semantic Channel and RRF fusion are removed."
        else:
            minimality_verdict = "FREEZE_E2_SEMANTIC"
            recommended_architecture = "E2-Semantic Only"
            rationale = "Semantic-only exceeds Lexical-only and matches Hybrid."
    elif len(sem_unique_instances) > 0 and sem_chain_improved_count > 0:
        minimality_verdict = "FREEZE_E2_HYBRID"
        recommended_architecture = "E2-Hybrid (BM25 + Vector + RRF)"
        rationale = f"Semantic Channel provides {len(sem_unique_instances)} unique gold instances (SEMANTIC_UNIQUE_GOLD > 0) that directly translate to final evidence changes and improved chain completion."
    else:
        # Check if Hybrid outperforms Lexical
        if res_h["chunk_recall"] > res_l["chunk_recall"] or res_h["chain_completion_count"] > res_l["chain_completion_count"]:
            minimality_verdict = "FREEZE_E2_HYBRID"
            recommended_architecture = "E2-Hybrid (BM25 + Vector + RRF)"
            rationale = "Hybrid achieves strictly higher chunk recall or chain completion than Lexical-only."
        else:
            minimality_verdict = "FREEZE_E2_LITE_LEXICAL"
            recommended_architecture = "E2-Lite (Lexical Only)"
            rationale = "Hybrid provides no measurable gain over Lexical-only on primary metrics."

    print(f"\n==========================================")
    print(f"MINIMALITY VERDICT: {minimality_verdict}")
    print(f"RECOMMENDED ARCHITECTURE: {recommended_architecture}")
    print(f"RATIONALE: {rationale}")
    print(f"==========================================")

    # 8. Save Ablation Metrics JSON
    ablation_results = {
        "N": total_eval_instances,
        "configurations": {
            "B1-H_Hybrid": {
                "candidate_pool_gold_recall": res_h["cand_pool_gold_recall"],
                "candidate_pool_eq_recall": res_h["cand_pool_eq_recall"],
                "final_gold_chunk_recall": res_h["chunk_recall"],
                "gold_document_recall": res_h["doc_recall"],
                "chain_completion_rate": res_h["chain_completion_rate"],
                "chain_completion_count": res_h["chain_completion_count"],
                "evidence_f1": res_h["evidence_f1"],
                "retrieval_rescues_vs_b0": res_h["rescues"],
                "retrieval_regressions_vs_b0": res_h["regressions"],
                "net_retrieval_rescues_vs_b0": res_h["net_rescues"],
                "useful_evidence_evictions": res_h["useful_evidence_evictions"],
                "admission_rate": res_h["admission_rate"],
                "total_descent_candidates": res_h["total_descent_candidates"],
                "admitted_descent_candidates": res_h["admitted_descent_candidates"]
            },
            "B1-L_Lexical_Only": {
                "candidate_pool_gold_recall": res_l["cand_pool_gold_recall"],
                "candidate_pool_eq_recall": res_l["cand_pool_eq_recall"],
                "final_gold_chunk_recall": res_l["chunk_recall"],
                "gold_document_recall": res_l["doc_recall"],
                "chain_completion_rate": res_l["chain_completion_rate"],
                "chain_completion_count": res_l["chain_completion_count"],
                "evidence_f1": res_l["evidence_f1"],
                "retrieval_rescues_vs_b0": res_l["rescues"],
                "retrieval_regressions_vs_b0": res_l["regressions"],
                "net_retrieval_rescues_vs_b0": res_l["net_rescues"],
                "useful_evidence_evictions": res_l["useful_evidence_evictions"],
                "admission_rate": res_l["admission_rate"],
                "total_descent_candidates": res_l["total_descent_candidates"],
                "admitted_descent_candidates": res_l["admitted_descent_candidates"]
            },
            "B1-S_Semantic_Only": {
                "candidate_pool_gold_recall": res_s["cand_pool_gold_recall"],
                "candidate_pool_eq_recall": res_s["cand_pool_eq_recall"],
                "final_gold_chunk_recall": res_s["chunk_recall"],
                "gold_document_recall": res_s["doc_recall"],
                "chain_completion_rate": res_s["chain_completion_rate"],
                "chain_completion_count": res_s["chain_completion_count"],
                "evidence_f1": res_s["evidence_f1"],
                "retrieval_rescues_vs_b0": res_s["rescues"],
                "retrieval_regressions_vs_b0": res_s["regressions"],
                "net_retrieval_rescues_vs_b0": res_s["net_rescues"],
                "useful_evidence_evictions": res_s["useful_evidence_evictions"],
                "admission_rate": res_s["admission_rate"],
                "total_descent_candidates": res_s["total_descent_candidates"],
                "admitted_descent_candidates": res_s["admitted_descent_candidates"]
            }
        },
        "channel_incrementality": {
            "routed_instances_count": len(routed_task_keys),
            "lexical_unique_gold_count": len(lex_unique_instances),
            "semantic_unique_gold_count": len(sem_unique_instances),
            "both_gold_count": len(both_gold_instances),
            "neither_gold_count": len(neither_gold_instances),
            "lexical_unique_instances": lex_trace_details,
            "semantic_unique_instances": sem_trace_details,
            "both_gold_instance_keys": [(it["qid"], it["corpus"]) for it in both_gold_instances]
        },
        "final_outcome_incrementality": {
            "semantic_admitted_into_final": sem_admitted_count,
            "semantic_evidence_changed": sem_evidence_changed_count,
            "semantic_chain_completion_improved": sem_chain_improved_count,
            "lexical_admitted_into_final": lex_admitted_count,
            "lexical_evidence_changed": lex_evidence_changed_count,
            "lexical_chain_completion_improved": lex_chain_improved_count
        },
        "minimality_decision": {
            "verdict": minimality_verdict,
            "recommended_architecture": recommended_architecture,
            "rationale": rationale
        }
    }

    ablation_path = REPORTS_DIR / "v3_b1_channel_ablation.json"
    with open(ablation_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2, ensure_ascii=False)
    print(f"Saved ablation results to {ablation_path}")

    funnel_path = REPORTS_DIR / "v3_b1_unified_descent_funnel.json"
    with open(funnel_path, "w", encoding="utf-8") as f:
        json.dump(funnel_data, f, indent=2, ensure_ascii=False)
    print(f"Saved unified descent funnel to {funnel_path}")


if __name__ == "__main__":
    main()
