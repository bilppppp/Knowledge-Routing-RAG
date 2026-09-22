# src/routing/e2_descent_router.py
"""
Candidate E2: Slot-Conditioned Hybrid Targeted Descent Router System.
Reference: V3 Stage B Specification (Sections XI - XIX).

Architecture:
  - Upstream Control & Data Planes: 100% FROZEN from E1 & C7-Clean.
    - Vector Top-5 FIB
    - Control Plane Shadow Candidate Plane (Top-20 RIB)
    - Document Prefix Aggregation & Selection
    - Lane Detection (FAST_PATH, TEMPORAL_BASIS, COMPOSITE_EVIDENCE)
    - Recursive Parent Lift / Hierarchical Resolution topology
  - Targeted Descent: Replaced with Slot-Conditioned Hybrid Targeted Descent (E2)
    - Dynamic unresolved slot identification based on seed coverage
    - Generic slot-conditioned query synthesis (zero statute lookup tables)
    - Dual-channel in-doc retrieval: Lexical FTS BM25 + Semantic Vector Search
    - Heading category overlap bonus
    - Reciprocal Rank Fusion (RRF) combining lexical and semantic rankings (up to ~4 local candidates)
    - Candidate channel tracking (lexical_only, semantic_only, both)
  - Admission Gate: 100% FROZEN E1 Coverage-Preserving Composition (E1 Composer).
  - Downstream Synthesis: 100% FROZEN B0 Raw Answer Prompt.
"""

import time
import re
from typing import List, Dict, Set, Any, Tuple, Optional
from src.common.models import ExecutionTrace, RoutingStep, EvidenceItem
from src.common.prompt import SYSTEM_PROMPT, pack_evidence_context, format_user_prompt
from src.services.search import SearchService
from src.services.llm import LLMService
from src.graph.lsdb import KnowledgeLSDB
from src.routing.e1_composer_router import E1ComposerRouterSystem
from src.composition.slots import extract_evidence_slots, EvidenceSlot
from src.composition.composer import compose_evidence, compute_set_slot_coverage, compute_set_utility
from src.composition.descent import identify_unresolved_slot, build_generic_descent_query, hybrid_targeted_descent


class E2DescentRouterSystem(E1ComposerRouterSystem):
    """
    E2 Router System:
    Replaces C7/E1 single-channel FTS descent with Slot-Conditioned Hybrid Targeted Descent.
    All other components (global vector search, graph relations, prefix aggregation,
    E1 composer, B0 prompt) are strictly frozen.
    """

    def __init__(
        self,
        search_service: SearchService,
        llm_service: LLMService,
        lsdb: KnowledgeLSDB,
        b0_traces: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
        top_k: int = 5,
        shadow_top_k: int = 20,
        max_hops: int = 2,
        max_evidence_tokens: int = 4000,
        generation_mode: str = "RAW",
        channel_mode: str = "hybrid"  # "hybrid", "lexical_only", "semantic_only"
    ):
        super().__init__(
            search_service=search_service,
            llm_service=llm_service,
            lsdb=lsdb,
            b0_traces=b0_traces,
            top_k=top_k,
            shadow_top_k=shadow_top_k,
            max_hops=max_hops,
            max_evidence_tokens=max_evidence_tokens,
            generation_mode=generation_mode
        )
        self.channel_mode = channel_mode

    def resolve_hierarchical_next_hop(
        self,
        source_chunk: EvidenceItem,
        required_relation: str,
        unresolved_slot_str: str,
        question: str = "",
        seed_items: Optional[List[EvidenceItem]] = None,
        slots: Optional[List[EvidenceSlot]] = None,
        corpus: str = "D20",
        max_targets: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Overrides C7 Clean resolution to use Slot-Conditioned Hybrid Descent.
        """
        parent_doc = source_chunk.doc_id
        if not parent_doc or parent_doc not in self.lsdb.G_routing:
            return None

        target_docs = []
        for tgt in self.lsdb.G_routing.successors(parent_doc):
            edge_data = self.lsdb.G_routing.get_edge_data(parent_doc, tgt)
            if edge_data.get('relation') == required_relation:
                if tgt in self.lsdb.doc_meta and self.lsdb.doc_meta[tgt].get(f'in_{corpus.lower()}', False):
                    target_docs.append(tgt)

        if not target_docs:
            return None

        def score_target(doc_id: str) -> int:
            t = self.lsdb.doc_meta[doc_id].get('title', '')
            import jieba
            return sum(1 for w in jieba.cut(unresolved_slot_str) if len(w) > 1 and w in t)

        target_docs.sort(key=score_target, reverse=True)
        selected_targets = target_docs[:max_targets]

        candidate_items: List[EvidenceItem] = []
        channel_metadata_list: List[Dict[str, Any]] = []

        for t_doc in selected_targets:
            t_title = self.lsdb.doc_meta[t_doc].get('title', '')
            # Use structured slot if available
            if question and seed_items:
                slot_obj, covered_slots = identify_unresolved_slot(
                    question=question,
                    seed_items=seed_items,
                    slots=slots,
                    target_doc_title=t_title
                )
                generic_query = build_generic_descent_query(
                    question=question,
                    unresolved_slot=slot_obj,
                    target_doc_title=t_title,
                    covered_slots=covered_slots
                )
            else:
                slot_obj = EvidenceSlot(slot_id="S_HIER", text=unresolved_slot_str, slot_type="LEGAL_BASIS", keywords=unresolved_slot_str.split())
                generic_query = self._clean_slot_query(unresolved_slot_str)

            hybrid_res = hybrid_targeted_descent(
                search_service=self.search_service,
                target_doc_id=t_doc,
                target_doc_title=t_title,
                unresolved_slot=slot_obj,
                generic_query=generic_query,
                corpus=corpus,
                top_k_candidates=2,
                lsdb=self.lsdb,
                channel_mode=self.channel_mode
            )

            for item, rrf_score, meta in hybrid_res:
                item.source_method = f"e2_hierarchical_{required_relation.lower()}"
                candidate_items.append(item)
                channel_metadata_list.append(meta)

        return {
            "source_chunk": source_chunk.chunk_id,
            "parent_doc": parent_doc,
            "relation": required_relation,
            "selected_targets": selected_targets,
            "slot_query": generic_query,
            "candidate_items": candidate_items,
            "channel_metadata": channel_metadata_list
        }

    def run(self, qid: str, question: str, corpus: str = "D20") -> ExecutionTrace:
        t0 = time.time()
        key = (qid, corpus)

        # 1. Initial Forwarding Table (B0 Vector Search)
        t_ret_0 = time.time()
        seed_items = self.search_service.vector_search(
            query=question,
            corpus=corpus,
            top_k=self.top_k
        )
        retrieval_latency_ms = (time.time() - t_ret_0) * 1000.0

        if not seed_items:
            return ExecutionTrace(
                qid=qid,
                system_id="E2-Descent",
                corpus=corpus,
                question=question,
                generated_answer="未检索到相关法律法规依据。",
                retrieved_chunk_ids=[],
                final_evidence_chunk_ids=[],
                evidence_tokens=0,
                total_latency_ms=(time.time() - t0) * 1000.0,
                llm_calls=0
            )

        # 2. Lane Detection (Frozen from C7-Clean)
        lane = self.detect_lane(question, seed_items)

        # Extract deterministic question slots
        question_slots = extract_evidence_slots(question)

        # FAST PATH: Single-scope query -> Direct Pass-Through
        if lane == "FAST_PATH":
            if key in self.b0_traces:
                b0 = self.b0_traces[key]
                return ExecutionTrace(
                    qid=qid,
                    system_id="E2-Descent",
                    corpus=corpus,
                    question=question,
                    generated_answer=b0["generated_answer"],
                    retrieved_chunk_ids=b0.get("retrieved_chunk_ids", b0.get("final_evidence_chunk_ids", [])),
                    final_evidence_chunk_ids=b0["final_evidence_chunk_ids"],
                    evidence_tokens=b0["evidence_tokens"],
                    routing_steps=[],
                    hub_encounters=0,
                    hub_expansions_prevented=0,
                    feasibility_rejects=0,
                    active_fallbacks=0,
                    retrieval_latency_ms=retrieval_latency_ms,
                    routing_latency_ms=0.0,
                    llm_latency_ms=b0.get("llm_latency_ms", 1500.0),
                    total_latency_ms=(time.time() - t0) * 1000.0,
                    input_tokens=b0.get("input_tokens", 0),
                    output_tokens=b0.get("output_tokens", 0),
                    llm_calls=1,
                    metadata={
                        "generation_mode": "FAST_PATH_PASSTHROUGH",
                        "is_reused_b0_output": True,
                        "question_slots": [s.dict() for s in question_slots],
                        "synthesis_flags": ["FAST_PATH"]
                    }
                )
            else:
                evidence_context = pack_evidence_context(seed_items, max_tokens=self.max_evidence_tokens)
                user_prompt = format_user_prompt(question, evidence_context)
                content, usage, llm_latency_ms = self.llm_service.generate(prompt=user_prompt, system_prompt=SYSTEM_PROMPT)
                retrieved_ids = [s.chunk_id for s in seed_items]
                return ExecutionTrace(
                    qid=qid,
                    system_id="E2-Descent",
                    corpus=corpus,
                    question=question,
                    generated_answer=content,
                    retrieved_chunk_ids=retrieved_ids,
                    final_evidence_chunk_ids=retrieved_ids,
                    evidence_tokens=len(evidence_context) // 2,
                    routing_steps=[],
                    hub_encounters=0,
                    hub_expansions_prevented=0,
                    feasibility_rejects=0,
                    active_fallbacks=0,
                    retrieval_latency_ms=retrieval_latency_ms,
                    routing_latency_ms=0.0,
                    llm_latency_ms=llm_latency_ms,
                    total_latency_ms=(time.time() - t0) * 1000.0,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    llm_calls=1,
                    metadata={
                        "generation_mode": "FAST_PATH_FRESH_RAW",
                        "is_reused_b0_output": False,
                        "question_slots": [s.dict() for s in question_slots],
                        "synthesis_flags": ["FAST_PATH"]
                    }
                )

        # 3. ROUTED PATH: Candidate Generation with E2 Hybrid Targeted Descent
        t_route_0 = time.time()
        routing_steps: List[RoutingStep] = []
        hub_encounters = 0
        feasibility_rejects = 0
        active_fallbacks = 0
        step_idx = 1

        candidate_pool: List[Tuple[EvidenceItem, float, str]] = []
        channel_diagnostics: List[Dict[str, Any]] = []
        primary_doc_id = seed_items[0].doc_id if seed_items else ""

        # LANE A: TEMPORAL_BASIS
        if lane == "TEMPORAL_BASIS":
            for s in seed_items[:3]:
                nbrs = self.lsdb.get_routing_neighbors(s.chunk_id, corpus=corpus, allowed_relations={"SUPERSEDES", "AMENDS"})
                for tgt, rel, cost in nbrs:
                    chk = self.lsdb.get_chunk_evidence(tgt)
                    if chk:
                        it = EvidenceItem(
                            chunk_id=chk["chunk_id"],
                            doc_id=chk["doc_id"],
                            title=chk["title"],
                            heading_path=chk["heading_path"],
                            text=chk["text"],
                            score=0.95,
                            source_method=f"e2_lane_a_{rel.lower()}"
                        )
                        candidate_pool.append((it, 1.2, f"Resolved version via {rel}"))

            if primary_doc_id:
                fts_repeal = self.search_service.fts_search_in_doc("施行 废止 附则", primary_doc_id, corpus=corpus, top_k=1)
                for f_item in fts_repeal:
                    f_item.score = 0.90
                    f_item.source_method = "e2_lane_a_repeal_clause"
                    candidate_pool.append((f_item, 1.1, "Repeal/effective date clause"))

            chunk_basis_found = False
            for s in seed_items[:3]:
                nbrs = self.lsdb.get_routing_neighbors(s.chunk_id, corpus=corpus, allowed_relations={"BASED_ON"})
                for tgt, rel, cost in nbrs:
                    chk = self.lsdb.get_chunk_evidence(tgt)
                    if chk:
                        it = EvidenceItem(
                            chunk_id=chk["chunk_id"],
                            doc_id=chk["doc_id"],
                            title=chk["title"],
                            heading_path=chk["heading_path"],
                            text=chk["text"],
                            score=0.95,
                            source_method="e2_lane_a_based_on"
                        )
                        candidate_pool.append((it, 1.2, "Followed legislative basis"))
                        chunk_basis_found = True

            if re.search(r"(上位|依据|根据|何法|哪两部)", question) and not chunk_basis_found and seed_items:
                unresolved_slot_str = self._extract_unresolved_slot(question)
                max_targets = 2 if re.search(r"(两部|两项|分别)", question) else 1
                hop_res = self.resolve_hierarchical_next_hop(
                    source_chunk=seed_items[0],
                    required_relation="BASED_ON",
                    unresolved_slot_str=unresolved_slot_str,
                    question=question,
                    seed_items=seed_items,
                    slots=question_slots,
                    corpus=corpus,
                    max_targets=max_targets
                )
                if hop_res and hop_res["candidate_items"]:
                    for idx, c_item in enumerate(hop_res["candidate_items"]):
                        candidate_pool.append((
                            c_item,
                            1.4,
                            f"Hierarchical resolution via {hop_res['parent_doc']} --BASED_ON--> {hop_res['selected_targets']}"
                        ))
                        if idx < len(hop_res.get("channel_metadata", [])):
                            channel_diagnostics.append(hop_res["channel_metadata"][idx])

                    routing_steps.append(RoutingStep(
                        step_num=step_idx,
                        action="HIERARCHICAL_RELATION_RESOLUTION",
                        current_node=hop_res["parent_doc"],
                        candidates=[c.chunk_id for c in hop_res["candidate_items"]],
                        feasible_successors=[c.chunk_id for c in hop_res["candidate_items"]],
                        selected_next=hop_res["selected_targets"][0] if hop_res["selected_targets"] else None,
                        cost=2.0,
                        reason=f"PARENT_LIFT: {hop_res['parent_doc']} --BASED_ON--> {hop_res['selected_targets']} -> E2_HYBRID_DESCENT"
                    ))
                    step_idx += 1

        # LANE B: COMPOSITE_EVIDENCE
        elif lane == "COMPOSITE_EVIDENCE":
            has_gap, missing_statutes = self._check_statutory_gap(question, seed_items)
            if missing_statutes:
                target_statute = missing_statutes[0]
                active_fallbacks += 1

                found_by_graph = False
                for s in seed_items:
                    nbrs = self.lsdb.get_routing_neighbors(s.chunk_id, corpus=corpus, allowed_relations={"SUPERSEDES", "AMENDS", "BASED_ON", "REFERENCES"})
                    for tgt, rel, cost in nbrs:
                        if tgt in self.lsdb.doc_meta:
                            d_meta = self.lsdb.doc_meta[tgt]
                            if target_statute in d_meta.get("title", ""):
                                d_chks = self.lsdb.get_document_chunks(tgt, corpus=corpus)
                                if d_chks:
                                    cdata = self.lsdb.get_chunk_evidence(d_chks[0])
                                    if cdata:
                                        it = EvidenceItem(
                                            chunk_id=cdata["chunk_id"],
                                            doc_id=cdata["doc_id"],
                                            title=cdata["title"],
                                            heading_path=cdata["heading_path"],
                                            text=cdata["text"],
                                            score=0.98,
                                            source_method=f"e2_lane_b_graph_gap_{rel.lower()}"
                                        )
                                        candidate_pool.append((it, 1.5, f"Missing statute {target_statute} resolved via graph {rel}"))
                                        found_by_graph = True

                if not found_by_graph and seed_items:
                    for s in seed_items[:2]:
                        hop_res = self.resolve_hierarchical_next_hop(
                            source_chunk=s,
                            required_relation="REFERENCES",
                            unresolved_slot_str=target_statute,
                            question=question,
                            seed_items=seed_items,
                            slots=question_slots,
                            corpus=corpus,
                            max_targets=1
                        )
                        if hop_res and hop_res["candidate_items"]:
                            for idx, c_item in enumerate(hop_res["candidate_items"]):
                                candidate_pool.append((
                                    c_item,
                                    1.45,
                                    f"Hierarchical resolution via {hop_res['parent_doc']} --REFERENCES--> {hop_res['selected_targets']}"
                                ))
                                if idx < len(hop_res.get("channel_metadata", [])):
                                    channel_diagnostics.append(hop_res["channel_metadata"][idx])
                            found_by_graph = True
                            break

            for s in seed_items[:3]:
                nbrs = self.lsdb.get_routing_neighbors(s.chunk_id, corpus=corpus, allowed_relations={"SUPERSEDES", "AMENDS", "BASED_ON", "REFERENCES"})
                for tgt, rel, cost in nbrs:
                    chk = self.lsdb.get_chunk_evidence(tgt)
                    if chk:
                        it = EvidenceItem(
                            chunk_id=chk["chunk_id"],
                            doc_id=chk["doc_id"],
                            title=chk["title"],
                            heading_path=chk["heading_path"],
                            text=chk["text"],
                            score=0.90,
                            source_method=f"e2_lane_b_{rel.lower()}"
                        )
                        candidate_pool.append((it, 1.1, f"Composite expansion via {rel}"))

        # CONTROL PLANE SHADOW CANDIDATE PLANE
        current_docs = set(s.doc_id for s in seed_items).union(set(c[0].doc_id for c in candidate_pool))
        missing_matches = [
            m for m in self._match_question_entities(question, corpus=corpus)
            if m[0] not in current_docs
        ]

        trigger_shadow = len(missing_matches) > 0 or len(candidate_pool) == 0
        shadow_rib_log: List[Dict[str, Any]] = []
        selected_prefixes_log: List[Tuple[str, float, str]] = []
        descent_cids: List[str] = []

        if trigger_shadow:
            shadow_rib, selected_prefixes = self._extract_shadow_prefixes(
                question=question,
                corpus=corpus,
                seed_items=seed_items,
                current_candidate_docs=current_docs
            )
            shadow_rib_log = shadow_rib
            selected_prefixes_log = selected_prefixes

            descent_k = 2 if len(selected_prefixes) >= 2 else 3
            for p_did, p_score, p_title in selected_prefixes:
                # E2: Slot-Conditioned Hybrid Targeted Descent!
                unresolved_slot_obj, covered_slots_list = identify_unresolved_slot(
                    question=question,
                    seed_items=seed_items,
                    slots=question_slots,
                    target_doc_title=p_title
                )
                descent_query = build_generic_descent_query(
                    question=question,
                    unresolved_slot=unresolved_slot_obj,
                    target_doc_title=p_title,
                    covered_slots=covered_slots_list
                )

                hybrid_res = hybrid_targeted_descent(
                    search_service=self.search_service,
                    target_doc_id=p_did,
                    target_doc_title=p_title,
                    unresolved_slot=unresolved_slot_obj,
                    generic_query=descent_query,
                    corpus=corpus,
                    top_k_candidates=descent_k,
                    lsdb=self.lsdb,
                    channel_mode=self.channel_mode
                )

                for d_item, rrf_score, meta in hybrid_res:
                    cand_pool_score = 1.45 if any(m[0] == p_did for m in missing_matches) else 1.35
                    candidate_pool.append((
                        d_item,
                        cand_pool_score,
                        f"Clean Shadow Prefix: {p_did} ({p_title[:20]}) -> E2 Hybrid Descent: {descent_query[:25]}"
                    ))
                    descent_cids.append(d_item.chunk_id)
                    channel_diagnostics.append(meta)

            if selected_prefixes:
                routing_steps.append(RoutingStep(
                    step_num=step_idx,
                    action="SHADOW_ROUTE_PREFIX_RESOLUTION",
                    current_node=primary_doc_id,
                    candidates=[p[0] for p in selected_prefixes],
                    feasible_successors=[p[0] for p in selected_prefixes],
                    selected_next=selected_prefixes[0][0],
                    cost=1.5,
                    reason=f"Resolved external document prefixes: {[p[0] for p in selected_prefixes]} -> Descended via E2 Hybrid into chunks: {descent_cids}"
                ))
                step_idx += 1

        routing_latency_ms = (time.time() - t_route_0) * 1000.0

        # 4. ADMISSION GATE: 100% FROZEN E1 Coverage-Preserving Composer
        final_evidence_items, comp_traces = compose_evidence(
            b0_evidence=seed_items,
            candidate_pool=candidate_pool,
            slots=question_slots,
            max_chunks=self.top_k,
            max_replacements=self.max_replacements,
            lsdb=self.lsdb
        )

        admitted = [t for t in comp_traces if t.get("action") == "REPLACE"]
        rejected = [t for t in comp_traces if t.get("action") == "REJECT"]

        # 5. DOWNSTREAM SYNTHESIS: 100% FROZEN Raw B0 Prompt
        evidence_context = pack_evidence_context(final_evidence_items, max_tokens=self.max_evidence_tokens)
        user_prompt = format_user_prompt(question, evidence_context)
        t_llm_0 = time.time()
        content, usage, llm_latency_ms = self.llm_service.generate(prompt=user_prompt, system_prompt=SYSTEM_PROMPT)

        total_latency_ms = (time.time() - t0) * 1000.0
        final_evidence_ids = [it.chunk_id for it in final_evidence_items]
        retrieved_ids = list(dict.fromkeys([s.chunk_id for s in seed_items] + [c[0].chunk_id for c in candidate_pool]))

        return ExecutionTrace(
            qid=qid,
            system_id="E2-Descent",
            corpus=corpus,
            question=question,
            generated_answer=content,
            retrieved_chunk_ids=retrieved_ids,
            final_evidence_chunk_ids=final_evidence_ids,
            evidence_tokens=len(evidence_context) // 2,
            routing_steps=routing_steps,
            hub_encounters=hub_encounters,
            hub_expansions_prevented=0,
            feasibility_rejects=feasibility_rejects,
            active_fallbacks=active_fallbacks,
            retrieval_latency_ms=retrieval_latency_ms,
            routing_latency_ms=routing_latency_ms,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=total_latency_ms,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            llm_calls=1,
            metadata={
                "generation_mode": "ROUTED_E2_HYBRID_DESCENT",
                "is_reused_b0_output": False,
                "synthesis_flags": ["E2_HYBRID_TARGETED_DESCENT", "E1_COVERAGE_COMPOSITION"],
                "lane": lane,
                "question_slots": [s.dict() for s in question_slots],
                "composition_traces": comp_traces,
                "admitted_chunks": admitted,
                "rejected_chunks": rejected,
                "shadow_rib": shadow_rib_log,
                "selected_prefixes": selected_prefixes_log,
                "descent_chunks": descent_cids,
                "channel_diagnostics": channel_diagnostics
            }
        )
