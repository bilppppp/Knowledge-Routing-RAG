# src/routing/e1_composer_router.py
"""
Candidate E1: Coverage-Preserving Evidence Composition System.
Reference: V3 Architecture Specification (Sections IX, XVI, XVII, XVIII, XXVIII).

Architecture:
  - Upstream Control & Data Planes: 100% FROZEN from C7-Clean.
    - Vector Top-5 FIB
    - Control Plane Shadow Candidate Plane (Top-20 RIB)
    - Document Prefix Aggregation & Selection
    - Generic Targeted Descent (algorithmic keywords)
    - Recursive Parent Lift / Hierarchical Resolution
  - Admission Gate: Replaced with Coverage-Preserving Set Composition (E1 Composer)
    - Deterministic generic question slot decomposition
    - Unique/dominant slot coverage protection (UNIQUE_COVERAGE_LOCK)
    - Removal cost evaluation (chain disruption, role degradation, redundancy discount)
    - Set utility marginal gain optimization (delta > 0)
    - Full explainable decision trace for every admitted/rejected candidate
  - Downstream Synthesis: 100% FROZEN B0 Prompt (Zero prompt pollution).
"""

import time
import re
from typing import List, Dict, Set, Any, Tuple, Optional
from src.common.models import ExecutionTrace, RoutingStep, EvidenceItem
from src.common.prompt import SYSTEM_PROMPT, pack_evidence_context, format_user_prompt
from src.services.search import SearchService
from src.services.llm import LLMService
from src.graph.lsdb import KnowledgeLSDB
from src.routing.c7_clean_router import C7CleanRouterSystem
from src.composition.slots import extract_evidence_slots
from src.composition.composer import compose_evidence, compute_set_slot_coverage, compute_set_utility


class E1ComposerRouterSystem(C7CleanRouterSystem):
    """
    E1 Router System: Exactly identical to C7CleanRouterSystem in retrieval,
    lane detection, next-hop resolution, shadow prefix resolution, and targeted descent.
    Replaces position-based slot 4/5 replacement with Coverage-Preserving Composition.
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
        generation_mode: str = "RAW"
    ):
        super().__init__(
            search_service=search_service,
            llm_service=llm_service,
            lsdb=lsdb,
            b0_traces=b0_traces,
            top_k=top_k,
            shadow_top_k=shadow_top_k,
            max_hops=max_hops,
            max_replacements=2,  # Not used by E1 composer, but passed to parent
            max_evidence_tokens=max_evidence_tokens,
            generation_mode=generation_mode
        )

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
                system_id="E1-Composer",
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

        # FAST PATH: Single-scope query -> Direct Pass-Through
        if lane == "FAST_PATH":
            if key in self.b0_traces:
                b0 = self.b0_traces[key]
                return ExecutionTrace(
                    qid=qid,
                    system_id="E1-Composer",
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
                        "question_slots": [],
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
                    system_id="E1-Composer",
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
                        "is_reused_b0_output": False
                    }
                )

        # 3. ROUTED PATH: Exactly identical upstream candidate generation to C7-Clean
        t_route_0 = time.time()
        routing_steps: List[RoutingStep] = []
        hub_encounters = 0
        feasibility_rejects = 0
        active_fallbacks = 0
        step_idx = 1

        candidate_pool: List[Tuple[EvidenceItem, float, str]] = []
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
                            source_method=f"clean_lane_a_{rel.lower()}"
                        )
                        candidate_pool.append((it, 1.2, f"Resolved version via {rel}"))

            if primary_doc_id:
                fts_repeal = self.search_service.fts_search_in_doc("施行 废止 附则", primary_doc_id, corpus=corpus, top_k=1)
                for f_item in fts_repeal:
                    f_item.score = 0.90
                    f_item.source_method = "clean_lane_a_repeal_clause"
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
                            source_method="clean_lane_a_based_on"
                        )
                        candidate_pool.append((it, 1.2, "Followed legislative basis"))
                        chunk_basis_found = True

            if re.search(r"(上位|依据|根据|何法|哪两部)", question) and not chunk_basis_found and seed_items:
                unresolved_slot = self._extract_unresolved_slot(question)
                max_targets = 2 if re.search(r"(两部|两项|分别)", question) else 1
                hop_res = self.resolve_hierarchical_next_hop(
                    source_chunk=seed_items[0],
                    required_relation="BASED_ON",
                    unresolved_slot=unresolved_slot,
                    corpus=corpus,
                    max_targets=max_targets
                )
                if hop_res and hop_res["candidate_items"]:
                    for c_item in hop_res["candidate_items"]:
                        candidate_pool.append((
                            c_item,
                            1.4,
                            f"Hierarchical resolution via {hop_res['parent_doc']} --BASED_ON--> {hop_res['selected_targets']}"
                        ))
                    routing_steps.append(RoutingStep(
                        step_num=step_idx,
                        action="HIERARCHICAL_RELATION_RESOLUTION",
                        current_node=hop_res["parent_doc"],
                        candidates=[c.chunk_id for c in hop_res["candidate_items"]],
                        feasible_successors=[c.chunk_id for c in hop_res["candidate_items"]],
                        selected_next=hop_res["selected_targets"][0] if hop_res["selected_targets"] else None,
                        cost=2.0,
                        reason=f"PARENT_LIFT: {hop_res['parent_doc']} --BASED_ON--> {hop_res['selected_targets']} -> TARGETED_DESCENT: {hop_res['slot_query']}"
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
                                            source_method=f"clean_lane_b_graph_gap_{rel.lower()}"
                                        )
                                        candidate_pool.append((it, 1.5, f"Missing statute {target_statute} resolved via graph {rel}"))
                                        found_by_graph = True

                if not found_by_graph and seed_items:
                    for s in seed_items[:2]:
                        hop_res = self.resolve_hierarchical_next_hop(
                            source_chunk=s,
                            required_relation="REFERENCES",
                            unresolved_slot=target_statute,
                            corpus=corpus,
                            max_targets=1
                        )
                        if hop_res and hop_res["candidate_items"]:
                            for c_item in hop_res["candidate_items"]:
                                candidate_pool.append((
                                    c_item,
                                    1.45,
                                    f"Hierarchical resolution via {hop_res['parent_doc']} --REFERENCES--> {hop_res['selected_targets']}"
                                ))
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
                            source_method=f"clean_lane_b_{rel.lower()}"
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

            descent_k = 1 if len(selected_prefixes) >= 2 else 2
            for p_did, p_score, p_title in selected_prefixes:
                descent_query = self._build_generic_descent_query(question, p_title, seed_items)
                descent_items = self.search_service.fts_search_in_doc(
                    query=descent_query,
                    doc_id=p_did,
                    corpus=corpus,
                    top_k=descent_k
                )
                if not descent_items:
                    c1_data = self.lsdb.get_chunk_evidence(f"{p_did}#c001")
                    if c1_data:
                        descent_items = [EvidenceItem(
                            chunk_id=c1_data["chunk_id"],
                            doc_id=c1_data["doc_id"],
                            title=c1_data["title"],
                            heading_path=c1_data["heading_path"],
                            text=c1_data["text"],
                            score=0.92,
                            source_method="clean_shadow_fallback_c001"
                        )]

                for d_item in descent_items:
                    d_item.score = 0.95
                    d_item.source_method = "clean_shadow_prefix_descent"
                    cand_pool_score = 1.45 if any(m[0] == p_did for m in missing_matches) else 1.35
                    candidate_pool.append((
                        d_item,
                        cand_pool_score,
                        f"Clean Shadow Prefix: {p_did} ({p_title[:20]}) -> Generic Descent: {descent_query[:25]}"
                    ))
                    descent_cids.append(d_item.chunk_id)

            if selected_prefixes:
                routing_steps.append(RoutingStep(
                    step_num=step_idx,
                    action="SHADOW_ROUTE_PREFIX_RESOLUTION",
                    current_node=primary_doc_id,
                    candidates=[p[0] for p in selected_prefixes],
                    feasible_successors=[p[0] for p in selected_prefixes],
                    selected_next=selected_prefixes[0][0],
                    cost=1.5,
                    reason=f"Resolved external document prefixes: {[p[0] for p in selected_prefixes]} -> Descended into chunks: {descent_cids}"
                ))
                step_idx += 1

        routing_steps.append(RoutingStep(
            step_num=step_idx,
            action=f"{lane}_COMPLETION",
            current_node=primary_doc_id,
            candidates=[c[0].chunk_id for c in candidate_pool],
            feasible_successors=[c[0].chunk_id for c in candidate_pool],
            selected_next=candidate_pool[0][0].chunk_id if candidate_pool else None,
            cost=1.0,
            reason=f"Routing completed with {len(candidate_pool)} total candidates in pool."
        ))
        step_idx += 1

        routing_latency_ms = (time.time() - t_route_0) * 1000.0

        # 4. COVERAGE-PRESERVING EVIDENCE COMPOSITION (E1 Composer)
        # Deconstruct question slots
        question_slots = extract_evidence_slots(question)

        # Perform set composition starting from initial B0 seed items
        final_evidence, composition_traces = compose_evidence(
            b0_evidence=seed_items,
            candidate_pool=candidate_pool,
            slots=question_slots,
            max_chunks=self.top_k,
            lsdb=self.lsdb
        )

        final_chunk_ids = [e.chunk_id for e in final_evidence]

        # Extract admitted vs rejected candidates for accounting
        admitted = [t for t in composition_traces if t["action"] in {"REPLACE", "ADD"}]
        rejected = [t for t in composition_traces if t["action"] == "REJECT"]
        locked_rejections = [t for t in rejected if "UNIQUE_COVERAGE_LOCK" in t["reason"]]

        # 5. SYNTHESIS GENERATION (100% Frozen B0 Raw Prompt)
        t_gen_0 = time.time()
        evidence_context = pack_evidence_context(
            final_evidence,
            max_tokens=self.max_evidence_tokens
        )
        user_prompt = format_user_prompt(question, evidence_context)
        content, usage, llm_latency_ms = self.llm_service.generate(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT
        )

        metadata: Dict[str, Any] = {
            "synthesis_flags": ["E1_COVERAGE_COMPOSITION"],
            "question_slots": [s.to_dict() for s in question_slots],
            "composition_traces": composition_traces,
            "admitted_chunks": admitted,
            "rejected_chunks": rejected,
            "locked_rejection_count": len(locked_rejections),
            "shadow_rib": shadow_rib_log[:8],
            "selected_prefixes": selected_prefixes_log,
            "descent_chunks": descent_cids,
            "is_reused_b0_output": False
        }

        total_latency_ms = (time.time() - t0) * 1000.0

        return ExecutionTrace(
            qid=qid,
            system_id="E1-Composer",
            corpus=corpus,
            question=question,
            generated_answer=content,
            retrieved_chunk_ids=[s.chunk_id for s in seed_items],
            final_evidence_chunk_ids=final_chunk_ids,
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
            metadata=metadata
        )
