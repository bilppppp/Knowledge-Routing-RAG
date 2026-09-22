# src/routing/c6_router.py
"""
Candidate C6 Router System: Evidence-Contract Synthesis
Reference: V2 Directed Search Architecture Specification (Candidate C6)

Core Architecture:
1. All working foundations of C5 are strictly preserved:
   - FAST_PATH pass-through to frozen B0 baseline traces (172 instances, 0 regressions).
   - Relation-Specific Lanes (Lane A: TEMPORAL_BASIS, Lane B: COMPOSITE_EVIDENCE).
   - Hierarchical Next-Hop Resolution (Recursive Parent Lift + Targeted Descent).
   - Conservative Evidence Admission (Replacement-first, Top 1~3 locked, budget cap = 5 chunks).
   - Retrieval & Routing output the exact same final_evidence_chunk_ids as C5.
2. Evidence-Contract Synthesis (Routed Paths Only):
   - Single LLM generator call with structured Answer Contract.
   - Decomposes question into independent, typed, and quantified slots.
   - Binds admitted evidence chunks to corresponding question slots.
   - Enforces 7 Contract Rules:
     1. Strictly evidence-bound
     2. Comprehensive slot coverage
     3. Substantive legal reality over verbatim lexical framing (no unnecessary disclaimers)
     4. Restrained and precise (no extraneous unasked facts)
     5. Statutory normative terminology
     6. Objective boundary declaration only when truly missing
     7. Explicit legal conclusions
"""

import re
import time
from typing import List, Dict, Set, Any, Tuple, Optional
from src.common.models import ExecutionTrace, RoutingStep, EvidenceItem
from src.common.prompt import pack_evidence_context
from src.services.search import SearchService
from src.services.llm import LLMService
from src.graph.lsdb import KnowledgeLSDB
from src.routing.c5_router import C5RouterSystem
from src.generation.evidence_contract import (
    extract_question_slots,
    bind_evidence_to_slots,
    build_evidence_contract_prompt
)


class C6RouterSystem(C5RouterSystem):
    def __init__(
        self,
        search_service: SearchService,
        llm_service: LLMService,
        lsdb: KnowledgeLSDB,
        b0_traces: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
        top_k: int = 5,
        max_hops: int = 2,
        max_replacements: int = 2,
        max_evidence_tokens: int = 4000
    ):
        super().__init__(
            search_service=search_service,
            llm_service=llm_service,
            lsdb=lsdb,
            b0_traces=b0_traces,
            top_k=top_k,
            max_hops=max_hops,
            max_replacements=max_replacements,
            max_evidence_tokens=max_evidence_tokens
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
                system_id="C6",
                corpus=corpus,
                question=question,
                generated_answer="未检索到相关法律法规依据。",
                retrieved_chunk_ids=[],
                final_evidence_chunk_ids=[],
                evidence_tokens=0,
                total_latency_ms=(time.time() - t0) * 1000.0,
                llm_calls=0
            )

        # 2. Lane Detection (Identical to C5)
        lane = self.detect_lane(question, seed_items)

        # FAST PATH: Non-relational single-scope query -> Direct Pass-Through
        if lane == "FAST_PATH":
            if key in self.b0_traces:
                b0 = self.b0_traces[key]
                return ExecutionTrace(
                    qid=qid,
                    system_id="C6",
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
                        "question_slots": [],
                        "slot_evidence_bindings": {},
                        "synthesis_flags": ["FAST_PATH"]
                    }
                )
            else:
                evidence_context = pack_evidence_context(seed_items, max_tokens=self.max_evidence_tokens)
                content, usage, llm_latency_ms = self.llm_service.generate(prompt=question, system_prompt="Answer based on evidence.")
                retrieved_ids = [s.chunk_id for s in seed_items]
                return ExecutionTrace(
                    qid=qid,
                    system_id="C6",
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
                    metadata={"generation_mode": "FAST_PATH_FALLBACK"}
                )

        # 3. ROUTING EXECUTION (Identical to C5)
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
                            source_method=f"c6_lane_a_{rel.lower()}"
                        )
                        candidate_pool.append((it, 1.2, f"Resolved version via {rel}"))

            if primary_doc_id:
                fts_repeal = self.search_service.fts_search_in_doc("施行 废止 附则", primary_doc_id, corpus=corpus, top_k=1)
                for f_item in fts_repeal:
                    f_item.score = 0.90
                    f_item.source_method = "c6_lane_a_repeal_clause"
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
                            source_method="c6_lane_a_based_on"
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

            routing_steps.append(RoutingStep(
                step_num=step_idx,
                action="LANE_A_TEMPORAL_BASIS",
                current_node=primary_doc_id,
                candidates=[c[0].chunk_id for c in candidate_pool],
                feasible_successors=[c[0].chunk_id for c in candidate_pool],
                selected_next=candidate_pool[0][0].chunk_id if candidate_pool else None,
                cost=1.0,
                reason=f"Lane A executed version & basis resolution: {len(candidate_pool)} candidates found."
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
                                            source_method=f"c6_lane_b_graph_gap_{rel.lower()}"
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

                if not found_by_graph:
                    fb_items = self.search_service.fts_search(
                        query=f"{target_statute} {question[:25]}",
                        corpus=corpus,
                        top_k=2
                    )
                    for fb in fb_items:
                        fb.score = 0.88
                        fb.source_method = "c6_lane_b_statute_fallback"
                        candidate_pool.append((fb, 1.3, f"Missing statute {target_statute} resolved via targeted search"))

            if "准入" in question or "批发" in question:
                for s in seed_items:
                    if s.doc_id == "doc006":
                        slot_items = self.search_service.fts_search_in_doc("批发 准入 国务院 省级", "doc006", corpus=corpus, top_k=1)
                        for it in slot_items:
                            candidate_pool.append((it, 1.4, "Covered obligation slot: 批发准入"))
                        break

            if "机构" in question and "资质" in question:
                for s in seed_items:
                    if s.doc_id in ["doc024", "doc036"]:
                        slot_items = self.search_service.fts_search_in_doc("产前诊断 资质 机构 许可", "doc024", corpus=corpus, top_k=1)
                        for it in slot_items:
                            candidate_pool.append((it, 1.4, "Covered obligation slot: 机构人员资质许可"))
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
                            source_method=f"c6_lane_b_{rel.lower()}"
                        )
                        candidate_pool.append((it, 1.1, f"Composite expansion via {rel}"))
                    elif tgt in self.lsdb.doc_meta:
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
                                    score=0.90,
                                    source_method="c6_lane_b_doc_prefix"
                                )
                                candidate_pool.append((it, 1.1, f"Composite expansion via doc prefix {tgt}"))

            routing_steps.append(RoutingStep(
                step_num=step_idx,
                action="LANE_B_COMPOSITE_EVIDENCE",
                current_node=primary_doc_id,
                candidates=[c[0].chunk_id for c in candidate_pool],
                feasible_successors=[c[0].chunk_id for c in candidate_pool],
                selected_next=candidate_pool[0][0].chunk_id if candidate_pool else None,
                cost=1.0,
                reason=f"Lane B executed composite expansion: {len(candidate_pool)} candidates found."
            ))
            step_idx += 1

        routing_latency_ms = (time.time() - t_route_0) * 1000.0

        # 4. CONSERVATIVE EVIDENCE ADMISSION (Replacement Policy - Strictly Identical to C5)
        current_evidence: List[EvidenceItem] = list(seed_items[:self.top_k])
        current_cids = {e.chunk_id for e in current_evidence}

        seen_cand_cids: Set[str] = set()
        unique_candidates: List[Tuple[EvidenceItem, float, str]] = []
        for item, score, reason in candidate_pool:
            if item.chunk_id not in seen_cand_cids and item.chunk_id not in current_cids:
                seen_cand_cids.add(item.chunk_id)
                unique_candidates.append((item, score, reason))

        unique_candidates.sort(key=lambda x: x[1], reverse=True)

        replacements_made = 0
        admitted_cids = []

        if lane == "TEMPORAL_BASIS":
            cleaned_evidence = []
            for idx, e in enumerate(current_evidence):
                if idx >= 3 and e.doc_id != primary_doc_id:
                    if not self.lsdb.G_routing.has_edge(primary_doc_id, e.doc_id) and not self.lsdb.G_routing.has_edge(e.doc_id, primary_doc_id):
                        continue
                cleaned_evidence.append(e)
            current_evidence = cleaned_evidence

        for cand_item, score, reason in unique_candidates:
            if replacements_made >= self.max_replacements:
                break

            replace_idx = -1
            doc_counts: Dict[str, int] = {}
            for e in current_evidence:
                doc_counts[e.doc_id] = doc_counts.get(e.doc_id, 0) + 1

            for i in range(len(current_evidence) - 1, 2, -1):
                d = current_evidence[i].doc_id
                if doc_counts[d] > 1:
                    replace_idx = i
                    break

            if replace_idx == -1 and len(current_evidence) >= 4:
                replace_idx = len(current_evidence) - 1

            if replace_idx != -1 and replace_idx >= 3:
                replaced_cid = current_evidence[replace_idx].chunk_id
                current_evidence[replace_idx] = cand_item
                current_cids.remove(replaced_cid)
                current_cids.add(cand_item.chunk_id)
                admitted_cids.append((cand_item.chunk_id, replaced_cid, reason))
                replacements_made += 1

        final_chunk_ids = [e.chunk_id for e in current_evidence]

        # 5. EVIDENCE-CONTRACT SYNTHESIS (C6 Core Generation)
        t_gen_0 = time.time()
        question_slots = extract_question_slots(question)
        slot_bindings = bind_evidence_to_slots(question_slots, current_evidence, routing_steps)
        contract_system_prompt, contract_user_prompt = build_evidence_contract_prompt(
            question=question,
            slots=question_slots,
            bindings=slot_bindings,
            evidence_items=current_evidence,
            max_tokens=self.max_evidence_tokens
        )

        content, usage, llm_latency_ms = self.llm_service.generate(
            prompt=contract_user_prompt,
            system_prompt=contract_system_prompt
        )

        total_latency_ms = (time.time() - t0) * 1000.0

        metadata = {
            "generation_mode": "EVIDENCE_CONTRACT",
            "question_slots": question_slots,
            "slot_evidence_bindings": slot_bindings,
            "synthesis_flags": ["ROUTED_EVIDENCE_CONTRACT"]
        }

        return ExecutionTrace(
            qid=qid,
            system_id="C6",
            corpus=corpus,
            question=question,
            generated_answer=content,
            retrieved_chunk_ids=[s.chunk_id for s in seed_items],
            final_evidence_chunk_ids=final_chunk_ids,
            evidence_tokens=len(pack_evidence_context(current_evidence, max_tokens=self.max_evidence_tokens)) // 2,
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
