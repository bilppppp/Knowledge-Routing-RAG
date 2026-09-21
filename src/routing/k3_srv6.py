# src/routing/k3_srv6.py
"""
K3: K2 + SRv6-style Segment Routing System
Reference: 实验方案.md Section 8, 16

Core Mechanics:
- Ingress Compiler: Translates natural language Question into an ordered list of fixed Knowledge SIDs:
    K.RESOLVE_VERSION
    K.FOLLOW_REFERENCE
    K.FOLLOW_BASIS
    K.FIND_DEFINITION
    K.CHECK_EXCEPTION
    K.CHECK_SCOPE
    K.VERIFY
- Segment Program Execution:
    Executes segment by segment, directing feasible next hops to match the active SID intent.
- Loop-free and Hub-protected via K2 feasibility gating.
"""

import time
import re
from typing import List, Dict, Set, Any, Tuple, Optional
from src.common.models import ExecutionTrace, RoutingStep, EvidenceItem
from src.common.prompt import SYSTEM_PROMPT, pack_evidence_context, format_user_prompt
from src.services.search import SearchService
from src.services.llm import LLMService
from src.graph.lsdb import KnowledgeLSDB

VALID_RELATIONS = {"SUPERSEDES", "AMENDS", "BASED_ON", "REFERENCES"}

class SRv6RouterSystem:
    def __init__(
        self,
        search_service: SearchService,
        llm_service: LLMService,
        lsdb: KnowledgeLSDB,
        seed_top_k: int = 5,
        max_hops: int = 3,
        max_branch_per_node: int = 2,
        max_final_evidence: int = 6,
        max_evidence_tokens: int = 4000
    ):
        self.search_service = search_service
        self.llm_service = llm_service
        self.lsdb = lsdb
        self.seed_top_k = seed_top_k
        self.max_hops = max_hops
        self.max_branch_per_node = max_branch_per_node
        self.max_final_evidence = max_final_evidence
        self.max_evidence_tokens = max_evidence_tokens

    def compile_segment_plan(self, question: str) -> List[str]:
        """
        Ingress translation of Question into explicit Knowledge SIDs.
        """
        sids = []

        # 1. Version / Supersedes Check
        if any(w in question for w in ["废止", "修订", "修正", "旧办法", "原条例", "历史", "生效", "何时"]):
            sids.append("K.RESOLVE_VERSION")

        # 2. Legislative Basis Check
        if any(w in question for w in ["上位法", "依据", "根据", "母法", "渊源", "制定本"]):
            sids.append("K.FOLLOW_BASIS")

        # 3. Exception / Emergency Check
        if any(w in question for w in ["例外", "除外", "紧急", "突发", "特殊", "豁免", "能否"]):
            sids.append("K.CHECK_EXCEPTION")

        # 4. Scope / Qualification Check
        if any(w in question for w in ["执业范围", "资质", "许可", "独立", "备案", "管辖", "层级", "权限"]):
            sids.append("K.CHECK_SCOPE")

        # 5. Definition / Standard Check
        if any(w in question for w in ["界定", "认定", "定义", "哪些", "情形", "标准"]):
            sids.append("K.FIND_DEFINITION")

        # 6. Reference / Compliance Check
        if any(w in question for w in ["规定", "责任", "处罚", "处理", "要求", "转运", "衔接"]):
            sids.append("K.FOLLOW_REFERENCE")

        # Fallback if no specific keyword matched
        if not sids:
            sids = ["K.FIND_DEFINITION", "K.FOLLOW_REFERENCE"]

        # Ensure terminal verification segment
        sids.append("K.VERIFY")
        return sids

    def run(self, qid: str, question: str, corpus: str = "D20") -> ExecutionTrace:
        t0 = time.time()
        routing_steps: List[RoutingStep] = []
        hub_encounters = 0
        hub_preventions = 0
        feasibility_rejects = 0

        # Ingress Compilation
        segment_plan = self.compile_segment_plan(question)

        # 1. Seed Retrieval
        t_ret_0 = time.time()
        seed_items = self.search_service.vector_search(
            query=question,
            corpus=corpus,
            top_k=self.seed_top_k
        )
        retrieval_latency_ms = (time.time() - t_ret_0) * 1000.0

        t_route_0 = time.time()
        path_history: Set[str] = set()
        active_scopes: Set[str] = set()
        evidence_pool: Dict[str, EvidenceItem] = {}

        for item in seed_items:
            evidence_pool[item.chunk_id] = item
            path_history.add(item.chunk_id)
            scope = self.lsdb.get_scope_prefix(item.chunk_id)
            if scope:
                active_scopes.add(scope)

        # 2. Segment-Programmed Execution
        current_frontier = [item.chunk_id for item in seed_items]
        step_idx = 1

        for sid in segment_plan:
            if sid == "K.VERIFY":
                routing_steps.append(RoutingStep(
                    step_num=step_idx,
                    action="SEGMENT_EXEC: K.VERIFY",
                    current_node="PROGRAM_TERMINAL",
                    candidates=[],
                    feasible_successors=[],
                    selected_next=None,
                    cost=0.0,
                    reason="Reached terminal segment K.VERIFY. Evidence consolidation complete."
                ))
                step_idx += 1
                break

            # Map SID to preferred relation families
            target_relations = set()
            if sid == "K.RESOLVE_VERSION":
                target_relations = {"SUPERSEDES", "AMENDS"}
            elif sid == "K.FOLLOW_BASIS":
                target_relations = {"BASED_ON"}
            elif sid in ["K.FOLLOW_REFERENCE", "K.CHECK_SCOPE", "K.FIND_DEFINITION"]:
                target_relations = {"REFERENCES", "BASED_ON"}
            elif sid == "K.CHECK_EXCEPTION":
                target_relations = {"REFERENCES"}

            next_frontier = []
            for node in current_frontier:
                if self.lsdb.is_hub(node):
                    hub_encounters += 1
                    hub_preventions += 1
                    routing_steps.append(RoutingStep(
                        step_num=step_idx,
                        action=f"HUB_ISOLATION ({sid})",
                        current_node=node,
                        candidates=[],
                        feasible_successors=[],
                        selected_next=None,
                        cost=4.0,
                        reason="Hub isolation: out-degree limit."
                    ))
                    step_idx += 1
                    continue

                raw_neighbors = self.lsdb.get_routing_neighbors(
                    node=node,
                    corpus=corpus,
                    prevent_hub_explosion=True
                )

                # Prioritize neighbors matching active SID
                prioritized = [n for n in raw_neighbors if n[1] in target_relations]
                if not prioritized:
                    prioritized = raw_neighbors

                feasible_candidates = []
                for tgt, rel, cost in prioritized:
                    if tgt in path_history:
                        feasibility_rejects += 1
                        continue
                    feasible_candidates.append((tgt, rel, cost))

                if feasible_candidates:
                    feasible_candidates.sort(key=lambda x: x[2])
                    successor = feasible_candidates[0][0]
                    feasible_successors = [c[0] for c in feasible_candidates[1:self.max_branch_per_node]]
                    selected_paths = [successor] + feasible_successors

                    for tgt in selected_paths:
                        path_history.add(tgt)
                        next_frontier.append(tgt)
                        chk = self.lsdb.get_chunk_evidence(tgt)
                        if chk:
                            evidence_pool[tgt] = EvidenceItem(
                                chunk_id=chk["chunk_id"],
                                doc_id=chk["doc_id"],
                                title=chk["title"],
                                heading_path=chk["heading_path"],
                                text=chk["text"],
                                score=1.2,
                                source_method=f"srv6_{sid.lower()}"
                            )
                        elif tgt in self.lsdb.doc_meta:
                            doc_chunks = self.lsdb.get_document_chunks(tgt, corpus=corpus)
                            if doc_chunks:
                                cid = doc_chunks[0]
                                cdata = self.lsdb.get_chunk_evidence(cid)
                                if cdata and cid not in evidence_pool:
                                    evidence_pool[cid] = EvidenceItem(
                                        chunk_id=cdata["chunk_id"],
                                        doc_id=cdata["doc_id"],
                                        title=cdata["title"],
                                        heading_path=cdata["heading_path"],
                                        text=cdata["text"],
                                        score=0.9,
                                        source_method=f"srv6_doc_{sid.lower()}"
                                    )

                    routing_steps.append(RoutingStep(
                        step_num=step_idx,
                        action=f"SEGMENT_EXEC: {sid}",
                        current_node=node,
                        candidates=[c[0] for c in raw_neighbors],
                        feasible_successors=selected_paths,
                        selected_next=successor,
                        cost=feasible_candidates[0][2],
                        reason=f"Executed {sid} towards successor {successor}."
                    ))
                else:
                    routing_steps.append(RoutingStep(
                        step_num=step_idx,
                        action=f"SEGMENT_BLOCKED: {sid}",
                        current_node=node,
                        candidates=[c[0] for c in raw_neighbors],
                        feasible_successors=[],
                        selected_next=None,
                        cost=0.0,
                        reason=f"No feasible path matching {sid} from node {node}."
                    ))

                step_idx += 1

            if next_frontier:
                current_frontier = next_frontier

        routing_latency_ms = (time.time() - t_route_0) * 1000.0

        # 3. Final Evidence Selection
        sorted_evidence = sorted(evidence_pool.values(), key=lambda x: x.score, reverse=True)[:self.max_final_evidence]
        final_chunk_ids = [e.chunk_id for e in sorted_evidence]

        # 4. Pack Context & Generate
        evidence_context = pack_evidence_context(sorted_evidence, max_tokens=self.max_evidence_tokens)
        user_prompt = format_user_prompt(question, evidence_context)

        content, usage, llm_latency_ms = self.llm_service.generate(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT
        )

        total_latency_ms = (time.time() - t0) * 1000.0

        return ExecutionTrace(
            qid=qid,
            system_id="K3",
            corpus=corpus,
            question=question,
            generated_answer=content,
            retrieved_chunk_ids=[s.chunk_id for s in seed_items],
            final_evidence_chunk_ids=final_chunk_ids,
            evidence_tokens=len(evidence_context) // 2,
            routing_steps=routing_steps,
            hub_encounters=hub_encounters,
            hub_expansions_prevented=hub_preventions,
            feasibility_rejects=feasibility_rejects,
            active_fallbacks=0,
            retrieval_latency_ms=retrieval_latency_ms,
            routing_latency_ms=routing_latency_ms,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=total_latency_ms,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            llm_calls=1,
            metadata={"segment_plan": segment_plan}
        )
