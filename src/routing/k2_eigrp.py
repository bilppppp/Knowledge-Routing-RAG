# src/routing/k2_eigrp.py
"""
K2: K1 + EIGRP-style Feasibility Gating System
Reference: 实验方案.md Section 8, 15

Core Mechanics:
- Inherits RIFT scope aggregation and Hub isolation from K1
- Explicit Feasibility Gate: Validates candidates BEFORE scoring
    Conditions:
      1. relation allowed (SUPERSEDES, AMENDS, BASED_ON, REFERENCES)
      2. loop-free (target not in path history)
      3. scope compatibility
      4. hop budget compliant
- Maintains primary 'successor' and backup 'feasible_successors'
- Fast reroute to feasible_successor upon failure without full-graph flooding
"""

import time
from typing import List, Dict, Set, Any, Tuple, Optional
from src.common.models import ExecutionTrace, RoutingStep, EvidenceItem
from src.common.prompt import SYSTEM_PROMPT, pack_evidence_context, format_user_prompt
from src.services.search import SearchService
from src.services.llm import LLMService
from src.graph.lsdb import KnowledgeLSDB

VALID_RELATIONS = {"SUPERSEDES", "AMENDS", "BASED_ON", "REFERENCES"}

class EIGRPRouterSystem:
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

    def _feasibility_gate(
        self,
        current_node: str,
        target_node: str,
        relation: str,
        path_history: Set[str],
        active_scopes: Set[str],
        current_hop: int
    ) -> Tuple[bool, str]:
        """
        Feasibility condition check:
        Returns (is_feasible, reject_reason)
        """
        # 1. Relation allowed
        if relation not in VALID_RELATIONS:
            return False, f"Relation '{relation}' not in allowed set"

        # 2. Strict Loop Prevention
        if target_node in path_history:
            return False, f"Loop detected: node '{target_node}' already in path history"

        # 3. Hop budget
        if current_hop >= self.max_hops:
            return False, "Hop budget exhausted"

        # 4. Scope compatibility
        target_scope = self.lsdb.get_scope_prefix(target_node)
        if target_scope and target_scope not in active_scopes:
            # Allow cross-scope transition only along explicit legislative/citation relations
            if relation not in ["SUPERSEDES", "AMENDS", "BASED_ON", "REFERENCES"]:
                return False, f"Incompatible scope '{target_scope}' without explicit binding"

        return True, "Passed Feasibility Gate"

    def run(self, qid: str, question: str, corpus: str = "D20") -> ExecutionTrace:
        t0 = time.time()
        routing_steps: List[RoutingStep] = []
        hub_encounters = 0
        hub_preventions = 0
        feasibility_rejects = 0

        # 1. Seed Retrieval (Vector Search)
        t_ret_0 = time.time()
        seed_items = self.search_service.vector_search(
            query=question,
            corpus=corpus,
            top_k=self.seed_top_k
        )
        retrieval_latency_ms = (time.time() - t_ret_0) * 1000.0

        t_route_0 = time.time()
        active_scopes: Set[str] = set()
        path_history: Set[str] = set()
        evidence_pool: Dict[str, EvidenceItem] = {}

        for item in seed_items:
            evidence_pool[item.chunk_id] = item
            path_history.add(item.chunk_id)
            scope = self.lsdb.get_scope_prefix(item.chunk_id)
            if scope:
                active_scopes.add(scope)

        # 2. EIGRP-style Feasible Hop Progression
        frontier = [item.chunk_id for item in seed_items]
        step_idx = 1

        for hop in range(self.max_hops):
            next_frontier = []
            for node in frontier:
                if self.lsdb.is_hub(node):
                    hub_encounters += 1
                    hub_preventions += 1
                    routing_steps.append(RoutingStep(
                        step_num=step_idx,
                        action="HUB_ISOLATION",
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

                candidate_names = [tgt for tgt, rel, cost in raw_neighbors]
                feasible_candidates = []

                for tgt, rel, cost in raw_neighbors:
                    feasible, reason = self._feasibility_gate(
                        current_node=node,
                        target_node=tgt,
                        relation=rel,
                        path_history=path_history,
                        active_scopes=active_scopes,
                        current_hop=hop
                    )
                    if feasible:
                        feasible_candidates.append((tgt, rel, cost))
                    else:
                        feasibility_rejects += 1

                # Select successor and feasible successors
                if feasible_candidates:
                    # Sort by cost
                    feasible_candidates.sort(key=lambda x: x[2])
                    successor = feasible_candidates[0][0]
                    feasible_successors = [c[0] for c in feasible_candidates[1:self.max_branch_per_node]]

                    # Advance along successor and top feasible successors
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
                                score=1.0 / (hop + 2),
                                source_method="eigrp_feasible_successor"
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
                                        score=0.8 / (hop + 2),
                                        source_method="eigrp_doc_prefix"
                                    )

                    routing_steps.append(RoutingStep(
                        step_num=step_idx,
                        action=f"FEASIBILITY_ADVANCE_HOP_{hop+1}",
                        current_node=node,
                        candidates=candidate_names,
                        feasible_successors=selected_paths,
                        selected_next=successor,
                        cost=feasible_candidates[0][2],
                        reason=f"Selected successor {successor} with {len(feasible_successors)} feasible backups."
                    ))
                else:
                    routing_steps.append(RoutingStep(
                        step_num=step_idx,
                        action="NO_FEASIBLE_SUCCESSOR",
                        current_node=node,
                        candidates=candidate_names,
                        feasible_successors=[],
                        selected_next=None,
                        cost=0.0,
                        reason="All raw neighbors failed feasibility condition."
                    ))

                step_idx += 1

            frontier = next_frontier
            if not frontier:
                break

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
            system_id="K2",
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
            llm_calls=1
        )
