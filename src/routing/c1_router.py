# src/routing/c1_router.py
"""
C1 Router System: Vector Entrance + Scope Narrowing + Feasibility Routing + Targeted Gap Fallback
Reference: V2 Directed Search Architecture Specification (Section III-VI)

Core Mechanics:
1. B0 Vector Entrance: Uses B0 vector retrieval to discover entry seeds.
2. RIFT Scope Narrowing: Anchors search space to active statutory scopes.
3. EIGRP-style Feasibility Gate: Enforces loop-free traversal, typed relations, and scope compatibility.
4. Soft Hub Handling: Replaces hard hub cuts with soft degree-dependent cost penalties and strict typed branching limits.
5. Evidence Sufficiency Check & Targeted Gap Fallback: Deterministic routing with at most ONE targeted fallback on evidence gaps.
6. Anchor Evidence Strategy: Preserves top B0 chunks as guaranteed anchors; remaining slots competed by routing candidates.
"""

import time
import numpy as np
from typing import List, Dict, Set, Any, Tuple, Optional
from src.common.models import ExecutionTrace, RoutingStep, EvidenceItem
from src.common.prompt import SYSTEM_PROMPT, pack_evidence_context, format_user_prompt
from src.services.search import SearchService
from src.services.llm import LLMService
from src.graph.lsdb import KnowledgeLSDB

VALID_RELATIONS = {"SUPERSEDES", "AMENDS", "BASED_ON", "REFERENCES"}

class C1RouterSystem:
    def __init__(
        self,
        search_service: SearchService,
        llm_service: LLMService,
        lsdb: KnowledgeLSDB,
        seed_top_k: int = 5,
        b0_anchor_slots: int = 3,
        max_hops: int = 2,
        max_branch_per_node: int = 2,
        max_final_evidence: int = 6,
        max_evidence_tokens: int = 4000
    ):
        self.search_service = search_service
        self.llm_service = llm_service
        self.lsdb = lsdb
        self.seed_top_k = seed_top_k
        self.b0_anchor_slots = b0_anchor_slots
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
        # 1. Relation allowed
        if relation not in VALID_RELATIONS:
            return False, f"Relation '{relation}' not in allowed set"

        # 2. Strict Loop Prevention
        if target_node in path_history:
            return False, f"Loop detected: node '{target_node}' already visited"

        # 3. Hop budget
        if current_hop >= self.max_hops:
            return False, "Hop budget exhausted"

        # 4. Scope compatibility
        target_scope = self.lsdb.get_scope_prefix(target_node)
        if target_scope and target_scope not in active_scopes:
            # Allow cross-scope transition only along explicit legislative/citation relations
            if relation not in ["SUPERSEDES", "AMENDS", "BASED_ON", "REFERENCES"]:
                return False, f"Incompatible scope '{target_scope}' without explicit legislative binding"

        return True, "Passed Feasibility Gate"

    def _check_evidence_gap(
        self,
        question: str,
        evidence_pool: Dict[str, EvidenceItem],
        active_scopes: Set[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Heuristic Evidence Sufficiency Check:
        Detects if query involves multi-hop/temporal aspects missing in current evidence.
        """
        temporal_keywords = ["修订", "废止", "现行", "替代", "旧", "新", "版本", "生效", "实施"]
        is_temporal = any(kw in question for kw in temporal_keywords)

        has_temporal_rel = False
        for chunk_id in evidence_pool:
            chk = self.lsdb.get_chunk_evidence(chunk_id)
            if chk and any(r in ["AMENDS", "SUPERSEDES"] for _, r, _ in self.lsdb.get_routing_neighbors(chunk_id, prevent_hub_explosion=False)):
                has_temporal_rel = True
                break

        if is_temporal and not has_temporal_rel:
            return {
                "missing": "法规版本演进与修订替代条款",
                "relation_family": ["SUPERSEDES", "AMENDS"],
                "query": f"{question} 修订 废止 替代"
            }

        # Check if query asks about multiple subjects/regulations but evidence is confined to single scope
        if len(active_scopes) == 1 and ("与" in question or "和" in question or "以及" in question):
            return {
                "missing": "关联法条与跨部门协同规范",
                "relation_family": ["REFERENCES", "BASED_ON"],
                "query": question
            }

        # Check if total evidence candidate count is too low (< 3)
        if len(evidence_pool) < 3:
            return {
                "missing": "关键事实佐证切片",
                "relation_family": ["REFERENCES"],
                "query": question
            }

        return None

    def run(self, qid: str, question: str, corpus: str = "D20") -> ExecutionTrace:
        t0 = time.time()
        routing_steps: List[RoutingStep] = []
        hub_encounters = 0
        feasibility_rejects = 0
        active_fallbacks = 0

        # 1. Entrance: B0 Vector Retrieval
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

        # 2. Scope Narrowing & Anchor Retention
        b0_anchors: List[EvidenceItem] = []
        for idx, item in enumerate(seed_items):
            evidence_pool[item.chunk_id] = item
            path_history.add(item.chunk_id)
            scope = self.lsdb.get_scope_prefix(item.chunk_id)
            if scope:
                active_scopes.add(scope)
            if idx < self.b0_anchor_slots:
                b0_anchors.append(item)

        # 3. Feasibility Routing with Soft Hub Handling
        frontier = [item.chunk_id for item in seed_items]
        step_idx = 1

        for hop in range(self.max_hops):
            next_frontier = []
            for node in frontier:
                is_hub = self.lsdb.is_hub(node)
                degree = self.lsdb.G_routing.degree(node) if self.lsdb.G_routing.has_node(node) else 1
                hub_penalty = float(np.log1p(degree)) if is_hub else 0.0

                if is_hub:
                    hub_encounters += 1

                # Typed expansion only (prevent_hub_explosion=False, but restricted by VALID_RELATIONS)
                raw_neighbors = self.lsdb.get_routing_neighbors(
                    node=node,
                    corpus=corpus,
                    allowed_relations=VALID_RELATIONS,
                    prevent_hub_explosion=False
                )

                candidate_names = [tgt for tgt, rel, cost in raw_neighbors]
                feasible_candidates = []

                for tgt, rel, edge_cost in raw_neighbors:
                    feasible, reason = self._feasibility_gate(
                        current_node=node,
                        target_node=tgt,
                        relation=rel,
                        path_history=path_history,
                        active_scopes=active_scopes,
                        current_hop=hop
                    )
                    if feasible:
                        total_cost = edge_cost + hub_penalty
                        feasible_candidates.append((tgt, rel, total_cost))
                    else:
                        feasibility_rejects += 1

                if feasible_candidates:
                    # Sort by total cost ascending
                    feasible_candidates.sort(key=lambda x: x[2])
                    selected_paths = [c[0] for c in feasible_candidates[:self.max_branch_per_node]]

                    for tgt in selected_paths:
                        path_history.add(tgt)
                        next_frontier.append(tgt)

                        # Look up chunk evidence
                        chk = self.lsdb.get_chunk_evidence(tgt)
                        if chk:
                            evidence_pool[tgt] = EvidenceItem(
                                chunk_id=chk["chunk_id"],
                                doc_id=chk["doc_id"],
                                title=chk["title"],
                                heading_path=chk["heading_path"],
                                text=chk["text"],
                                score=1.0 / (hop + 1.5 + hub_penalty * 0.3),
                                source_method="c1_feasible_expansion"
                            )
                        elif tgt in self.lsdb.doc_meta:
                            doc_chunks = self.lsdb.get_document_chunks(tgt, corpus=corpus)
                            if doc_chunks:
                                for cid in doc_chunks[:2]:
                                    cdata = self.lsdb.get_chunk_evidence(cid)
                                    if cdata and cid not in evidence_pool:
                                        evidence_pool[cid] = EvidenceItem(
                                            chunk_id=cdata["chunk_id"],
                                            doc_id=cdata["doc_id"],
                                            title=cdata["title"],
                                            heading_path=cdata["heading_path"],
                                            text=cdata["text"],
                                            score=0.9 / (hop + 1.5 + hub_penalty * 0.3),
                                            source_method="c1_doc_prefix"
                                        )
                        elif tgt in self.lsdb.G_routing.nodes:
                            doc_id = self.lsdb.G_routing.nodes[tgt].get("doc_id")
                            if doc_id and doc_id in self.lsdb.doc_meta:
                                doc_chunks = self.lsdb.get_document_chunks(doc_id, corpus=corpus)
                                if doc_chunks:
                                    for cid in doc_chunks[:2]:
                                        cdata = self.lsdb.get_chunk_evidence(cid)
                                        if cdata and cid not in evidence_pool:
                                            evidence_pool[cid] = EvidenceItem(
                                                chunk_id=cdata["chunk_id"],
                                                doc_id=cdata["doc_id"],
                                                title=cdata["title"],
                                                heading_path=cdata["heading_path"],
                                                text=cdata["text"],
                                                score=0.9 / (hop + 1.5 + hub_penalty * 0.3),
                                                source_method="c1_doc_prefix"
                                            )

                    routing_steps.append(RoutingStep(
                        step_num=step_idx,
                        action=f"C1_FEASIBLE_HOP_{hop+1}",
                        current_node=node,
                        candidates=candidate_names,
                        feasible_successors=selected_paths,
                        selected_next=selected_paths[0],
                        cost=feasible_candidates[0][2],
                        reason=f"Advanced {len(selected_paths)} paths (hub_penalty={hub_penalty:.2f})."
                    ))
                else:
                    routing_steps.append(RoutingStep(
                        step_num=step_idx,
                        action=f"C1_TERMINATE_HOP_{hop+1}",
                        current_node=node,
                        candidates=candidate_names,
                        feasible_successors=[],
                        selected_next=None,
                        cost=0.0,
                        reason="No feasible next-hop candidates."
                    ))

                step_idx += 1

            frontier = next_frontier
            if not frontier:
                break

        # 4. Evidence Sufficiency Check & Targeted Gap Fallback
        gap = self._check_evidence_gap(question, evidence_pool, active_scopes)
        if gap:
            active_fallbacks += 1
            t_gap_query = gap["query"]
            fallback_items = self.search_service.hybrid_search(
                query=t_gap_query,
                corpus=corpus,
                top_k=2
            )
            for fb in fallback_items:
                if fb.chunk_id not in evidence_pool:
                    fb.source_method = "c1_targeted_gap_fallback"
                    fb.score = 0.95  # Priority for targeted gap evidence
                    evidence_pool[fb.chunk_id] = fb
                    path_history.add(fb.chunk_id)

            routing_steps.append(RoutingStep(
                step_num=step_idx,
                action="C1_TARGETED_GAP_FALLBACK",
                current_node="GAP_CONTROLLER",
                candidates=[fb.chunk_id for fb in fallback_items],
                feasible_successors=[fb.chunk_id for fb in fallback_items],
                selected_next=fallback_items[0].chunk_id if fallback_items else None,
                cost=1.5,
                reason=f"Gap identified: {gap['missing']}. Targeted search added {len(fallback_items)} items."
            ))
            step_idx += 1

        routing_latency_ms = (time.time() - t_route_0) * 1000.0

        # 5. Evidence Slot Allocation:
        # Guaranteed B0 Anchors (Slots 1..b0_anchor_slots) + Competitive Routed Candidates (Remaining slots)
        final_evidence_items: List[EvidenceItem] = []
        selected_ids: Set[str] = set()

        # Step A: Place guaranteed B0 anchors first
        for anch in b0_anchors:
            final_evidence_items.append(anch)
            selected_ids.add(anch.chunk_id)

        # Step B: Fill remaining slots with routed candidates
        remaining_slots = self.max_final_evidence - len(final_evidence_items)
        routed_candidates = [
            item for cid, item in evidence_pool.items()
            if cid not in selected_ids and item.source_method in ["c1_feasible_expansion", "c1_doc_prefix", "c1_targeted_gap_fallback"]
        ]
        routed_candidates.sort(key=lambda x: x.score, reverse=True)
        
        assigned_routed = routed_candidates[:remaining_slots]
        final_evidence_items.extend(assigned_routed)
        for item in assigned_routed:
            selected_ids.add(item.chunk_id)

        # Step C: If slots still remain, fill with remaining original B0 seeds
        if len(final_evidence_items) < self.max_final_evidence:
            still_needed = self.max_final_evidence - len(final_evidence_items)
            remaining_seeds = [
                s for s in seed_items
                if s.chunk_id not in selected_ids
            ]
            final_evidence_items.extend(remaining_seeds[:still_needed])

        final_chunk_ids = [e.chunk_id for e in final_evidence_items]

        # 6. Pack Context & Generate Answer
        evidence_context = pack_evidence_context(
            final_evidence_items,
            max_tokens=self.max_evidence_tokens
        )
        user_prompt = format_user_prompt(question, evidence_context)

        content, usage, llm_latency_ms = self.llm_service.generate(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT
        )

        total_latency_ms = (time.time() - t0) * 1000.0

        return ExecutionTrace(
            qid=qid,
            system_id="C1",
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
            llm_calls=1
        )
