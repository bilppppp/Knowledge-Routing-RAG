# src/routing/c2_router.py
"""
Candidate C2 Router System: Hierarchical Scope-Aware Router with Diversity Admission & Confidence Bypass
Reference: V2 Directed Search Architecture Specification (Branch B)

Core Architectural Mechanisms:
1. B0 Vector Entrance & Confidence Bypass Gate (Fast Path):
   - Evaluates initial B0 vector search confidence (top1 score, top1-top2 margin, statutory gap).
   - If confidence is high and no cross-statutory gap exists, bypasses graph expansion to preserve baseline precision.
2. Hierarchical Macro/Micro Graph Routing (Dual-Plane Routing):
   - Operates across both chunk-level edges (micro) and document/statute-level citations (macro).
   - Resolves the 93.6% chunk degree bottleneck by allowing inter-statutory routing via parent document nodes.
   - Restricts traversal to typed relations: SUPERSEDES, AMENDS, BASED_ON, REFERENCES.
   - Enforces EIGRP feasibility (loop prevention, hop budget, scope compatibility).
   - Employs soft hub penalty (np.log1p(degree)) rather than hard cuts.
3. Target Statute Vector Guidance:
   - When routing reaches a new statute (target_doc), uses vector search inside target_doc to locate
     the most relevant chunks for the specific query.
4. Diversity-Preserving Context Admission Gate:
   - Limits any single document to at most 2 chunks from initial seeds.
   - Reclaims redundant slots monopolized by single documents (e.g. 5 duplicate chunks from one statute)
     to admit routed cross-statute evidence.
   - Preserves all unique document seeds (e.g. Q105), preventing context displacement.
   - Accommodates up to 6 final evidence chunks (<= 4000 tokens).
5. Deterministic Targeted Gap Fallback:
   - At most one targeted search if an explicit query statute is missing.
"""

import re
import time
import numpy as np
from typing import List, Dict, Set, Any, Tuple, Optional
from src.common.models import ExecutionTrace, RoutingStep, EvidenceItem
from src.common.prompt import SYSTEM_PROMPT, pack_evidence_context, format_user_prompt
from src.services.search import SearchService
from src.services.llm import LLMService
from src.graph.lsdb import KnowledgeLSDB

VALID_RELATIONS = {"SUPERSEDES", "AMENDS", "BASED_ON", "REFERENCES"}

CROSS_RELATION_PATTERN = re.compile(
    r"(依据.+制定|根据.+制定|上位法|与.+衔接|协同衔接|同时.+满足|废止.+法规|废止.+办法|废止.+规定|废止了|分别规定|交叉门槛|法律渊源)"
)

class C2RouterSystem:
    def __init__(
        self,
        search_service: SearchService,
        llm_service: LLMService,
        lsdb: KnowledgeLSDB,
        seed_top_k: int = 5,
        max_per_doc_seeds: int = 2,
        confidence_top1_thresh: float = 0.76,
        confidence_margin_thresh: float = 0.035,
        max_hops: int = 2,
        max_branch_per_node: int = 2,
        max_final_evidence: int = 6,
        max_evidence_tokens: int = 4000
    ):
        self.search_service = search_service
        self.llm_service = llm_service
        self.lsdb = lsdb
        self.seed_top_k = seed_top_k
        self.max_per_doc_seeds = max_per_doc_seeds
        self.confidence_top1_thresh = confidence_top1_thresh
        self.confidence_margin_thresh = confidence_margin_thresh
        self.max_hops = max_hops
        self.max_branch_per_node = max_branch_per_node
        self.max_final_evidence = max_final_evidence
        self.max_evidence_tokens = max_evidence_tokens

    def _extract_query_entities(self, question: str) -> List[str]:
        """
        Extract statutory titles from question enclosed in 《...》 or named regulations.
        """
        matches = re.findall(r"《([^》]+)》", question)
        return [m.strip() for m in matches if len(m.strip()) > 2]

    def _check_statutory_gap(self, question: str, seed_items: List[EvidenceItem]) -> Tuple[bool, List[str]]:
        """
        Detects if an explicit statute title mentioned in question is completely absent from seeds.
        """
        entities = self._extract_query_entities(question)
        if not entities:
            return False, []

        seed_titles = " ".join([it.title for it in seed_items])
        missing = []
        for ent in entities:
            clean_ent = ent.replace("中华人民共和国", "")
            if clean_ent not in seed_titles and ent not in seed_titles:
                missing.append(ent)

        return len(missing) > 0, missing

    def _feasibility_gate(
        self,
        current_node: str,
        target_node: str,
        relation: str,
        path_history: Set[str],
        active_scopes: Set[str],
        current_hop: int
    ) -> Tuple[bool, str]:
        if relation not in VALID_RELATIONS:
            return False, f"Relation '{relation}' not in allowed set"

        if target_node in path_history:
            return False, f"Loop detected: node '{target_node}' already visited"

        if current_hop >= self.max_hops:
            return False, "Hop budget exhausted"

        if relation == "REFERENCES":
            target_scope = self.lsdb.get_scope_prefix(target_node)
            if target_scope and target_scope not in active_scopes:
                if target_node not in self.lsdb.doc_meta:
                    return False, f"Reference target '{target_node}' outside active statutory scopes"

        return True, "Passed Feasibility Gate"

    def run(self, qid: str, question: str, corpus: str = "D20") -> ExecutionTrace:
        t0 = time.time()
        routing_steps: List[RoutingStep] = []
        hub_encounters = 0
        feasibility_rejects = 0
        active_fallbacks = 0

        # 1. Seed Retrieval (B0 Vector Entrance)
        t_ret_0 = time.time()
        seed_items = self.search_service.vector_search(
            query=question,
            corpus=corpus,
            top_k=self.seed_top_k
        )
        retrieval_latency_ms = (time.time() - t_ret_0) * 1000.0

        if not seed_items:
            return ExecutionTrace(
                qid=qid,
                system_id="C2",
                corpus=corpus,
                question=question,
                generated_answer="未检索到相关法律法规依据。",
                retrieved_chunk_ids=[],
                final_evidence_chunk_ids=[],
                evidence_tokens=0,
                total_latency_ms=(time.time() - t0) * 1000.0,
                llm_calls=0
            )

        # 2. Check Confidence & Statutory Gaps & Cross-Relation signals
        has_gap, missing_entities = self._check_statutory_gap(question, seed_items)
        is_cross_relation = bool(CROSS_RELATION_PATTERN.search(question))

        scores = [it.score for it in seed_items]
        top1 = scores[0] if scores else 0.0
        top2 = scores[1] if len(scores) > 1 else top1
        margin = top1 - top2

        is_high_confidence = (
            not is_cross_relation and
            not has_gap and
            top1 >= self.confidence_top1_thresh and
            margin >= self.confidence_margin_thresh
        )

        # FAST PATH: High Confidence Bypass
        if is_high_confidence:
            evidence_context = pack_evidence_context(
                evidence_items=seed_items,
                max_tokens=self.max_evidence_tokens
            )
            user_prompt = format_user_prompt(question, evidence_context)
            content, usage, llm_latency_ms = self.llm_service.generate(
                prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT
            )
            total_latency_ms = (time.time() - t0) * 1000.0
            retrieved_ids = [e.chunk_id for e in seed_items]
            return ExecutionTrace(
                qid=qid,
                system_id="C2",
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
                total_latency_ms=total_latency_ms,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                llm_calls=1
            )

        # 3. ROUTING PATH: Hierarchical Scope Routing
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

        # Frontier contains both chunk nodes (micro) and their parent doc nodes (macro)
        frontier: List[str] = []
        for s in seed_items:
            frontier.append(s.chunk_id)
            doc_id = s.doc_id
            if doc_id and doc_id not in path_history:
                frontier.append(doc_id)
                path_history.add(doc_id)

        step_idx = 1
        routed_items: List[EvidenceItem] = []

        for hop in range(self.max_hops):
            next_frontier: List[str] = []

            for node in frontier:
                # Soft Hub Cost Penalty
                degree = 0
                if self.lsdb.G_routing.has_node(node):
                    degree = self.lsdb.G_routing.out_degree(node)

                hub_penalty = 0.0
                if self.lsdb.is_hub(node) or degree >= 11:
                    hub_encounters += 1
                    hub_penalty = float(np.log1p(degree) * 0.5)

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
                    feasible_candidates.sort(key=lambda x: x[2])
                    selected_targets = [c[0] for c in feasible_candidates[:self.max_branch_per_node]]

                    for tgt in selected_targets:
                        path_history.add(tgt)
                        next_frontier.append(tgt)

                        # Resolution A: Target is a CHUNK
                        chk = self.lsdb.get_chunk_evidence(tgt)
                        if chk and chk["chunk_id"] not in evidence_pool:
                            new_item = EvidenceItem(
                                chunk_id=chk["chunk_id"],
                                doc_id=chk["doc_id"],
                                title=chk["title"],
                                heading_path=chk["heading_path"],
                                text=chk["text"],
                                score=1.0 / (hop + 1.5 + hub_penalty * 0.3),
                                source_method="c2_hierarchical_routing"
                            )
                            evidence_pool[tgt] = new_item
                            routed_items.append(new_item)

                        # Resolution B: Target is a DOCUMENT -> Use vector guidance to find top chunks
                        elif tgt in self.lsdb.doc_meta:
                            doc_top_chunks = self.search_service.vector_search_in_doc(
                                query=question,
                                doc_id=tgt,
                                corpus=corpus,
                                top_k=2
                            )
                            for d_chk in doc_top_chunks:
                                if d_chk.chunk_id not in evidence_pool:
                                    d_chk.score = 0.95 / (hop + 1.5 + hub_penalty * 0.3)
                                    d_chk.source_method = "c2_doc_guided_vector"
                                    evidence_pool[d_chk.chunk_id] = d_chk
                                    routed_items.append(d_chk)

                        # Resolution C: Target is a concept / prefix
                        elif tgt in self.lsdb.G_routing.nodes:
                            doc_id = self.lsdb.G_routing.nodes[tgt].get("doc_id")
                            if doc_id and doc_id in self.lsdb.doc_meta:
                                doc_top_chunks = self.search_service.vector_search_in_doc(
                                    query=question,
                                    doc_id=doc_id,
                                    corpus=corpus,
                                    top_k=2
                                )
                                for d_chk in doc_top_chunks:
                                    if d_chk.chunk_id not in evidence_pool:
                                        d_chk.score = 0.90 / (hop + 1.5 + hub_penalty * 0.3)
                                        d_chk.source_method = "c2_prefix_guided_vector"
                                        evidence_pool[d_chk.chunk_id] = d_chk
                                        routed_items.append(d_chk)

                    routing_steps.append(RoutingStep(
                        step_num=step_idx,
                        action="HIERARCHICAL_ADVANCE",
                        current_node=node,
                        candidates=candidate_names,
                        feasible_successors=[c[0] for c in feasible_candidates],
                        selected_next=selected_targets[0] if selected_targets else None,
                        cost=feasible_candidates[0][2] if feasible_candidates else 2.0,
                        reason=f"Advanced {len(selected_targets)} hierarchical paths (hub_penalty={hub_penalty:.2f})."
                    ))
                    step_idx += 1
                else:
                    if candidate_names:
                        routing_steps.append(RoutingStep(
                            step_num=step_idx,
                            action="FEASIBILITY_BLOCK",
                            current_node=node,
                            candidates=candidate_names,
                            feasible_successors=[],
                            selected_next=None,
                            cost=3.0,
                            reason="All candidates failed feasibility gate."
                        ))
                        step_idx += 1

            frontier = next_frontier
            if not frontier:
                break

        # 4. Targeted Gap Fallback (at most 1 targeted search if statutory gap remains)
        pool_titles = " ".join([it.title for it in evidence_pool.values()])
        unresolved_missing = [m for m in missing_entities if m not in pool_titles]
        if unresolved_missing:
            target_entity = unresolved_missing[0]
            active_fallbacks += 1
            fb_items = self.search_service.fts_search(
                query=f"{target_entity} {question[:30]}",
                corpus=corpus,
                top_k=2
            )
            for fb in fb_items:
                if fb.chunk_id not in evidence_pool:
                    fb.score = 0.85
                    fb.source_method = "c2_targeted_gap_fallback"
                    evidence_pool[fb.chunk_id] = fb
                    routed_items.append(fb)

            routing_steps.append(RoutingStep(
                step_num=step_idx,
                action="TARGETED_GAP_FALLBACK",
                current_node="GLOBAL",
                candidates=[target_entity],
                feasible_successors=[fb.chunk_id for fb in fb_items],
                selected_next=fb_items[0].chunk_id if fb_items else None,
                cost=1.5,
                reason=f"Gap resolved: {target_entity}. Added {len(fb_items)} items."
            ))
            step_idx += 1

        routing_latency_ms = (time.time() - t_route_0) * 1000.0

        # 5. Diversity-Preserving Context Admission Gate
        # Step A: Collect seeds respecting max_per_doc_seeds (default 2)
        # Unique documents are preserved; redundant chunks from the same statute are deferred
        selected_items: List[EvidenceItem] = []
        selected_cids: Set[str] = set()
        doc_counts: Dict[str, int] = {}
        deferred_seeds: List[EvidenceItem] = []

        for s in seed_items:
            d = s.doc_id
            if doc_counts.get(d, 0) < self.max_per_doc_seeds:
                selected_items.append(s)
                selected_cids.add(s.chunk_id)
                doc_counts[d] = doc_counts.get(d, 0) + 1
            else:
                deferred_seeds.append(s)

        # Step B: Fill remaining slots with routed items (sorted by score descending)
        remaining_slots = self.max_final_evidence - len(selected_items)
        routed_items.sort(key=lambda x: x.score, reverse=True)
        for r in routed_items:
            if remaining_slots <= 0:
                break
            if r.chunk_id not in selected_cids:
                selected_items.append(r)
                selected_cids.add(r.chunk_id)
                remaining_slots -= 1

        # Step C: If slots still remain, fill with deferred seeds
        for s in deferred_seeds:
            if remaining_slots <= 0:
                break
            if s.chunk_id not in selected_cids:
                selected_items.append(s)
                selected_cids.add(s.chunk_id)
                remaining_slots -= 1

        final_chunk_ids = [e.chunk_id for e in selected_items]

        # 6. Pack Context & Generate Answer
        evidence_context = pack_evidence_context(
            selected_items,
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
            system_id="C2",
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
