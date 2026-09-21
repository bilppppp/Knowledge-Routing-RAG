# src/routing/k1_rift.py
"""
K1: RIFT-style Routing System
Reference: 实验方案.md Section 8, 12, 13, 14

Core Mechanics:
- Prefix / Document aggregation & scope narrowing
- Directional expansion along validated routing edges (SUPERSEDES, AMENDS, BASED_ON, REFERENCES)
- Hub protection: Prohibits unconstrained neighbors(*) expansion when degree >= threshold
- Caps evidence context under 4000 tokens
"""

import time
from typing import List, Dict, Set, Any, Tuple
from src.common.models import ExecutionTrace, RoutingStep, EvidenceItem
from src.common.prompt import SYSTEM_PROMPT, pack_evidence_context, format_user_prompt
from src.services.search import SearchService
from src.services.llm import LLMService
from src.graph.lsdb import KnowledgeLSDB

class RIFTRouterSystem:
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

    def run(self, qid: str, question: str, corpus: str = "D20") -> ExecutionTrace:
        t0 = time.time()
        routing_steps: List[RoutingStep] = []
        hub_encounters = 0
        hub_preventions = 0

        # 1. Seed Retrieval (Vector Search)
        t_ret_0 = time.time()
        seed_items = self.search_service.vector_search(
            query=question,
            corpus=corpus,
            top_k=self.seed_top_k
        )
        retrieval_latency_ms = (time.time() - t_ret_0) * 1000.0

        t_route_0 = time.time()
        # 2. Scope Narrowing via Route Prefixes
        active_scopes: Set[str] = set()
        visited_nodes: Set[str] = set()
        evidence_pool: Dict[str, EvidenceItem] = {}

        for item in seed_items:
            evidence_pool[item.chunk_id] = item
            visited_nodes.add(item.chunk_id)
            scope = self.lsdb.get_scope_prefix(item.chunk_id)
            if scope:
                active_scopes.add(scope)

        # 3. Directional Routing Expansion with Hub Isolation
        frontier = [item.chunk_id for item in seed_items]
        step_idx = 1

        for hop in range(self.max_hops):
            next_frontier = []
            for node in frontier:
                # Check Hub status
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
                        reason="Degree exceeded threshold: neighbors(*) suppressed by RIFT aggregation."
                    ))
                    step_idx += 1
                    continue

                # Normal directional expansion
                neighbors = self.lsdb.get_routing_neighbors(
                    node=node,
                    corpus=corpus,
                    prevent_hub_explosion=True
                )

                candidates = [tgt for tgt, rel, cost in neighbors if tgt not in visited_nodes]
                selected = candidates[:self.max_branch_per_node]

                for tgt in selected:
                    visited_nodes.add(tgt)
                    next_frontier.append(tgt)
                    # If target is chunk, add to evidence
                    chk = self.lsdb.get_chunk_evidence(tgt)
                    if chk:
                        evidence_pool[tgt] = EvidenceItem(
                            chunk_id=chk["chunk_id"],
                            doc_id=chk["doc_id"],
                            title=chk["title"],
                            heading_path=chk["heading_path"],
                            text=chk["text"],
                            score=1.0 / (hop + 2),
                            source_method="rift_routing"
                        )
                    elif tgt in self.lsdb.doc_meta:
                        # Target is a document: pull its primary preamble/first chunk
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
                                    source_method="rift_doc_prefix"
                                )

                routing_steps.append(RoutingStep(
                    step_num=step_idx,
                    action=f"EXPAND_HOP_{hop+1}",
                    current_node=node,
                    candidates=candidates,
                    feasible_successors=selected,
                    selected_next=selected[0] if selected else None,
                    cost=1.0,
                    reason=f"Expanded {len(selected)} successors within active scope."
                ))
                step_idx += 1

            frontier = next_frontier
            if not frontier:
                break

        routing_latency_ms = (time.time() - t_route_0) * 1000.0

        # 4. Final Evidence Selection (Rank by score, cap at max_final_evidence)
        sorted_evidence = sorted(evidence_pool.values(), key=lambda x: x.score, reverse=True)[:self.max_final_evidence]
        final_chunk_ids = [e.chunk_id for e in sorted_evidence]

        # 5. Pack Context & Generate
        evidence_context = pack_evidence_context(sorted_evidence, max_tokens=self.max_evidence_tokens)
        user_prompt = format_user_prompt(question, evidence_context)

        content, usage, llm_latency_ms = self.llm_service.generate(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT
        )

        total_latency_ms = (time.time() - t0) * 1000.0

        return ExecutionTrace(
            qid=qid,
            system_id="K1",
            corpus=corpus,
            question=question,
            generated_answer=content,
            retrieved_chunk_ids=[s.chunk_id for s in seed_items],
            final_evidence_chunk_ids=final_chunk_ids,
            evidence_tokens=len(evidence_context) // 2,
            routing_steps=routing_steps,
            hub_encounters=hub_encounters,
            hub_expansions_prevented=hub_preventions,
            feasibility_rejects=0,
            active_fallbacks=0,
            retrieval_latency_ms=retrieval_latency_ms,
            routing_latency_ms=routing_latency_ms,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=total_latency_ms,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            llm_calls=1
        )
