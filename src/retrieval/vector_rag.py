# src/retrieval/vector_rag.py
"""
B0: Vector RAG System (Primary Baseline)
Reference: 实验方案.md Section 8, 9, 20
"""

import time
from typing import Optional
from src.common.models import ExecutionTrace
from src.common.prompt import SYSTEM_PROMPT, pack_evidence_context, format_user_prompt
from src.services.search import SearchService
from src.services.llm import LLMService

class VectorRAGSystem:
    def __init__(
        self,
        search_service: SearchService,
        llm_service: LLMService,
        top_k: int = 5,
        max_evidence_tokens: int = 4000
    ):
        self.search_service = search_service
        self.llm_service = llm_service
        self.top_k = top_k
        self.max_evidence_tokens = max_evidence_tokens

    def run(self, qid: str, question: str, corpus: str = "D20") -> ExecutionTrace:
        t0 = time.time()

        # 1. Vector Retrieval
        t_ret_0 = time.time()
        evidence_items = self.search_service.vector_search(
            query=question,
            corpus=corpus,
            top_k=self.top_k
        )
        retrieval_latency_ms = (time.time() - t_ret_0) * 1000.0

        # 2. Pack Evidence Context
        evidence_context = pack_evidence_context(
            evidence_items=evidence_items,
            max_tokens=self.max_evidence_tokens
        )
        user_prompt = format_user_prompt(question, evidence_context)

        # 3. LLM Generation
        content, usage, llm_latency_ms = self.llm_service.generate(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT
        )

        total_latency_ms = (time.time() - t0) * 1000.0
        retrieved_ids = [e.chunk_id for e in evidence_items]

        return ExecutionTrace(
            qid=qid,
            system_id="B0",
            corpus=corpus,
            question=question,
            generated_answer=content,
            retrieved_chunk_ids=retrieved_ids,
            final_evidence_chunk_ids=retrieved_ids,
            evidence_tokens=len(evidence_context) // 2,
            retrieval_latency_ms=retrieval_latency_ms,
            llm_latency_ms=llm_latency_ms,
            total_latency_ms=total_latency_ms,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            llm_calls=1
        )
