#!/usr/bin/env python3
"""
scripts/release_smoke_test.py
Minimum Release Smoke Test for Knowledge-Routing-RAG.

Verifies:
1. All core imports succeed without syntax or dependency errors.
2. Runtime configurations load cleanly (b0_baseline.yaml, v3_frozen.yaml).
3. Embedded Knowledge LSDB loads successfully from SQLite.
4. One B0 Vector RAG query executes cleanly.
5. One V3-Frozen Knowledge Routing query executes cleanly.
6. The final evidence context strictly adheres to the budget: <= 5 chunks.
"""

import sys
import os
import yaml
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_pass(msg: str):
    print(f"  {GREEN}[PASS]{RESET} {msg}")

def print_fail(msg: str):
    print(f"  {RED}[FAIL]{RESET} {msg}")

def main():
    print(f"\n{BLUE}======================================================{RESET}")
    print(f"{BLUE} Knowledge-Routing-RAG — Release Smoke Test {RESET}")
    print(f"{BLUE}======================================================{RESET}")

    # 1. Imports Verification
    try:
        from src.services.search import SearchService
        from src.services.llm import LLMService
        from src.graph.lsdb import KnowledgeLSDB
        from src.retrieval.vector_rag import VectorRAGSystem
        from src.routing.e2_descent_router import E2DescentRouterSystem
        from src.composition.composer import compose_evidence
        from src.composition.descent import hybrid_targeted_descent
        from src.composition.slots import extract_evidence_slots
        from src.common.models import ExecutionTrace
        from src.common.prompt import SYSTEM_PROMPT, pack_evidence_context
        print_pass("Core modules imported successfully.")
    except Exception as e:
        print_fail(f"Import failed: {e}")
        return 1

    # 2. Config Loading
    try:
        b0_cfg_path = PROJECT_ROOT / "configs" / "b0_baseline.yaml"
        v3_cfg_path = PROJECT_ROOT / "configs" / "v3_frozen.yaml"
        with open(b0_cfg_path, "r", encoding="utf-8") as f:
            b0_cfg = yaml.safe_load(f)
        with open(v3_cfg_path, "r", encoding="utf-8") as f:
            v3_cfg = yaml.safe_load(f)
        assert b0_cfg["system"]["system_id"] == "B0"
        assert v3_cfg["system"]["system_id"] == "V3"
        print_pass("Runtime configuration files loaded and validated.")
    except Exception as e:
        print_fail(f"Config loading failed: {e}")
        return 1

    # 3. Knowledge LSDB Database Verification
    try:
        sqlite_path = str(PROJECT_ROOT / "data" / "knowledge_lsdb.sqlite")
        lsdb = KnowledgeLSDB(sqlite_path=sqlite_path)
        doc_count = len(lsdb.doc_meta)
        chunk_count = len(lsdb.chunk_meta)
        assert doc_count == 100, f"Expected 100 documents, got {doc_count}"
        assert chunk_count == 2862, f"Expected 2862 chunks, got {chunk_count}"
        print_pass(f"Knowledge LSDB loaded ({doc_count} documents, {chunk_count} chunks, {lsdb.G_routing.number_of_edges()} routing edges).")
    except Exception as e:
        print_fail(f"Knowledge LSDB initialization failed: {e}")
        return 1

    # Mock LLM for deterministic offline smoke test
    class MockLLMService:
        def generate(self, prompt: str, system_prompt: str):
            return "Smoke test mock answer: requirements met.", {"total_tokens": 120}, 45.0

    mock_llm = MockLLMService()

    # 4. SearchService & B0 Single Query
    try:
        search_service = SearchService()
        b0_system = VectorRAGSystem(
            search_service=search_service,
            llm_service=mock_llm,
            top_k=5,
            max_evidence_tokens=4000
        )
        test_question = "医疗机构执业许可证应当在有效期满前多长时间申请换发？"
        b0_trace = b0_system.run(qid="smoke_b0_001", question=test_question, corpus="D100")
        assert len(b0_trace.retrieved_chunk_ids) <= 5
        print_pass(f"B0 single query executed (retrieved {len(b0_trace.retrieved_chunk_ids)} chunks).")
    except Exception as e:
        print_fail(f"B0 execution failed: {e}")
        return 1

    # 5. V3-Frozen Single Query & Evidence Constraint Verification
    try:
        v3_system = E2DescentRouterSystem(
            search_service=search_service,
            llm_service=mock_llm,
            lsdb=lsdb,
            channel_mode="lexical_only",
            top_k=5,
            shadow_top_k=20,
            max_evidence_tokens=4000
        )
        v3_trace = v3_system.run(qid="smoke_v3_001", question=test_question, corpus="D100")
        n_chunks = len(v3_trace.retrieved_chunk_ids)
        assert n_chunks <= 5, f"V3 evidence violated cap: {n_chunks} chunks"
        assert len(v3_trace.generated_answer) > 0
        print_pass(f"V3-Frozen single query executed (retrieved {n_chunks} chunks, budget constraint <= 5 chunks satisfied).")
    except Exception as e:
        print_fail(f"V3 execution failed: {e}")
        return 1

    print(f"\n{GREEN}All smoke tests passed successfully! Release candidate is ready.{RESET}\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
