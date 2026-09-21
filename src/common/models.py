# src/common/models.py
"""
Pydantic Data Models for Knowledge Routing RAG.
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    section_id: str
    heading_path: str
    order_num: int
    text: str
    char_count: int
    in_d20: bool
    in_d50: bool
    in_d100: bool

class EvidenceItem(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    heading_path: str
    text: str
    score: float = 0.0
    source_method: str = "vector"  # vector, fts, graph_step, fallback
    step_info: Optional[Dict[str, Any]] = None

class RoutingStep(BaseModel):
    step_num: int
    action: str
    current_node: str
    candidates: List[str] = Field(default_factory=list)
    feasible_successors: List[str] = Field(default_factory=list)
    selected_next: Optional[str] = None
    cost: float = 0.0
    reason: str = ""

class ExecutionTrace(BaseModel):
    qid: str
    system_id: str  # B0, B1, K1, K2, K3, K4
    corpus: str     # D20, D50, D100
    question: str
    generated_answer: str = ""
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    final_evidence_chunk_ids: List[str] = Field(default_factory=list)
    evidence_tokens: int = 0
    routing_steps: List[RoutingStep] = Field(default_factory=list)
    hub_encounters: int = 0
    hub_expansions_prevented: int = 0
    feasibility_rejects: int = 0
    active_fallbacks: int = 0
    retrieval_latency_ms: float = 0.0
    routing_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
