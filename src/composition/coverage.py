# src/composition/coverage.py
"""
Slot Coverage and Evidence Relevance Module.
Reference: V3 Architecture Specification (Sections XIV, XVI, XVIII).

Calculates deterministic slot coverage, unique coverage locks, and redundancy
for evidence chunks and candidate sets.
Zero LLM calls; pure deterministic signals.
"""

import re
from typing import List, Dict, Set, Any, Tuple, Optional
from src.common.models import EvidenceItem
from src.composition.slots import EvidenceSlot
from src.composition.roles import infer_evidence_role

COVERAGE_THRESHOLD = 0.35
DOMINANT_MARGIN = 0.20


def estimate_chunk_slot_coverage(
    chunk: EvidenceItem,
    slots: List[EvidenceSlot],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Computes deterministic relevance of a chunk to each question slot in [0.0, 1.0].
    Signals:
      1. Lexical keyword recall (chunk title + heading + text vs slot keywords)
      2. Role alignment (e.g. EXCEPTION chunk matching EXCEPTION slot)
      3. Statutory / entity alignment (slot entity matching chunk doc title)
      4. Retrieval confidence prior
    """
    role = infer_evidence_role(chunk, slots)
    search_text = f"{chunk.title} {chunk.heading_path} {chunk.text}".lower()

    scores: Dict[str, float] = {}
    for slot in slots:
        kws = slot.keywords
        if not kws:
            scores[slot.slot_id] = 0.10
            continue

        # 1. Lexical overlap
        hits = sum(1 for kw in kws if kw.lower() in search_text)
        lexical_recall = hits / max(1, len(kws))

        # 2. Role alignment bonus
        role_bonus = 0.0
        if slot.slot_type == "EXCEPTION" and role == "EXCEPTION":
            role_bonus = 0.30
        elif slot.slot_type == "PENALTY_SANCTION" and role == "SANCTION":
            role_bonus = 0.25
        elif slot.slot_type == "REQUIREMENT_CONDITION" and role in {"CONDITION", "DIRECT"}:
            role_bonus = 0.20
        elif slot.slot_type == "LEGAL_BASIS" and role in {"BASIS", "BRIDGE"}:
            role_bonus = 0.30
        elif slot.slot_type == "TIMELINE_PROCEDURE" and any(t in search_text for t in ["日", "月", "年", "小时", "期", "程序"]):
            role_bonus = 0.20
        elif role == "DIRECT" and lexical_recall > 0.20:
            role_bonus = 0.15

        # 3. Entity match bonus (damped if no substantive keyword recall or role match)
        entity_bonus = 0.0
        if lexical_recall > 0.10 or role_bonus > 0.0:
            for ent in slot.target_entities:
                clean_ent = ent.replace("中华人民共和国", "")
                if clean_ent in chunk.title or ent in chunk.title:
                    entity_bonus = 0.15
                    break

        # 4. Source method prior
        source_bonus = 0.0
        method = (chunk.source_method or "").lower()
        if "based_on" in method and slot.slot_type == "LEGAL_BASIS":
            source_bonus = 0.20
        elif "repeal" in method and slot.slot_type == "LEGAL_BASIS":
            source_bonus = 0.20
        elif "gap" in method and entity_bonus > 0:
            source_bonus = 0.15

        # 5. Composite score
        composite = (lexical_recall * 0.55) + role_bonus + entity_bonus + source_bonus
        if chunk.score > 0:
            composite += min(chunk.score, 1.0) * 0.05

        scores[slot.slot_id] = round(min(1.0, composite), 4)

    return scores


def compute_set_slot_coverage(
    evidence_set: List[EvidenceItem],
    slots: List[EvidenceSlot]
) -> Dict[str, Any]:
    """
    Computes comprehensive coverage status for a candidate evidence set S:
      - slot_covering_chunks: Map[slot_id -> List[chunk_id]]
      - covered_slots: List of slots covered by at least one chunk (score >= COVERAGE_THRESHOLD)
      - unique_coverage_map: Map[chunk_id -> List[slot_id]] (slots uniquely or dominantly covered)
      - locked_chunks: Set of chunk_ids with UNIQUE_COVERAGE_LOCK
    """
    chunk_scores: Dict[str, Dict[str, float]] = {}
    for item in evidence_set:
        chunk_scores[item.chunk_id] = estimate_chunk_slot_coverage(item, slots)

    slot_covering: Dict[str, List[str]] = {s.slot_id: [] for s in slots}
    for item in evidence_set:
        cov = chunk_scores[item.chunk_id]
        for s_id, score in cov.items():
            if score >= COVERAGE_THRESHOLD:
                slot_covering[s_id].append(item.chunk_id)

    covered_slots = [s_id for s_id, cids in slot_covering.items() if len(cids) > 0]
    uncovered_slots = [s_id for s_id, cids in slot_covering.items() if len(cids) == 0]
    coverage_rate = len(covered_slots) / max(1, len(slots))

    # Identify unique or dominant coverage per slot
    unique_coverage_map: Dict[str, List[str]] = {item.chunk_id: [] for item in evidence_set}
    for s in slots:
        s_id = s.slot_id
        covering_cids = slot_covering[s_id]
        if len(covering_cids) == 1:
            unique_coverage_map[covering_cids[0]].append(s_id)
        elif len(covering_cids) > 1:
            # Check for dominant coverage
            sorted_cids = sorted(covering_cids, key=lambda c: chunk_scores[c][s_id], reverse=True)
            top_cid = sorted_cids[0]
            second_cid = sorted_cids[1]
            top_score = chunk_scores[top_cid][s_id]
            second_score = chunk_scores[second_cid][s_id]
            if top_score - second_score >= DOMINANT_MARGIN and top_score >= 0.50:
                unique_coverage_map[top_cid].append(s_id)

    locked_chunks = {cid for cid, s_ids in unique_coverage_map.items() if len(s_ids) > 0}

    return {
        "slot_covering": slot_covering,
        "covered_slots": covered_slots,
        "uncovered_slots": uncovered_slots,
        "coverage_rate": coverage_rate,
        "chunk_scores": chunk_scores,
        "unique_coverage_map": unique_coverage_map,
        "locked_chunks": locked_chunks
    }


def compute_set_redundancy(
    evidence_set: List[EvidenceItem],
    slots: List[EvidenceSlot]
) -> float:
    """
    Calculates evidence redundancy penalty.
    Penalizes sets where multiple chunks from the same document cover the same slot
    without adding independent slot coverage or novel role value.
    """
    if len(evidence_set) <= 1:
        return 0.0

    redundancy = 0.0
    doc_chunks: Dict[str, List[EvidenceItem]] = {}
    for item in evidence_set:
        doc_chunks.setdefault(item.doc_id, []).append(item)

    for doc_id, items in doc_chunks.items():
        if len(items) > 1:
            for i in range(len(items)):
                cov_i = estimate_chunk_slot_coverage(items[i], slots)
                slots_i = {s for s, sc in cov_i.items() if sc >= COVERAGE_THRESHOLD}
                for j in range(i + 1, len(items)):
                    cov_j = estimate_chunk_slot_coverage(items[j], slots)
                    slots_j = {s for s, sc in cov_j.items() if sc >= COVERAGE_THRESHOLD}
                    if slots_i and slots_j and slots_i == slots_j:
                        redundancy += 0.35
                    elif len(items) >= 3:
                        redundancy += 0.20

    return min(1.5, redundancy)
