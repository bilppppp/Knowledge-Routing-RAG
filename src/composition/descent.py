# src/composition/descent.py
"""
E2: Slot-Conditioned Hybrid Targeted Descent Module.
Reference: V3 Stage B Specification (Sections XI - XVIII).

Design Principles:
1. Domain-Agnostic & Zero Lookup Tables:
   - Zero hardcoded statute names (no "医师法", "传染病", etc.).
   - Zero hardcoded qid or benchmark answer strings.
   - Derived strictly from unresolved question slots, target document title, and linguistic structure.
2. Slot Conditioning:
   - Accurately determines unresolved slots from current seed coverage.
   - Queries focus strictly on what is missing in the current context.
3. Dual-Channel Hybrid Retrieval:
   - Lexical Channel: FTS / BM25 inside target document using concise slot keywords.
   - Semantic Channel: Dense vector search inside target document using natural slot text.
4. Rank Fusion & Heading Bonus:
   - Reciprocal Rank Fusion (RRF) combining lexical and semantic rankings.
   - Structural heading overlap bonus (penalty-free bonus for category matches like 处罚, 许可, 附则, etc.).
"""

import re
import jieba
from typing import List, Dict, Set, Any, Tuple, Optional
from src.common.models import EvidenceItem
from src.composition.slots import EvidenceSlot, extract_evidence_slots
from src.composition.coverage import compute_set_slot_coverage
from src.services.search import SearchService
from src.graph.lsdb import KnowledgeLSDB

STOP_WORDS = {
    '对于', '在', '前', '有何', '何种', '吗', '呢', '什么', '哪些', '如何',
    '应当', '以及', '的和', '产生的', '关于', '根据', '依据', '制定', '可以', '是否',
    '请问', '分别', '具体', '属于', '进行', '相关', '国家', '规定', '要求', '条例',
    '两部', '两项', '两个', '办法', '细则', '决定', '法律', '法规', '上位法'
}

STRUCTURAL_CATEGORIES = {
    '处罚', '法律责任', '责任', '许可', '资格', '附则', '生效', '废止',
    '预防', '控制', '救治', '考核', '监督', '管理', '总则', '原则', '标准'
}


def identify_unresolved_slot(
    question: str,
    seed_items: List[EvidenceItem],
    slots: Optional[List[EvidenceSlot]] = None,
    target_doc_title: str = ""
) -> Tuple[EvidenceSlot, List[EvidenceSlot]]:
    """
    Identifies the primary unresolved slot and the set of covered slots.
    A slot is covered if current seed items achieve coverage >= 0.40.
    If multiple slots are unresolved, selects the one most relevant to the target doc title
    or the one with the lowest coverage score.
    """
    if not slots:
        slots = extract_evidence_slots(question)

    if not slots:
        # Fallback slot if question had no punctuation
        fallback = EvidenceSlot(
            slot_id="S1",
            text=question,
            slot_type="CONTENT_DEFINITION",
            keywords=[w for w in jieba.cut(question) if len(w.strip()) > 1 and w.strip() not in STOP_WORDS]
        )
        return fallback, []

    cov_info = compute_set_slot_coverage(seed_items, slots)
    chunk_scores = cov_info["chunk_scores"]

    # Maximum score achieved for each slot by any seed chunk
    slot_peak_scores = {}
    for s in slots:
        peak = max([chunk_scores[it.chunk_id].get(s.slot_id, 0.0) for it in seed_items] or [0.0])
        slot_peak_scores[s.slot_id] = peak

    covered_slots = [s for s in slots if slot_peak_scores.get(s.slot_id, 0.0) >= 0.40]
    unresolved_candidates = [s for s in slots if slot_peak_scores.get(s.slot_id, 0.0) < 0.40]

    if not unresolved_candidates:
        unresolved_candidates = sorted(slots, key=lambda s: slot_peak_scores.get(s.slot_id, 0.0))

    # If target_doc_title is provided, prioritize unresolved slot matching target doc topic
    if target_doc_title and len(unresolved_candidates) > 1:
        clean_target = re.sub(r"^(?:中华人民共和国)?", "", target_doc_title)
        target_words = set(w for w in jieba.cut(clean_target) if len(w) > 1 and w not in STOP_WORDS)

        def score_slot_title_match(slot: EvidenceSlot) -> int:
            return sum(1 for kw in slot.keywords if kw in target_words)

        unresolved_candidates.sort(key=score_slot_title_match, reverse=True)

    primary_unresolved = unresolved_candidates[0]
    filtered_covered = [s for s in covered_slots if s.slot_id != primary_unresolved.slot_id]
    return primary_unresolved, filtered_covered


def build_generic_descent_query(
    question: str,
    unresolved_slot: EvidenceSlot,
    target_doc_title: str = "",
    covered_slots: Optional[List[EvidenceSlot]] = None
) -> str:
    """
    Constructs a generic, slot-conditioned targeted descent query (Section XIII).
    Composed of:
      unresolved slot keywords
      + action/verb/condition/obligation terms
      + target document topic terms (stripped of boilerplate)
      - stopwords
      - question boilerplate
      - terms already heavily covered by covered_slots
    Target length: 3 to 8 high-information terms.
    """
    covered_kws = set()
    if covered_slots:
        for cs in covered_slots:
            covered_kws.update(cs.keywords)

    # 1. Unresolved slot terms (highest priority)
    slot_kws = [kw for kw in unresolved_slot.keywords if kw not in STOP_WORDS and kw not in covered_kws]

    # 2. Extract action/condition terms directly from unresolved slot text
    slot_text_words = [
        w for w in jieba.cut(unresolved_slot.text)
        if len(w.strip()) > 1 and w.strip() not in STOP_WORDS and w.strip() not in covered_kws
    ]

    # 3. Clean target doc title words (excluding generic suffixes)
    clean_title = re.sub(r"^(?:中华人民共和国)?", "", target_doc_title)
    clean_title = re.sub(r"[《》？?。；;，,\n（）()、“”\"']", " ", clean_title)
    title_words = [
        w for w in jieba.cut(clean_title)
        if len(w.strip()) > 1 and w.strip() not in STOP_WORDS and w.strip() not in ["关于", "规定", "条例", "办法", "批复", "细则", "法"]
    ]

    # Combine in prioritized order: slot terms first, then title disambiguation terms
    combined: List[str] = []
    seen: Set[str] = set()

    for w in slot_kws + slot_text_words + title_words:
        if w not in seen and len(w) > 1:
            seen.add(w)
            combined.append(w)
            if len(combined) >= 8:
                break

    if not combined:
        # Fallback to general question keywords
        combined = [w for w in jieba.cut(question) if len(w) > 1 and w not in STOP_WORDS][:6]

    return " ".join(combined)


def compute_heading_overlap_bonus(
    heading_path: str,
    unresolved_slot: EvidenceSlot,
    query_terms: List[str]
) -> float:
    """
    Computes a category-based heading overlap bonus (Section XVII).
    If the unresolved slot or query mentions structural categories like
    处罚, 许可, 资格, 责任, 附则, 应急, 考核 and the chunk's heading matches,
    award a small positive bonus (0.005 ~ 0.015 in RRF scale).
    """
    if not heading_path:
        return 0.0

    bonus = 0.0
    for cat in STRUCTURAL_CATEGORIES:
        if cat in unresolved_slot.text or any(cat in qt for qt in query_terms):
            if cat in heading_path:
                bonus += 0.010
                break  # Max 1 structural bonus per chunk

    # Also check if any high-value query term appears directly in the heading
    for qt in query_terms[:4]:
        if len(qt) > 1 and qt in heading_path:
            bonus += 0.005
            break

    return bonus


def hybrid_targeted_descent(
    search_service: SearchService,
    target_doc_id: str,
    target_doc_title: str,
    unresolved_slot: EvidenceSlot,
    generic_query: str,
    corpus: str = "D20",
    top_k_candidates: int = 4,
    rrf_k: int = 60,
    lsdb: Optional[KnowledgeLSDB] = None,
    channel_mode: str = "hybrid"  # "hybrid", "lexical_only", "semantic_only"
) -> List[Tuple[EvidenceItem, float, Dict[str, Any]]]:
    """
    Executes Slot-Conditioned Hybrid / Lexical-only / Semantic-only Targeted Descent:
    1. Lexical Channel: FTS BM25 search in target doc with generic_query -> top-3.
    2. Semantic Channel: Vector search in target doc with unresolved_slot.text -> top-3.
    3. Heading Bonus: Category overlap between unresolved slot and heading_path.
    4. Rank scoring / Fusion -> merged top-k candidates.
    Returns list of (item, fused_score, channel_metadata).
    """
    # 1. Lexical Channel (Top-3)
    if channel_mode in ("hybrid", "lexical_only"):
        lexical_items = search_service.fts_search_in_doc(
            query=generic_query,
            doc_id=target_doc_id,
            corpus=corpus,
            top_k=3
        )
    else:
        lexical_items = []

    # 2. Semantic Channel (Top-3)
    if channel_mode in ("hybrid", "semantic_only"):
        semantic_items = search_service.vector_search_in_doc(
            query=unresolved_slot.text,
            doc_id=target_doc_id,
            corpus=corpus,
            top_k=3
        )
    else:
        semantic_items = []

    # If active channels returned nothing, fallback to c001 if available
    if not lexical_items and not semantic_items:
        if lsdb:
            c1_data = lsdb.get_chunk_evidence(f"{target_doc_id}#c001")
            if c1_data:
                fallback_item = EvidenceItem(
                    chunk_id=c1_data["chunk_id"], doc_id=c1_data["doc_id"],
                    title=c1_data["title"], heading_path=c1_data["heading_path"],
                    text=c1_data["text"], score=0.92, source_method=f"e2_{channel_mode}_fallback_c001"
                )
                return [(fallback_item, 1.30, {"lexical_rank": None, "semantic_rank": None, "channel": "fallback"})]
        return []

    # 3. Fusion / Ranking + Heading Bonus
    query_terms = generic_query.split()
    rrf_scores: Dict[str, float] = {}
    item_map: Dict[str, EvidenceItem] = {}
    channel_meta: Dict[str, Dict[str, Any]] = {}

    for rank, item in enumerate(lexical_items, 1):
        cid = item.chunk_id
        item_map[cid] = item
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
        channel_meta.setdefault(cid, {})["lexical_rank"] = rank

    for rank, item in enumerate(semantic_items, 1):
        cid = item.chunk_id
        if cid not in item_map:
            item_map[cid] = item
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))
        channel_meta.setdefault(cid, {})["semantic_rank"] = rank

    # Apply Heading Overlap Bonus
    for cid, item in item_map.items():
        bonus = compute_heading_overlap_bonus(item.heading_path, unresolved_slot, query_terms)
        rrf_scores[cid] += bonus
        channel_meta[cid]["heading_bonus"] = bonus
        lex_r = channel_meta[cid].get("lexical_rank")
        sem_r = channel_meta[cid].get("semantic_rank")
        if lex_r and sem_r:
            channel_meta[cid]["channel"] = "both"
        elif lex_r:
            channel_meta[cid]["channel"] = "lexical_only"
        else:
            channel_meta[cid]["channel"] = "semantic_only"

    # Sort descending by fused / ranking score
    sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)
    results: List[Tuple[EvidenceItem, float, Dict[str, Any]]] = []

    method_name = (
        "e2_hybrid_targeted_descent" if channel_mode == "hybrid"
        else "e2_lexical_targeted_descent" if channel_mode == "lexical_only"
        else "e2_semantic_targeted_descent"
    )

    for cid in sorted_cids[:top_k_candidates]:
        it = item_map[cid]
        it.score = 0.96
        it.source_method = method_name
        results.append((it, rrf_scores[cid], channel_meta[cid]))

    return results
