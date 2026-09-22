# src/composition/composer.py
"""
Coverage-Preserving Evidence Composer Module (Candidate E1).
Reference: V3 Architecture Specification (Sections IX, XVI, XVII, XVIII, XXIV, XXV).

Replaces rigid position-based slot replacement with Coverage-Preserving Set Composition:
  - Decomposes questions into generic evidence slots
  - Protects unique/dominant slot coverage via UNIQUE_COVERAGE_LOCK
  - Distinguishes primary document representatives from secondary peer chunks
  - Estimates removal cost based on unique coverage, role degradation, and redundancy
  - Selects replacements based on set-level marginal utility gain (delta > 0)
  - Enforces conservative admission limits (max_replacements <= 2)
  - Generates detailed, explainable decision traces for every candidate evaluation

Zero benchmark-specific tuning; pure coverage and marginal utility optimization.
"""

from typing import List, Dict, Set, Tuple, Any, Optional
import numpy as np
from src.common.models import EvidenceItem
from src.composition.slots import EvidenceSlot, extract_evidence_slots
from src.composition.roles import infer_evidence_role
from src.composition.coverage import (
    estimate_chunk_slot_coverage,
    compute_set_slot_coverage,
    compute_set_redundancy,
    COVERAGE_THRESHOLD
)
from src.graph.lsdb import KnowledgeLSDB


def estimate_removal_cost(
    chunk: EvidenceItem,
    current_set: List[EvidenceItem],
    candidate: EvidenceItem,
    slots: List[EvidenceSlot],
    lsdb: Optional[KnowledgeLSDB] = None
) -> Tuple[float, str]:
    """
    Computes the removal cost of chunk E from current_set S when considering candidate C:
      - If E provides UNIQUE or DOMINANT coverage for any slot S_k, and C does NOT
        cover S_k with comparable score -> LOCKED (cost = 999.0).
      - Identifies the PRIMARY representative per document using composite slot value:
          comp_val = 0.5 * max(slot_cov) + 0.5 * sum(slot_cov)
      - Primary document representatives covering operative slots receive high retention cost.
      - Secondary peer chunks from the same document receive a substantial redundancy discount.
      - If E maintains an active multi-hop chain edge that C does not -> cost penalty (+1.0).
      - Low coverage chunks (comp_val < 0.20) receive an irrelevance discount (-0.40).
    """
    cov_info = compute_set_slot_coverage(current_set, slots)
    unique_slots = cov_info["unique_coverage_map"].get(chunk.chunk_id, [])

    # 1. UNIQUE COVERAGE LOCK (Core Safety Mechanism)
    if unique_slots:
        cand_cov = estimate_chunk_slot_coverage(candidate, slots)
        chunk_cov = cov_info["chunk_scores"].get(chunk.chunk_id, {})
        for s_id in unique_slots:
            e_score = chunk_cov.get(s_id, 0.0)
            c_score = cand_cov.get(s_id, 0.0)
            if c_score < COVERAGE_THRESHOLD or c_score < (e_score - 0.15):
                return 999.0, f"UNIQUE_COVERAGE_LOCK: chunk uniquely covers {s_id} (score={e_score:.2f} vs cand={c_score:.2f})"

    # Compute composite question value per chunk: balances peak coverage and multi-slot coverage
    chunk_cov = cov_info["chunk_scores"].get(chunk.chunk_id, {})
    max_c = max(chunk_cov.values()) if chunk_cov else 0.0
    sum_c = sum(chunk_cov.values()) if chunk_cov else 0.0
    comp_val = 0.5 * max_c + 0.5 * sum_c

    # 2. Identify primary representative per document in current_set
    doc_best: Dict[str, Tuple[str, float]] = {}
    for it in current_set:
        it_cov = cov_info["chunk_scores"].get(it.chunk_id, {})
        it_max = max(it_cov.values()) if it_cov else 0.0
        it_sum = sum(it_cov.values()) if it_cov else 0.0
        it_comp = 0.5 * it_max + 0.5 * it_sum
        d = it.doc_id
        if d not in doc_best or it_comp > doc_best[d][1]:
            doc_best[d] = (it.chunk_id, it_comp)

    is_primary = (chunk.chunk_id in {info[0] for info in doc_best.values()})

    cost = 0.20
    reason_parts = ["preservation_prior(+0.20)"]

    # 3. Multi-hop chain continuity check
    if lsdb and len(current_set) > 1:
        other_dids = {it.doc_id for it in current_set if it.chunk_id != chunk.chunk_id}
        has_edge_in_set = any(
            lsdb.G_routing.has_edge(chunk.doc_id, o_did) or lsdb.G_routing.has_edge(o_did, chunk.doc_id)
            for o_did in other_dids
        )
        if has_edge_in_set:
            cand_maintains_edge = any(
                lsdb.G_routing.has_edge(candidate.doc_id, o_did) or lsdb.G_routing.has_edge(o_did, candidate.doc_id)
                for o_did in other_dids
            )
            if not cand_maintains_edge:
                cost += 1.0
                reason_parts.append("breaks_chain_edge(+1.0)")

    # 4. Primary representative retention vs Secondary peer discount
    role = infer_evidence_role(chunk, slots)
    if is_primary:
        cost += 0.50 * comp_val
        reason_parts.append(f"primary_doc_chunk(+{0.50 * comp_val:.2f})")
        if role == "DIRECT":
            cost += 0.30
            reason_parts.append("direct_role(+0.30)")
        elif role in {"BASIS", "EXCEPTION"}:
            cost += 0.25
            reason_parts.append(f"{role.lower()}_role(+0.25)")
    else:
        # Secondary peer chunk: high discount, cheap to evict in favor of cross-document evidence
        cost -= 0.60
        cost += 0.40 * comp_val
        reason_parts.append(f"secondary_peer_discount(-0.60+{0.40 * comp_val:.2f})")

    # 5. Low coverage / irrelevant discount
    if comp_val < 0.20:
        cost -= 0.40
        reason_parts.append("low_coverage_discount(-0.40)")

    return round(cost, 4), "; ".join(reason_parts)


def compute_set_utility(
    evidence_set: List[EvidenceItem],
    slots: List[EvidenceSlot],
    lsdb: Optional[KnowledgeLSDB] = None
) -> float:
    """
    Evaluates total utility of an evidence set S:
      Utility(S) =
          w_cov   * SlotCoverage(S)
        + w_dir   * DirectSupport(S)
        + w_chain * ChainContinuity(S)
        + w_rel   * RelationSupport(S)
        + w_doc   * DocComplementarity(S)
        - w_red   * Redundancy(S)
        - w_irr   * IrrelevantContext(S)
    """
    if not evidence_set:
        return 0.0

    cov_info = compute_set_slot_coverage(evidence_set, slots)
    coverage_rate = cov_info["coverage_rate"]

    # 1. Direct support strength
    chunk_scores = cov_info["chunk_scores"]
    active_scores = []
    for cid, scores in chunk_scores.items():
        high = [sc for sc in scores.values() if sc >= COVERAGE_THRESHOLD]
        if high:
            active_scores.append(max(high))
    direct_support = float(np.mean(active_scores)) if active_scores else 0.0

    # 2. Chain continuity
    chain_bonus = 0.0
    distinct_dids = list({it.doc_id for it in evidence_set})
    if lsdb and len(distinct_dids) >= 2:
        connected_pairs = 0
        for i in range(len(distinct_dids)):
            for j in range(i + 1, len(distinct_dids)):
                d1, d2 = distinct_dids[i], distinct_dids[j]
                if lsdb.G_routing.has_edge(d1, d2) or lsdb.G_routing.has_edge(d2, d1):
                    connected_pairs += 1
        chain_bonus = min(connected_pairs * 0.40, 0.80)

    # 3. Relation support
    relation_bonus = 0.0
    has_basis_slot = any(s.slot_type == "LEGAL_BASIS" for s in slots)
    if has_basis_slot:
        has_basis_chunk = any(
            infer_evidence_role(it, slots) in {"BASIS", "BRIDGE"}
            for it in evidence_set
        )
        if has_basis_chunk:
            relation_bonus = 0.50

    # 4. Document complementarity
    doc_bonus = 0.0
    all_target_entities = {ent for s in slots for ent in s.target_entities}
    if len(all_target_entities) >= 2:
        covered_entities = sum(
            1 for ent in all_target_entities
            if any(ent.replace("中华人民共和国", "") in it.title for it in evidence_set)
        )
        if covered_entities >= 2:
            doc_bonus = 0.40

    # 5. Redundancy penalty
    redundancy = compute_set_redundancy(evidence_set, slots)

    # 6. Irrelevant context penalty
    irrelevant_count = 0
    for cid, scores in chunk_scores.items():
        if all(sc < COVERAGE_THRESHOLD for sc in scores.values()):
            irrelevant_count += 1
    irrelevant_penalty = irrelevant_count * 0.40

    # Composite Utility
    w_cov = 3.50
    w_dir = 1.00
    w_chain = 1.00
    w_rel = 0.50
    w_doc = 0.40
    w_red = 0.60
    w_irr = 0.50

    utility = (
        (w_cov * coverage_rate)
        + (w_dir * direct_support)
        + (w_chain * chain_bonus)
        + (w_rel * relation_bonus)
        + (w_doc * doc_bonus)
        - (w_red * redundancy)
        - (w_irr * irrelevant_penalty)
    )

    return round(float(utility), 4)


def compose_evidence(
    b0_evidence: List[EvidenceItem],
    candidate_pool: List[Tuple[EvidenceItem, float, str]],
    slots: List[EvidenceSlot],
    max_chunks: int = 5,
    max_replacements: int = 2,
    lsdb: Optional[KnowledgeLSDB] = None
) -> Tuple[List[EvidenceItem], List[Dict[str, Any]]]:
    """
    Performs Preservation-Aware Greedy Composition starting from B0 evidence set.
    Evaluates candidate marginal gains against removal costs with UNIQUE_COVERAGE_LOCK.
    Limits replacements to max_replacements (default 2) for admission safety.
    """
    current_set: List[EvidenceItem] = list(b0_evidence[:max_chunks])
    current_cids = {it.chunk_id for it in current_set}

    # Deduplicate candidates (preserve highest initial score)
    seen_cids = set(current_cids)
    unique_candidates: List[Tuple[EvidenceItem, float, str]] = []
    for item, score, reason in candidate_pool:
        if item.chunk_id not in seen_cids:
            seen_cids.add(item.chunk_id)
            unique_candidates.append((item, score, reason))

    # Sort candidates by candidate pool score descending
    unique_candidates.sort(key=lambda x: x[1], reverse=True)

    # Cap internal pool to at most 10 candidates
    unique_candidates = unique_candidates[:10]

    traces: List[Dict[str, Any]] = []
    replacements_made = 0

    # If initial B0 evidence has fewer than max_chunks, fill first
    while len(current_set) < max_chunks and unique_candidates:
        cand_item, cand_score, cand_reason = unique_candidates.pop(0)
        u_before = compute_set_utility(current_set, slots, lsdb)
        trial_set = current_set + [cand_item]
        u_after = compute_set_utility(trial_set, slots, lsdb)
        delta = u_after - u_before
        if delta > 0:
            current_set = trial_set
            current_cids.add(cand_item.chunk_id)
            traces.append({
                "candidate": cand_item.chunk_id,
                "action": "ADD",
                "evicted": None,
                "marginal_delta": round(delta, 4),
                "reason": f"Initial pool fill: added {cand_item.chunk_id} with delta={delta:+.4f}"
            })

    # Evaluate candidate replacements against existing chunks
    for cand_item, cand_score, cand_reason in unique_candidates:
        if replacements_made >= max_replacements:
            break
        if cand_item.chunk_id in current_cids:
            continue

        u_current = compute_set_utility(current_set, slots, lsdb)

        # 1. Evaluate removal cost for each chunk in current_set
        removal_evals = []
        for e in current_set:
            cost, r_reason = estimate_removal_cost(e, current_set, cand_item, slots, lsdb)
            removal_evals.append((e, cost, r_reason))

        # Sort by removal cost ascending
        removal_evals.sort(key=lambda x: x[1])
        best_removable, best_cost, cost_reason = removal_evals[0]

        # 2. Check if all candidates for removal are locked
        if best_cost >= 100.0:
            traces.append({
                "candidate": cand_item.chunk_id,
                "action": "REJECT",
                "evicted": None,
                "marginal_delta": 0.0,
                "reason": f"UNIQUE_COVERAGE_LOCK: All current evidence chunks locked ({cost_reason})"
            })
            continue

        # 3. Compute marginal utility delta of trial replacement
        trial_set = [it for it in current_set if it.chunk_id != best_removable.chunk_id] + [cand_item]
        u_trial = compute_set_utility(trial_set, slots, lsdb)
        delta = u_trial - u_current

        # 4. Replacement decision: require positive marginal delta
        if delta > 0.0:
            current_set = trial_set
            current_cids.remove(best_removable.chunk_id)
            current_cids.add(cand_item.chunk_id)
            replacements_made += 1
            traces.append({
                "candidate": cand_item.chunk_id,
                "action": "REPLACE",
                "evicted": best_removable.chunk_id,
                "evicted_cost": best_cost,
                "evicted_reason": cost_reason,
                "marginal_delta": round(delta, 4),
                "reason": f"Accepted: Evicted {best_removable.chunk_id} ({cost_reason}), Admitted {cand_item.chunk_id} (Delta={delta:+.4f})"
            })
        else:
            traces.append({
                "candidate": cand_item.chunk_id,
                "action": "REJECT",
                "proposed_eviction": best_removable.chunk_id,
                "evicted_cost": best_cost,
                "marginal_delta": round(delta, 4),
                "reason": f"Rejected: Marginal utility delta non-positive (Delta={delta:+.4f})"
            })

    return current_set[:max_chunks], traces
