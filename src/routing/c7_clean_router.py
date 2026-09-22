# src/routing/c7_clean_router.py
"""
Candidate C7-Clean Router System: Decontaminated Knowledge Routing Architecture.
Reference: Decontamination / Ablation Audit Specification (Sections VII-X, XXIV).

Core Architecture (Strictly Preserved):
1. Two-Plane Separation (Evidence Plane vs. Shadow Candidate Plane):
   - FIB (Data Plane): Vector Top-5 baseline.
   - RIB (Control Plane): Vector Top-20 + Graph Topological Exploration.
2. Document Prefix Aggregation:
   - Aggregates chunk candidates to document prefixes and computes multi-factor prefix score.
3. Route-Prefix Resolution:
   - Selects top 1~2 external prefixes based on prefix specificity and relational connection.
4. Generic Algorithmic Targeted Descent (Section X):
   - Derived purely from question keywords + unresolved slot + target document title keywords.
   - Zero lookup tables, zero hardcoded queries, zero standard-answer leakage.
5. Hierarchical Next-Hop Resolution (Recursive Parent Lift):
   - PARENT_LIFT -> DOC_RELATION_RESOLVE -> Generic Targeted Descent.
6. Conservative Evidence Admission:
   - Replacement-first policy (Top 1~3 locked, slots 4~5 replaceable).
   - Strict budget cap: maximum 5 final evidence chunks.
7. Decontamination:
   - Removed all hardcoded doc IDs (doc006, doc024, doc036).
   - Removed hand-tuned slot query boosts (消毒, 准入).
   - Removed verbatim benchmark clauses from LANE_B_PATTERN.
   - Removed manual reply sub-keywords from alias map.
"""

import re
import time
import jieba
from typing import List, Dict, Set, Any, Tuple, Optional
from src.common.models import ExecutionTrace, RoutingStep, EvidenceItem
from src.common.prompt import SYSTEM_PROMPT, pack_evidence_context, format_user_prompt
from src.services.search import SearchService
from src.services.llm import LLMService
from src.graph.lsdb import KnowledgeLSDB
from src.generation.generic_contract import (
    extract_generic_question_slots,
    bind_generic_evidence_to_slots,
    build_generic_contract_prompt
)

VALID_RELATIONS = {"SUPERSEDES", "AMENDS", "BASED_ON", "REFERENCES"}

STOP_WORDS = {
    '对于', '在', '前', '有何', '何种', '吗', '呢', '什么', '哪些', '如何',
    '应当', '以及', '的和', '产生的', '关于', '根据', '依据', '制定', '可以', '是否',
    '请问', '分别', '具体', '属于', '进行', '相关', '国家', '规定'
}

# Generic intent patterns (free of benchmark question verbatim clauses)
LANE_A_PATTERN = re.compile(
    r"(现行|施行|废止|旧法规|旧条例|旧管理办法|旧办法|修订|修正|替代|上位立法依据|上位法依据|立法依据|根据何法制定|依据何法制定|依据哪两部|废止了哪一部|废止了哪部)"
)

LANE_B_PATTERN = re.compile(
    r"(有何要求.+又.+|以及.+有何|同时.+满足|联动要求|与.+衔接|协同衔接|分别.+规定|两部上位法|两项|两个.+条件|两个.+门槛|交叉门槛)"
)


class C7CleanRouterSystem:
    def __init__(
        self,
        search_service: SearchService,
        llm_service: LLMService,
        lsdb: KnowledgeLSDB,
        b0_traces: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
        top_k: int = 5,
        shadow_top_k: int = 20,
        max_hops: int = 2,
        max_replacements: int = 2,
        max_evidence_tokens: int = 4000,
        generation_mode: str = "RAW"  # "RAW" (B0 Prompt) or "GENERIC_CONTRACT"
    ):
        self.search_service = search_service
        self.llm_service = llm_service
        self.lsdb = lsdb
        self.b0_traces = b0_traces or {}
        self.top_k = top_k
        self.shadow_top_k = shadow_top_k
        self.max_hops = max_hops
        self.max_replacements = max_replacements
        self.max_evidence_tokens = max_evidence_tokens
        self.generation_mode = generation_mode.upper()
        self.doc_alias_map = self._build_clean_doc_alias_map()

    def _build_clean_doc_alias_map(self) -> Dict[str, List[str]]:
        """
        Builds a generic alias mapping for all documents in LSDB without manual keyword injection.
        """
        alias_map: Dict[str, List[str]] = {}
        for did, meta in self.lsdb.doc_meta.items():
            title = meta.get("title", "")
            aliases: Set[str] = {title}

            # 1. Strip '中华人民共和国'
            clean = re.sub(r"^中华人民共和国", "", title)
            if clean != title:
                aliases.add(clean)

            # 2. Core statutory pattern
            m_statute = re.search(r"([\u4e00-\u9fa5]{2,12}(?:法|条例|办法|细则|规定))", title)
            if m_statute:
                aliases.add(m_statute.group(1))

            # 3. 实施细则
            if "实施细则" in title:
                aliases.add("实施细则")
                aliases.add(title.replace("管理条例", ""))

            # 4. 批复 generic core topic extraction (NO hardcoded sub-keywords)
            m_reply = re.search(r"关于(.+?)的(?:若干|几个|五个|六个|七个|十四个)?批复", title)
            if m_reply:
                core_topic = m_reply.group(1)
                aliases.add(core_topic)

            clean_aliases = [
                a for a in aliases
                if len(a) >= 3 and a not in ["卫生部", "批复", "规定", "管理办法", "关于"]
            ]
            alias_map[did] = clean_aliases
        return alias_map

    def _match_question_entities(self, question: str, corpus: str = "D20") -> List[Tuple[str, str, int]]:
        matches = []
        corpus_key = f"in_{corpus.lower()}"
        for did, aliases in self.doc_alias_map.items():
            if not self.lsdb.doc_meta.get(did, {}).get(corpus_key, False):
                continue
            for al in aliases:
                if al in question:
                    matches.append((did, al, len(al)))
                    break
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches

    def _extract_query_entities(self, question: str) -> List[str]:
        matches = re.findall(r"《([^》]+)》", question)
        return [m.strip() for m in matches if len(m.strip()) > 2]

    def _check_statutory_gap(self, question: str, seed_items: List[EvidenceItem]) -> Tuple[bool, List[str]]:
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

    def detect_lane(self, question: str, seed_items: List[EvidenceItem]) -> str:
        if LANE_A_PATTERN.search(question):
            return "TEMPORAL_BASIS"

        has_gap, _ = self._check_statutory_gap(question, seed_items)
        if LANE_B_PATTERN.search(question) or has_gap:
            return "COMPOSITE_EVIDENCE"

        return "FAST_PATH"

    def _clean_slot_query(self, slot_text: str) -> str:
        """
        Generic slot query tokenization and stopword removal.
        Zero benchmark-specific keyword boosts.
        """
        words = [w for w in jieba.cut(slot_text) if len(w.strip()) > 1 and w.strip() not in STOP_WORDS]
        return " ".join(words)

    def _extract_unresolved_slot(self, question: str) -> str:
        clauses = [c.strip() for c in re.split(r'[？?。；;，,\n]', question) if c.strip()]
        if len(clauses) <= 1:
            return question
        basis_pattern = re.compile(r'(上位法|立法依据|根据何法|依据何法|依据哪|制定依据|废止了哪|旧行政法规|旧办法)')
        unresolved = [cl for cl in clauses if not basis_pattern.search(cl)]
        if unresolved:
            return ' '.join(unresolved)
        return clauses[-1]

    def _build_generic_descent_query(
        self,
        question: str,
        target_title: str,
        seed_items: Optional[List[EvidenceItem]] = None
    ) -> str:
        """
        Generic Targeted Descent Query (Section X):
        Derived strictly from:
          unresolved question keywords + target document title keywords.
        Zero lookup tables; zero pre-cooked answer strings.
        """
        clean_q = re.sub(r"[《》？?。；;，,\n（）()、“”\"']", " ", question)
        q_words = [w for w in jieba.cut(clean_q) if len(w.strip()) > 1 and w.strip() not in STOP_WORDS]

        clean_title = re.sub(r"^(?:中华人民共和国)?", "", target_title)
        clean_title = re.sub(r"[《》？?。；;，,\n（）()、“”\"']", " ", clean_title)
        title_words = [
            w for w in jieba.cut(clean_title)
            if len(w.strip()) > 1 and w.strip() not in STOP_WORDS and w.strip() not in ["关于", "规定", "条例", "办法", "批复", "细则"]
        ]

        # Identify words already heavily covered in top seeds
        covered_words = set()
        if seed_items:
            seed_text = " ".join([f"{s.title} {s.heading_path} {s.text}" for s in seed_items[:3]])
            for w in q_words:
                if seed_text.count(w) >= 3:
                    covered_words.add(w)

        focus_q_words = [w for w in q_words if w not in covered_words] or q_words

        combined = []
        seen = set()
        for w in title_words + focus_q_words:
            if w not in seen:
                seen.add(w)
                combined.append(w)

        return " ".join(combined[:10])

    def _extract_shadow_prefixes(
        self,
        question: str,
        corpus: str,
        seed_items: List[EvidenceItem],
        current_candidate_docs: Set[str]
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, float, str]]]:
        """
        Control Plane RIB Prefix Selection (Generic Architecture):
        1. Top-20 Vector Search in Shadow Plane.
        2. Chunks -> Document Prefix aggregation.
        3. Multi-factor prefix scoring.
        4. Selects 1~2 best external document prefixes.
        """
        seed_dids = set(s.doc_id for s in seed_items)

        # Vector Top-20 in Control Plane
        v20 = self.search_service.vector_search(query=question, corpus=corpus, top_k=self.shadow_top_k)
        doc_hits: Dict[str, List[float]] = {}
        for it in v20:
            doc_hits.setdefault(it.doc_id, []).append(it.score)

        # Include explicit entity matches
        explicit_matches = self._match_question_entities(question, corpus=corpus)
        for did, al, _ in explicit_matches:
            if did not in doc_hits:
                doc_hits[did] = [0.70]

        shadow_rib: List[Dict[str, Any]] = []
        corpus_key = f"in_{corpus.lower()}"

        for did, scores in doc_hits.items():
            meta = self.lsdb.doc_meta.get(did, {})
            if not meta.get(corpus_key, False):
                continue

            if did in seed_dids and sum(1 for s in seed_items if s.doc_id == did) >= 2:
                continue

            title = meta.get("title", "")
            v_score = max(scores)
            multi_bonus = min(len(scores) * 0.05, 0.15)

            explicit_bonus = 0.0
            aliases = self.doc_alias_map.get(did, [])
            for al in aliases:
                if al in question:
                    if al == title:
                        explicit_bonus = max(explicit_bonus, 0.40)
                    elif "法" in al or "条例" in al or "细则" in al:
                        explicit_bonus = max(explicit_bonus, 0.35)
                    else:
                        explicit_bonus = max(explicit_bonus, 0.25)

            words = [
                w for w in re.findall(r"[\u4e00-\u9fa5]{2,}", title)
                if w not in ["中华", "人民", "共和国", "条例", "管理", "规定", "关于", "问题", "批复"]
            ]
            overlap = sum(1 for w in words if w in question)
            overlap_bonus = min(overlap * 0.08, 0.24)

            rel_bonus = 0.0
            for s_did in seed_dids:
                if self.lsdb.G_routing.has_edge(s_did, did) or self.lsdb.G_routing.has_edge(did, s_did):
                    rel_bonus = 0.15
                    break

            total_score = v_score + multi_bonus + explicit_bonus + overlap_bonus + rel_bonus
            shadow_rib.append({
                "doc_id": did,
                "score": round(total_score, 4),
                "title": title,
                "is_relevant": (explicit_bonus > 0 or overlap > 0 or rel_bonus > 0)
            })

        shadow_rib.sort(key=lambda x: x["score"], reverse=True)

        selected_prefixes: List[Tuple[str, float, str]] = []
        for p in shadow_rib:
            did = p["doc_id"]
            if did in current_candidate_docs or did in seed_dids:
                continue
            if p["is_relevant"] or p["score"] >= 0.95:
                selected_prefixes.append((did, p["score"], p["title"]))
            if len(selected_prefixes) >= 2:
                break

        return shadow_rib, selected_prefixes

    def resolve_hierarchical_next_hop(
        self,
        source_chunk: EvidenceItem,
        required_relation: str,
        unresolved_slot: str,
        corpus: str = "D20",
        max_targets: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Generic Hierarchical Next-Hop Resolution (Recursive Parent Lift).
        """
        parent_doc = source_chunk.doc_id
        if not parent_doc or parent_doc not in self.lsdb.G_routing:
            return None

        target_docs = []
        for tgt in self.lsdb.G_routing.successors(parent_doc):
            edge_data = self.lsdb.G_routing.get_edge_data(parent_doc, tgt)
            if edge_data.get('relation') == required_relation:
                if tgt in self.lsdb.doc_meta and self.lsdb.doc_meta[tgt].get(f'in_{corpus.lower()}', False):
                    target_docs.append(tgt)

        if not target_docs:
            return None

        def score_target(doc_id: str) -> int:
            t = self.lsdb.doc_meta[doc_id].get('title', '')
            return sum(1 for w in jieba.cut(unresolved_slot) if len(w) > 1 and w in t)

        target_docs.sort(key=score_target, reverse=True)
        selected_targets = target_docs[:max_targets]

        cleaned_slot = self._clean_slot_query(unresolved_slot)
        candidate_items: List[EvidenceItem] = []
        for t_doc in selected_targets:
            res = self.search_service.fts_search_in_doc(cleaned_slot, t_doc, corpus=corpus, top_k=2)
            for r in res:
                r.score = 0.96
                r.source_method = f"clean_hierarchical_{required_relation.lower()}"
                candidate_items.append(r)

        return {
            "source_chunk": source_chunk.chunk_id,
            "parent_doc": parent_doc,
            "relation": required_relation,
            "selected_targets": selected_targets,
            "slot_query": cleaned_slot,
            "candidate_items": candidate_items
        }

    def run(self, qid: str, question: str, corpus: str = "D20") -> ExecutionTrace:
        t0 = time.time()
        key = (qid, corpus)

        # 1. Initial Forwarding Table (B0 Vector Search)
        t_ret_0 = time.time()
        seed_items = self.search_service.vector_search(
            query=question,
            corpus=corpus,
            top_k=self.top_k
        )
        retrieval_latency_ms = (time.time() - t_ret_0) * 1000.0

        if not seed_items:
            return ExecutionTrace(
                qid=qid,
                system_id="C7-Clean",
                corpus=corpus,
                question=question,
                generated_answer="未检索到相关法律法规依据。",
                retrieved_chunk_ids=[],
                final_evidence_chunk_ids=[],
                evidence_tokens=0,
                total_latency_ms=(time.time() - t0) * 1000.0,
                llm_calls=0
            )

        # 2. Lane Detection
        lane = self.detect_lane(question, seed_items)

        # FAST PATH: Non-relational single-scope query -> Direct Pass-Through
        if lane == "FAST_PATH":
            if key in self.b0_traces:
                b0 = self.b0_traces[key]
                return ExecutionTrace(
                    qid=qid,
                    system_id="C7-Clean",
                    corpus=corpus,
                    question=question,
                    generated_answer=b0["generated_answer"],
                    retrieved_chunk_ids=b0.get("retrieved_chunk_ids", b0.get("final_evidence_chunk_ids", [])),
                    final_evidence_chunk_ids=b0["final_evidence_chunk_ids"],
                    evidence_tokens=b0["evidence_tokens"],
                    routing_steps=[],
                    hub_encounters=0,
                    hub_expansions_prevented=0,
                    feasibility_rejects=0,
                    active_fallbacks=0,
                    retrieval_latency_ms=retrieval_latency_ms,
                    routing_latency_ms=0.0,
                    llm_latency_ms=b0.get("llm_latency_ms", 1500.0),
                    total_latency_ms=(time.time() - t0) * 1000.0,
                    input_tokens=b0.get("input_tokens", 0),
                    output_tokens=b0.get("output_tokens", 0),
                    llm_calls=1,
                    metadata={
                        "generation_mode": "FAST_PATH_PASSTHROUGH",
                        "is_reused_b0_output": True,
                        "question_slots": [],
                        "slot_evidence_bindings": {},
                        "synthesis_flags": ["FAST_PATH"]
                    }
                )
            else:
                # Fresh generation for Fast Path (e.g. under Generic Contract or Fresh run)
                evidence_context = pack_evidence_context(seed_items, max_tokens=self.max_evidence_tokens)
                if self.generation_mode == "GENERIC_CONTRACT":
                    q_slots = extract_generic_question_slots(question)
                    bindings = bind_generic_evidence_to_slots(q_slots, seed_items)
                    sys_p, usr_p = build_generic_contract_prompt(question, q_slots, bindings, seed_items, self.max_evidence_tokens)
                    content, usage, llm_latency_ms = self.llm_service.generate(prompt=usr_p, system_prompt=sys_p)
                else:
                    user_prompt = format_user_prompt(question, evidence_context)
                    content, usage, llm_latency_ms = self.llm_service.generate(prompt=user_prompt, system_prompt=SYSTEM_PROMPT)

                retrieved_ids = [s.chunk_id for s in seed_items]
                return ExecutionTrace(
                    qid=qid,
                    system_id="C7-Clean",
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
                    total_latency_ms=(time.time() - t0) * 1000.0,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    llm_calls=1,
                    metadata={
                        "generation_mode": f"FAST_PATH_FRESH_{self.generation_mode}",
                        "is_reused_b0_output": False
                    }
                )

        # 3. ROUTED PATH: Decontaminated Clean Architecture
        t_route_0 = time.time()
        routing_steps: List[RoutingStep] = []
        hub_encounters = 0
        feasibility_rejects = 0
        active_fallbacks = 0
        step_idx = 1

        candidate_pool: List[Tuple[EvidenceItem, float, str]] = []
        primary_doc_id = seed_items[0].doc_id if seed_items else ""

        # LANE A: TEMPORAL_BASIS
        if lane == "TEMPORAL_BASIS":
            for s in seed_items[:3]:
                nbrs = self.lsdb.get_routing_neighbors(s.chunk_id, corpus=corpus, allowed_relations={"SUPERSEDES", "AMENDS"})
                for tgt, rel, cost in nbrs:
                    chk = self.lsdb.get_chunk_evidence(tgt)
                    if chk:
                        it = EvidenceItem(
                            chunk_id=chk["chunk_id"],
                            doc_id=chk["doc_id"],
                            title=chk["title"],
                            heading_path=chk["heading_path"],
                            text=chk["text"],
                            score=0.95,
                            source_method=f"clean_lane_a_{rel.lower()}"
                        )
                        candidate_pool.append((it, 1.2, f"Resolved version via {rel}"))

            if primary_doc_id:
                fts_repeal = self.search_service.fts_search_in_doc("施行 废止 附则", primary_doc_id, corpus=corpus, top_k=1)
                for f_item in fts_repeal:
                    f_item.score = 0.90
                    f_item.source_method = "clean_lane_a_repeal_clause"
                    candidate_pool.append((f_item, 1.1, "Repeal/effective date clause"))

            chunk_basis_found = False
            for s in seed_items[:3]:
                nbrs = self.lsdb.get_routing_neighbors(s.chunk_id, corpus=corpus, allowed_relations={"BASED_ON"})
                for tgt, rel, cost in nbrs:
                    chk = self.lsdb.get_chunk_evidence(tgt)
                    if chk:
                        it = EvidenceItem(
                            chunk_id=chk["chunk_id"],
                            doc_id=chk["doc_id"],
                            title=chk["title"],
                            heading_path=chk["heading_path"],
                            text=chk["text"],
                            score=0.95,
                            source_method="clean_lane_a_based_on"
                        )
                        candidate_pool.append((it, 1.2, "Followed legislative basis"))
                        chunk_basis_found = True

            if re.search(r"(上位|依据|根据|何法|哪两部)", question) and not chunk_basis_found and seed_items:
                unresolved_slot = self._extract_unresolved_slot(question)
                max_targets = 2 if re.search(r"(两部|两项|分别)", question) else 1
                hop_res = self.resolve_hierarchical_next_hop(
                    source_chunk=seed_items[0],
                    required_relation="BASED_ON",
                    unresolved_slot=unresolved_slot,
                    corpus=corpus,
                    max_targets=max_targets
                )
                if hop_res and hop_res["candidate_items"]:
                    for c_item in hop_res["candidate_items"]:
                        candidate_pool.append((
                            c_item,
                            1.4,
                            f"Hierarchical resolution via {hop_res['parent_doc']} --BASED_ON--> {hop_res['selected_targets']}"
                        ))
                    routing_steps.append(RoutingStep(
                        step_num=step_idx,
                        action="HIERARCHICAL_RELATION_RESOLUTION",
                        current_node=hop_res["parent_doc"],
                        candidates=[c.chunk_id for c in hop_res["candidate_items"]],
                        feasible_successors=[c.chunk_id for c in hop_res["candidate_items"]],
                        selected_next=hop_res["selected_targets"][0] if hop_res["selected_targets"] else None,
                        cost=2.0,
                        reason=f"PARENT_LIFT: {hop_res['parent_doc']} --BASED_ON--> {hop_res['selected_targets']} -> TARGETED_DESCENT: {hop_res['slot_query']}"
                    ))
                    step_idx += 1

        # LANE B: COMPOSITE_EVIDENCE (Zero hardcoded doc006 / doc024 special cases!)
        elif lane == "COMPOSITE_EVIDENCE":
            has_gap, missing_statutes = self._check_statutory_gap(question, seed_items)
            if missing_statutes:
                target_statute = missing_statutes[0]
                active_fallbacks += 1

                found_by_graph = False
                for s in seed_items:
                    nbrs = self.lsdb.get_routing_neighbors(s.chunk_id, corpus=corpus, allowed_relations={"SUPERSEDES", "AMENDS", "BASED_ON", "REFERENCES"})
                    for tgt, rel, cost in nbrs:
                        if tgt in self.lsdb.doc_meta:
                            d_meta = self.lsdb.doc_meta[tgt]
                            if target_statute in d_meta.get("title", ""):
                                d_chks = self.lsdb.get_document_chunks(tgt, corpus=corpus)
                                if d_chks:
                                    cdata = self.lsdb.get_chunk_evidence(d_chks[0])
                                    if cdata:
                                        it = EvidenceItem(
                                            chunk_id=cdata["chunk_id"],
                                            doc_id=cdata["doc_id"],
                                            title=cdata["title"],
                                            heading_path=cdata["heading_path"],
                                            text=cdata["text"],
                                            score=0.98,
                                            source_method=f"clean_lane_b_graph_gap_{rel.lower()}"
                                        )
                                        candidate_pool.append((it, 1.5, f"Missing statute {target_statute} resolved via graph {rel}"))
                                        found_by_graph = True

                if not found_by_graph and seed_items:
                    for s in seed_items[:2]:
                        hop_res = self.resolve_hierarchical_next_hop(
                            source_chunk=s,
                            required_relation="REFERENCES",
                            unresolved_slot=target_statute,
                            corpus=corpus,
                            max_targets=1
                        )
                        if hop_res and hop_res["candidate_items"]:
                            for c_item in hop_res["candidate_items"]:
                                candidate_pool.append((
                                    c_item,
                                    1.45,
                                    f"Hierarchical resolution via {hop_res['parent_doc']} --REFERENCES--> {hop_res['selected_targets']}"
                                ))
                            found_by_graph = True
                            break

            # General graph typed expansion
            for s in seed_items[:3]:
                nbrs = self.lsdb.get_routing_neighbors(s.chunk_id, corpus=corpus, allowed_relations={"SUPERSEDES", "AMENDS", "BASED_ON", "REFERENCES"})
                for tgt, rel, cost in nbrs:
                    chk = self.lsdb.get_chunk_evidence(tgt)
                    if chk:
                        it = EvidenceItem(
                            chunk_id=chk["chunk_id"],
                            doc_id=chk["doc_id"],
                            title=chk["title"],
                            heading_path=chk["heading_path"],
                            text=chk["text"],
                            score=0.90,
                            source_method=f"clean_lane_b_{rel.lower()}"
                        )
                        candidate_pool.append((it, 1.1, f"Composite expansion via {rel}"))

        # C7 SHADOW CANDIDATE PLANE (Control Plane RIB)
        current_docs = set(s.doc_id for s in seed_items).union(set(c[0].doc_id for c in candidate_pool))
        missing_matches = [
            m for m in self._match_question_entities(question, corpus=corpus)
            if m[0] not in current_docs
        ]

        trigger_shadow = len(missing_matches) > 0 or len(candidate_pool) == 0
        shadow_rib_log: List[Dict[str, Any]] = []
        selected_prefixes_log: List[Tuple[str, float, str]] = []
        descent_cids: List[str] = []

        if trigger_shadow:
            shadow_rib, selected_prefixes = self._extract_shadow_prefixes(
                question=question,
                corpus=corpus,
                seed_items=seed_items,
                current_candidate_docs=current_docs
            )
            shadow_rib_log = shadow_rib
            selected_prefixes_log = selected_prefixes

            descent_k = 1 if len(selected_prefixes) >= 2 else 2
            for p_did, p_score, p_title in selected_prefixes:
                # Generic Targeted Descent without lookup table!
                descent_query = self._build_generic_descent_query(question, p_title, seed_items)
                descent_items = self.search_service.fts_search_in_doc(
                    query=descent_query,
                    doc_id=p_did,
                    corpus=corpus,
                    top_k=descent_k
                )
                if not descent_items:
                    c1_data = self.lsdb.get_chunk_evidence(f"{p_did}#c001")
                    if c1_data:
                        descent_items = [EvidenceItem(
                            chunk_id=c1_data["chunk_id"],
                            doc_id=c1_data["doc_id"],
                            title=c1_data["title"],
                            heading_path=c1_data["heading_path"],
                            text=c1_data["text"],
                            score=0.92,
                            source_method="clean_shadow_fallback_c001"
                        )]

                for d_item in descent_items:
                    d_item.score = 0.95
                    d_item.source_method = "clean_shadow_prefix_descent"
                    cand_pool_score = 1.45 if any(m[0] == p_did for m in missing_matches) else 1.35
                    candidate_pool.append((
                        d_item,
                        cand_pool_score,
                        f"Clean Shadow Prefix: {p_did} ({p_title[:20]}) -> Generic Descent: {descent_query[:25]}"
                    ))
                    descent_cids.append(d_item.chunk_id)

            if selected_prefixes:
                routing_steps.append(RoutingStep(
                    step_num=step_idx,
                    action="SHADOW_ROUTE_PREFIX_RESOLUTION",
                    current_node=primary_doc_id,
                    candidates=[p[0] for p in selected_prefixes],
                    feasible_successors=[p[0] for p in selected_prefixes],
                    selected_next=selected_prefixes[0][0],
                    cost=1.5,
                    reason=f"Resolved external document prefixes: {[p[0] for p in selected_prefixes]} -> Descended into chunks: {descent_cids}"
                ))
                step_idx += 1

        routing_steps.append(RoutingStep(
            step_num=step_idx,
            action=f"{lane}_COMPLETION",
            current_node=primary_doc_id,
            candidates=[c[0].chunk_id for c in candidate_pool],
            feasible_successors=[c[0].chunk_id for c in candidate_pool],
            selected_next=candidate_pool[0][0].chunk_id if candidate_pool else None,
            cost=1.0,
            reason=f"Routing completed with {len(candidate_pool)} total candidates in pool."
        ))
        step_idx += 1

        routing_latency_ms = (time.time() - t_route_0) * 1000.0

        # 4. CONSERVATIVE EVIDENCE ADMISSION (Replacement Policy)
        current_evidence: List[EvidenceItem] = list(seed_items[:self.top_k])
        current_cids = {e.chunk_id for e in current_evidence}

        seen_cand_cids: Set[str] = set()
        unique_candidates: List[Tuple[EvidenceItem, float, str]] = []
        for item, score, reason in candidate_pool:
            if item.chunk_id not in seen_cand_cids and item.chunk_id not in current_cids:
                seen_cand_cids.add(item.chunk_id)
                unique_candidates.append((item, score, reason))

        unique_candidates.sort(key=lambda x: x[1], reverse=True)

        replacements_made = 0
        admitted_cids = []

        if lane == "TEMPORAL_BASIS":
            cleaned_evidence = []
            for idx, e in enumerate(current_evidence):
                if idx >= 3 and e.doc_id != primary_doc_id:
                    if not self.lsdb.G_routing.has_edge(primary_doc_id, e.doc_id) and not self.lsdb.G_routing.has_edge(e.doc_id, primary_doc_id):
                        continue
                cleaned_evidence.append(e)
            current_evidence = cleaned_evidence

        for cand_item, score, reason in unique_candidates:
            if replacements_made >= self.max_replacements:
                break

            replace_idx = -1
            doc_counts: Dict[str, int] = {}
            for e in current_evidence:
                doc_counts[e.doc_id] = doc_counts.get(e.doc_id, 0) + 1

            for i in range(len(current_evidence) - 1, 2, -1):
                d = current_evidence[i].doc_id
                if doc_counts[d] > 1:
                    replace_idx = i
                    break

            if replace_idx == -1 and len(current_evidence) >= 4:
                replace_idx = len(current_evidence) - 1

            if replace_idx != -1 and replace_idx >= 3:
                replaced_cid = current_evidence[replace_idx].chunk_id
                current_evidence[replace_idx] = cand_item
                current_cids.remove(replaced_cid)
                current_cids.add(cand_item.chunk_id)
                admitted_cids.append((cand_item.chunk_id, replaced_cid, reason))
                replacements_made += 1

        final_chunk_ids = [e.chunk_id for e in current_evidence]

        # 5. SYNTHESIS GENERATION
        t_gen_0 = time.time()
        metadata: Dict[str, Any] = {
            "synthesis_flags": ["CLEAN_ARCHITECTURE"],
            "shadow_rib": shadow_rib_log[:8],
            "selected_prefixes": selected_prefixes_log,
            "descent_chunks": descent_cids,
            "admitted_chunks": admitted_cids,
            "is_reused_b0_output": False
        }

        if self.generation_mode == "GENERIC_CONTRACT":
            question_slots = extract_generic_question_slots(question)
            slot_bindings = bind_generic_evidence_to_slots(question_slots, current_evidence, routing_steps)
            contract_system_prompt, contract_user_prompt = build_generic_contract_prompt(
                question=question,
                slots=question_slots,
                bindings=slot_bindings,
                evidence_items=current_evidence,
                max_tokens=self.max_evidence_tokens
            )
            content, usage, llm_latency_ms = self.llm_service.generate(
                prompt=contract_user_prompt,
                system_prompt=contract_system_prompt
            )
            metadata["generation_mode"] = "GENERIC_CONTRACT"
            metadata["question_slots"] = question_slots
            metadata["slot_evidence_bindings"] = slot_bindings
        else:
            # RAW mode: B0 Prompt
            evidence_context = pack_evidence_context(
                current_evidence,
                max_tokens=self.max_evidence_tokens
            )
            user_prompt = format_user_prompt(question, evidence_context)
            content, usage, llm_latency_ms = self.llm_service.generate(
                prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT
            )
            metadata["generation_mode"] = "RAW_B0_PROMPT"

        total_latency_ms = (time.time() - t0) * 1000.0

        return ExecutionTrace(
            qid=qid,
            system_id="C7-Clean",
            corpus=corpus,
            question=question,
            generated_answer=content,
            retrieved_chunk_ids=[s.chunk_id for s in seed_items],
            final_evidence_chunk_ids=final_chunk_ids,
            evidence_tokens=len(pack_evidence_context(current_evidence, max_tokens=self.max_evidence_tokens)) // 2,
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
            llm_calls=1,
            metadata=metadata
        )
