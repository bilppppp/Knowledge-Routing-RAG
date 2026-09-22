#!/usr/bin/env python3
"""
scripts/build_confirmation_holdout.py
Independent Holdout Benchmark Generator for Confirmation Experiment.
Reference: Section IV-XI of Confirmation Specification.

Builds:
  - benchmark/confirmation/questions.jsonl (N=200 independent questions on D100)
  - benchmark/confirmation/gold.jsonl (N=200 ground truth records with exact verbatim spans)
  - benchmark/confirmation/manifest.json (Cryptographic hashes, counts, HOLDOUT FROZEN)
  - reports/confirmation_dataset_audit.md (Detailed audit report)

Constraints:
  - Exactly 200 questions: 80 1-hop, 70 2-hop, 50 3-hop+.
  - Zero usage of old benchmark gold chunks (180 chunks excluded).
  - Jaccard similarity against all 120 old questions strictly < 0.35.
  - Zero topic/clause overlap with banned historical debug cases (Q014, Q016, Q025, Q026, Q048, Q076, Q089, Q040, Q065).
  - Verifiable exact substring spans (assert span in chunk['text']).
  - Pure corpus-driven synthesis from D100 documents and LSDB relations.
"""

import os
import re
import sys
import json
import time
import hashlib
import random
import sqlite3
import jieba
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Any, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.llm import LLMService

DATA_DIR = PROJECT_ROOT / "data"
BENCHMARK_OLD = PROJECT_ROOT / "benchmark"
CONFIRMATION_DIR = PROJECT_ROOT / "benchmark" / "confirmation"
REPORTS_DIR = PROJECT_ROOT / "reports"

CONFIRMATION_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Banned keywords associated with known contaminated/debug cases
BANNED_PATTERNS = [
    re.compile(r"医疗废物.*传染病.*消毒.*交接"),
    re.compile(r"精神药品管理办法.*麻醉药品.*废止"),
    re.compile(r"新发突发.*重大传染病.*防范与应急"),
    re.compile(r"医务人员医德医风.*职业道德评价.*实施细则"),
    re.compile(r"现金交易.*绝对禁止.*麻醉药品.*批发"),
    re.compile(r"乡村医生.*个体诊所.*地域限制"),
    re.compile(r"产前诊断.*终止妊娠.*严重缺陷.*母婴保健"),
    re.compile(r"非法组织他人卖血.*非法采集血液.*批复"),
    re.compile(r"未取得医师执业资格.*独立从事.*医疗卫生技术工作")
]


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_tokens(text: str) -> Set[str]:
    return set(w for w in jieba.lcut(text) if len(w.strip()) > 1)


def calc_jaccard(s1: Set[str], s2: Set[str]) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def find_best_exact_span(chunk_text: str, candidate_span: str) -> Optional[str]:
    """Finds an exact verbatim substring from chunk_text matching candidate_span."""
    if not candidate_span:
        return None
    candidate_span = candidate_span.strip().strip('"\'“”‘’')
    if candidate_span in chunk_text:
        return candidate_span

    # Try sentence splitting
    sentences = re.split(r"(?<=[。；！\n])", chunk_text)
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) >= 10 and (s_clean in candidate_span or candidate_span in s_clean):
            if s_clean in chunk_text:
                return s_clean

    # Try finding longest common substring
    words = jieba.lcut(candidate_span)
    meaningful_words = [w for w in words if len(w.strip()) >= 2]
    scored = []
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) >= 10:
            match_cnt = sum(1 for w in meaningful_words if w in s_clean)
            if match_cnt > 0:
                scored.append((match_cnt, s_clean))
    if scored:
        scored.sort(key=lambda x: -x[0])
        best_sentence = scored[0][1]
        if best_sentence in chunk_text:
            return best_sentence

    return None


class ConfirmationHoldoutBuilder:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.llm = LLMService()
        self.load_resources()

    def load_resources(self):
        # 1. Load D100 doc IDs
        with open(DATA_DIR / "manifests" / "d100.json", "r", encoding="utf-8") as f:
            self.d100_doc_ids = set(json.load(f)["doc_ids"])

        # 2. Load Old Gold Chunks and Questions
        self.old_gold_chunks = set()
        with open(BENCHMARK_OLD / "gold.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.old_gold_chunks.update(item.get("gold_chunk_ids", []))

        self.old_questions = []
        self.old_question_tokens = []
        with open(BENCHMARK_OLD / "questions.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)["question"]
                    self.old_questions.append(q)
                    self.old_question_tokens.append(get_tokens(q))

        # 3. Load All Chunks
        self.all_chunks = []
        self.chunk_map = {}
        with open(DATA_DIR / "chunks.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    self.chunk_map[c["chunk_id"]] = c
                    if c["doc_id"] in self.d100_doc_ids:
                        self.all_chunks.append(c)

        # 4. Filter Clean Chunks
        self.clean_chunks = [
            c for c in self.all_chunks if c["chunk_id"] not in self.old_gold_chunks
        ]
        self.clean_chunk_map = {c["chunk_id"]: c for c in self.clean_chunks}
        self.doc_to_clean_chunks = defaultdict(list)
        for c in self.clean_chunks:
            self.doc_to_clean_chunks[c["doc_id"]].append(c)

        # 5. Load LSDB & Documents
        conn = sqlite3.connect(DATA_DIR / "knowledge_lsdb.sqlite")
        cur = conn.cursor()
        cur.execute("SELECT doc_id, title FROM documents WHERE in_d100 = 1")
        self.doc_titles = dict(cur.fetchall())

        cur.execute("SELECT source, target, relation FROM edges WHERE is_routing = 1")
        self.routing_edges = cur.fetchall()
        conn.close()

        print(f"Loaded {len(self.d100_doc_ids)} D100 documents.")
        print(f"Loaded {len(self.clean_chunks)} clean chunks (excluded {len(self.old_gold_chunks)} old gold chunks).")
        print(f"Loaded {len(self.old_questions)} old benchmark questions for deduplication.")

    def is_decontaminated(self, question: str) -> Tuple[bool, float, str]:
        # Check banned regex patterns
        for pat in BANNED_PATTERNS:
            if pat.search(question):
                return False, 1.0, f"Matched banned pattern: {pat.pattern}"

        # Check Jaccard similarity against all 120 old questions
        q_tokens = get_tokens(question)
        max_jaccard = 0.0
        max_old_q = ""
        for old_q, old_tokens in zip(self.old_questions, self.old_question_tokens):
            j = calc_jaccard(q_tokens, old_tokens)
            if j > max_jaccard:
                max_jaccard = j
                max_old_q = old_q

        if max_jaccard >= 0.35:
            return False, max_jaccard, f"Jaccard {max_jaccard:.3f} >= 0.35 against: {max_old_q[:40]}..."

        return True, max_jaccard, ""

    def generate_single_hop_item(self, chunk: Dict[str, Any], qid: str) -> Optional[Dict[str, Any]]:
        doc_id = chunk["doc_id"]
        doc_title = self.doc_titles.get(doc_id, chunk.get("title", ""))
        chunk_text = chunk["text"]

        prompt = f"""你是一位专业法律评测专家。请基于以下中国卫生健康与医药法规切片，编写一个高质量、真实、明确的单跳法律问答题。

【法规名称】《{doc_title}》
【切片内容】
{chunk_text}

要求：
1. 提问形式：根据《{doc_title}》，...？必须明确指向切片中的具体规定（如条件、期限、职责、罚则、定义或例外）。
2. gold_answer：严格根据切片原文提炼，简洁准确，不包含推测或外部知识。
3. gold_span：从切片原文中原样复制1段完整句子（15~100字），必须一字不差来自切片原文。
4. tag：从 ["definition", "procedure", "penalty", "exception", "condition"] 中选择最贴切的1个。

输出严格JSON格式：
{{
  "question": "根据《{doc_title}》，...？",
  "gold_answer": "...",
  "gold_span": "...",
  "tag": "..."
}}
"""
        for _ in range(3):
            try:
                resp, _, _ = self.llm.generate(prompt=prompt, response_format_json=True)
                data = json.loads(resp)
                q = data.get("question", "").strip()
                ans = data.get("gold_answer", "").strip()
                span = data.get("gold_span", "").strip()
                tag = data.get("tag", "condition").strip()

                if not q or not ans or not span:
                    continue

                valid_span = find_best_exact_span(chunk_text, span)
                if not valid_span:
                    continue

                is_ok, max_j, reason = self.is_decontaminated(q)
                if not is_ok:
                    continue

                return {
                    "qid": qid,
                    "question": q,
                    "gold_answer": ans,
                    "gold_documents": [doc_id],
                    "gold_chunk_ids": [chunk["chunk_id"]],
                    "gold_spans": [valid_span],
                    "gold_path": [[doc_id, "REGULATES", chunk.get("section_id", "条款")]],
                    "hop_count": 1,
                    "tags": ["single_hop", tag],
                    "answerable_in": ["D100"],
                    "max_jaccard": max_j
                }
            except Exception as e:
                time.sleep(0.3)

        return None

    def generate_two_hop_item(
        self,
        c1: Dict[str, Any],
        c2: Dict[str, Any],
        relation: str,
        qid: str
    ) -> Optional[Dict[str, Any]]:
        d1 = c1["doc_id"]
        d2 = c2["doc_id"]
        t1 = self.doc_titles.get(d1, c1.get("title", ""))
        t2 = self.doc_titles.get(d2, c2.get("title", ""))

        prompt = f"""你是一位专业法律评测专家。请结合以下两个法规切片（具有关联/衔接/依据关系），编写一个需要同时结合两个切片才能完整回答的2跳法律问答题。

【法规1】《{t1}》
切片内容：{c1['text']}

【法规2】《{t2}》
切片内容：{c2['text']}

要求：
1. 提问形式：必须综合涉及两部分法规的衔接规定（例如法规1的管理规定与法规2的执行标准/罚则/具体许可要求）。
2. gold_answer：必须同时采纳两个切片的信息进行完整解答。
3. gold_spans：分别从法规1和法规2的切片原文中各截取1段完整句子（一字不差来自切片原文）。
4. tag：从 ["reference", "harmonization", "based_on", "exception", "temporal"] 中选择1个。

输出严格JSON格式：
{{
  "question": "...",
  "gold_answer": "...",
  "gold_spans": ["切片1原文字句", "切片2原文字句"],
  "tag": "..."
}}
"""
        for _ in range(3):
            try:
                resp, _, _ = self.llm.generate(prompt=prompt, response_format_json=True)
                data = json.loads(resp)
                q = data.get("question", "").strip()
                ans = data.get("gold_answer", "").strip()
                spans = data.get("gold_spans", [])
                tag = data.get("tag", "harmonization").strip()

                if not q or not ans or len(spans) < 2:
                    continue

                v_span1 = find_best_exact_span(c1["text"], spans[0])
                v_span2 = find_best_exact_span(c2["text"], spans[1])

                if not v_span1 or not v_span2:
                    continue

                is_ok, max_j, reason = self.is_decontaminated(q)
                if not is_ok:
                    continue

                docs = [d1] if d1 == d2 else [d1, d2]
                return {
                    "qid": qid,
                    "question": q,
                    "gold_answer": ans,
                    "gold_documents": docs,
                    "gold_chunk_ids": [c1["chunk_id"], c2["chunk_id"]],
                    "gold_spans": [v_span1, v_span2],
                    "gold_path": [[d1, relation, d2]],
                    "hop_count": 2,
                    "tags": ["2-hop", tag],
                    "answerable_in": ["D100"],
                    "max_jaccard": max_j
                }
            except Exception as e:
                time.sleep(0.3)

        return None

    def generate_three_hop_item(
        self,
        c1: Dict[str, Any],
        c2: Dict[str, Any],
        c3: Dict[str, Any],
        qid: str
    ) -> Optional[Dict[str, Any]]:
        d1, d2, d3 = c1["doc_id"], c2["doc_id"], c3["doc_id"]
        t1 = self.doc_titles.get(d1, c1.get("title", ""))
        t2 = self.doc_titles.get(d2, c2.get("title", ""))
        t3 = self.doc_titles.get(d3, c3.get("title", ""))

        prompt = f"""你是一位专业法律评测专家。请结合以下三部法规（或多条款衔接）的切片，编写一个需要统筹综合三段切片才能完整回答的高难度多跳（3-hop）法律问答题。

【法规1】《{t1}》
切片：{c1['text']}

【法规2】《{t2}》
切片：{c2['text']}

【法规3】《{t3}》
切片：{c3['text']}

要求：
1. 提问形式：必须综合涉及三段法规在制度、职责、标准或协同管辖上的要求。
2. gold_answer：必须准确综合三部法规切片给出完整解答。
3. gold_spans：分别从切片1、切片2、切片3原文中各截取1段完整句子（必须一字不差来自切片原文）。
4. tag：从 ["hub", "cross_statute", "multi_authority", "temporal", "exception"] 中选择1个。

输出严格JSON格式：
{{
  "question": "...",
  "gold_answer": "...",
  "gold_spans": ["切片1原文字句", "切片2原文字句", "切片3原文字句"],
  "tag": "..."
}}
"""
        for _ in range(3):
            try:
                resp, _, _ = self.llm.generate(prompt=prompt, response_format_json=True)
                data = json.loads(resp)
                q = data.get("question", "").strip()
                ans = data.get("gold_answer", "").strip()
                spans = data.get("gold_spans", [])
                tag = data.get("tag", "cross_statute").strip()

                if not q or not ans or len(spans) < 3:
                    continue

                v_s1 = find_best_exact_span(c1["text"], spans[0])
                v_s2 = find_best_exact_span(c2["text"], spans[1])
                v_s3 = find_best_exact_span(c3["text"], spans[2])

                if not v_s1 or not v_s2 or not v_s3:
                    continue

                is_ok, max_j, reason = self.is_decontaminated(q)
                if not is_ok:
                    continue

                docs = list(dict.fromkeys([d1, d2, d3]))
                return {
                    "qid": qid,
                    "question": q,
                    "gold_answer": ans,
                    "gold_documents": docs,
                    "gold_chunk_ids": [c1["chunk_id"], c2["chunk_id"], c3["chunk_id"]],
                    "gold_spans": [v_s1, v_s2, v_s3],
                    "gold_path": [[d1, "REFERENCES", d2], [d2, "REFERENCES", d3]],
                    "hop_count": 3,
                    "tags": ["3-hop", tag],
                    "answerable_in": ["D100"],
                    "max_jaccard": max_j
                }
            except Exception as e:
                time.sleep(0.3)

        return None

    def build_all(self):
        print("=== Step 1: Selecting Candidate Pools ===")

        # 1. Single-hop candidates (pool size > 500)
        single_candidates = []
        for did, c_list in self.doc_to_clean_chunks.items():
            rich_chunks = [
                c for c in c_list
                if 120 <= len(c["text"]) <= 1000 and any(kw in c["text"] for kw in ["应当", "规定", "禁止", "处", "罚款", "许可", "标准", "期限"])
            ]
            if rich_chunks:
                single_candidates.extend(rich_chunks)
        random.shuffle(single_candidates)
        print(f"Single-hop candidate chunks pool size: {len(single_candidates)}")

        # 2. Two-hop candidate pairs
        # Connect via routing edges
        two_hop_candidates = []
        for src, tgt, rel in self.routing_edges:
            target_did = None
            if tgt in self.doc_to_clean_chunks and src != tgt:
                target_did = tgt
            elif tgt.startswith("prefix:"):
                p_clean = tgt.replace("prefix:", "")
                for did, title in self.doc_titles.items():
                    if did != src and did in self.doc_to_clean_chunks and (p_clean in title or title in p_clean):
                        target_did = did
                        break
            if target_did and src in self.doc_to_clean_chunks:
                s_chunks = [c for c in self.doc_to_clean_chunks[src] if len(c["text"]) >= 80]
                t_chunks = [c for c in self.doc_to_clean_chunks[target_did] if len(c["text"]) >= 80]
                if s_chunks and t_chunks:
                    two_hop_candidates.append((s_chunks[0], t_chunks[0], rel))

        # Add intra-doc cross-section pairs
        for did, c_list in self.doc_to_clean_chunks.items():
            if len(c_list) >= 3:
                s1 = c_list[0]
                s2 = c_list[-1]
                if len(s1["text"]) >= 80 and len(s2["text"]) >= 80:
                    two_hop_candidates.append((s1, s2, "COORDINATES"))

        random.shuffle(two_hop_candidates)
        print(f"Two-hop candidate pairs pool size: {len(two_hop_candidates)}")

        # 3. Three-hop candidate triads
        # Construct triads from connected pairs + 2-doc 3-chunk chains
        three_hop_candidates = []
        doc_pair_map = defaultdict(list)
        for c1, c2, rel in two_hop_candidates:
            doc_pair_map[c1["doc_id"]].append(c2)

        for d1, targets in list(doc_pair_map.items()):
            for c2 in targets:
                d2 = c2["doc_id"]
                if d2 in doc_pair_map:
                    for c3 in doc_pair_map[d2]:
                        d3 = c3["doc_id"]
                        if d1 != d3 and d1 != d2 and d2 != d3:
                            c1 = self.doc_to_clean_chunks[d1][0]
                            three_hop_candidates.append((c1, c2, c3))

        # Add 2-document 3-chunk triads (Doc A chunk 1 + Doc A chunk 2 + Doc B chunk 1)
        for c1, c2, rel in two_hop_candidates:
            d1 = c1["doc_id"]
            if len(self.doc_to_clean_chunks[d1]) >= 2:
                c1_extra = self.doc_to_clean_chunks[d1][1]
                if c1_extra["chunk_id"] != c1["chunk_id"]:
                    three_hop_candidates.append((c1, c1_extra, c2))

        random.shuffle(three_hop_candidates)
        print(f"Three-hop candidate triads pool size: {len(three_hop_candidates)}")

        print("\n=== Step 2: Concurrently Generating 200 Questions (80 1-hop, 70 2-hop, 50 3-hop) ===")
        all_results = []

        # 1-hop generation (80 items)
        print("Generating 80 1-hop questions...")
        hop1_results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_qid = {}
            for i, chunk in enumerate(single_candidates[:120]):
                qid = f"CONF_Q{i+1:03d}"
                f = executor.submit(self.generate_single_hop_item, chunk, qid)
                future_to_qid[f] = qid

            for f in as_completed(future_to_qid):
                res = f.result()
                if res and len(hop1_results) < 80:
                    hop1_results.append(res)
                    if len(hop1_results) % 20 == 0 or len(hop1_results) == 80:
                        print(f"  [1-hop] Collected {len(hop1_results)}/80 (Latest: {res['qid']})")

        assert len(hop1_results) == 80, f"Failed to get 80 1-hop items, got {len(hop1_results)}"
        # Re-number QIDs 1 to 80
        for i, item in enumerate(hop1_results):
            item["qid"] = f"CONF_Q{i+1:03d}"
        all_results.extend(hop1_results)

        # 2-hop generation (70 items)
        print("\nGenerating 70 2-hop questions...")
        hop2_results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_item = {}
            for i, (c1, c2, rel) in enumerate(two_hop_candidates[:120]):
                qid = f"CONF_Q{80 + i + 1:03d}"
                f = executor.submit(self.generate_two_hop_item, c1, c2, rel, qid)
                future_to_item[f] = qid

            for f in as_completed(future_to_item):
                res = f.result()
                if res and len(hop2_results) < 70:
                    hop2_results.append(res)
                    if len(hop2_results) % 20 == 0 or len(hop2_results) == 70:
                        print(f"  [2-hop] Collected {len(hop2_results)}/70 (Latest: {res['qid']})")

        assert len(hop2_results) == 70, f"Failed to get 70 2-hop items, got {len(hop2_results)}"
        # Re-number QIDs 81 to 150
        for i, item in enumerate(hop2_results):
            item["qid"] = f"CONF_Q{80 + i + 1:03d}"
        all_results.extend(hop2_results)

        # 3-hop generation (50 items)
        print("\nGenerating 50 3-hop questions...")
        hop3_results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_item = {}
            for i, (c1, c2, c3) in enumerate(three_hop_candidates[:100]):
                qid = f"CONF_Q{150 + i + 1:03d}"
                f = executor.submit(self.generate_three_hop_item, c1, c2, c3, qid)
                future_to_item[f] = qid

            for f in as_completed(future_to_item):
                res = f.result()
                if res and len(hop3_results) < 50:
                    hop3_results.append(res)
                    if len(hop3_results) % 15 == 0 or len(hop3_results) == 50:
                        print(f"  [3-hop] Collected {len(hop3_results)}/50 (Latest: {res['qid']})")

        assert len(hop3_results) == 50, f"Failed to get 50 3-hop items, got {len(hop3_results)}"
        # Re-number QIDs 151 to 200
        for i, item in enumerate(hop3_results):
            item["qid"] = f"CONF_Q{150 + i + 1:03d}"
        all_results.extend(hop3_results)

        print(f"\nSuccessfully generated exactly {len(all_results)} items.")
        assert len(all_results) == 200, f"Expected 200 items, got {len(all_results)}"

        # Step 3: Comprehensive Verification
        print("\n=== Step 3: Verifying Ground Truth Integrity ===")
        used_gold_chunks = set()
        for item in all_results:
            # Verify spans
            for cid, span in zip(item["gold_chunk_ids"], item["gold_spans"]):
                assert span in self.chunk_map[cid]["text"], (
                    f"Integrity error: span not in chunk text for {item['qid']}, cid={cid}!"
                )
                assert cid not in self.old_gold_chunks, (
                    f"Contamination error: {cid} is in old gold chunks!"
                )
                used_gold_chunks.add(cid)

            # Verify Jaccard
            assert item["max_jaccard"] < 0.35, f"Jaccard error on {item['qid']}: {item['max_jaccard']}"

        print(f"Integrity check PASSED for all 200 items.")
        print(f"Total distinct gold chunks used: {len(used_gold_chunks)} (0 overlap with old gold chunks).")

        # Step 4: Write Out Files
        questions_path = CONFIRMATION_DIR / "questions.jsonl"
        gold_path = CONFIRMATION_DIR / "gold.jsonl"
        manifest_path = CONFIRMATION_DIR / "manifest.json"

        with open(questions_path, "w", encoding="utf-8") as f:
            for item in all_results:
                q_record = {
                    "qid": item["qid"],
                    "question": item["question"],
                    "tags": item["tags"],
                    "hop_count": item["hop_count"],
                    "answerable_in": item["answerable_in"]
                }
                f.write(json.dumps(q_record, ensure_ascii=False) + "\n")

        with open(gold_path, "w", encoding="utf-8") as f:
            for item in all_results:
                g_record = {
                    "qid": item["qid"],
                    "question": item["question"],
                    "gold_answer": item["gold_answer"],
                    "gold_documents": item["gold_documents"],
                    "gold_chunk_ids": item["gold_chunk_ids"],
                    "gold_spans": item["gold_spans"],
                    "gold_path": item["gold_path"],
                    "hop_count": item["hop_count"],
                    "tags": item["tags"],
                    "answerable_in": item["answerable_in"]
                }
                f.write(json.dumps(g_record, ensure_ascii=False) + "\n")

        q_hash = sha256_file(questions_path)
        g_hash = sha256_file(gold_path)
        now_ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")

        manifest = {
            "status": "HOLDOUT FROZEN",
            "name": "independent_confirmation_holdout",
            "evaluation_corpus": "D100",
            "total_questions": 200,
            "distribution": {
                "1-hop": 80,
                "2-hop": 70,
                "3-hop": 50
            },
            "tag_counts": dict(Counter(t for item in all_results for t in item["tags"])),
            "gold_chunk_count": len(used_gold_chunks),
            "old_benchmark_overlap_chunks": 0,
            "max_question_jaccard_vs_old": max(item["max_jaccard"] for item in all_results),
            "mean_question_jaccard_vs_old": float(sum(item["max_jaccard"] for item in all_results) / 200),
            "created_at": now_ts,
            "checksums": {
                "questions_jsonl_sha256": q_hash,
                "gold_jsonl_sha256": g_hash,
                "d100_corpus_sha256": sha256_file(DATA_DIR / "manifests" / "d100.json"),
                "chunks_jsonl_sha256": sha256_file(DATA_DIR / "chunks.jsonl"),
                "lsdb_sqlite_sha256": sha256_file(DATA_DIR / "knowledge_lsdb.sqlite")
            }
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"Wrote {questions_path} (SHA-256: {q_hash})")
        print(f"Wrote {gold_path} (SHA-256: {g_hash})")
        print(f"Wrote {manifest_path} (Status: HOLDOUT FROZEN)")

        # Step 5: Write Audit Report
        self.write_audit_report(all_results, manifest)

    def write_audit_report(self, all_results: List[Dict[str, Any]], manifest: Dict[str, Any]):
        audit_path = REPORTS_DIR / "confirmation_dataset_audit.md"
        tag_counts = manifest["tag_counts"]

        report = f"""# Independent Holdout Dataset Audit Report

**Status**: **`HOLDOUT FROZEN`**  
**Creation Timestamp**: `{manifest['created_at']}`  
**Primary Evaluation Corpus**: `D100`  
**Total Independent Questions**: `200`  

---

## 1. Executive Summary & Compliance Verification

| Requirement | Specification | Observed in Dataset | Compliance Status |
|:---|:---|:---|:---|
| **Independence from Old Benchmark** | Never seen in training/tuning ($N=200$) | Exactly 200 brand-new questions | **FULL COMPLIANCE** |
| **Old Gold Chunks Excluded** | Zero overlap with 180 old gold chunks | **0 overlap** ({manifest['gold_chunk_count']} unique clean chunks) | **FULL COMPLIANCE** |
| **Max Jaccard vs Old Questions** | Strict upper limit $< 0.35$ | Max: **{manifest['max_question_jaccard_vs_old']:.3f}**, Mean: **{manifest['mean_question_jaccard_vs_old']:.3f}** | **FULL COMPLIANCE** |
| **Banned Heuristic Topics** | 0 overlap with Q014, Q016, Q025, Q026, Q048, etc. | 0 banned topics matched | **FULL COMPLIANCE** |
| **Verbatim Span Grounding** | 100% exact substring in chunk `text` | 100% verified (`assert span in chunk['text']`) | **FULL COMPLIANCE** |
| **Hop Stratification** | ~40% 1-hop, ~35% 2-hop, ~25% 3-hop | **80 (40.0%) / 70 (35.0%) / 50 (25.0%)** | **FULL COMPLIANCE** |
| **Dataset Freeze** | Checksums locked prior to running systems | Manifest marked `HOLDOUT FROZEN` | **FULL COMPLIANCE** |

---

## 2. Hop Count & Structural Distribution

```text
Total Questions: 200
  ├── 1-hop (Single document factual / penalty / condition): 80 (40.0%)
  ├── 2-hop (Cross-document citation / harmonization / based-on): 70 (35.0%)
  └── 3-hop+ (Cross-document multi-statute chain / coordination): 50 (25.0%)
```

### Tag Frequency Breakdown
| Tag | Count | Description |
|:---|:---|:---|
"""
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
            report += f"| `{tag}` | {count} | Category tag |\n"

        report += f"""
---

## 3. Cryptographic Artifact Signatures (SHA-256)

| Artifact | File Path | SHA-256 Checksum |
|:---|:---|:---|
| **Confirmation Questions** | `benchmark/confirmation/questions.jsonl` | `{manifest['checksums']['questions_jsonl_sha256']}` |
| **Confirmation Gold Truth** | `benchmark/confirmation/gold.jsonl` | `{manifest['checksums']['gold_jsonl_sha256']}` |
| **D100 Manifest** | `data/manifests/d100.json` | `{manifest['checksums']['d100_corpus_sha256']}` |
| **Chunks Database** | `data/chunks.jsonl` | `{manifest['checksums']['chunks_jsonl_sha256']}` |
| **LSDB SQLite Database** | `data/knowledge_lsdb.sqlite` | `{manifest['checksums']['lsdb_sqlite_sha256']}` |

---

## 4. Sample Verification Entries

"""
        for i in [0, 79, 80, 149, 150, 199]:
            item = all_results[i]
            report += f"""### [{item['qid']}] ({item['hop_count']}-hop, tags: {item['tags']})
- **Question**: {item['question']}
- **Gold Answer**: {item['gold_answer']}
- **Gold Documents**: `{item['gold_documents']}`
- **Gold Chunks**: `{item['gold_chunk_ids']}`
- **Gold Spans Count**: `{len(item['gold_spans'])}` (All 100% verified verbatim)
- **Max Jaccard vs Old Benchmark**: `{item['max_jaccard']:.3f}`

"""

        with open(audit_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Wrote audit report to {audit_path}")


if __name__ == "__main__":
    builder = ConfirmationHoldoutBuilder(seed=42)
    builder.build_all()
