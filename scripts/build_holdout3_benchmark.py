#!/usr/bin/env python3
"""
scripts/build_holdout3_benchmark.py
Mechanism Holdout-3 Benchmark Generator.
Constructs N=240 completely new, unseen questions from the corrected frozen D100 corpus.

Experimental Strata:
  - Q-E (Explicit): 80 questions (all target document names explicitly in prompt).
  - Q-P (Partial): 80 questions (only 1 starting statute in prompt; target statute discovered via relations).
  - Q-I (Implicit): 80 questions (no statute names in prompt; realistic real-world legal inquiry).

Strict Guarantees:
  1. Complete Decontamination:
     - All historical questions (Dev N=120, Holdout-1 N=200, Holdout-2 N=250, ScaleSet-1 N=80, HubSet-1 N=100) excluded.
     - Maximum Jaccard similarity strictly < 0.35 against all historical questions.
  2. Ground Truth Verifiability:
     - Exact verbatim substring assertions (assert span in chunk['text']).
  3. Grounded in Corrected Graph:
     - Gold paths exist in corrected frozen knowledge_lsdb.sqlite edges table.
  4. Cryptographic Manifest & Freeze:
     - Outputs benchmark/holdout3/questions.jsonl, gold.jsonl, manifest.json.
     - Status: HOLDOUT3_FROZEN.
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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Any, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.llm import LLMService

HOLDOUT3_DIR = PROJECT_ROOT / "benchmark" / "holdout3"
HOLDOUT3_DIR.mkdir(parents=True, exist_ok=True)


def get_tokens(text: str) -> Set[str]:
    return set(w for w in jieba.lcut(text) if len(w.strip()) > 1)


def calc_jaccard(s1: Set[str], s2: Set[str]) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def find_best_exact_span(chunk_text: str, candidate_span: str) -> Optional[str]:
    if not candidate_span:
        return None
    candidate_span = candidate_span.strip().strip('"\'“”‘’')
    if candidate_span in chunk_text and len(candidate_span) >= 10:
        return candidate_span

    # Try sentence splitting
    sentences = re.split(r"(?<=[。；！\n])", chunk_text)
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) >= 10 and (s_clean in candidate_span or candidate_span in s_clean):
            if s_clean in chunk_text:
                return s_clean

    words = jieba.lcut(candidate_span)
    meaningful_words = [w for w in words if len(w.strip()) >= 2]
    scored = []
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) >= 10:
            score = sum(1 for w in meaningful_words if w in s_clean)
            scored.append((score, s_clean))
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_sent = scored[0]
        if best_score >= 3 and best_sent in chunk_text:
            return best_sent

    return None


class Holdout3Builder:
    def __init__(self):
        self.llm = LLMService()
        self.load_historical_benchmarks()
        self.load_corpus_and_graph()

    def load_historical_benchmarks(self):
        self.excluded_questions = []
        historical_paths = [
            PROJECT_ROOT / "benchmark" / "gold.jsonl",
            PROJECT_ROOT / "benchmark" / "confirmation" / "gold.jsonl",
            PROJECT_ROOT / "benchmark" / "holdout2" / "gold.jsonl",
            PROJECT_ROOT / "benchmark" / "scale1" / "gold.jsonl",
            PROJECT_ROOT / "benchmark" / "hub1" / "gold.jsonl"
        ]
        for p in historical_paths:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            self.excluded_questions.append(d["question"])
        self.historical_token_sets = [get_tokens(q) for q in self.excluded_questions]
        print(f"Loaded {len(self.excluded_questions)} historical questions for decontamination checks.")

    def load_corpus_and_graph(self):
        with open(PROJECT_ROOT / "data" / "manifests" / "corpus_manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.doc_titles = {d["doc_id"]: d["title"] for d in manifest if d.get("in_d100", True)}
        self.short_titles = {
            did: t.replace("中华人民共和国", "")
            for did, t in self.doc_titles.items()
        }

        conn = sqlite3.connect(PROJECT_ROOT / "data" / "knowledge_lsdb.sqlite")
        cur = conn.cursor()

        # Load chunks
        cur.execute("SELECT chunk_id, doc_id, title, heading_path, text FROM chunks WHERE in_d100 = 1")
        self.chunks_by_id = {}
        self.chunks_by_doc = defaultdict(list)
        for cid, did, title, hp, txt in cur.fetchall():
            item = {"chunk_id": cid, "doc_id": did, "title": title, "heading_path": hp, "text": txt}
            self.chunks_by_id[cid] = item
            if len(txt) >= 80 and any(kw in txt for kw in ["应当", "规定", "禁止", "处", "罚款", "许可", "条件", "职责", "要求"]):
                self.chunks_by_doc[did].append(item)

        # Load cross-doc routing edges in D100
        cur.execute("""
            SELECT source, target, relation 
            FROM edges 
            WHERE is_routing = 1 AND source LIKE 'doc%' AND target LIKE 'doc%'
        """)
        raw_edges = cur.fetchall()
        conn.close()

        self.candidate_pairs = []
        for s, t, rel in raw_edges:
            s_did = s.split("#")[0]
            t_did = t.split("#")[0]
            if s_did != t_did and s_did in self.doc_titles and t_did in self.doc_titles:
                s_chunks = [self.chunks_by_id[s]] if "#" in s and s in self.chunks_by_id else self.chunks_by_doc[s_did]
                t_chunks = [self.chunks_by_id[t]] if "#" in t and t in self.chunks_by_id else self.chunks_by_doc[t_did]
                if s_chunks and t_chunks:
                    self.candidate_pairs.append({
                        "source_doc": s_did,
                        "target_doc": t_did,
                        "relation": rel,
                        "source_chunks": s_chunks,
                        "target_chunks": t_chunks
                    })
        print(f"Loaded {len(self.candidate_pairs)} candidate routing pairs for Holdout-3 construction.")

    def is_decontaminated(self, question: str, current_token_sets: List[Set[str]]) -> Tuple[bool, float, str]:
        q_tokens = get_tokens(question)
        if len(q_tokens) < 5:
            return False, 1.0, "Question too short"

        max_hist_j = 0.0
        for h_set in self.historical_token_sets:
            j = calc_jaccard(q_tokens, h_set)
            if j > max_hist_j:
                max_hist_j = j
                if max_hist_j >= 0.35:
                    return False, max_hist_j, "Violates historical decontamination threshold (J >= 0.35)"

        for c_set in current_token_sets:
            j = calc_jaccard(q_tokens, c_set)
            if j >= 0.40:
                return False, j, "Violates intra-dataset diversity threshold (J >= 0.40)"

        return True, max_hist_j, "Pass"

    def generate_qe_item(self, pair: Dict[str, Any], qid: str, current_tokens: List[Set[str]]) -> Optional[Dict[str, Any]]:
        """Q-E: Explicit stratum. Both target documents named in prompt."""
        d1, d2, rel = pair["source_doc"], pair["target_doc"], pair["relation"]
        t1, t2 = self.doc_titles[d1], self.doc_titles[d2]
        c1 = random.choice(pair["source_chunks"])
        c2 = random.choice(pair["target_chunks"])

        prompt = f"""你是一位专业法律评测专家。请结合以下两部法规的切片内容，编写一道高水准多跳法律问答题。

【法规1】《{t1}》
切片：{c1['text']}

【法规2】《{t2}》
切片：{c2['text']}

【必须严格遵守的规则】：
1. 题干显式度（Explicit）：题干中【必须明确写出】《{t1}》和《{t2}》两部法规的全称（或公认规范简称）。
2. 问答深度：问题必须同时涉及两部法规在制度衔接、职责分工、法律责任或执行依据上的具体要求，只有综合两个切片才能完整回答。
3. gold_answer：依据切片内容给出客观、准确、详尽的解答。
4. gold_spans：分别从切片1和切片2中截取1段原文字句（必须一字不差截取自切片原文，长度>=10字）。
5. 可以根据需要指明具体的条款号（如“第十条”），也可以不指明。

请输出严格JSON格式：
{{
  "question": "根据《...》第X条与《...》第Y条...",
  "gold_answer": "...",
  "gold_spans": ["来自切片1的原文句子", "来自切片2的原文句子"]
}}
"""
        for _ in range(3):
            try:
                resp, _, _ = self.llm.generate(prompt=prompt, response_format_json=True)
                data = json.loads(resp)
                q = data.get("question", "").strip()
                ans = data.get("gold_answer", "").strip()
                spans = data.get("gold_spans", [])

                if not q or not ans or len(spans) < 2:
                    continue

                # Check explicitness: both titles must be in question
                st1, st2 = self.short_titles[d1], self.short_titles[d2]
                has_t1 = (t1 in q or st1 in q)
                has_t2 = (t2 in q or st2 in q)
                if not (has_t1 and has_t2):
                    continue

                v_s1 = find_best_exact_span(c1["text"], spans[0])
                v_s2 = find_best_exact_span(c2["text"], spans[1])
                if not v_s1 or not v_s2:
                    continue

                ok, _, _ = self.is_decontaminated(q, current_tokens)
                if not ok:
                    continue

                article_exp = bool(re.search(r"第[一二三四五六七八九十百0-9]+条", q))
                return {
                    "qid": qid,
                    "question": q,
                    "gold_answer": ans,
                    "gold_documents": [d1, d2],
                    "gold_chunk_ids": [c1["chunk_id"], c2["chunk_id"]],
                    "gold_spans": [v_s1, v_s2],
                    "gold_path": [[d1, rel, d2]],
                    "required_relations": [rel],
                    "hop_count": 2,
                    "doc_explicitness": "E",
                    "article_explicit": article_exp,
                    "answerable_in": ["D100"]
                }
            except Exception:
                time.sleep(0.2)
        return None

    def generate_qp_item(self, pair: Dict[str, Any], qid: str, current_tokens: List[Set[str]]) -> Optional[Dict[str, Any]]:
        """Q-P: Partial stratum. Only source document named; target document omitted."""
        d1, d2, rel = pair["source_doc"], pair["target_doc"], pair["relation"]
        t1, t2 = self.doc_titles[d1], self.doc_titles[d2]
        c1 = random.choice(pair["source_chunks"])
        c2 = random.choice(pair["target_chunks"])

        prompt = f"""你是一位专业法律评测专家。请结合以下两部法规切片，编写一道【部分显式（Partial Explicitness）】的多跳法律问答题。

【起始法规】《{t1}》
切片：{c1['text']}

【关联法规】《{t2}》（关系：{rel}）
切片：{c2['text']}

【必须严格遵守的绝对规则】：
1. 题干【只能显式提及起始法规《{t1}》】！
2. 题干【严禁出现关联法规名称《{t2}》或其任何规范简称】！关联法规必须通过法律关系推导发现！
3. 提问方式：从《{t1}》的规定出发，询问其引用的相关规定、上位法职责、前置审批要求或衔接罚则。例如：“根据《{t1}》，对于某某行为，若要追究某某责任或依据相关专门法规实施行政处罚，应当满足哪些条件/由谁负责？”
4. gold_answer：必须准确综合起始法规和关联法规两个切片的内容给出完整解答。
5. gold_spans：分别从切片1和切片2中截取1段原文字句（必须一字不差截取自切片原文，长度>=10字）。

请输出严格JSON格式：
{{
  "question": "根据《{t1}》关于...的规定，相关事项如何衔接/具体要求是什么？",
  "gold_answer": "...",
  "gold_spans": ["来自切片1的原文句子", "来自切片2的原文句子"]
}}
"""
        for _ in range(3):
            try:
                resp, _, _ = self.llm.generate(prompt=prompt, response_format_json=True)
                data = json.loads(resp)
                q = data.get("question", "").strip()
                ans = data.get("gold_answer", "").strip()
                spans = data.get("gold_spans", [])

                if not q or not ans or len(spans) < 2:
                    continue

                # Check partial explicitness: t1 MUST be present, t2 MUST NOT be present
                st1, st2 = self.short_titles[d1], self.short_titles[d2]
                has_t1 = (t1 in q or st1 in q)
                has_t2 = (t2 in q or st2 in q)
                if not has_t1 or has_t2:
                    continue

                v_s1 = find_best_exact_span(c1["text"], spans[0])
                v_s2 = find_best_exact_span(c2["text"], spans[1])
                if not v_s1 or not v_s2:
                    continue

                ok, _, _ = self.is_decontaminated(q, current_tokens)
                if not ok:
                    continue

                article_exp = bool(re.search(r"第[一二三四五六七八九十百0-9]+条", q))
                return {
                    "qid": qid,
                    "question": q,
                    "gold_answer": ans,
                    "gold_documents": [d1, d2],
                    "gold_chunk_ids": [c1["chunk_id"], c2["chunk_id"]],
                    "gold_spans": [v_s1, v_s2],
                    "gold_path": [[d1, rel, d2]],
                    "required_relations": [rel],
                    "hop_count": 2,
                    "doc_explicitness": "P",
                    "article_explicit": article_exp,
                    "answerable_in": ["D100"]
                }
            except Exception:
                time.sleep(0.2)
        return None

    def generate_qi_item(self, pair: Dict[str, Any], qid: str, current_tokens: List[Set[str]]) -> Optional[Dict[str, Any]]:
        """Q-I: Implicit stratum. NO statute names in prompt. Realistic case/consultation inquiry."""
        d1, d2, rel = pair["source_doc"], pair["target_doc"], pair["relation"]
        t1, t2 = self.doc_titles[d1], self.doc_titles[d2]
        c1 = random.choice(pair["source_chunks"])
        c2 = random.choice(pair["target_chunks"])

        prompt = f"""你是一位专业法律评测专家。请结合以下两部法规切片，编写一道【完全隐式（Implicit Explicitness）】的真实法律实务咨询题。

【法规切片1】
切片内容：{c1['text']}

【法规切片2】
切片内容：{c2['text']}

【必须严格遵守的绝对规则】：
1. 题干【严禁出现任何一部法规的名称或全称】！绝对不得出现《{t1}》或《{t2}》！
2. 提问形式：必须完全以卫生健康行政执法人员、医疗机构管理者或患者律师面临的【真实具体业务咨询、合规审查或行政争议案例】形式提问！
   例如：“某基层卫生院在开展某某活动时...发生某某纠纷，按国家现行法律法规要求，该卫生院应当履行哪些法定程序？主管行政机关有权实施何种监管或处罚措施？”
3. 禁止制造不可回答的谜语：问题中的事实细节必须与切片涉及的法定概念紧密对应，真实反映两项法规制度交叉时的实务问答。
4. gold_answer：必须综合两段切片中的规定给出权威法律实务解答。
5. gold_spans：分别从切片1和切片2中截取1段原文字句（必须一字不差截取自切片原文，长度>=10字）。

请输出严格JSON格式：
{{
  "question": "某医疗卫生机构在开展...时发生...，请问相关责任人应当履行何种法定程序？依法可能面临哪些监管措施或法律责任？",
  "gold_answer": "...",
  "gold_spans": ["来自切片1的原文句子", "来自切片2的原文句子"]
}}
"""
        for _ in range(3):
            try:
                resp, _, _ = self.llm.generate(prompt=prompt, response_format_json=True)
                data = json.loads(resp)
                q = data.get("question", "").strip()
                ans = data.get("gold_answer", "").strip()
                spans = data.get("gold_spans", [])

                if not q or not ans or len(spans) < 2:
                    continue

                # Check implicit: NEITHER title nor short title can appear in question!
                st1, st2 = self.short_titles[d1], self.short_titles[d2]
                if (t1 in q or st1 in q or t2 in q or st2 in q or "《" in q):
                    continue

                v_s1 = find_best_exact_span(c1["text"], spans[0])
                v_s2 = find_best_exact_span(c2["text"], spans[1])
                if not v_s1 or not v_s2:
                    continue

                ok, _, _ = self.is_decontaminated(q, current_tokens)
                if not ok:
                    continue

                article_exp = bool(re.search(r"第[一二三四五六七八九十百0-9]+条", q))
                return {
                    "qid": qid,
                    "question": q,
                    "gold_answer": ans,
                    "gold_documents": [d1, d2],
                    "gold_chunk_ids": [c1["chunk_id"], c2["chunk_id"]],
                    "gold_spans": [v_s1, v_s2],
                    "gold_path": [[d1, rel, d2]],
                    "required_relations": [rel],
                    "hop_count": 2,
                    "doc_explicitness": "I",
                    "article_explicit": article_exp,
                    "answerable_in": ["D100"]
                }
            except Exception:
                time.sleep(0.2)
        return None

    def generate_stratum_parallel(self, stratum: str, target: int, seed: int, start_qid: int) -> List[Dict[str, Any]]:
        print(f"\n--- Parallel Generating {target} Q-{stratum} questions ---")
        pairs = list(self.candidate_pairs)
        random.seed(seed)
        random.shuffle(pairs)

        gen_func = {
            "E": self.generate_qe_item,
            "P": self.generate_qp_item,
            "I": self.generate_qi_item
        }[stratum]

        results = []
        token_sets = []
        pair_idx = 0

        with ThreadPoolExecutor(max_workers=8) as executor:
            while len(results) < target and pair_idx < len(pairs) * 6:
                batch_pairs = []
                for _ in range(16):
                    batch_pairs.append(pairs[pair_idx % len(pairs)])
                    pair_idx += 1

                futures = [executor.submit(gen_func, p, f"H3_{start_qid + len(results) + i:03d}", token_sets) for i, p in enumerate(batch_pairs)]
                for fut in as_completed(futures):
                    it = fut.result()
                    if it and len(results) < target:
                        it["qid"] = f"H3_{start_qid + len(results):03d}"
                        results.append(it)
                        token_sets.append(get_tokens(it["question"]))
                        if len(results) % 10 == 0 or len(results) == target:
                            print(f"[Q-{stratum}] Collected {len(results)} / {target} items...")

        return results

    def build_dataset(self, target_per_stratum: int = 80):
        print(f"\n=======================================================")
        print(f"Building Mechanism Holdout-3 Benchmark (Target = {target_per_stratum * 3})")
        print(f"Stratum Allocation: {target_per_stratum} Q-E + {target_per_stratum} Q-P + {target_per_stratum} Q-I")
        print(f"=======================================================")

        qe_items = self.generate_stratum_parallel("E", target_per_stratum, seed=101, start_qid=1)
        qp_items = self.generate_stratum_parallel("P", target_per_stratum, seed=202, start_qid=len(qe_items) + 1)
        qi_items = self.generate_stratum_parallel("I", target_per_stratum, seed=303, start_qid=len(qe_items) + len(qp_items) + 1)

        items = qe_items + qp_items + qi_items
        for idx, it in enumerate(items, 1):
            it["qid"] = f"H3_{idx:03d}"

        print(f"\nTotal Holdout-3 items generated: {len(items)}")
        counts = {
            "E": sum(1 for it in items if it["doc_explicitness"] == "E"),
            "P": sum(1 for it in items if it["doc_explicitness"] == "P"),
            "I": sum(1 for it in items if it["doc_explicitness"] == "I")
        }
        print("Strata Breakdown:", counts)

        # Save files
        questions_path = HOLDOUT3_DIR / "questions.jsonl"
        gold_path = HOLDOUT3_DIR / "gold.jsonl"
        manifest_path = HOLDOUT3_DIR / "manifest.json"

        with open(gold_path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

        with open(questions_path, "w", encoding="utf-8") as f:
            for it in items:
                q_rec = {
                    "qid": it["qid"],
                    "question": it["question"],
                    "doc_explicitness": it["doc_explicitness"],
                    "article_explicit": it["article_explicit"],
                    "hop_count": it["hop_count"]
                }
                f.write(json.dumps(q_rec, ensure_ascii=False) + "\n")

        h_sha = hashlib.sha256()
        with open(gold_path, "rb") as f:
            while chunk := f.read(8192):
                h_sha.update(chunk)
        gold_hash = h_sha.hexdigest()

        manifest_data = {
            "dataset_name": "Mechanism Holdout-3",
            "status": "HOLDOUT3_FROZEN",
            "creation_date": "2026-09-22",
            "total_questions": len(items),
            "strata": counts,
            "gold_sha256": gold_hash,
            "target_corpus": "D100 (Corrected Frozen Corpus)",
            "article_explicit_count": sum(1 for it in items if it["article_explicit"])
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        print(f"Saved Holdout-3: questions.jsonl, gold.jsonl, manifest.json. SHA256: {gold_hash}")
        return items


if __name__ == "__main__":
    builder = Holdout3Builder()
    builder.build_dataset(target_per_stratum=80)
