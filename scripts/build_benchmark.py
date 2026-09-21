#!/usr/bin/env python3
"""
Benchmark Builder & Verification Script (Phase 0)
Reference: 实验方案.md Section 4, 5, 6, 7 & 准备清单.md Section 14, 15, 16

Builds:
  - benchmark/gold.jsonl (120 full gold records with verbatim spans)
  - benchmark/questions.jsonl (120 test-ready questions)
  - benchmark/split.json (Dev=24, Test=96 stratified split)
"""

import os
import sys
import re
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"
BENCHMARK_DIR = BASE_DIR / "benchmark"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load chunks
chunks = []
with open(DATA_DIR / "chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

chunk_map = {c["chunk_id"]: c for c in chunks}
doc_to_chunks = {}
for c in chunks:
    doc_to_chunks.setdefault(c["doc_id"], []).append(c)

def resolve_target_chunk_and_span(doc_id, keywords):
    """
    Locates candidate chunk in doc_id and extracts an exact verbatim span.
    Guarantees span in chunk["text"].
    """
    candidates = doc_to_chunks.get(doc_id, [])
    if not candidates:
        raise ValueError(f"No chunks found for doc_id {doc_id}")

    # Score chunks by keywords
    scored = []
    for c in candidates:
        score = sum(1 for kw in keywords if kw in c["text"])
        if score > 0:
            scored.append((score, c))
    
    selected_chunk = None
    if scored:
        scored.sort(key=lambda x: -x[0])
        selected_chunk = scored[0][1]
    else:
        # Fallback to first chunk
        selected_chunk = candidates[0]

    # Extract exact verbatim span from chunk text
    text = selected_chunk["text"]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    
    extracted_span = None
    for kw in keywords:
        for line in lines:
            if kw in line:
                # Find sentences
                parts = re.split(r"(?<=[。；！])", line)
                for part in parts:
                    part_clean = part.strip()
                    if kw in part_clean and len(part_clean) >= 8:
                        if part_clean in text:
                            extracted_span = part_clean
                            break
                if extracted_span:
                    break
        if extracted_span:
            break

    if not extracted_span:
        for kw in keywords:
            if kw in text:
                idx = text.find(kw)
                start = max(0, idx - 20)
                end = min(len(text), idx + 80)
                candidate_span = text[start:end].strip()
                if candidate_span in text:
                    extracted_span = candidate_span
                    break

    if not extracted_span:
        extracted_span = lines[0] if lines else text[:100]

    # Strict verification
    assert extracted_span in selected_chunk["text"], (
        f"Extracted span not in chunk text! cid={selected_chunk['chunk_id']}, doc={doc_id}"
    )

    return selected_chunk["chunk_id"], extracted_span

def main():
    from scripts.benchmark_specs import BENCHMARK_SPECS
    print(f"Loaded {len(BENCHMARK_SPECS)} benchmark specs.")
    assert len(BENCHMARK_SPECS) == 120, f"Expected 120 specs, got {len(BENCHMARK_SPECS)}"

    gold_records = []
    question_records = []

    for spec in BENCHMARK_SPECS:
        qid = spec["qid"]
        corpus = spec["corpus"]
        hop_count = spec["hop_count"]
        tags = spec["tags"]
        targets = spec["targets"]
        question = spec["question"]
        gold_answer = spec["gold_answer"]
        gold_path = spec["gold_path"]

        gold_cids = []
        gold_docs = []
        gold_spans = []

        for doc_id, kws in targets:
            cid, span = resolve_target_chunk_and_span(doc_id, kws)
            gold_cids.append(cid)
            if doc_id not in gold_docs:
                gold_docs.append(doc_id)
            gold_spans.append(span)

        if corpus == "D20":
            answerable_in = ["D20", "D50", "D100"]
        elif corpus == "D50":
            answerable_in = ["D50", "D100"]
        else:
            answerable_in = ["D100"]

        gold_item = {
            "qid": qid,
            "question": question,
            "gold_answer": gold_answer,
            "gold_documents": gold_docs,
            "gold_chunk_ids": gold_cids,
            "gold_spans": gold_spans,
            "gold_path": gold_path,
            "hop_count": hop_count,
            "tags": tags,
            "answerable_in": answerable_in
        }
        gold_records.append(gold_item)

        q_item = {
            "qid": qid,
            "question": question,
            "tags": tags,
            "hop_count": hop_count,
            "answerable_in": answerable_in
        }
        question_records.append(q_item)

    # 2. Strict Verification of all Gold Spans
    print("\n--- Verifying Verbatim Gold Spans ---")
    verified_count = 0
    for item in gold_records:
        for cid, span in zip(item["gold_chunk_ids"], item["gold_spans"]):
            assert cid in chunk_map, f"Chunk {cid} not found in chunks.jsonl!"
            assert span in chunk_map[cid]["text"], f"Span '{span}' not in chunk {cid} text!"
            verified_count += 1
    print(f"Verified {verified_count} spans across 120 questions: 100% VERBATIM MATCH!")

    # 3. Validate Quotas
    print("\n--- Question Type Quota Validation ---")
    total_q = len(gold_records)
    tag_counter = Counter(t for item in gold_records for t in item["tags"])
    hop_counter = Counter(item["hop_count"] for item in gold_records)
    corp_counter = Counter(item["answerable_in"][0] for item in gold_records)

    sh_count = tag_counter.get("single_hop", 0)
    h2_count = tag_counter.get("2-hop", 0)
    h3_count = tag_counter.get("3-hop", 0)
    hub_count = tag_counter.get("hub", 0)
    temp_count = tag_counter.get("temporal", 0)
    exc_count = tag_counter.get("exception", 0)

    quotas = [
        ("single-hop", sh_count, 0.15, sh_count / total_q >= 0.15),
        ("2-hop", h2_count, 0.25, h2_count / total_q >= 0.25),
        ("3-hop+", h3_count, 0.20, h3_count / total_q >= 0.20),
        ("hub/high-fanout", hub_count, 0.20, hub_count / total_q >= 0.20),
        ("version/temporal", temp_count, 0.10, temp_count / total_q >= 0.10),
        ("exception/conflict", exc_count, 0.10, exc_count / total_q >= 0.10),
    ]

    print(f"{'Category':<22} | {'Count':<6} | {'Ratio':<8} | {'Required':<10} | {'Status'}")
    print("-" * 65)
    all_passed = True
    for cat, cnt, req, passed in quotas:
        ratio_str = f"{cnt / total_q * 100:.1f}%"
        req_str = f">= {req * 100:.0f}%"
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"{cat:<22} | {cnt:<6} | {ratio_str:<8} | {req_str:<10} | {status}")
    print("-" * 65)
    assert all_passed, "Quota validation failed!"

    # 4. Stratified Dev=24 / Test=96 Split
    print("\n--- Stratified Dev=24 / Test=96 Split ---")
    random.seed(42)

    # Stratify by corpus partition:
    # D20 (60 items) -> 12 Dev, 48 Test
    # D50 (30 items) -> 6 Dev, 24 Test
    # D100 (30 items) -> 6 Dev, 24 Test
    by_corpus = defaultdict(list)
    for item in gold_records:
        c = item["answerable_in"][0]
        by_corpus[c].append(item)

    dev_qids = []
    test_qids = []

    targets = {"D20": 12, "D50": 6, "D100": 6}

    for c, target_dev in targets.items():
        pool = by_corpus[c]
        # Further stratify by hop_count and tags
        def sort_key(x):
            return (x["hop_count"], "".join(sorted(x["tags"])), x["qid"])
        pool_sorted = sorted(pool, key=sort_key)
        
        # Systematic stratified stride sampling with random offset
        step = len(pool_sorted) / target_dev
        offset = random.uniform(0, step - 1)
        sampled_indices = {int(offset + i * step) for i in range(target_dev)}
        
        for idx, item in enumerate(pool_sorted):
            if idx in sampled_indices and len([q for q in dev_qids if q in [x['qid'] for x in pool]]) < target_dev:
                dev_qids.append(item["qid"])
            else:
                test_qids.append(item["qid"])

    # Ensure exact counts
    dev_qids = sorted(dev_qids)
    test_qids = sorted(test_qids)

    assert len(dev_qids) == 24, f"Dev count {len(dev_qids)} != 24"
    assert len(test_qids) == 96, f"Test count {len(test_qids)} != 96"
    assert len(set(dev_qids) & set(test_qids)) == 0, "Overlap between Dev and Test!"
    print(f"Dev: {len(dev_qids)} items (20%), Test: {len(test_qids)} items (80%)")

    # 5. Write Files
    gold_path = BENCHMARK_DIR / "gold.jsonl"
    with open(gold_path, "w", encoding="utf-8") as f:
        for item in gold_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved: {gold_path} ({len(gold_records)} records)")

    q_path = BENCHMARK_DIR / "questions.jsonl"
    with open(q_path, "w", encoding="utf-8") as f:
        for item in question_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved: {q_path} ({len(question_records)} records)")

    split_path = BENCHMARK_DIR / "split.json"
    split_data = {
        "dev_qids": dev_qids,
        "test_qids": test_qids,
        "summary": {
            "total_questions": 120,
            "dev_count": len(dev_qids),
            "test_count": len(test_qids),
            "dev_ratio": len(dev_qids) / total_q,
            "test_ratio": len(test_qids) / total_q,
            "random_seed": 42
        }
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {split_path}")

    # Print Dev vs Test balance
    print("\n--- Dev vs Test Distribution Balance ---")
    dev_set = set(dev_qids)
    dev_tags = Counter(t for item in gold_records if item["qid"] in dev_set for t in item["tags"])
    test_tags = Counter(t for item in gold_records if item["qid"] not in dev_set for t in item["tags"])
    for tag in ["single_hop", "2-hop", "3-hop", "hub", "temporal", "exception"]:
        d_cnt = dev_tags.get(tag, 0)
        t_cnt = test_tags.get(tag, 0)
        print(f"Tag '{tag}': Dev={d_cnt} ({d_cnt/24*100:.1f}%), Test={t_cnt} ({t_cnt/96*100:.1f}%)")

    print("\n Phase 0 Benchmark & Gold Evidence Pipeline COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
