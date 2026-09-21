#!/usr/bin/env python3
"""
Benchmark Generation & Verification Script (Phase 0)
Reference: 实验方案.md Section 4, 5, 6, 7 & 准备清单.md Section 14, 15, 16

Features:
1. 120 benchmark questions total:
   - Core-60 (Q001~Q060): gold evidence in D20, answerable in ["D20", "D50", "D100"]
   - Extension-50 (Q061~Q090): gold evidence requires D50, answerable in ["D50", "D100"]
   - Extension-100 (Q091~Q120): gold evidence requires D100, answerable in ["D100"]
2. Question Type Quotas:
   - single-hop >= 15% (18/120)
   - 2-hop >= 25% (30/120)
   - 3-hop+ >= 20% (24/120)
   - hub >= 20% (24/120)
   - temporal >= 10% (12/120)
   - exception >= 10% (12/120)
3. Verbatim Gold Spans:
   - Every single gold_span is strictly verified to be a substring of the gold chunk.
4. Stratified Split:
   - Dev = 24 (20%), Test = 96 (80%)
   - Stratified across corpus & tags, seed=42.
5. Outputs:
   - benchmark/gold.jsonl
   - benchmark/questions.jsonl
   - benchmark/split.json
"""

import os
import re
import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
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

def find_best_chunk(doc_id, keywords, section_hint=None):
    candidates = doc_to_chunks.get(doc_id, [])
    if section_hint:
        for c in candidates:
            if section_hint in c["section_id"] or section_hint in c["heading_path"]:
                return c
    scored = []
    for c in candidates:
        score = sum(1 for kw in keywords if kw in c["text"])
        if score > 0:
            scored.append((score, c))
    if scored:
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]
    return candidates[0] if candidates else None

def extract_verbatim_span(chunk_text, query_kw, max_chars=120):
    lines = [l.strip() for l in chunk_text.splitlines() if l.strip()]
    for line in lines:
        if any(kw in line for kw in query_kw):
            # clean line
            if len(line) <= max_chars:
                return line
            # find index of keyword
            for kw in query_kw:
                if kw in line:
                    idx = line.find(kw)
                    start = max(0, idx - 20)
                    end = min(len(line), idx + max_chars - 20)
                    sub = line[start:end].strip("，、。； ")
                    if sub in chunk_text:
                        return sub
    # Fallback to first non-empty line
    return lines[0][:max_chars] if lines else chunk_text[:max_chars]

