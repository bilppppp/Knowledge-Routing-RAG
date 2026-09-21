#!/usr/bin/env python3
"""
scripts/analyze_c1.py
Detailed transition and feature analysis of C1 vs B0.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    b0_traces = {}
    for p in (PROJECT_ROOT / "runs" / "test").glob("B0_*/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            b0_traces[(d["qid"], d["corpus"])] = d

    c1_traces = {}
    for p in (PROJECT_ROOT / "runs" / "v2" / "C1").glob("C1_*/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            c1_traces[(d["qid"], d["corpus"])] = d

    with open(PROJECT_ROOT / "benchmark" / "gold.jsonl", "r", encoding="utf-8") as f:
        gold_map = {json.loads(l)["qid"]: json.loads(l) for l in f if l.strip()}

    keys = sorted(list(b0_traces.keys()))

    rescues = []
    regressions = []

    for k in keys:
        b0_corr = b0_traces[k]["is_correct"]
        c1_corr = c1_traces[k]["is_correct"]
        if not b0_corr and c1_corr:
            rescues.append(k)
        elif b0_corr and not c1_corr:
            regressions.append(k)

    print(f"Total: Rescues={len(rescues)}, Regressions={len(regressions)}")

    print("\n=======================================================")
    print(f" 11 REGRESSIONS (B0 Correct -> C1 Wrong)")
    print("=======================================================")
    for k in regressions:
        qid, corpus = k
        g = gold_map[qid]
        b0_t = b0_traces[k]
        c1_t = c1_traces[k]
        gold_chunks = set(g["gold_chunk_ids"])
        b0_chunks = set(b0_t["final_evidence_chunk_ids"])
        c1_chunks = set(c1_t["final_evidence_chunk_ids"])
        print(f"\n[{qid} - {corpus}] Hops={g['hop_count']} Tags={g['tags']}")
        print(f"  Q: {g['question']}")
        print(f"  Gold chunks: {gold_chunks}")
        print(f"  B0 chunks: {b0_chunks} (Gold in B0: {len(gold_chunks & b0_chunks)}/{len(gold_chunks)})")
        print(f"  C1 chunks: {c1_chunks} (Gold in C1: {len(gold_chunks & c1_chunks)}/{len(gold_chunks)})")

    print("\n=======================================================")
    print(f" 8 RESCUES (B0 Wrong -> C1 Correct)")
    print("=======================================================")
    for k in rescues:
        qid, corpus = k
        g = gold_map[qid]
        b0_t = b0_traces[k]
        c1_t = c1_traces[k]
        gold_chunks = set(g["gold_chunk_ids"])
        b0_chunks = set(b0_t["final_evidence_chunk_ids"])
        c1_chunks = set(c1_t["final_evidence_chunk_ids"])
        print(f"\n[{qid} - {corpus}] Hops={g['hop_count']} Tags={g['tags']}")
        print(f"  Q: {g['question']}")
        print(f"  Gold chunks: {gold_chunks}")
        print(f"  B0 chunks: {b0_chunks} (Gold in B0: {len(gold_chunks & b0_chunks)}/{len(gold_chunks)})")
        print(f"  C1 chunks: {c1_chunks} (Gold in C1: {len(gold_chunks & c1_chunks)}/{len(gold_chunks)})")

if __name__ == "__main__":
    main()
