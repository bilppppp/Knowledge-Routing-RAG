#!/usr/bin/env python3
"""
scripts/analyze_transitions.py
Deep dive into question-level transitions between B0, K2, and K4.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs" / "test"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"

def main():
    traces = []
    for p in RUNS_DIR.glob("*/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            traces.append(json.load(f))

    with open(BENCHMARK_DIR / "gold.jsonl", "r", encoding="utf-8") as f:
        gold_map = {json.loads(l)["qid"]: json.loads(l) for l in f if l.strip()}

    by_key = {}
    for t in traces:
        key = (t["qid"], t["corpus"])
        if key not in by_key:
            by_key[key] = {}
        by_key[key][t["system_id"]] = t

    keys = sorted(list(by_key.keys()))

    print("=================================================================")
    print("1. K2 RESCUES (B0 Wrong -> K2 Correct) [5 cases]")
    print("=================================================================")
    for k in keys:
        b0_t = by_key[k].get("B0")
        k2_t = by_key[k].get("K2")
        if not b0_t["is_correct"] and k2_t["is_correct"]:
            qid, corpus = k
            g = gold_map[qid]
            b0_ev = set(b0_t["final_evidence_chunk_ids"])
            k2_ev = set(k2_t["final_evidence_chunk_ids"])
            gold_ev = set(g["gold_chunk_ids"])
            print(f"\n[{qid} - {corpus}] Hops={g['hop_count']} Tags={g['tags']} Docs={g['gold_documents']}")
            print(f"  Q: {g['question']}")
            print(f"  Gold chunks: {gold_ev}")
            print(f"  B0 chunks: {b0_ev} (gold overlap: {len(b0_ev & gold_ev)}/{len(gold_ev)})")
            print(f"  K2 chunks: {k2_ev} (gold overlap: {len(k2_ev & gold_ev)}/{len(gold_ev)})")

    print("\n=================================================================")
    print("2. K2 REGRESSIONS (B0 Correct -> K2 Wrong) [7 cases]")
    print("=================================================================")
    for k in keys:
        b0_t = by_key[k].get("B0")
        k2_t = by_key[k].get("K2")
        if b0_t["is_correct"] and not k2_t["is_correct"]:
            qid, corpus = k
            g = gold_map[qid]
            b0_ev = set(b0_t["final_evidence_chunk_ids"])
            k2_ev = set(k2_t["final_evidence_chunk_ids"])
            gold_ev = set(g["gold_chunk_ids"])
            print(f"\n[{qid} - {corpus}] Hops={g['hop_count']} Tags={g['tags']} Docs={g['gold_documents']}")
            print(f"  Q: {g['question']}")
            print(f"  Gold chunks: {gold_ev}")
            print(f"  B0 chunks: {b0_ev} (gold overlap: {len(b0_ev & gold_ev)}/{len(gold_ev)})")
            print(f"  K2 chunks: {k2_ev} (gold overlap: {len(k2_ev & gold_ev)}/{len(gold_ev)})")

    print("\n=================================================================")
    print("3. K4 RESCUES (B0 Wrong -> K4 Correct) [7 cases]")
    print("=================================================================")
    for k in keys:
        b0_t = by_key[k].get("B0")
        k4_t = by_key[k].get("K4")
        if not b0_t["is_correct"] and k4_t["is_correct"]:
            qid, corpus = k
            g = gold_map[qid]
            b0_ev = set(b0_t["final_evidence_chunk_ids"])
            k4_ev = set(k4_t["final_evidence_chunk_ids"])
            gold_ev = set(g["gold_chunk_ids"])
            print(f"\n[{qid} - {corpus}] Hops={g['hop_count']} Tags={g['tags']} Docs={g['gold_documents']}")
            print(f"  Q: {g['question']}")
            print(f"  Gold chunks: {gold_ev}")
            print(f"  B0 chunks: {b0_ev} (gold overlap: {len(b0_ev & gold_ev)}/{len(gold_ev)})")
            print(f"  K4 chunks: {k4_ev} (gold overlap: {len(k4_ev & gold_ev)}/{len(gold_ev)})")
            print(f"  K4 fallbacks triggered: {k4_t.get('active_fallbacks', 0)}")

    print("\n=================================================================")
    print("4. K4 REGRESSIONS (B0 Correct -> K4 Wrong) [12 cases]")
    print("=================================================================")
    for k in keys:
        b0_t = by_key[k].get("B0")
        k4_t = by_key[k].get("K4")
        if b0_t["is_correct"] and not k4_t["is_correct"]:
            qid, corpus = k
            g = gold_map[qid]
            b0_ev = set(b0_t["final_evidence_chunk_ids"])
            k4_ev = set(k4_t["final_evidence_chunk_ids"])
            gold_ev = set(g["gold_chunk_ids"])
            print(f"\n[{qid} - {corpus}] Hops={g['hop_count']} Tags={g['tags']} Docs={g['gold_documents']}")
            print(f"  Q: {g['question']}")
            print(f"  Gold chunks: {gold_ev}")
            print(f"  B0 chunks: {b0_ev} (gold overlap: {len(b0_ev & gold_ev)}/{len(gold_ev)})")
            print(f"  K4 chunks: {k4_ev} (gold overlap: {len(k4_ev & gold_ev)}/{len(gold_ev)})")

if __name__ == "__main__":
    main()
