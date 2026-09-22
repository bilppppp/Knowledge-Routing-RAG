# Corrected Corpus Freeze Manifest (CORRECTED_CORPUS_FROZEN)

**Freeze Date**: 2026-09-22  
**Target Repository**: `Knowledge-Routing-RAG`  
**Status**: `CORRECTED_CORPUS_FROZEN`  
**Prerequisite Gate**: Gate I (Verdict: `I-B — CORRECTABLE INTEGRITY ISSUES`)

---

## 1. Objective Rationale & Scope of Correction

As verified during the Gate I census, two documents in the initial 100-document corpus had corrupted metadata resulting from raw file naming errors during the initial import pipeline:
1. **`doc034`**:
   - *Previous Title*: 《医疗器械监督管理条例》
   - *Corrected Canonical Title*: 《中华人民共和国红十字会法》
   - *Physical Content*: 100% 《中华人民共和国红十字会法》（1993年通过，2009/2017修正）。
2. **`doc086`**:
   - *Previous Title*: 《关于采供血机构管理有关事宜的批复（国卫医函[2018]35号）》
   - *Corrected Canonical Title*: 《卫生部关于X射线诊断机等医用诊断设备不属于计量器具的批复（卫法监发[2002]119号）》
   - *Physical Content*: 100% 《卫生部关于X射线诊断机等医用诊断设备不属于计量器具的批复》（卫法监发[2002]119号）。

No text content, chunking boundaries, embedding weights, graph edges, BM25 tokenizer rules, router algorithms, or composer parameters were altered. Only the objectively corrupted metadata labels were corrected across all repository data layers.

---

## 2. Cryptographic Hashes and Freeze Record

```text
================================================================================
CORRECTED CORPUS CRYPTOGRAPHIC RECORD:
--------------------------------------------------------------------------------
Raw Corpus Hash (data/documents/*.txt):
  236a3aff7c898f045378a1177dbebb7e7b43772ad440b5307ed21a7d5f7b2fd3

Corpus Manifest Hash (data/manifests/corpus_manifest.json):
  bb21f21fb3f4785c23ac82e18d926e5185252d498d1f0a1da26dad8325add47f

Chunks JSONL Hash (data/chunks.jsonl):
  d3c054be36cd5c805b9d6f0612aa2771990aa3bd890551ca72579d9d3f32513c

Graph Hash (data/knowledge_lsdb.sqlite nodes + edges):
  f744713290c07c7c2eab9fb95011ba07682df0a6157d4f230140aad77b4a66f1

Vector Index Version (Qdrant):
  Collection: knowledge_routing_exp
  Points: 2862
  Vector Dimensions: 1024 (bge-large-zh-v1.5)
  Distance Metric: Cosine

FTS Index Version (SQLite FTS5):
  Table: chunks_fts
  Indexed Documents: 2862
================================================================================
```

---

## 3. Synchronized Stores

The following storage tiers have been verified as fully synchronized with the corrected canonical titles:
1. **Manifest File**: `data/manifests/corpus_manifest.json` (100 documents).
2. **Chunk Archive**: `data/chunks.jsonl` (2,862 chunks).
3. **SQLite Relational Store**: `data/knowledge_lsdb.sqlite` tables `documents`, `nodes`, and `chunks`.
4. **SQLite Full-Text Search**: `data/knowledge_lsdb.sqlite` virtual table `chunks_fts`.
5. **Vector Store**: Qdrant collection `knowledge_routing_exp` payloads for `doc034` (45 points) and `doc086` (1 point).

From this timestamp forward, all experiments are bound to this `CORRECTED_CORPUS_FROZEN` state.
