# HubSet-1 Dataset Integrity & Stratification Audit Report

**Date**: 2026-09-22T15:38:12+0800  
**Status**: `HUBSET1_FROZEN`  
**Phase**: Phase D — Hub / High-Fanout Stress Characterization  
**Total QIDs**: 100  

---

## 1. Executive Summary & Verification Guarantees

1. **Strict Stratified Degree Distribution**:
   - Exactly 5 pre-registered degree buckets, balanced at **20 questions per bucket** ($N = 100$).
   - All critical hub degrees computed directly from frozen `G_routing` in `knowledge_lsdb.sqlite`.
2. **Decontamination Guarantees**:
   - Audited against all 650 historical benchmark items (Dev-216, Holdout-1, Holdout-2, ScaleSet-1).
   - Maximum Jaccard similarity across all historical items strictly $< 0.35$.
   - Banned pattern matches: **0**.
   - Historical gold chunk overlap: **0**.
3. **Verifiable Ground Truth**:
   - $100\%$ of gold spans (198 total spans) verified as exact verbatim substrings within designated chunks (`span in chunk['text']`).
   - Independent LLM judge verified $100\%$ answerability using only gold chunks.
4. **Decoupled Hop & Degree Topology**:
   - Hop counts and hub degrees independently sampled to prevent confounding multi-hop reasoning difficulty with graph degree.

---

## 2. Degree Bucket Distribution

| Degree Bucket | Degree Range | Question Count | Critical Node Examples |
|---|---|---|---|
| **Bucket A** | Degree < 5 | 20 | doc003, doc007, doc009, doc012, doc014, doc020 |
| **Bucket B** | Degree 5–10 | 20 | doc001, doc004, doc008, doc011, doc016, doc018, doc019 |
| **Bucket C** | Degree 11–20 | 20 | doc002, doc005, doc015, doc034, doc045, doc073 |
| **Bucket D** | Degree 21–50 | 20 | doc025, doc026, doc056, doc061, doc063, doc065 |
| **Bucket E** | Degree > 50 | 20 | doc010 (deg 58), doc033 (deg 74) |
| **Total** | — | **100** | **Balanced across 5 tiers** |

---

## 3. Hop Count Distribution

| Hop Count | Number of Questions | Percentage |
|---|---|---|
| **1-Hop** | 27 | 27.0% |
| **2-Hop** | 48 | 48.0% |
| **3-Hop+** | 25 | 25.0% |

---

## 4. Cryptographic Signatures

```json
{
  "questions_sha256": "aff5e827ec164d7120a8e2a377a423f2e26ede4870236ff2ff67bb1cc568e8ef",
  "gold_sha256": "87f278d05820d04e1d7abedb3118a7378b0c5e24a291441957fb68452c2df5a7",
  "d100_manifest_sha256": "852865dc239b65c10fc8e99b0469f43886b624f315ca6060eeb26a4666f0fd22"
}
```

**AUDIT VERDICT**: **`APPROVED & FROZEN FOR PHASE D CONFIRMATION`**
