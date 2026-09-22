# Independent Holdout-2 Dataset Audit Report

**Status**: **`HOLDOUT2_FROZEN`**  
**Creation Date**: 2026-09-22 12:20:27  
**Sample Size**: $N = 250$ Unique Questions  
**Evaluation Environment**: D100 Distraction Corpus  

---

## 1. Hop Count Distribution

| Hop Count | Category | Count | Percentage |
|:---|:---|:---:|:---:|
| **1-Hop** | Simple / Single-Document Direct | 94 | 37.6% |
| **2-Hop** | Multi-Section / Cross-Statute | 100 | 40.0% |
| **3-Hop+**| Complex Multi-Authority Regulatory | 56 | 22.4% |
| **Total** | | **250** | **100.0%** |

---

## 2. Decontamination & Independence Guarantees

1. **Dual Historical Benchmark Exclusion**:
   - Total historical questions compared: 320 (72 from Dev-216 + 200 from Holdout-1).
   - Maximum Jaccard word-overlap against historical questions: strictly bounded $< 0.35$.
2. **Zero Debug Case Leakage**:
   - 10 targeted regex patterns banning historical debug topics (Q014, Q016, Q025, Q026, Q078, Q089, Q098, etc.).
   - Zero violations detected.
3. **Internal Diversity**:
   - Pairwise Jaccard similarity within Holdout-2 strictly $< 0.45$.
4. **Verbatim Ground Truth Verification**:
   - 100% of gold spans are exact verbatim substrings in source chunks (`assert span in chunk['text']`).

---

## 3. Cryptographic Hashes

- `questions.jsonl`: `a893e3d05f99ddc261541d9f20f96d8de518c33258ef74aa4b9cdac7a75741b3`
- `gold.jsonl`: `9be7df8abb2fa462adb5b485fcd3e4ff49e21cce821f37785b257e70749de1d3`
- `knowledge_lsdb.sqlite`: `5495e97f18f2e414622cca162f8f6051cf88b77e8d3aefd565dcb7950ec9bfa8`
