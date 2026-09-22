# ScaleSet-1 Dataset Integrity & Decontamination Audit Report

**Date**: 2026-09-22T15:05:33+0800  
**Status**: `SCALESET1_FROZEN`  
**Evaluation Scope**: Phase C — Scale & Distractor Robustness Confirmation ($D_{20} \subset D_{50} \subset D_{100}$)  
**Total QIDs**: 80  

---

## 1. Executive Summary & Verification Guarantees

1. **Strict Corpus Containment in $D_{20}$**:
   - $100\%$ of gold documents belong to $D_{20}$ (`doc001` through `doc020`).
   - $100\%$ of gold chunks (160 chunk instances) reside strictly in $D_{20}$.
   - $100\%$ of questions are answerable in $D_{20}$, $D_{50}$, and $D_{100}$ with the exact same gold answer.
2. **Decontamination Guarantees**:
   - Evaluated against all 570 historical benchmark items (Dev-216, Holdout-1, Holdout-2).
   - Maximum Jaccard similarity across all historical items: $< 0.35$.
   - Banned pattern matches: **0**.
   - Historical gold chunk overlap: **0**.
3. **Ground Truth Verifiability**:
   - Every single gold span (160 total spans) was verified as an exact verbatim substring within its designated chunk (`span in chunk['text']`).
   - Independent LLM answerability verification confirmed $100\%$ answerability using only the gold evidence chunks.
4. **Distractor Invariance**:
   - Audited against $D_{21}-D_{100}$: No distractor documents supersede or alter the legal conclusions of the questions.

---

## 2. Hop Count & Topology Distribution

| Hop Count | Number of Questions | Target Percentage | Actual Percentage | Description |
|---|---|---|---|---|
| **1-hop** | 24 | ~30% | 30.0% | Single-document, single-provision direct inquiries |
| **2-hop** | 32 | ~40% | 40.0% | Same-doc multi-section & cross-statute harmonization |
| **3-hop+** | 24 | ~30% | 30.0% | Multi-statute, multi-tier composite reasoning chains |
| **Total** | **80** | **100%** | **100.0%** | Comprehensive benchmark |

---

## 3. Tag & Facet Coverage

| Tag Facet | Question Count | Description |
|---|---|---|
| `high_fanout_hub` | 43 | Regulatory facet coverage |
| `cross_doc_reference` | 42 | Regulatory facet coverage |
| `2-hop` | 32 | Regulatory facet coverage |
| `1-hop` | 24 | Regulatory facet coverage |
| `3-hop+` | 24 | Regulatory facet coverage |
| `same_doc_multi_section` | 14 | Regulatory facet coverage |
| `direct_factual` | 11 | Regulatory facet coverage |
| `procedure` | 6 | Regulatory facet coverage |
| `qualification_condition` | 5 | Regulatory facet coverage |
| `sanction_consequence` | 4 | Regulatory facet coverage |
| `exception` | 4 | Regulatory facet coverage |

---

## 4. D20 Document Coverage

Total D20 documents covered: **20 / 20** (100% coverage).

| Doc ID | Statutory Title | Question Count |
|---|---|---|
| `doc001` | 《中华人民共和国食品安全法》 | 6 |
| `doc002` | 《中华人民共和国药品管理法》 | 12 |
| `doc003` | 《中华人民共和国疫苗管理法》 | 3 |
| `doc004` | 《中华人民共和国职业病防治法》 | 4 |
| `doc005` | 《中华人民共和国传染病防治法》 | 15 |
| `doc006` | 《麻醉药品和精神药品管理条例》 | 4 |
| `doc007` | 《中华人民共和国基本医疗卫生与健康促进法》 | 14 |
| `doc008` | 《病原微生物实验室生物安全管理条例》 | 6 |
| `doc009` | 《中华人民共和国生物安全法》 | 9 |
| `doc010` | 《医疗机构管理条例实施细则》 | 14 |
| `doc011` | 《放射性同位素与射线装置安全和防护条例》 | 2 |
| `doc012` | 《中华人民共和国精神卫生法》 | 4 |
| `doc013` | 《中华人民共和国食品安全法实施条例》 | 3 |
| `doc014` | 《中华人民共和国医师法》 | 14 |
| `doc015` | 《医疗事故处理条例》 | 6 |
| `doc016` | 《艾滋病防治条例》 | 4 |
| `doc017` | 《血吸虫病防治条例》 | 6 |
| `doc018` | 《医疗纠纷预防和处理条例》 | 6 |
| `doc019` | 《医疗废物管理条例》 | 7 |
| `doc020` | 《中华人民共和国中医药法》 | 7 |

---

## 5. Cryptographic Signatures

```json
{
  "questions_sha256": "510ad0768746069a58729619bace988d53252b3c535f9ed71522eba38df396c8",
  "gold_sha256": "1fd9c31a6521c4db76cb8ffcba150b6ec279da232dc30160b727774f3fec652f",
  "d20_manifest_sha256": "e0e6b19097853c2856d86bc0b3ee8ac982a589ca1b6a256f0941f05b3f8e97e4",
  "d50_manifest_sha256": "9673555854eb93180fbb530af50cd7a97b17c3ac25d7b5ae6071744b79c9a807",
  "d100_manifest_sha256": "852865dc239b65c10fc8e99b0469f43886b624f315ca6060eeb26a4666f0fd22"
}
```

**AUDIT VERDICT**: **`APPROVED & FROZEN FOR PHASE C CONFIRMATION`**
