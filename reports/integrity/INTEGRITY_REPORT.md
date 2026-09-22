# Independent Data and Evaluation Integrity Audit Report (Gate I)

**Audit Date**: 2026-09-22  
**Target Repository**: `https://github.com/bilppppp/Knowledge-Routing-RAG`  
**Evaluation Target**: Frozen Core Claims, Corpus Metadata, Provenance, Benchmark Explicitness, and Runner Integrity  
**Auditor**: Independent Data & Evaluation Integrity Auditor  
**Artifact Directory**: `reports/integrity/`

---

# Executive Summary & Gate Verdict

```text
================================================================================
GATE I INTEGRITY VERDICT:
I-B — CORRECTABLE INTEGRITY ISSUES
================================================================================
```

### Verdict Justification
1. **No Material Falsification (Not I-C)**:
   - **Chunk Text Provenance**: 2,862 of 2,862 chunks (**100.0%**) are genuine substrings physically present in the source files. No chunks were hallucinated or fabricated.
   - **Gold Span Authenticity**: 1,425 of 1,425 gold spans (**100.0%**) across all 5 benchmark suites match source chunks verbatim.
   - **Holdout-2 Gain Survival**: On the decontaminated Holdout-2 dataset ($N=248$), the core accuracy gain is **$+5.65\text{pp}$** (vs. $+5.60\text{pp}$ on $N=250$). The two contaminated questions did not contribute to any transitions ($T\to T$ and $F\to F$).
2. **Defects Requiring Formal Demotion and Correction (Not I-A)**:
   - **Two Document Swaps**: `doc034` (manifest says 《医疗器械监督管理条例》, raw text is 《中华人民共和国红十字会法》) and `doc086` (manifest says 《关于采供血机构管理有关事宜的批复》, raw text is 《关于X射线诊断机...的批复》).
   - **10 Contaminated Benchmark Questions**: 10 questions across 4 benchmark suites reference these misattributed documents.
   - **Runner Metric Invariant**: In `scripts/run_holdout2_experiment.py`, `same_evidence_flips = 0` was enforced by short-circuit caching (`if h1_cids == h0_cids: h1_ans = h0_ans`), not observed empirically.
   - **Eviction Claim Demotion**: On Holdout-2, Composer had **10 useful evictions (4.00%)**, refuting earlier claims of "universal 0% eviction".
   - **Benchmark Explicitness**: 94.87% of multi-hop questions explicitly specify the statutes in their question text (Tiers E2 and E3), which accounts for 100% of the net multi-hop gain.

---

# 1. Full 100-Document Corpus Metadata Audit

A 100% census was conducted across all 100 corpus documents in `data/documents/` comparing:
- Manifest title (`data/manifests/corpus_manifest.json`)
- Raw text contents and canonical headings
- SQLite database titles (`data/knowledge_lsdb.sqlite`)
- Chunk titles and `heading_path` roots for all 2,862 chunks

### 1.1 Census Summary

| Category | Count | Percentage | Status |
|:---|:---:|:---:|:---|
| **Fully Verified Canonical Matches** | 94 | 94.0% | `PASS` |
| **Composite Multi-Reply Collections** (`doc026`, `doc042`, `doc045`, `doc051`) | 4 | 4.0% | `VERIFIED_COLLECTION` |
| **Fatal Content-Title Mismatches** (`doc034`, `doc086`) | 2 | 2.0% | `CONTAMINATED_CORPUS_SWAP` |
| **Total** | **100** | **100.0%** | **100% Census Complete** |

### 1.2 Detail on Fatal Mismatches
1. **`doc034`**:
   - *Manifest / SQLite / Chunk Title*: `医疗器械监督管理条例`
   - *Actual Raw Text*: 100% 《中华人民共和国红十字会法》（1993年通过、2009/2017年修正案全文，带知否卫监微信公众号排版头部）。
   - *Substantive Occurrences*: "医疗器械" appears **0 times**; "红十字" appears **53 times**.
2. **`doc086`**:
   - *Manifest / SQLite / Chunk Title*: `关于采供血机构管理有关事宜的批复（国卫医函[2018]35号）`
   - *Actual Raw Text*: 100% 《卫生部关于X射线诊断机等医用诊断设备不属于计量器具的批复》（卫法监发[2002]119号）。
   - *Substantive Occurrences*: "采供血" / "血液" appears **0 times**; "X射线" / "计量器具" appears throughout.

Full 100-row audit record generated at: [reports/integrity/corpus_document_audit.csv](file:///Users/gravity/Desktop/AI/Knowledge-Routing-RAG/reports/integrity/corpus_document_audit.csv).

---

# 2. Chunk Provenance Audit (2,862 Chunks)

Deterministic string and whitespace-normalized matching was executed for all 2,862 chunks against the normalized raw files in `data/documents/`:

```text
============================================================
Total Chunks in SQLite:             2,862
Verbatim Exact Substring Matches:   2,719 (95.00%)
Whitespace-Normalized Matches:        143  (5.00%)
------------------------------------------------------------
Total Substring Provenance Matches: 2,862 (100.00%)
Failed / Hallucinated Chunks:           0  (0.00%)
Orphan Chunks (No Doc ID):              0  (0.00%)
------------------------------------------------------------
Chunks with Corrupted Metadata:        46  (1.61%)
  - doc034 chunks: 45
  - doc086 chunks: 1
============================================================
```

Full 2,862-row audit record generated at: [reports/integrity/chunk_provenance_audit.csv](file:///Users/gravity/Desktop/AI/Knowledge-Routing-RAG/reports/integrity/chunk_provenance_audit.csv).

---

# 3. Gold Benchmark Audit Across 5 Benchmark Suites

All 750 questions across the 5 frozen benchmark sets were audited for span integrity and contaminated document references:

| Benchmark Suite | Total Questions | Gold Spans Audited | Spans Verified Verbatim | Contaminated Questions | Contamination Rate | Contaminated QIDs |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Dev** | 120 | 120 | 120 (100%) | 2 | 1.67% | `Q074`, `Q119` |
| **Holdout-1** | 200 | 448 | 448 (100%) | 3 | 1.50% | `CONF_Q097`, `CONF_Q136`, `CONF_Q152` |
| **Holdout-2** | 250 | 462 | 462 (100%) | 2 | 0.80% | `H2_020`, `H2_093` |
| **ScaleSet-1** | 80 | 160 | 160 (100%) | 0 | 0.00% | *None* |
| **HubSet-1** | 100 | 235 | 235 (100%) | 3 | 3.00% | `HUB_038`, `HUB_047`, `HUB_059` |
| **Total / Overall** | **750** | **1,425** | **1,425 (100%)** | **10** | **1.33%** | **10 items total** |

### 3.1 Impact of Decontamination on Holdout-2

In Holdout-2:
- `H2_020`: Prompt asks about 《医疗器械监督管理条例》 regarding 红十字. Both $H_0$ and $H_1$ answered **True** ($T\to T$).
- `H2_093`: Prompt asks about 《医疗器械监督管理条例》 vs 《红十字会法》 effective dates. Both $H_0$ and $H_1$ answered **False** ($F\to F$).

```text
--------------------------------------------------------------------------------
Dataset                  N    H0 (B0) Acc    H1 (V3) Acc    Accuracy Delta    Net Rescues
--------------------------------------------------------------------------------
Full Holdout-2         250       67.60%         73.20%         +5.60pp           +14
Decontaminated Clean   248       67.74%         73.39%         +5.65pp           +14
--------------------------------------------------------------------------------
```

**Conclusion**: Decontamination changes accuracy delta from $+5.60\text{pp}$ to $+5.65\text{pp}$. The main confirmation conclusion of the paper/report is **100% statistically preserved**.

Full 750-row audit record generated at: [reports/integrity/gold_integrity_audit.csv](file:///Users/gravity/Desktop/AI/Knowledge-Routing-RAG/reports/integrity/gold_integrity_audit.csv).

---

# 4. Benchmark Explicitness Audit (Holdout-2 Multi-hop N=156)

All 156 multi-hop questions in Holdout-2 were categorized into 4 tiers based on statutory naming explicitness:
- **E0 — Implicit**: Cross-domain concept without naming target statutes.
- **E1 — Partial**: Names 1 statute but leaves subsequent hops unnamed.
- **E2 — Explicit Docs**: Explicitly names all target statutes in question text.
- **E3 — Explicit Docs & Articles**: Explicitly names all target statutes AND specific article numbers (e.g. 《条例》第四条 与 《法》第四条).

### 4.1 Quantitative Tier Breakdown

| Explicitness Tier | Questions ($N$) | Proportion | $H_0$ (B0) Acc | $H_1$ (V3) Acc | Accuracy Delta | Rescues | Regressions | Net Rescues |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E0 — Implicit** | 4 | 2.56% | 100.00% | 100.00% | $+0.00\text{pp}$ | 0 | 0 | 0 |
| **E1 — Partial** | 4 | 2.56% | 25.00% | 25.00% | $+0.00\text{pp}$ | 0 | 0 | 0 |
| **E2 — Explicit Docs** | 77 | 49.36% | 37.66% | 46.75% | **$+9.09\text{pp}$** | 11 | 4 | +7 |
| **E3 — Explicit Docs & Articles** | 71 | 45.51% | 59.15% | 69.01% | **$+9.86\text{pp}$** | 7 | 0 | +7 |
| **Total Multi-Hop** | **156** | **100.0%** | **48.72%** | **57.69%** | **$+8.97\text{pp}$** | **18** | **4** | **+14** |

### 4.2 Key Architectural Insight
1. **Explicitness Dominance**: **94.87%** of multi-hop questions belong to Tiers E2 and E3 ($77 + 71 = 148 / 156$).
2. **Origin of Gain**: **100.0% of the net multi-hop gain** (+14 net rescues) occurs in Tiers E2 and E3.
3. **Functional Reality of Knowledge Routing**: The system's empirical gain does not stem from discovering deep, unstated semantic links across unindexed text. Rather, it functions as a high-precision **statutory entity recognizer and targeted document descent mechanism**: when a user prompt mentions two statutes, Knowledge Routing ensures chunks from both designated statutes enter the prompt context, overcoming Vector Search's tendency to retrieve all chunks from only one statute.

Full 156-row audit record generated at: [reports/integrity/holdout2_explicitness_audit.csv](file:///Users/gravity/Desktop/AI/Knowledge-Routing-RAG/reports/integrity/holdout2_explicitness_audit.csv).

---

# 5. Evaluation Runner and Metric Invariants Audit

Detailed inspection of [scripts/run_holdout2_experiment.py](file:///Users/gravity/Desktop/AI/Knowledge-Routing-RAG/scripts/run_holdout2_experiment.py) established two critical reporting corrections:

### 5.1 Same-Evidence Flips
- **Code Reality**: When `h1_cids == h0_cids`, the runner executes `h1_ans = h0_ans` and `h1_judge = h0_judge`.
- **Finding**: `SAME_EVIDENCE_FLIPS = 0 / 250 (0.00%)` is **an architectural invariant enforced by runner caching**, not an empirical finding of LLM token stability.
- **Correction**: This behavior is valid experimental control (preventing generation noise when retrieval is unchanged), but must be documented as caching rather than an empirical measurement.

### 5.2 Useful Evidence Eviction
- **Code Reality**: Eviction tracking recorded 10 useful evictions in Holdout-2.
- **Finding**: The earlier claim of "0% eviction rate across all benchmarks" is refuted. On Holdout-2, Composer had a **4.00% eviction rate (10 / 250)**.
- **Correction**: The claim must be demoted to: *"Composer maintains a low eviction rate of 4.0% on Holdout-2, achieving a net retrieval rescue of +8 (+11 rescues vs. 3 regressions)."*

Full runner audit report: [reports/integrity/runner_freshness_audit.md](file:///Users/gravity/Desktop/AI/Knowledge-Routing-RAG/reports/integrity/runner_freshness_audit.md).

---

# 6. Actionable Protocol & Final Gate Recommendation

1. **Gate Acceptance**: Acceptance under status **`I-B — CORRECTABLE INTEGRITY ISSUES`**.
2. **Release Documentation Updates**:
   - Keep Holdout-2 primary metrics reported on the full $N=250$ suite (+5.60pp), with an explicit footnote documenting the decontaminated $N=248$ result (+5.65pp).
   - Clarify the nature of `same_evidence_flips = 0` as deterministic short-circuit caching.
   - Demote universal "0% eviction" claims to "4.0% eviction on Holdout-2".
   - Disclose that 94.9% of multi-hop questions are explicit (E2/E3), defining the operational scope of Knowledge Routing as statutory entity-guided descent.
