# Independent Holdout Confirmation: Code Freeze Protocol

**Status**: **`CODE FROZEN`**  
**Frozen Git Commit**: `4e5f406`  
**Timestamp**: `2026-09-22T09:42:00+08:00`  
**Protocol Requirement**: Established prior to creation, inspection, or execution of the independent confirmation holdout set. Once frozen, systems H0, H1, and H2 must not be modified.

---

## 1. Frozen Experimental Systems

### H0 — B0 Fresh (Baseline)
- **Architecture**: Pure Vector RAG (Vector Top-5 retrieved over D100 corpus).
- **Prompt**: Baseline `SYSTEM_PROMPT` + `format_user_prompt` from `src/common/prompt.py`.
- **Execution Mode**: 100% Fresh generation and fresh judge across all runs (Zero answer pass-through, zero cache reuse).

### H1 — Clean-Raw (Primary Confirmation Hypothesis)
- **Architecture**: `C7CleanRouterSystem` from `src/routing/c7_clean_router.py`.
  - Control Plane: Vector Top-20 Shadow Candidate Plane (RIB) + Document Prefix Aggregation.
  - Route-Prefix Resolution: Specificity & relation-based selection of external prefixes.
  - Next-Hop Resolution: Recursive Parent Lift (`PARENT_LIFT` $\to$ `DOC_RELATION_RESOLVE`).
  - Targeted Descent: Algorithmic synthesis without lookup tables (`keywords(unresolved_slot) + keywords(target_doc_title)`).
  - Data Plane: Conservative Evidence Admission Gate (Top 1–3 locked, replaceable slots 4–5, budget cap = 5 chunks).
  - Decontamination: 100% free of hardcoded doc IDs, law-specific query tables, or benchmark question regexes.
- **Prompt**: Raw B0 `SYSTEM_PROMPT` + `format_user_prompt` (Identical prompt to H0).
- **Execution Mode**: 100% Fresh generation and fresh judge across all runs.

### H2 — Clean-Generic (Secondary Hypothesis)
- **Architecture**: H1 Clean Retrieval & Routing Architecture (`C7CleanRouterSystem`).
- **Prompt**: `GENERIC_CONTRACT_SYSTEM_PROMPT` + `build_generic_contract_prompt` from `src/generation/generic_contract.py`.
  - Strictly generic 5-rule legal answer contract.
  - Zero specific statute names, article numbers, years, or benchmark cases.
- **Execution Mode**: 100% Fresh generation and fresh judge across all runs.

---

## 2. Cryptographic Implementation Hashes (SHA-256)

| Component | File Path | SHA-256 Checksum |
|:---|:---|:---|
| **B0 Retrieval Engine** | `src/retrieval/vector_rag.py` | `9da475a64725684a9e63e0b799bd15b913747fc7935c44bc0262891d29fef6fd` |
| **Clean Router (H1)** | `src/routing/c7_clean_router.py` | `4892ce9e1251d5e9641ead47ad771179ef2268a151f5a559411223f3f80c0c0a` |
| **Generic Contract (H2)** | `src/generation/generic_contract.py` | `8bf1a2281724fca350f9f1f748b09e176710d764355d9e30e238a6b9deadb630` |
| **B0 Prompt Module** | `src/common/prompt.py` | `cc9c8ad349c642275b233db5f9e823751735377c80e30f62cd3bf7c16d023961` |
| **LLM Service Client** | `src/services/llm.py` | `f01554a078e8abe23d609fa909d6819e2b9ebe44dde0c72433e704c9342a668a` |
| **Embedding Client** | `src/services/embedding.py` | `f083d6768f12a7134d4bc808b25fe4a618b14b283ef482cec5b5092dcd619402` |
| **Search Service** | `src/services/search.py` | `aaf449a539f68d68d328a5b6e652d8a80235c26c697a11ca88c6c06c5cd83a8e` |
| **Knowledge LSDB Module** | `src/graph/lsdb.py` | `82e64586bc3c3d6d4a3b1a0c822e5f2ba6ecba123b356a65e005fbeafa0e266c` |
| **Evaluation Engine** | `src/evaluation/metrics.py` | `6d455ffcf3ff5dcc3bbbe173bd0c18cb2385da05bd8670aeb492d037c763ebde` |

---

## 3. Data & Graph Artifact Hashes (SHA-256)

| Artifact | File Path | SHA-256 Checksum |
|:---|:---|:---|
| **LSDB SQLite Database** | `data/knowledge_lsdb.sqlite` | `5495e97f18f2e414622cca162f8f6051cf88b77e8d3aefd565dcb7950ec9bfa8` |
| **Corpus Chunks** | `data/chunks.jsonl` | `7763192631df6bd55a0db1188ae8fcf3e7ff212304a0a5e75605d4539efeca7a` |
| **D100 Manifest** | `data/manifests/d100.json` | `852865dc239b65c10fc8e99b0469f43886b624f315ca6060eeb26a4666f0fd22` |
| **Relation Statistics** | `data/relation_statistics.json` | `468d0bbc6812225d63095dcf1b0bf92a767f79e4dff98fe4d6ba360998203196` |

---

## 4. Frozen Runtime Parameters

- **Generator Model**: `deepseek-chat`
- **Judge Model**: `deepseek-chat`
- **Temperature**: `0.0` (Strict deterministic setting)
- **Pre-registered Seeds**:
  - **Run 1**: `seed = 101`
  - **Run 2**: `seed = 202`
  - **Run 3**: `seed = 303`
- **Retrieval Hyperparameters**:
  - Data Plane Top-K (`top_k`): `5`
  - Control Plane Shadow Top-K (`shadow_top_k`): `20`
  - Max Routing Hops (`max_hops`): `2`
  - Max Admission Replacements (`max_replacements`): `2`
  - Final Evidence Token Limit (`max_evidence_tokens`): `4000` (~5 chunks)
  - Primary Evaluation Corpus: **`D100`**

---

## 5. Non-Intervention Commitment

From this point forward:
1. No modifications may be made to `src/routing/c7_clean_router.py`, `src/generation/generic_contract.py`, or `src/retrieval/vector_rag.py`.
2. The independent confirmation holdout dataset will be constructed strictly from the frozen corpus without inspecting any system responses or past candidate error analyses.
3. If an implementation defect is discovered during holdout evaluation, the confirmation must be terminated, the bug corrected, a new freeze issued, and a completely fresh holdout dataset constructed.
