# Independent Holdout-2 Pre-Registration Document

**Study Title**: Independent Confirmation Experiment of V3 Knowledge Routing vs Pure Vector RAG  
**Date of Pre-Registration**: 2026-09-22  
**Target Environment**: D100 Distraction Benchmark  
**Sample Size**: $N = 250$ Unique, Unseen Questions  
**Primary Endpoint**: 3-Run Majority Answer Correctness (H0 Majority vs H1 Majority)  
**Status**: PRE-REGISTERED & SYSTEM FROZEN (Prior to Holdout-2 Data Generation)  

---

## 1. Frozen System Architectures

### System H0: Vector Baseline (B0)
- **Retrieval**: Pure Dense Vector Search (Qdrant `law_chunks`, Top-5 chunks).
- **Prompt**: Original B0 Raw Answer Prompt (`src/common/prompt.py`).
- **Context Budget**: Max 4000 tokens (up to 5 chunks).
- **Generation**: LLMService greedy decoding (`temperature=0.0`).

### System H1: Frozen V3 Knowledge Routing
- **FIB Entrance**: Vector Search Top-5 (`top_k=5`).
- **Control Plane**: Shadow Candidate Plane Top-20 RIB (`shadow_top_k=20`).
- **Prefix Discovery**: Multi-signal scoring over document prefixes.
- **Topology**: Hierarchical parent lift (`BASED_ON`, `REFERENCES`) and graph gap filling.
- **Targeted Descent**: `E2-Lite` (Slot-Conditioned Generic Lexical Targeted Descent, `channel_mode="lexical_only"`, FTS BM25 + Heading Bonus).
- **Admission Gate**: `E1 Composer` (Coverage-Preserving Evidence Composition, `max_chunks=5`, `max_replacements=2`, `coverage_threshold=0.40`, `useful_evidence_eviction=0`).
- **Prompt**: Original B0 Raw Answer Prompt (`src/common/prompt.py`).
- **Context Budget**: Max 4000 tokens (up to 5 chunks).
- **Generation**: LLMService greedy decoding (`temperature=0.0`).

---

## 2. Cryptographic Code Signatures

```yaml
git_commit: "4e5f40679086a5f403e0af42d120dcea79effc89"
components:
  router:
    path: "src/routing/e2_descent_router.py"
    sha256: "946e59c96acc9c572f6edf95a99851b6d65e66c3b2e8f51b91270bf240e7fec5"
  descent_module:
    path: "src/composition/descent.py"
    sha256: "001497c30cc07d51995771834d2d3e0c98738ddf4517aadfafab6943d8b7fa0c"
  composer:
    path: "src/composition/composer.py"
    sha256: "224ef0e88364a969a7c919a4195d4a26ab7f86373a33d1c0625c7f12fa09291d"
  slots:
    path: "src/composition/slots.py"
    sha256: "e6d5b960c924e7d350d9cc8bb9eba8aaf070e3b4419f4ec0dbd8648efde9953f"
  coverage:
    path: "src/composition/coverage.py"
    sha256: "e75d4af5cae647c61fcfebea2b2ad3d657dca634c9f1273e3e47944fbdd3b7c5"
  prompt:
    path: "src/common/prompt.py"
    sha256: "cc9c8ad349c642275b233db5f9e823751735377c80e30f62cd3bf7c16d023961"
  search_service:
    path: "src/services/search.py"
    sha256: "aaf449a539f68d68d328a5b6e652d8a80235c26c697a11ca88c6c06c5cd83a8e"
  evaluator_metrics:
    path: "src/evaluation/metrics.py"
    sha256: "6d455ffcf3ff5dcc3bbbe173bd0c18cb2385da05bd8670aeb492d037c763ebde"
```

---

## 3. Pre-Registered Experimental Protocol

1. **Benchmark Scale & Environment**:
   - $N = 250$ unique questions constructed strictly from the D100 corpus.
   - Primary evaluation environment: **D100** (high-distraction full corpus setting).
   - Each unique qid constitutes a single primary statistical unit.

2. **Fresh Execution Protocol**:
   - 3 independent, paired Fresh Runs for H0 and H1 using random seeds: `101`, `202`, `303`.
   - Zero historical caching or response reuse; every run executes fresh generation and judging.
   - Paired execution within identical API timeframes.

3. **Primary Endpoint**:
   - **3-Run Majority Correctness**: For each question, correctness is determined by majority vote across the 3 runs (at least 2 out of 3 runs correct).
   - **Primary Metric**: $\Delta = \text{Accuracy}(\text{H1 Majority}) - \text{Accuracy}(\text{H0 Majority})$.

4. **Paired Transition Metrics**:
   - **Stable Rescue**: H0 Majority = Incorrect, H1 Majority = Correct.
   - **Stable Regression**: H0 Majority = Correct, H1 Majority = Incorrect.
   - **Net Stable Rescue**: $\text{Stable Rescue} - \text{Stable Regression}$.

5. **Statistical Significance Tests**:
   - Primary Paired Test: **Exact McNemar Test** (two-tailed, $\alpha = 0.05$).
   - Confidence Interval: **Paired Bootstrap 95% CI** (10,000 resamples) for $\Delta$.

6. **Safety & Subgroup Endpoints**:
   - Simple (1-hop) Accuracy & Simple Regression count.
   - Multi-Hop (2-hop+) Accuracy & Multi-Hop Rescue count.
   - Retrieval Layer Mechanism Metrics:
     - Gold Document Recall
     - Gold Chunk Recall
     - Evidence F1
     - Chain Completion Rate
     - Net Retrieval Rescue

7. **Blind Human Adjudication Protocol**:
   - All discordant cases (Stable Rescues, Stable Regressions, and unstable critical flips) are blinded (system labels anonymized as System A and System B).
   - Independent verification of answer correctness and judge alignment.
   - Final report incorporates **Adjudicated Majority Result**.

---

## 4. Pre-Registered Decision Rules (Verdicts)

### VERDICT A — END-TO-END CONFIRMED
Requires **all 6 criteria** to be simultaneously met:
1. **Majority Accuracy**: H1 Majority Accuracy > H0 Majority Accuracy.
2. **Directional Stability**: At least 2 of the 3 individual runs show H1 > H0, with no large negative run offsetting the gain.
3. **Stable Transition**: $\text{Stable Rescue} > \text{Stable Regression}$.
4. **Retrieval Mechanism**: Positive gain on at least two core retrieval metrics (Gold Document Recall, Gold Chunk Recall, Chain Completion), with $\text{Net Retrieval Rescue} > 0$.
5. **Blind Review**: $\text{Adjudicated Stable Rescue} > \text{Adjudicated Stable Regression}$.
6. **Statistical Evidence**: Exact McNemar $p < 0.05$ OR Paired Bootstrap 95% CI strictly bounded above 0.

### VERDICT B — MECHANISM CONFIRMED, END-TO-END NOT CONFIRMED
Applied when:
- Retrieval/Evidence metrics show clear, robust improvement ($\text{Gold Document Recall} \uparrow$, $\text{Gold Chunk Recall} \uparrow$, $\text{Net Retrieval Rescue} > 0$);
- BUT End-to-End Majority Accuracy fails to achieve statistical significance ($p \ge 0.05$, CI crosses zero) or Stable Rescue $\approx$ Stable Regression.

### VERDICT C — NOT CONFIRMED
Applied when:
- Retrieval metrics fail to show stable improvement; OR
- H1 Majority Accuracy $\le$ H0 Majority Accuracy; OR
- $\text{Stable Regression} \ge \text{Stable Rescue}$; AND blind review does not support a net positive effect.

*No fourth/ambiguous verdict (e.g. "Promising", "Mixed", "Positive Trend") is permitted.*
