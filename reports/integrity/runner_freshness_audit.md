# Runner Freshness and Evaluation Logic Integrity Audit

**Audit Date**: 2026-09-22  
**Target Repository**: `Knowledge-Routing-RAG`  
**Auditor**: Independent Data & Evaluation Integrity Auditor (Gate I)  
**Target Scripts**: `scripts/run_holdout2_experiment.py`, `scripts/run_confirmation_evaluation.py`, `scripts/run_v3_e2_evaluation.py`  
**Status**: COMPLETE

---

## 1. Executive Summary

This audit evaluates the algorithmic neutrality, caching behaviors, short-circuit implementations, and metric definitions of the evaluation runners in `Knowledge-Routing-RAG`.

Key findings:
1. **Identical Evidence Short-Circuit**: In `scripts/run_holdout2_experiment.py`, when $H_1$ (V3 retrieval) produces the exact same chunk IDs as $H_0$ (Vector baseline), the runner copies $H_0$'s generated answer and LLM judge verdict without calling the generator or evaluator (`h1_ans = h0_ans`, `h1_judge = h0_judge`).
2. **`SAME_EVIDENCE_FLIPS = 0` Metric Status**: The reported metric `Same-Evidence Flips: 0 / 250 (0.00%)` is **not an empirical discovery of zero LLM stochasticity**, but an **enforced architectural invariant** of the runner pipeline.
3. **Effect Attribution Invariant**: The finding `0 Generation-only Flips` in effect attribution is mathematically guaranteed by the same short-circuit, because transitions require $H_1 \neq H_0$, which cannot occur when $H_1$ chunk IDs match $H_0$.
4. **Eviction Metric Reality**: The claim that the Composer "universally eliminates useful evidence eviction (0% eviction)" is refuted by the Holdout-2 data. Holdout-2 experienced **10 useful evidence evictions (4.00%)**, leading to 3 retrieval regressions. The metric must be reported honestly as **low eviction rate (4.0%)**, not universal zero.

---

## 2. Code Inspection: The Identical Evidence Short-Circuit

### 2.1 Code Location
In [scripts/run_holdout2_experiment.py](file:///Users/gravity/Desktop/AI/Knowledge-Routing-RAG/scripts/run_holdout2_experiment.py#L454-L466):

```python
# scripts/run_holdout2_experiment.py:454-466
# H1 Run
h1_cids = h1_traces[qid]["final_evidence_chunk_ids"]
if h1_cids == h0_cids:
    h1_ans = h0_ans
    h1_judge = h0_judge
else:
    h1_evidence = [EvidenceItem(**lsdb.get_chunk_evidence(c)) for c in h1_cids if lsdb.get_chunk_evidence(c)]
    h1_context = pack_evidence_context(h1_evidence, max_tokens=4000)
    h1_prompt = format_user_prompt(question, h1_context)
    h1_ans, _, _ = llm_service.generate(prompt=h1_prompt, system_prompt=SYSTEM_PROMPT, seed=seed)
    h1_judge = evaluator.judge_answer(question=question, gold_answer=gold_answer, gold_spans=gold_spans, generated_answer=h1_ans)
```

And in [scripts/run_holdout2_experiment.py](file:///Users/gravity/Desktop/AI/Knowledge-Routing-RAG/scripts/run_holdout2_experiment.py#L550-L554):

```python
# scripts/run_holdout2_experiment.py:550-554
# Check Same-Evidence Flips
if h0_traces[qid]["final_evidence_chunk_ids"] == h1_traces[qid]["final_evidence_chunk_ids"]:
    if h0_maj != h1_maj:
        same_evidence_flips.append(qid)
```

### 2.2 Mechanism and Consequence
When `h1_cids == h0_cids`:
- In Run 1 (Seed 101): `h1_judge == h0_judge`
- In Run 2 (Seed 202): `h1_judge == h0_judge`
- In Run 3 (Seed 303): `h1_judge == h0_judge`

Consequently, for every question where retrieval yields identical chunks:
$$\text{Votes}(H_1) = \text{Votes}(H_0) \implies H_1^{\text{majority}} \equiv H_0^{\text{majority}}$$

Therefore:
$$\text{same\_evidence\_flips} = \sum_{q \in Q} \mathbb{I}(c_1(q) = c_0(q) \land H_1(q) \neq H_0(q)) \equiv 0$$

### 2.3 Scientific Assessment
- **Engineering Justification**: When context and prompt are identical, avoiding redundant LLM generation calls saves ~50% of API calls and token spend on fast-path / identical retrieval cases. Furthermore, in paired RAG benchmarking, isolating retrieval causal effects from LLM token sampling variance is standard experimental control.
- **Reporting Misrepresentation**: Describing `Same-Evidence Flips: 0 (0.00%)` as an empirical measurement of generator reproducibility is circular. The runner code literally guarantees it cannot be any other number.
- **Audit Requirement**: Any public release or report must state that same-evidence flip rate is 0 **by construction via deterministic caching**, not by empirical sampling across independent API generations.

---

## 3. Eviction Rate: Dev vs. Holdout-2

### 3.1 Definition of Useful Evidence Eviction
A useful evidence eviction is defined in [scripts/run_holdout2_experiment.py](file:///Users/gravity/Desktop/AI/Knowledge-Routing-RAG/scripts/run_holdout2_experiment.py#L337-L342):

```python
evicted = set(h0_cids) - set(h1_cids)
for ec in evicted:
    if ec in gold_chunks and not any(r in gold_chunks for r in (set(h1_cids) - set(h0_cids))):
        useful_evictions += 1
```
A chunk $ec \in H_0$ is a "useful eviction" if $ec$ was in the gold evidence set, but was dropped by Composer during slot substitution without bringing in an alternative gold chunk.

### 3.2 Quantitative Results

| Benchmark | Total Questions | Useful Evictions | Eviction Rate | Retrieval Rescues | Retrieval Regressions | Net Retrieval Rescue |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dev E1** | 120 | 0 | 0.00% | 14 | 0 | +14 |
| **Holdout-1** | 200 | 2 | 1.00% | 19 | 2 | +17 |
| **Holdout-2** | 250 | 10 | 4.00% | 11 | 3 | +8 |

### 3.3 Integrity Rectification
The claim:
> *"V3 Slot Composer completely eliminates useful evidence eviction (0% eviction rate across all benchmarks)."*

Is **factually invalid** on Holdout-2. On Holdout-2, Composer experienced **10 useful evictions (4.0%)**, leading directly to 3 retrieval regressions where $H_0$ had complete gold evidence while $H_1$ lost it.

The verified, scientifically honest statement is:
> *"V3 Slot Composer reduces useful evidence eviction to a low rate of 4.0% on unseen Holdout-2 data (compared to 12–18% under unconstrained score-based replacement), resulting in a net retrieval rescue of +8 (+11 rescues vs. 3 regressions)."*

---

## 4. Run Independence, Seeds, and Fallback Handlers

### 4.1 Seed Control
The runner employs 3 distinct seeds:
- Run 1: `seed = 101`
- Run 2: `seed = 202`
- Run 3: `seed = 303`

These seeds are passed directly to `llm_service.generate(prompt=..., seed=seed)` and `evaluator.judge_answer(...)`.

### 4.2 Fallback Handler Audit
Lines 483-486 of `scripts/run_holdout2_experiment.py`:
```python
fallback_count = sum(1 for it in h0_results.values() if it["reasoning"] == "Heuristic fallback evaluation.") + \
                 sum(1 for it in h1_results.values() if it["reasoning"] == "Heuristic fallback evaluation.")
if fallback_count > 0:
    raise RuntimeError(f"Run {run_id} aborted: {fallback_count} judge evaluations used heuristic fallback...")
```
This check ensures that if LLM judge calls timed out or encountered API errors, the runner aborts rather than silently substituting heuristic fallback scores into the final dataset. In the frozen runs, `fallback_count == 0` for all 3 runs.

---

## 5. Audit Recommendations for Runner Implementations

1. **Explicit Caching Documentation**: Add an explicit docstring in `run_holdout2_experiment.py` stating that `h1_ans = h0_ans` when `h1_cids == h0_cids` is an intentional experimental control for isolating retrieval variance.
2. **Metric Label Demotion**: Rename `Same-Evidence Flips` to `Same-Evidence Flips (Cached Invariant = 0)`.
3. **Eviction Claim Demotion**: Demote all documentation claiming "0% eviction" to "4.0% eviction on Holdout-2".
