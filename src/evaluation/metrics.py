# src/evaluation/metrics.py
"""
Evaluation and Statistical Analysis Engine.
Reference: 实验方案.md Section 21, 22, 23, 24, 25, 26

Computes:
1. Retrieval Metrics: Evidence Precision, Recall, F1
2. Context Pollution Rate (CPR)
3. Answer Accuracy (LLM Judge + Rule Verification)
4. Rescue Rate & Regression Rate
5. McNemar Exact Test (Paired Comparison)
6. 95% Bootstrap Confidence Intervals
7. Latency (P50, P95) and Token Accounting
"""

import json
import numpy as np
from scipy import stats
from typing import List, Dict, Any, Tuple, Optional
from src.common.models import ExecutionTrace
from src.services.llm import LLMService

JUDGE_SYSTEM_PROMPT = """你是一位客观严谨的法学法律判卷专家。
你需要评判模型生成的解答与标准参考答案（及法定证据要点）在法理实质上是否一致。
评分标准：
- 如果模型回答的核心法律结论、关键法条内容和限制条件与标准答案相符（允许表述措辞差异，但核心法律定性、倍数、时限或程序必须准确），判定为正确（is_correct: true）。
- 如果模型回答出现明显事实错误、遗漏关键核心限制/例外、或得出与法律规定相反的结论，判定为错误（is_correct: false）。
输出格式必须为 JSON:
{
  "is_correct": true/false,
  "confidence": 0.0~1.0,
  "reasoning": "一句话判卷理由"
}"""

class Evaluator:
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()

    def evaluate_retrieval(
        self,
        retrieved_ids: List[str],
        gold_ids: List[str]
    ) -> Dict[str, float]:
        """
        Computes Precision, Recall, F1, CPR.
        """
        if not retrieved_ids:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "cpr": 0.0}

        ret_set = set(retrieved_ids)
        gold_set = set(gold_ids)

        hits = len(ret_set & gold_set)
        precision = hits / len(ret_set) if ret_set else 0.0
        recall = hits / len(gold_set) if gold_set else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        cpr = (len(ret_set) - hits) / len(ret_set) if ret_set else 0.0

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "cpr": float(cpr)
        }

    def judge_answer(
        self,
        question: str,
        gold_answer: str,
        gold_spans: List[str],
        generated_answer: str
    ) -> Dict[str, Any]:
        """
        Judges generated answer using LLM Judge with deterministic fallback.
        """
        spans_str = "\n".join([f"- {s}" for s in gold_spans])
        judge_prompt = f"""【问题】
{question}

【法定依据要点】
{spans_str}

【标准答案】
{gold_answer}

【待评测模型回答】
{generated_answer}

请根据评分标准客观裁定该回答是否在实质法理与核心事实上正确："""

        try:
            content, usage, lat = self.llm_service.generate(
                prompt=judge_prompt,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                response_format_json=True,
                max_tokens=1024
            )
            data = json.loads(content)
            return {
                "is_correct": bool(data.get("is_correct", False)),
                "confidence": float(data.get("confidence", 1.0)),
                "reasoning": data.get("reasoning", "")
            }
        except Exception:
            # Fallback heuristic: check overlap of core statutory keywords
            key_words = [w for w in gold_answer.replace("，", " ").replace("。", " ").split() if len(w) >= 3]
            overlap = sum(1 for kw in key_words if kw in generated_answer)
            is_corr = (overlap / max(1, len(key_words))) >= 0.5
            return {
                "is_correct": is_corr,
                "confidence": 0.6,
                "reasoning": "Heuristic fallback evaluation."
            }

    @staticmethod
    def bootstrap_ci(
        data: List[float],
        num_resamples: int = 2000,
        alpha: float = 0.05
    ) -> Tuple[float, float, float]:
        """
        Returns (mean, ci_lower, ci_upper)
        """
        arr = np.array(data)
        if len(arr) == 0:
            return 0.0, 0.0, 0.0
        means = []
        for _ in range(num_resamples):
            sample = np.random.choice(arr, size=len(arr), replace=True)
            means.append(np.mean(sample))
        mean = float(np.mean(arr))
        lower = float(np.percentile(means, 100 * (alpha / 2)))
        upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
        return mean, lower, upper

    @staticmethod
    def mcnemar_exact_test(
        baseline_correct: List[bool],
        router_correct: List[bool]
    ) -> Dict[str, Any]:
        """
        McNemar's exact paired test for 2x2 contingency table.
        Table:
                 Router +   Router -
        Base +     a           b (regression)
        Base -     c (rescue)  d
        """
        assert len(baseline_correct) == len(router_correct), "Length mismatch in McNemar test!"
        a = sum(1 for b, r in zip(baseline_correct, router_correct) if b and r)
        b = sum(1 for b, r in zip(baseline_correct, router_correct) if b and not r)  # Regressions
        c = sum(1 for b, r in zip(baseline_correct, router_correct) if not b and r)  # Rescues
        d = sum(1 for b, r in zip(baseline_correct, router_correct) if not b and not r)

        # Exact binomial test on discordant pairs (b, c)
        n_discordant = b + c
        if n_discordant > 0:
            result = stats.binomtest(k=min(b, c), n=n_discordant, p=0.5, alternative="two-sided")
            p_value = float(result.pvalue)
        else:
            p_value = 1.0

        rescue_rate = c / (c + d) if (c + d) > 0 else 0.0
        regression_rate = b / (a + b) if (a + b) > 0 else 0.0

        return {
            "contingency_matrix": {
                "base_correct_router_correct": a,
                "base_correct_router_wrong (regressions)": b,
                "base_wrong_router_correct (rescues)": c,
                "base_wrong_router_wrong": d
            },
            "rescues": c,
            "regressions": b,
            "rescue_rate": float(rescue_rate),
            "regression_rate": float(regression_rate),
            "p_value": float(p_value),
            "significant_at_05": bool(p_value < 0.05)
        }
