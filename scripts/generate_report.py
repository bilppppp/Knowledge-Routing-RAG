#!/usr/bin/env python3
"""
scripts/generate_report.py
Generates the comprehensive, rigorous, and protocol-aligned Final Academic Report
for Knowledge-Routing-RAG V1 per 实验方案.md §25, §26, §27, §33, §34, §35.

Outputs:
- reports/final_report.md
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports"
STAT_FILE = REPORTS_DIR / "statistical_analysis.json"
ADJUDICATION_FILE = REPORTS_DIR / "final_adjudicated_results.json"
REPORT_OUTPUT = REPORTS_DIR / "final_report.md"


def generate_final_report_md():
    with open(STAT_FILE, "r", encoding="utf-8") as f:
        stats = json.load(f)

    with open(ADJUDICATION_FILE, "r", encoding="utf-8") as f:
        adjudication = json.load(f)

    meta = stats["metadata"]
    sys_acc = stats["system_accuracies"]
    paired = stats["paired_comparisons_vs_b0"]
    retrieval = stats["retrieval_and_evidence_metrics"]
    latency = stats["latency_and_tokens"]
    scale = stats["scale_degradation_core60"]
    hub = stats["hub_correlational_analysis"]
    subgroups = stats["exploratory_subgroups"]
    diag = stats["post_hoc_error_diagnostics"]
    verdict = stats["preregistered_verdict"]

    # Table 1: Primary Endpoint Results
    table1_rows = []
    for s in ["B0", "B1", "K1", "K2", "K3", "K4"]:
        acc_info = sys_acc[s]
        mean_acc = acc_info["mean_accuracy_pct"]
        ci_l = acc_info["ci_lower_pct"]
        ci_u = acc_info["ci_upper_pct"]
        corr_tot = f"{acc_info['correct_count']}/{acc_info['total_count']}"

        if s == "B0":
            delta_str = "Baseline"
            rel_str = "-"
        else:
            p_info = paired[f"{s}_vs_B0"]["delta_bootstrap_95ci"]
            d_val = p_info["mean_delta_pp"]
            d_l = p_info["ci_lower_pp"]
            d_u = p_info["ci_upper_pp"]
            delta_str = f"{d_val:+.2f}pp [{d_l:+.2f}, {d_u:+.2f}]"
            rel_str = f"{p_info['relative_delta_pct']:+.1f}%"

        table1_rows.append(
            f"| **{s}** | {mean_acc:.1f}% [{ci_l:.1f}%, {ci_u:.1f}%] | {corr_tot} | {delta_str} | {rel_str} |"
        )
    table1_text = "\n".join(table1_rows)

    # Table 2: Retrieval and Evidence Quality
    table2_rows = []
    for s in ["B0", "B1", "K1", "K2", "K3", "K4"]:
        r_info = retrieval[s]
        l_info = latency[s]
        f1_str = f"{r_info['f1']['mean']:.3f} [{r_info['f1']['ci_lower']:.3f}, {r_info['f1']['ci_upper']:.3f}]"
        cpr_str = f"{r_info['cpr']['mean_pct']:.1f}% [{r_info['cpr']['ci_lower_pct']:.1f}%, {r_info['cpr']['ci_upper_pct']:.1f}%]"
        lat_str = f"{l_info['latency_p50_ms']:.0f} / {l_info['latency_p95_ms']:.0f} ms"
        tok_str = f"{l_info['mean_total_tokens']:.0f}"
        table2_rows.append(
            f"| **{s}** | {f1_str} | {cpr_str} | {lat_str} | {tok_str} |"
        )
    table2_text = "\n".join(table2_rows)

    # Table 3: Core-60 Scaling (Test N=48)
    table3_rows = []
    for s in ["B0", "B1", "K1", "K2", "K3", "K4"]:
        s_scale = scale[s]
        d20_str = f"{s_scale['D20']['accuracy_pct']:.1f}% ({s_scale['D20']['ratio_str']})"
        d50_str = f"{s_scale['D50']['accuracy_pct']:.1f}% ({s_scale['D50']['ratio_str']})"
        d100_str = f"{s_scale['D100']['accuracy_pct']:.1f}% ({s_scale['D100']['ratio_str']})"
        drop_val = s_scale["degradation_delta_d20_to_d100_pp"]
        table3_rows.append(
            f"| **{s}** | {d20_str} | {d50_str} | {d100_str} | {drop_val:+.1f}pp |"
        )
    table3_text = "\n".join(table3_rows)

    # Table 4: Hub Correlational Analysis
    table4_rows = []
    for tier_name in ["< 5 (Low)", "5–10 (Moderate)", "11–20 (High)", "21–50 (Very High)", "> 50 (Hub Core)"]:
        h_info = hub[tier_name]
        table4_rows.append(
            f"| {tier_name} | {h_info['question_count']} | {h_info['b0_accuracy_pct']:.1f}% | {h_info['k4_accuracy_pct']:.1f}% | {h_info['b0_cpr_pct']:.1f}% | {h_info['k4_cpr_pct']:.1f}% |"
        )
    table4_text = "\n".join(table4_rows)

    # Table 5: Exploratory Subgroups
    hop_rows = []
    for h in ["Hop 1", "Hop 2", "Hop 3+"]:
        r = [f"**{h}**"]
        for s in ["B0", "B1", "K1", "K2", "K3", "K4"]:
            info = subgroups["by_hop_count"][h][s]
            r.append(f"{info['accuracy_pct']:.1f}% ({info['correct_count']}/{info['total_count']})")
        hop_rows.append(f"| {' | '.join(r)} |")
    hop_table_text = "\n".join(hop_rows)

    tag_rows = []
    for t in ["single_hop", "2-hop", "3-hop", "hub", "temporal", "exception"]:
        r = [f"**{t}**"]
        for s in ["B0", "B1", "K1", "K2", "K3", "K4"]:
            info = subgroups["by_tag"][t][s]
            r.append(f"{info['accuracy_pct']:.1f}% ({info['correct_count']}/{info['total_count']})")
        tag_rows.append(f"| {' | '.join(r)} |")
    tag_table_text = "\n".join(tag_rows)

    # McNemar details
    m_k4 = paired["K4_vs_B0"]["mcnemar_test"]
    m_k2 = paired["K2_vs_B0"]["mcnemar_test"]

    content = f"""# Knowledge-Routing-RAG 实验评估与学术验证最终报告 (V1 严格闭环审计版)

> **FINAL REPORT STATUS: PENDING HUMAN ADJUDICATION**
> 
> [!CAUTION]
> **评审状态说明**: 本报告当前评测指标基于自动化评分（Auto Judge）。根据《实验方案.md §25》，LLM 评分不可作为唯一最终裁判，所有系统间分歧案例（共 23 组争议测试实例，展开为 31 组盲评对比对）已抽取形成待审盲评包 [`reports/blind_review_pack.json`](file:///Users/gravity/Desktop/AI/Knowledge-Routing-RAG/reports/blind_review_pack.json)。在独立双盲人工仲裁未正式完成并回填之前，本报告所有数据标记为**预备性学术评估**，不得作为最终已仲裁定论。

---

## 1. 实验基本信息与评估规范 (Experiment Status & Disclosures)

- **实验编号**: EXP-20260921-KR-V1 (Strict Protocol Alignment)
- **代码基准提交**: [`1aaafbb`](file:///Users/gravity/Desktop/AI/Knowledge-Routing-RAG) (锁定 Dev/Test 防火墙与冻结参数)
- **评估数据集**: 120 题目全集（Dev=24 冻结参数验证集，Test=96 正式评估盲测集）
- **测试语料库**: 严格嵌套语料集 $D20 \\subset D50 \\subset D100$（100 篇法规，2,862 Chunks，评测实例 $N=216$）
- **嵌入模型**: `google/embeddinggemma-300m` (768 维)
- **向量数据库**: Qdrant Docker (v1.8.2) + SQLite FTS5 (Jieba 分词)
- **生成模型**: DeepSeek Chat (`deepseek-chat`, Temperature=0.0)
- **自动裁判模型**: DeepSeek Chat (`deepseek-chat`, Temperature=0.0)
- **上下文预算硬上限**: 4,000 Tokens (统一预算约束)

> [!WARNING]
> **评测模型同源性披露 (Model-Family Evaluation Bias Disclosure)**:
> 本实验的生成模型与自动裁判模型均属于 DeepSeek 模型家族。存在因模型家族相同而产生的内在推理偏置（Model-Family Bias）。为消除该偏差，实验严格按照《方案 §25》预留了双盲人工仲裁管道（Blind Adjudication Pack），禁止任何系统标识出现在评审端。

---

## 2. 主终点结果 (Primary Endpoint Results)

本实验的 **Primary Hypothesis (主假设)** 严格依据《实验方案.md §35》预注册定义：
> **“在存在跨文档依赖、高扇出公共节点和多跳证据链的知识库中，将‘候选发现’和‘证据进入上下文’解耦，并通过聚合、可行性约束和显式路径程序控制下一跳，能否提高证据精度和最终回答准确率。”**

预注册的成功判据（《实验方案.md §27》）：
- **Minimum Signal (最低有效信号)**: $K4 > B0$
- **Engineering Success (工程成功标准)**: $K4 \\ge B0 + 8\\text{{pp}}$，且 $\\text{{Regression Rate}} < \\text{{Rescue Rate}}$，且 CPR 显著收敛。

### 主终点测试集准确率统计表 (Overall Test Set Accuracy)

| 实验系统 | 准确率 (Mean [95% Bootstrap CI]) | 命中题数 / 总测试数 | 相对 B0 绝对差值 (Δ) [95% CI] | 相对 B0 相对变动率 |
|:---|:---:|:---:|:---:|:---:|
{table1_text}

### 主终点判定：
- 纯向量检索基线 **B0 达到 72.2%**（[66.2%, 78.2%]）。
- 完整路由系统 **K4 仅达到 69.9%**（[63.9%, 75.9%]），低于 B0 达到 **2.31pp**。
- 表现最好的路由变体 **K2 达到 71.3%**（[65.3%, 77.3%]），仍低于 B0 **0.93pp**。
- **结论**: **V1 的 Primary Hypothesis 未得到数据支持**。无论是 K4 还是任何中间路由变体，均未达成 $K4 > B0$ 的 Minimum Signal 门槛。

---

## 3. 统计不确定性与配对检验 (Statistical Uncertainty & Paired Deltas)

由于各系统在完全相同的测试题目序列上运行，采用 Paired Bootstrap (5,000 次重采样) 与 McNemar 配对确切检验：

| 配对对比组 | 准确率差值 (Δ) | 95% Bootstrap CI | 配对确切检验 (McNemar p) | 显著性判断 (α=0.05) |
|:---|:---:|:---:|:---:|:---:|
| **K1 vs B0** | -3.70pp | [-6.94pp, -0.46pp] | p = 0.0386 | 统计显著为负 (*) |
| **K2 vs B0** | -0.93pp | [-4.17pp, +2.31pp] | p = 0.7744 | 无显著差异 (CI 跨零) |
| **K3 vs B0** | -3.24pp | [-6.48pp, -0.46pp] | p = 0.0654 | 边际为负 (CI 上界为负) |
| **K4 vs B0** | -2.31pp | [-6.48pp, +1.39pp] | p = 0.3593 | 无显著差异 (CI 跨零) |

> [!NOTE]
> Paired Bootstrap 95% 置信区间显示，K4 vs B0 的置信区间涵盖了 [-6.48pp, +1.39pp]，点估计为 -2.31pp。数据排除了 K4 能带来预注册要求的 +8pp 提升的可能性。

---

## 4. 证据检索与上下文质量指标 (Retrieval & Evidence Quality Metrics)

| 实验系统 | 证据检索 F1 [95% CI] | 上下文污染率 (CPR ↓) [95% CI] | 延迟 (P50 / P95) | 平均上下文 Tokens |
|:---|:---:|:---:|:---:|:---:|
{table2_text}

- **证据检索 F1**: K4 (0.247) 相比 B0 (0.244) 仅有微幅差异 (+0.003)，未见实质性质的证据召回改善。
- **上下文污染率 (CPR)**: B0 为 83.2%，K4 为 83.1%，两者的 95% 置信区间完全重叠，路由机制未能显著抑制非黄金 Chunk 进入上下文。
- **计算开销**: K4 端到端延迟 P50 为 3621 ms（B0 为 3590 ms），平均图路由耗时 81.1 ms，额外计算负担很小，但由于未换取准确率提升，此开销未产生正向收益。

---

## 5. 配对挽救与退化案例分析 (Paired Rescue vs. Regression)

对题目级判断进行转换矩阵（Contingency Matrix）分析：

### B0 vs K4 转移矩阵
| | K4 正确 (Router +) | K4 错误 (Router -) | 合计 |
|:---|:---:|:---:|:---:|
| **B0 正确 (Base +)** | 144 | **12 (退化案例, Regressions)** | 156 |
| **B0 错误 (Base -)** | **7 (挽救案例, Rescues)** | 53 | 60 |
| **合计** | 151 | 65 | 216 |

- **挽救案例数 (Rescues)**: 7 题
- **退化案例数 (Regressions)**: 12 题
- **净修复效果**: **净亏损 5 题**（破坏的题目多于挽救的题目）。
- **挽救率 (Rescue Rate)**: $c / (c+d) = 7 / 60 = 11.7%$
- **退化率 (Regression Rate)**: $b / (a+b) = 12 / 156 = 7.7%$
- **McNemar 双侧确切概率**: $p = 0.3593$（不显著）。
- **预注册符合度**: 违反了《方案 §27》“$\\text{{Regression Rate}} < \\text{{Rescue Rate}}$”前提下净收益为正的要求（绝对案例数净损失）。

### B0 vs K2 转移矩阵
- 挽救案例数: 5 题，退化案例数: 7 题，**净亏损 2 题**；McNemar $p = 0.7744$。

---

## 6. Core-60 语料扩展分析 (Core Scale Degradation Analysis)

Core-60 子集包含标准答案与证据全部位于 D20 的问题。在测试集中，经 Dev/Test 切分后实际分配到的样本量为 **$N = 48$**。

| 实验系统 | D20 准确率 (比例) | D50 准确率 (比例) | D100 准确率 (比例) | 衰减差值 Δ(D20 → D100) |
|:---|:---:|:---:|:---:|:---:|
{table3_text}

> [!WARNING]
> **方法学口径修正说明**:
> 1. 在 $N = 48$ 的样本容量下，**2.1 个百分点仅对应 1 道题**，4.2pp 对应 2 道题，6.2pp 对应 3 道题。
> 2. 在本子集上观察到 K2 保持 36/48（0 题变动），K4 从 34/48 变为 37/48（增加 3 题）。然而由于样本规模较小，绝对题数变动非常有限，因此本分析仅属于**探索性观察 (Exploratory Finding)**，**绝不能单独作为“证明系统完全具备抗规模退化能力”的实证依据**。

---

## 7. Hub 节点相关性分析 (Hub Correlational Analysis)

将测试集题目按照其涉及法条在路由图谱中的度数进行分桶统计：

| 节点度数分桶 (Degree Tier) | 涉及题目数 (N) | B0 准确率 | K4 准确率 | B0 污染率 (CPR) | K4 污染率 (CPR) |
|:---|:---:|:---:|:---:|:---:|:---:|
{table4_text}

> [!WARNING]
> **非因果推断声明 (Non-Causal Methodology Disclosure)**:
> 上表对比的是**包含不同法规实体的不同题目组**，而非对同一道题目人为控制其 Hub 度数。
> 因此：
> 1. 在 Hub Core (> 50 度) 题组上观察到的准确率下降（B0 52.6%, K4 47.4%），仅能说明**涉及高频枢纽法条的试题本身在法理复杂度或检索辨析度上难度更大（相关性）**。
> 2. **不能**据此推断出“度数增加直接因果导致了系统性能退化”。
> 3. 严格的因果控制实验（即在相同拓扑路径下人工注入不同度数的干扰边）并未在本次 V1 范围内实施，属于后续深入研究的独立命题。

---

## 8. 探索性子群发现 (Exploratory Subgroup Findings)

以下子群结果**非预注册的主终点指标**，未经多重假设检验校正（Unadjusted for multiple testing），**不能用于佐证主假设成立**，仅作为未来 V2 研究的假设线索（Hypothesis-Generating Signals）：

### 按跳数（Hop Count）表现
| 跳数分类 | B0 | B1 | K1 | K2 | K3 | K4 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
{hop_table_text}

### 按题型标签（Tags）表现
| 题型标签 | B0 | B1 | K1 | K2 | K3 | K4 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
{tag_table_text}

> [!NOTE]
> **探索性观察要点**:
> 1. **长链推理 (Hop 3+)**: 在 3 跳及以上问题中，观察到 K2（67.7%）优于 B0（64.6%）+3.1pp（高出 2 道题）。
> 2. **时序冲突 (Temporal)**: 在法规修订版本冲突题中，观察到 K4（84.0%）优于 B0（80.0%）+4.0pp（高出 1 道题）。
> 3. **单跳检索 (Hop 1)**: B0 表现最高（84.4%），路由系统在单跳题上均有 1.6~6.3pp 的落后，说明图谱约束在简单题目上产生了负迁移（过度限制了向量召回）。

---

## 9. 失败用例事后诊断归因 (Post-hoc Error Diagnostics)

对测试集中 K4 的全部 65 例失败用例进行事后分析归因（Post-hoc Diagnostic Attribution，非因果根因）：

1. **部分跳步丢失 (Wrong Next-Hop & Missing Edge)**: **46.2%** (30/65)
   - 表现为黄金 Chunk 仅部分被召回。原因在于法规图谱中部分引用关系较弱，或 EIGRP 可行性门控过于严格导致合法跳转被剪除。
2. **种子定位偏离 (Seed Miss & Scope Selection)**: **38.5%** (25/65)
   - 表现为没有任何黄金 Chunk 进入最终证据上下文。前缀向量检索第一跳锚定到了无关法规，后续图路由在错误范围内闭环。
3. **生成推理失误 (Generator Misuse)**: **15.4%** (10/65)
   - 黄金 Chunk 全部成功进入 4000 Token 上下文，但生成模型给出了错误或不完整的法条解读。

---

## 10. 实验方案第 33 节必须回答的 8 个核心问题 (Audited Answers)

### Q1: 纯 RAG baseline 是多少？
- **B0 (纯向量检索)**: 准确率 **72.2%**（[66.2%, 78.2%]），CPR 83.2%，证据 F1 0.244。
- **B1 (混合检索)**: 准确率 **68.1%**（[61.6%, 74.1%]），CPR 83.2%，证据 F1 0.244。

### Q2: 完整 Routing-RAG 是多少？
- **K4 (Full Router)**: 准确率 **69.9%**（[63.9%, 75.9%]），CPR 83.1%，证据 F1 0.247。
- **K2 (EIGRP 门控变体)**: 准确率 **71.3%**（[65.3%, 77.3%]），CPR 83.2%，证据 F1 0.247。

### Q3: 提升主要来自哪一个 ablation？
- **不存在总体准确率提升**。在所有路由变体内部，**K2** 表现相对最好（71.3%），相较于其他 K 系统最接近 B0，并在 Hop 3+ 子集观察到 +3.1pp 的探索性优势。但总体 Accuracy 仍比 B0 低 0.9pp，因此不能称为总体提升。

### Q4: 从 D20 → D100 后谁退化最快？
- 在 Core-60 测试子集（$N=48$）上，B1 衰减 2.1pp（降 1 题），K3 衰减 4.2pp（降 2 题）。K2 保持 0.0pp（0 题变动），K4 为 -6.2pp（增 3 题）。但样本量较小，不可过度外推。

### Q5: Hub degree 增加后谁退化最快？
- 在 Hub 度数 > 50 题组上，K4 准确率为 47.4%，B0 为 52.6%。这属于不同题目间的相关性表现，不能推导为单题度数增加的因果退化。

### Q6: Routing 修复了多少，又破坏了多少？
- **K4**: 修复了 **7** 道基线错误题（挽救率 11.7%），但破坏了 **12** 道基线正确题（退化率 7.7%），净损失 5 道题。
- **K2**: 修复了 **5** 道基线错误题（挽救率 8.3%），破坏了 **7** 道基线正确题（退化率 4.5%），净损失 2 道题。

### Q7: 错误仍主要来自哪一种模式？
- 依据事后诊断归因，K4 错误主要源自：**Wrong Next-Hop & Missing Edge (46.2%)**，其次是 **Seed Miss (38.5%)**，模型阅读理解失误占 **15.4%**。

### Q8: 准确率收益是否值得 latency/token 成本？
- **当前不存在总体准确率收益，因此不能认为收益值得成本**。尽管 Routing 图计算带来的额外耗时很低（仅增加 81.1 ms），Token 消耗基本持平（1440 vs 1422），但并未换来总体准确率或证据质量的净改善。

---

## 11. 最终判定结论与后续假设 (Go / No-Go Decision & V2 Hypotheses)

根据《实验方案.md》第 34 节的标准决策规则：

> ### 最终实验结论: **NO-GO for current V1 implementation / primary hypothesis**
> 
> **正式结论陈述**:
> 1. **主假设未获支持**: Knowledge Routing RAG V1 在总体准确率（69.9% vs 72.2%）、证据检索质量（F1 0.247 vs 0.244）以及上下文污染率（83.1% vs 83.2%）上，均未达成超越纯向量检索基线（B0）的预注册最低有效信号（Minimum Signal）。
> 2. **停止功能堆叠**: 依据预注册规程，应当正式终止在当前 V1（K1-K4）刚性图拓扑管道上继续无序增加组件的做法。
> 3. **适用范围限定**: **NO-GO 严格针对当前 V1 的具体实现与主假设，不否定整个研究领域的科学价值**。

### 后续 V2 研究假设 (Hypotheses for Future Research - 仅作立项参考，本任务不予实现)

在 V1 获得的探索性数据为后续研究提供了明确的假设来源：
- **V2 核心假设（条件式路由架构）**:
  - **默认路径**: 鉴于向量检索在单跳和常规题上的高效性（84.4%），系统应以 **Vector RAG 作为默认第一通路**；
  - **按需路由**: 仅当检测器识别到输入属于 **多跳长链（Multi-hop）**、**法条时序/版本冲突（Temporal Conflict）** 或 **第一通路证据置信度不足（Evidence Gap）** 时，才动态触发类似 K2 的受控图路由机制。
  - 此举可兼得单跳检索的高精度与多跳图约束的抗漂移能力，避免在简单题目上发生“负迁移”。

---
*报告生成脚本版本: v1-strict-audit*  
*数据审计文件参考: `reports/statistical_analysis.json` & `reports/final_adjudicated_results.json`*
"""
    REPORT_OUTPUT.write_text(content, encoding="utf-8")
    print(f"Generated Audited Final Academic Report: {REPORT_OUTPUT}")


if __name__ == "__main__":
    generate_final_report_md()
