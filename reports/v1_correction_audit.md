# Knowledge-Routing-RAG V1 实验闭环修正审计备忘录 (Correction Audit)

**审计基准**: 《实验方案.md》预注册规程（§25、§26、§27、§33、§34、§35）  
**审计目标**: 彻底消除报告中的过度解释与阳性偏差，恢复科学实证的客观性，使 V1 的结论完全由冻结的数据与预注册标准决定。

---

## 1. 核心结论与判定对照表

| 审计维度 | 原报告陈述 (Original Claim) | 修正后科学结论 (Corrected Verdict) | 修正依据与预注册规程 (Protocol Basis) |
|:---|:---|:---|:---|
| **最终结论判定** | **GO (强烈推荐推进工程化投产)**<br>声称“核心有效性成立、反常退化被克服、准确率收益完全值得”。 | **NO-GO for current V1 implementation / primary hypothesis**<br>明确 V1 主假设未获数据支持，停止在当前架构上堆叠功能。 | **《方案 §27, §34》**<br>K4 (69.9%) < B0 (72.2%)，未达到 Minimum Signal ($K4 > B0$)，更未达 Engineering Success ($K4 \ge B0+8\text{pp}$)。复杂度与开销上升且无净收益，按 §34 必须判定为 NO-GO。 |
| **评判裁决机制** | 单纯依赖 DeepSeek LLM Judge 的一次性自动打分作为最终 Accuracy 定论。 | **保留自动评分为 `auto_judge`，正式启动双盲人工仲裁包，标记 `PENDING HUMAN ADJUDICATION`**。 | **《方案 §25》**<br>明确规定“LLM Judge 可以作为辅助，但不要作为唯一最终裁判。对 baseline/router 不一致的记录进行人工仲裁”。 |
| **统计不确定性** | 仅展示单点估计（Point Estimates），未呈现置信区间，掩盖了抽样误差。 | **全面补齐各系统及 Paired Delta 的 95% Bootstrap CI**（5,000 次重采样）。 | **《方案 §26》**<br>要求必须报告 95% bootstrap CI。K4 vs B0 差值 CI 涵盖 [-6.48pp, +1.39pp]，点估计 -2.31pp，统计上彻底排除了 +8pp 的可能性。 |
| **挽救与退化案例** | 宣称“挽救案例数压倒性超越退化案例数 (Rescue ≫ Regression)”。 | **如实报告 12 例退化 vs 7 例挽救（净损失 5 道题）**，退化案例数显著多于挽救案例数。 | **《方案 §27》**<br>数据事实显示：K4 破坏了 12 道基线答对的题，仅挽救了 7 道基线答错的题；McNemar 检验 $p = 0.3593$ 不显著。 |
| **Core-60 规模扩展** | 宣称“实证证明 Router 架构成功抵御语料膨胀、完全克服噪声”。 | **修正为探索性发现（Exploratory Finding）**；披露测试集实际样本量仅 **$N=48$**，2.1pp 仅对应 1 道题。 | **样本量限制与科学谨慎原则**<br>在 $N=48$ 下绝对变动仅为 1~3 道题，样本容量不足以支持“完全克服规模退化”的外推结论。 |
| **Hub 压力测试** | 宣称“随着度数增加系统退化，证明 hard cut 导致退化”，做因果推断。 | **明确为相关性观察（Correlational），非因果推断**；披露该分析系跨题目横向比较，混淆了试题本身难度。 | **控制变量因果准则**<br>不同 degree 分桶包含不同试题。真正因果受控实验（同一路径改变 degree）留待后续，不伪装成已完成。 |
| **消融分析 Q3** | 宣称“提升主要来自 K2”。 | **修正为“不存在总体准确率提升”**；K2 仅在路由变体中最接近 B0，总体依然低 0.93pp。 | **事实一致性**<br>K2 (71.3%) 总体低于 B0 (72.2%)，负值不能定义为“提升”。 |
| **成本效益 Q8** | 宣称“准确率收益完全值得 latency/token 成本”。 | **修正为“当前不存在总体准确率收益，不能回答为值得”**。额外计算未带来正向回报。 | **经济性评估准则**<br>在没有正向准确率收益的前提下，任何额外图计算延迟与开销均为净成本。 |
| **局部子群信号** | 宣称“多跳与长链推理取得结构性优势”。 | **降级并隔离为“探索性子群发现 (Exploratory Findings)”**，明确未经多重检验校正，不可作为主假设成立依据。 | **多重比较防范 (Multiplicity Control)**<br>事后子群分析（Hop 3+ 增 2 题，Temporal 增 1 题）属于假设生成，不能替换主终点。 |
| **失败归因性质** | 宣称“精准定位根因、因果归因”。 | **规范措辞为“事后诊断归因 (Post-hoc Diagnostic Attribution)”**。 | **方法学严谨性**<br>非独立双盲人工逐题根因审计，属于基于规则的事后数据切片。 |
| **评测偏置披露** | 未提及生成模型与裁判模型同源性。 | **明确披露 Generator 与 Auto Judge 均使用 DeepSeek 模型家族，存在潜在偏置**。 | **评测透明度原则**<br>指出同源模型可能存在偏好一致性，双盲人工仲裁是消除此偏置的必要保障。 |

---

## 2. 交付与生成文件清单

1. [`reports/v1_historical_report.md`](v1_historical_report.md): 严格遵照 §25~§35 规程重写的完整学术报告（标记 `PENDING HUMAN ADJUDICATION`，结论锁定为 `NO-GO`）。
2. [`reports/blind_review_pack.json`](blind_review_pack.json): 包含全部 31 组争议对比对的双盲仲裁包（完全抹去所有系统和算法名称，仅保留随机化 Output A / Output B）。
3. [`reports/blind_review_mapping.json`](blind_review_mapping.json): 双盲仲裁对照密钥，单独留存，不提供给仲裁员。
4. [`reports/blind_review_decisions.json`](blind_review_decisions.json): 人工仲裁决策模板（不伪造仲裁数据，待真实评审回填）。
5. [`reports/final_adjudicated_results.json`](final_adjudicated_results.json): 追踪每题判断来源（`auto_high_confidence` vs `provisional_auto_pending_human_adjudication`）。
6. [`reports/statistical_analysis.json`](statistical_analysis.json): 包含 95% Bootstrap CI、Paired Deltas、McNemar 矩阵、Core-60 比例与诊断分布的完整机器可读统计库。

---

## 3. 核心原则总结

> **“一个可信、诚实、具有完备统计不确定性披露的阴性结果（Negative Result），其学术价值与工程指导意义远胜于一个被粉饰修饰后的阳性假象。”**

V1 的 NO-GO 结论严格证明了：在当前的知识库与模型能力下，直接在检索全流程上刚性强加图拓扑约束无法击败纯向量检索。这一结论为后续探索“以 Vector 为基座、仅在局部异常分支按需触发图路由”的条件式架构（V2 假设）奠定了真实的科学基石。
