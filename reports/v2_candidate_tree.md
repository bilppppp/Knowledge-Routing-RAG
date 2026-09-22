# V2 Directed Search Champion Tree

```text
B0 Vector RAG (Anchor Baseline)
│  Accuracy: 72.22% (156/216)
│  F1: 0.244 | CPR: 83.2%
│
├── C1 Vector Entrance + Scope Narrowing + Feasibility Routing + Soft Hub + Targeted Gap + B0 Anchors
│     Accuracy: 71.30% (-0.93pp) | FAILED
│     Net Rescue: -2 (7 Rescues, 9 Regressions)
│     Evidence F1: 0.249 | Chain Comp: 36.1%
│     │
│     ├── C2 Hierarchical Scope Routing + Diversity Cap 2/doc + Confidence Fast Path
│     │     Accuracy: 68.52% (-3.70pp) | FAILED
│     │     Net Rescue: -8 (6 Rescues, 14 Regressions)
│     │     Evidence F1: 0.213 | Chain Comp: 29.6%
│     │     [Severe Context Dilution: Over-aggressive macro expansion evicted single-statute chunks]
│     │
│     └── C3 Intent-Gated Relational Router + 4 Guaranteed B0 Anchors + B0 Fast Path
│           Accuracy: 73.15% (+0.93pp vs B0, +1.85pp vs C1) | SUPERSEDED BY C4
│           Net Rescue: +2 (2 Rescues, 0 Regressions)
│           Evidence F1: 0.246 | Chain Comp: 34.3%
│           Latency P50: 95 ms | CPR: 83.1%
│           [Zero-Regression Architecture: Selectively engages routing only on cross-statute tasks]
│           │
│           └── C4 Relation-Specific Lanes (Temporal/Composite) + Conservative Evidence Admission
│                 Accuracy: 74.54% (+2.31pp vs B0, +1.39pp vs C3) | SUPERSEDED BY C5
│                 Net Rescue: +5 (5 Rescues, 0 Regressions)
│                 Evidence F1: 0.250 | Chain Comp: 34.3%
│                 Latency P50: 126 ms | CPR: 82.8%
│                 [High-Precision Architecture: Specialized Micro-Programs + Strict 5-chunk Budget Replacement]
│                 │
│                 └── C5 Hierarchical Next-Hop Resolution (Recursive Parent Lift + Targeted Descent)
│                       Accuracy: 75.93% (+3.70pp vs B0, +1.39pp vs C4) | SUPERSEDED BY C6
│                       Net Rescue: +8 (8 Rescues, 0 Regressions)
│                       Evidence F1: 0.251 | Chain Comp: 35.6%
│                       Latency P50: 121 ms | CPR: 82.7%
│                       [Recursive Architecture: Lift only to resolve; descend immediately to retrieve]
│                       │
│                       └── C6 Evidence-Contract Synthesis (Slot Decomposition + Semantic Binding + Substantive Contract)
│                             Accuracy: 77.31% (+5.09pp vs B0, +1.38pp vs C5) | SUPERSEDED BY C7
│                             Net Rescue: +11 vs B0 (+3 vs C5, 0 regressions vs B0)
│                             Evidence F1: 0.251 | Chain Comp: 35.6%
│                             Latency P50: 143 ms | CPR: 82.7%
│                             Evidence-Complete Accuracy: 90.91% (70/77 vs B0 84.42%)
│                             [Contract Architecture: Route to evidence; bind evidence to question; substantive reality over literal framing]
│                             │
│                             └── C7 Shadow Candidate Plane (Top-20 RIB) + Route-Prefix Resolution + Targeted Descent
│                                   Accuracy: 80.09% (+7.87pp vs B0, +2.78pp vs C6) | CANDIDATE / NEW INCUMBENT ★
│                                   Net Rescue: +17 vs B0 (+6 vs C6, 0 regressions vs B0)
│                                   Evidence F1: 0.253 | Chain Comp: 34.7%
│                                   Latency P50: 140 ms | CPR: 82.6%
│                                   Evidence-Complete Accuracy: 90.67% (68/75 vs B0 86.67%)
│                                   [Decoupled Architecture: Wide Control Plane RIB, Narrow Data Plane FIB; Longest Prefix Match]
```

**Current Champion**: `C7` (Accuracy: **80.09% (173/216)**, Net Rescue vs B0: **+17**, Net vs C6: **+6**, Regressions vs B0: **0**)
