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
│           Accuracy: 73.15% (+0.93pp vs B0, +1.85pp vs C1) | CANDIDATE / NEW INCUMBENT ★
│           Net Rescue: +2 (2 Rescues, 0 Regressions)
│           Evidence F1: 0.246 | Chain Comp: 34.3%
│           Latency P50: 95 ms | CPR: 83.1%
│           [Zero-Regression Architecture: Selectively engages routing only on cross-statute tasks]
```

**Current Champion**: `C3` (Accuracy: **73.15%**, Net Rescue: **+2**, Regressions: **0**)
