# Decontamination Rule Audit Report: C3–C7 Codebase Inspection

**Audit Target**: Knowledge-Routing-RAG (Commit `d202f0b`)  
**Evaluation Set**: N = 216 test instances (D20, D50, D100)  
**Audit Objective**: Identify, classify, and isolate benchmark-conditioned heuristics, prompt contamination, and human-in-the-loop test leakage from generic retrieval/routing architecture.

---

## 1. Classification Taxonomy

All rules, heuristics, and prompt components in C3–C7 are classified into three strict categories:

1. **`GENERIC_ARCHITECTURE`**:
   Mechanisms that embody domain-independent knowledge routing, graph control planes, and conservative admission principles. These are retained in `C7-Clean`.
   - Examples: Vector Top-5 Evidence Plane, Vector Top-20 Shadow Candidate Plane (RIB), Document-Prefix Aggregation, Prefix Scoring, EIGRP Feasibility Gate, Recursive Parent Lift, Conservative Evidence Admission (Replacement Policy, Budget Cap = 5).

2. **`DOMAIN_GENERAL`**:
   Patterns and heuristics that represent general legal or statutory document structures, not tied to specific benchmark instances or standard answers. These are retained in `C7-Clean`.
   - Examples: Chinese statutory citation syntax (`《...》`), statutory suffix normalization (`法/条例/办法/细则/规定`), jurisdiction stripping (`中华人民共和国`), general intent indicators (`施行/废止/依据/上位法`), generic reply title topic extraction.

3. **`BENCHMARK_CONDITIONED`**:
   Specialized heuristics, hardcoded identifiers, hand-crafted queries, and prompt guides directly inspired by error analysis of specific benchmark test questions (e.g., Q014, Q016, Q025, Q026, Q048, Q076, Q089).
   - **MANDATORY REMOVAL in `C7-Clean` and `Generic Contract`**.

---

## 2. Itemized Rule Audit Inventory

### Item 1: In-Document Targeted Descent Lookup Table

- **File**: `src/routing/c7_router.py`
- **Function**: `C7RouterSystem._build_descent_query(question: str, target_title: str)` (Lines 225–242)
- **Rule**:
  ```python
  if "医师法" in target_title and ("医德" in question or "职业道德" in question):
      return "医德医风 职业道德 评价 考评 考核 定期考核 暂停执业"
  elif "医师法" in target_title and ("个体诊所" in question or "乡村医生" in question):
      return "设立 个体诊所 医师 执业满五年 审批 备案 执业证书"
  elif "医疗机构管理条例" in target_title and "个体诊所" in question:
      return "设立 个体诊所 规划 许可证 登记"
  elif "传染病防治法" in target_title and "应急" in question:
      return "新发突发重大传染病 疫情 应急控制体系 紧急措施 疫区封锁"
  elif "医疗广告" in target_title:
      return "未取得 医疗机构执业许可证 擅自发布 医疗广告 查处 取缔"
  elif "母婴保健" in target_title and ("产前诊断" in question or "终止妊娠" in question):
      return "产前诊断 母婴保健技术服务 许可证 考核合格 执业 资质"
  ```
- **Classification**: `BENCHMARK_CONDITIONED`
- **Keep / Remove in Clean**: **REMOVE**
- **Reason**: Direct hardcoded lookup table mapping document titles and question keywords to specific pre-known gold standard answer terminology (e.g. injecting "定期考核 暂停执业" for Q026, "新发突发重大传染病...紧急措施 疫区封锁" for Q025, "产前诊断 母婴保健技术服务 许可证 考核合格" for Q076). Must be replaced with algorithmic, generic in-doc query construction.

---

### Item 2: Hardcoded Document Identifiers in Lane B Micro-Program

- **File**: `src/routing/c4_router.py` (Lines 299–316), `c5_router.py` (Lines 419–435), `c6_router.py` (Lines 311–326), `c7_router.py` (Lines 466–481)
- **Function**: `detect_lane` / `LANE_B_COMPOSITE_EVIDENCE` micro-program
- **Rule**:
  ```python
  # Special case A: "批发实行何种准入" / "准入" within doc006
  if "准入" in question or "批发" in question:
      for s in seed_items:
          if s.doc_id == "doc006":
              slot_items = self.search_service.fts_search_in_doc("批发 准入 国务院 省级", "doc006", corpus=corpus, top_k=1)
              for it in slot_items:
                  candidate_pool.append((it, 1.4, "Covered obligation slot: 批发准入"))
              break

  # Special case B: "机构" / "人员" / "资质" / "许可证" within doc024
  if "机构" in question and "资质" in question:
      for s in seed_items:
          if s.doc_id in ["doc024", "doc036"]:
              slot_items = self.search_service.fts_search_in_doc("产前诊断 资质 机构 许可", "doc024", corpus=corpus, top_k=1)
              for it in slot_items:
                  candidate_pool.append((it, 1.4, "Covered obligation slot: 机构人员资质许可"))
              break
  ```
- **Classification**: `BENCHMARK_CONDITIONED`
- **Keep / Remove in Clean**: **REMOVE**
- **Reason**: Directly specifies internal corpus document IDs (`doc006` = 麻醉药品和精神药品管理条例, `doc024` = 母婴保健法实施办法, `doc036` = 中华人民共和国母婴保健法) and manually crafted search strings to rescue Q048 and Q076. Violates generalized routing architecture.

---

### Item 3: Manually Injected Benchmark Topic Keywords in Document Alias Map

- **File**: `src/routing/c7_router.py`
- **Function**: `C7RouterSystem._build_doc_alias_map()` (Lines 98–105)
- **Rule**:
  ```python
  m_reply = re.search(r"关于(.+?)的(?:若干|几个|五个|六个|七个|十四个)?批复", title)
  if m_reply:
      core_topic = m_reply.group(1)
      aliases.add(core_topic)
      for sub in [
          "医疗广告", "乡村医生", "执业登记", "超范围执业",
          "个体诊所", "非法行医", "放射诊疗", "产前诊断",
          "药品使用", "医疗美容", "涉嫌犯罪", "继续犯罪"
      ]:
          if sub in core_topic:
              aliases.add(sub)
  ```
- **Classification**: `BENCHMARK_CONDITIONED`
- **Keep / Remove in Clean**: **REMOVE** sub-keyword loop; **KEEP** regex extraction of `core_topic`.
- **Reason**: The list of 12 sub-strings (`医疗广告`, `乡村医生`, `超范围执业`, `产前诊断`, etc.) was compiled from error traces on regulatory reply questions in this benchmark. The generic rule is simply to extract the core topic from `关于(.*?)的批复`.

---

### Item 4: Benchmark-Targeted Slot Query Expansion in Hierarchical Resolution

- **File**: `src/routing/c5_router.py` (Lines 96–104), inherited in `c6_router.py` and `c7_router.py`
- **Function**: `C5RouterSystem._clean_slot_query(slot_text: str)`
- **Rule**:
  ```python
  words = [w for w in jieba.cut(slot_text) if len(w.strip()) > 1 and w.strip() not in STOP_WORDS]
  action_boost = []
  if '消毒' in words:
      action_boost.extend(['严格消毒', '污物'])
  if '准入' in words or '批发' in words:
      action_boost.extend(['审批', '许可'])
  return ' '.join(words + action_boost)
  ```
- **Classification**: `BENCHMARK_CONDITIONED`
- **Keep / Remove in Clean**: **REMOVE**
- **Reason**: Injects exact statutory wording from Q014 gold chunk `doc005#c030` (`严格消毒 污物`) and Q048 (`审批 许可`). Slot query cleaning in clean architecture must rely strictly on generic stopword filtering.

---

### Item 5: Verbatim Benchmark Question Patterns in Lane Routing Regex

- **File**: `src/routing/c4_router.py` (Line 48), `c5_router.py` (Line 38), `c6_router.py`, `c7_router.py`
- **Constant**: `LANE_B_PATTERN`
- **Rule**:
  ```python
  LANE_B_PATTERN = re.compile(
      r"(有何要求.+又.+|以及.+有何|同时.+满足|联动要求|与.+衔接|协同衔接|分别.+规定|两部上位法|两项|两个.+条件|两个.+门槛|交叉门槛|何种准入.+在结算方式上|批发实行何种准入|出具医学意见终止妊娠的法定条件)"
  )
  ```
- **Classification**: Mixed
  - Generic conjunctions (`有何要求.+又.+`, `联动要求`, `与.+衔接`, `两部上位法`, etc.): `DOMAIN_GENERAL`
  - Suffix clauses (`何种准入.+在结算方式上|批发实行何种准入|出具医学意见终止妊娠的法定条件`): `BENCHMARK_CONDITIONED`
- **Keep / Remove in Clean**: **REMOVE** benchmark-specific suffix clauses; **KEEP** generic structural clauses.
- **Reason**: The last three clauses were lifted directly from the text of Q048 ("国家对特殊管理药品的批发实行何种准入？医疗机构购进...在结算方式上有何绝对禁止性规定？") and Q076 ("...出具医学意见终止妊娠的法定条件是什么？").

---

### Item 6: Benchmark-Specific Standard Answer Hints in Slot Guidance

- **File**: `src/generation/evidence_contract.py`
- **Function**: `extract_question_slots(question: str)` (Lines 60–86)
- **Rule**:
  ```python
  if re.search(r"(依据|上位法|根据何法|制定依据|哪两部|上位立法依据)", req):
      guide = "请依据条文明确指出上位法制定的完整法律全称（如《中华人民共和国药品管理法》、《中华人民共和国传染病防治法》等）"
  elif re.search(r"(废止|旧法规|旧条例|旧管理办法|旧办法)", req):
      guide = "请依据条文准确指明废止的旧法规全称及施行时间（若条文包含多部废止法规，应重点明确对应管理领域的专门旧法规1988年《精神药品管理办法》等）"
  elif re.search(r"(准入|审批|许可|结算方式|禁止)", req):
      guide = "请明确指出适用的法定准入审批层级（国务院/省级药监部门）或法定结算绝对禁止性规定（如禁止使用现金进行业务结算）"
  elif re.search(r"(条件|资质|联动|机构|门槛)", req):
      guide = "请完整列明法条关于该法定条件的所有法定情形（如母婴保健法第十八条包含的三种情形）以及产前诊断与终止妊娠的机构、人员资质联动许可与考核合格要求"
  elif re.search(r"(例外|特殊|特别|消毒|交接|处置|要求|义务)", req):
      guide = "请根据证据中的专门法定义务直接回答具体要求（明确其在交接处置前须经严格消毒处理达到卫生要求的前置义务），切勿因未出现字面‘例外’等标签词发表免责声明"
  ```
- **Classification**: `BENCHMARK_CONDITIONED`
- **Keep / Remove in Clean**: **REMOVE** all specific statute names, dates, and answers; **REPLACE** with generic legal analysis directives.
- **Reason**: These guidance strings explicitly prompt the generator with the exact answers for:
  - Q014: "在交接处置前须经严格消毒处理达到卫生要求的前置义务"
  - Q016: "1988年《精神药品管理办法》"
  - Q048: "禁止使用现金进行业务结算"
  - Q076: "母婴保健法第十八条包含的三种情形以及产前诊断与终止妊娠的机构、人员资质联动许可与考核合格要求"

---

### Item 7: Specific Benchmark Fact Injection in Evidence Contract System Prompt

- **File**: `src/generation/evidence_contract.py`
- **Constant**: `EVIDENCE_CONTRACT_SYSTEM_PROMPT` (Lines 156–167)
- **Rule**:
  ```text
  3. 法律实质优先，禁止形式主义免责声明：
     ...只要参考证据中已包含针对该主体、对象或特定情形的专门法定义务、规制措施或前置条件（例如传染病病原体污染污物在交接处置前须经严格消毒处理），必须直接作为实质依据给出明确回答；...
  ```
- **Classification**: `BENCHMARK_CONDITIONED`
- **Keep / Remove in Clean**: **REMOVE** the concrete example.
- **Reason**: Injects the verbatim factual answer of benchmark question Q014 into the system prompt. In a general legal contract, guidance must remain abstract.

---

### Item 8: Benchmark Keyword Boosts in Semantic Evidence-Slot Binding

- **File**: `src/generation/evidence_contract.py`
- **Function**: `bind_evidence_to_slots()` (Lines 130–142)
- **Rule**:
  ```python
  if s["answer_type"] == "substantive_obligation":
      if "消毒" in ev.text or "第二十七条" in ev.heading_path:
          score += 5.0
  elif s["answer_type"] == "market_access_or_settlement":
      if "现金" in ev.text or "禁止" in ev.text or "审批" in ev.text or "准入" in ev.text:
          score += 5.0
  elif s["answer_type"] == "condition_or_qualification":
      if "终止妊娠" in ev.text or "产前诊断" in ev.text or "资质" in ev.text or "许可" in ev.text:
          score += 5.0
  elif s["answer_type"] == "legal_coordination":
      if "追究刑事责任" in ev.text or "构成犯罪" in ev.text or "献血法" in ev.text:
          score += 5.0
  ```
- **Classification**: `BENCHMARK_CONDITIONED`
- **Keep / Remove in Clean**: **REMOVE**
- **Reason**: Uses task-specific keywords and exact article numbers (`第二十七条`, `献血法`, `现金`, `终止妊娠`) to artificially boost specific evidence chunks for Q014, Q048, Q076, and Q089. Binding must use purely general lexical overlap between the slot text and evidence text/metadata.

---

## 3. Retained Architectural Components in C7-Clean

The following components are strictly preserved as legitimate, generalizable architecture:

1. **Two-Plane Separation**:
   - Data Plane (Evidence Plane / FIB): Top-5 Vector Search baseline.
   - Control Plane (Candidate Plane / RIB): Top-20 Vector Search + Graph Topological Exploration.
2. **Document Prefix Aggregation**:
   - Chunks aggregated to parent document prefixes with multi-factor prefix scoring (vector score, density, entity match, typed edge connection).
3. **Route-Prefix Resolution**:
   - Longest Prefix Match / Specificity selection of up to 2 high-value external document prefixes.
4. **Hierarchical Next-Hop Resolution**:
   - PARENT_LIFT: elevating chunk to document node in LSDB.
   - DOC_RELATION_RESOLVE: traversing typed graph relations (`BASED_ON`, `REFERENCES`, `SUPERSEDES`, `AMENDS`).
   - TARGETED_DESCENT: in-document precision descent.
5. **Generic Algorithmic Targeted Descent**:
   - Automatically derives descent query as `keywords(unresolved_slot) + keywords(target_doc_title)`.
   - Zero lookup tables; zero pre-cooked answers.
6. **Conservative Evidence Admission**:
   - Replacement-first policy: Top 1–3 seeds locked; slots 4–5 replaceable.
   - Strict budget limit: maximum 5 final evidence chunks.
7. **Fast Path Routing Gate**:
   - Single-scope queries bypass graph exploration, eliminating graph pollution and regression on simple lookups.

---

## 4. Decontamination Audit Summary Table

| Component | Location | Original Implementation | Decontaminated Status | Rationale |
|:---|:---|:---|:---|:---|
| Descent Query | `c7_router.py:225` | 6 hardcoded `if/elif` branches | Algorithmic slot + title synthesis | Eliminates memorized answer terms |
| Lane B Special Cases | `c4~c7:300` | Hardcoded `doc006`, `doc024`, `doc036` | Deleted entirely | Enforces graph/prefix discovery |
| Document Aliases | `c7_router.py:98` | 12 hardcoded reply sub-keywords | Generic regex topic extraction | Prevents keyword-specific routing bias |
| Slot Query Cleaning | `c5~c7:99` | Hand-tuned `消毒`, `准入` boosts | Generic stopword & length filtering | Prevents targeted chunk hunting |
| Lane B Trigger Regex | `c4~c7:48` | Benchmark question verbatim snippets | General relational conjunctions only | Prevents query overfitting |
| Slot Guidance | `evidence_contract.py:60`| Mentions specific laws, dates, rules | Abstract statutory analysis tasks | Eliminates direct answer leakage |
| Contract System Prompt | `evidence_contract.py:164`| Q014 disinfection example | General legal compliance principles | Eliminates prompt contamination |
| Slot Evidence Binding | `evidence_contract.py:130`| Hardcoded keywords (`第二十七条` etc.) | Pure lexical overlap & relation tags | Eliminates artificial evidence favoritism |
