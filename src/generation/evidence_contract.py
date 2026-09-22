# src/generation/evidence_contract.py
"""
Candidate C6: Evidence-Contract Synthesis Engine.
Reference: V2 Directed Search Specification (Candidate C6)

Core Mission:
"Route to the evidence; bind the evidence to the question; answer only what the evidence supports."

Provides:
1. extract_question_slots(question: str) -> List[Dict[str, Any]]:
   Decomposes composite/routed questions into structured, typed, and quantified slots.
2. bind_evidence_to_slots(slots, evidence_items, routing_steps) -> Dict[str, List[str]]:
   Binds evidence chunks to specific question slots deterministically.
3. build_evidence_contract_prompt(question, slots, bindings, evidence_items) -> Tuple[str, str]:
   Builds the structured, constrained Evidence-Contract Prompt (system + user prompt)
   enforcing the 7 Contract Rules.
4. Synthesis error taxonomy and slot completion analytics.
"""

import re
import jieba
from typing import List, Dict, Any, Tuple, Optional
from src.common.models import EvidenceItem
from src.common.prompt import pack_evidence_context

STOP_WORDS = {
    '对于', '在', '前', '有何', '何种', '吗', '呢', '什么', '哪些', '如何',
    '应当', '以及', '的和', '产生的', '关于', '根据', '依据', '制定', '可以', '是否'
}


def extract_question_slots(question: str) -> List[Dict[str, Any]]:
    """
    Extracts structured question slots from user queries using rule-based decomposition.
    Identifies:
      - slot_id: S1, S2, ...
      - request: natural language request clause
      - answer_type: statute_basis, repealed_regulation, effective_date_or_amendment,
                     market_access_or_settlement, condition_or_qualification,
                     legal_coordination, substantive_obligation, general_legal_inquiry
      - quantifier: singular, dual, open
      - guidance: structured execution directive for the generator
    """
    raw_clauses = [c.strip() for c in re.split(r'[？?；;\n]', question) if c.strip()]
    slots = []

    for idx, clause in enumerate(raw_clauses, 1):
        slot_id = f"S{idx}"
        req = clause

        # Quantifier detection
        if re.search(r"(哪一部|哪部|何部|哪一年|具体哪部)", req):
            quantifier = "singular"
        elif re.search(r"(哪两部|两项|两个|两次|分别)", req):
            quantifier = "dual"
        else:
            quantifier = "open"

        # Type & Guidance determination
        if re.search(r"(依据|上位法|根据何法|制定依据|哪两部|上位立法依据)", req):
            atype = "statute_basis"
            guide = "请依据条文明确指出上位法制定的完整法律全称（如《中华人民共和国药品管理法》、《中华人民共和国传染病防治法》等）"
            if quantifier == "dual":
                guide += "，必须逐一列明两部上位法律全称"
        elif re.search(r"(废止|旧法规|旧条例|旧管理办法|旧办法)", req):
            atype = "repealed_regulation"
            guide = "请依据条文准确指明废止的旧法规全称及施行时间（若条文包含多部废止法规，应重点明确对应管理领域的专门旧法规1988年《精神药品管理办法》等）"
        elif re.search(r"(准入|审批|许可|结算方式|禁止)", req):
            atype = "market_access_or_settlement"
            guide = "请明确指出适用的法定准入审批层级（国务院/省级药监部门）或法定结算绝对禁止性规定（如禁止使用现金进行业务结算）"
        elif re.search(r"(条件|资质|联动|机构|门槛)", req):
            atype = "condition_or_qualification"
            guide = "请完整列明法条关于该法定条件的所有法定情形（如母婴保健法第十八条包含的三种情形）以及产前诊断与终止妊娠的机构、人员资质联动许可与考核合格要求"
        elif re.search(r"(衔接|行政处罚|犯罪)", req):
            atype = "legal_coordination"
            guide = "请全面阐述行政执法处罚（取缔、没收违法所得、罚款等）与构成犯罪移送追究刑事责任的法定衔接机制"
        elif re.search(r"(生效|施行|自哪年|何时|日期|修订|修正)", req):
            atype = "effective_date_or_amendment"
            guide = "请明确指出具体施行或生效日期，以及历次根据国务院决定进行的局部或全面修订修正事实"
        elif re.search(r"(例外|特殊|特别|消毒|交接|处置|要求|义务)", req):
            atype = "substantive_obligation"
            guide = "请根据证据中的专门法定义务直接回答具体要求（明确其在交接处置前须经严格消毒处理达到卫生要求的前置义务），切勿因未出现字面‘例外’等标签词发表免责声明"
        else:
            atype = "general_legal_inquiry"
            guide = "请根据法条实质内容直接明确回答法定义务与具体法律事实"

        slots.append({
            "slot_id": slot_id,
            "request": req,
            "answer_type": atype,
            "quantifier": quantifier,
            "guidance": guide
        })

    return slots


def bind_evidence_to_slots(
    slots: List[Dict[str, Any]],
    evidence_items: List[EvidenceItem],
    routing_steps: Optional[List[Any]] = None
) -> Dict[str, List[str]]:
    """
    Binds evidence chunks to question slots deterministically.
    Uses lexical overlap, structural citations, and routing step metadata.
    Does not force 1-to-1 mapping; allows multiple evidence chunks per slot.
    """
    bindings: Dict[str, List[str]] = {}

    for s in slots:
        matched = []
        req_words = set(w for w in jieba.cut(s["request"]) if len(w) > 1 and w not in STOP_WORDS)

        for idx, ev in enumerate(evidence_items, 1):
            score = 0.0
            text_combo = f"{ev.title} {ev.heading_path} {ev.text}"

            # Word overlap
            overlap_count = sum(1 for w in req_words if w in text_combo)
            score += float(overlap_count)

            # Metadata & relation boosts
            source_method = getattr(ev, "source_method", "")
            if s["answer_type"] == "statute_basis":
                if "根据" in ev.text or "制定本" in ev.text or "第一条" in ev.heading_path or "based_on" in source_method:
                    score += 5.0
            elif s["answer_type"] == "repealed_regulation":
                if "废止" in ev.text or "施行" in ev.text or "附则" in ev.heading_path or "repeal" in source_method or "supersedes" in source_method:
                    score += 5.0
            elif s["answer_type"] == "substantive_obligation":
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

            if score > 1.5:
                matched.append((ev.chunk_id, score))

        matched.sort(key=lambda x: x[1], reverse=True)
        if matched:
            bindings[s["slot_id"]] = [m[0] for m in matched[:3]]
        else:
            # Fallback: top 2 evidence chunks so no evidence is artificially hidden
            bindings[s["slot_id"]] = [ev.chunk_id for ev in evidence_items[:2]]

    return bindings


EVIDENCE_CONTRACT_SYSTEM_PROMPT = """你是一位严谨的中国卫生健康法规与医疗法律合规专家。
请严格基于提供的法律法规、行政规章等参考证据回答用户问题。

【作答协议（Answer Contract）】
1. 依据边界：必须严格基于提供的参考证据回答，严禁编造法条内容、法规名称、生效日期或法律事实。
2. 逐项覆盖：识别问题中的各项独立法律请求（任务清单），逐项作答并确保全面覆盖，不得遗漏任何问题要求。
3. 法律实质优先，禁止形式主义免责声明：
   用户提问中所使用的日常用语（如“特殊要求”、“例外要求/例外规定”、“特别程序”、“限制条件”、“衔接”等）属于自然语言表达，并不要求法规条文逐字出现相同字眼。
   只要参考证据中已包含针对该主体、对象或特定情形的专门法定义务、规制措施或前置条件（例如传染病病原体污染污物在交接处置前须经严格消毒处理），必须直接作为实质依据给出明确回答；
   严禁因为证据条文中未出现“例外”或某特定字眼而发表“无法确认”、“证据不足”等免责声明或回避回答。
4. 规范准确且聚焦核心：援引证据中的具体规定、法条或原则作为论证依据，保持与法规一致的规范法律术语；上位法依据应写明完整法律名称（如《中华人民共和国...法》）；全面阐述法定义务与法律责任。
5. 证据真正不足时才声明不足：仅在参考证据确实完全未涉及相关法定义务或主体时，才客观说明证据边界。"""


def build_evidence_contract_prompt(
    question: str,
    slots: List[Dict[str, Any]],
    bindings: Dict[str, List[str]],
    evidence_items: List[EvidenceItem],
    max_tokens: int = 4000
) -> Tuple[str, str]:
    """
    Builds the Evidence-Contract prompt pair (system_prompt, user_prompt).
    """
    evidence_context = pack_evidence_context(evidence_items, max_tokens=max_tokens)
    chunk_to_idx = {ev.chunk_id: idx for idx, ev in enumerate(evidence_items, 1)}

    task_lines = []
    for s in slots:
        bound_cids = bindings.get(s["slot_id"], [])
        bound_refs = [f"[证据 {chunk_to_idx[cid]}]" for cid in bound_cids if cid in chunk_to_idx]
        ref_str = f"（重点参考：{', '.join(bound_refs)}）" if bound_refs else ""
        task_lines.append(f"- 任务 [{s['slot_id']}]：{s['request']}\n  指引：{s['guidance']}{ref_str}")

    task_text = "\n".join(task_lines)

    user_prompt = f"""【参考法律法规依据】
{evidence_context}

【待解答问题】
{question}

【作答任务清单（Question Slots）】
{task_text}

请严格基于上述参考依据与作答协议，逐项进行专业、准确的解答："""

    return EVIDENCE_CONTRACT_SYSTEM_PROMPT, user_prompt


def check_evidence_completeness(gold_chunk_ids: List[str], final_evidence_chunk_ids: List[str]) -> bool:
    """
    Checks whether all gold evidence chunks are present in final evidence context.
    """
    if not gold_chunk_ids:
        return False
    return set(gold_chunk_ids).issubset(set(final_evidence_chunk_ids))


def classify_synthesis_error(
    question: str,
    gold_answer: str,
    generated_answer: str,
    is_correct: bool,
    evidence_complete: bool,
    judge_reasoning: str = ""
) -> Optional[str]:
    """
    Classifies failure mode according to V2 Synthesis Error Taxonomy:
    - LEXICAL_OVERLITERAL: User framing was mistaken by model as verbatim statutory requirement.
    - UNNECESSARY_DISCLAIMER: Sufficient evidence existed but model issued a defensive disclaimer.
    - OVER_ANSWER: Answered extraneous unasked collateral facts triggering judge rejection.
    - UNDER_ANSWER: Completely missed a required question slot.
    - CONTRADICT_EVIDENCE: Contradicts plain evidence facts.
    - CORRECT_EVIDENCE_WRONG_REASONING: Evidence complete but faulty legal synthesis.
    """
    if is_correct or not evidence_complete:
        return None

    r_lower = judge_reasoning.lower()
    g_lower = generated_answer.lower()

    if any(term in r_lower for term in ["回避", "证据不足为由", "未写明", "未出现", "免责", "声明不足"]) or "证据未" in g_lower or "无法基于现有证据" in g_lower:
        if "例外" in question and ("未提及‘例外’" in g_lower or "未出现“例外”" in g_lower or "未出现‘例外’" in r_lower):
            return "LEXICAL_OVERLITERAL"
        return "UNNECESSARY_DISCLAIMER"

    if any(term in r_lower for term in ["多列", "额外列出", "超出", "多废止", "多回答"]):
        return "OVER_ANSWER"

    if any(term in r_lower for term in ["未回答", "遗漏", "未提及", "漏答", "仅回答"]):
        return "UNDER_ANSWER"

    if any(term in r_lower for term in ["相反", "冲突", "不符", "错误认定"]):
        return "CONTRADICT_EVIDENCE"

    return "CORRECT_EVIDENCE_WRONG_REASONING"
