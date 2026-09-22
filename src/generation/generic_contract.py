# src/generation/generic_contract.py
"""
Generic Answer Contract & Slot Synthesis Engine (Decontaminated).
Implements Sections XI & XII of Decontamination / Ablation Audit Specification:
- Generic Answer Contract (5 core principles, zero benchmark-specific hints).
- Generic Question Slot Decomposition (task-type guidance without law names or answers).
- Pure Lexical / Structural Evidence Binding (no hardcoded topic keywords).
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

GENERIC_CONTRACT_SYSTEM_PROMPT = """你是一位严谨的法律法规合规专家。
请严格基于提供的法律法规参考证据回答用户问题。

【作答协议】
1. 依据边界：必须严格基于提供的参考证据回答，严禁编造法律法规内容、名称、日期或事实。
2. 逐项覆盖：问题包含多个独立请求时，应识别各项任务，逐项作答并确保全面覆盖，不得遗漏任何问题要求。
3. 实质理解优先：用户自然语言措辞不要求与法规原文逐字一致；应根据证据表达的实际法律含义与法定义务直接回答，切勿因措辞差异发表形式主义免责声明或回避回答。
4. 规范克制：不主动回答问题未询问的额外事项，紧扣核心法律事实，保持专业严谨。
5. 证据真正不足时才声明不足：只有在参考证据确实完全未涉及解答所需依据时，才客观说明证据边界。"""


def extract_generic_question_slots(question: str) -> List[Dict[str, Any]]:
    """
    Decomposes question into structured question slots using generic structural patterns.
    Strictly free of any specific statute names, article numbers, or answer hints.
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

        # Type & Guidance determination (Generic legal categories)
        if re.search(r"(依据|上位法|根据何法|制定依据|哪两部|上位立法依据|法律渊源)", req):
            atype = "statute_basis"
            guide = "请依据参考证据明确指出上位法制定的完整法律全称"
            if quantifier == "dual":
                guide += "，若涉及多部，必须逐一列明全部上位法全称"
        elif re.search(r"(废止|旧法规|旧条例|旧管理办法|旧办法|替代)", req):
            atype = "repealed_regulation"
            guide = "请依据参考证据准确指明废止或替代的旧法规全称及施行或废止时间"
        elif re.search(r"(准入|审批|许可|结算方式|禁止|限制)", req):
            atype = "market_access_or_restriction"
            guide = "请明确指出适用的法定准入条件、审批主管层级或法定禁止性与限制性规定"
        elif re.search(r"(条件|资质|联动|机构|门槛|程序)", req):
            atype = "condition_or_qualification"
            guide = "请完整列明参考证据关于该法定事项的具体条件、资质要求或程序规范"
        elif re.search(r"(衔接|行政处罚|犯罪|责任|制裁)", req):
            atype = "legal_coordination"
            guide = "请全面阐述行政监管处罚措施与刑事责任或其他法律责任的法定衔接规定"
        elif re.search(r"(生效|施行|自哪年|何时|日期|修订|修正)", req):
            atype = "effective_date_or_amendment"
            guide = "请明确指出具体施行生效日期或历次修订修正事实"
        elif re.search(r"(例外|特殊|特别|要求|义务)", req):
            atype = "substantive_obligation"
            guide = "请根据参考证据中的专门法定义务直接回答具体法律要求，切勿因用词差异发表免责声明"
        else:
            atype = "general_legal_inquiry"
            guide = "请根据参考证据实质内容直接明确回答具体法定义务与法律事实"

        slots.append({
            "slot_id": slot_id,
            "request": req,
            "answer_type": atype,
            "quantifier": quantifier,
            "guidance": guide
        })

    return slots


def bind_generic_evidence_to_slots(
    slots: List[Dict[str, Any]],
    evidence_items: List[EvidenceItem],
    routing_steps: Optional[List[Any]] = None
) -> Dict[str, List[str]]:
    """
    Binds evidence chunks to question slots using purely lexical overlap and generic structural markers.
    Strictly avoids hardcoded domain keywords (e.g. 消毒, 现金, 第二十七条, 终止妊娠, etc.).
    """
    bindings: Dict[str, List[str]] = {}

    for s in slots:
        matched = []
        req_words = set(w for w in jieba.cut(s["request"]) if len(w) > 1 and w not in STOP_WORDS)

        for idx, ev in enumerate(evidence_items, 1):
            score = 0.0
            text_combo = f"{ev.title} {ev.heading_path} {ev.text}"

            # Word overlap with slot request
            overlap_count = sum(1 for w in req_words if w in text_combo)
            score += float(overlap_count)

            # Generic structural boosts based on source_method or general legislative patterns
            source_method = getattr(ev, "source_method", "")
            if s["answer_type"] == "statute_basis":
                if "based_on" in source_method or "根据" in ev.text or "制定本" in ev.text or "第一条" in ev.heading_path:
                    score += 3.0
            elif s["answer_type"] == "repealed_regulation":
                if "repeal" in source_method or "supersedes" in source_method or "废止" in ev.text or "施行" in ev.text or "附则" in ev.heading_path:
                    score += 3.0

            if score > 1.5:
                matched.append((ev.chunk_id, score))

        matched.sort(key=lambda x: x[1], reverse=True)
        if matched:
            bindings[s["slot_id"]] = [m[0] for m in matched[:3]]
        else:
            # Fallback: Top 2 evidence chunks so no slot is starved of context
            bindings[s["slot_id"]] = [ev.chunk_id for ev in evidence_items[:2]]

    return bindings


def build_generic_contract_prompt(
    question: str,
    slots: List[Dict[str, Any]],
    bindings: Dict[str, List[str]],
    evidence_items: List[EvidenceItem],
    max_tokens: int = 4000
) -> Tuple[str, str]:
    """
    Builds the Generic Contract prompt pair (system_prompt, user_prompt).
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

【作答任务清单】
{task_text}

请严格基于上述参考依据与作答协议，逐项进行专业、准确的解答："""

    return GENERIC_CONTRACT_SYSTEM_PROMPT, user_prompt
