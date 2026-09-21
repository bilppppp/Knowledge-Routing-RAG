# src/common/prompt.py
"""
Prompt formatting and context token packing for Generator.
Reference: 实验方案.md Section 9, 20
"""

from typing import List
from src.common.models import EvidenceItem

SYSTEM_PROMPT = """你是一位严谨的中国卫生健康法规与医疗法律合规专家。
请严格基于提供的法律法规、行政规章和官方批复等参考证据回答用户问题。
要求：
1. 观点明确、结论清晰、法理准确；
2. 援引证据中的具体规定、法条或原则作为论证依据；
3. 如果参考证据不足以完全解答问题，必须如实客观说明已有规定和证据边界，严禁编造法律法规内容。"""

def pack_evidence_context(
    evidence_items: List[EvidenceItem],
    max_tokens: int = 4000
) -> str:
    """
    Packs evidence items into unified prompt text while strictly capping under max_tokens.
    (Approx 1.5 Chinese characters per token / ~6000 chars limit).
    """
    char_limit = int(max_tokens * 1.5)
    
    sections = []
    current_chars = 0
    
    for idx, ev in enumerate(evidence_items, 1):
        sec = f"[证据 {idx}] 《{ev.title}》 {ev.heading_path}\n{ev.text}\n"
        if current_chars + len(sec) > char_limit and sections:
            # Reached budget
            break
        sections.append(sec)
        current_chars += len(sec)
        
    return "\n".join(sections)

def format_user_prompt(question: str, evidence_context: str) -> str:
    return f"""【参考法律法规依据】
{evidence_context}

【待解答问题】
{question}

请严格基于上述依据进行专业解答："""
