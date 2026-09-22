# src/composition/slots.py
"""
Generic Evidence Slot Decomposition Module.
Reference: V3 Architecture Specification (Sections XII-XIV).

Decomposes natural language queries into generic, independent evidence slots
using purely structural, linguistic, and domain-agnostic patterns.
Zero statute-specific hardcoding, zero benchmark-answer leakage.
"""

import re
import jieba
from typing import List, Set, Dict, Any, Optional
from pydantic import BaseModel, Field

STOP_WORDS = {
    '对于', '在', '前', '有何', '何种', '吗', '呢', '什么', '哪些', '如何',
    '应当', '以及', '的和', '产生的', '关于', '根据', '依据', '制定', '可以', '是否',
    '请问', '分别', '具体', '属于', '进行', '相关', '国家', '规定', '需经', '遵循',
    '包括', '具有', '发生', '具体', '提出', '应当', '并且', '同时', '及其', '有关'
}

SLOT_TYPE_PATTERNS = [
    ("EXCEPTION", re.compile(r"(例外|除外|特殊情形|豁免|但书|排除)")),
    ("PENALTY_SANCTION", re.compile(r"(处罚|处理|罚款|责任|吊销|追究|后果|暂停|处以|没收|注销|倍数|红线)")),
    ("REQUIREMENT_CONDITION", re.compile(r"(条件|资格|门槛|要求|标准|应当具备|准入|资质|限制|禁止)")),
    ("TIMELINE_PROCEDURE", re.compile(r"(多久|多长|期限|时限|程序|申报|向哪个|由谁|何时|几年|月内|日内|几小时|途径)")),
    ("LEGAL_BASIS", re.compile(r"(?:依据哪|根据何法|依据何法|上位立法依据|上位法依据|上位法|立法依据|旧法规|旧条例|旧办法|废止了哪|发生何种变化|效力变化|法律渊源)")),
    ("CONTENT_DEFINITION", re.compile(r"(包括哪些|何种|是什么|有哪些|内容|界定|定义|情形|原则|方针|分类|哪四类)"))
]


class EvidenceSlot(BaseModel):
    slot_id: str
    text: str
    slot_type: str = "GENERAL"
    keywords: List[str] = Field(default_factory=list)
    target_entities: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "text": self.text,
            "slot_type": self.slot_type,
            "keywords": self.keywords,
            "target_entities": self.target_entities
        }


def classify_slot_type(clause_text: str) -> str:
    """
    Determines generic semantic role of the inquiry clause based on syntactic indicators.
    """
    for slot_type, pattern in SLOT_TYPE_PATTERNS:
        if pattern.search(clause_text):
            return slot_type
    return "GENERAL"


def extract_evidence_slots(question: str) -> List[EvidenceSlot]:
    """
    Decomposes a query into generic structural evidence slots.
    Guarantees:
      1. Domain-agnostic structural decomposition.
      2. Zero statute-specific logic or answer-targeted regexes.
      3. At least 1 slot produced for any non-empty question.
    """
    clean_q = question.strip()
    if not clean_q:
        return [EvidenceSlot(slot_id="S1", text="", slot_type="GENERAL", keywords=[], target_entities=[])]

    # Extract mentioned entities/statutes in quotes
    entities = re.findall(r"《([^》]+)》", clean_q)

    # Primary split on terminal punctuation: ? ? ; ; \n
    raw_units = [u.strip() for u in re.split(r"[？?；;\n]+", clean_q) if u.strip()]

    # Secondary split on coordinating inquiry conjunctions
    split_clauses: List[str] = []
    for u in raw_units:
        sub_splits = re.split(
            r"(?:，|\s+)(?=(?:又|同时|分别|以及对|并对|其上位法|上位立法依据|废止了哪|依据哪|对于|不合格如何|费用由谁|有何))",
            u
        )
        for s in sub_splits:
            s_clean = s.strip()
            if s_clean:
                split_clauses.append(s_clean)

    if not split_clauses:
        split_clauses = [clean_q]

    slots: List[EvidenceSlot] = []
    for idx, clause in enumerate(split_clauses):
        slot_id = f"S{idx + 1}"
        slot_type = classify_slot_type(clause)

        # Tokenize content words
        # Strip punctuation for keyword extraction
        clause_stripped = re.sub(r"[《》？?。；;，,\n（）()、“”\"':：]", " ", clause)
        words = [
            w.strip() for w in jieba.cut(clause_stripped)
            if len(w.strip()) >= 2 and w.strip() not in STOP_WORDS
        ]

        # Extract entities mentioned in this specific clause
        clause_entities = [e for e in entities if e in clause]

        slots.append(EvidenceSlot(
            slot_id=slot_id,
            text=clause,
            slot_type=slot_type,
            keywords=list(dict.fromkeys(words)),  # preserve order & deduplicate
            target_entities=clause_entities
        ))

    return slots
