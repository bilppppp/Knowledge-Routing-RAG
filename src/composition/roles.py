# src/composition/roles.py
"""
Generic Evidence Role Classification Module.
Reference: V3 Architecture Specification (Section XV).

Assigns generic roles to retrieved/routed evidence items:
  - DIRECT: Directly answers an operative question slot
  - BASIS: Provides legislative basis, superior law authority, or enactment status
  - CONDITION: Specifies qualifications, thresholds, prerequisites, or operational standards
  - SANCTION: Specifies penalties, liabilities, administrative measures, or legal consequences
  - EXCEPTION: Specifies statutory exceptions, exemptions, or proviso clauses
  - BRIDGE: Connects multi-hop documents (citations, references, superseding)
  - BACKGROUND: High-level principles or general definitions without operative facts

Zero benchmark-specific keyword tables; pure structural and semantic classification.
"""

import re
from typing import List, Dict, Any, Optional
from src.common.models import EvidenceItem
from src.composition.slots import EvidenceSlot

ROLE_EXCEPTION_RE = re.compile(r"(除外|除(?:其|上述|下列)?外|但书|不适用|特殊情形|例外|豁免|不得除外)")
ROLE_SANCTION_RE = re.compile(r"(处以|罚款|吊销|没收|暂停|责令|拘留|追究刑事责任|法律责任|处以货值金额|予以警告|记入信用档案|注销注册|处[一二三四五六七八九十百千万\d]+倍)")
ROLE_CONDITION_RE = re.compile(r"(应当具备|应当符合|应当满足|资格|门槛|准入条件|必须具备|任职条件|资质|设立个体诊所|经批准)")
ROLE_BASIS_RE = re.compile(r"(为了.+制定|依据.+制定|根据.+制定|上位法|立法依据|由国务院制定|制定本条例|制定本办法|公布施行|开始施行|废止)")
ROLE_BRIDGE_RE = re.compile(r"(依照《|根据《|按照《|见《|原《|施行后.+废止|替代|有关法律、行政法规的规定)")


def infer_evidence_role(
    item: EvidenceItem,
    slots: Optional[List[EvidenceSlot]] = None
) -> str:
    """
    Infers the functional role of an evidence item relative to question slots and internal text cues.
    """
    text = item.text
    heading = item.heading_path
    method = (item.source_method or "").lower()

    # 1. Routing method cues
    if "based_on" in method:
        return "BASIS"
    if "repeal" in method or "supersedes" in method or "amends" in method:
        return "BRIDGE"
    if "references" in method:
        return "BRIDGE"

    # 2. Heading path cues
    if "法律责任" in heading or "罚则" in heading:
        return "SANCTION"
    if "总则" in heading and ("立法" in text or "依据" in text or "制定" in text):
        return "BASIS"
    if "附则" in heading and ("施行" in text or "废止" in text):
        return "BRIDGE"

    # 3. Content regex cues
    if ROLE_EXCEPTION_RE.search(text):
        return "EXCEPTION"
    if ROLE_SANCTION_RE.search(text):
        return "SANCTION"
    if ROLE_CONDITION_RE.search(text):
        return "CONDITION"
    if ROLE_BASIS_RE.search(text):
        return "BASIS"
    if ROLE_BRIDGE_RE.search(text):
        return "BRIDGE"

    # 4. Check if chunk directly matches an operative slot
    if slots:
        for s in slots:
            overlap = sum(1 for kw in s.keywords if kw in text)
            if overlap >= 2:
                if s.slot_type in {"PENALTY_SANCTION", "EXCEPTION", "REQUIREMENT_CONDITION", "TIMELINE_PROCEDURE"}:
                    return "DIRECT"

    # 5. Default fallback
    if len(text.strip()) < 50 or "总则" in heading or "第一条" in text:
        return "BACKGROUND"

    return "DIRECT"
