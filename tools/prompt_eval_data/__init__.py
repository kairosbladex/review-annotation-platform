"""Prompt-gold snapshot tooling: read-only exporter, auditor and unified scorer.

Implements docs/PROMPT_GOLD_CONTRACT.md. No backend imports; stdlib only.
"""

SCHEMA_VERSION = "prompt-gold/2"
SCORE_SCHEMA_VERSION = "prompt-gold-score/1"

# Fixed class map; 0-9 matches data/ktd-formal-review-20260903/classes.txt,
# 车牌 appended as 10. Never reorder (see contract section 1).
CLASS_MAP = [
    (0, "三轮车"),
    (1, "两轮车"),
    (2, "人员"),
    (3, "光伏板"),
    (4, "光斑"),
    (5, "公交车"),
    (6, "卡车"),
    (7, "没戴头盔"),
    (8, "车"),
    (9, "违建"),
    (10, "车牌"),
]
CLASS_NAME_TO_ID = {name: cid for cid, name in CLASS_MAP}
CLASS_ID_TO_NAME = {cid: name for cid, name in CLASS_MAP}

SPLIT_SALT = "prompt-gold-split-v1"
