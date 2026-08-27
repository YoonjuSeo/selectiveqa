# -*- coding: utf-8 -*-
"""check_preds.py — preds 파일들의 건수·중복·평가셋 정합성 점검."""
import json
from pathlib import Path

# 기준: 평가셋별 question_id 집합
def load_ids(p):
    return {json.loads(l)["question_id"] for l in open(p, encoding="utf-8")}

eval_v2_ids = load_ids("data/processed/eval_full_v2.jsonl")
print(f"eval_full_v2: {len(eval_v2_ids)}건\n")

for f in sorted(Path("results").glob("preds_*.jsonl")):
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    ids = [r["question_id"] for r in rows]
    dup = len(ids) - len(set(ids))
    in_v2 = len(set(ids) & eval_v2_ids)
    match = "= v2 전체 일치" if set(ids) == eval_v2_ids else f"(v2와 교집합 {in_v2})"
    print(f"{f.name}: {len(rows)}건, 중복 {dup}, {match}")