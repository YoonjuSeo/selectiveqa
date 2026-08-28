# -*- coding: utf-8 -*-
"""check_fields.py — eval_full_v2와 preds의 필드 구조 확인 (easy/hard UA 구분자 탐색)."""
import json
from collections import Counter

rows = [json.loads(l) for l in open("data/processed/eval_full_v2.jsonl", encoding="utf-8")]
print("eval_full_v2 필드:", sorted(rows[0].keys()))
ua = [r for r in rows if not r.get("answerable", True)]
print(f"UA {len(ua)}건. UA 전용 필드 값 분포:")
for k in ua[0].keys():
    vals = Counter(str(r.get(k))[:30] for r in ua)
    if 1 < len(vals) <= 6:
        print(f"  {k}: {dict(vals)}")

p = [json.loads(l) for l in open("results/preds_M2_r05_s42.jsonl", encoding="utf-8")]
print("\npreds 필드:", sorted(p[0].keys()))
print("orig_type 분포(UA만):", Counter(r.get("orig_type") for r in p if r["type"] == "unanswerable"))

print("\n--- easy/hard 구분자 검증 ---")
ua_none = [r["question_id"] for r in ua if r.get("gold_answer") is None]
ua_empty = [r["question_id"] for r in ua if r.get("gold_answer") == ""]
print(f"gold=None ({len(ua_none)}건) ID 예시:", ua_none[:5])
print(f"gold=''   ({len(ua_empty)}건) ID 예시:", ua_empty[:5])
# hard UA 생성 방식(근거 제거/대상 치환) 구분자도 있는지
print("gold='' ID 접두사 분포:", Counter(q.split("_")[0] for q in ua_empty))
print("gold=None ID 접두사 분포:", Counter(q.split("_")[0] for q in ua_none))