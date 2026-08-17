# count_reused.py — easy UA 재사용 소스의 실소비 확인 (변경 관리 기록용)
import json
from collections import Counter
rows = [json.loads(l) for l in open("data/processed/hard_ua.jsonl", encoding="utf-8")]
reused = [r for r in rows if r.get("easy_ua_reused")]
print(f"전체 {len(rows)}건 중 easy UA 재사용 소스: {len(reused)}건")
print("유형×방식:", dict(Counter((r["orig_type"], r["ua_kind"]) for r in reused)))
for r in reused:
    print(" ", r["question_id"])