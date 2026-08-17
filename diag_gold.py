# diag_gold.py — 손상 gold 진단: 동결 목록 교차 검증 + 경계 구간 의심 건 출력
import json
from collections import Counter

rows = [json.loads(l) for l in open("data/processed/eval.jsonl", encoding="utf-8")]
frozen = set(json.load(open("results/excluded_gold_v2.json", encoding="utf-8"))["question_ids"])

leak, suspects = [], []
for r in rows:
    if not r.get("answerable", True):
        continue
    g = str(r.get("gold_answer", ""))
    if len(g) > 100 and r["question_id"] not in frozen:
        leak.append(r)                       # (a) 규칙상 잡혔어야 하는데 목록에 없음
    elif 60 < len(g) <= 100:
        suspects.append(r)                   # (b) 경계 구간 — 육안 판정 필요

print(f"(a) >100자인데 동결 목록 밖: {len(leak)}건  {'✓ 무결' if not leak else '⚠ 원인 확인 필요'}")
for r in leak:
    print(f"    [{len(str(r['gold_answer']))}자] {r['question_id']} ({r['type']})")

print(f"\n(b) 60~100자 의심 구간: {len(suspects)}건 (유형별 {dict(Counter(r['type'] for r in suspects))})")
for r in suspects:
    g = str(r["gold_answer"])
    print(f"    [{len(g)}자] {r['question_id']} ({r['type']}): {g[:70]}...")