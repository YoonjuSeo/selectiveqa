# -*- coding: utf-8 -*-
"""M0 gate 판정 + answer_span 폴백 원인 분해 (Stage F).

사용: python src/analysis/m0_gate_check.py results/ax31/preds_M0_v2.jsonl
"""
import json
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "results/ax31/preds_M0_v2.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8")]


def kind(r):
    q = r["question_id"]
    if q.startswith("unans_"):
        return "easy_UA"
    if "-hd-" in q:
        return "hard_UA"
    return "answerable"


groups = {}
for r in rows:
    groups.setdefault(kind(r), []).append(r)

print(f"전체 {len(rows)}건 · " + " · ".join(f"{k} {len(v)}" for k, v in groups.items()))
print()
print(f"{'구분':<11}{'무응답률':>9}{'파싱실패':>9}{'폴백':>8}{'폴백중 무응답':>14}{'평균토큰':>9}{'128토큰':>8}")
for k in ("answerable", "easy_UA", "hard_UA"):
    g = groups.get(k, [])
    if not g:
        continue
    n = len(g)
    abst = sum(r.get("answerable_pred") is False for r in g)
    pfail = sum(not r.get("parse_ok", True) for r in g)
    fb = [r for r in g if r.get("conf_scope") == "full_sequence"]
    fb_abst = sum(r.get("answerable_pred") is False for r in fb)
    ntok = sum(r.get("n_gen_tokens", 0) for r in g) / n
    cap = sum(r.get("n_gen_tokens", 0) >= 128 for r in g)
    print(f"{k:<11}{abst / n:>9.1%}{pfail / n:>9.1%}{len(fb) / n:>8.1%}"
          f"{(fb_abst / len(fb) if fb else 0):>14.1%}{ntok:>9.1f}{cap:>8}")

easy = groups.get("easy_UA", [])
if easy:
    rate = sum(r.get("answerable_pred") is False for r in easy) / len(easy)
    print(f"\n[Gate] easy UA 무응답률 = {rate:.1%} → {'통과' if rate >= 0.90 else '중단 (90% 미만)'}")

fb_all = [r for r in rows if r.get("conf_scope") == "full_sequence"]
reasons = Counter()
for r in fb_all:
    if r.get("answerable_pred") is False or r.get("prediction") is None:
        reasons["무응답(answer=null) — 정상적인 폴백"] += 1
    elif not r.get("parse_ok", True):
        reasons["파싱 실패"] += 1
    else:
        reasons["답은 있으나 생성문에서 answer 문자열을 못 찾음"] += 1
print(f"\n[폴백 {len(fb_all)}건 원인]")
for k, v in reasons.most_common():
    print(f"  {v:>4}건  {k}")
