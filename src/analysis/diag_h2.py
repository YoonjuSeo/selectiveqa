# -*- coding: utf-8 -*-
"""diag_h2.py — H2 EM 급락(−0.34)의 원인 분해.

거짓 무응답으로 설명되는 몫 vs 응답 정확도 하락 몫을 유형별로 나누고,
'M1 정답 → M2 답변했으나 오답' 사례를 표본 출력해 채점 인공물 여부를 확인한다.
사용: python src/analysis/diag_h2.py  (프로젝트 루트에서)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import re
import string
import unicodedata
from collections import Counter

_PUNCT = set(string.punctuation) | {"·", "…", "「", "」", "『", "』", "%"}
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize(t):
    if t is None:
        return ""
    t = unicodedata.normalize("NFKC", str(t)).lower()
    return "".join(ch for ch in t if not ch.isspace() and ch not in _PUNCT)


def extract_numbers(t):
    return [] if t is None else [float(m.replace(",", "")) for m in _NUM_RE.findall(str(t))]


def is_correct(pred, gold, qtype, tol=1e-4):
    if qtype == "numeric_reasoning":
        g, p = extract_numbers(gold), extract_numbers(pred)
        if g and p:
            return any(abs(x - y) <= tol * max(1.0, abs(y)) for y in g for x in p)
        return normalize(pred) == normalize(gold)
    if qtype == "yes_no":
        from evaluation.polarity_v2 import polarity_v2 as polarity
        gp, pp = polarity(gold), polarity(pred)
        if gp is not None and pp is not None:
            return gp == pp
        return normalize(pred) == normalize(gold)
    return normalize(pred) == normalize(gold)


def load(p):
    return {r["question_id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}


m1 = load("results/preds_M1_v2_s42.jsonl")
m2 = load("results/preds_M2_r05_s42.jsonl")

ans_ids = [q for q, r in m1.items() if r.get("gold_answerable", True)]
print(f"응답가능 {len(ans_ids)}건 (시드 42 기준)\n")

hdr = f"{'유형':<20}{'n':>5}{'M1 EM':>8}{'M2 EM':>8}{'M2무응답':>9}{'M2조건부EM':>11}"
print(hdr)
rows_by_type = {}
for t in ("text_span", "table_lookup", "numeric_reasoning", "yes_no"):
    ids = [q for q in ans_ids if m1[q]["type"] == t]
    em1 = sum(is_correct(m1[q]["prediction"], m1[q]["gold_answer"], t) for q in ids) / len(ids)
    abst2 = [q for q in ids if m2[q].get("answerable_pred") is False]
    answered2 = [q for q in ids if m2[q].get("answerable_pred") is not False]
    corr2 = [q for q in answered2 if is_correct(m2[q]["prediction"], m2[q]["gold_answer"], t)]
    em2 = len(corr2) / len(ids)
    cond2 = len(corr2) / len(answered2) if answered2 else float("nan")
    print(f"{t:<20}{len(ids):>5}{em1:>8.3f}{em2:>8.3f}{len(abst2)/len(ids):>9.1%}{cond2:>11.3f}")
    rows_by_type[t] = (ids, answered2)

# 손실 분해: 거짓 무응답 몫 vs 응답 오답 몫
m1_correct = {q for q in ans_ids
              if is_correct(m1[q]["prediction"], m1[q]["gold_answer"], m1[q]["type"])}
fr_loss = [q for q in m1_correct if m2[q].get("answerable_pred") is False]
wrong_loss = [q for q in m1_correct
              if m2[q].get("answerable_pred") is not False
              and not is_correct(m2[q]["prediction"], m2[q]["gold_answer"], m2[q]["type"])]
n = len(ans_ids)
print(f"\nM1 정답 {len(m1_correct)}건 중 M2에서의 손실:")
print(f"  거짓 무응답으로 손실: {len(fr_loss)}건 ({len(fr_loss)/n:.1%}p)")
print(f"  답변했으나 오답:     {len(wrong_loss)}건 ({len(wrong_loss)/n:.1%}p)")
print(f"  오답 전환의 유형 분포: {dict(Counter(m1[q]['type'] for q in wrong_loss))}")

print("\n--- '답변했으나 오답' 표본 8건 (채점 인공물 여부 육안 확인) ---")
for q in wrong_loss[:8]:
    print(f"[{m1[q]['type']}] gold={m1[q]['gold_answer']!r}")
    print(f"   M1 pred={m1[q]['prediction']!r}")
    print(f"   M2 pred={m2[q]['prediction']!r}")