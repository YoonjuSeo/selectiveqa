# -*- coding: utf-8 -*-
"""diag_h2_decompose.py — M2 EM 손실의 3분해 + 출력 길이 회귀 확인.

손실 = 거짓 무응답 + 형식 회귀(내용은 정답: 포함/극성/수치 일치) + 진짜 오답
사용: python src/analysis/diag_h2_decompose.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import re
import string
import unicodedata
from collections import Counter

import numpy as np

_PUNCT = set(string.punctuation) | {"·", "…", "「", "」", "『", "』", "%"}
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize(t):
    if t is None:
        return ""
    t = unicodedata.normalize("NFKC", str(t)).lower()
    return "".join(ch for ch in t if not ch.isspace() and ch not in _PUNCT)


def nums(t):
    return [] if t is None else [float(m.replace(",", "")) for m in _NUM_RE.findall(str(t))]


def strict_correct(pred, gold, qtype, tol=1e-4):
    if qtype == "numeric_reasoning":
        g, p = nums(gold), nums(pred)
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


def lenient_correct(pred, gold, qtype, tol=1e-4):
    """형식 회귀 판별용 관대 채점: 내용이 맞으면 True.
    - text/table: 정규화 후 gold ⊆ pred 또는 pred ⊆ gold (장황·단위 추가 흡수)
    - numeric: gold 수치가 pred 수치 집합에 존재 (기존과 동일 + 포함)
    - yes_no: 극성 일치 또는 포함
    """
    if strict_correct(pred, gold, qtype, tol):
        return True
    np_, ng = normalize(pred), normalize(gold)
    if qtype == "numeric_reasoning":
        g, p = nums(gold), nums(pred)
        return bool(g and p and any(abs(x - y) <= tol * max(1.0, abs(y))
                                    for y in g for x in p)) or (ng and ng in np_)
    if qtype == "yes_no":
        return False  # 극성 판정 실패 시 관대 인정 없음 (보수적)
    return bool(ng and (ng in np_ or (np_ and np_ in ng)))


def load(p):
    return {r["question_id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}


# polarity_v2 커버리지 확인
from evaluation.polarity_v2 import polarity_v2 as polarity
print("polarity_v2 스팟체크:",
      {w: polarity(w) for w in ["Yes", "No", "예", "아니오", "그렇다", "아니다", "맞다"]})

for seed in ("42", "43", "44"):
    m1 = load(f"results/preds_M1_v2_s{seed}.jsonl")
    m2 = load(f"results/preds_M2_r05_s{seed}.jsonl")
    ans = [q for q, r in m1.items() if r.get("gold_answerable", True)]
    n = len(ans)

    m1_ok = {q for q in ans if strict_correct(m1[q]["prediction"], m1[q]["gold_answer"], m1[q]["type"])}
    fr = [q for q in m1_ok if m2[q].get("answerable_pred") is False]
    answered_wrong = [q for q in m1_ok
                      if m2[q].get("answerable_pred") is not False
                      and not strict_correct(m2[q]["prediction"], m2[q]["gold_answer"], m2[q]["type"])]
    fmt = [q for q in answered_wrong
           if lenient_correct(m2[q]["prediction"], m2[q]["gold_answer"], m2[q]["type"])]
    true_wrong = [q for q in answered_wrong if q not in set(fmt)]

    len1 = np.mean([m1[q].get("n_gen_tokens", np.nan) for q in ans])
    len2 = np.mean([m2[q].get("n_gen_tokens", np.nan) for q in ans])

    print(f"\n===== 시드 {seed} (응답가능 {n}건, M1 정답 {len(m1_ok)}건) =====")
    print(f"손실 3분해 (M1 정답 → M2 손실 {len(fr)+len(answered_wrong)}건):")
    print(f"  ① 거짓 무응답        : {len(fr):>4}건 ({len(fr)/n:5.1%}p)")
    print(f"  ② 형식 회귀(내용 정답): {len(fmt):>4}건 ({len(fmt)/n:5.1%}p)  ← 관대 채점 시 복구")
    print(f"  ③ 진짜 오답          : {len(true_wrong):>4}건 ({len(true_wrong)/n:5.1%}p)")
    print(f"  ③ 유형 분포: {dict(Counter(m1[q]['type'] for q in true_wrong))}")
    print(f"평균 생성 토큰 (응답가능): M1 {len1:.1f} → M2 {len2:.1f}")

    # 관대 채점 기준 EM 차이 (탐색적 — H2 판정은 사전등록 EM 그대로)
    em1_len = sum(lenient_correct(m1[q]["prediction"], m1[q]["gold_answer"], m1[q]["type"]) for q in ans) / n
    em2_len = sum(m2[q].get("answerable_pred") is not False
                  and lenient_correct(m2[q]["prediction"], m2[q]["gold_answer"], m2[q]["type"])
                  for q in ans) / n
    print(f"[탐색적] 관대 채점 EM: M1 {em1_len:.3f} → M2 {em2_len:.3f} (Δ {em2_len-em1_len:+.3f})")