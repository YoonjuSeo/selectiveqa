# -*- coding: utf-8 -*-
"""diag_yesno_vocab.py — [민감도 분석] yes_no 확장 어휘 재채점.

동결 원칙(polarity_v2.py)에 따라 본판정 채점기는 불변. 이 스크립트는
확장 어휘(그렇다/아니다 계열) 적용 시 수치가 어떻게 이동하는지만 별도 산출한다.
사용: python src/analysis/diag_yesno_vocab.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
from collections import Counter

from evaluation.polarity_v2 import (YES_WORDS, NO_WORDS, normalize,
                                    polarity_v2)

# ---- 확장 어휘 (민감도 전용 — 본판정 미사용) ------------------------------
EXT_YES = {"그렇다", "그렇습니다", "그래", "맞다", "맞아요", "맞음", "옳다"}
EXT_NO = {"그렇지 않다", "그렇지 않습니다", "아니다", "틀리다", "틀립니다",
          "옳지 않다", "안된다", "안 된다"}

_EXT_KEYWORDS = sorted(
    [(normalize(w), 1) for w in (YES_WORDS | EXT_YES)]
    + [(normalize(w), 0) for w in (NO_WORDS | EXT_NO)],
    key=lambda kv: (-len(kv[0]), kv[0]),
)


def polarity_ext(text):
    t = normalize(text)
    for kw, pol in _EXT_KEYWORDS:
        if kw in t:
            return pol
    return None


def load(p):
    return {r["question_id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}


print("확장 어휘:", sorted(EXT_YES), "/", sorted(EXT_NO), "\n")

for seed in ("42", "43", "44"):
    m1 = load(f"results/preds_M1_v2_s{seed}.jsonl")
    m2 = load(f"results/preds_M2_r05_s{seed}.jsonl")
    yn = [q for q, r in m1.items()
          if r.get("gold_answerable", True) and r["type"] == "yes_no"]
    n = len(yn)

    def em(preds, pol_fn):
        ok = 0
        for q in yn:
            r = preds[q]
            if r.get("answerable_pred") is False:
                continue
            gp, pp = pol_fn(r["gold_answer"]), pol_fn(r["prediction"])
            if gp is not None and pp is not None:
                ok += int(gp == pp)
            else:
                ok += int(normalize(r["prediction"]) == normalize(r["gold_answer"]))
        return ok / n

    # M2 미판정 출력의 패턴 (동결 채점기 기준)
    unparsed = [m2[q]["prediction"] for q in yn
                if m2[q].get("answerable_pred") is not False
                and polarity_v2(m2[q]["prediction"]) is None]
    rescued = sum(1 for p in unparsed if polarity_ext(p) is not None)

    print(f"===== 시드 {seed} (yes_no 응답가능 {n}건) =====")
    print(f"M2 답변 중 동결 채점기 미판정: {len(unparsed)}건"
          f" → 확장 어휘로 판정 가능: {rescued}건 (잔여 {len(unparsed)-rescued}건은 서술형 이탈)")
    print("미판정 출력 상위 패턴:",
          Counter(str(p)[:20] for p in unparsed).most_common(5))
    e1f, e2f = em(m1, polarity_v2), em(m2, polarity_v2)
    e1x, e2x = em(m1, polarity_ext), em(m2, polarity_ext)
    print(f"yes_no EM — 동결: M1 {e1f:.3f} / M2 {e2f:.3f} (Δ {e2f-e1f:+.3f})")
    print(f"yes_no EM — 확장: M1 {e1x:.3f} / M2 {e2x:.3f} (Δ {e2x-e1x:+.3f})")
    print(f"→ 확장으로 인한 Δ EM 이동: {(e2x-e1x)-(e2f-e1f):+.3f}\n")

    # [추가] 서술형 이탈 문항에서 M2 답이 M0 답과 유사한가 (M0 회귀 가설)
    m0 = load("results/preds_M0_v2.jsonl")
    drift = [q for q in yn if m2[q].get("answerable_pred") is not False
             and polarity_ext(m2[q]["prediction"]) is None
             and normalize(m2[q]["prediction"]) != normalize(m2[q]["gold_answer"])]
    sim = sum(1 for q in drift
              if normalize(m2[q]["prediction"]) and normalize(m0[q]["prediction"])
              and (normalize(m2[q]["prediction"]) in normalize(m0[q]["prediction"])
                   or normalize(m0[q]["prediction"]) in normalize(m2[q]["prediction"])))
    print(f"[M0 회귀 검증] 서술형 이탈 {len(drift)}건 중 M0 답과 포함관계 유사: {sim}건")