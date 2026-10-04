# -*- coding: utf-8 -*-
"""
klue_m1_check.py — KLUE M1 소거 판정 (사전등록 prereg_klue_m1_outproj_261004.md §3).

M1 예측 파일마다 다음을 계산한다. 판정 로직은 사전등록 §3 의 문턱을 그대로 옮긴 것이다.
  - easy / hard / 전체 UA 무응답률 (answerable_pred is False) — 1차 종점은 easy
  - 응답가능 EM (max-over-golds, evaluate.py 의 is_correct 그대로), false abstention
  - 정답 틀 적합률: 응답한 응답가능 문항 중 evidence_span 이 answer 와 같은 비율
    (학습 정답 틀 {"evidence_span": 정답} 을 따르는지. 금융 M1 은 이 틀을 재현했고 원 KLUE M1 은 지문 문장을 옮겼다)
  - 응답가능 평균 생성 토큰
  - H3 정의의 AUROC (응답가능 정답 vs hard UA 환각, confidence)
easy UA 무응답률과 AUROC 는 문항 재표집 bootstrap 95% CI 를 함께 낸다.

원 KLUE M1(q/k/v) 예측에도 같은 스크립트를 적용해 같은 정의로 비교한다
(원 m1fix_eval.py 는 저장소에 없어 정의가 같은지 확인할 수 없기 때문).

사용:
  python src/analysis/klue_m1_check.py results/klue_exaone_outproj/preds_M1_pin_s42.jsonl
  python src/analysis/klue_m1_check.py results/klue_exaone_outproj/preds_M1_pin_s*.jsonl --out results/klue_exaone_outproj/klue_m1_check_pin.json
  python src/analysis/klue_m1_check.py <원 KLUE M1 예측 경로들> --out results/klue_exaone/klue_m1_check_orig.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/

import argparse
import glob
import json
import re

import numpy as np

from evaluation.evaluate import is_correct, normalize

# ── 사전등록 §3 문턱 (결과 확인 전 고정) ─────────────────────────────
ERASED_MAX = 0.10      # easy UA 무응답률 ≤ 0.10 → 소거 성립
RETAINED_MIN = 0.60    # easy UA 무응답률 ≥ 0.60 → 소거 없음 (원 KLUE 결과 재현)
N_BOOT = 2000
BOOT_SEED = 42


def ua_kind(qid):
    if qid.startswith("unans_"):
        return "easy"
    if "-hd-" in qid:
        return "hard"
    return None


def parse_json_obj(text):
    """gen_text 에서 첫 JSON 객체를 꺼낸다. 실패하면 None."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = np.asarray(pos + neg, float)
    ranks = s.argsort().argsort().astype(float) + 1
    # 동점 평균 순위
    for v in np.unique(s):
        idx = s == v
        if idx.sum() > 1:
            ranks[idx] = ranks[idx].mean()
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def verdict(easy_rate):
    if easy_rate <= ERASED_MAX:
        return "소거 성립"
    if easy_rate >= RETAINED_MIN:
        return "소거 없음"
    return "부분 소거"


def analyze(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    ans, easy, hard = [], [], []
    for r in rows:
        k = ua_kind(r["question_id"])
        (easy if k == "easy" else hard if k == "hard" else ans).append(r)

    abst = lambda r: r.get("answerable_pred") is False
    correct = lambda r: (not abst(r)) and is_correct(r.get("prediction"), r["gold_answer"], r["type"], 1e-4)

    # 정답 틀 적합: 응답한 응답가능 문항에서 evidence_span == answer
    answered = [r for r in ans if not abst(r)]
    tmpl = []
    for r in answered:
        if r.get("gen_text") is None:      # 구버전 추론 파일은 gen_text 를 저장하지 않음 → 계산 제외
            continue
        obj = parse_json_obj(r["gen_text"])
        if obj is None:
            tmpl.append(False)
            continue
        tmpl.append(normalize(obj.get("evidence_span")) == normalize(obj.get("answer"))
                    and normalize(obj.get("answer")) != "")

    pos = [float(r["confidence"]) for r in ans if correct(r)]
    neg = [float(r["confidence"]) for r in hard if not abst(r)]

    easy_ab = np.array([abst(r) for r in easy], float)
    out = {
        "file": str(path),
        "n": {"answerable": len(ans), "easy": len(easy), "hard": len(hard)},
        "easy_ua_abstain": float(easy_ab.mean()),
        "hard_ua_abstain": float(np.mean([abst(r) for r in hard])),
        "all_ua_abstain": float(np.mean([abst(r) for r in easy + hard])),
        "answerable_em": float(np.mean([correct(r) for r in ans])),
        "false_abstention": float(np.mean([abst(r) for r in ans])),
        "template_adherence": float(np.mean(tmpl)) if tmpl else float("nan"),   # gen_text 없으면 nan
        "template_n": len(tmpl),
        "mean_gen_tokens_answerable": float(np.mean([r.get("n_gen_tokens", 0) for r in ans])),
        "auroc_h3": auroc(pos, neg),
    }

    rng = np.random.default_rng(BOOT_SEED)
    be, ba = [], []
    pos_a, neg_a = np.array(pos), np.array(neg)
    for _ in range(N_BOOT):
        be.append(easy_ab[rng.integers(0, len(easy_ab), len(easy_ab))].mean())
        if len(pos_a) and len(neg_a):
            ba.append(auroc(list(pos_a[rng.integers(0, len(pos_a), len(pos_a))]),
                            list(neg_a[rng.integers(0, len(neg_a), len(neg_a))])))
    out["easy_ua_abstain_ci"] = [float(np.percentile(be, 2.5)), float(np.percentile(be, 97.5))]
    out["auroc_h3_ci"] = ([float(np.nanpercentile(ba, 2.5)), float(np.nanpercentile(ba, 97.5))]
                          if ba else [float("nan")] * 2)
    out["verdict"] = verdict(out["easy_ua_abstain"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", help="M1 예측 파일 (glob 가능)")
    ap.add_argument("--out", default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()

    files = sorted({f for p in args.preds for f in (glob.glob(p) or [p])})
    results = []
    for f in files:
        o = analyze(f)
        results.append(o)
        lo, hi = o["easy_ua_abstain_ci"]
        alo, ahi = o["auroc_h3_ci"]
        print(f"\n== {f}  (응답가능 {o['n']['answerable']} · easy {o['n']['easy']} · hard {o['n']['hard']})")
        print(f"  easy UA 무응답률  {o['easy_ua_abstain']:.3f} [{lo:.3f}, {hi:.3f}]  → {o['verdict']}")
        print(f"  hard UA 무응답률  {o['hard_ua_abstain']:.3f} · 전체 UA {o['all_ua_abstain']:.3f}")
        print(f"  응답가능 EM {o['answerable_em']:.3f} · false abstention {o['false_abstention']:.3f}"
              f" · 정답 틀 적합률 {o['template_adherence']:.3f} · 평균 토큰 {o['mean_gen_tokens_answerable']:.1f}")
        print(f"  H3 AUROC {o['auroc_h3']:.3f} [{alo:.3f}, {ahi:.3f}]")

    vs = [o["verdict"] for o in results]
    print("\n== 시드 집계: " + " · ".join(f"{v} {vs.count(v)}" for v in ("소거 성립", "부분 소거", "소거 없음")))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"per_file": results,
                   "thresholds": {"erased_max": ERASED_MAX, "retained_min": RETAINED_MIN},
                   "prereg": "prereg_klue_m1_outproj_261004.md"},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
