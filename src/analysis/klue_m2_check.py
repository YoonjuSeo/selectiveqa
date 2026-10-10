# -*- coding: utf-8 -*-
"""
klue_m2_check.py — KLUE M2-r05 복원·비용 판정 (사전등록 prereg_klue_m2_r05_outproj_261010.md §3).

H1·H2·H3·H4·H5 의 점추정·CI 는 evaluate_followup.py 가 만든 metrics_followup_<tag>.json 을
그대로 읽는다(판정 로직을 다시 구현하지 않는다). 이 스크립트가 새로 계산하는 것은
evaluate_followup.py 에 없는 항목뿐이다.

  - R_tok(응답가능): M2 / M1 응답가능 평균 생성 토큰 (rtok_check.py 와 같은 정의)
  - M0 대비 복원: easy·hard UA 무응답률의 M2 − M0 차이 (question_id 로 짝지은 paired bootstrap 95% CI)
    와 비율 M2 / M0. 금융과 KLUE 의 hard UA 는 종류가 달라(D5) 도메인 간에는 비교하지 않는다.
  - 정답 틀 적합률(M2), false abstention(M1·M2)
  - 사전등록 §3 의 시드별 판정(복원 / 비용)과 3시드 집계

제외 목록은 적용하지 않는다. KLUE 평가셋은 제외 문항이 0건이다(Phase 2 사전등록 §4).
금융 예측에 쓰면 응답가능이 1,200건으로 잡혀 evaluate_followup.py(1,170건)와 n 이 다르다.

사용:
  python src/analysis/klue_m2_check.py \
      --metrics results/klue_exaone_outproj/metrics_followup_klue_r05_pin.json \
      --m0 results/klue_exaone/preds_M0.jsonl \
      --m1-glob "results/klue_exaone_outproj/preds_M1_pin_s*.jsonl" \
      --m2-glob "results/klue_exaone_outproj/preds_M2_r05_pin_s*.jsonl" \
      --out results/klue_exaone_outproj/klue_m2_check_r05_pin.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/

import argparse
import glob
import json
import re

import numpy as np

from analysis.klue_m1_check import parse_json_obj, ua_kind
from evaluation.evaluate import normalize

# ── 사전등록 §3 기준 (결과 확인 전 고정) ─────────────────────────────
RTOK_BAND = (0.90, 1.15)   # 무손실 띠 — Phase 2 사전등록 §3 의 무손실형 조건과 같은 값
COST_MARGIN = -0.03        # H2 마진. ΔEM CI 상한 < −0.03 이면 '비용 있음'
N_BOOT = 10000
BOOT_SEED = 42


def load(path):
    return {r["question_id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def abst(r):
    return r.get("answerable_pred") is False


def seed_of(p):
    m = re.search(r"_s(\d+)\.jsonl$", str(p))
    return m.group(1) if m else Path(p).stem


def mean_tokens(rows):
    return float(np.mean([r.get("n_gen_tokens", 0) for r in rows])) if rows else float("nan")


def template_adherence(rows):
    vals = []
    for r in rows:
        if abst(r) or r.get("gen_text") is None:
            continue
        obj = parse_json_obj(r["gen_text"])
        vals.append(obj is not None
                    and normalize(obj.get("answer")) != ""
                    and normalize(obj.get("evidence_span")) == normalize(obj.get("answer")))
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def paired_diff(m0, m2, qids, rng):
    """같은 문항에서 M2 무응답 − M0 무응답. 점추정, 95% CI, 비율 M2/M0."""
    a0 = np.array([abst(m0[q]) for q in qids], float)
    a2 = np.array([abst(m2[q]) for q in qids], float)
    boots = []
    n = len(qids)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boots.append(a2[idx].mean() - a0[idx].mean())
    return {
        "M0": float(a0.mean()), "M2": float(a2.mean()),
        "diff": float(a2.mean() - a0.mean()),
        "diff_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "ratio": float(a2.mean() / a0.mean()) if a0.mean() > 0 else float("nan"),
        "n": n,
    }


def judge(ci, judge_h, rtok):
    """사전등록 §3 시드별 판정."""
    restore = "복원 성립" if judge_h["H1_easy"] else "복원 불성립"
    d_lo, d_hi = ci["d_em"]["lo"], ci["d_em"]["hi"]
    in_band = RTOK_BAND[0] <= rtok <= RTOK_BAND[1]
    if judge_h["H2"] and in_band:
        cost = "비용 없음"
    elif d_hi < COST_MARGIN or rtok > RTOK_BAND[1]:
        cost = "비용 있음"
    else:
        cost = "비용 미결"
    return restore, cost


def aggregate(labels, target):
    k = sum(l == target for l in labels)
    n = len(labels)
    if k == 0:
        return "해당 없음", k
    return ("강건" if k == n and n >= 3 else "조건부" if n >= 3 and k >= 2 else "미결"), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="evaluate_followup.py 출력 JSON")
    ap.add_argument("--m0", required=True)
    ap.add_argument("--m1-glob", required=True)
    ap.add_argument("--m2-glob", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    met = json.load(open(args.metrics, encoding="utf-8"))
    m0 = load(args.m0)
    m1_files = {seed_of(p): p for p in sorted(glob.glob(args.m1_glob))}
    m2_files = {seed_of(p): p for p in sorted(glob.glob(args.m2_glob))}
    seeds = sorted(set(m1_files) & set(m2_files) & set(met["per_seed"]))
    if not seeds:
        raise SystemExit("M1·M2 예측과 metrics JSON 의 시드가 맞지 않습니다.")

    rng = np.random.default_rng(BOOT_SEED)
    per_seed, restores, costs = {}, [], []
    for s in seeds:
        m1, m2 = load(m1_files[s]), load(m2_files[s])
        common = sorted(set(m0) & set(m1) & set(m2))
        ans = [q for q in common if ua_kind(q) is None]
        easy = [q for q in common if ua_kind(q) == "easy"]
        hard = [q for q in common if ua_kind(q) == "hard"]

        t1 = mean_tokens([m1[q] for q in ans])
        t2 = mean_tokens([m2[q] for q in ans])
        rtok = t2 / t1
        tmpl, tmpl_n = template_adherence([m2[q] for q in ans])

        ps = met["per_seed"][s]
        restore, cost = judge(ps["ci"], ps["judge"], rtok)
        restores.append(restore)
        costs.append(cost)

        per_seed[s] = {
            "n": {"answerable": len(ans), "easy": len(easy), "hard": len(hard)},
            "rtok_answerable": rtok, "tokens_M1": t1, "tokens_M2": t2,
            "false_abstention_M1": float(np.mean([abst(m1[q]) for q in ans])),
            "false_abstention_M2": float(np.mean([abst(m2[q]) for q in ans])),
            "template_adherence_M2": tmpl, "template_n": tmpl_n,
            "vs_M0_easy": paired_diff(m0, m2, easy, rng),
            "vs_M0_hard": paired_diff(m0, m2, hard, rng),
            "from_metrics": {k: ps["point"][k] for k in
                             ("hall_easy_M1", "hall_easy_M2", "hall_hard_M1", "hall_hard_M2",
                              "em_M1", "em_M2", "d_em", "auroc_h3", "auroc_h4")},
            "d_em_ci": [ps["ci"]["d_em"]["lo"], ps["ci"]["d_em"]["hi"]],
            "H": ps["judge"],
            "verdict_restore": restore, "verdict_cost": cost,
        }

        e, h = per_seed[s]["vs_M0_easy"], per_seed[s]["vs_M0_hard"]
        p = ps["point"]
        print(f"\n== seed {s}  (응답가능 {len(ans)} · easy {len(easy)} · hard {len(hard)})")
        print(f"  easy UA 무응답  M0 {e['M0']:.3f} · M1 {1 - p['hall_easy_M1']:.3f} · M2 {e['M2']:.3f}"
              f"  (M2−M0 {e['diff']:+.3f} [{e['diff_ci'][0]:+.3f}, {e['diff_ci'][1]:+.3f}], M2/M0 {e['ratio']:.2f})"
              f"  H1_easy {'충족' if ps['judge']['H1_easy'] else '미충족'} → {restore}")
        print(f"  hard UA 무응답  M0 {h['M0']:.3f} · M1 {1 - p['hall_hard_M1']:.3f} · M2 {h['M2']:.3f}"
              f"  (M2−M0 {h['diff']:+.3f} [{h['diff_ci'][0]:+.3f}, {h['diff_ci'][1]:+.3f}], M2/M0 {h['ratio']:.2f})  [보조]")
        print(f"  ΔEM {p['d_em']:+.3f} [{ps['ci']['d_em']['lo']:+.3f}, {ps['ci']['d_em']['hi']:+.3f}]"
              f" (M1 {p['em_M1']:.3f} → M2 {p['em_M2']:.3f}) · R_tok ×{rtok:.2f} ({t1:.1f}→{t2:.1f})"
              f" · 허위 무응답 {per_seed[s]['false_abstention_M1']:.3f}→{per_seed[s]['false_abstention_M2']:.3f}"
              f" → {cost}")
        print(f"  정답 틀 적합률(M2) {tmpl:.3f} (n={tmpl_n}) · H3 {p['auroc_h3']:.3f} · H4 {p['auroc_h4']:.3f}")

    agg_r, kr = aggregate(restores, "복원 성립")
    agg_c, kc = aggregate(costs, "비용 없음")
    agg_cost_yes, ky = aggregate(costs, "비용 있음")
    print(f"\n== 시드 집계: 복원 성립 {kr}/{len(seeds)} ({agg_r}) · 비용 없음 {kc}/{len(seeds)} ({agg_c})"
          f" · 비용 있음 {ky}/{len(seeds)}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"per_seed": per_seed,
                   "summary": {"restore": {"pass": kr, "verdict": agg_r},
                               "no_cost": {"pass": kc, "verdict": agg_c},
                               "cost": {"pass": ky, "verdict": agg_cost_yes},
                               "n_seeds": len(seeds)},
                   "thresholds": {"rtok_band": RTOK_BAND, "cost_margin": COST_MARGIN, "n_boot": N_BOOT},
                   "prereg": "prereg_klue_m2_r05_outproj_261010.md"},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
