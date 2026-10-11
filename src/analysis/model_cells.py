# -*- coding: utf-8 -*-
"""
model_cells.py — 4장 표 4-1·4-3·4-5·4-6·4-9의 모델별 칸을 예측 파일(M0, M1, M2-r05)에서 직접 산출.
GPU·판정 코드와 무관한 기술 통계이며 H1~H5 판정에는 영향이 없다.
evaluate_followup.py의 헬퍼(is_correct, auroc, ua_kind, is_abstain, load_exclusions)를 그대로 쓴다.

사용 (Kanana):
  python src/analysis/model_cells.py --config config_kanana.yaml
사용 (EXAONE out_proj, 파일명이 _pin):
  python src/analysis/model_cells.py --config config_abl_exaone_outproj.yaml \
      --m1-glob "preds_M1_pin_s*.jsonl" --m2-glob "preds_M2_r05_pin_s*.jsonl"
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse, re
import numpy as np
from evaluate_followup import (load_config, load_preds, load_exclusions, ua_kind,
                               is_abstain, is_correct, auroc)


def rows_of(preds, tol, excluded):
    out = []
    for qid, r in preds.items():
        if qid in excluded:
            continue
        ga = bool(r.get("gold_answerable", True))
        row = dict(qid=qid, type=r["type"], ga=ga, ua=ua_kind(qid) if not ga else None,
                   conf=float(r["confidence"]), minp=r.get("min_token_prob"),
                   ab=is_abstain(r), n=r.get("n_gen_tokens"),
                   fb=(r.get("conf_scope") != "answer_span"))
        if ga:
            row["ok"] = bool(is_correct(r["prediction"], r["gold_answer"], r["type"], tol)) and not row["ab"]
        out.append(row)
    return out


def pct(x):
    return f"{100 * x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--m0", default="preds_M0_v2.jsonl")
    ap.add_argument("--m1-glob", default="preds_M1_v2_s*.jsonl")
    ap.add_argument("--m2-glob", default="preds_M2_r05_s*.jsonl")
    a = ap.parse_args()
    cfg = load_config(a.config)
    res = Path(cfg["paths"]["results_dir"]); tol = cfg["eval"]["numeric_tolerance"]
    exc = load_exclusions(res, ["excluded_gold_v2.json", "excluded_gold_v2_manual.json"])
    sid = lambda p: re.search(r"_s(\d+)\.jsonl$", p.name).group(1)
    m0 = rows_of(load_preds(res / a.m0), tol, exc)
    m1 = {sid(p): rows_of(load_preds(p), tol, exc) for p in sorted(res.glob(a.m1_glob))}
    m2 = {sid(p): rows_of(load_preds(p), tol, exc) for p in sorted(res.glob(a.m2_glob))}
    seeds = sorted(set(m1) & set(m2))
    print(f"##### {a.config} | seeds {seeds} | n(M0)={len(m0)}")

    fb = lambda rs: pct(np.mean([r["fb"] for r in rs]))
    fb_ans = lambda rs: pct(np.mean([r["fb"] for r in rs if not r["ab"]]))
    f1 = [float(fb(m1[s])) for s in seeds]
    f2 = [float(fb(m2[s])) for s in seeds]; f2a = [float(fb_ans(m2[s])) for s in seeds]
    print("\n[표 4-1] answer span 폴백 비율(%)")
    print(f"  M0 (전체 / 응답 문항): {fb(m0)} / {fb_ans(m0)}")
    print(f"  M1 (전체, seed 범위): {min(f1):.1f}~{max(f1):.1f}")
    print(f"  M2-r05 (전체 / 응답 문항): {min(f2):.1f}~{max(f2):.1f} / {min(f2a):.1f}~{max(f2a):.1f}")

    ynem = lambda rs: np.mean([r["ok"] for r in rs if r["ga"] and r["type"] == "yes_no"])
    tok = lambda rs: np.mean([r["n"] for r in rs if r["ga"] and r["n"] is not None])
    em = lambda rs: np.mean([r["ok"] for r in rs if r["ga"]])
    print("\n[표 4-3] M0 → M1 (3 seed 평균)")
    print(f"  Answerable EM: {pct(em(m0))} → {pct(np.mean([em(m1[s]) for s in seeds]))}")
    print(f"  yes/no EM: {pct(ynem(m0))} → {pct(np.mean([ynem(m1[s]) for s in seeds]))}")
    print(f"  평균 응답 토큰(Answerable): {tok(m0):.1f} → {np.mean([tok(m1[s]) for s in seeds]):.1f}")

    def conf_of(rs, f):
        v = [r["conf"] for r in rs if f(r)]
        return np.mean(v) if v else np.nan
    cor = lambda r: r["ga"] and r["ok"]
    eh = lambda r: r["ua"] == "easy" and not r["ab"]
    hh = lambda r: r["ua"] == "hard" and not r["ab"]
    print("\n[표 4-5] M1 평균 신뢰도 (정답 / easy 환각 / hard 환각), 3 seed 평균")
    print("  " + " / ".join(f"{np.mean([conf_of(m1[s], f) for s in seeds]):.2f}" for f in (cor, eh, hh)))

    print("\n[표 4-6] M1, seed별")
    wrong = lambda r: r["ga"] and not r["ab"] and not r["ok"]
    for s in seeds:
        rs = m1[s]
        w = [r["conf"] for r in rs if wrong(r)]; h = [r["conf"] for r in rs if hh(r)]
        c = [r["minp"] for r in rs if cor(r) and r["minp"] is not None]
        hm = [r["minp"] for r in rs if hh(r) and r["minp"] is not None]
        a_wh = auroc(w + h, [1] * len(w) + [0] * len(h))
        a_min = auroc(c + hm, [1] * len(c) + [0] * len(hm))
        print(f"  s{s} | Answerable 오답 vs hard UA 환각 AUROC {a_wh:.3f} (반대 방향 {1 - a_wh:.3f}) | 최소 토큰 확률 분리 AUROC {a_min:.3f}")

    print("\n[표 4-9] M2-r05 vs M1")
    types = sorted({r["type"] for r in m1[seeds[0]] if r["ga"]})
    dts, rat = [], []
    for s in seeds:
        d = {t: 100 * (np.mean([r["ok"] for r in m2[s] if r["ga"] and r["type"] == t])
                       - np.mean([r["ok"] for r in m1[s] if r["ga"] and r["type"] == t])) for t in types}
        dts += list(d.values()); rat.append(tok(m2[s]) / tok(m1[s]))
        print(f"  s{s} | 유형별 ΔEM(%p) " + ", ".join(f"{t} {v:+.1f}" for t, v in d.items()) + f" | 응답 길이 배율 ×{rat[-1]:.2f}")
    print(f"  유형별 ΔEM 범위: {min(dts):+.1f} ~ {max(dts):+.1f} | 응답 길이 배율 평균 ×{np.mean(rat):.2f} ({min(rat):.2f}~{max(rat):.2f})")


if __name__ == "__main__":
    main()