# -*- coding: utf-8 -*-
"""
fill_gaps.py — 보고서 표의 "—" 칸을 기존 예측 파일만으로 채우는 사후 분석 (GPU 불필요).

evaluate_followup.py의 헬퍼(is_correct, auroc, ua_kind, is_abstain, load_exclusions)를
그대로 재사용하며 판정 코드는 손대지 않는다. 산출 지표는 모두 보고용 기술 통계이며
H1~H5 판정에는 영향이 없다.

산출 항목 (해당 파일이 있을 때만):
  [M0]     hard UA 환각 시 confidence, 정답 vs hard UA 환각 AUROC, 응답 시 평균 confidence
  [M1]     easy UA AUROC / hard UA AUROC (+95% CI), 환각 시 confidence(easy/hard),
           응답가능 평균 답변 토큰
  [H4]     m1_conf 신호의 easy / hard 분해 AUROC
  [entropy] M2 자체 entropy_full 신호로 무응답 예측 AUROC ("확신에 찬 무응답" 확인)
  [natua]  검색 불일치형 UA 107건: M0 무응답률, M1 환각률·AUROC·환각 conf, M2 무응답률
  [epoch]  M2-r05 ep1 vs ep2: hard/easy UA 무응답률, EM, 오무응답, 답변 토큰, hard AUROC

사용: python src/analysis/fill_gaps.py --config config_llama.yaml [--n-boot 10000]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import glob
import json
import numpy as np

from evaluate_followup import (load_config, load_preds, load_exclusions, ua_kind,
                               is_abstain, is_correct, auroc)


def ci_boot(fn, rng, n_boot):
    """fn(rng) → 한 리샘플 통계. 백분위 95% CI."""
    vals = np.array([fn(rng) for _ in range(n_boot)], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def auroc_ci(pos, neg, rng, n_boot):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, (np.nan, np.nan)
    pt = auroc(list(pos) + list(neg), [1] * len(pos) + [0] * len(neg))

    def one(r):
        p = pos[r.integers(0, len(pos), len(pos))]
        n = neg[r.integers(0, len(neg), len(neg))]
        return auroc(list(p) + list(n), [1] * len(p) + [0] * len(n))
    return float(pt), ci_boot(one, rng, n_boot)


def rows_from(preds, tol, excluded):
    rows = []
    for qid, r in preds.items():
        if qid in excluded:
            continue
        ga = bool(r.get("gold_answerable", True))
        row = {"qid": qid, "type": r["type"], "ga": ga,
               "ua": ua_kind(qid) if not ga else None,
               "conf": float(r["confidence"]), "abstain": is_abstain(r),
               "entropy_full": r.get("entropy_full"),
               "n_gen": r.get("n_gen_tokens"), "parse_ok": r.get("parse_ok")}
        if ga:
            row["correct"] = int(is_correct(r["prediction"], r["gold_answer"], r["type"], tol))
        rows.append(row)
    return rows


def m1_signal_stats(rows, rng, n_boot, label):
    ans = [r for r in rows if r["ga"]]
    easy = [r for r in rows if r["ua"] == "easy"]
    hard = [r for r in rows if r["ua"] == "hard"]
    pos = [r["conf"] for r in ans if r["correct"] and not r["abstain"]]
    out = {}
    for kind, sub in (("easy", easy), ("hard", hard)):
        neg = [r["conf"] for r in sub if not r["abstain"]]
        pt, ci = auroc_ci(pos, neg, rng, n_boot)
        out[f"{kind}_auroc"] = pt
        out[f"{kind}_auroc_ci"] = ci
        out[f"{kind}_hall_rate"] = float(np.mean([not r["abstain"] for r in sub])) if sub else np.nan
        out[f"{kind}_hall_conf"] = float(np.mean(neg)) if neg else np.nan
    answered = [r for r in ans if not r["abstain"]]
    out["ans_mean_conf"] = float(np.mean([r["conf"] for r in answered])) if answered else np.nan
    toks = [r["n_gen"] for r in ans if r["n_gen"] is not None]
    out["ans_mean_tokens"] = float(np.mean(toks)) if toks else np.nan
    out["false_abstain"] = float(np.mean([r["abstain"] for r in ans])) if ans else np.nan
    out["em"] = float(np.mean([r["correct"] and not r["abstain"] for r in ans])) if ans else np.nan
    out["parse_fail"] = float(np.mean([r["parse_ok"] is False for r in rows]))
    return out


def fmt(v, d=3):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def fmt_ci(ci):
    return "—" if ci is None or np.isnan(ci[0]) else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    res = Path(cfg["paths"]["results_dir"])
    tol = cfg["eval"]["numeric_tolerance"]
    excluded = load_exclusions(res, ["excluded_gold_v2.json", "excluded_gold_v2_manual.json"])
    rng = np.random.default_rng(cfg.get("seed", 42))
    nb = args.n_boot
    report = {"config": args.config, "results_dir": str(res), "n_boot": nb}

    def p(name):
        f = res / name
        return load_preds(f) if f.exists() else None

    print(f"\n######## {args.config} → {res}  (제외 {len(excluded)}건, bootstrap {nb}회) ########")

    # ---------------- M0
    m0 = p("preds_M0_v2.jsonl")
    if m0:
        r0 = rows_from(m0, tol, excluded)
        s0 = m1_signal_stats(r0, rng, nb, "M0")
        report["M0"] = s0
        print("\n[M0] 정답 vs hard UA 환각 AUROC:", fmt(s0["hard_auroc"]), fmt_ci(s0["hard_auroc_ci"]))
        print("     hard UA 환각 시 confidence:", fmt(s0["hard_hall_conf"], 2),
              "| 응답 시 평균 confidence:", fmt(s0["ans_mean_conf"], 2))
        print("     easy/hard UA 무응답률:", fmt(1 - s0["easy_hall_rate"]), "/", fmt(1 - s0["hard_hall_rate"]),
              "| Answerable 오무응답률:", fmt(s0["false_abstain"]),
              "| 평균 응답 토큰:", fmt(s0["ans_mean_tokens"], 1), "| 파싱실패:", fmt(s0["parse_fail"]))

    # ---------------- M1 (시드별)
    m1_files = sorted(glob.glob(str(res / "preds_M1_v2_s*.jsonl")))
    m1_rows = {}
    if m1_files:
        report["M1"] = {}
        print("\n[M1] seed | easy AUROC [CI] | hard AUROC [CI] | 환각 conf easy/hard | 답변 토큰 | EM")
        for f in m1_files:
            seed = Path(f).stem.split("_s")[-1]
            rows = rows_from(load_preds(f), tol, excluded)
            m1_rows[seed] = rows
            s = m1_signal_stats(rows, rng, nb, "M1")
            report["M1"][seed] = s
            print(f"     s{seed} | {fmt(s['easy_auroc'])} {fmt_ci(s['easy_auroc_ci'])} | "
                  f"{fmt(s['hard_auroc'])} {fmt_ci(s['hard_auroc_ci'])} | "
                  f"{fmt(s['easy_hall_conf'],2)} / {fmt(s['hard_hall_conf'],2)} | "
                  f"{fmt(s['ans_mean_tokens'],1)} | {fmt(s['em'])}")
        # 3-seed pooling AUROC
        for kind in ("easy", "hard"):
            pos, neg = [], []
            for rows in m1_rows.values():
                pos += [r["conf"] for r in rows if r["ga"] and r["correct"] and not r["abstain"]]
                neg += [r["conf"] for r in rows if r["ua"] == kind and not r["abstain"]]
            pt, ci = auroc_ci(pos, neg, rng, nb)
            report["M1"][f"pool_{kind}_auroc"] = pt
            report["M1"][f"pool_{kind}_auroc_ci"] = ci
            print(f"     pooling {kind} AUROC: {fmt(pt)} {fmt_ci(ci)}")
    else:
        print("\n[M1] preds_M1_v2_s*.jsonl 없음 → M1 의존 지표(easy AUROC, H4 분해) 산출 불가. "
              "Modal 볼륨에서 회수 필요.")

    # ---------------- M2 r05: H4 easy/hard 분해 (m1_conf) + 자체 entropy AUROC
    m2_files = sorted(glob.glob(str(res / "preds_M2_r05_s*.jsonl")))
    m2_files = [f for f in m2_files if "_ep1" not in f and "natua" not in f]
    if m2_files:
        report["M2"] = {}
        print("\n[M2-r05] seed | H4 easy / hard 분해 (m1_conf) | M2 자체 entropy AUROC (UA 전체) | hard UA 무응답률 | 오무응답")
        for f in m2_files:
            seed = Path(f).stem.split("_s")[-1]
            m2 = load_preds(f)
            rows = rows_from(m2, tol, excluded)
            byq = {r["qid"]: r for r in rows}
            d = {}
            # 자체 entropy 신호
            ua = [r for r in rows if r["ua"] and r["entropy_full"] is not None]
            sc = [r["entropy_full"] for r in ua]; lb = [int(r["abstain"]) for r in ua]
            d["entropy_auroc"] = auroc(sc, lb) if ua else np.nan
            # H4 분해 (M1 conf 필요)
            if seed in m1_rows:
                m1q = {r["qid"]: r for r in m1_rows[seed]}
                for kind in ("easy", "hard"):
                    sub = [r for r in rows if r["ua"] == kind and r["qid"] in m1q]
                    pos = [1 - m1q[r["qid"]]["conf"] for r in sub if r["abstain"]]
                    neg = [1 - m1q[r["qid"]]["conf"] for r in sub if not r["abstain"]]
                    pt, ci = auroc_ci(pos, neg, rng, nb)
                    d[f"h4_{kind}"] = pt; d[f"h4_{kind}_ci"] = ci
            hard = [r for r in rows if r["ua"] == "hard"]
            ans = [r for r in rows if r["ga"]]
            d["hard_abstain"] = float(np.mean([r["abstain"] for r in hard]))
            d["false_abstain"] = float(np.mean([r["abstain"] for r in ans]))
            report["M2"][seed] = d
            print(f"     s{seed} | {fmt(d.get('h4_easy'))} {fmt_ci(d.get('h4_easy_ci'))} / "
                  f"{fmt(d.get('h4_hard'))} {fmt_ci(d.get('h4_hard_ci'))} | "
                  f"{fmt(d['entropy_auroc'])} | {fmt(d['hard_abstain'])} | {fmt(d['false_abstain'])}")

    # ---------------- natua (검색 불일치형 UA)
    n0 = p("preds_M0_natua.jsonl")
    if n0:
        report["natua"] = {}
        print("\n[natua n=%d]" % len(n0))
        r0 = [{"conf": float(r["confidence"]), "abstain": is_abstain(r)} for r in n0.values()]
        report["natua"]["M0_abstain"] = float(np.mean([r["abstain"] for r in r0]))
        print("     M0 자발적 무응답률:", fmt(report["natua"]["M0_abstain"]))
        for f in sorted(glob.glob(str(res / "preds_M1_natua_s*.jsonl"))):
            seed = Path(f).stem.split("_s")[-1]
            n1 = load_preds(f)
            hall = [r for r in n1.values() if not is_abstain(r)]
            hall_rate = len(hall) / len(n1)
            hall_conf = float(np.mean([float(r["confidence"]) for r in hall])) if hall else np.nan
            # AUROC: 같은 시드 M1 응답가능 정답 conf vs natua 환각 conf
            if seed in m1_rows:
                pos = [r["conf"] for r in m1_rows[seed] if r["ga"] and r["correct"] and not r["abstain"]]
                neg = [float(r["confidence"]) for r in hall]
                pt, ci = auroc_ci(pos, neg, rng, nb)
            else:
                pt, ci = np.nan, (np.nan, np.nan)
            report["natua"][f"M1_s{seed}"] = {"hall_rate": hall_rate, "hall_conf": hall_conf,
                                              "auroc": pt, "auroc_ci": ci}
            print(f"     M1 s{seed}: 환각률 {fmt(hall_rate)} | 분리 AUROC {fmt(pt)} {fmt_ci(ci)} | 환각 시 conf {fmt(hall_conf,2)}")
        for f in sorted(glob.glob(str(res / "preds_M2_r05_natua_s*.jsonl"))):
            seed = Path(f).stem.split("_s")[-1]
            n2 = load_preds(f)
            ab = float(np.mean([is_abstain(r) for r in n2.values()]))
            report["natua"][f"M2_s{seed}_abstain"] = ab
            print(f"     M2-r05 s{seed} 무응답률: {fmt(ab)}")

    # ---------------- epoch (M2-r05 ep1 vs ep2)
    ep1_files = sorted(glob.glob(str(res / "preds_M2_r05_ep1_s*.jsonl")))
    if ep1_files:
        report["epoch"] = {}
        print("\n[epoch M2-r05] seed | hard UA 무응답률 ep1→ep2 | easy | EM | 오무응답 | 답변토큰 | hard AUROC(M1 conf 기준은 M1 고정이므로 생략)")
        for f in ep1_files:
            seed = Path(f).stem.split("_s")[-1]
            f2 = res / f"preds_M2_r05_s{seed}.jsonl"
            if not f2.exists():
                continue
            a = rows_from(load_preds(f), tol, excluded)
            b = rows_from(load_preds(f2), tol, excluded)

            def stats(rows):
                hard = [r for r in rows if r["ua"] == "hard"]; easy = [r for r in rows if r["ua"] == "easy"]
                ans = [r for r in rows if r["ga"]]
                toks = [r["n_gen"] for r in ans if r["n_gen"] is not None]
                return dict(hard=float(np.mean([r["abstain"] for r in hard])),
                            easy=float(np.mean([r["abstain"] for r in easy])),
                            em=float(np.mean([r["correct"] and not r["abstain"] for r in ans])),
                            fa=float(np.mean([r["abstain"] for r in ans])),
                            tok=float(np.mean(toks)) if toks else np.nan)
            sa, sb = stats(a), stats(b)
            report["epoch"][seed] = {"ep1": sa, "ep2": sb}
            print(f"     s{seed} | {fmt(sa['hard'])}→{fmt(sb['hard'])} (Δ{sa['hard']-sb['hard']:+.3f}) | "
                  f"{fmt(sa['easy'])}→{fmt(sb['easy'])} | {fmt(sa['em'])}→{fmt(sb['em'])} | "
                  f"{fmt(sa['fa'])}→{fmt(sb['fa'])} | {fmt(sa['tok'],1)}→{fmt(sb['tok'],1)}")

    out = Path(args.out) if args.out else res / "gap_fill.json"
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"\n→ 저장: {out}")


if __name__ == "__main__":
    main()
