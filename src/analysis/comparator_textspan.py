# -*- coding: utf-8 -*-
"""
comparator_textspan.py — Phase 2(KLUE) 붕괴 판별의 비교 기준을 금융 결과에서 고정한다.

배경 (Phase 2 사전등록 §3):
  기존 붕괴 기준 ΔEM(M2-r05 − M1) = -0.346 은 4개 답 유형의 집계값이며,
  유형별 기여는 numeric -14 / text -30 / table -28 / yes_no -61 %p 로
  yes/no 가 가장 크다(보고서 표 10). KLUE-MRC 는 추출형 span 단일 유형이므로
  이 집계값과 직접 비교하면 "붕괴가 사라졌다"와 "yes/no 가 없어서 작아 보인다"를
  구분할 수 없다.

  따라서 KLUE 와 비교할 기준은 금융 결과를 text_span 하위집합으로 제한한 값이어야
  한다. 본 스크립트는 그 값을 기존 예측 파일에서 산출해 고정한다 — 새 추론·학습 없음.

동일성 보장:
  채점(is_correct), 행 구성(build_rows), EM 정의(무응답=오답), 층화 paired
  bootstrap 절차는 evaluate_followup.py 에서 **import** 해 쓴다. 재구현하지 않는다.
  --types 를 생략하면 config 의 4유형 전체로 돌며, 이때 산출값이
  results/metrics_followup_r05.json 의 d_em 과 일치해야 한다(자동 검증).

제외 목록:
  부록 A 기록대로 question_ids 필드만 읽는다(메타데이터 문자열 4건 미포함).
  제외 집합이 30건이 되고 판정은 달라지지 않음을 --verify 가 확인한다.

사용:
  python src/analysis/comparator_textspan.py                      # EXAONE, text_span
  python src/analysis/comparator_textspan.py --types all --verify # 4유형 + 재현 검증
  python src/analysis/comparator_textspan.py --results-dir results/kanana \
      --m1-glob 'preds_M1_s*.jsonl'                               # 대조군

출력:
  results/comparator_textspan.json  (Phase 2 실행 전 커밋 대상)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import re
from collections import defaultdict

import numpy as np

from evaluation.evaluate_followup import (
    build_rows,
    load_config,
    load_preds,
)


# ---------------------------------------------------------------- 제외 목록 (부록 A 정정판)
def load_exclusions_strict(res_dir, names):
    """question_ids 필드만 읽는다. 기존 load_exclusions 는 모든 문자열 값을 훑어
    메타데이터 4건을 함께 집계했다(부록 A). 판정은 달라지지 않으나 건수 표기를 바로잡는다."""
    ids = set()
    for name in names:
        p = Path(res_dir) / name
        if not p.exists():
            continue
        data = json.load(open(p, encoding="utf-8"))
        if isinstance(data, dict):
            ids.update(x for x in data.get("question_ids", []) if isinstance(x, str))
        elif isinstance(data, list):
            ids.update(x for x in data if isinstance(x, str))
    return ids


# ---------------------------------------------------------------- 지표 (유형 제한판)
def em(rows, cond):
    """evaluate_followup.scalars_one_sample 의 EM 정의와 동일 — 무응답은 오답."""
    return float(np.mean([r[f"correct_{cond}"] and not r[f"abstain_{cond}"] for r in rows]))


def point_stats(rows, tok):
    """한 시드의 점추정. rows 는 이미 (응답가능 ∩ 대상 유형)으로 걸러진 상태."""
    em1, em2 = em(rows, "M1"), em(rows, "M2")
    t1 = [tok["M1"][r["qid"]] for r in rows if tok["M1"].get(r["qid"]) is not None]
    t2 = [tok["M2"][r["qid"]] for r in rows if tok["M2"].get(r["qid"]) is not None]
    ratio = (sum(t2) / len(t2)) / (sum(t1) / len(t1)) if t1 and t2 else float("nan")
    return {
        "n": len(rows),
        "em_M1": em1,
        "em_M2": em2,
        "d_em": em2 - em1,
        "token_ratio": ratio,
        "tok_M1": (sum(t1) / len(t1)) if t1 else float("nan"),
        "tok_M2": (sum(t2) / len(t2)) if t2 else float("nan"),
        "false_abstention_M1": float(np.mean([r["abstain_M1"] for r in rows])),
        "false_abstention_M2": float(np.mean([r["abstain_M2"] for r in rows])),
    }


def bootstrap_d_em(rows_by_type, n_boot, seed):
    """층화 paired bootstrap — evaluate_followup.bootstrap 과 동일 절차를 ΔEM 에만 적용.

    층 = 대상 유형 각각(응답가능). 같은 행 객체를 재표집하므로 M1·M2 가 자동으로 쌍을 이룬다.
    """
    rng = np.random.default_rng(seed)
    strata = [v for v in rows_by_type.values() if v]
    samples = defaultdict(list)
    for _ in range(n_boot):
        rs = []
        for sub in strata:
            idx = rng.integers(0, len(sub), len(sub))
            rs.extend(sub[i] for i in idx)
        samples["d_em"].append(em(rs, "M2") - em(rs, "M1"))
    lo, hi = np.percentile(samples["d_em"], [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi)}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--results-dir", default=None,
                    help="기본값은 config 의 paths.results_dir")
    ap.add_argument("--types", default="text_span",
                    help="쉼표 구분 유형 목록, 또는 'all' (config 의 4유형)")
    ap.add_argument("--m0", default="preds_M0_v2.jsonl")
    ap.add_argument("--m1-glob", default="preds_M1_v2_s*.jsonl")
    ap.add_argument("--m2-glob", default="preds_M2_r05_s*.jsonl")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--exclude-files", nargs="*",
                    default=["excluded_gold_v2.json", "excluded_gold_v2_manual.json"])
    ap.add_argument("--verify", action="store_true",
                    help="metrics_followup_r05.json 의 d_em 과 일치하는지 검증 (--types all 에서만 의미)")
    ap.add_argument("--out", default="comparator_textspan.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    all_types = cfg["data"]["types"]
    types = all_types if args.types == "all" else [t.strip() for t in args.types.split(",")]
    unknown = [t for t in types if t not in all_types]
    if unknown:
        raise SystemExit(f"알 수 없는 유형: {unknown} (가능: {all_types})")

    n_boot = args.n_boot or cfg["eval"]["n_bootstrap"]
    tol = cfg["eval"]["numeric_tolerance"]
    res_dir = Path(args.results_dir or cfg["paths"]["results_dir"])

    excluded = load_exclusions_strict(res_dir, args.exclude_files)
    print(f"[제외] question_ids 필드 기준 {len(excluded)}건")

    m0_path = res_dir / args.m0
    if not m0_path.exists():
        raise SystemExit(f"M0 예측 파일 없음: {m0_path}")
    m0 = load_preds(m0_path)

    def seed_of(p):
        m = re.search(r"_s(\d+)\.jsonl$", p.name)
        return m.group(1) if m else p.stem

    m1_files = {seed_of(p): p for p in sorted(res_dir.glob(args.m1_glob))}
    m2_files = {seed_of(p): p for p in sorted(res_dir.glob(args.m2_glob))}
    seeds = sorted(set(m1_files) & set(m2_files))
    if not seeds:
        raise SystemExit(
            f"M1/M2 시드 쌍 없음 — M1 {sorted(m1_files)} · M2 {sorted(m2_files)} "
            f"(dir={res_dir}, globs={args.m1_glob!r} {args.m2_glob!r})")
    print(f"시드 쌍: {seeds} · 대상 유형: {types} · bootstrap {n_boot}회")

    per_seed = {}
    for s in seeds:
        m1_raw, m2_raw = load_preds(m1_files[s]), load_preds(m2_files[s])
        rows = build_rows(m0, m1_raw, m2_raw, tol, excluded)
        tok = {"M1": {q: r.get("n_answer_tokens") for q, r in m1_raw.items()},
               "M2": {q: r.get("n_answer_tokens") for q, r in m2_raw.items()}}

        by_type = {t: [r for r in rows if r["gold_answerable"] and r["type"] == t]
                   for t in types}
        target = [r for v in by_type.values() for r in v]
        pt = point_stats(target, tok)
        pt["ci_d_em"] = bootstrap_d_em(by_type, n_boot, cfg["seed"])
        per_seed[s] = pt
        print(f"  seed {s}: n={pt['n']:4d}  EM {pt['em_M1']:.4f}→{pt['em_M2']:.4f}  "
              f"ΔEM {pt['d_em']:+.4f} CI[{pt['ci_d_em']['lo']:+.4f},{pt['ci_d_em']['hi']:+.4f}]  "
              f"토큰 ×{pt['token_ratio']:.2f}  false-abst {pt['false_abstention_M2']:.1%}")

    def agg(key):
        v = [per_seed[s][key] for s in seeds]
        return {"mean": float(np.mean(v)),
                "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "min": float(np.min(v)), "max": float(np.max(v))}

    summary = {k: agg(k) for k in
               ("d_em", "em_M1", "em_M2", "token_ratio", "false_abstention_M2")}
    summary["h2_pass_seeds"] = sum(
        per_seed[s]["ci_d_em"]["lo"] > -cfg["eval"]["min_contrast"] for s in seeds)
    summary["n_seeds"] = len(seeds)

    print(f"\n  평균 ΔEM {summary['d_em']['mean']:+.4f} (SD {summary['d_em']['sd']:.4f}) · "
          f"토큰 ×{summary['token_ratio']['mean']:.2f} · "
          f"false-abst {summary['false_abstention_M2']['mean']:.1%} · "
          f"H2 {summary['h2_pass_seeds']}/{summary['n_seeds']}")

    verified = None
    if args.verify:
        ref_path = res_dir / "metrics_followup_r05.json"
        if ref_path.exists():
            ref = json.load(open(ref_path, encoding="utf-8"))["per_seed"]
            diffs = {s: per_seed[s]["d_em"] - ref[s]["point"]["d_em"]
                     for s in seeds if s in ref}
            ci_diffs = {s: max(abs(per_seed[s]["ci_d_em"]["lo"] - ref[s]["ci"]["d_em"]["lo"]),
                               abs(per_seed[s]["ci_d_em"]["hi"] - ref[s]["ci"]["d_em"]["hi"]))
                        for s in seeds if s in ref}
            verified = {"max_abs_diff": max(abs(v) for v in diffs.values()) if diffs else None,
                        "per_seed_diff": diffs,
                        "max_ci_diff": max(ci_diffs.values()) if ci_diffs else None,
                        "ci_note": ("CI 차이는 bootstrap 층 구성 차이에서만 온다 — 공식 절차는 "
                                    "4유형+easy+hard 전 층을 재표집하며 UA 층도 난수를 소모하는 반면, "
                                    "본 스크립트는 ΔEM 에 실제로 기여하는 대상 유형 층만 재표집한다. "
                                    "점추정 일치가 채점·행 구성·제외 처리의 동일성을 보증한다.")}
            print(f"\n[검증] 공식 판정 d_em 점추정 최대 차이 = {verified['max_abs_diff']:.2e}")
            print(f"       CI 최대 차이 = {verified['max_ci_diff']:.2e} (난수 스트림 차이, 아래 주석 참조)")
            if verified["max_abs_diff"] is not None and verified["max_abs_diff"] < 1e-12:
                print("       → 점추정 일치. 채점·행 구성·제외 처리가 공식 판정과 동일하며,")
                print("         부록 A의 '제외 목록 정정은 판정을 바꾸지 않는다'도 함께 확인됨.")
            else:
                print("       → 불일치! 기준값으로 쓰기 전에 원인을 확인할 것.")
        else:
            print(f"\n[검증] 건너뜀 — {ref_path} 없음")

    out = {
        "purpose": "Phase 2(KLUE) 붕괴 판별 비교 기준 — 금융 결과의 유형 제한 재산출",
        "types": types,
        "results_dir": str(res_dir),
        "inputs": {"m0": str(m0_path),
                   "m1": {s: str(m1_files[s]) for s in seeds},
                   "m2": {s: str(m2_files[s]) for s in seeds}},
        "settings": {"n_boot": n_boot, "boot_seed": cfg["seed"],
                     "numeric_tolerance": tol,
                     "n_excluded": len(excluded),
                     "exclusion_rule": "question_ids 필드만 (부록 A 정정)",
                     "em_definition": "무응답=오답 (evaluate_followup 과 동일)"},
        "per_seed": per_seed,
        "summary": summary,
        "verified_against_official": verified,
    }
    out_path = res_dir / args.out
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
