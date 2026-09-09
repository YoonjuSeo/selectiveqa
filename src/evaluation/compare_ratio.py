# -*- coding: utf-8 -*-
"""
compare_ratio.py — 혼입률 격자 판정 (다이얼 / 스위치 / 미결정).

사전 지정 기준 (prereg_grid_r30_260908.md):
  통계량  Δ_s = (기준B hard UA 무응답률) − (기준A hard UA 무응답률)
          동일 시드·동일 hard UA 문항에서 question_id 페어링,
          hard 층 재표집 paired bootstrap.
  마진    δ = 0.10 (기본값, --delta 로 변경 가능)
  판정    다이얼   Δ_s 95% CI 하한 >  +δ
          스위치   Δ_s 95% CI 상한 <  +δ   (동등성 형태)
          미결정   그 외 (CI가 δ를 걸침)
  집계    3시드 모두 같은 판정이면 강건, 2/3이면 조건부, 그 외 미결정.

부수 보고 (판정 없음, 기술 통계):
  easy UA 무응답률 Δ, 응답가능 문항 오무응답률 Δ, 각 조건의 시드 범위.
  ※ EM 손실은 각 혼입률의 metrics_followup_*.json 에서 H2로 이미 판정된다.

사용법 (프로젝트 루트에서):
  python src/evaluation/compare_ratio.py --config config_qwen3.yaml
  python src/evaluation/compare_ratio.py --config config_llama.yaml --tag-a r05 --tag-b r30
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import re

import numpy as np
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_preds(path):
    return {r["question_id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def load_exclusions(res_dir, names):
    """evaluate_followup.load_exclusions 와 동일 규칙."""
    ids = set()
    for name in names:
        p = Path(res_dir) / name
        if not p.exists():
            continue
        data = json.load(open(p, encoding="utf-8"))
        items = data.values() if isinstance(data, dict) else data
        for it in items:
            if isinstance(it, str):
                ids.add(it)
            elif isinstance(it, dict) and "question_id" in it:
                ids.add(it["question_id"])
            elif isinstance(it, list):
                ids.update(x for x in it if isinstance(x, str))
        print(f"[제외] {name}: 누적 {len(ids)}건")
    return ids


def ua_kind(qid):
    if qid.startswith("unans_"):
        return "easy"
    if "-hd-" in qid:
        return "hard"
    return None


def is_abstain(rec):
    return rec.get("answerable_pred") is False


def paired_delta_ci(a_flags, b_flags, n_boot, seed):
    """무응답 여부 0/1 배열 쌍에 대한 Δ(=B−A) 점추정과 백분위 95% CI."""
    a = np.asarray(a_flags, float)
    b = np.asarray(b_flags, float)
    point = float(b.mean() - a.mean())
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))          # 같은 인덱스로 두 조건 동시 재표집
    deltas = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return point, float(lo), float(hi)


def verdict_of(lo, hi, delta):
    if lo > delta:
        return "다이얼"
    if hi < delta:
        return "스위치"
    return "미결정"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tag-a", default="r05", help="기준 혼입률 (낮은 쪽)")
    ap.add_argument("--tag-b", default="r30", help="비교 혼입률 (높은 쪽)")
    ap.add_argument("--delta", type=float, default=0.10,
                    help="사전 지정 마진 (기본 0.10)")
    ap.add_argument("--n-boot", type=int, default=None, help="미지정 시 config 값")
    ap.add_argument("--exclude-files", nargs="*",
                    default=["excluded_gold_v2.json", "excluded_gold_v2_manual.json"])
    ap.add_argument("--out", default=None, help="미지정 시 metrics_ratio_{A}_{B}.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    res_dir = Path(cfg["paths"]["results_dir"])
    n_boot = args.n_boot or cfg["eval"]["n_bootstrap"]
    boot_seed = cfg["seed"]

    excluded = load_exclusions(res_dir, args.exclude_files)

    def seed_of(p):
        m = re.search(r"_s(\d+)\.jsonl$", p.name)
        return m.group(1) if m else p.stem

    files_a = {seed_of(p): p for p in sorted(res_dir.glob(f"preds_M2_{args.tag_a}_s*.jsonl"))}
    files_b = {seed_of(p): p for p in sorted(res_dir.glob(f"preds_M2_{args.tag_b}_s*.jsonl"))}
    seeds = sorted(set(files_a) & set(files_b))
    if not seeds:
        raise SystemExit(f"{args.tag_a}/{args.tag_b} 시드 쌍을 찾지 못했습니다. 경로·태그 확인.")
    print(f"시드 쌍: {seeds} · bootstrap {n_boot}회 · 마진 δ={args.delta:+.2f}")
    print(f"비교: {args.tag_a} → {args.tag_b}  (Δ = {args.tag_b} − {args.tag_a})\n")

    per_seed, verdicts = {}, {}
    rate_a, rate_b = {}, {}          # 시드 범위 기술 통계용
    for s in seeds:
        A, B = load_preds(files_a[s]), load_preds(files_b[s])
        common = sorted((set(A) & set(B)) - excluded)
        strata = {"hard": [], "easy": [], "ans": []}
        for q in common:
            k = ua_kind(q)
            if k == "hard":
                strata["hard"].append(q)
            elif k == "easy":
                strata["easy"].append(q)
            elif bool(A[q].get("gold_answerable", True)):
                strata["ans"].append(q)

        row = {"n": {k: len(v) for k, v in strata.items()}}
        for k, qs in strata.items():
            fa = [is_abstain(A[q]) for q in qs]
            fb = [is_abstain(B[q]) for q in qs]
            pt, lo, hi = paired_delta_ci(fa, fb, n_boot, boot_seed)
            row[k] = {"rate_a": float(np.mean(fa)), "rate_b": float(np.mean(fb)),
                      "delta": pt, "lo": lo, "hi": hi}

        v = verdict_of(row["hard"]["lo"], row["hard"]["hi"], args.delta)
        row["verdict"] = v
        per_seed[s], verdicts[s] = row, v
        rate_a[s], rate_b[s] = row["hard"]["rate_a"], row["hard"]["rate_b"]

        h, e, n = row["hard"], row["easy"], row["ans"]
        print(f"======== 시드 {s} (hard={row['n']['hard']}, easy={row['n']['easy']}, "
              f"응답가능={row['n']['ans']}) ========")
        print(f"hard UA 무응답률: {h['rate_a']:.3f} → {h['rate_b']:.3f}"
              f"  Δ {h['delta']:+.3f} [{h['lo']:+.3f}, {h['hi']:+.3f}] → {v}")
        print(f"  [기술] easy UA 무응답률: {e['rate_a']:.3f} → {e['rate_b']:.3f}"
              f"  Δ {e['delta']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}]")
        print(f"  [기술] 응답가능 오무응답률: {n['rate_a']:.3f} → {n['rate_b']:.3f}"
              f"  Δ {n['delta']:+.3f} [{n['lo']:+.3f}, {n['hi']:+.3f}]\n")

    print("======== 시드 집계 ========")
    counts = {v: sum(1 for s in seeds if verdicts[s] == v) for v in set(verdicts.values())}
    top = max(counts, key=counts.get)
    n_top = counts[top]
    if n_top == len(seeds) >= 3:
        overall = f"{top} (강건)"
    elif len(seeds) >= 3 and n_top >= 2:
        overall = f"{top} (조건부)"
    else:
        overall = "미결정 (시드 간 불일치)"
    print(" · ".join(f"{v} {c}/{len(seeds)}" for v, c in sorted(counts.items())))
    print(f"→ 종합: {overall}")

    span_a = max(rate_a.values()) - min(rate_a.values())
    span_b = max(rate_b.values()) - min(rate_b.values())
    print(f"[기술] hard UA 무응답률 시드 범위: {args.tag_a} {span_a:.3f} · {args.tag_b} {span_b:.3f}")

    out = {"per_seed": per_seed, "verdicts": verdicts, "overall": overall,
           "seed_span": {args.tag_a: span_a, args.tag_b: span_b},
           "settings": {"tag_a": args.tag_a, "tag_b": args.tag_b,
                        "delta": args.delta, "n_boot": n_boot,
                        "boot_seed": boot_seed,
                        "files_a": {s: str(files_a[s]) for s in seeds},
                        "files_b": {s: str(files_b[s]) for s in seeds},
                        "excluded": sorted(excluded),
                        "statistic": "hard UA 무응답률 차이 (B−A), qid 페어링, hard 층 재표집",
                        "prereg": "prereg_grid_r30_260908.md (사전 지정 탐색적 판정)"}}
    out_path = res_dir / (args.out or f"metrics_ratio_{args.tag_a}_{args.tag_b}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()