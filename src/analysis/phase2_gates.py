# -*- coding: utf-8 -*-
"""
phase2_gates.py — Phase 2 사전등록 §7 실행 관문을 예측 파일에서 판정한다 (로컬 CPU).

관문 1 (M0): easy UA 무응답률 ≥ 90% 이어야 M1 학습으로 진행.
관문 2 (M1): 응답가능 EM 3 seed 평균 ≥ 0.55 이어야 ΔEM 비교가 유효.
             (미달이어도 M2 는 진행한다 — 토큰 배율·false abstention 판별은 유효하므로.)

사용 (Modal 볼륨에서 예측 파일을 회수한 뒤):
  python src/analysis/phase2_gates.py --config config_klue_exaone.yaml --gate 1
  python src/analysis/phase2_gates.py --config config_klue_exaone.yaml --gate 2
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

from evaluation.evaluate_followup import is_correct, load_config, ua_kind


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def gate1(res_dir):
    p = res_dir / "preds_M0_v2.jsonl"
    rows = load(p)
    easy = [r for r in rows if ua_kind(r["question_id"]) == "easy"]
    hard = [r for r in rows if ua_kind(r["question_id"]) == "hard"]
    ans = [r for r in rows if r.get("gold_answerable", True)]
    abst = lambda xs: sum(r.get("answerable_pred") is False for r in xs) / len(xs) if xs else float("nan")
    parse_fail = sum(not r.get("parse_ok", True) for r in rows) / len(rows)
    print(f"[관문 1] {p}  (n={len(rows)})")
    print(f"  easy UA 무응답률  {abst(easy):.1%}  (기준 ≥ 90%)")
    print(f"  hard UA 무응답률  {abst(hard):.1%}  (참고)")
    print(f"  응답가능 false abstention  {abst(ans):.1%}  (참고)")
    print(f"  파싱 실패율  {parse_fail:.1%}  (참고 — 부록 C 기준 10% 초과 시 보고)")
    ok = abst(easy) >= 0.90
    print(f"  → {'통과: M1 학습 진행' if ok else '미달: Stage F 중단, 사실만 보고'}")
    return ok


def gate2(res_dir, tol, seeds):
    ems = []
    print(f"[관문 2] {res_dir}/preds_M1_v2_s*.jsonl")
    for s in seeds:
        p = res_dir / f"preds_M1_v2_s{s}.jsonl"
        if not p.exists():
            print(f"  seed {s}: 파일 없음 — 건너뜀"); continue
        rows = [r for r in load(p) if r.get("gold_answerable", True)]
        em = sum(bool(is_correct(r["prediction"], r["gold_answer"], r["type"], tol))
                 and r.get("answerable_pred") is not False for r in rows) / len(rows)
        ems.append(em)
        print(f"  seed {s}: 응답가능 EM {em:.4f}  (n={len(rows)})")
    if not ems:
        raise SystemExit("M1 예측 파일이 없습니다.")
    mean = sum(ems) / len(ems)
    ok = mean >= 0.55
    print(f"  3 seed 평균 EM {mean:.4f}  (기준 ≥ 0.55; 금융 text_span M1 = 0.73)")
    print(f"  → {'통과: ΔEM 비교 유효' if ok else '미달: ΔEM 항목은 inconclusive 로 보고. 토큰 배율·false abstention 판별은 계속 유효 — M2 진행'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gate", type=int, choices=[1, 2], required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res_dir = Path(cfg["paths"]["results_dir"])
    if args.gate == 1:
        gate1(res_dir)
    else:
        gate2(res_dir, cfg["eval"]["numeric_tolerance"], cfg["train"]["seeds"])


if __name__ == "__main__":
    main()
