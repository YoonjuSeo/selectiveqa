# -*- coding: utf-8 -*-
"""
gate_check.py — 재현 관문 판정 (modal_inference_cb.py의 gate_fn과 로컬 재판정이 같은 함수를 쓴다).

원 예측(A)과 같은 고정 환경에서 새로 돌린 예측(B)을 같은 문항끼리 비교한다.
  기준 (2026-11-06 개정, 사전 계획 3절):
    G1 answerable 판단 일치 >= 0.99
    G2 정답 여부(is_correct) 일치 >= 0.97   # 무응답은 오답으로 본다. 문자열 완전 일치는 보고만 하고 판정에 쓰지 않는다.
    G3 신뢰도 |차이| 평균 <= 0.01
사용(로컬, GPU 불필요):
  python src/analysis/gate_check.py --config config_qwen3.yaml --b results/qwen3/preds_M0_cbgate.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate import is_correct  # noqa: E402

G1_AB_MIN = 0.99
G2_CORRECT_MIN = 0.97
G3_DCONF_MEAN_MAX = 0.01


def _load(p):
    return {r["question_id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}


def _hit(r, tol):
    if r.get("answerable_pred") is False or not r.get("prediction"):
        return False
    return bool(is_correct(r["prediction"], r["gold_answer"], r["type"], tol))


def gate_metrics(A, B, tol):
    common = [q for q in B if q in A]
    n = len(common)
    if not n:
        raise SystemExit("공통 문항 없음")
    ab = sum(A[q].get("answerable_pred") == B[q].get("answerable_pred") for q in common) / n
    # 정답 여부는 gold가 있는 B 레코드의 type/gold_answer로 양쪽을 같은 기준으로 채점
    cor = 0
    for q in common:
        ra, rb = dict(A[q]), B[q]
        ra["gold_answer"], ra["type"] = rb["gold_answer"], rb["type"]
        cor += _hit(ra, tol) == _hit(rb, tol)
    cor /= n
    exact = sum(A[q].get("prediction") == B[q].get("prediction") for q in common) / n
    dconf = [abs(float(A[q]["confidence"]) - float(B[q]["confidence"])) for q in common]
    m = {"n": n, "answerable_match": ab, "correct_agree": cor, "answer_exact_match": exact,
         "mean_abs_conf_diff": sum(dconf) / n, "max_abs_conf_diff": max(dconf)}
    m["pass"] = bool(ab >= G1_AB_MIN and cor >= G2_CORRECT_MIN and m["mean_abs_conf_diff"] <= G3_DCONF_MEAN_MAX)
    m["criteria"] = {"answerable_min": G1_AB_MIN, "correct_agree_min": G2_CORRECT_MIN,
                     "mean_conf_diff_max": G3_DCONF_MEAN_MAX}
    return m


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--b", required=True, help="새로 돌린 관문 예측(preds_M0_cbgate.jsonl)")
    ap.add_argument("--a", default=None, help="원 예측. 기본: <results_dir>/preds_M0_v2.jsonl")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    tol = cfg["eval"]["numeric_tolerance"]
    pa = a.a or str(Path(cfg["paths"]["results_dir"]) / "preds_M0_v2.jsonl")
    m = gate_metrics(_load(pa), _load(a.b), tol)
    print(json.dumps(m, ensure_ascii=False, indent=2))
    print("→ 통과" if m["pass"] else "→ 미통과")


if __name__ == "__main__":
    main()
