# -*- coding: utf-8 -*-
"""
compare_preds.py — 같은 문항에 대한 두 예측 파일의 일치도 (BOS 점검 4.8.3절의 재현 관문용).

  A: 원 예측 파일 (예: results/kanana/preds_M1_v2_s42.jsonl)
  B: 새로 돌린 예측 파일 (예: results/kanana/preds_repro_M1_s42.jsonl)
공통 question_id만 비교한다. 출력: 생성 텍스트·answer·answerable 일치율, 신뢰도 차이.

사용:
  python src/analysis/compare_preds.py A.jsonl B.jsonl [--show 5]
"""
import argparse
import json


def load(p):
    return {r["question_id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--show", type=int, default=5)
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    common = [q for q in B if q in A]
    if not common:
        raise SystemExit("공통 문항 없음")
    same_text = [q for q in common if A[q].get("gen_text") == B[q].get("gen_text")] \
        if all("gen_text" in A[q] for q in common) else None
    same_ans = [q for q in common if A[q].get("prediction") == B[q].get("prediction")]
    same_ab = [q for q in common if A[q].get("answerable_pred") == B[q].get("answerable_pred")]
    dconf = [abs(float(A[q]["confidence"]) - float(B[q]["confidence"])) for q in common]
    n = len(common)
    print(f"공통 문항 {n}건")
    if same_text is not None:
        print(f"  생성 텍스트 완전 일치: {len(same_text)}/{n} ({100 * len(same_text) / n:.1f}%)")
    else:
        print("  생성 텍스트: A 파일에 gen_text 없음 → answer 기준으로만 비교")
    print(f"  answer 일치: {len(same_ans)}/{n} ({100 * len(same_ans) / n:.1f}%)")
    print(f"  answerable 판단 일치: {len(same_ab)}/{n} ({100 * len(same_ab) / n:.1f}%)")
    print(f"  신뢰도 |차이|: 평균 {sum(dconf) / n:.4f}, 최대 {max(dconf):.4f}")
    if "bos_dedup" in next(iter(B.values())):
        k = sum(bool(B[q].get("bos_dedup")) for q in common)
        print(f"  B의 BOS 중복 제거 문항: {k}/{n}")
    diff = [q for q in common if A[q].get("prediction") != B[q].get("prediction")
            or A[q].get("answerable_pred") != B[q].get("answerable_pred")]
    for q in diff[: a.show]:
        print(f"  - {q}\n      A: {A[q].get('answerable_pred')} | {str(A[q].get('prediction'))[:70]}"
              f"\n      B: {B[q].get('answerable_pred')} | {str(B[q].get('prediction'))[:70]}")


if __name__ == "__main__":
    main()
