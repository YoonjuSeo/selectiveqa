# -*- coding: utf-8 -*-
"""
sensitivity_yesno.py — [탐색적/부록] 두 가지 사후 민감도 분석.

A. yes_no 방향 기반 재채점:
   gold 라벨("Yes"/"No")과 M0의 한국어 문장형 답("예, ~입니다") 사이의
   형식 불일치로 인한 EM 과소평가를 교정한 민감도 채점.
   명시적 방향 표지(예/네/맞/그렇 vs 아니/그렇지 않)가 있는 경우만 매핑하고
   나머지는 원채점 유지 (보수적). 사전등록 EM 판정은 변경하지 않음.

B. 환각 confidence 분리도:
   응답불가능 문항에서 M1이 지어낸 답의 confidence 분포가
   정답 confidence 분포와 얼마나 분리되는지 (selective prediction 가능성).

사용법: python sensitivity_yesno.py   (main_code 폴더에서)
"""

import glob
import json
import re
import unicodedata

import numpy as np


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    return re.sub(r"[\s\.\,\!\?]+$", "", s)


# 방향 표지 + 경계(문장끝/쉼표/공백/구두점) — '네이버' 같은 명사 오인 방지
_YES = re.compile(
    r"^(?:yes|예|네|넵|맞다|맞아요|맞습니다|맞음|그렇다|그렇습니다|그렇음|그럼|그래요)"
    r"(?=$|[\s,.!?~])", re.I)
_NO = re.compile(
    r"^(?:no|아니오|아니요|아닙니다|아니다|아니에요|아님|틀리다|틀렸다)(?=$|[\s,.!?~])"
    r"|^그렇지\s*않", re.I)


def map_direction(pred):
    """명시적 방향 표지가 있으면 'yes'/'no'로 매핑, 없으면 None."""
    if pred is None:
        return None
    p = norm(pred)
    # "예, ..." / "네, 맞습니다 ..." / "아니오, ..." 등 문두 표지
    if _NO.match(p):
        return "no"
    if _YES.match(p):
        return "yes"
    return None


def ece(confs, corrs, n_bins=15):
    confs, corrs = np.asarray(confs, float), np.asarray(corrs, float)
    edges = np.linspace(0, 1, n_bins + 1)
    total, val = len(confs), 0.0
    if total == 0:
        return float("nan")
    for i in range(n_bins):
        m = (confs > edges[i]) & (confs <= edges[i + 1]) if i else (confs >= 0) & (confs <= edges[1])
        if m.sum() == 0:
            continue
        val += (m.sum() / total) * abs(corrs[m].mean() - confs[m].mean())
    return float(val)


def auroc(pos, neg):
    """pos가 neg보다 큰 경향의 rank AUROC (동률 평균순위)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allc = np.concatenate([pos, neg])
    order = np.argsort(allc, kind="mergesort")
    ranks = np.empty(len(allc))
    sv, i = allc[order], 0
    while i < len(allc):
        j = i
        while j + 1 < len(allc) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r = ranks[: len(pos)].sum()
    return float((r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def yn_correct_mapped(gold, pred):
    g = norm(gold)                      # 'yes' / 'no'
    d = map_direction(pred)
    if d is not None:
        return int(d == g)
    return int(norm(pred) == g)         # 매핑 불가 시 원채점과 동일 기준


def main():
    m0 = load("results/preds_M0.jsonl")
    m1_files = sorted(glob.glob("results/preds_M1_s*.jsonl"))

    print("=" * 66)
    print("A. yes_no 방향 기반 재채점 (탐색적 — 사전등록 EM 판정 불변)")
    print("=" * 66)

    def yn_stats(rows, label):
        yn = [r for r in rows if r["type"] == "yes_no" and r["gold_answerable"]]
        raw = [int(norm(r["prediction"]) == norm(r["gold_answer"])) for r in yn]
        mapped = [yn_correct_mapped(r["gold_answer"], r["prediction"]) for r in yn]
        confs = [r["confidence"] for r in yn]
        n_dir = sum(1 for r in yn if map_direction(r["prediction"]) is not None)
        n_abst = sum(1 for r in yn if r["answerable_pred"] is False)
        print(f"{label:<8} EM(원채점) {np.mean(raw):.3f} → EM(방향매핑) {np.mean(mapped):.3f}"
              f" | 방향표지 검출 {n_dir}/{len(yn)}건, 무응답 {n_abst}건"
              f" | ECE(원) {ece(confs, raw):.3f} → ECE(매핑) {ece(confs, mapped):.3f}")
        return np.mean(mapped), ece(confs, mapped)

    em0_m, ece0_m = yn_stats(m0, "M0")
    for f in m1_files:
        tag = f.split("preds_")[1].split(".jsonl")[0]
        em1_m, ece1_m = yn_stats(load(f), tag)
        print(f"         → 재채점 기준 yes_no: Δacc {em1_m - em0_m:+.3f}, "
              f"ΔECE {ece1_m - ece0_m:+.3f}")

    print()
    print("=" * 66)
    print("B. M1 환각 confidence 분리도 (selective prediction 가능성)")
    print("=" * 66)
    for f in m1_files:
        tag = f.split("preds_")[1].split(".jsonl")[0]
        rows = load(f)
        hall = [r["confidence"] for r in rows
                if not r["gold_answerable"] and r["answerable_pred"] is not False]
        corr = [r["confidence"] for r in rows
                if r["gold_answerable"]
                and norm(r["prediction"]) == norm(r["gold_answer"])]
        hall_a, corr_a = np.array(hall), np.array(corr)
        au = auroc(corr_a, hall_a)
        print(f"{tag}: 환각 conf 평균 {hall_a.mean():.3f} "
              f"(사분위 {np.percentile(hall_a, 25):.2f}/{np.percentile(hall_a, 50):.2f}/{np.percentile(hall_a, 75):.2f}, n={len(hall_a)})"
              f" vs 정답 conf 평균 {corr_a.mean():.3f} (n={len(corr_a)})"
              f" | 분리 AUROC {au:.3f}")
        # 운영 관점: 정답 90%를 보존하는 임계값에서 환각을 얼마나 거르는가
        thr = np.percentile(corr_a, 10)
        blocked = float((hall_a < thr).mean())
        print(f"      임계값 conf≥{thr:.3f} (정답 90% 보존) 적용 시 환각 차단율 {blocked:.1%}")


if __name__ == "__main__":
    main()