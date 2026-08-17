# -*- coding: utf-8 -*-
"""
rescore_yesno_diff.py — yes_no 극성 채점 v1→v2 재채점 diff (후속 실험 단계 0)

목적:
  본실험의 M0·M1(3시드) 예측 원본을 재추론 없이 v1(현행)·v2(최장 일치
  우선) 두 채점기로 채점하고, (a) 판정이 뒤집힌 문항의 전수 목록,
  (b) 조건·시드별 EM 변화, (c) yes_no ECE·ΔECE 변화를 산출한다.

판정 영향 범위 (사전 확인 사항):
  - H2 contrast 는 (table, numeric) vs text 로 정의되어 yes_no 와 무관
    → 극성 수정은 H2 판정에 영향 없음 (콘솔에 명시 출력)
  - 영향 가능 지표: yes_no EM · yes_no ECE/ΔECE · 전체 EM · 탐색 지표
  - 보고서 7.3절 예상 방향: v1 결함은 M0(문장형 답변)에 불리(보수적)
    → v2 에서 M0 yes_no EM 이 상승 또는 불변이어야 정합

사용:
  python rescore_yesno_diff.py --config config.yaml
  # evaluate.py, polarity_v2.py 와 같은 폴더에서 실행

출력:
  results/yesno_rescore_diff.jsonl   # 뒤집힌 문항 전수 (감사 로그)
  results/yesno_rescore_summary.json # 조건·시드별 요약
  콘솔 요약표
"""
import argparse
import json
from pathlib import Path

import numpy as np

# 동결 로직 재사용: 정규화·ECE·로더는 evaluate.py 원본을 그대로 쓴다
import evaluate as ev
from polarity_v2 import polarity_v1, polarity_v2


# --------------------------------------------------------------- 채점 유틸
def score_yesno(pred, gold, pol_fn):
    """극성 함수 pol_fn 으로 yes_no 1건 채점. 반환: (correct, gp, pp, path)
    path: 'polarity'(양쪽 표지) | 'string_fallback'(미표지 폴백)"""
    gp, pp = pol_fn(gold), pol_fn(pred)
    if gp is not None and pp is not None:
        return int(gp == pp), gp, pp, "polarity"
    return int(ev.normalize(pred) == ev.normalize(gold)), gp, pp, "string_fallback"


def rescore_condition(preds, tag):
    """한 조건(M0 또는 M1_sXX)의 yes_no 전 문항을 v1/v2로 채점.
    반환: (rows: 문항별 기록 목록, summary: dict)"""
    rows = []
    for qid, r in sorted(preds.items()):
        if r.get("type") != "yes_no" or not r.get("gold_answerable", True):
            continue
        pred, gold = r.get("prediction"), r.get("gold_answer")
        c1, gp1, pp1, path1 = score_yesno(pred, gold, polarity_v1)
        c2, gp2, pp2, path2 = score_yesno(pred, gold, polarity_v2)
        pol2, kw2, amb2 = polarity_v2(pred, return_match=True)
        rows.append({
            "question_id": qid, "condition": tag,
            "prediction": pred, "gold": gold,
            "v1": {"correct": c1, "gold_pol": gp1, "pred_pol": pp1, "path": path1},
            "v2": {"correct": c2, "gold_pol": gp2, "pred_pol": pp2, "path": path2,
                   "matched_kw": kw2, "ambiguous_coexist": amb2},
            "flip_polarity": (pp1 != pp2) or (gp1 != gp2),
            "flip_correct": c1 != c2,
            "confidence": r.get("confidence"),
        })

    n = len(rows)
    em1 = float(np.mean([x["v1"]["correct"] for x in rows])) if n else float("nan")
    em2 = float(np.mean([x["v2"]["correct"] for x in rows])) if n else float("nan")
    flips = [x for x in rows if x["flip_correct"]]
    summary = {
        "n_yes_no": n,
        "em_v1": em1, "em_v2": em2, "delta_em": em2 - em1,
        "n_flip_polarity": sum(x["flip_polarity"] for x in rows),
        "n_flip_correct": len(flips),
        "n_flip_to_correct": sum(1 for x in flips if x["v2"]["correct"] == 1),
        "n_flip_to_wrong": sum(1 for x in flips if x["v2"]["correct"] == 0),
        "n_ambiguous_coexist": sum(x["v2"]["ambiguous_coexist"] for x in rows),
        "n_string_fallback_v2": sum(1 for x in rows if x["v2"]["path"] == "string_fallback"),
    }
    return rows, summary


def yesno_ece_pair(m0_preds, m1_preds, version, n_bins):
    """yes_no 유형의 ECE(M0), ECE(M1), ΔECE 를 지정 채점기로 재계산.
    (ECE 는 correctness 에 의존하므로 채점기 버전에 따라 달라진다)"""
    pol_fn = polarity_v1 if version == "v1" else polarity_v2
    out = {}
    for tag, preds in (("M0", m0_preds), ("M1", m1_preds)):
        confs, corrs = [], []
        for qid, r in preds.items():
            if r.get("type") != "yes_no" or not r.get("gold_answerable", True):
                continue
            c, _, _, _ = score_yesno(r.get("prediction"), r.get("gold_answer"), pol_fn)
            confs.append(float(r.get("confidence") or 0.0))
            corrs.append(c)
        out[tag] = ev.ece(confs, corrs, n_bins) if confs else float("nan")
    out["delta_ece"] = out["M1"] - out["M0"]
    return out


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = ev.load_config(args.config)
    n_bins = cfg["eval"]["n_bins"]
    res_dir = Path(cfg["paths"]["results_dir"])

    m0 = ev.load_preds(res_dir / "preds_M0.jsonl")
    m1_files = ev.find_m1_files(res_dir, None)

    all_rows, summaries, ece_report = [], {}, {}

    rows, summaries["M0"] = rescore_condition(m0, "M0")
    all_rows += rows
    for tag, path in m1_files:
        m1 = ev.load_preds(path)
        rows, summaries[tag] = rescore_condition(m1, tag)
        all_rows += rows
        ece_report[tag] = {
            "v1": yesno_ece_pair(m0, m1, "v1", n_bins),
            "v2": yesno_ece_pair(m0, m1, "v2", n_bins),
        }

    # ---- 저장: 뒤집힌 문항 전수 + 요약 ------------------------------------
    diff_path = res_dir / "yesno_rescore_diff.jsonl"
    with open(diff_path, "w", encoding="utf-8") as f:
        for x in all_rows:
            if x["flip_polarity"] or x["flip_correct"]:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    summary = {
        "note_h2": "H2 contrast 는 (table_lookup, numeric_reasoning) vs text_span "
                   "으로 정의되어 yes_no 채점 변경의 영향을 받지 않음 (판정 불변).",
        "per_condition": summaries,
        "yesno_ece": ece_report,
    }
    sum_path = res_dir / "yesno_rescore_summary.json"
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # ---- 콘솔 요약 --------------------------------------------------------
    print("=" * 72)
    print("yes_no 극성 재채점 diff (v1: 현행 / v2: 최장 일치 우선)")
    print("=" * 72)
    print(f"{'조건':<10} {'n':>4} {'EM v1':>8} {'EM v2':>8} {'ΔEM':>8} "
          f"{'극성뒤집힘':>10} {'채점뒤집힘':>10} {'→정답':>6} {'→오답':>6}")
    for tag, s in summaries.items():
        print(f"{tag:<10} {s['n_yes_no']:>4} {s['em_v1']:>8.3f} {s['em_v2']:>8.3f} "
              f"{s['delta_em']:>+8.3f} {s['n_flip_polarity']:>10} "
              f"{s['n_flip_correct']:>10} {s['n_flip_to_correct']:>6} "
              f"{s['n_flip_to_wrong']:>6}")
    print("-" * 72)
    for tag, e in ece_report.items():
        print(f"[{tag}] yes_no ΔECE  v1: {e['v1']['delta_ece']:+.4f}  →  "
              f"v2: {e['v2']['delta_ece']:+.4f}")
    print("-" * 72)
    print(summary["note_h2"])
    print(f"\n저장: {diff_path}\n저장: {sum_path}")

    # ---- 정합성 자동 점검 (보고서 7.3절 예상 방향) ------------------------
    d = summaries["M0"]["delta_em"]
    if d < 0:
        print("\n[경고] M0 yes_no EM 이 v2에서 하락 — 예상 방향(결함은 M0에 "
              "불리=보수적)과 반대. 뒤집힌 문항을 수동 확인할 것.")
    else:
        print(f"\n[정합] M0 yes_no EM 변화 {d:+.3f} — '원채점이 보수적' 해석과 부합.")


if __name__ == "__main__":
    main()