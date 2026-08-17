# -*- coding: utf-8 -*-
"""
make_baseline_v2.py — 후속 실험 기준선(baseline_metrics_v2) 확정 (단계 0 후반)

목적:
  손상 gold 24건(numeric, gold_answer에 원문 저장)을 평가셋에서 사전
  제외하는 것을 확정하고(설계서 5.3절), 수정 채점기(v2) 기준으로
  M0·M1(3시드)의 기준 지표를 산출해 단일 파일로 동결한다.
  이 파일이 후속 실험 H2(비열등성) 판정의 비교 대상이다.

sensitivity_goldlen.py 와의 차이:
  민감도(판정 유지 확인)가 아니라 기준선 확정이 목적이므로,
  (a) 제외 목록을 question_id 단위로 파일에 동결하고 재실행 시 불변을
      검증하며, (b) v2 패치 적용 여부를 실행 전에 자동 확인하고,
  (c) H2 비교 대상(응답가능 전체 EM)과 H3·H4 참조치(분리 AUROC)를
      포함한 전체 스냅샷을 저장한다.

사용:
  python make_baseline_v2.py --config config.yaml
  # v2 패치된 evaluate.py, polarity_v2.py 와 같은 폴더에서 실행

출력:
  results/excluded_gold_v2.json     # 제외 목록 동결 (최초 1회 생성, 이후 불변 검증)
  results/baseline_metrics_v2.json  # 기준선 스냅샷
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

import evaluate as ev

GOLD_LEN_LIMIT = 100      # sensitivity_goldlen.py 와 동일 규칙 (동결)
EXPECTED_N_BAD = 24       # 보고서 2.3절 기록과의 대조용


# ------------------------------------------------------------- 사전 검증
def assert_v2_patched():
    """evaluate.is_correct 가 v2 극성(최장 일치 우선)으로 동작하는지 확인.
    v1이면 '불가능합니다' 가 긍정 오판되어 gold 'No' 와 불일치 처리된다."""
    ok = ev.is_correct("불가능합니다", "No", "yes_no", tol=0.01)
    if not ok:
        raise SystemExit(
            "[중단] evaluate.py 가 아직 v1 극성 채점입니다.\n"
            "  polarity_v2.PATCH_NOTE 대로 is_correct 의 yes_no 분기를 교체한 뒤\n"
            "  재실행하세요. (기준선은 반드시 수정 채점기 기준으로 확정)"
        )
    print("[확인] evaluate.py v2 극성 패치 적용됨")


# ------------------------------------------------------------- 제외 목록
def freeze_exclusions(m0, res_dir):
    """손상 gold 제외 목록을 확정·동결. 기존 파일이 있으면 불변 검증."""
    bad = sorted(
        qid for qid, r in m0.items()
        if r.get("gold_answerable", True)
        and r.get("gold_answer") and len(str(r["gold_answer"])) > GOLD_LEN_LIMIT
    )
    by_type = {}
    for qid in bad:
        t = m0[qid]["type"]
        by_type[t] = by_type.get(t, 0) + 1

    record = {
        "rule": f"gold_answerable and len(str(gold_answer)) > {GOLD_LEN_LIMIT}",
        "n": len(bad), "by_type": by_type, "question_ids": bad,
    }
    path = Path(res_dir) / "excluded_gold_v2.json"
    if path.exists():
        prev = json.loads(path.read_text(encoding="utf-8"))
        if prev["question_ids"] != bad:
            raise SystemExit(
                f"[중단] 기존 제외 목록({prev['n']}건)과 재도출 결과({len(bad)}건)가 "
                f"불일치 — 동결 원칙 위반. 원인 확인 필요."
            )
        print(f"[확인] 기존 제외 목록과 일치 ({len(bad)}건) — 동결 유지")
    else:
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"[동결] 제외 목록 신규 저장: {path} ({len(bad)}건, 유형별 {by_type})")

    if len(bad) != EXPECTED_N_BAD:
        print(f"[주의] 제외 건수 {len(bad)} ≠ 보고서 기록 {EXPECTED_N_BAD} — 원인 확인 후 진행")
    return set(bad)


# ------------------------------------------------------------- 참조 지표
def separation_auroc(kept_rows, cond):
    """분리 AUROC: 응답가능 정답(양성) vs 응답불가능 환각(음성)의 신뢰도.
    보고서 7.4절 지표의 재현 — H3(hard UA 판정)·H4 의 easy UA 참조치."""
    pos = [r["conf"][cond] for r in kept_rows
           if r["gold_answerable"] and r["correct"][cond] == 1]
    neg = [r["conf"][cond] for r in kept_rows
           if not r["gold_answerable"] and not r["abstain"][cond]]
    if not pos or not neg:
        return None, len(pos), len(neg)
    confs = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    return float(ev.auroc(confs, labels)), len(pos), len(neg)


def overall_answerable_em(kept_rows, cond):
    ans = [r for r in kept_rows if r["gold_answerable"]]
    return float(np.mean([r["correct"][cond] for r in ans])), len(ans)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = ev.load_config(args.config)
    types = cfg["data"][ev.ANSWERABLE_TYPES_KEY]
    n_bins = cfg["eval"]["n_bins"]
    tol = cfg["eval"]["numeric_tolerance"]
    res_dir = Path(cfg["paths"]["results_dir"])

    assert_v2_patched()

    m0 = ev.load_preds(res_dir / "preds_M0.jsonl")
    bad = freeze_exclusions(m0, res_dir)
    m1_files = ev.find_m1_files(res_dir, None)

    baseline = {
        "meta": {
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "scorer": "v2 (longest-match-first polarity)",
            "exclusion": f"excluded_gold_v2.json ({len(bad)}건)",
            "purpose": "후속 실험 H2 비교 기준선. M2 판정 시 동일 제외 목록·"
                       "동일 채점기 적용 후 paired bootstrap 으로 차이 CI 산출.",
        },
        "per_seed": {},
    }

    print()
    print(f"{'시드':<8} {'응답가능n':>8} {'EM_M0':>8} {'EM_M1':>8} "
          f"{'환각률M1':>8} {'AUROC_M1':>9}")
    for tag, path in m1_files:
        rows = ev.join_pair(m0, ev.load_preds(path), tol)
        kept = [r for r in rows if r["question_id"] not in bad]

        stats = ev.compute_stats(kept, types, n_bins, full=True)
        em_m0, n_ans = overall_answerable_em(kept, "M0")
        em_m1, _ = overall_answerable_em(kept, "M1")
        auroc_m1, n_pos, n_neg = separation_auroc(kept, "M1")

        baseline["per_seed"][tag] = {
            "n_answerable": n_ans,
            "n_excluded": len(rows) - len(kept),
            "overall_em": {"M0": em_m0, "M1": em_m1},          # ← H2 비교 대상
            "separation_auroc_M1": auroc_m1,                    # ← easy UA 참조치
            "separation_n": {"pos_correct": n_pos, "neg_hall": n_neg},
            "stats": stats,                                     # per_type 전체 스냅샷
        }
        # unans 지표가 stats["unans"] 또는 최상위에 있을 수 있음 — 안전 접근
        u = stats.get("unans") or {k: v for k, v in stats.items()
                                   if k.startswith(("hall_", "abstain_acc"))}
        hall = u.get("hall_rate_M1", float("nan"))
        print(f"{tag:<8} {n_ans:>8} {em_m0:>8.3f} {em_m1:>8.3f} "
              f"{hall:>8.3f} {auroc_m1 if auroc_m1 is None else round(auroc_m1, 4):>9}")

    out_path = res_dir / "baseline_metrics_v2.json"
    out_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2,
                                   default=float), encoding="utf-8")
    print(f"\n저장: {out_path}")
    print("\n[다음 단계] H2 판정 시 이 파일의 overall_em.M1 이 아니라, 동일 제외"
          "\n  목록을 적용한 M1·M2 예측의 paired bootstrap 차이 CI 를 사용한다."
          "\n  이 파일은 기준선의 동결 스냅샷(감사·표 작성용)이다.")


if __name__ == "__main__":
    main()