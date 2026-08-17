# -*- coding: utf-8 -*-
"""
sensitivity_goldlen.py — [탐색적/부록] 손상 gold(정답 필드에 원문 텍스트가 들어간
numeric 문항 24건) 제외 재판정 민감도.

배경: 수동 검수 중 numeric 유형 일부에서 gold_answer에 정답 대신 기사 본문이
저장된 것을 확인. numeric 채점은 gold 내 임의 숫자와의 허용오차 일치를 정답으로
보므로, 숫자가 다수 포함된 손상 gold는 M0·M1 양쪽에서 우연 일치(가짜 정답)를
유발할 수 있다. 해당 문항 제외 시 numeric 지표와 H2 contrast가 유지되는지
확인한다. 사전등록 판정은 변경하지 않는다.

사용법: python sensitivity_goldlen.py   (main_code 폴더, CPU 가능)
"""

import glob

from evaluate import (CONTRAST_NEG, CONTRAST_POS, bootstrap_ci, compute_stats,
                      join_pair, load_config, load_preds)

GOLD_LEN_LIMIT = 100   # 이 길이를 넘는 gold는 손상으로 간주


def main():
    cfg = load_config()
    n_bins = cfg["eval"]["n_bins"]
    tol = cfg["eval"]["numeric_tolerance"]
    n_boot = cfg["eval"]["n_bootstrap"]
    types = [CONTRAST_NEG, "yes_no"] + list(CONTRAST_POS)
    rd = cfg["paths"]["results_dir"]

    m0 = load_preds(f"{rd}/preds_M0.jsonl")
    bad = {qid for qid, r in m0.items()
           if r.get("gold_answer") and len(str(r["gold_answer"])) > GOLD_LEN_LIMIT}
    by_type = {}
    for qid in bad:
        t = m0[qid]["type"]
        by_type[t] = by_type.get(t, 0) + 1
    print(f"손상 gold 제외 대상: {len(bad)}건 (유형별 {by_type})")
    print()

    for path in sorted(glob.glob(f"{rd}/preds_M1_s*.jsonl")):
        tag = path.split("preds_")[1].split(".jsonl")[0]
        rows = join_pair(m0, load_preds(path), tol)
        kept = [r for r in rows if r["question_id"] not in bad]

        base = compute_stats(rows, types, n_bins)
        excl = compute_stats(kept, types, n_bins)
        ci = bootstrap_ci(kept, types, n_bins, n_boot, cfg["seed"])

        nb = base["per_type"]["numeric_reasoning"]
        ne = excl["per_type"]["numeric_reasoning"]
        nc = ci["delta_ece:numeric_reasoning"]
        cc = ci.get("contrast", {})
        print(f"== {tag} ==  (제외 후 numeric n={ne['n']})")
        print(f"  numeric EM M0 {nb['em_M0']:.3f}→{ne['em_M0']:.3f} | "
              f"EM M1 {nb['em_M1']:.3f}→{ne['em_M1']:.3f} | "
              f"Δconf {nb['delta_conf']:.3f}→{ne['delta_conf']:.3f}")
        print(f"  numeric ΔECE {nb['delta_ece']:.3f} → {ne['delta_ece']:.3f} "
              f"[{nc['lo']:.3f}, {nc['hi']:.3f}]")
        print(f"  contrast {base['contrast']:.3f} → {excl['contrast']:.3f} "
              f"[{cc.get('lo', float('nan')):.3f}, {cc.get('hi', float('nan')):.3f}]"
              f"  ← CI 0 제외 & |값|≥0.03 유지 시 H2 강건")
        print()


if __name__ == "__main__":
    main()