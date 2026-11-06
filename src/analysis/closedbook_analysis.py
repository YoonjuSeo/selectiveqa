# -*- coding: utf-8 -*-
"""
closedbook_analysis.py — closed-book 통제 실험 분석 (GPU 불필요, 로컬 CPU).
사전 계획: docs/prereg_closedbook_261106.md

입력 (results_dir 기준):
  preds_M0_cb_force.jsonl          closed-book(force) M0 예측  ← modal_inference_cb.py
  preds_M0_v2.jsonl / preds_M1_*_s{seed}.jsonl / preds_M2_r05_s{seed}.jsonl   기존 예측 (재사용)
  excluded_gold_v2*.json           기존 제외 목록 (evaluate_followup.py와 동일 규칙)
평가셋: data/processed/eval_full_v2.jsonl 의 orig_gold(UA 편집 전 원 정답), gold_answer(Answerable)

산출:
  closedbook_<mode>.json         유형별 closed-book 정답률, '사전지식 가능' 문항 목록, 민감도 분석 수치
  closedbook_<mode>.md           논문 표용 마크다운
채점: evaluate.is_correct (numeric 허용오차·yes/no 극성·정규화 규칙 기존과 동일)

사용:
  python src/analysis/closedbook_analysis.py --config config_qwen3.yaml
  python src/analysis/closedbook_analysis.py --config config_abl_exaone_outproj.yaml --m1-glob "preds_M1_pin_s*.jsonl" --m2-glob "preds_M2_r05_pin_s*.jsonl"
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from collections import defaultdict

import yaml

from evaluation.evaluate import is_correct, extract_numbers, normalize
from evaluation.evaluate_followup import load_exclusions


def ua_kind(qid):
    if qid.startswith("unans_"):
        return "easy"
    if "-hd-er" in qid:
        return "hard_er"
    if "-hd-ts" in qid:
        return "hard_ts"
    return "ans"


def load_jsonl(p):
    return {r["question_id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}


def is_binary(g):
    """지문 없이도 50% 확률로 맞히는 문항: yes/no, 질문 안에 두 선택지('A와 B 중')가 있는 경우."""
    q = g["question"]
    return g.get("orig_type") == "yes_no" or (" 중 " in q or "중에" in q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", default="force", choices=["force", "abstain"])
    ap.add_argument("--cb-file", default=None, help="기본: preds_M0_cb_{mode}.jsonl")
    ap.add_argument("--m1-glob", default="preds_M1_v2_s*.jsonl")
    ap.add_argument("--m2-glob", default="preds_M2_r05_s*.jsonl")
    ap.add_argument("--eval-file", default="eval_full_v2.jsonl")
    ap.add_argument("--exclude-files", nargs="*", default=["excluded_gold_v2.json", "excluded_gold_v2_manual.json"])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    res_dir = Path(cfg["paths"]["results_dir"])
    tol = cfg["eval"]["numeric_tolerance"]
    gold = load_jsonl(Path(cfg["paths"]["processed_dir"]) / args.eval_file)
    excluded = load_exclusions(res_dir, args.exclude_files)
    cb = load_jsonl(res_dir / (args.cb_file or f"preds_M0_cb_{args.mode}.jsonl"))

    # 1) closed-book 정답률: UA는 orig_gold, Answerable은 gold_answer 기준
    stats = defaultdict(lambda: {"n": 0, "hit": 0, "abstain": 0, "n_nonbinary": 0, "hit_nonbinary": 0})
    knowable = {"easy": [], "hard_er": [], "hard_ts": [], "ans": []}
    for qid, r in cb.items():
        if qid in excluded or qid not in gold:
            continue
        g = gold[qid]; k = ua_kind(qid)
        ref = g["gold_answer"] if k == "ans" else g.get("orig_gold")
        qtype = g["type"] if k == "ans" else g.get("orig_type")
        s = stats[k]; s["n"] += 1
        if r.get("answerable_pred") is False or not r.get("prediction"):
            s["abstain"] += 1
            continue
        hit = bool(ref) and is_correct(r["prediction"], ref, qtype, tol)
        s["hit"] += hit
        if k != "ans" and not is_binary(g):
            s["n_nonbinary"] += 1; s["hit_nonbinary"] += hit
        if hit:
            knowable[k].append(qid)
    for k in list(stats):
        s = stats[k]
        s["hit_rate"] = s["hit"] / s["n"] if s["n"] else None
        s["hit_rate_nonbinary"] = (s["hit_nonbinary"] / s["n_nonbinary"]) if s["n_nonbinary"] else None
    know_ids = set(knowable["easy"] + knowable["hard_er"] + knowable["hard_ts"])

    # 2) 민감도 분석: '사전지식 가능' UA를 제외했을 때 기존 지표가 달라지는가
    #    (M1 hard UA 환각률, M2-r05 hard UA 무응답률, M0 대비 복원 비율)
    def abst_rate(preds, kinds, drop):
        ids = [q for q in preds if q in gold and ua_kind(q) in kinds and q not in excluded and q not in drop]
        if not ids: return None, 0
        return sum(preds[q].get("answerable_pred") is False for q in ids) / len(ids), len(ids)

    sens = {}
    m0_path = res_dir / "preds_M0_v2.jsonl"
    m0 = load_jsonl(m0_path) if m0_path.exists() else None
    m1_files = sorted(res_dir.glob(args.m1_glob)); m2_files = sorted(res_dir.glob(args.m2_glob))
    hard = {"hard_er", "hard_ts"}
    for label, files in [("M1", m1_files), ("M2_r05", m2_files)]:
        for p in files:
            preds = load_jsonl(p)
            row = {}
            for nm, drop in [("all", set()), ("excl_knowable", know_ids)]:
                for kk, kinds in [("easy", {"easy"}), ("hard", hard), ("hard_er", {"hard_er"}), ("hard_ts", {"hard_ts"})]:
                    rate, n = abst_rate(preds, kinds, drop)
                    row[f"abst_{kk}_{nm}"] = rate; row[f"n_{kk}_{nm}"] = n
            if m0 is not None and label == "M2_r05":
                for nm, drop in [("all", set()), ("excl_knowable", know_ids)]:
                    r0, _ = abst_rate(m0, hard, drop)
                    row[f"restoration_hard_{nm}"] = (row[f"abst_hard_{nm}"] / r0) if r0 else None
            sens[p.name] = row
    if m0 is not None:
        sens["M0"] = {f"abst_{kk}_{nm}": abst_rate(m0, kinds, drop)[0]
                      for nm, drop in [("all", set()), ("excl_knowable", know_ids)]
                      for kk, kinds in [("easy", {"easy"}), ("hard", hard)]}

    out = {"config": args.config, "mode": args.mode, "n_excluded_gold": len(excluded),
           "closed_book": dict(stats), "knowable_question_ids": knowable, "n_knowable_ua": len(know_ids),
           "sensitivity": sens}
    (res_dir / f"closedbook_{args.mode}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) 마크다운 표
    name = {"easy": "주제 불일치형(easy UA)", "hard_er": "근거 제거형", "hard_ts": "대상 치환형", "ans": "Answerable"}
    md = [f"# closed-book({args.mode}) 결과 — {cfg['model']['base_model']}", "",
          "| 문항 유형 | n | closed-book 정답 | 정답률 | 무응답 | 이지선다 제외 정답률 |", "|---|---|---|---|---|---|"]
    for k in ["ans", "easy", "hard_er", "hard_ts"]:
        s = stats.get(k)
        if not s: continue
        nb = f"{s['hit_nonbinary']}/{s['n_nonbinary']} ({100*s['hit_rate_nonbinary']:.1f}%)" if s["hit_rate_nonbinary"] is not None else "–"
        md.append(f"| {name[k]} | {s['n']} | {s['hit']} | {100*s['hit_rate']:.1f}% | {s['abstain']} | {nb} |")
    md += ["", f"사전지식 가능 UA 문항: {len(know_ids)}건 (easy {len(knowable['easy'])}, ER {len(knowable['hard_er'])}, TS {len(knowable['hard_ts'])})", "",
           "## 민감도: 사전지식 가능 UA 제외 전/후", "", "| 예측 파일 | hard UA 무응답률 (전체) | hard UA 무응답률 (제외 후) | ER 제외 후 | TS 제외 후 | 복원 비율 전→후 |", "|---|---|---|---|---|---|"]
    f = lambda v: "–" if v is None else f"{100*v:.1f}%"
    for fn, row in sens.items():
        if fn == "M0":
            md.append(f"| M0 | {f(row.get('abst_hard_all'))} | {f(row.get('abst_hard_excl_knowable'))} | – | – | – |"); continue
        rr = row.get("restoration_hard_all"); rr2 = row.get("restoration_hard_excl_knowable")
        rtxt = f"{rr:.2f} → {rr2:.2f}" if rr is not None and rr2 is not None else "–"
        md.append(f"| {fn} | {f(row['abst_hard_all'])} | {f(row['abst_hard_excl_knowable'])} | {f(row['abst_hard_er_excl_knowable'])} | {f(row['abst_hard_ts_excl_knowable'])} | {rtxt} |")
    (res_dir / f"closedbook_{args.mode}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n저장: {res_dir / f'closedbook_{args.mode}.json'}, {res_dir / f'closedbook_{args.mode}.md'}")


if __name__ == "__main__":
    main()
