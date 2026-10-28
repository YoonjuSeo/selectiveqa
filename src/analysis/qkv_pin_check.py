# -*- coding: utf-8 -*-
"""qkv_pin_check.py — 사전등록 prereg_exaone_qkv_pin_261028.md 4절 판정.

  python src/analysis/qkv_pin_check.py
  python src/analysis/qkv_pin_check.py --m2 results/exaone_qkv_pin/preds_M2_r05_qkvpin_s43.jsonl --m1 results/preds_M1_v2_s43.jsonl
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.evaluate_followup import is_correct  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--m1", default="results/preds_M1_v2_s42.jsonl")          # 원 M1 (4.x, q/k/v)
ap.add_argument("--m2", default="results/exaone_qkv_pin/preds_M2_r05_qkvpin_s42.jsonl")
ap.add_argument("--orig-m2", default="results/preds_M2_r05_s42.jsonl")     # 참고: 원 M2 (5.x, q/k/v)
a = ap.parse_args()

ex = set()
for n in ["excluded_gold_v2.json", "excluded_gold_v2_manual.json"]:
    ex |= set(json.load(open(Path("results") / n, encoding="utf-8"))["question_ids"])

def load(p):
    return {r["question_id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}

def stats(pr, ids):
    ans = [pr[q] for q in ids if pr[q]["gold_answerable"]]
    em = sum(r.get("answerable_pred") is not False and is_correct(r["prediction"], r["gold_answer"], r["type"], 1e-4) for r in ans) / len(ans)
    tok = sum(r.get("n_gen_tokens", 0) for r in ans) / len(ans)
    fab = sum(r.get("answerable_pred") is False for r in ans) / len(ans)
    out = {"n_ans": len(ans), "EM": em, "tok": tok, "false_abst": fab}
    for k, f in [("easy", lambda q: q.startswith("unans_")), ("hard", lambda q: "-hd-" in q)]:
        g = [pr[q] for q in ids if f(q)]
        out[f"hall_{k}"] = sum(r.get("answerable_pred") is not False for r in g) / len(g)
    return out

m1, m2 = load(a.m1), load(a.m2)
ids = sorted((set(m1) & set(m2)) - ex)
s1, s2 = stats(m1, ids), stats(m2, ids)
d_em, r_tok = s2["EM"] - s1["EM"], s2["tok"] / s1["tok"]
verdict = ("Q1 붕괴 재현 → 원인: out_proj 누락" if d_em <= -0.20 and r_tok >= 1.8 else
           "Q2 붕괴 없음 → 원인: transformers 5.x 학습 환경" if d_em >= -0.05 and r_tok <= 1.2 else
           "Q3 부분 → 원인 특정 안 함 (seed 43 추가)")
print(f"문항 {len(ids)} (Answerable {s1['n_ans']})")
for name, s in [("원 M1 (4.x, q/k/v)", s1), ("이번 M2 (4.57.1, q/k/v)", s2)]:
    print(f"  {name:24s} EM {s['EM']:.3f}  토큰 {s['tok']:.1f}  허위무응답 {s['false_abst']:.3f}  easy환각 {s['hall_easy']:.3f}  hard환각 {s['hall_hard']:.3f}")
if Path(a.orig_m2).exists():
    s0 = stats(load(a.orig_m2), [q for q in ids if q in load(a.orig_m2)])
    print(f"  {'(참고) 원 M2 (5.x, q/k/v)':24s} EM {s0['EM']:.3f}  토큰 {s0['tok']:.1f}")
print(f"ΔEM {d_em*100:+.1f}%p  R_tok ×{r_tok:.2f}  →  {verdict}")
