# -*- coding: utf-8 -*-
"""diag_fallback.py — preds 파일의 answer_span 폴백 원인 분해."""
import json
from collections import Counter


def diag(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    fb = [r for r in rows if r["conf_scope"] != "answer_span"]
    refuse_fb = sum(1 for r in fb if r.get("answerable_pred") is False)
    print(f"--- {path}")
    print(f"  폴백 {len(fb)}/{len(rows)} ({100 * len(fb) / len(rows):.1f}%)")
    print(f"  |- 거절 응답(answerable_pred=False): {refuse_fb}")
    print(f"  |- 유형 분포: {dict(Counter(r['type'] for r in fb))}")
    print(f"  |- scope 분포: {dict(Counter(r['conf_scope'] for r in fb))}")
    print(f"  |- parse_ok 실패: {sum(1 for r in rows if not r['parse_ok'])}")
    n_refuse = sum(1 for r in rows if r.get("answerable_pred") is False)
    n_ua = sum(1 for r in rows if not r.get("gold_answerable", True))
    print(f"  [미리보기] 전체 거절 {n_refuse}건 / gold unanswerable {n_ua}건")


for s in (42, 43, 44):
    diag(f"results/preds_M2_r05_s{s}.jsonl")
    diag("results/preds_M1_s42.jsonl")
    diag("results/preds_M2_r10_s42.jsonl")
    diag("results/preds_M2_r30_s42.jsonl")