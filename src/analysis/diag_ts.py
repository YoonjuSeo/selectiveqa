# diag_ts.py — 유형별 TS 적격(치환 대상 존재 + 원대상이 지문에 존재) 소스 수 확인
import json
from collections import Counter
from data_prep.prepare_data import _norm
from data_prep.make_hard_unanswerable import find_swap_targets

rows = [json.loads(l) for l in open("data/processed/eval.jsonl", encoding="utf-8")]
c = Counter()
for r in rows:
    if r.get("type") in (None, "unanswerable"):
        continue
    ok = any(_norm(t[1]) in _norm(r["context"]) for t in find_swap_targets(r["question"]))
    c[(r["type"], "ts_ok" if ok else "ts_NO")] += 1
print(dict(c))