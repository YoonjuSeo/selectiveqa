    # diag_er.py — gold_not_located의 유형별 분포 확인
import json
from collections import Counter
from prepare_data import _norm
from make_hard_unanswerable import gold_variants, split_segments

rows = [json.loads(l) for l in open("data/processed/eval.jsonl", encoding="utf-8")]
c = Counter()
for r in rows:
    if r.get("type") in (None, "unanswerable"):
        continue
    segs = split_segments(r["context"])
    hit = any(any(v in _norm(s) for v in gold_variants(r["gold_answer"])) for s in segs)
    c[(r["type"], "located" if hit else "NOT_located")] += 1
print(dict(c))