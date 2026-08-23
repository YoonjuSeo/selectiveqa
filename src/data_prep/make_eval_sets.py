# -*- coding: utf-8 -*-
"""make_eval_sets.py — 후속 실험용 평가 파일 3종 파생. eval.jsonl(원본)은 불변."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
proc = Path(cfg["paths"]["processed_dir"])
rng = random.Random(cfg["seed"])

ev = [json.loads(l) for l in open(proc / "eval.jsonl", encoding="utf-8")]
hard = [json.loads(l) for l in open(proc / "hard_ua_final.jsonl", encoding="utf-8")]
ans = [r for r in ev if r.get("type") != "unanswerable"]
easy = [r for r in ev if r.get("type") == "unanswerable"]
print(f"원본: 응답가능 {len(ans)} · easy UA {len(easy)} · hard UA {len(hard)}")

# (a) M2 본추론용
full = ans + easy + hard
# (b) M0·M1 보강 추론용 (hard만)
# (c) 에폭 분석용: easy 300 + 응답가능 유형별 50 층화
by_t = {}
for r in ans:
    by_t.setdefault(r["type"], []).append(r)
sample = []
for t, rs in sorted(by_t.items()):
    rs = rs[:]
    rng.shuffle(rs)
    sample += rs[:50]
epoch_set = easy + sample

for name, rows in (("eval_full_v2.jsonl", full), ("eval_hard.jsonl", hard),
                   ("eval_epoch.jsonl", epoch_set)):
    ids = [r["question_id"] for r in rows]
    dup = [q for q, c in Counter(ids).items() if c > 1]
    assert not dup, f"{name}: question_id 중복 {len(dup)}건"
    with open(proc / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)}건 (중복 0 ✓)")