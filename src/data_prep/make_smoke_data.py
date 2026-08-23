# -*- coding: utf-8 -*-
"""make_smoke_data.py — 스모크 테스트용 축소 학습 데이터 (무응답 20 + 응답가능 180)."""
import json
from pathlib import Path

import yaml

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
proc = Path(cfg["paths"]["processed_dir"])

rows = [json.loads(l) for l in open(proc / "train_mix_r05.jsonl", encoding="utf-8")]
ua = [r for r in rows if not r.get("answerable", True)][:20]
an = [r for r in rows if r.get("answerable", True)][:180]
smoke = ua + an

out = proc / "train_smoke.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for r in smoke:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"저장: {out} — 총 {len(smoke)}건 (무응답 {len(ua)} + 응답가능 {len(an)})")
assert len(ua) == 20, "무응답 예시가 20건 미만 — 스모크의 목적(무응답 타깃 경로 검증) 미달"
print("✓ 무응답 예시 포함 확인")