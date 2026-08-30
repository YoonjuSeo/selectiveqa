# -*- coding: utf-8 -*-
"""scan_raw_ua.py — 원본 라벨링 데이터에서 무응답 관련 필드 탐색."""
import json
from pathlib import Path
from collections import Counter

UA_KEYS = {"is_impossible", "impossible", "unanswerable", "is_unanswerable",
           "answerable", "no_answer", "isImpossible"}

def walk_keys(obj, found, depth=0):
    """중첩 구조에서 전체 키 수집 + UA 후보 키의 값 분포."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            found["keys"][k] += 1
            if k in UA_KEYS:
                found["ua"][(k, str(v)[:20])] += 1
            if k in ("answers", "answer") and (v == [] or v == "" or v is None):
                found["empty_answer"] += 1
            walk_keys(v, found, depth + 1)
    elif isinstance(obj, list):
        for item in obj[:2000]:          # 파일당 표본 제한 (속도)
            walk_keys(item, found, depth + 1)

for p in sorted(Path("data/raw").rglob("*.json")):
    if not (p.name.startswith(("TL_", "VL_"))):
        continue
    try:
        data = json.load(open(p, encoding="utf-8-sig"))
    except Exception as e:
        print(f"{p.name}: 로드 실패 ({e})"); continue
    found = {"keys": Counter(), "ua": Counter(), "empty_answer": 0}
    walk_keys(data, found)
    ua_str = dict(found["ua"]) if found["ua"] else "없음"
    print(f"\n=== {p.relative_to('data/raw')} ===")
    print(f"  UA 후보 필드: {ua_str}")
    print(f"  빈 answer: {found['empty_answer']}건 (표본 내)")
    interesting = [k for k in found["keys"] if any(
        w in k.lower() for w in ("possib", "answer", "impos", "unans"))]
    print(f"  answer 관련 키: {sorted(interesting)}")