# -*- coding: utf-8 -*-
"""diag_trainmix.py — train.jsonl(M1) vs train_mix_r05.jsonl(M2)의 응답가능 예시 동일성 검증.

M2의 장황화가 '혼입의 효과'인지 '학습 타깃이 달랐던 인공물'인지 판별한다.
사용: python src/analysis/diag_trainmix.py
"""
import json
import numpy as np

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

base = load("data/processed/train.jsonl")
mix = load("data/processed/train_mix_r30.jsonl")

print(f"train.jsonl: {len(base)}건 · train_mix_r30.jsonl: {len(mix)}건")
mix_ans = [r for r in mix if r.get("answerable", True)]
mix_ua = [r for r in mix if not r.get("answerable", True)]
print(f"mix 구성: 응답가능 {len(mix_ans)} + 무응답 {len(mix_ua)} (기대: 2850 + 150)\n")

base_by_id = {r["question_id"]: r for r in base}
overlap = [r for r in mix_ans if r["question_id"] in base_by_id]
print(f"응답가능 중 train.jsonl과 question_id 겹침: {len(overlap)}/{len(mix_ans)}")

if overlap:
    same_gold = sum(str(r["gold_answer"]) == str(base_by_id[r["question_id"]]["gold_answer"])
                    for r in overlap)
    print(f"gold_answer 완전 일치: {same_gold}/{len(overlap)}")
    gl_mix = [len(str(r["gold_answer"])) for r in overlap]
    gl_base = [len(str(base_by_id[r["question_id"]]["gold_answer"])) for r in overlap]
    print(f"gold 길이(문자) 평균: base {np.mean(gl_base):.1f} vs mix {np.mean(gl_mix):.1f}")
    diff = [r for r in overlap
            if str(r["gold_answer"]) != str(base_by_id[r["question_id"]]["gold_answer"])][:5]
    for r in diff:
        b = base_by_id[r["question_id"]]
        print(f"\n[불일치 예] {r['question_id']} ({r['type']})")
        print(f"  base gold: {str(b['gold_answer'])[:120]!r}")
        print(f"  mix  gold: {str(r['gold_answer'])[:120]!r}")
else:
    # ID 체계가 다르면 길이 분포로라도 비교
    gl_mix = [len(str(r["gold_answer"])) for r in mix_ans]
    gl_base = [len(str(r["gold_answer"])) for r in base]
    print("[주의] ID 겹침 없음 — gold 길이 분포로 간접 비교")
    print(f"gold 길이 평균: base {np.mean(gl_base):.1f} vs mix {np.mean(gl_mix):.1f}")
    print(f"gold 길이 90분위: base {np.percentile(gl_base, 90):.0f} vs mix {np.percentile(gl_mix, 90):.0f}")
    print("\nmix 응답가능 예시 3건의 gold:")
    for r in mix_ans[:3]:
        print(f"  [{r['type']}] {str(r['gold_answer'])[:120]!r}")
    print("base 예시 3건의 gold:")
    for r in base[:3]:
        print(f"  [{r['type']}] {str(r['gold_answer'])[:120]!r}")

# 무응답 예시 형태 확인
if mix_ua:
    print(f"\n무응답 예시 3건 (answerable={mix_ua[0].get('answerable')}):")
    for r in mix_ua[:3]:
        print(f"  gold={str(r.get('gold_answer'))[:80]!r} · question={str(r['question'])[:60]!r}")