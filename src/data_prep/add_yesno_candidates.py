# -*- coding: utf-8 -*-
"""add_yesno_candidates.py — 검색 불일치형 UA의 yes_no 후보 추가 생성.

1차 생성(make_natural_ua.py)에서 yes_no 채택이 3건에 그친 문제 보완:
  - 표본 75개 → 가용 yes_no 질문 전체 사용
  - 질문당 최유사 1개 → 상위 3개 지문까지 후보로 확장
  - 기존 후보(natua_candidates.jsonl)와 동일 쌍 자동 제외
출력: natua_candidates_yesno2.jsonl + natua_review_yesno2.tsv (검수용)

사용: python src/data_prep/add_yesno_candidates.py
검수 후 확정 시 make_natural_ua.py --finalize 가 두 후보 파일을 함께 읽도록
아래 finalize 패치 참고(파일 하단 주석).
"""
import json
import re
from pathlib import Path

EVAL = Path("data/processed/eval_full_v2.jsonl")
CAND1 = Path("data/processed/natua_candidates.jsonl")
CAND2 = Path("data/processed/natua_candidates_yesno2.jsonl")
REVIEW2 = Path("data/processed/natua_review_yesno2.tsv")

TOP_K = 3


def norm(t):
    return re.sub(r"\s+", "", str(t)).lower()


def tokens(t):
    t = str(t)
    ws = set(t.split())
    bg = {t[i:i + 2] for i in range(len(t) - 1) if not t[i:i + 2].isspace()}
    return ws | bg


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


rows = [json.loads(l) for l in open(EVAL, encoding="utf-8")]
ans = [r for r in rows if r.get("gold_answer") not in (None, "")]

# easy UA 기사용 질문 제외 (1차와 동일 규칙)
ua_rows = [r for r in rows if r.get("gold_answer") in (None, "")]
easy_q = {norm(r["question"]) for r in ua_rows if "-hd-" not in r["question_id"]}
yn = [r for r in ans if r["type"] == "yes_no" and norm(r["question"]) not in easy_q]
print(f"yes_no 가용 질문: {len(yn)}건 (easy UA 기사용 제외 후)")

# 기존 후보와의 중복 쌍 방지: (src_question_id, doc_id) 집합
existing = set()
if CAND1.exists():
    for r in map(json.loads, open(CAND1, encoding="utf-8")):
        existing.add((r.get("src_question_id"), r.get("doc_id")))

toks_q = {r["question_id"]: tokens(r["question"]) for r in yn}
cands = []
for q in yn:
    gold_n = norm(q["gold_answer"])
    scored = []
    for c in yn:
        if c["doc_id"] == q["doc_id"]:
            continue
        if gold_n and gold_n in norm(c["context"]):
            continue
        scored.append((jaccard(toks_q[q["question_id"]], tokens(c["context"][:800])), c))
    scored.sort(key=lambda x: -x[0])
    for rank, (sim, c) in enumerate(scored[:TOP_K], 1):
        if (q["question_id"], c["doc_id"]) in existing:
            continue
        cands.append({
            "question_id": f"natua_yes_no_{q['question_id']}_k{rank}",
            "type": "unanswerable",
            "orig_type": "yes_no",
            "question": q["question"],
            "context": c["context"],
            "doc_id": c["doc_id"],
            "gold_answer": None,
            "src_question_id": q["question_id"],
            "src_gold_answer": q["gold_answer"],
            "topic_sim": round(sim, 4),
        })

cands.sort(key=lambda r: -r["topic_sim"])
with open(CAND2, "w", encoding="utf-8") as f:
    for r in cands:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(REVIEW2, "w", encoding="utf-8") as f:
    f.write("keep\tquestion_id\ttopic_sim\torig_type\tquestion\t"
            "src_gold(참고)\tcontext_head\n")
    for r in cands:
        f.write("\t".join([
            "", r["question_id"], str(r["topic_sim"]), "yes_no",
            r["question"].replace("\t", " "),
            str(r["src_gold_answer"])[:60],
            r["context"].replace("\t", " ").replace("\n", " ")[:200],
        ]) + "\n")
print(f"추가 후보 {len(cands)}건 → {CAND2}")
print(f"검수 파일 → {REVIEW2}")
print("검수 유의: 상식으로 답 가능한 일반 명제(예: '중앙은행은 물가안정이 목표다')는 X — "
      "특정 문서·수치·기관에 결부된 진위 질문만 O")

# ── finalize 패치 안내 ─────────────────────────────────────────────
# make_natural_ua.py의 finalize()에서 후보 로드를 아래처럼 두 파일 합산으로 교체:
#   cands = [json.loads(l) for l in open(CAND, encoding="utf-8")]
#   p2 = Path("data/processed/natua_candidates_yesno2.jsonl")
#   if p2.exists():
#       cands += [json.loads(l) for l in open(p2, encoding="utf-8")]
# 그리고 검수 파일도 natua_review.tsv와 natua_review_yesno2.tsv 두 개를 읽어
# keep_ids를 합치면 된다 (REVIEW 순회 블록을 두 파일 루프로).