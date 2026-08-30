# -*- coding: utf-8 -*-
"""make_natural_ua.py — 검색 불일치형(retrieval-mismatch) UA 후보 생성.

원리: eval_full_v2의 응답가능 질문(자연 질문, 학습 미노출)을 같은 유형의
다른 문서 지문 중 '주제 유사도 최상위 + 정답 문자열 부재' 지문과 교차 매칭.
easy UA(무작위 교차)와의 차별점: (1) 최유사 지문 매칭 (2) 전수 검수 (3) easy UA
기사용 질문 제외. hard UA(규칙 편집)와의 차별점: 질문·지문 무편집.

절차:
  python src/data_prep/make_natural_ua.py            # 후보 생성 → 검수 파일
  (natua_review.tsv 를 열어 keep 열에 O/X 전수 검수)
  python src/data_prep/make_natural_ua.py --finalize # 통과분으로 eval_natua.jsonl 확정
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

EVAL = Path("data/processed/eval_full_v2.jsonl")
CAND = Path("data/processed/natua_candidates.jsonl")
REVIEW = Path("data/processed/natua_review.tsv")
OUT = Path("data/processed/eval_natua.jsonl")

N_PER_TYPE = 75          # 유형별 후보 수 (총 300 → 검수 후 100~150 목표)
SEED = 42


def norm(t):
    return re.sub(r"\s+", "", str(t)).lower()


def tokens(t):
    """공백 토큰 + 문자 2-gram — 형태소 분석기 없이 쓰는 경량 유사도 재료."""
    t = str(t)
    ws = set(t.split())
    bg = {t[i:i + 2] for i in range(len(t) - 1) if not t[i:i + 2].isspace()}
    return ws | bg


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def build_candidates():
    rows = [json.loads(l) for l in open(EVAL, encoding="utf-8")]
    ans = [r for r in rows if r.get("gold_answer") not in (None, "")]  # 응답가능만

    # easy UA로 이미 쓰인 질문 제외 (세트 간 독립성).
    # ID 체계가 아닌 질문 텍스트로 매칭: 데이터 계보상 easy UA ID가
    # 재포맷된 흔적이 있어(-unans 접미 → unans_ 접두) 텍스트 매칭이 안전.
    ua_rows = [r for r in rows if r.get("gold_answer") in (None, "")]
    easy_questions = {norm(r["question"]) for r in ua_rows
                      if "-hd-" not in r["question_id"]}
    before = len(ans)
    ans = [r for r in ans if norm(r["question"]) not in easy_questions]
    print(f"easy UA 기사용 질문 제외: {before} → {len(ans)}건")

    by_type = defaultdict(list)
    for r in ans:
        by_type[r["type"]].append(r)

    rng = random.Random(SEED)
    cands = []
    for qtype, items in by_type.items():
        toks = {r["question_id"]: tokens(r["question"]) for r in items}
        picked = rng.sample(items, min(N_PER_TYPE, len(items)))
        for q in picked:
            gold_n = norm(q["gold_answer"])
            best, best_sim = None, -1.0
            for c in items:
                if c["doc_id"] == q["doc_id"]:
                    continue                          # 같은 문서 제외
                ctx = c["context"]
                if gold_n and gold_n in norm(ctx):
                    continue                          # 정답 문자열 포함 지문 제외
                sim = jaccard(toks[q["question_id"]], tokens(ctx[:800]))
                if sim > best_sim:
                    best, best_sim = c, sim
            if best is None:
                continue
            cands.append({
                "question_id": f"natua_{qtype}_{q['question_id']}",
                "type": "unanswerable",
                "orig_type": qtype,
                "question": q["question"],
                "context": best["context"],
                "doc_id": best["doc_id"],
                "gold_answer": None,
                "src_question_id": q["question_id"],
                "src_gold_answer": q["gold_answer"],   # 검수용 (최종본에서 제거)
                "topic_sim": round(best_sim, 4),
            })
    # 유사도 상위 순 정렬 (주제 근접 후보 우선 검수)
    cands.sort(key=lambda r: -r["topic_sim"])

    with open(CAND, "w", encoding="utf-8") as f:
        for r in cands:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 검수 파일: 엑셀에서 열어 keep 열에 O/X 기입 (탭 구분 유지하여 저장)
    with open(REVIEW, "w", encoding="utf-8") as f:
        f.write("keep\tquestion_id\ttopic_sim\torig_type\tquestion\t"
                "src_gold(참고: 이 답이 지문에 없어야 함)\tcontext_head\n")
        for r in cands:
            f.write("\t".join([
                "", r["question_id"], str(r["topic_sim"]), r["orig_type"],
                r["question"].replace("\t", " "),
                str(r["src_gold_answer"]).replace("\t", " ")[:60],
                r["context"].replace("\t", " ").replace("\n", " ")[:200],
            ]) + "\n")
    print(f"후보 {len(cands)}건 생성 → {CAND}")
    print(f"검수 파일 → {REVIEW}")
    print("검수 기준: ① 질문이 자연스러움 ② 지문 주제가 질문과 유관 "
          "③ 지문에 답·부분답·의역답이 정말 없음 ④ 애매하면 X")


def finalize():
    pairs = [
        (REVIEW, CAND),
        (Path("data/processed/natua_review_yesno2.tsv"),
         Path("data/processed/natua_candidates_yesno2.jsonl")),
    ]
    keep_ids, cands = set(), []
    for review_path, cand_path in pairs:
        if not review_path.exists() or not cand_path.exists():
            print(f"[건너뜀] {review_path.name} 또는 {cand_path.name} 없음")
            continue
        raw = open(review_path, "rb").read()
        for enc in ("utf-8-sig", "utf-16", "cp949"):
            try:
                text = raw.decode(enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        n_o = 0
        for line in text.splitlines()[1:]:
            cols = line.split("\t")
            if cols and cols[0].strip().upper() in ("O", "1", "Y", "KEEP"):
                keep_ids.add(cols[1].strip())
                n_o += 1
        cands += [json.loads(l) for l in open(cand_path, encoding="utf-8")]
        print(f"{review_path.name}: O {n_o}건")

    final = [r for r in cands if r["question_id"] in keep_ids]
    for r in final:
        r.pop("src_gold_answer", None)
        r.pop("src_question_id", None)
        r.pop("topic_sim", None)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"확정 {len(final)}건 → {OUT}")
    print("유형 분포:", dict(Counter(r["orig_type"] for r in final)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    finalize() if args.finalize else build_candidates()