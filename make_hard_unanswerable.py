# -*- coding: utf-8 -*-
"""
make_hard_unanswerable.py — hard unanswerable 평가 문항 생성기 (설계서 5.2절)

기존 easy UA(교차 지문, prepare_data.make_unanswerable)와 달리 '같은 문서 내
근접 부재' 방식으로, 주제는 일치하지만 답만 없는 어려운 응답불가능 문항을 만든다.

  (a) 근거 제거형 (hd-er): 원문항 지문에서 정답 근거 문장(들)만 제거.
      기계 검증 — 제거 후 정답 문자열·수치 변형의 잔존 0건.
  (b) 대상 치환형 (hd-ts): 질문의 연도·분기·월·순위·개체를 같은 지문에
      등장하지 않는 것으로 치환. 기계 검증 — 원대상은 지문에 존재했고,
      치환 대상은 지문에 부재.

파일럿 워크플로 (설계서 단계 1):
  1) python make_hard_unanswerable.py --pilot 20
       → data/processed/hard_ua_pilot.jsonl + hard_ua_pilot_review.txt (전건 검수용)
  2) 검수(A/B/C)에서 C(지문만으로 답 가능) 발견 시 아래 TODO 규칙 보강 후 재생성
  3) 검수 기준 확정 후: python make_hard_unanswerable.py --n 300
       → hard_ua.jsonl + hard_ua_review.txt (무작위 50건 검수용)

검수 판정 범례 (unans_review.txt 관례 유지):
  A = 응답 불가능 확실 (합격)  /  B = 경계 사례 (재논의)  /  C = 지문만으로 답 가능 (불합격)
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from prepare_data import _norm  # 검증 정규화 — 기존 무결성 검사와 동일 원칙

# --------------------------------------------------------------- 생성 파라미터
MAX_REMOVE_FRAC = 0.5       # ER: 지문의 이 비율 초과 제거되면 폐기 (문맥 붕괴 방지)
MIN_CONTEXT_CHARS = 200     # ER: 제거 후 지문 최소 길이
ENTITY_SUFFIXES = (         # TS: 개체 후보 인식용 접미(금융 도메인 휴리스틱)
    # 기업형
    "은행", "증권", "카드", "보험", "생명", "화재", "자산운용", "캐피탈",
    "저축은행", "그룹", "전자", "페이", "금융지주",
    # 기관형 (text_span·yes_no 질문의 주어 커버)
    "위원회", "거래소", "감독원", "협회", "공사", "공단", "연구원",
    "진흥원", "결제원", "금융공사",
)
# ER 잔존 검사 보강용 — 파일럿 검수 결과 반영
ALIAS_RE = re.compile(r"([가-힣A-Za-z]{4,})\s*\(\s*([가-힣A-Za-z]{2,8})\s*\)")
PERSON_TITLES = ("CFO", "CEO", "대표", "회장", "부회장", "사장", "전무", "상무",
                 "이사", "위원장", "총재", "장관", "씨", "연구원", "애널리스트",
                 "본부장", "팀장")
COMPARATIVE_RE = re.compile(r"중\s|더 (?:적|많|크|작|높|낮)|가장\s")
# 불릿 계층: 헤더 삭제 시 하위 항목 동반 삭제용 (파일럿 [15] 결함 대응)
BULLET_LEVEL = {"□": 0, "○": 1, "ㅇ": 1, "◆": 1, "•": 1, "￭": 2, "-": 2, "–": 2}
YEAR_RE = re.compile(r"(?:19|20)\d{2}(?=년)")
QUARTER_RE = re.compile(r"[1-4](?=분기)")
MONTH_RE = re.compile(r"(?<![0-9])(?:1[0-2]|[1-9])(?=월)")
RANK_RE = re.compile(r"[0-9]+(?=(?:순)?위)")
ENTITY_RE = re.compile(
    r"[가-힣A-Za-z0-9&]+(?:" + "|".join(ENTITY_SUFFIXES) + r")"
)


# ------------------------------------------------------------------- 공통 유틸
def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gold_variants(gold):
    """정답 잔존 검사용 변형 집합. 수치형은 콤마 제거·숫자열도 검사.
    TODO(검수 후): 검수에서 잔존 패턴 발견 시 변형 규칙 추가 (단위 환산 등)"""
    g = str(gold).strip()
    out = {_norm(g)}
    digits = re.sub(r"[^\d.]", "", g)
    if digits and any(ch.isdigit() for ch in digits):
        out.add(_norm(digits))
    return {v for v in out if v}


def split_segments(context):
    """지문을 근거 제거 단위로 분할: 줄 → 문장(종결부 기준).
    표 형태 줄(table_lookup 지문)은 줄 단위 그대로 유지된다."""
    segs = []
    for line in context.split("\n"):
        line = line.rstrip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?。])\s+", line)
        segs.extend(p for p in parts if p.strip())
    return segs


# ------------------------------------------------------- (a) 근거 제거형 (ER)
def make_evidence_removal(ex):
    """정답 근거 문장(들)을 제거한 지문을 만든다. 실패 시 (None, 사유).

    파일럿 검수 반영 보강:
      - 비교형 질문 배제([06]): 한쪽 피연산자만 제거해도 잔여 서사로 복원 가능
      - 약칭 수확([02]): 지문의 '전체명(약칭)' 병기에서 gold의 약칭을 잔존 검사에 추가
      - 인명 축약([05]): gold가 인명이면 성+직함 조합을 잔존 검사에 추가
      - gold 토큰([14]): 다단어 gold는 구성 토큰(2자 이상)도 잔존 검사 — 의역 잔존 차단
      - 불릿 계층([15]): 헤더 문장 삭제 시 하위 불릿 동반 삭제 — 오귀속 차단
    """
    # 비교형 질문은 ER 부적격
    if COMPARATIVE_RE.search(ex["question"]):
        return None, "comparative_question"

    gold_raw = str(ex["gold_answer"]).strip()
    gold_n = _norm(gold_raw)
    variants = gold_variants(gold_raw)
    # (1) '전체명(약칭)' 병기에서 약칭·전체명 수확
    for m in ALIAS_RE.finditer(ex["context"]):
        full, alias = _norm(m.group(1)), _norm(m.group(2))
        if full and alias and (full in gold_n or gold_n in full):
            variants |= {full, alias}
    # (2) 인명 gold의 축약 지칭(성+직함/씨)
    name_m = re.match(r"([가-힣])[가-힣]{1,2}(?=\s|$)", gold_raw)
    if name_m:
        variants |= {_norm(name_m.group(1) + t) for t in PERSON_TITLES}
    # (3) 다단어 gold의 구성 토큰 (의역 잔존 차단, 과잉 제거는 폐기로 흡수)
    if " " in gold_raw:
        variants |= {_norm(tok) for tok in gold_raw.split() if len(_norm(tok)) >= 2}
    variants = {v for v in variants if v}

    segs = split_segments(ex["context"])
    if len(segs) < 3:
        return None, "segments_too_few"

    hit = set(i for i, s in enumerate(segs)
              if any(v in _norm(s) for v in variants))
    if not hit:
        return None, "gold_not_located"
    # (4) 불릿 헤더가 삭제되면 그 하위 항목도 함께 삭제 (고아 불릿 오귀속 방지)
    changed = True
    while changed:
        changed = False
        for i in sorted(hit):
            lv = BULLET_LEVEL.get(segs[i][:1])
            if lv is None:
                continue
            j = i + 1
            while j < len(segs):
                lv_j = BULLET_LEVEL.get(segs[j][:1])
                if lv_j is None or lv_j <= lv:
                    break
                if j not in hit:
                    hit.add(j)
                    changed = True
                j += 1
    if len(hit) / len(segs) > MAX_REMOVE_FRAC:
        return None, "removal_too_large"

    kept = [s for i, s in enumerate(segs) if i not in hit]
    new_ctx = "\n".join(kept)
    if len(new_ctx) < MIN_CONTEXT_CHARS:
        return None, "context_too_short"
    if any(v in _norm(new_ctx) for v in variants):
        return None, "gold_residual"
    return {
        "context": new_ctx,
        "removed_segments": [segs[i] for i in sorted(hit)],
    }, None


# ------------------------------------------------------- (b) 대상 치환형 (TS)
def _swap_candidates(kind, orig):
    """치환 후보 생성. TODO(검수 후): 후보 폭·현실성 규칙 보강"""
    if kind == "year":
        y = int(orig)
        return [str(y + d) for d in (-3, -2, -1, 1, 2, 3) if 1995 <= y + d <= 2030]
    if kind == "quarter":
        return [q for q in "1234" if q != orig]
    if kind == "month":
        return [str(m) for m in range(1, 13) if str(m) != orig]
    if kind == "rank":
        r = int(orig)
        return [str(r + d) for d in (1, 2, 3, -1, -2) if r + d >= 1 and str(r + d) != orig]
    return []


def find_swap_targets(question):
    """질문에서 치환 가능한 대상을 우선순위대로 수집."""
    targets = []
    for kind, pat in (("year", YEAR_RE), ("quarter", QUARTER_RE),
                      ("month", MONTH_RE), ("rank", RANK_RE)):
        m = pat.search(question)
        if m:
            targets.append((kind, m.group(), m.span()))
    m = ENTITY_RE.search(question)
    if m:
        targets.append(("entity", m.group(), m.span()))
    return targets


def make_target_swap(ex, entity_pool, rng):
    """질문의 대상 하나를 지문에 없는 것으로 치환. 실패 시 (None, 사유).

    파일럿 검수 반영: 같은 값의 전체 출현을 일괄 치환([03] — 첫 출현만 바꾸면
    답을 결정하는 두 번째 출현이 남아 응답 가능해질 수 있음).
    """
    ctx_n = _norm(ex["context"])
    targets = find_swap_targets(ex["question"])
    if not targets:
        return None, "no_swappable_target"

    unit = {"year": "년", "quarter": "분기", "month": "월", "rank": "위"}
    rng.shuffle(targets)
    for kind, orig, (s, e) in targets:
        if _norm(orig) not in ctx_n:
            continue  # 원대상이 지문에 존재해야 '근접 부재' 성립
        if kind == "entity":
            cands = [c for c in entity_pool
                     if c != orig and c not in ex["context"]]
            rng.shuffle(cands)
            cands = cands[:10]
        else:
            cands = _swap_candidates(kind, orig)
            rng.shuffle(cands)
        for cand in cands:
            if _norm(cand) in ctx_n:
                continue
            if kind == "entity":
                new_q = ex["question"].replace(orig, cand)      # 전체 출현 치환
            else:
                new_q = re.sub(re.escape(orig) + f"(?={unit[kind]})",
                               cand, ex["question"])            # 단위 결합 전체 치환
            if new_q == ex["question"]:
                continue
            return {
                "question": new_q,
                "swap": {"kind": kind, "from": orig, "to": cand},
            }, None
    return None, "no_valid_candidate"


# ----------------------------------------------------------------- 파이프라인
def build_source_pool(eval_rows, excluded_ids, reuse_ok=()):
    """응답가능 문항 중 소스 풀 구성: 손상 gold·easy UA 재사용 문항 제외.
    reuse_ok에 지정된 유형은 easy UA 재사용을 허용(소스 고갈 시 예비책)."""
    easy_ua_qids = set()
    easy_ua_questions = set()
    for r in eval_rows:
        if r.get("type") == "unanswerable":
            qid = r["question_id"]
            easy_ua_qids.add(qid.removesuffix("-unans").removeprefix("unans_"))
            easy_ua_questions.add(r["question"])

    pool, dropped = [], Counter()
    for r in eval_rows:
        if r.get("type") == "unanswerable":
            continue
        if r["question_id"] in excluded_ids:
            dropped["excluded_gold"] += 1
            continue
        if (r["question_id"] in easy_ua_qids
                or r["question"] in easy_ua_questions):
            if r["type"] not in reuse_ok:
                dropped["easy_ua_reuse"] += 1
                continue
            dropped["easy_ua_reuse_allowed"] += 1
            r = dict(r, _easy_reused=True)      # 최후순위 사용 + 감사 표식
        pool.append(r)
    return pool, dropped


def build_entity_pool(eval_rows):
    """교차 문서 개체 풀: 다른 문항의 질문·지문에서 개체 후보 수집."""
    ents = Counter()
    for r in eval_rows:
        for m in ENTITY_RE.finditer(r.get("question", "")):
            ents[m.group()] += 1
    # 너무 희귀한 것(오탐 가능성)과 과도하게 흔한 것 제외
    return [e for e, c in ents.items() if 1 <= c <= 50]


def generate(pool, entity_pool, n_target, rng):
    """유형×방식 층화 생성. 반환: (rows, fail_counter)"""
    by_type = defaultdict(list)
    for ex in pool:
        by_type[ex["type"]].append(ex)
    # 유형별 허용 방식 (파일럿 검수 반영):
    #  - yes_no 제외: 폐쇄형 질문은 전제 위조 시 '무응답'이 아니라 '아니오'가
    #    규범적 정답이 되어(파일럿 [17]~[20]) UA 라벨의 타당도를 훼손
    #  - text_span: TS 부적격(치환 대상 부족) → ER 전용
    #  - numeric·table: ER+TS (numeric ER은 정답 직접 등장 문항 한정 — 자동 처리)
    TYPE_KINDS = {"text_span": ("er",),
                  "table_lookup": ("er", "ts"),
                  "numeric_reasoning": ("er", "ts")}
    types = [t for t in sorted(by_type) if t in TYPE_KINDS]
    quota = {(t, k): 0 for t in types for k in ("er", "ts")}
    per_type = n_target // len(types)
    rem_t = n_target - per_type * len(types)
    for t in types:
        alloc = per_type + (1 if rem_t > 0 else 0)
        rem_t -= 1
        kinds = TYPE_KINDS[t]
        per_kind = alloc // len(kinds)
        for j, k in enumerate(kinds):
            quota[(t, k)] = per_kind + (1 if j < alloc - per_kind * len(kinds) else 0)

    rows, fails = [], Counter()
    for t in types:
        cand = by_type[t][:]
        rng.shuffle(cand)
        cand.sort(key=lambda ex: ex.get("_easy_reused", False))  # 재사용은 최후순위
        made = {"er": 0, "ts": 0}
        kinds = TYPE_KINDS[t]
        type_total = sum(quota[(t, k)] for k in ("er", "ts"))
        for ex in cand:
            if sum(made.values()) >= type_total:
                break
            # 방식 균형(quota)을 소프트 목표로 우선 시도하되,
            # 한쪽 소스 고갈 시 다른 방식이 유형 총량까지 이월 충당
            order = sorted(kinds, key=lambda k: made[k] - quota[(t, k)])
            placed = False
            for k in order:
                if k == "er":
                    res, err = make_evidence_removal(ex)
                    if err:
                        fails[f"er:{err}"] += 1
                        continue
                    rows.append(_make_row(ex, "hd-er", "hard_evidence_removal",
                                          context=res["context"],
                                          gen_meta={"removed_segments":
                                                    res["removed_segments"]}))
                else:
                    res, err = make_target_swap(ex, entity_pool, rng)
                    if err:
                        fails[f"ts:{err}"] += 1
                        continue
                    rows.append(_make_row(ex, "hd-ts", "hard_target_swap",
                                          question=res["question"],
                                          gen_meta={"swap": res["swap"],
                                                    "orig_question":
                                                    ex["question"]}))
                made[k] += 1
                placed = True
                break
            if not placed:
                fails["unusable_source"] += 1
        if sum(made.values()) < type_total:
            fails[f"quota_short:{t}"] += type_total - sum(made.values())
    return rows, fails


def _make_row(ex, suffix, ua_kind, context=None, question=None, gen_meta=None):
    return {
        "question_id": f"{ex['question_id']}-{suffix}",
        "doc_id": ex["doc_id"],                  # 같은 문서 (근접 부재 방식)
        "type": "unanswerable",
        "ua_kind": ua_kind,
        "orig_type": ex["type"],
        "context": context if context is not None else ex["context"],
        "question": question if question is not None else ex["question"],
        "gold_answer": "",
        "orig_gold": ex["gold_answer"],
        "answerable": False,
        "easy_ua_reused": bool(ex.get("_easy_reused")),
        "gen_meta": gen_meta or {},
    }


# ------------------------------------------------------------- 검수 파일 출력
def write_review(rows, path, n_sample, rng):
    sample = rows if len(rows) <= n_sample else rng.sample(rows, n_sample)
    with open(path, "w", encoding="utf-8") as f:
        f.write("판정 범례: A=응답불가 확실 / B=경계(재논의) / C=지문만으로 답 가능(불합격)\n")
        f.write("합격 기준(설계서 5.2절): C 0건\n\n")
        for i, r in enumerate(sample, 1):
            f.write("=" * 70 + "\n")
            f.write(f"[{i:02d}] QID: {r['question_id']}  |  원유형: {r['orig_type']}"
                    f"  |  방식: {r['ua_kind']}\n")
            f.write(f"질문: {r['question']}\n")
            if r["ua_kind"] == "hard_target_swap":
                sw = r["gen_meta"]["swap"]
                f.write(f"원질문: {r['gen_meta']['orig_question']}\n")
                f.write(f"치환: [{sw['kind']}] {sw['from']} → {sw['to']}\n")
            f.write(f"원정답(이 지문에서 도출 불가여야 정상): {r['orig_gold']}\n")
            if r["ua_kind"] == "hard_evidence_removal":
                f.write("--- 제거된 근거 문장 ---\n")
                for s in r["gen_meta"]["removed_segments"]:
                    f.write(f"  [삭제] {s}\n")
            f.write("--- 지문 (수정 후) ---\n")
            f.write(r["context"][:2500] + ("\n...(생략)" if len(r["context"]) > 2500 else "") + "\n")
            f.write("판정: [   ]   ← A/B/C 기입\n")
    print(f"검수 파일 저장: {path} ({len(sample)}건)")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--pilot", type=int, default=0,
                    help="파일럿 건수 (지정 시 전건 검수 파일 생성)")
    ap.add_argument("--n", type=int, default=300, help="본생성 건수")
    ap.add_argument("--review-sample", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rng = random.Random(cfg["seed"])
    proc = Path(cfg["paths"]["processed_dir"])
    res_dir = Path(cfg["paths"]["results_dir"])

    eval_rows = load_jsonl(proc / "eval.jsonl")
    excluded = set()
    for fname in ("excluded_gold_v2.json", "excluded_gold_v2_manual.json"):
        p = res_dir / fname
        if p.exists():
            excluded |= set(json.loads(p.read_text(encoding="utf-8"))["question_ids"])
        else:
            print(f"[주의] {fname} 없음 — 해당 제외 미적용")

    pool, dropped = build_source_pool(eval_rows, excluded, reuse_ok=("numeric_reasoning",))
    entity_pool = build_entity_pool(eval_rows)
    print(f"소스 풀: {len(pool)}건 (제외: {dict(dropped)}) · "
          f"개체 풀: {len(entity_pool)}종")

    n_target = args.pilot if args.pilot else args.n
    rows, fails = generate(pool, entity_pool, n_target, rng)

    # ---- 무결성 요약 (설계서 5.2절 완료 기준의 기계 검증부) ----
    dup = [q for q, c in Counter(r["question_id"] for r in rows).items() if c > 1]
    print(f"\n생성: {len(rows)}건 / 목표 {n_target}건")
    print(f"  방식×유형: {dict(Counter((r['orig_type'], r['ua_kind'][:14]) for r in rows))}")
    print(f"  생성 실패 사유: {dict(fails) or '없음'}")
    print(f"  question_id 중복: {len(dup)}건 {'⚠' if dup else '✓'}")

    tag = "hard_ua_pilot" if args.pilot else "hard_ua"
    out_path = proc / f"{tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"저장: {out_path}")

    n_review = len(rows) if args.pilot else args.review_sample
    write_review(rows, proc / f"{tag}_review.txt", n_review, rng)

    if args.pilot:
        print("\n[다음] 검수 파일 전건에 A/B/C 기입 → C 발견 시 해당 생성 규칙"
              "(gold_variants / _swap_candidates 등 TODO 표시부) 보강 후 재생성"
              " → C 0건 확인 후 --n 300 본생성")


if __name__ == "__main__":
    main()