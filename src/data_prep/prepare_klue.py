# -*- coding: utf-8 -*-
"""
prepare_klue.py — Phase 2(도메인 축 분리)용 KLUE-MRC 데이터셋 구축.

금융 파이프라인과 **동형**으로 만드는 것이 목적이다. 판정 코드
(evaluate_followup.py)를 한 줄도 바꾸지 않고 돌릴 수 있도록 question_id 규약과
행 스키마를 그대로 따른다.

  easy UA : question_id 가 'unans_' 로 시작   (ua_kind() → "easy")
  hard UA : question_id 에 '-hd-' 포함        (ua_kind() → "hard")
  학습 UA : question_id 에 '-trainua' 접미     (make_train_mix 규약)

설계 결정 (Phase 2 사전등록 §4에 근거 기록):

  [D1] 한국경제(hankyung) 출처 제외 — 기본값.
       KLUE-MRC 원천은 위키피디아·한국경제·ACROFAN 이고, 실사 결과 한국경제가
       train 56.4% · dev 55.8% 를 차지한다(results/klue_audit.json). 한국경제는
       경제·금융 저널리즘이므로 남겨 두면 "금융 도메인을 떠난다"는 Phase 2 의
       전제 자체가 무너진다. --include-hankyung 으로 2차(출처 층화) 분석 가능.

  [D2] train·dev 를 합친 뒤 title 단위로 재분할.
       KLUE 자체 분할은 train·dev 가 title 2,620개를 공유한다(실사 [7]). 금융
       파이프라인의 문서 단위 분리 원칙을 지키려면 재분할이 불가피하다.

  [D3] 답 유형은 전부 text_span.
       KLUE-MRC 는 추출형 span 단일 유형이다. 그래서 붕괴 판별의 비교 기준도
       금융의 4유형 집계(-0.346)가 아니라 text_span 한정값(-0.303)이어야 한다
       (results/comparator_textspan.json).

  [D4] 학습 무응답 예시는 KLUE 의 is_impossible 을 쓰지 않고 **교차 지문**으로 만든다.
       금융에서 학습 무응답은 교차 지문(easy)만 사용했고(make_train_mix.py 주석),
       그 덕분에 hard UA 복원이 "훈련 분포 밖 일반화"로 해석된다. KLUE 의 사람이
       쓴 적대적 무응답을 학습에 넣으면 처치 자체가 달라져 도메인만 바꾼 대조가
       아니게 되고, hard UA 의 OOD 성질도 사라진다.

  [D5] hard UA = KLUE 네이티브 is_impossible (question_type 3).
       금융의 근거제거·대상치환과는 다른 종류의 hard 이므로 복원 깊이를 도메인 간에
       직접 비교하지 않는다. Phase 2 의 1차 종점(붕괴 여부)은 응답가능 문항에서만
       측정되므로 이 비대칭은 주 판정에 영향을 주지 않는다.

  [D6] 응답가능 평가 1,200건 — 금융(1,170)과 동형.
       n=300 으로 줄이면 ΔEM 의 95% CI 반폭이 0.023~0.027 로 넓어져 H2 문턱
       (-0.03)에 걸린다. 실제로 무손실 모델을 금융 text_span(n=300)만으로 판정하면
       Qwen3 2/3 · Llama 1/3 로 떨어진다(comparator 산출). 검정력을 금융과 맞추려면
       1,200건이 필요하다.

사용:
  python src/data_prep/prepare_klue.py --klue-dir <KLUE>/klue_benchmark/klue-mrc-v1.1
출력:
  data/processed/klue_train.jsonl          (응답가능 3,000)
  data/processed/klue_train_mix_r05.jsonl  (3,000 중 150건을 교차지문 UA 로 교체)
  data/processed/klue_eval_full.jsonl      (응답가능 1,200 + easy UA 300 + hard UA 267)
  data/processed/klue_manifest.json        (무결성 검사 결과)
"""
import argparse
import hashlib
import json
import random
import string
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------- 정규화 (prepare_data._norm 과 동일)
_PUNCT = set(string.punctuation) | {"·", "…", "「", "」", "『", "』", "%"}


def _norm(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return "".join(ch for ch in text if not ch.isspace() and ch not in _PUNCT)


def doc_id_of(title):
    return "klue-" + hashlib.sha1(title.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- 로드
def load_klue(klue_dir, exclude_sources, max_ctx_chars):
    """train·dev 를 합쳐 문항 단위로 평탄화한다 (D2)."""
    rows, dropped = [], Counter()
    for fn in ("klue-mrc-v1.1_train.json", "klue-mrc-v1.1_dev.json"):
        p = Path(klue_dir) / fn
        if not p.exists():
            raise SystemExit(f"KLUE 파일 없음: {p}")
        data = json.load(open(p, encoding="utf-8"))
        for art in data["data"]:
            src = art.get("source")
            if src in exclude_sources:
                dropped[f"source:{src}"] += sum(len(pa["qas"]) for pa in art["paragraphs"])
                continue
            for para in art["paragraphs"]:
                ctx = para["context"]
                if max_ctx_chars and len(ctx) > max_ctx_chars:
                    dropped["context_too_long"] += len(para["qas"])
                    continue
                for qa in para["qas"]:
                    golds = []
                    for a in qa.get("answers", []):
                        t = a.get("text", "")
                        if t and t not in golds:
                            golds.append(t)
                    if not qa["is_impossible"] and not golds:
                        dropped["no_gold"] += 1
                        continue
                    rows.append({
                        "guid": qa["guid"],
                        "title": art["title"],
                        "doc_id": doc_id_of(art["title"]),
                        "source": src,
                        "news_category": art.get("news_category"),
                        "klue_qtype": qa["question_type"],
                        "is_impossible": qa["is_impossible"],
                        "context": ctx,
                        "question": qa["question"],
                        "golds": golds,
                    })
    print(f"[로드] {len(rows):,}건 · 제외 {dict(dropped)}")
    return rows


# ---------------------------------------------------------------- 분할 (D2)
def split_by_title(rows, n_eval_ans, n_train_ans, n_hard, rng):
    """title 단위로 평가/학습 문서를 먼저 가르고, 그 안에서 정원을 채운다.

    문서를 섞은 뒤 평가 정원(응답가능 + 무응답)이 찰 때까지 '문서 전체'를 평가측에
    배정한다. 나머지 문서가 학습 풀이 된다 — 같은 title 이 양쪽에 오지 않는다.
    """
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[r["doc_id"]].append(r)
    docs = list(by_doc)
    rng.shuffle(docs)

    eval_docs, train_docs = [], []
    n_a = n_i = 0
    for d in docs:
        grp = by_doc[d]
        if n_a < n_eval_ans or n_i < n_hard:
            eval_docs.append(d)
            n_a += sum(not r["is_impossible"] for r in grp)
            n_i += sum(r["is_impossible"] for r in grp)
        else:
            train_docs.append(d)

    ev = [r for d in eval_docs for r in by_doc[d]]
    tr = [r for d in train_docs for r in by_doc[d]]

    ev_ans = [r for r in ev if not r["is_impossible"]]
    ev_imp = [r for r in ev if r["is_impossible"]]
    tr_ans = [r for r in tr if not r["is_impossible"]]
    rng.shuffle(ev_ans); rng.shuffle(ev_imp); rng.shuffle(tr_ans)

    for name, have, want in (("평가 응답가능", len(ev_ans), n_eval_ans),
                             ("hard UA", len(ev_imp), n_hard),
                             ("학습 응답가능", len(tr_ans), n_train_ans)):
        if have < want:
            raise SystemExit(f"{name} 부족: {have} < {want} — 제외 조건을 완화하거나 정원을 낮출 것")
    return ev_ans[:n_eval_ans], ev_imp[:n_hard], tr_ans[:n_train_ans]


# ---------------------------------------------------------------- 행 변환
def to_answerable(r):
    """gold_answer 를 리스트로 둔다 — 채점기의 max-over-golds 분기가 받는다."""
    return {
        "question_id": r["guid"],
        "doc_id": r["doc_id"],
        "type": "text_span",                      # D3
        "context": r["context"],
        "question": r["question"],
        "gold_answer": r["golds"],                # ★ 복수 정답 (KLUE 응답가능의 35.5%)
        "answerable": True,
        "klue_qtype": r["klue_qtype"],
        "source": r["source"],
    }


def to_hard_ua(r):
    """KLUE 네이티브 무응답 (D5). question_id 에 '-hd-' 를 넣어 ua_kind()가 hard 로 읽게 한다."""
    return {
        "question_id": f"{r['guid']}-hd-na",      # na = native adversarial
        "doc_id": r["doc_id"],
        "type": "unanswerable",
        "ua_kind": "hard_klue_native",
        "orig_type": "text_span",
        "context": r["context"],
        "question": r["question"],
        "gold_answer": "",
        "orig_gold": None,
        "answerable": False,
        "klue_qtype": r["klue_qtype"],
        "source": r["source"],
    }


# ---------------------------------------------------------------- 교차 지문 UA (D4)
def cross_pair(targets, context_pool, rng, id_fn, max_tries=30):
    """금융 make_unanswerable 과 동일한 규칙: 다른 문서의 지문으로 교체하되,
    교체 지문에 정답 문자열이 우연히 들어 있으면 재시도한다.
    KLUE 는 정답이 여럿이므로 **모든** gold 가 부재해야 통과시킨다(더 보수적)."""
    out, failed = [], 0
    for r in targets:
        gold_ns = [g for g in (_norm(x) for x in r["golds"]) if g]
        swapped = None
        for _ in range(max_tries):
            cand = context_pool[rng.randrange(len(context_pool))]
            if cand["doc_id"] == r["doc_id"]:
                continue
            cn = _norm(cand["context"])
            if any(g in cn for g in gold_ns):
                continue
            swapped = cand
            break
        if swapped is None:
            failed += 1
            continue
        out.append({
            "question_id": id_fn(r["guid"]),
            "doc_id": r["doc_id"],
            "type": "unanswerable",
            "orig_type": "text_span",
            "context": swapped["context"],
            "question": r["question"],
            "gold_answer": "",
            "orig_gold": r["golds"][0] if r["golds"] else None,
            "ctx_doc_id": swapped["doc_id"],
            "answerable": False,
            "klue_qtype": r["klue_qtype"],
            "source": r["source"],
        })
    if failed:
        print(f"[경고] 교차 지문 생성 실패 {failed}건")
    return out


def make_train_ua(train_rows, n_replace, rng):
    """학습셋 3,000건 중 n_replace 건을 교차 지문 UA 로 '교체'한다.
    총량·질문 집합·행 순서를 보존하는 make_train_mix.py 의 설계 특성을 그대로 따른다."""
    idx = list(range(len(train_rows)))
    rng.shuffle(idx)
    chosen = sorted(idx[:n_replace])
    pool = [{"doc_id": r["doc_id"], "context": r["context"]} for r in train_rows]
    ua = cross_pair([train_rows[i] for i in chosen], pool, rng,
                    id_fn=lambda g: f"{g}-trainua")
    if len(ua) != n_replace:
        raise SystemExit(f"학습 UA 생성 부족: {len(ua)} < {n_replace}")
    mix = [to_answerable(r) for r in train_rows]
    for pos, row in zip(chosen, ua):
        row_keep = dict(row)
        row_keep["type"] = "text_span"     # 유형 비율 보존 (금융 train_mix 와 동일)
        mix[pos] = row_keep
    return mix, chosen


# ---------------------------------------------------------------- 출력
def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"저장: {path} ({len(rows):,}건)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klue-dir", required=True)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-hankyung", action="store_true",
                    help="D1 해제 — 출처 층화 2차 분석용")
    ap.add_argument("--max-context-chars", type=int, default=0,
                    help="0 = 제한 없음. 학습 전 실제 토크나이저로 길이를 반드시 확인할 것")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-eval-answerable", type=int, default=1200)
    ap.add_argument("--n-easy-ua", type=int, default=300)
    ap.add_argument("--n-hard-ua", type=int, default=267)
    ap.add_argument("--n-train-ua", type=int, default=150)   # r05
    args = ap.parse_args()

    rng = random.Random(args.seed)
    exclude = set() if args.include_hankyung else {"hankyung"}
    print(f"[D1] 제외 출처: {exclude or '없음 (2차 분석 모드)'}")

    rows = load_klue(args.klue_dir, exclude, args.max_context_chars)
    ev_ans, ev_imp, tr_ans = split_by_title(
        rows, args.n_eval_answerable, args.n_train, args.n_hard_ua, rng)

    # --- 문서 누출 검증 (금융 prepare_data 의 assert 와 동일 원칙)
    tr_docs = {r["doc_id"] for r in tr_ans}
    ev_docs = {r["doc_id"] for r in ev_ans} | {r["doc_id"] for r in ev_imp}
    overlap = tr_docs & ev_docs
    assert not overlap, f"문서 누출 발생: {len(overlap)}개"
    print(f"[검증] 문서 누출 없음 (학습 {len(tr_docs):,}문서 · 평가 {len(ev_docs):,}문서)")

    # --- easy UA: 평가측 응답가능 문항을 평가측 지문끼리 교차 (금융과 동일 규칙)
    ev_pool = [{"doc_id": r["doc_id"], "context": r["context"]} for r in ev_ans]
    easy_src = ev_ans[:]
    rng.shuffle(easy_src)
    easy = cross_pair(easy_src[:int(args.n_easy_ua * 1.3)], ev_pool, rng,
                      id_fn=lambda g: f"unans_{g}")[:args.n_easy_ua]
    if len(easy) < args.n_easy_ua:
        raise SystemExit(f"easy UA 부족: {len(easy)} < {args.n_easy_ua}")

    hard = [to_hard_ua(r) for r in ev_imp]
    eval_full = [to_answerable(r) for r in ev_ans] + easy + hard

    # --- 학습셋 / 혼입셋
    train = [to_answerable(r) for r in tr_ans]
    mix, replaced = make_train_ua(tr_ans, args.n_train_ua, rng)

    out = Path(args.out_dir)
    write_jsonl(out / "klue_train.jsonl", train)
    write_jsonl(out / "klue_train_mix_r05.jsonl", mix)
    write_jsonl(out / "klue_eval_full.jsonl", eval_full)

    # --- 매니페스트 / 무결성
    n_multi = sum(len(r["gold_answer"]) > 1 for r in eval_full if r["answerable"])
    ctx_len = sorted(len(r["context"]) for r in eval_full)
    manifest = {
        "seed": args.seed,
        "excluded_sources": sorted(exclude),
        "counts": {
            "train": len(train), "train_mix_r05": len(mix),
            "train_ua_replaced": len(replaced),
            "eval_answerable": len(ev_ans), "eval_easy_ua": len(easy),
            "eval_hard_ua": len(hard), "eval_total": len(eval_full),
        },
        "doc_leakage": len(overlap),
        "eval_multi_gold": n_multi,
        "eval_context_chars": {"median": ctx_len[len(ctx_len) // 2],
                               "p90": ctx_len[int(0.9 * len(ctx_len))],
                               "max": ctx_len[-1]},
        "source_mix_eval": dict(Counter(r.get("source") for r in eval_full)),
        "klue_qtype_eval": dict(Counter(r.get("klue_qtype") for r in eval_full)),
        "id_conventions": {"easy": "unans_*", "hard": "*-hd-na", "train_ua": "*-trainua"},
    }
    json.dump(manifest, open(out / "klue_manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("\n=== 구성 ===")
    print(f"  학습 응답가능      {len(train):,}")
    print(f"  학습 혼입 r05      {len(mix):,} (무응답 교체 {len(replaced)}건, 총량·질문집합 불변)")
    print(f"  평가 응답가능      {len(ev_ans):,}  (복수 정답 {n_multi:,}건)")
    print(f"  평가 easy UA       {len(easy):,}  (교차 지문)")
    print(f"  평가 hard UA       {len(hard):,}  (KLUE 네이티브 is_impossible)")
    print(f"  평가 합계          {len(eval_full):,}")
    print(f"  지문 길이(문자)    중앙 {manifest['eval_context_chars']['median']} · "
          f"p90 {manifest['eval_context_chars']['p90']} · 최대 {manifest['eval_context_chars']['max']}")
    print(f"\n저장: {out / 'klue_manifest.json'}")
    print("\n★ 학습 전 확인: 실제 토크나이저로 (시스템프롬프트+지문+질문+타깃) 길이가")
    print("  max_seq_len 2048 을 넘지 않는지 표본 검사할 것. 넘으면 --max-context-chars 로 제한.")


if __name__ == "__main__":
    main()
