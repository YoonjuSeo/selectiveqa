# -*- coding: utf-8 -*-
"""
audit_klue.py — Phase 2 실행 전 KLUE-MRC 데이터 실사 (결과가 아니라 데이터 구조 확인).

이 스크립트는 모델을 전혀 돌리지 않는다. 사전등록 전에 수행해도 선택적 배제 위험이
없으며, prepare_klue.py 의 설계 결정(출처 제외, 평가셋 크기, 무응답 정의)의 근거를
수치로 남기는 것이 목적이다.

확인 항목:
  1. 스키마 — question_type / is_impossible / 복수 정답 / guid
  2. **출처 교란**: KLUE-MRC 원천은 위키피디아 · 한국경제 · ACROFAN 이다.
     한국경제는 경제·금융 저널리즘이므로 "금융 도메인을 떠난다"는 Phase 2 전제와
     직접 충돌한다. 출처별·카테고리별 분포를 세어 제외 규모를 정한다.
  3. 응답가능 문항 수 — 평가셋 1,200건(금융과 동형)과 학습 3,000건이 가능한가
  4. 지문 길이 — max_seq_len 2048 안에 들어가는가 (초과분은 제외 대상)
  5. 정답 길이 — 금융 평가셋의 gold 문장형 제외 규칙과 비교
  6. doc(title) 단위 분리 가능성 — 학습·평가 문서 겹침 방지

사용:
  python src/data_prep/audit_klue.py --klue-dir <KLUE repo>/klue_benchmark/klue-mrc-v1.1
출력:
  results/klue_audit.json + 콘솔 요약
"""
import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load_split(path):
    """KLUE-MRC JSON(SQuAD 형식)을 문항 단위로 평탄화."""
    data = json.load(open(path, encoding="utf-8"))
    rows = []
    for art in data["data"]:
        for para in art["paragraphs"]:
            for qa in para["qas"]:
                rows.append({
                    "guid": qa["guid"],
                    "title": art["title"],
                    "source": art.get("source"),
                    "news_category": art.get("news_category"),
                    "context": para["context"],
                    "question": qa["question"],
                    "question_type": qa["question_type"],
                    "is_impossible": qa["is_impossible"],
                    "answers": [a["text"] for a in qa.get("answers", [])],
                })
    return rows


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "—"


def audit(rows, name):
    n = len(rows)
    out = {"n": n}
    print(f"\n{'=' * 64}\n{name}  (문항 {n:,}건)\n{'=' * 64}")

    # --- 1. question_type × is_impossible
    ct = Counter((r["question_type"], r["is_impossible"]) for r in rows)
    out["question_type_x_impossible"] = {f"qt{k[0]}_imp{int(k[1])}": v
                                         for k, v in sorted(ct.items())}
    n_imp = sum(r["is_impossible"] for r in rows)
    out["n_impossible"] = n_imp
    out["n_answerable"] = n - n_imp
    print(f"\n[1] question_type × is_impossible")
    for k in sorted(ct):
        print(f"    question_type={k[0]}  is_impossible={str(k[1]):5s}  {ct[k]:6,}건 ({pct(ct[k], n)})")
    print(f"    → 응답가능 {n - n_imp:,}건 · 무응답 {n_imp:,}건 ({pct(n_imp, n)})")

    # --- 2. 출처 (금융 교란의 핵심)
    src = Counter(r["source"] for r in rows)
    src_imp = Counter(r["source"] for r in rows if r["is_impossible"])
    out["source"] = dict(src)
    out["source_answerable"] = {k: src[k] - src_imp.get(k, 0) for k in src}
    print(f"\n[2] 출처별 분포  ★ 금융 교란 확인")
    for k, v in src.most_common():
        print(f"    {str(k):12s} {v:6,}건 ({pct(v, n)})  응답가능 {v - src_imp.get(k, 0):6,}건")
    hk = [k for k in src if k and "hankyung" in str(k).lower()]
    if hk:
        nh = sum(src[k] for k in hk)
        print(f"    → 한국경제(경제·금융지) {nh:,}건 = 전체의 {pct(nh, n)} — Phase 2 전제와 충돌, 제외 검토 대상")

    # --- 2b. 뉴스 카테고리
    cat = Counter(r["news_category"] for r in rows)
    out["news_category"] = dict(cat.most_common())
    print(f"\n[2b] news_category 상위 12")
    for k, v in cat.most_common(12):
        print(f"    {str(k):16s} {v:6,}건 ({pct(v, n)})")
    fin_kw = ("경제", "금융", "증권", "부동산", "산업", "기업")
    fin = sum(v for k, v in cat.items() if k and any(w in str(k) for w in fin_kw))
    out["n_finance_like_category"] = fin
    print(f"    → 경제·금융 계열 카테고리 합계 {fin:,}건 ({pct(fin, n)})")

    # --- 3. 복수 정답
    ans_rows = [r for r in rows if not r["is_impossible"]]
    multi = sum(len(set(r["answers"])) > 1 for r in ans_rows)
    out["n_multi_gold"] = multi
    out["max_n_gold"] = max((len(r["answers"]) for r in ans_rows), default=0)
    print(f"\n[3] 복수 정답 — 응답가능 {len(ans_rows):,}건 중 서로 다른 정답이 2개 이상: "
          f"{multi:,}건 ({pct(multi, len(ans_rows))}), 최대 {out['max_n_gold']}개")
    print(f"    → 채점기에 max-over-golds 분기 필요 (금융은 단일 gold 전제)")

    # --- 4. 길이
    ctx = [len(r["context"]) for r in rows]
    gold = [len(a) for r in ans_rows for a in r["answers"][:1]]
    out["context_len"] = {"mean": statistics.mean(ctx), "median": statistics.median(ctx),
                          "p90": sorted(ctx)[int(0.9 * len(ctx))], "max": max(ctx)}
    out["gold_len"] = {"mean": statistics.mean(gold), "median": statistics.median(gold),
                       "p90": sorted(gold)[int(0.9 * len(gold))], "max": max(gold)}
    over = sum(c > 1500 for c in ctx)          # 대략 2048 토큰 여유선(한국어 문자≈토큰 상한)
    out["n_context_over_1500char"] = over
    print(f"\n[4] 지문 길이(문자)  평균 {out['context_len']['mean']:.0f} · "
          f"중앙 {out['context_len']['median']:.0f} · p90 {out['context_len']['p90']} · "
          f"최대 {out['context_len']['max']}")
    print(f"    1,500자 초과 {over:,}건 ({pct(over, n)}) — max_seq_len 2048 에서 절단 위험")
    print(f"[5] 정답 길이(문자)  평균 {out['gold_len']['mean']:.1f} · "
          f"중앙 {out['gold_len']['median']} · p90 {out['gold_len']['p90']} · "
          f"최대 {out['gold_len']['max']}")
    long_gold = sum(g > 100 for g in gold)
    out["n_gold_over_100char"] = long_gold
    print(f"    100자 초과 {long_gold:,}건 — 금융 제외 규칙(문장형 gold)과 동일 기준 적용 대상")

    # --- 6. 문서 단위
    titles = Counter(r["title"] for r in rows)
    out["n_titles"] = len(titles)
    out["q_per_title"] = {"mean": n / len(titles), "max": max(titles.values())}
    print(f"\n[6] 문서(title) {len(titles):,}개 · 문서당 문항 평균 "
          f"{n / len(titles):.1f} · 최대 {max(titles.values())}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klue-dir", required=True)
    ap.add_argument("--out", default="results/klue_audit.json")
    args = ap.parse_args()

    d = Path(args.klue_dir)
    report = {"klue_dir": str(d)}
    splits = {}
    for split, fn in (("train", "klue-mrc-v1.1_train.json"),
                      ("dev", "klue-mrc-v1.1_dev.json")):
        p = d / fn
        if not p.exists():
            print(f"[경고] {p} 없음 — 건너뜀")
            continue
        rows = load_split(p)
        splits[split] = rows
        report[split] = audit(rows, f"KLUE-MRC {split}")

    # --- 분할 간 문서 누출
    if "train" in splits and "dev" in splits:
        t = {r["title"] for r in splits["train"]}
        v = {r["title"] for r in splits["dev"]}
        overlap = t & v
        report["title_overlap_train_dev"] = len(overlap)
        print(f"\n{'=' * 64}\n[7] train·dev 문서(title) 겹침: {len(overlap):,}개")
        print("    → 0이면 KLUE 자체 분할을 그대로 써도 문서 누출이 없다.")
        print("      0이 아니면 prepare_klue.py 에서 title 단위로 재분할해야 한다.")

    # --- Phase 2 실현 가능성 요약
    if "dev" in splits:
        dv = splits["dev"]
        ans = [r for r in dv if not r["is_impossible"]]
        imp = [r for r in dv if r["is_impossible"]]
        nonhk_ans = [r for r in ans if "hankyung" not in str(r["source"]).lower()]
        nonhk_imp = [r for r in imp if "hankyung" not in str(r["source"]).lower()]
        report["feasibility"] = {
            "dev_answerable": len(ans), "dev_impossible": len(imp),
            "dev_answerable_excl_hankyung": len(nonhk_ans),
            "dev_impossible_excl_hankyung": len(nonhk_imp),
        }
        print(f"\n{'=' * 64}\n[8] Phase 2 평가셋 실현 가능성 (dev 기준)")
        print(f"    응답가능 {len(ans):,} · 무응답 {len(imp):,}")
        print(f"    한국경제 제외 시  응답가능 {len(nonhk_ans):,} · 무응답 {len(nonhk_imp):,}")
        print(f"    금융 평가셋 동형 목표: 응답가능 1,200 · hard UA 267 · easy UA 300")
        ok = len(nonhk_ans) >= 1200 and len(nonhk_imp) >= 267
        print(f"    → 한국경제 제외해도 동형 구성 {'가능' if ok else '불가 — 구성 재검토 필요'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
