# -*- coding: utf-8 -*-
"""
make_train_mix.py — 학습용 무응답 혼입 데이터 3세트 생성 (설계서 5.1절)

train.jsonl(3,000건 전건 응답가능)에서 유형 비율을 보존한 k건을
교차 지문(easy) 방식의 무응답 예시로 '교체'하여 혼입 세트를 만든다.
  r=5%  → k=150 → train_mix_r05.jsonl
  r=10% → k=300 → train_mix_r10.jsonl
  r=30% → k=900 → train_mix_r30.jsonl

설계 특성 (용량-반응 해석의 교란 통제):
  1) 총량 고정: 모든 세트가 정확히 3,000건 — 노출 총량 통제
  2) 질문 불변: 3,000개 질문 집합이 원본·전 세트에서 동일 —
     바뀌는 것은 감독 신호(응답 vs 무응답)뿐
  3) 중첩(nested) 전환: 전환 집합이 r=5% ⊂ r=10% ⊂ r=30% —
     혼입률 간 차이가 '어떤 문항이 뽑혔나'가 아니라 '얼마나 뽑혔나'만 반영
  4) 지문 고정: 같은 문항의 교체 지문은 전 세트에서 동일 —
     세트 간 차이를 혼입량으로만 국한
  5) 행 순서 불변: 원본 train.jsonl의 행 순서를 유지하고 해당 위치만 교체
     (셔플은 Trainer가 시드로 수행 — 학습 순서 재현성은 기존 체계 그대로)

학습·평가 생성 방식 분리(기록): 학습 무응답은 교차 지문(easy)만 사용.
hard UA 평가에서의 복원은 훈련 분포 밖 일반화 검증이 된다.

사용:
  python make_train_mix.py --config config.yaml
출력:
  data/processed/train_mix_r05.jsonl / _r10 / _r30
  data/processed/train_mix_manifest.json  (무결성 검사 결과·전환 목록)
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from data_prep.prepare_data import _norm

RATIOS = [("r05", 150), ("r10", 300), ("r30", 900)]   # (태그, k) — 오름차순 필수
MAX_TRIES = 30


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stratified_nested_selection(rows, ks, rng):
    """유형 비율 보존 + 중첩 보장 전환 집합 선정.

    유형별로 한 번 섞어 순서를 고정한 뒤, 각 k의 유형별 정원(최대잔여법)
    만큼 '앞에서부터' 취한다. 정원이 k에 대해 단조증가하면 중첩이 자동 보장.
    반환: {k: [row_index, ...]}
    """
    by_type = defaultdict(list)
    for i, r in enumerate(rows):
        by_type[r["type"]].append(i)
    types = sorted(by_type)
    for t in types:
        rng.shuffle(by_type[t])               # 유형별 순서 1회 고정

    n_total = len(rows)
    quotas = {}                               # {k: {type: 정원}}
    for k in ks:
        exact = {t: k * len(by_type[t]) / n_total for t in types}
        base = {t: int(exact[t]) for t in types}
        rem = k - sum(base.values())
        for t in sorted(types, key=lambda t: exact[t] - base[t], reverse=True)[:rem]:
            base[t] += 1
        quotas[k] = base

    # 최대잔여법의 드문 비단조 사례 보정 (중첩 보장)
    for k_small, k_large in zip(ks, ks[1:]):
        for t in types:
            if quotas[k_large][t] < quotas[k_small][t]:
                quotas[k_large][t] = quotas[k_small][t]
        excess = sum(quotas[k_large].values()) - k_large
        if excess > 0:                        # 보정으로 초과 시 여유 유형에서 감축
            for t in sorted(types, key=lambda t: quotas[k_large][t] - quotas[k_small][t],
                            reverse=True):
                cut = min(excess, quotas[k_large][t] - quotas[k_small][t])
                quotas[k_large][t] -= cut
                excess -= cut
                if excess == 0:
                    break

    selection = {k: [] for k in ks}
    for k in ks:
        for t in types:
            selection[k].extend(by_type[t][: quotas[k][t]])
    return selection


def assign_swapped_contexts(rows, convert_indices, rng):
    """전환 대상 문항마다 교체 지문을 1회 확정 배정 (전 세트 공유).

    prepare_data.make_unanswerable 과 동일 원칙:
      - 다른 doc_id 의 지문만 사용 (같은 문서는 응답 가능할 수 있음)
      - 교체 지문에 원정답이 (정규화 부분일치로) 존재하면 재시도
    반환: {row_index: (ctx_doc_id, context)}, 실패 인덱스 목록
    """
    contexts = [(r["doc_id"], r["context"]) for r in rows]
    assigned, failed = {}, []
    for i in sorted(convert_indices):
        ex = rows[i]
        gold_n = _norm(ex["gold_answer"])
        pick = None
        for _ in range(MAX_TRIES):
            doc_id, ctx = contexts[rng.randrange(len(contexts))]
            if doc_id == ex["doc_id"]:
                continue
            if gold_n and gold_n in _norm(ctx):
                continue
            pick = (doc_id, ctx)
            break
        if pick is None:
            failed.append(i)
        else:
            assigned[i] = pick
    return assigned, failed


def make_ua_row(ex, ctx_doc_id, context):
    """전환된 무응답 학습 행. build_target 이 answerable=False 로 무응답
    타깃({"answerable": false, "answer": null, ...})을 생성하도록 표시."""
    row = dict(ex)
    row.update({
        "question_id": ex["question_id"] + "-trainua",
        "context": context,
        "gold_answer": "",
        "answerable": False,
        "orig_gold": ex["gold_answer"],       # 감사용 (학습 타깃에는 미사용)
        "ctx_doc_id": ctx_doc_id,             # 교체 지문 출처 (무결성 검사용)
        "orig_type": ex["type"],
    })
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rng = random.Random(cfg["seed"])
    proc = Path(cfg["paths"]["processed_dir"])

    train = load_jsonl(proc / "train.jsonl")
    eval_rows = load_jsonl(proc / "eval.jsonl")
    eval_docs = {r["doc_id"] for r in eval_rows}
    ks = [k for _, k in RATIOS]

    selection = stratified_nested_selection(train, ks, rng)
    all_convert = set(selection[ks[-1]])      # 최대 k의 전환 집합이 전체
    assigned, failed = assign_swapped_contexts(train, all_convert, rng)
    if failed:
        # 실패분은 전환 집합에서 제거 — 총량 유지를 위해 대체 표본이 필요하나,
        # 교차 지문 실패는 실질적으로 희귀하므로 발생 시 수동 확인으로 처리
        print(f"[주의] 교체 지문 배정 실패 {len(failed)}건 — 수동 확인 필요")

    manifest = {
        "seed": cfg["seed"],
        "design": "총량 3,000 고정 · 질문 집합 불변 · 중첩 전환 · 지문 고정 · 행 순서 불변",
        "ua_generation": "교차 지문(easy)만 사용 — hard UA 평가는 훈련 분포 밖 일반화 검증",
        "sets": {},
    }

    for tag, k in RATIOS:
        convert = set(selection[k]) - set(failed)
        out_rows = []
        for i, ex in enumerate(train):
            if i in convert:
                out_rows.append(make_ua_row(ex, *assigned[i]))
            else:
                out_rows.append(dict(ex))

        # ---- 무결성 검사 (설계서 5.1절) ----
        n_ua = sum(1 for r in out_rows if not r.get("answerable", True))
        type_dist = Counter(r.get("orig_type", r["type"]) for r in out_rows)
        used_docs = ({r["doc_id"] for r in out_rows}
                     | {r["ctx_doc_id"] for r in out_rows if "ctx_doc_id" in r})
        leak = used_docs & eval_docs
        gold_resid = sum(
            1 for r in out_rows if not r.get("answerable", True)
            and _norm(r["orig_gold"]) and _norm(r["orig_gold"]) in _norm(r["context"])
        )
        checks = {
            "total": len(out_rows), "n_unanswerable": n_ua, "expected_k": k,
            "type_dist_surface": dict(type_dist),
            "eval_doc_leak": sorted(leak), "gold_in_context": gold_resid,
        }
        ok = (len(out_rows) == len(train) and n_ua == k - len([f for f in failed if f in selection[k]])
              and not leak and gold_resid == 0)
        checks["pass"] = bool(ok)

        out_path = proc / f"train_mix_{tag}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["sets"][tag] = {
            "k": k, "path": str(out_path), "checks": checks,
            "converted_qids": sorted(train[i]["question_id"] for i in convert),
        }
        print(f"[{tag}] k={k}: 총 {len(out_rows)}건 (무응답 {n_ua}) · "
              f"eval 누출 {len(leak)}건 · 정답 잔존 {gold_resid}건 · "
              f"{'PASS' if ok else 'FAIL'}  → {out_path}")

    # 중첩 검증
    c05 = set(manifest["sets"]["r05"]["converted_qids"])
    c10 = set(manifest["sets"]["r10"]["converted_qids"])
    c30 = set(manifest["sets"]["r30"]["converted_qids"])
    nested = c05 <= c10 <= c30
    manifest["nested_check"] = {"r05⊆r10": c05 <= c10, "r10⊆r30": c10 <= c30}
    print(f"중첩 검증: r05⊆r10⊆r30 = {nested} {'✓' if nested else '⚠'}")

    man_path = proc / "train_mix_manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"매니페스트 저장: {man_path}")


if __name__ == "__main__":
    main()