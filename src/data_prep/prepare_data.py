# -*- coding: utf-8 -*-
"""
prepare_data.py — AI Hub JSON을 표준 JSONL로 변환하고 문서 단위로 분할한다.

출력 스키마 (한 줄 = 한 질문):
  {"question_id", "doc_id", "type", "context", "question", "gold_answer"}

사용법:
  1) python prepare_data.py --inspect        # 원본 JSON 구조를 먼저 눈으로 확인
  2) FIELD_MAP / TYPE_MAP 을 실제 스키마에 맞게 수정
  3) python prepare_data.py                  # train.jsonl / eval.jsonl 생성

주의: AI Hub 데이터셋은 버전에 따라 필드명이 다르다.
      --inspect 결과를 보고 아래 두 맵을 반드시 맞춘 뒤 실행할 것.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# TODO(사용자): --inspect 출력을 보고 실제 스키마에 맞게 수정하세요.
# AI Hub 기계독해 계열은 대체로 SQuAD 유사 구조입니다:
#   {"data": [{"paragraphs": [{"context": ..., "qas": [{"question": ...,
#       "answers": [{"text": ...}], "question_type"(또는 유사 필드): ...}]}]}]}
# ---------------------------------------------------------------------------
FIELD_MAP = {
    "root_list": "data",            # 문서 리스트가 담긴 최상위 키
    "paragraphs": "paragraphs",     # 문서 내 지문 리스트 키
    "context": "context",           # 지문 텍스트 키
    "qas": "qas",                   # 질문 리스트 키
    "question": "question",         # 질문 텍스트 키
    "answers": "answers",           # 정답 리스트 키
    "answer_text": "text",          # 정답 텍스트 키
    "qtype": "question_type",       # 질문 유형 키 (없으면 None으로 두고 아래 참조)
    "doc_id": "doc_id",             # 문서 식별자 키 (없으면 파일명+인덱스로 대체됨)
}

# TODO(사용자): 원본 유형 라벨 → 표준 4유형 매핑. --inspect 로 라벨 값 확인 후 수정.
TYPE_MAP = {
    # 예시 (실제 라벨로 교체):
    "본문 추출": "text_span",
    "테이블 추출": "table_lookup",
    "숫자 연산": "numeric_reasoning",
    "Yes/No": "yes_no",
}

# 숫자연산 기계독해 데이터처럼 유형 필드가 아예 없는 소스에 부여할 기본 유형
DEFAULT_TYPE_BY_SOURCE = {
    "finance": None,                 # 유형 필드를 사용
    "numeric": "numeric_reasoning",  # 전부 숫자 연산형으로 간주
}


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_json_files(directory):
    d = Path(directory)
    if not d.exists():
        print(f"[경고] 폴더 없음: {d} — 건너뜀")
        return
    yield from sorted(d.rglob("*.json"))


def inspect(cfg):
    """각 소스에서 파일 1개를 열어 구조를 출력한다. FIELD_MAP 수정용."""

    def show(obj, indent=0, depth=3):
        pad = "  " * indent
        if depth == 0:
            print(pad + "...")
            return
        if isinstance(obj, dict):
            for k, v in list(obj.items())[:8]:
                print(f"{pad}{k}: ({type(v).__name__})")
                show(v, indent + 1, depth - 1)
        elif isinstance(obj, list) and obj:
            print(f"{pad}[list, len={len(obj)}] 첫 원소:")
            show(obj[0], indent + 1, depth - 1)
        else:
            s = str(obj)
            print(pad + (s[:80] + ("..." if len(s) > 80 else "")))

    for name, key in [("finance", "raw_finance_dir"), ("numeric", "raw_numeric_dir")]:
        files = list(iter_json_files(cfg["paths"][key]))
        print(f"\n===== 소스 '{name}' ({cfg['paths'][key]}) : 파일 {len(files)}개 =====")
        if files:
            print(f"--- 예시 파일: {files[0]} ---")
            with open(files[0], encoding="utf-8") as f:
                show(json.load(f), depth=4)


def parse_source(directory, source_name):
    """한 소스 폴더의 모든 JSON을 표준 스키마 리스트로 변환한다."""
    fm = FIELD_MAP
    examples, skipped = [], Counter()
    default_type = DEFAULT_TYPE_BY_SOURCE.get(source_name)

    for fp in iter_json_files(directory):
        with open(fp, encoding="utf-8") as f:
            raw = json.load(f)
        docs = raw.get(fm["root_list"], raw if isinstance(raw, list) else [])
        for di, doc in enumerate(docs):
            doc_id = str(doc.get(fm["doc_id"], f"{fp.stem}-{di}"))
            for para in doc.get(fm["paragraphs"], []):
                context = para.get(fm["context"], "")
                if not context:
                    skipped["no_context"] += 1
                    continue
                for qa in para.get(fm["qas"], []):
                    question = qa.get(fm["question"], "")
                    answers = qa.get(fm["answers"], [])
                    if not question or not answers:
                        skipped["no_q_or_a"] += 1
                        continue
                    gold = answers[0].get(fm["answer_text"], "") if isinstance(answers[0], dict) else str(answers[0])
                    if not gold:
                        skipped["empty_answer"] += 1
                        continue
                    if default_type:
                        qtype = default_type
                    else:
                        raw_type = qa.get(fm["qtype"], "")
                        qtype = TYPE_MAP.get(raw_type)
                        if qtype is None:
                            skipped[f"unmapped_type:{raw_type}"] += 1
                            continue
                    examples.append({
                        "question_id": f"{source_name}-{doc_id}-{qa.get('id', len(examples))}",
                        "doc_id": f"{source_name}-{doc_id}",
                        "type": qtype,
                        "context": context,
                        "question": question,
                        "gold_answer": str(gold),
                    })
    print(f"[{source_name}] 파싱 {len(examples)}건, 제외 사유: {dict(skipped)}")
    return examples


def split_by_document(examples, cfg, rng):
    """문서 단위 분할: 같은 doc_id의 질문이 train/eval 양쪽에 존재하지 않게 한다.

    문서를 섞은 뒤, 유형별 평가 정원(eval_per_type)이 찰 때까지
    '문서 전체'를 평가셋에 배정하고, 나머지 문서에서 학습셋을 샘플링한다.
    """
    per_type_quota = cfg["data"]["eval_per_type"]
    types = cfg["data"]["types"]

    by_doc = defaultdict(list)
    for ex in examples:
        by_doc[ex["doc_id"]].append(ex)
    doc_ids = list(by_doc)
    rng.shuffle(doc_ids)

    eval_set, eval_count = [], Counter()
    train_pool = []
    for doc_id in doc_ids:
        doc_examples = by_doc[doc_id]
        # 이 문서가 채워줄 수 있는 미달 유형이 있으면 평가셋으로
        helps = any(eval_count[ex["type"]] < per_type_quota for ex in doc_examples)
        if helps:
            for ex in doc_examples:
                if eval_count[ex["type"]] < per_type_quota:
                    eval_set.append(ex)
                    eval_count[ex["type"]] += 1
                # 정원 초과분은 누출 방지를 위해 학습에도 넣지 않고 버린다
        else:
            train_pool.extend(doc_examples)

    missing = {t: per_type_quota - eval_count[t] for t in types if eval_count[t] < per_type_quota}
    if missing:
        print(f"[경고] 평가 정원 미달 유형: {missing} — 데이터 양 또는 TYPE_MAP 확인 필요")

    rng.shuffle(train_pool)
    train_set = train_pool[: cfg["data"]["train_size"]]
    return train_set, eval_set


def _norm(text):
    """응답 불가능 검증용 간이 정규화 (evaluate.py normalize와 동일 원칙)."""
    import string
    import unicodedata
    punct = set(string.punctuation) | {"·", "…", "「", "」", "『", "』", "%"}
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return "".join(ch for ch in text if not ch.isspace() and ch not in punct)


def make_unanswerable(eval_set, n_target, rng, max_tries=30):
    """평가셋 질문을 '다른 문서'의 지문과 교차 배치해 응답 불가능 문항을 만든다.

    설계서 §4.3: 표면 형태(유형 분포)를 응답가능 평가셋과 맞추기 위해
    4개 유형에서 균등하게 표집한다. 교차 지문에 정답 문자열이 우연히
    포함되면(정규화 부분일치) 다른 지문으로 재시도한다.

    ※ 자동 검증은 '정답 문자열 부재'만 확인하므로, 생성 후 무작위 50건
      수동 검수(설계서 §4.3)가 반드시 필요하다. 검수용 필드로 원 정답을
      orig_gold 에 보존한다.
    """
    by_type = defaultdict(list)
    for ex in eval_set:
        by_type[ex["type"]].append(ex)
    types = sorted(by_type)
    per_type = n_target // len(types)
    quotas = {t: per_type for t in types}
    for t in types[: n_target - per_type * len(types)]:  # 나머지 배분
        quotas[t] += 1

    contexts = [(ex["doc_id"], ex["context"]) for ex in eval_set]
    out, failed = [], 0
    for t in types:
        pool = by_type[t][:]
        rng.shuffle(pool)
        made = 0
        for ex in pool:
            if made >= quotas[t]:
                break
            gold_n = _norm(ex["gold_answer"])
            swapped = None
            for _ in range(max_tries):
                doc_id, ctx = contexts[rng.randrange(len(contexts))]
                if doc_id == ex["doc_id"]:
                    continue  # 같은 문서 지문은 응답 가능할 수 있음
                if gold_n and gold_n in _norm(ctx):
                    continue  # 교차 지문에 정답이 우연히 존재 → 재시도
                swapped = ctx
                break
            if swapped is None:
                failed += 1
                continue
            out.append({
                "question_id": ex["question_id"] + "-unans",
                "doc_id": ex["doc_id"],
                "type": "unanswerable",
                "orig_type": t,               # 표면 형태 분석용
                "context": swapped,
                "question": ex["question"],
                "gold_answer": "",            # 정답 없음 (무응답이 정답)
                "orig_gold": ex["gold_answer"],  # 수동 검수용 참조
                "answerable": False,
            })
            made += 1
    if failed:
        print(f"[경고] 응답 불가능 생성 실패 {failed}건 (교차 지문 탐색 실패)")
    from collections import Counter as _C
    print(f"응답 불가능 생성: {len(out)}건, 원유형 분포 {dict(_C(r['orig_type'] for r in out))}")
    return out


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"저장: {path} ({len(rows)}건)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--inspect", action="store_true", help="원본 구조만 출력하고 종료")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.inspect:
        inspect(cfg)
        return

    rng = random.Random(cfg["seed"])
    examples = parse_source(cfg["paths"]["raw_finance_dir"], "finance")
    examples += parse_source(cfg["paths"]["raw_numeric_dir"], "numeric")
    if not examples:
        raise SystemExit("파싱된 예제가 없습니다. --inspect 로 스키마를 확인하고 FIELD_MAP을 수정하세요.")

    train_set, eval_set = split_by_document(examples, cfg, rng)

    # 분할 검증: 문서 누출 확인
    overlap = {e["doc_id"] for e in train_set} & {e["doc_id"] for e in eval_set}
    assert not overlap, f"문서 누출 발생: {overlap}"

    # 응답가능 표시 (본 실험 신규 필드)
    for ex in train_set + eval_set:
        ex["answerable"] = True

    # ★ 본 실험: 응답 불가능 평가 문항 생성 후 평가셋에 추가 (설계서 §4.2)
    n_unans = cfg["data"].get("unanswerable_eval", 0)
    if n_unans > 0:
        eval_set = eval_set + make_unanswerable(eval_set, n_unans, rng)

    out = Path(cfg["paths"]["processed_dir"])
    write_jsonl(out / "train.jsonl", train_set)
    write_jsonl(out / "eval.jsonl", eval_set)

    print("\n=== 유형 분포 ===")
    for name, rows in [("train", train_set), ("eval", eval_set)]:
        print(name, dict(Counter(r["type"] for r in rows)))


if __name__ == "__main__":
    main()
