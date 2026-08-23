# -*- coding: utf-8 -*-
"""
apply_discards.py — hard UA 문항 단위 폐기 적용 (검수 확정분 → 최종본 생성)

동작:
  1) results/hard_ua_discards.json 이 없으면 2차 표본 검수에서 확정된
     4건(C 3 + B폐기 1)으로 시드 생성
  2) 표적 검수(numeric ER 전수, TS 따옴표-연도)에서 추가 확정된 건은
     이 파일의 items 에 {question_id, judgement, reason} 형식으로 수동 추가
  3) 본 스크립트 재실행 → data/processed/hard_ua_final.jsonl 생성
     (원본 hard_ua.jsonl 은 감사용으로 보존, 항상 원본에서 다시 필터)

사용 (프로젝트 루트에서):
  python src\\data_prep\\apply_discards.py
"""
import datetime
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

SEED_DISCARDS = [
    {"question_id": "numeric-cf34910d-fb39-4cbb-8ed2-83b62e423778-hd-er",
     "judgement": "C",
     "reason": "주제명 gold 재서술 잔존 — '규제의 핵심' 문장이 답을 명시 (표본 [13])"},
    {"question_id": "numeric-d4db13da-63c3-4cdd-934a-5e5f8ca788c8-hd-er",
     "judgement": "C",
     "reason": "worked-example 역산 — 예시 계산(900만→500만 등)으로 요율 0.5% 복원 가능 (표본 [33])"},
    {"question_id": "numeric-caa24385-2c87-4d89-8694-bef8d839df5b-hd-er",
     "judgement": "C",
     "reason": "접두 정규화 실패 — ㈜ NFKC 변형으로 '불스원' 잔존 검사 누락 (표본 [35])"},
    {"question_id": "numeric-c6cd5012-608f-4fee-a53e-b0bb78e4a704-hd-ts",
     "judgement": "B(폐기)",
     "reason": "제목 라벨 연도 치환 — 묻는 사실이 상대 참조로 유일 대응, 라벨 계쟁 (표본 [25])"},
]


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    cfg = load_config()
    proc = Path(cfg["paths"]["processed_dir"])
    res_dir = Path(cfg["paths"]["results_dir"])
    dpath = res_dir / "hard_ua_discards.json"

    # 1) 폐기 목록 로드 또는 시드 생성
    if dpath.exists():
        record = json.loads(dpath.read_text(encoding="utf-8"))
        print(f"폐기 목록 로드: {len(record['items'])}건")
    else:
        record = {
            "note": "hard UA 문항 단위 폐기 목록. 검수(C·B폐기) 확정분만 추가. "
                    "원본 hard_ua.jsonl 은 불변, 최종본은 hard_ua_final.jsonl.",
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "items": SEED_DISCARDS,
        }
        dpath.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                         encoding="utf-8")
        print(f"폐기 목록 시드 생성: {dpath} ({len(SEED_DISCARDS)}건 — 2차 표본 확정분)")

    discard_ids = [d["question_id"] for d in record["items"]]
    dup = [q for q, c in Counter(discard_ids).items() if c > 1]
    if dup:
        raise SystemExit(f"[중단] 폐기 목록에 중복 qid: {dup}")
    discard_set = set(discard_ids)

    # 2) 원본에서 필터 (항상 원본 기준 — 멱등)
    rows = load_jsonl(proc / "hard_ua.jsonl")
    all_ids = {r["question_id"] for r in rows}
    unknown = sorted(discard_set - all_ids)
    if unknown:
        raise SystemExit(f"[중단] 폐기 목록의 qid 가 원본에 없음 (오타 확인): {unknown}")

    kept = [r for r in rows if r["question_id"] not in discard_set]
    out_path = proc / "hard_ua_final.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 3) 요약 기록 갱신
    dist = Counter((r["orig_type"], r["ua_kind"]) for r in kept)
    record["last_applied"] = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "original_n": len(rows), "discarded_n": len(discard_set),
        "final_n": len(kept),
        "final_dist": {f"{t}|{k}": c for (t, k), c in sorted(dist.items())},
        "final_path": str(out_path),
    }
    dpath.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                     encoding="utf-8")

    print(f"\n원본 {len(rows)}건 − 폐기 {len(discard_set)}건 → 최종 {len(kept)}건")
    print("최종 분포:")
    for (t, k), c in sorted(dist.items()):
        print(f"  {t:<20} {k:<24} {c}")
    print(f"\n저장: {out_path}\n기록 갱신: {dpath}")
    print("\n[주의] 이후 파이프라인(평가셋 병합·추론)은 hard_ua_final.jsonl 을 사용할 것")


if __name__ == "__main__":
    main()