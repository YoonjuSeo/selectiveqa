# -*- coding: utf-8 -*-
"""
make_review_targets.py — hard UA 위험층 표적 검수 파일 생성 (본생성 2차 표본 후속)

배경 (변경 관리 기록과 연동):
  2차 표본 50건 검수 결과 C 3건이 전부 numeric ER(표본 내 12건 중 3건),
  B(폐기) 1건이 따옴표 안 제목 연도 치환(TS)에서 발생.
  → 재생성 대신 위험층 표적 전수 검수로 마무리:
     (a) numeric ER 전수  (b) TS 중 '따옴표 안 연도' 치환 건

사용 (프로젝트 루트에서):
  python src\\data_prep\\make_review_targets.py
출력:
  data/processed/hard_ua_numeric_er_review.txt   (numeric ER 전수)
  data/processed/hard_ua_ts_quoted_year_review.txt (스캔 해당분)
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

# 따옴표 쌍: 한국어 기사·보고서에서 제목 인용에 쓰이는 부호들
QUOTE_SPAN_RE = re.compile(r"['\"‘“」『「]([^'\"’”」』]{2,60})['\"’”」』]")

LEGEND = (
    "판정 범례:\n"
    "  C      = 남은 지문이 질문에 대한 답을 명시하거나 정합적으로 귀속시킴\n"
    "  B(유지) = 답 불성립이나 인접 정보가 특정 오답을 강하게 유도 (경계 기록, 문항 유지)\n"
    "  B(폐기) = 라벨 계쟁 — 잔존 제한자(인명·인용 문자열·제한자 조합)만으로\n"
    "           대상이 유일 특정되어, 무응답과 수선 응답 중 규범적 답 판별 불가\n"
    "  A      = 답 불성립 + 강한 유도 없음\n"
    "처리 원칙: C·B(폐기) → 문항 단위 폐기 (results/hard_ua_discards.json 에 qid·사유 추가)\n"
)

CHECKPOINTS_NUMERIC_ER = (
    "numeric ER 검수 관점 3패턴 (2차 표본에서 발견):\n"
    "  1) 주제명 재서술: gold가 지문 주제의 이름이라 개념 기술이 지문 전체에 잔존\n"
    "  2) worked-example 역산: 규칙·요율 문장은 지워졌으나 예시 계산으로 값 복원 가능\n"
    "  3) 표기 변형 잔존: ㈜·(주) 등 접두, 단위 변형으로 잔존 검사가 놓친 경우\n"
)

CHECKPOINTS_TS_QY = (
    "TS 따옴표-연도 검수 관점 (2차 표본 [25] 패턴):\n"
    "  치환된 연도가 데이터 인덱스가 아니라 보고서·문서 제목의 라벨이면,\n"
    "  묻는 사실이 상대 참조('직전 대비' 등)로 지문에 유일 대응해 라벨 계쟁 → B(폐기)\n"
    "  연도가 실제 데이터를 선택하는 인덱스면 정상 트랩 → A\n"
)


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def year_in_quotes(question, year):
    """질문의 따옴표 인용 구간 안에 해당 연도가 등장하는지."""
    for m in QUOTE_SPAN_RE.finditer(question):
        if year in m.group(1):
            return True
    return False


def write_review(rows, path, header_note):
    with open(path, "w", encoding="utf-8") as f:
        f.write(LEGEND + "\n" + header_note + "\n")
        for i, r in enumerate(rows, 1):
            f.write("=" * 70 + "\n")
            f.write(f"[{i:02d}] QID: {r['question_id']}  |  원유형: {r['orig_type']}"
                    f"  |  방식: {r['ua_kind']}"
                    f"{'  |  easy재사용' if r.get('easy_ua_reused') else ''}\n")
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
            f.write(r["context"][:2500]
                    + ("\n...(생략)" if len(r["context"]) > 2500 else "") + "\n")
            f.write("판정: [   ]\n")
    print(f"검수 파일 저장: {path} ({len(rows)}건)")


def main():
    cfg = load_config()
    proc = Path(cfg["paths"]["processed_dir"])
    res_dir = Path(cfg["paths"]["results_dir"])

    rows = load_jsonl(proc / "hard_ua.jsonl")

    # 이미 폐기 확정된 문항은 검수 대상에서 제외
    discards = set()
    dpath = res_dir / "hard_ua_discards.json"
    if dpath.exists():
        discards = {d["question_id"]
                    for d in json.loads(dpath.read_text(encoding="utf-8"))["items"]}
        print(f"기존 폐기 목록 {len(discards)}건 — 검수 대상에서 제외")

    # (a) numeric ER 전수
    num_er = [r for r in rows
              if r["orig_type"] == "numeric_reasoning"
              and r["ua_kind"] == "hard_evidence_removal"
              and r["question_id"] not in discards]
    write_review(num_er, proc / "hard_ua_numeric_er_review.txt",
                 CHECKPOINTS_NUMERIC_ER)

    # (b) TS 따옴표-연도 스캔
    ts_qy = []
    for r in rows:
        if r["ua_kind"] != "hard_target_swap" or r["question_id"] in discards:
            continue
        sw = r["gen_meta"]["swap"]
        if sw["kind"] == "year" and year_in_quotes(
                r["gen_meta"]["orig_question"], sw["from"]):
            ts_qy.append(r)
    write_review(ts_qy, proc / "hard_ua_ts_quoted_year_review.txt",
                 CHECKPOINTS_TS_QY)

    print(f"\n요약: numeric ER 전수 {len(num_er)}건 / TS 따옴표-연도 {len(ts_qy)}건"
          f" — 두 파일 검수 후 C·B(폐기) 건을 hard_ua_discards.json 에 추가하고"
          f" apply_discards.py 실행")


if __name__ == "__main__":
    main()