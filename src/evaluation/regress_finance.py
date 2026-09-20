# -*- coding: utf-8 -*-
"""
regress_finance.py — 금융 판정 결과가 Phase 2 코드 변경 이후에도 불변임을 증명한다.

Phase 2 를 위해 채점기 is_correct() 에 딱 하나의 분기를 넣었다 — gold 가 리스트면
max-over-golds. KLUE-MRC 응답가능 문항의 35.5%가 복수 정답을 갖기 때문이다.
금융 데이터의 gold_answer 는 항상 문자열이므로 이 분기는 발동하지 않아야 하고,
이 스크립트가 그 사실을 판정 산출물 전체 비교로 확인한다.

사전등록 원칙상 "채점 코드는 손대지 않는다"를 깨는 변경이므로, 변경과 함께 이
검증을 커밋해 판정 불변성을 기록으로 남긴다.

사용:
  # 1) 변경 전 커밋에서 산출된 metrics 를 기준본으로 보관
  cp results/metrics_followup_r05.json results/metrics_followup_r05.ref.json
  # 2) 변경 후 재산출 (반드시 사전등록 확정 신호 m1_conf 로)
  python src/evaluation/evaluate_followup.py --h4-signal m1_conf
  # 3) 비교
  python src/evaluation/regress_finance.py

  종료 코드 0 = 판정 불변, 1 = 불일치(원인 확인 전 Phase 2 진행 금지)
"""
import argparse
import json
import sys
from pathlib import Path

# 격자 실행 시 추가된 기록 필드(6.2절). 판정에 쓰이지 않으므로 비교에서 제외한다.
IGNORE_PREFIXES = ("/settings/m2_files", "/settings/m2_glob", "/settings/m1_glob")
TOL = 1e-12


def flatten(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flatten(v, f"{p}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flatten(v, f"{p}[{i}]")
    else:
        yield p, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="results/metrics_followup_r05.ref.json")
    ap.add_argument("--new", default="results/metrics_followup_r05.json")
    args = ap.parse_args()

    for p in (args.ref, args.new):
        if not Path(p).exists():
            raise SystemExit(f"파일 없음: {p} (사용법은 docstring 참조)")

    a = dict(flatten(json.load(open(args.ref, encoding="utf-8"))))
    b = dict(flatten(json.load(open(args.new, encoding="utf-8"))))

    keys = [k for k in sorted(set(a) | set(b))
            if not any(k.startswith(x) for x in IGNORE_PREFIXES)]
    diffs = []
    for k in keys:
        x, y = a.get(k), b.get(k)
        if (isinstance(x, (int, float)) and isinstance(y, (int, float))
                and not isinstance(x, bool) and not isinstance(y, bool)):
            if abs(x - y) > TOL:
                diffs.append((k, x, y))
        elif x != y:
            diffs.append((k, x, y))

    print(f"기준본 {args.ref}\n신규본 {args.new}")
    print(f"판정 관련 비교 항목 {len(keys):,}개 · 불일치 {len(diffs)}개")
    if diffs:
        for k, x, y in diffs[:30]:
            print(f"  {k} | {x} -> {y}")
        if len(diffs) > 30:
            print(f"  ... 외 {len(diffs) - 30}개")
        print("\n✗ 판정이 변했다. Phase 2 진행 전에 원인을 규명할 것.")
        sys.exit(1)

    print("\n✓ 판정 불변 — is_correct 의 max-over-golds 분기는 금융 경로에서 발동하지 않는다.")
    print("  (H1~H5 점추정·CI, risk-coverage, 운영점 전 항목 1e-12 이내 일치)")
    sys.exit(0)


if __name__ == "__main__":
    main()
