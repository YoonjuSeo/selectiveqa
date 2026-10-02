# -*- coding: utf-8 -*-
"""
preflight_tokenizer.py — 새 모델 투입 전 입력 형식 점검 (GPU 불필요, 토크나이저만 로드)

사용:
  python src/analysis/preflight_tokenizer.py --config config_ax31.yaml

확인 항목 (하나라도 FAIL 이면 exit 1):
  1. 저장소 최신 커밋 전체 해시 출력 → config 의 revision 에 기록
  2. 실제 학습·추론 프롬프트 렌더링 결과 출력 (눈으로 확인)
  3. 과제 지시문(SYSTEM_PROMPT)이 정확히 1회 들어가고, 템플릿이 기본 system 문구를 끼워 넣지 않는가
  4. 학습 경로(add_special_tokens=False)와 추론 경로(tokenizer(prompt)) 토큰열이 같은가 (BOS 중복·누락)
  5. 학습 타깃에 붙는 eos_token 이 generation_config 의 eos_token_id 에 포함되는가 (생성 종료 보장)
  6. UA 타깃 / 예시 Answerable 타깃의 토큰 수 (모델 간 비교용 기록)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import yaml
from transformers import AutoTokenizer, GenerationConfig

from inference.prompts import (SYSTEM_PROMPT, build_messages, build_target,
                               apply_template, set_system_prompt_style,
                               current_system_prompt)

# 템플릿이 몰래 넣을 수 있는 기본 system 문구의 흔적 (A.X 템플릿의 도구 호출 안내 등)
SUSPECT_DEFAULT_SYSTEM = ["도구 호출", "You are a helpful assistant", "Cutting Knowledge Date"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    set_system_prompt_style(cfg)
    name = cfg["model"]["base_model"]
    rev = cfg["model"].get("revision")
    et = cfg["model"].get("enable_thinking")
    fails = []

    # 1. 커밋 해시
    try:
        from huggingface_hub import HfApi
        sha = HfApi().model_info(name).sha
        print(f"[1] {name} 최신 커밋: {sha}")
        if not rev:
            fails.append(f"config revision 이 비어 있음 → revision: \"{sha}\" 로 기록할 것")
        elif not sha.startswith(str(rev)):
            print(f"    (config revision={rev} 은 최신과 다름 — 의도한 고정이면 무시)")
    except Exception as e:
        print(f"[1] 커밋 조회 실패: {e}")

    tok = AutoTokenizer.from_pretrained(name, revision=rev, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # train_qlora.py 와 동일 처리

    # 2. 렌더링
    msgs = build_messages("지문 예시 문장입니다. 2024년 매출은 100억원이다.", "2024년 매출은?")
    prompt = apply_template(tok, msgs, et)
    print("[2] 렌더링된 프롬프트 (repr):")
    print(repr(prompt))

    # 3. system 문구
    n_sys = prompt.count(SYSTEM_PROMPT)
    print(f"[3] 과제 지시문 등장 횟수: {n_sys}")
    if n_sys != 1:
        fails.append(f"과제 지시문이 {n_sys}회 등장")
    rest = prompt.replace(current_system_prompt(), "")   # 의도한 system 문구는 제외하고 검사
    for s in SUSPECT_DEFAULT_SYSTEM:
        if s in rest:
            # 경고만: Llama-3.1 처럼 템플릿이 날짜 헤더를 항상 넣는 경우도 있어 판단은 사람이 한다
            print(f"    WARN 템플릿이 추가한 문구 의심: '{s}' — 기존 4모델과 같은 성격인지 확인")

    # 4. 학습/추론 토큰열 일치
    ids_train = tok(prompt, add_special_tokens=False)["input_ids"]
    ids_infer = tok(prompt)["input_ids"]
    head = tok.convert_ids_to_tokens(ids_infer[:4])
    print(f"[4] bos_token={tok.bos_token!r} (id {tok.bos_token_id}) · 추론 입력 앞 4토큰: {head}")
    if ids_train != ids_infer:
        extra = len(ids_infer) - len(ids_train)
        fails.append(f"학습/추론 토큰열 불일치 (추론 쪽 {extra:+d}토큰: special token 자동 삽입)")

    # 5. eos 정합성
    gen = GenerationConfig.from_pretrained(name, revision=rev)
    gen_eos = gen.eos_token_id if isinstance(gen.eos_token_id, list) else [gen.eos_token_id]
    print(f"[5] tokenizer.eos_token={tok.eos_token!r} (id {tok.eos_token_id}) · generation eos ids={gen_eos}")
    if tok.eos_token_id not in gen_eos:
        fails.append("학습 타깃 끝 eos 가 generation 종료 토큰에 없음 → 생성이 멈추지 않을 수 있음")

    # 6. 타깃 토큰 수
    ua = build_target(None, answerable=False) + tok.eos_token
    ans = build_target("100억원", evidence_span="2024년 매출은 100억원이다.") + tok.eos_token
    print(f"[6] UA 타깃 {len(tok(ua, add_special_tokens=False)['input_ids'])}토큰 · "
          f"예시 Answerable 타깃 {len(tok(ans, add_special_tokens=False)['input_ids'])}토큰")

    print("\n" + ("PASS" if not fails else "FAIL:\n  - " + "\n  - ".join(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
