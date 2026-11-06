# 사전 계획 — closed-book 통제 실험 (사전지식 점검)

작성일 2026-11-06 · 어떤 closed-book 예측도 확인하기 전에 커밋한다.
상위 문서: 논문 3.2.2절(사전지식 혼입의 점검), 4.1절([표 4-2a]), 5.3절(사전지식의 통제)
관련: `prereg_bos_check_261011.md` 2단계(재현 관문)와 같은 절차를 따른다.

## 1. 배경과 질문

1차 발표(2026-11-06) 심사 의견: "hard UA(근거 제거형·대상 치환형)의 환각이 사실은 모델이 사전지식으로 답을 알고 있어서
생긴 것일 수 있다. 원래 답할 수 있는 문항은 실험에서 배제했어야 한다."
기존 점검([표 4-2a])은 M1의 환각 답이 편집 전 원 정답(`orig_gold`)과 일치하는 비율(근거 제거형 1~4%)로 간접 확인한 것이다.
이 실험은 **지문 없이 질문만 주었을 때** 원본 모델(M0)이 원 정답을 맞히는 비율을 직접 측정한다.

- Q1. 지문 없이 M0가 UA 문항의 원 정답을 맞히는 비율(유형별)은 얼마인가?
- Q2. 그렇게 맞힌 '사전지식 가능' UA 문항을 제외하면 본 논문의 hard UA 환각률·무응답률·복원 비율(RQ1·RQ2)이 달라지는가?

## 2. 설계 (판정·채점 코드 무변경)

- 조건: **M0 (원본 instruct 모델, 어댑터 없음)**, 다섯 모델 전부. M0는 greedy라 seed 변동이 없으므로 모델당 1회.
- 입력: `eval_full_v2.jsonl` 1,767건 전부(Answerable 포함). 지문을 주지 않고 `[질문]`만 준다 (`prompts.build_messages_closed_book`).
- 프롬프트 모드
  - `force` (주 조건): "확실하지 않더라도 가장 가능성이 높은 답을 하나 반드시 쓰세요" → 사전지식 정답률의 **상한**.
  - `abstain` (탐색적, 선택): 무응답 형식 허용. 모델이 '모른다'고 아는지 보는 용도이며 판정에 쓰지 않는다.
- 출력 형식·파서·신뢰도 계산·디코딩(greedy, max_new_tokens 128)·revision·Qwen3 non-thinking: 본 실험과 동일.
- 채점: `evaluate.is_correct` (수치 허용오차 1e-4, yes/no 극성, NFKC 정규화) — UA는 `orig_gold`, Answerable은 `gold_answer` 기준.
  제외 목록 `excluded_gold_v2*.json`은 기존 규칙 그대로 적용.
- '이지선다' 문항(yes/no 75건, 질문 안에 두 선택지가 있는 40건)은 지문 없이도 50%로 맞힐 수 있으므로 **별도 열로 함께 보고**하고,
  해석은 이를 제외한 문항을 기준으로 한다.

## 3. 환경 통제 (다섯 모델 동일)

- 래퍼 `modal_inference_cb.py`: `modal_train_pinned.py`와 같은 PINNED 이미지
  (torch 2.8.0, transformers 4.57.1, peft 0.20.0, accelerate 1.14.0, bitsandbytes 0.50.0 = `results/env_versions.txt`).
  실행 전 버전을 검사해 하나라도 다르면 중단. 실행마다 `env_closedbook_<mode>.json`(버전·revision·GPU·명령)을 결과 폴더에 남긴다.
- **재현 관문(모델마다 필수)**: 같은 이미지에서 정상 프롬프트 M0 앞 100건을 다시 추론해 원 `preds_M0_v2.jsonl`과 비교한다.
  통과 기준(2026-11-06 개정, `src/analysis/gate_check.py`): **G1** answerable 판단 일치 ≥ 99% **그리고** **G2** 정답 여부
  (`evaluate.is_correct`, 무응답은 오답) 일치 ≥ 97% **그리고** **G3** 신뢰도 |차이| 평균 ≤ 0.01. 답 문자열 완전 일치율은 보고만 하고
  판정에 쓰지 않는다. 미통과 모델은 closed-book 결과를 본 실험과 같은 환경으로 해석할 수 없으므로 원인을 기록하고 보류한다(판정에 넣지 않음).
  - 개정 사유: 최초 기준(answer 문자열 일치 ≥ 98%)은 "의미가 같은 표기 차이"(조사·따옴표 모양)와 숫자 토큰의 수치 노이즈에
    과민하다. 또 원 `preds_M0_v2.jsonl`은 torch를 고정하지 않은 `modal_inference.py`로 생성돼 원 환경의 버전이 기록에 없다.
    G2는 이 연구가 실제로 쓰는 지표(정답 여부)로, G3는 수치 환경 차이의 크기로 환경 동일성을 점검한다.
  - **개정 시점의 공개**: Qwen3 관문의 첫 실행(문자열 일치 0.950, answerable 0.990, 신뢰도 평균 차 0.0045)을 본 뒤에 개정했다.
    다만 개정 기준은 closed-book 본 실험 결과(주 판정 C1~C3)를 보기 전에 고정했고, 본 실험 결과는 관문 기준에 영향을 주지 않는다.
    논문에는 이 순서를 그대로 적는다. 개정 전 기준의 수치(문자열 일치 0.950)도 함께 보고한다.
- BOS: 본 판정 예측과 같게 `--dedupe-bos` 끔(Kanana·Llama 포함).

## 4. 판정 기준 (결과 확인 전 고정)

| 항목 | 기준 | 의미 |
|---|---|---|
| C1 사전지식 상한 | 근거 제거형 closed-book 정답률(이지선다 제외) **< 10%**, 다섯 모델 모두 | hard UA 환각의 대부분은 사전지식으로 설명되지 않음 |
| C2 지표 강건성 | '사전지식 가능' UA를 제외한 뒤 M2-r05 hard UA 복원 비율(RR)의 변화 **< 0.05**, 15 seed 모두 | 본 논문 RQ2 수치는 사전지식 혼입에 둔감 |
| C3 M1 환각률 | 제외 후에도 M1 hard UA 무응답률 = 0% (15 seed) | RQ1 소거 결과 불변 |

- C1·C2·C3 모두 충족: 본문은 그대로 두고 4.1절에 "closed-book 정답률 x~y%" 한 문장과 부록 표를 추가한다.
- C1 미충족(어느 모델이든 ≥ 10%): 해당 유형의 '사전지식 가능' 문항을 제외한 수치를 본문의 **주 수치**로 바꾸고, 제외 전 수치를 부록으로 내린다.
- 대상 치환형은 정의상 원 질문의 답이 지문에 있어 closed-book 정답률이 높아도 '지문 복사'와 구분되지 않으므로 C1의 대상에서 제외하고 보고만 한다.
- 이 실험은 사후에 추가된 통제 분석이며, 사전등록 가설 H1~H5의 판정에는 영향을 주지 않는다(4.1절 보조 분석).

## 5. 절차와 명령 (`run_closedbook.ps1` 참고)

1. `modal run modal_inference_cb.py --config <cfg> --check-only` — 버전 확인
2. `modal run modal_inference_cb.py --config <cfg> --gate` — 재현 관문 (원 `preds_M0_v2.jsonl`이 Volume의 `<results_dir>/`에 있어야 함)
3. `modal run --detach modal_inference_cb.py --config <cfg> --mode force` — 본 실험
4. `modal volume get selectiveqa-results <results_dir>/preds_M0_cb_force.jsonl .\results\<results_dir>` (+ env_closedbook_*.json)
5. `python src/analysis/closedbook_analysis.py --config <cfg> [--m1-glob ... --m2-glob ...]` — 로컬 분석 → `closedbook_force.{json,md}`

모델별 config / results_dir / M1·M2 glob:
| 모델 | config | results_dir | --m1-glob | --m2-glob |
|---|---|---|---|---|
| EXAONE | config_abl_exaone_outproj.yaml | results/exaone_outproj | preds_M1_pin_s*.jsonl | preds_M2_r05_pin_s*.jsonl |
| Qwen3 | config_qwen3.yaml | results/qwen3 | preds_M1_v2_s*.jsonl | preds_M2_r05_s*.jsonl |
| Llama | config_llama.yaml | results/llama | preds_M1_v2_s*.jsonl | preds_M2_r05_s*.jsonl |
| Kanana | config_kanana.yaml | results/kanana | preds_M1_v2_s*.jsonl | preds_M2_r05_s*.jsonl |
| A.X | config_ax31.yaml | results/ax31 | preds_M1_v2_s*.jsonl | preds_M2_r05_s*.jsonl |

예상 비용: 모델당 관문 100건 + 본 실험 1,767건(지문이 없어 입력이 짧음) → L4 기준 모델당 약 20~40분.

## 6. 기록

| 날짜 | 내용 |
|---|---|
| 2026-11-06 | 계획 작성. 코드 변경: `prompts.py`(closed-book 프롬프트 추가, 기존 함수 무변경), `run_inference.py`(`--closed-book` 옵션, 기본 꺼짐), `modal_inference_cb.py`, `src/analysis/closedbook_analysis.py`, `run_closedbook.ps1` 신규. 결과 확인 전. |
| 2026-11-06 | Qwen3 `--check-only` 통과(PINNED 일치). 관문 1차: 문자열 일치 0.950(미통과, 개정 전 기준 0.98), answerable 0.990, 신뢰도 평균 차 0.0045. 불일치 5건 = 표기 차이 2건 + `numeric_reasoning` 3건(숫자 토큰 갈림). 관문 기준을 G1~G3으로 개정(3절). 개정 기준으로 재판정(로컬, 추가 추론 없음): answerable 0.990, 정답 여부 일치 1.000, 신뢰도 평균 차 0.0045 → 통과. 본 실험 결과 확인 전. |
