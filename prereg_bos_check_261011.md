# 사전 계획 — 추론 입력 BOS 중복 점검 (Kanana, Llama)

작성일 2026-10-11 · 어떤 BOS 1개 추론 결과도 확인하기 전에 커밋한다.
상위 문서: 3장 3.2.9절(추론 토큰화), 3.4.3절(강건성 점검), `prereg_stageF_ax31_260929.md` 103행(Kanana BOS 중복 발견)

## 1. 배경

`run_inference.py`는 대화 템플릿을 문자열로 만든 뒤 `tokenizer(prompt)`를 기본값(특수 토큰 자동 추가)으로 호출한다.
템플릿이 이미 BOS를 넣는 모델에서는 추론 입력 맨 앞에 BOS가 두 번 들어간다. 학습 경로는
`add_special_tokens=False`라 BOS가 하나뿐이므로 학습과 추론의 토큰열이 다르다.
Kanana는 `preflight_tokenizer.py`로 중복을 확인했고(학습 1개, 추론 2개), Llama는 이 문서의 1단계에서 확인한다.
EXAONE·Qwen3·A.X는 템플릿에 BOS가 없어 해당하지 않는다(3.2.9절).

이 점검은 판정 가설(H1~H5)에 쓰지 않는 강건성 점검이며(3.4.3절), 본 판정은 원 예측(BOS 2개)으로 유지한다.

## 2. 코드 변경 (판정·채점 코드 무변경)

- `run_inference.py`에 `--dedupe-bos` 옵션 추가. 입력 맨 앞 두 토큰이 모두 BOS일 때만 하나를 지운다.
  중복이 없으면 입력을 바꾸지 않는다. 기본값은 꺼짐이라 기존 실행과 동일하다.
  켜면 예측 레코드에 `bos_dedup` 필드를 남기고, 실행 끝에 제거 건수를 출력한다.
- `modal_inference.py`에 같은 옵션 전달 인자 추가.
- `src/analysis/compare_preds.py` 추가 (두 예측 파일의 문항별 일치도).
- 프롬프트, 템플릿, 생성 설정(greedy, max_new_tokens 128), 어댑터, 평가셋(`eval_full_v2.jsonl`), 채점·판정 코드는 바꾸지 않는다.

## 3. 절차

### 1단계: 대상 확인 (GPU 불필요)
- `python src/analysis/preflight_tokenizer.py --config config_llama.yaml` 4번 항목에서 BOS 중복 여부를 확인한다.
  중복이 없으면 Llama는 대상에서 빼고 그 사실을 기록한다.
- 같은 점검을 Kanana에 다시 돌려 중복을 재확인한다.

### 2단계: 재현 관문
원 예측은 2026년 9월에 만들었고 추론 래퍼는 transformers 4.57.1·peft 0.20.0·bitsandbytes 0.50.0으로 고정했으나
torch는 고정하지 않았다. 지금 환경이 원 예측을 재현하는지 먼저 확인한다.
- 모델별로 M1 seed 42를 **옵션 없이**(BOS 2개, 원 설정 그대로) 앞 100건만 추론하고 원 `preds_M1_v2_s42.jsonl`과 비교한다.
- **통과 기준**: answer 일치 ≥ 98% 그리고 answerable 판단 일치 ≥ 99%.
- 통과하면 원 예측을 BOS 2개 대조군으로 쓴다. 통과하지 못하면 환경 차이가 섞이므로,
  BOS 2개 대조군도 현재 환경에서 같은 범위로 다시 추론한 뒤 그것과 비교한다(비용 2배).

### 3단계: BOS 1개 추론 — seed 42 우선
모델별 3회: M0, M1 seed 42, M2-r05 seed 42 (각 1,767건, `--dedupe-bos`).
실행 로그의 "BOS 중복 제거" 건수가 1,767건 전부여야 한다(아니면 중단하고 원인 확인).

### 4단계: seed 43·44 확장 규칙 (결과 보기 전 고정)
seed 42에서 아래 중 **하나라도** 해당하면 그 모델은 seed 43·44의 M1·M2-r05도 추론한다. 하나도 해당하지 않으면 seed 42로 끝낸다.
- H1·H2·H3·H5 중 seed 42 판정이 하나라도 바뀜 (원 예측 판정 대비)
- easy UA 무응답률 또는 hard UA 무응답률(M0, M1, M2-r05 각각)의 차이 |Δ| ≥ 5%p
- Answerable EM(M0, M1, M2-r05 각각)의 차이 |Δ| ≥ 2%p
- M1 분리 AUROC(H3 정의)의 차이 |Δ| ≥ 0.02

## 4. 분석과 보고

- `evaluate_followup.py --m0 preds_M0_bos1.jsonl --m1-glob "preds_M1_bos1_s*.jsonl" --m2-glob "preds_M2_r05_bos1_s*.jsonl" --tag r05_bos1`
  로 BOS 1개 판정을 내고 원 판정(`metrics_followup_r05.json`)과 seed별로 나란히 놓는다.
- `model_cells.py`(같은 인자)로 4장 표 지표를 BOS 1개 기준으로 다시 산출한다.
- `compare_preds.py`로 원 예측과의 문항별 일치도(answer, answerable)를 보고한다.
- 4.8.3절에는 위 결과를 기술 통계로 싣는다. 본 판정과 4장 본문 수치는 바꾸지 않는다.
  BOS 1개에서 결론이 달라지면 그 사실과 범위를 4.8.3절과 5장 한계에 적는다.

## 5. 출력 파일 이름

`results/{kanana,llama}/preds_repro_M1_s42.jsonl`(재현 관문), `preds_M0_bos1.jsonl`, `preds_M1_bos1_s{seed}.jsonl`, `preds_M2_r05_bos1_s{seed}.jsonl`.
대조군을 다시 돌리는 경우 `preds_*_bos2now*.jsonl`.

## 6. 기록

| 날짜 | 내용 |
|---|---|
| 2026-10-11 | 본 문서와 코드 변경 커밋 (BOS 1개 결과 확인 전) |
| 2026-10-11 | 1단계: preflight_tokenizer.py로 Kanana·Llama 모두 추론 입력 BOS 중복 확인(앞 두 토큰 <\|begin_of_text\|> ×2, 학습 대비 +1토큰) → 두 모델 모두 대상 |
| 2026-10-11 | 2단계 재현 관문 통과: M1 seed 42 앞 100건을 옵션 없이 재추론(Modal somajuju08, L4), Kanana·Llama 모두 answer 100/100·answerable 100/100·신뢰도 차이 최대 0.0000 → 원 예측을 BOS 2개 대조군으로 사용 |
