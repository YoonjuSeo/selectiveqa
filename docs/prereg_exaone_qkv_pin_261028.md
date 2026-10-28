# 사전등록: EXAONE 원 M2 붕괴의 원인 분리 (q/k/v, transformers 4.57.1) — 2026-10-28

**학습 제출 전 커밋.** 이 문서의 판정 기준은 결과를 보기 전에 고정한다.

## 1. 배경
- 원 EXAONE(q/k/v, out_proj 누락)은 M1이 transformers 4.x(2026-08-08, loss 0.096~0.101)에서, M2-r05를 포함한 모든 M2가 5.x(2026-08-25 이후)에서 학습되었다(어댑터 tokenizer_config 저장 형식으로 확인).
- 원 M2-r05는 ΔEM 약 −34%p, Answerable 응답 길이 약 ×2.4의 붕괴를 보였다.
- out_proj를 포함하고 4.57.1로 고정한 재학습(prereg_exaone_outproj_261002)에서 붕괴가 사라졌다(P2 3/3). 이 재학습은 out_proj 포함 여부와 라이브러리 버전을 동시에 바꾸었으므로 원인을 가르지 못한다.
- 5.x는 EXAONE 원격 코드와 호환되지 않는다는 정황이 있다(5.x 학습 loss 0.778, 진단 재학습 loss 0.4~0.9, 5.x forward 실패).

## 2. 질문
원 설정(q/k/v)을 4.57.1에서 학습하면 M2-r05의 붕괴가 재현되는가?

## 3. 설계 (단일 변수)
- 모델: LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct, revision 0ff6b5ec7c13 (원 실험과 동일)
- LoRA 대상: [q_proj, k_proj, v_proj] (원 실험과 동일, 3종 × 32층 = 96모듈, 9,437,184 파라미터 예상)
- 학습 데이터: data/processed/train_mix_r05.jsonl (원 M2-r05와 동일)
- 하이퍼파라미터: config.yaml과 동일 (r 16, α 32, dropout 0.05, lr 2e-4, 실효 배치 16, 2 epoch, 376 step)
- 라이브러리: modal_train_pinned.py (torch 2.8.0, transformers 4.57.1, peft 0.20.0, accelerate 1.14.0, bitsandbytes 0.50.0)
- seed: 42 한 번
- 비교 기준: 원 M1 seed 42 (results/qlora_adapter_s42, 4.x, q/k/v) 예측 results/preds_M1_v2_s42.jsonl
- 추론: modal_inference.py (transformers 4.57.1), eval_full_v2.jsonl, greedy
- 설정 파일: config_abl_exaone_qkv_pin.yaml / 산출물: results/exaone_qkv_pin/

따라서 원 M1과 이번 M2의 차이는 학습 데이터(혼입 5%)뿐이고, 원 M2와 이번 M2의 차이는 라이브러리 버전뿐이다.

## 4. 판정 기준 (prereg_exaone_outproj_261002와 같은 기준)
지표: ΔEM = EM(이번 M2) − EM(원 M1 s42), Answerable 1,170건(제외 30건), evaluate_followup.is_correct.
R_tok = Answerable 평균 생성 토큰 수(이번 M2) / (원 M1 s42).

| 판정 | 조건 | 해석 |
|---|---|---|
| Q1 붕괴 재현 | ΔEM ≤ −0.20 그리고 R_tok ≥ 1.8 | 4.x에서도 q/k/v만으로 붕괴 → 원인은 out_proj 누락(LoRA 설정) |
| Q2 붕괴 없음 | ΔEM ≥ −0.05 그리고 R_tok ≤ 1.2 | 4.x에서는 q/k/v로도 정상 → 원 붕괴의 원인은 transformers 5.x 학습 환경 |
| Q3 부분 | 그 외 | 두 요인이 모두 기여했거나 판정 불가. 원인을 특정하지 않는다 |

- 보조(판정에 쓰지 않음): train_loss, easy/hard UA 환각률, 허위 무응답률, LoRA 적용 검사 출력(96모듈).
- 이 실험은 본문 결과(out_proj 재학습)를 바꾸지 않는다. 결과는 부록 A의 원인 서술에만 반영한다.
- seed 1개의 탐색적 판정이다. 결과가 Q3이거나 경계(ΔEM −0.05~−0.20)에 가까우면 seed 43을 추가할지 결과 확인 전에 정한 이 규칙으로 결정한다: Q3이면 seed 43 추가, Q1·Q2이면 추가하지 않는다.

## 5. 결과 (학습 후 기록)
- 학습 일시·계정:
- train_loss:
- LoRA 적용 검사 출력:
- ΔEM / R_tok / 판정:
