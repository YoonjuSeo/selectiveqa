# 사전등록 — EXAONE LoRA 대상 모듈 절제 (out_proj 포함)

작성일 2026-10-02 · 재학습·추론 결과 확인 전 커밋

## 1. 발견 경위

Stage F(A.X) 어댑터의 Tier 0-b 코사인을 계산하던 중, LoRA 모듈 수가 EXAONE 96개(32층×3),
다른 모델 128개(32층×4)로 다른 것을 확인했다(2026-10-02).

- 학습 코드는 `target_modules=["q_proj","k_proj","v_proj","o_proj"]`로 고정되어 있다.
- EXAONE-3.5의 attention 출력층 이름은 `o_proj`가 아니라 `out_proj`다(공식 modeling_exaone.py).
- 따라서 **EXAONE만 attention 출력층에 LoRA가 적용되지 않았다.** 저장된 어댑터 키도
  `q_proj, k_proj, v_proj` 3종뿐임을 확인했다.
- 학습 가능 파라미터: EXAONE 원 실험 9,437,184개. out_proj를 포함하면 13,631,488개로,
  같은 차원(hidden 4096, GQA KV 1024)을 가진 Llama·Kanana와 정확히 같아진다.

지금까지 확인된 원인 후보 중 **EXAONE에만 해당하는 실험 설정 차이**는 이것이 처음이다.

## 2. 질문

EXAONE의 M2-r05 붕괴(ΔEM ≈ −0.34, 응답 토큰 ×2.4, M1↔M2 코사인 0.026)는
attention 출력층에 LoRA가 적용되지 않은 설정에 의존하는가?

## 3. 조건 (이 항목 외 모든 설정은 config.yaml과 동일)

| 항목 | 원 실험 (config.yaml) | 절제 (config_abl_exaone_outproj.yaml) |
|---|---|---|
| LoRA 대상 | q_proj, k_proj, v_proj (out_proj 누락) | q_proj, k_proj, v_proj, **out_proj** |
| 학습 가능 파라미터 | 9.44M | 13.63M |
| 그 외 | QLoRA r16/α32, 2 epoch, lr 2e-4, 실효 배치 16, 과제 지시문 system prompt, BOS 미삽입 | 동일 |

주의: 이 절제는 "출력층 포함 여부"와 "LoRA 용량(파라미터 수)"을 함께 바꾼다. 두 요인을
분리하지는 않는다. 다만 바뀐 뒤의 설정은 다른 모델들과 같은 조건이 된다.

## 4. 실행 범위

- 1단계: M1·M2-r05, seed 42. 학습 후 (a) 로컬 LoRA 코사인, (b) eval_full_v2 추론.
- 2단계(조건부): 판정이 P2 또는 P3이면 seed 43·44를 추가해 3 seed로 판정한다.
  P1이면 seed 42로 종료할 수 있다.
- M0는 LoRA와 무관하므로 원 실험의 preds_M0_v2.jsonl을 그대로 쓴다.

## 5. 판정 기준 (권장 프롬프트 절제와 같은 문턱)

ΔEM = EM(M2-r05) − EM(M1), R_tok = 평균 응답 토큰(M2)/평균 응답 토큰(M1), Answerable 기준.

| 판정 | 조건 | 해석 |
|---|---|---|
| P1 붕괴 유지 | ΔEM ≤ −0.20 그리고 R_tok ≥ 1.8 | 출력층 누락은 붕괴의 필요조건이 아니다. 기존 주장 유지, 설정 차이는 한계로 명기 |
| P2 붕괴 소멸 | ΔEM ≥ −0.05 그리고 R_tok ≤ 1.2 | 붕괴는 "출력층 미적용 LoRA + 무응답 혼입" 조건에 의존한다 |
| P3 부분 완화 | 그 외 | 출력층 누락이 붕괴 크기에 기여한다 |

## 6. 보조 지표 (탐색적)

M1↔M2 LoRA 코사인(모듈별 평균, `lora_cosine.py`). 비교 기준: 원 실험 EXAONE 0.022~0.026,
원 실험 EXAONE M1 시드 노이즈 0.149~0.157, 무손실형 0.68~0.77.
코사인만으로 판정하지 않으며, 5절 판정과 방향이 다르면 둘 다 보고한다.

## 7. 결과와 무관하게 본문에 반영할 사항

3장 학습 설정에 "모듈 이름 차이로 EXAONE은 q/k/v 3종에만 LoRA가 적용되었다"를 명기한다.
기존 EXAONE 결과는 이 설정에서 얻은 것으로 서술한다.

## 8. 기록

(실행 로그의 trainable params 값, 이탈 사항을 날짜와 함께 적는다)
