# 사전등록 보완 — EXAONE(out_proj) 혼입률 격자 r05 → r30

작성일 2026-10-03 · r30 학습 제출 전 커밋
상위 문서: `docs/prereg_grid_r30_260908.md` (판정 기준 원본), `prereg_exaone_outproj_261002.md` (out_proj 절제)

## 1. 배경

out_proj 절제(2026-10-03, P2 3/3)로 원 EXAONE의 붕괴(ΔEM −0.34)가 LoRA 대상 모듈에서
attention 출력층이 빠진 설정에서 발생했음을 확인했다. 본 분석의 EXAONE은 q/k/v/out_proj
설정으로 교체한다. 원 EXAONE의 r05·r10·r30·r01 격자 결과는 이 설정을 따르지 않으므로
본문 4.7절에 쓰지 않고 부록("원 설정(q/k/v) 결과")으로 옮긴다.

4.7절의 질문은 다음 두 가지다. EXAONE r30을 추가하면 Qwen3·Llama와 같은 설정의 세 번째 모델이 된다.

1. hard UA 복원이 혼입량에 비례하는가 (r30 − r05의 Δ, 다이얼/스위치 판정)
2. 혼입을 늘리면 응답가능 문항 비용이 생기는가 (r30의 H2)

## 2. 판정 기준 — 원본과 동일 (변경 없음)

`docs/prereg_grid_r30_260908.md`의 기준을 그대로 적용한다.

- 통계량: Δ_s = (r30 hard UA 무응답률) − (r05 hard UA 무응답률). 동일 시드·동일 hard UA 문항
  question_id 페어링, hard 층 재표집 paired bootstrap 10,000회, 백분위 95% CI.
- 마진 δ = 0.10.
- 판정(시드별): 다이얼 = CI 하한 > +δ / 스위치 = CI 상한 < +δ / 미결정 = 그 외.
- 집계: 3시드 동일 판정이면 강건, 2/3이면 조건부, 그 외 미결정.
- 부수 조건: r30에서도 H2(ΔEM)와 응답가능 오무응답률이 r05 수준을 유지해야 "이득"으로 해석한다.
- H1~H5는 `evaluate_followup.py`의 기존 정의를 그대로 쓴다.

## 3. 원본 대비 이탈 사항

| 항목 | 원본(원 EXAONE 격자) | 이번 실행 |
|---|---|---|
| LoRA 대상 모듈 | q_proj, k_proj, v_proj (out_proj 누락) | q_proj, k_proj, v_proj, **out_proj** |
| config | `config.yaml` | `config_abl_exaone_outproj.yaml` |
| 학습 래퍼 | `modal_train.py` | `modal_train_pinned.py` (torch 2.8.0, transformers 4.57.1, peft 0.20.0, accelerate 1.14.0, bitsandbytes 0.50.0 고정) |
| r05 비교 기준 | 원 EXAONE `preds_M2_r05_s*` | `results/exaone_outproj/preds_M2_r05_pin_s*` |
| M1 | 원 EXAONE `preds_M1_v2_s*` | `results/exaone_outproj/preds_M1_pin_s*` (재학습 없이 재사용) |
| M0 | `preds_M0_v2.jsonl` | 동일 (LoRA 무관) |
| 학습 데이터 | `train_mix_r30.jsonl` | 동일 (sha256 `9b5be1fa…74325ab`, Qwen3·Llama r30과 같은 파일) |

나머지(r16/α32, 2 epoch, lr 2e-4, 실효 배치 16, 모델 리비전 0ff6b5ec7c13, eval_full_v2,
greedy, max_new_tokens 128, 제외 목록)는 out_proj r05와 동일하다.

## 4. 판정 격 — 사전 지정된 탐색적 판정

원본과 같이 확증적 판정이 아니라 **사전 지정된 탐색적 판정**으로 보고한다. 이번에는 그
이유가 하나 더 있다. 기준을 적용하기 전에 아래 결과를 이미 알고 있다.

- EXAONE(out_proj) r05: hard UA 무응답률 약 39%(시드별 .49/.32/.36), H2 3/3 채택
- Qwen3 r30: H2 3/3 유지 / Llama r30: H2 0/3 (−1.7~−3.0%p)
- 원 EXAONE(q/k/v)의 r30 결과 (붕괴 설정)

판정 기준은 위 결과를 보고 바꾸지 않았다(원본 문서 그대로).

## 5. 실행 범위와 순서

- 학습: M2-r30, seed 42 → 학습 파라미터 13,631,488개와 train_loss 정상 범위(r05 0.088~0.090
  부근)를 확인한 뒤 seed 43·44 제출. seed 42 결과(평가 지표)를 보고 43·44 실행 여부를 바꾸지
  않는다. 3시드 모두 실행한다.
- 태그: 학습 `_r30_pin` → 어댑터 `results/exaone_outproj/qlora_adapter_r30_pin_s{seed}`,
  추론 `--out-tag M2_r30_pin_s{seed}`.
- 평가:
  - `evaluate_followup.py --config config_abl_exaone_outproj.yaml --m1-glob "preds_M1_pin_s*.jsonl" --m2-glob "preds_M2_r30_pin_s*.jsonl" --tag r30_pin --n-boot 10000`
  - `compare_ratio.py --config config_abl_exaone_outproj.yaml --tag-a r05_pin --tag-b r30_pin`
  - R_tok: `rtok_check.py`
- 학습 중단·선점 시 같은 시드·같은 설정으로 재제출한다. 라이브러리 버전 검사 실패 시 학습하지 않는다.

## 6. 해석 시 미리 밝혀둘 점

- r05 hard UA 무응답률이 이미 약 39%로 Llama(21.7%)보다 높아, Δ가 δ=0.10을 넘을 여지가
  상대적으로 작을 수 있다. 이 상한 효과를 결과와 함께 적는다.
- 원 EXAONE r30과의 비교는 부록에서 기술적으로만 하며 판정에 쓰지 않는다.
- 실행 계정(yoonjuseo/somayj08)이 바뀌어도 이미지·GPU(L4)·코드는 동일하다. 실제 사용 계정은 결과란에 기록한다.

## 7. 결과 (실행 후 작성)

(비워둠)
