# 사전등록 보완 — KLUE-MRC EXAONE M2-r05 (LoRA q/k/v/out_proj, 버전 고정)

작성일 2026-10-10 · **KLUE M2-r05 학습 제출 전 커밋**
상위 문서: `docs/prereg_phase2_klue_260920.md` (Phase 2 원 사전등록), `prereg_klue_m1_outproj_261004.md` (KLUE M1 재학습, §7·§8.5에서 이 실행을 예고)

## 1. 배경

KLUE M1(out_proj)에서 소거가 3 seed 모두 성립했다(easy UA 무응답 0/900, 강건; `prereg_klue_m1_outproj_261004.md` §8).
같은 문서 §7은 이 경우 "비교 기준을 금융 out_proj text_span으로 교체한 사전등록 보완 후 KLUE M2-r05 실행"으로 정해 두었다.
이 문서가 그 보완이다.

Phase 2 원래 질문("EXAONE 붕괴가 도메인을 떠나도 재현되는가")은 붕괴가 설정 오류(out_proj 누락)로
밝혀져 성립하지 않는다. 원 사전등록 §3의 붕괴 판별 규칙과 비교 기준(금융 원 EXAONE text_span ΔEM −0.303,
토큰 배율 ×2.33)은 쓰지 않는다. 질문을 논문의 RQ2·RQ3에 맞춰 다시 세운다.

## 2. 질문

**금융 EXAONE(out_proj)에서 관찰한 M2-r05의 복원과 비용이 KLUE-MRC에서도 같은 방향으로 나타나는가.**

- 복원: 교차 지문 무응답 예시 5%(150건) 혼입으로 easy UA 무응답이 M1 대비 되살아나는가.
- 비용: 그 대가로 응답가능 EM과 응답 길이가 변하는가.
- 금융 EXAONE(out_proj) 결과: H1 3/3, H2 3/3(ΔEM −0.003/−0.005/−0.009), R_tok(응답가능) ×1.00,
  허위 무응답 0%, M2 hard UA 무응답 49/32/36%.

결과는 1장 RQ3·기여 ③의 "복원의 도메인 일반성은 범위 밖" 문장과 3.2.8절·4.9절(KLUE)에 반영한다.

## 3. 판정 기준 (결과 확인 전 고정)

### 3.1 시드별 판정

| 축 | 판정 | 조건 |
|---|---|---|
| 복원 | 복원 성립 | **H1_easy 충족**: M2 easy UA 환각률 95% CI 상한 < M1 easy UA 환각률 CI 하한 |
| | 복원 불성립 | 그 외 |
| 비용 | 비용 없음 | **H2 충족**(ΔEM 95% CI 하한 > −0.03) **그리고** R_tok(응답가능) ∈ [0.90, 1.15] |
| | 비용 있음 | ΔEM 95% CI 상한 < −0.03 **또는** R_tok > 1.15 |
| | 비용 미결 | 그 외 (CI가 −0.03에 걸침 등) |

- H1·H2의 정의, 층화 paired bootstrap 10,000회, 마진 −0.03은 원 사전등록(260817·부칙 260828)과 Phase 2 §6 그대로다.
- R_tok 띠 [0.90, 1.15]는 Phase 2 사전등록 §3의 무손실형 조건과 같은 값이다.
- 응답가능 1,200건(D6)은 ΔEM CI 반폭을 금융과 비슷하게 맞추기 위해 정한 정원이며 바꾸지 않는다.

### 3.2 3시드 집계

같은 판정 3/3이면 강건, 2/3이면 조건부, 그 외는 미결. 복원과 비용을 따로 집계한다.

### 3.3 보조 지표 (판정에 쓰지 않고 함께 보고)

- hard UA: H1_hard, M2 hard UA 무응답률, **M0 대비** 차이(M2 − M0, 문항 짝 bootstrap 95% CI)와 비율(M2/M0).
  KLUE hard UA는 네이티브 `is_impossible`이라(D5) **금융 hard UA와 수치를 직접 비교하지 않는다.** 도메인 안에서 M0 대비로만 본다.
- easy UA의 M0 대비 차이·비율(논문 3.1.4의 "M0 대비 = 기술적 비교")
- 허위 무응답(M1·M2), 정답 틀 적합률(M2), H3·H4(`m1_conf`)·H5(text_span 단일)
- risk-coverage: M1+필터 곡선, M2 운영점, M2+필터 곡선, 운영점 2개(evaluate_followup.py 기본 산출). RQ4의 도메인 일반성 보조 근거
- train_loss, 학습 시간

### 3.4 비교 기준 — 금융 EXAONE(out_proj) text_span 하위집합

KLUE는 text_span 단일 유형이므로, 금융과의 크기 비교는 금융 EXAONE(out_proj)의 **text_span 하위집합**으로 한다
(Phase 2 §2와 같은 논리). 값은 기존 예측 파일에서 산출하며 새 학습·추론은 없다.
**KLUE M2 결과를 보기 전에** 산출해 `results/exaone_outproj/comparator_textspan_outproj.json`으로 커밋한다(§6 단계 0).
이 값은 서술용 비교 기준이며 §3.1 판정 문턱에는 들어가지 않는다.

## 4. 원본 대비 이탈 사항

| 항목 | Phase 2 원 사전등록 (M2-r05) | 이번 실행 |
|---|---|---|
| 질문 | EXAONE 붕괴의 도메인 재현 | 복원·비용의 도메인 재현 (§2) |
| 판별 규칙 | 붕괴형/무손실형/미결 (§3) | 복원 성립/불성립 × 비용 없음/있음/미결 (§3.1) |
| 비교 기준 | 금융 원 EXAONE(q/k/v) text_span ΔEM −0.303 | 금융 EXAONE(out_proj) text_span (§3.4) |
| LoRA 대상 모듈 | q/k/v (out_proj 누락) | q/k/v/out_proj (128모듈, 적용 층 수 검사) |
| config | `config_klue_exaone.yaml` | `config_klue_exaone_outproj.yaml` (KLUE M1 재학습과 같음) |
| 학습 래퍼 | `modal_train.py` (버전 미고정) | `modal_train_pinned.py` (torch 2.8.0 / transformers 4.57.1 / peft 0.20.0 / accelerate 1.14.0 / bitsandbytes 0.50.0) |
| 대조군 | Kanana KLUE M1·M2 | 실행하지 않음 (붕괴 질문이 사라져 대조군의 역할이 없음) |

학습 데이터: `klue_train_mix_r05_str.jsonl` (3,000건, sha256 `021f7c0b62a2…d7c369a`).
`klue_train_str.jsonl`(M1 학습 파일, sha256 `6f117c90…99f5671`)과 행 순서·질문 집합이 같고,
2,850행이 완전히 같으며 150행만 교차 지문 무응답(`*-trainua`, 정답 `{"answerable": false, ...}`)으로 바뀌었다(2026-10-10 확인).
평가 데이터 `klue_eval_full.jsonl`(1,767건), M0 예측(`results/klue_exaone/preds_klue_M0_s42.jsonl`, LoRA 무관이라 재추론 안 함. M0는 greedy라 `preds_klue_M0_s42/43/44.jsonl` 세 파일의 SHA256이 모두 같음을 2026-10-10에 확인: `4C2EDBC427B8330DB55426421FD111CDE3995D4844FA21985ABAABD977A582C5`),
M1 예측(`results/klue_exaone_outproj/preds_M1_pin_s{42,43,44}.jsonl`)은 그대로 쓴다.
나머지 하이퍼파라미터(r16/α32, dropout 0.05, 2 epoch, lr 2e-4, 실효 배치 16, max_seq_len 2048,
모델 리비전 0ff6b5ec7c13, greedy, max_new_tokens 128, 시드 42/43/44)는 KLUE M1 재학습과 같다.

## 5. 판정 격 — 사전 지정 탐색적 판정

기준을 정하기 전에 알고 있는 결과:

- 금융 다섯 모델의 M2-r05 결과 전부(EXAONE(out_proj) 무비용, A.X 중간형 등)
- KLUE M1(out_proj): easy UA 무응답 0%, 응답가능 EM 0.849~0.854, 응답가능 토큰 27.5~27.6, H3 AUROC 0.72~0.73
- KLUE M0: UA 무응답 77.2%

H1·H2 문턱은 원 사전등록 그대로이고 새로 정한 것은 R_tok 띠의 사용(Phase 2 §3에서 가져옴)과 §3.1의 결합 방식뿐이다.
금융 결과를 알고 정했으므로 사전 지정 탐색적 판정으로 보고한다. **KLUE M2 결과는 어떤 것도 보지 않았다.**

관문: Phase 2 §7 관문 2(M1 응답가능 EM 3 seed 평균 ≥ 0.55)는 KLUE M1(out_proj) 평균 0.852로 통과했다.
관문 1(M0 easy UA 무응답 ≥ 90%)은 원 Phase 2에서 확인한 값을 따른다.

## 6. 실행 순서

0. **비교 기준 산출·커밋 (KLUE M2 학습 전)**
   ```
   python src/analysis/comparator_textspan.py --config config_abl_exaone_outproj.yaml \
       --results-dir results/exaone_outproj --m0 preds_M0_v2.jsonl \
       --m1-glob "preds_M1_pin_s*.jsonl" --m2-glob "preds_M2_r05_pin_s*.jsonl" \
       --out comparator_textspan_outproj.json
   ```
   M0와 제외 목록은 `results/exaone_outproj/` 안의 파일(커밋 054ca1c)을 쓴다. 제외 30건이 잡혔는지 콘솔에서 확인한다.
   이 문서, 산출 JSON, `src/analysis/klue_m2_check.py`를 함께 커밋한 뒤 학습을 제출한다.

1. **학습** — seed 42 먼저. 학습 파라미터 13,631,488개, LoRA 적용 검사 통과(4종 × 32층), loss 발산 없음만 확인하고 43·44 제출.
   **seed 42의 평가 지표를 보고 43·44 실행 여부를 바꾸지 않는다.**
   ```
   modal run --detach modal_train_pinned.py --train-file klue_train_mix_r05_str.jsonl --tag _r05_pin --seed {seed} --config config_klue_exaone_outproj.yaml
   ```
   어댑터: `results/klue_exaone_outproj/qlora_adapter_r05_pin_s{seed}` (에폭별 `_ep1` 함께 저장, 판정에는 미사용)

2. **추론**
   ```
   modal run --detach modal_inference.py --condition M1 --adapter-dir results/klue_exaone_outproj/qlora_adapter_r05_pin_s{seed} --out-tag M2_r05_pin_s{seed} --eval-file klue_eval_full.jsonl --config config_klue_exaone_outproj.yaml
   ```
   (`--condition M1`은 "어댑터를 얹어 추론"이라는 뜻이며 금융 M2 추론과 같은 방식이다.)

3. **판정**
   ```
   python src/evaluation/evaluate_followup.py --config config_klue_exaone_outproj.yaml \
       --m0 ../klue_exaone/preds_klue_M0_s42.jsonl \
       --m1-glob "preds_M1_pin_s*.jsonl" --m2-glob "preds_M2_r05_pin_s*.jsonl" \
       --tag klue_r05_pin --h4-signal m1_conf --exclude-files
   python src/analysis/klue_m2_check.py \
       --metrics results/klue_exaone_outproj/metrics_followup_klue_r05_pin.json \
       --m0 results/klue_exaone/preds_klue_M0_s42.jsonl \
       --m1-glob "results/klue_exaone_outproj/preds_M1_pin_s*.jsonl" \
       --m2-glob "results/klue_exaone_outproj/preds_M2_r05_pin_s*.jsonl" \
       --out results/klue_exaone_outproj/klue_m2_check_r05_pin.json
   ```
   `--exclude-files`를 빈 값으로 주어 제외 목록을 쓰지 않는다(KLUE 제외 0건).

- 학습 중단·선점 시 같은 시드·같은 설정으로 재제출한다. 라이브러리 버전 검사나 LoRA 적용 검사가 실패하면 학습하지 않는다.
- 예상 소요: seed당 학습 약 2시간 15분, 추론 약 1시간 40분(KLUE M1 재학습 기준, L4).

## 7. 결과에 따른 다음 단계 (미리 정해 둠)

| 복원 | 비용 | 논문 서술 | 다음 단계 |
|---|---|---|---|
| 성립 (강건·조건부) | 없음 (강건·조건부) | EXAONE에서 easy UA 복원과 무비용이 KLUE에서도 재현. 1장 RQ3·기여 ③의 "복원의 도메인 일반성은 범위 밖"을 "EXAONE 한 모델에서 KLUE-MRC로 확인"으로 바꾼다 | hard UA는 M0 대비로만 서술 |
| 성립 | 있음 | 복원은 도메인을 넘어 재현, 비용은 도메인에 따라 달라짐. 비용 크기를 금융 text_span 기준(§3.4)과 나란히 보고 | 원인 해석은 하지 않고 관찰로 둔다. 추가 실험은 별도 결정 |
| 성립 | 미결 | 그대로 보고, 범주를 강제하지 않음 | 없음 |
| 불성립 | — | 혼입에 의한 복원이 도메인에 의존. RQ3 경계로 서술 | 학습 정답 틀·train_loss와 함께 해석. 추가 실험은 별도 결정 |

어느 경우든 결과는 보고한다. M0 대비 hard UA 복원 수치는 결과와 무관하게 보조 지표로 함께 싣는다.

## 8. 결과 (실행 후 작성)

### 8.1 실행 기록

- KLUE M0 예측 원본 위치: Modal 계정 `yoonjuseo08`, 볼륨 `selectiveqa-results/klue_exaone/` (2026-10-10 확인)
- M2-r05 학습·추론 실행 계정: (기입)

| seed | 학습 파라미터 | LoRA 적용 검사 | train_loss | 학습 시간 | 추론 시간 |
|---|---|---|---|---|---|
| 42 | | | | | |
| 43 | | | | | |
| 44 | | | | | |

### 8.2 판정

(`results/klue_exaone_outproj/klue_m2_check_r05_pin.json`)

### 8.3 해석
