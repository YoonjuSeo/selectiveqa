# 사전등록 보완 — KLUE-MRC EXAONE M1 재학습 (LoRA 대상 모듈 수정)

작성일 2026-10-04 · KLUE M1(out_proj) 학습 제출 전 커밋
상위 문서: `docs/prereg_phase2_klue_260920.md` (Phase 2 원 사전등록), `prereg_exaone_outproj_261002.md` (금융 out_proj 절제)

## 1. 배경

Phase 2의 KLUE EXAONE M1(정답 수정본 klue_m1fix)에서는 무응답이 소거되지 않았다(UA 무응답 75~76%,
M0 77.2%). 출력도 학습 정답 틀(`evidence_span` = 정답)을 따르지 않고 지문 문장을 옮겼다. 보고서
4.8절은 이를 "소거는 학습 정답 틀에 대한 적합을 전제로 한다"로 해석했고, 적합이 왜 일어나지
않았는지는 분리하지 못했다.

`config_klue_exaone.yaml`에는 `lora_target_modules` 키가 없어 학습 코드 기본 목록(q/k/v/o_proj)이
쓰였다. EXAONE의 attention 출력층 이름은 `out_proj`라서 실제로는 q/k/v 3종(96모듈)에만 LoRA가
적용됐다. 금융에서는 같은 누락이 ΔEM −0.34 붕괴를 만들었고, out_proj를 넣으면 3시드 모두
사라졌다(2026-10-03, P2 3/3). 따라서 KLUE의 "적합 실패"도 이 설정에서 나왔을 가능성을 배제할 수 없다.

## 2. 질문

**LoRA를 q/k/v/out_proj 4종에 적용하면 KLUE에서도 M1 무응답 소거가 일어나는가.**

- 일어나면: 원 KLUE의 무소거는 설정에서 나왔을 가능성이 크고, 4.8절 결론("소거는 적합을
  전제로 한다", 사실상 "KLUE에서는 소거 없음")을 고쳐야 한다. KLUE M2-r05가 다시 의미를 가진다.
- 일어나지 않으면: 4.8절 결론이 올바른 설정에서 재확인된다.

Phase 2 원래 질문("EXAONE 붕괴가 도메인을 떠나도 재현되는가")은 붕괴가 설정 오류로 밝혀져
성립하지 않는다. 원 사전등록 §3의 붕괴 판별 규칙과 비교 기준(금융 EXAONE text_span ΔEM −0.303)은
이번 실행에 쓰지 않는다.

## 3. 판정 기준 (결과 확인 전 고정)

**1차 종점: M1의 easy UA 무응답률** (`answerable_pred is False`, n=300).

easy UA를 쓰는 이유: KLUE easy UA는 금융과 같은 방식(교차 지문)으로 만들었다(원 사전등록 D4).
hard UA는 KLUE 네이티브 `is_impossible`이라 금융 hard와 종류가 다르다(D5). 금융 M1의 소거도
easy UA 환각률 100%(H1의 M1 쪽)로 확인했다.

| 판정 (시드별) | 조건 |
|---|---|
| 소거 성립 | easy UA 무응답률 ≤ 0.10 |
| 소거 없음 | easy UA 무응답률 ≥ 0.60 (원 KLUE 결과 재현) |
| 부분 소거 | 그 사이 |

- 문턱 근거: 금융 M1은 5모델 모두 0~0.6%, 원 KLUE M1은 75~76%, M0는 77.2%. 두 관측 띠 바깥에 잡았다.
- 집계: 3시드 같은 판정이면 강건, 2/3이면 조건부, 그 외는 미결.
- 판정 코드: `src/analysis/klue_m1_check.py`(이 문서와 함께 커밋). 문턱 상수는 이 문서와 같다.

**보조 지표 (판정에 쓰지 않고 함께 보고):** hard·전체 UA 무응답률, 응답가능 EM(max-over-golds),
false abstention, 정답 틀 적합률(응답한 응답가능 문항 중 `evidence_span` = `answer` 비율),
응답가능 평균 생성 토큰, H3 정의의 AUROC(응답가능 정답 vs hard UA 환각), train_loss.

**비교 가능성:** 원 KLUE M1 결과는 저장소에 없는 `m1fix_eval.py`로 산출했다. 같은 정의로
비교하기 위해 원 KLUE M1 예측 파일에도 `klue_m1_check.py`를 적용해 함께 보고한다.
원 예측 파일에 `gen_text`가 없으면 정답 틀 적합률은 원 쪽만 비워 둔다.

## 4. 원본 대비 이탈 사항

| 항목 | 원 KLUE M1 (klue_m1fix) | 이번 실행 |
|---|---|---|
| LoRA 대상 모듈 | q/k/v (out_proj 누락, 96모듈) | q/k/v/**out_proj** (128모듈, 학습 코드의 적용 층 수 검사로 확인) |
| config | `config_klue_exaone.yaml` | `config_klue_exaone_outproj.yaml` (★OP 3곳만 다름) |
| 학습 래퍼 | `modal_train.py` (requirements.txt 최소 버전, **실제 버전 기록 없음**) | `modal_train_pinned.py` (torch 2.8.0 / transformers 4.57.1 / peft 0.20.0 / accelerate 1.14.0 / bitsandbytes 0.50.0 고정) |
| 결과 경로 | `results/klue_exaone/` | `results/klue_exaone_outproj/` |
| 판정 스크립트 | `m1fix_eval.py` (저장소 밖) | `src/analysis/klue_m1_check.py` |
| 학습 데이터 | `klue_train_str.jsonl` | 동일 (sha256 `6f117c90…99f5671`, 3,000건) |
| 평가 데이터 | `klue_eval_full.jsonl` | 동일 (sha256 `ee95d815…7c97d952`, 1,767건) |
| M0 | `results/klue_exaone/preds_M0*.jsonl` | 동일 (LoRA 무관, 재추론 안 함) |

나머지(r16/α32, dropout 0.05, 2 epoch, lr 2e-4, 실효 배치 16, max_seq_len 2048, 모델 리비전
0ff6b5ec7c13, greedy, max_new_tokens 128, 시드 42/43/44)는 원 KLUE와 같다.

**주의 — 변수가 둘일 수 있다.** 원 KLUE M1의 라이브러리 버전은 기록되지 않았다.
`modal_train.py` 이미지가 2026-10-02 금융 1차 시도처럼 transformers 5.x로 빌드됐다면, 이번
실행은 대상 모듈과 라이브러리 버전 두 가지를 함께 바꾼 것이 된다. 이 경우 결과가 달라져도
out_proj 하나의 효과로 귀속하지 않는다(원 버그판 loss 0.635는 금융 5.x 1차 시도 loss 0.778과
비슷한 크기라 이 가능성을 무시할 수 없다). 원 학습 로그나 Modal 이미지 기록으로 버전을 확인할 수
있으면 결과란에 적는다.

## 5. 판정 격 — 사전 지정 탐색적 판정

기준을 정하기 전에 아래 결과를 알고 있다.

- 금융 EXAONE(out_proj) M1: easy UA 무응답 0% (3시드), 소거 성립
- 원 KLUE EXAONE M1(q/k/v): UA 무응답 75~76%, 정답 틀 미적합, 신뢰도 AUROC 0.58~0.59
- KLUE M0: text_span UA 무응답 77.2%

판정 기준은 위 결과를 보고 정했으므로(문턱을 두 관측 띠 바깥에 둠) 확증적 판정이 아니라
사전 지정 탐색적 판정으로 보고한다. KLUE M1(out_proj) 결과는 어떤 것도 보지 않았다.

## 6. 실행 범위와 순서

- 학습: seed 42 먼저 → 학습 파라미터 13,631,488개, LoRA 적용 검사 통과(4종 × 32층), loss가
  발산하지 않았는지만 확인하고 seed 43·44 제출. **seed 42의 평가 지표를 보고 43·44 실행 여부를
  바꾸지 않는다.** 3시드 모두 실행한다.
- 명령:
  - 학습 `modal run --detach modal_train_pinned.py --train-file klue_train_str.jsonl --tag _pin --seed {seed} --config config_klue_exaone_outproj.yaml`
  - 추론 `modal run --detach modal_inference.py --condition M1 --adapter-dir results/klue_exaone_outproj/qlora_adapter_pin_s{seed} --out-tag M1_pin_s{seed} --eval-file klue_eval_full.jsonl --config config_klue_exaone_outproj.yaml`
  - 판정 `python src/analysis/klue_m1_check.py results/klue_exaone_outproj/preds_M1_pin_s*.jsonl --out results/klue_exaone_outproj/klue_m1_check_pin.json`
- 학습 중단·선점 시 같은 시드·같은 설정으로 재제출한다. 라이브러리 버전 검사나 LoRA 적용 검사가
  실패하면 학습하지 않는다.

## 7. 결과에 따른 다음 단계 (미리 정해 둠)

| 집계 결과 | 4.8절 서술 | 다음 단계 |
|---|---|---|
| 소거 성립 (강건·조건부) | 원 KLUE 무소거는 설정(대상 모듈 ± 라이브러리 버전)에서 나왔을 가능성. 소거는 금융 밖에서도 일어난다 | KLUE M2-r05 사전등록 보완(비교 기준을 금융 out_proj text_span으로 교체) 후 실행 |
| 소거 없음 (강건·조건부) | 올바른 설정에서도 KLUE는 소거되지 않음. 기존 결론 유지, 설정 확인을 각주로 | Phase 2 종료 유지 |
| 부분 소거·미결 | 그대로 보고, 범주를 강제하지 않음 | 정답 틀 적합률·train_loss와 함께 해석. 추가 실험은 별도 결정 |

원 KLUE M1(q/k/v) 결과는 어느 경우든 부록 "원 설정(q/k/v) 결과"로 옮긴다.

## 8. 결과 (실행 후 작성)

(비워둠)
