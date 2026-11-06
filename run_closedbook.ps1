# closed-book 통제 실험 실행 순서 (Windows PowerShell, 레포 루트에서). 사전 계획: docs/prereg_closedbook_261106.md
# 모델마다 1) 버전 검사 → 2) 재현 관문 → 3) 본 실험(detach) → 4) 회수 → 5) 분석 순서로 진행한다.
# 재현 관문이 미통과(G1 answerable 일치 < 99% 또는 G2 정답 여부 일치 < 97% 또는 G3 신뢰도 평균 차 > 0.01, gate_check.py)면 그 모델의 3)을 실행하지 않는다.

$models = @(
  @{ name="EXAONE"; cfg="config_abl_exaone_outproj.yaml"; dir="exaone_outproj"; m1="preds_M1_pin_s*.jsonl"; m2="preds_M2_r05_pin_s*.jsonl" },
  @{ name="Qwen3";  cfg="config_qwen3.yaml";              dir="qwen3";          m1="preds_M1_v2_s*.jsonl";  m2="preds_M2_r05_s*.jsonl" },
  @{ name="Llama";  cfg="config_llama.yaml";              dir="llama";          m1="preds_M1_v2_s*.jsonl";  m2="preds_M2_r05_s*.jsonl" },
  @{ name="Kanana"; cfg="config_kanana.yaml";             dir="kanana";         m1="preds_M1_v2_s*.jsonl";  m2="preds_M2_r05_s*.jsonl" },
  @{ name="A.X";    cfg="config_ax31.yaml";               dir="ax31";           m1="preds_M1_v2_s*.jsonl";  m2="preds_M2_r05_s*.jsonl" }
)

# 0) 원 M0 예측이 Volume에 없으면 올린다 (관문 비교용, 최초 1회)
#   modal volume put selectiveqa-results .\results\qwen3\preds_M0_v2.jsonl qwen3/preds_M0_v2.jsonl

# 1) 버전 검사 (이미지 빌드 포함, 1회)
modal run modal_inference_cb.py --config config.yaml --check-only

foreach ($m in $models) {
  Write-Host "=== $($m.name): 재현 관문 ===" -ForegroundColor Cyan
  modal run modal_inference_cb.py --config $m.cfg --gate
  # 출력의 "→ 통과"를 확인한 뒤 본 실험 제출. 자동화하지 않고 사람이 확인한다(사전 계획 3절).
  Read-Host "관문 통과를 확인했으면 Enter (미통과면 Ctrl+C)"
  modal run --detach modal_inference_cb.py --config $m.cfg --mode force
}

# 4) 회수 (완료 후 모델마다)
# foreach ($m in $models) {
#   modal volume get selectiveqa-results "$($m.dir)/preds_M0_cb_force.jsonl"   ".\results\$($m.dir)" --force
#   modal volume get selectiveqa-results "$($m.dir)/env_closedbook_force.json"  ".\results\$($m.dir)" --force
#   modal volume get selectiveqa-results "$($m.dir)/env_closedbook_gate.json"   ".\results\$($m.dir)" --force
#   modal volume get selectiveqa-results "$($m.dir)/preds_M0_cbgate.jsonl"      ".\results\$($m.dir)" --force
# }

# 5) 로컬 분석 (GPU 불필요)
# foreach ($m in $models) {
#   python src\analysis\closedbook_analysis.py --config $m.cfg --m1-glob $m.m1 --m2-glob $m.m2
# }
