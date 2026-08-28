# -*- coding: utf-8 -*-
"""
modal_eval.py — evaluate_followup.py를 Modal CPU 컨테이너에서 실행 (GPU 불필요).

사전 준비 (제외 목록 2개를 Volume에 업로드, 최초 1회):
  modal volume put selectiveqa-results .\\results\\excluded_gold_v2.json excluded_gold_v2.json
  modal volume put selectiveqa-results .\\results\\excluded_gold_v2_manual.json excluded_gold_v2_manual.json

실행:
  modal run --detach modal_eval.py --h4-signal m1_conf

결과 회수 (완료 후):
  modal volume get selectiveqa-results metrics_followup_r05.json .\\results
  modal volume get selectiveqa-results risk_coverage_r05.png .\\results
"""
import modal

app = modal.App("selectiveqa-eval")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pyyaml", "matplotlib")
    .add_local_file("config.yaml", "/root/proj/config.yaml")
    .add_local_dir("src", "/root/proj/src")
)

results_vol = modal.Volume.from_name("selectiveqa-results")


@app.function(
    image=image,
    cpu=4.0,                      # 부트스트랩 가속용 (그래도 시간당 몇십 센트 수준)
    memory=4096,
    volumes={"/root/proj/results": results_vol},
    timeout=4 * 60 * 60,
)
def evaluate(h4_signal: str, n_boot: int | None, tag: str):
    import os
    import subprocess
    import sys

    os.chdir("/root/proj")
    cmd = [sys.executable, "src/evaluation/evaluate_followup.py",
           "--h4-signal", h4_signal, "--tag", tag]
    if n_boot is not None:
        cmd += ["--n-boot", str(n_boot)]
    print("실행:", " ".join(cmd))
    try:
        result = subprocess.run(cmd)
    finally:
        results_vol.commit()      # metrics json + png를 Volume에 보존
    if result.returncode != 0:
        raise RuntimeError(f"평가 스크립트 비정상 종료 (code={result.returncode})")
    print("✓ 판정 완료. metrics_followup_*.json / risk_coverage_*.png 를 회수하세요.")


@app.local_entrypoint()
def main(h4_signal: str = "m1_conf", n_boot: int = None, tag: str = "r05"):
    call = evaluate.spawn(h4_signal=h4_signal, n_boot=n_boot, tag=tag)
    print(f"작업 제출 완료 (function call id: {call.object_id})")
    print("진행 상황: modal.com 대시보드 → selectiveqa-eval → App Logs")