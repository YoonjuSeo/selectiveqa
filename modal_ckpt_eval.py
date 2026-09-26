# -*- coding: utf-8 -*-
"""
modal_ckpt_eval.py — diag_ckpt_eval.py(Tier1 Q5: 스텝별 체크포인트 EM/응답길이/
M1 코사인)를 Modal GPU에서 실행하는 래퍼.

프로젝트 루트에서 실행 (--detach 필수, 체크포인트 19개 x 200문항 추론이라
시간이 꽤 걸림 — 중단돼도 완료된 step은 건너뛰고 이어감):

  modal run --detach modal_ckpt_eval.py --model exaone \
      --ckpt-glob "qlora_adapter_r05_diag_s42_step*" \
      --m1-adapter results/qlora_adapter_s42 --out-tag r05_diag_s42

결과 회수:
  modal volume get selectiveqa-results diag_ckpt_eval_r05_diag_s42.json ./results
"""
import modal

app = modal.App("selectiveqa-ckpt-eval")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .pip_install(
        # modal_inference.py / modal_diag_tier0.py와 동일하게 버전 고정.
        # requirements.txt의 느슨한 transformers>=4.44만 쓰면 EXAONE 커스텀
        # modeling 코드의 generate()가 cache_position=None으로 깨진다.
        "transformers==4.57.1",
        "peft==0.20.0",
        "accelerate==1.14.0",
        "bitsandbytes==0.50.0",
    )
    .add_local_file("config.yaml", "/root/proj/config.yaml")
    .add_local_file("config_qwen3.yaml", "/root/proj/config_qwen3.yaml")
    .add_local_file("config_llama.yaml", "/root/proj/config_llama.yaml")
    .add_local_file("config_kanana.yaml", "/root/proj/config_kanana.yaml")
    .add_local_dir("src", "/root/proj/src")
    .add_local_dir("data/processed", "/root/proj/data/processed")
)

results_vol = modal.Volume.from_name("selectiveqa-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

CONFIG_MAP = {
    "exaone": "config.yaml",
    "qwen3": "config_qwen3.yaml",
    "llama": "config_llama.yaml",
    "kanana": "config_kanana.yaml",
}


@app.function(
    gpu="L4",
    image=image,
    volumes={
        "/root/proj/results": results_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=8 * 60 * 60,
)
def ckpt_eval(args: list):
    import os
    import subprocess
    import sys

    os.chdir("/root/proj")
    cmd = [sys.executable, "src/analysis/diag_ckpt_eval.py"] + args
    print("실행:", " ".join(cmd))
    try:
        result = subprocess.run(cmd)
    finally:
        results_vol.commit()
    if result.returncode != 0:
        raise RuntimeError(f"체크포인트 평가 스크립트 비정상 종료 (code={result.returncode})")
    print("✓ 완료. `modal volume get selectiveqa-results diag_ckpt_eval_<태그>.json ./results` 로 회수하세요.")


@app.local_entrypoint()
def main(
    model: str = "exaone",
    config: str = None,
    ckpt_glob: str = "qlora_adapter_r05_diag_s42_step*",
    m1_adapter: str = "results/qlora_adapter_s42",
    eval_file: str = "eval_full_v2.jsonl",
    qids_file: str = None,
    out_tag: str = "r05_diag_s42",
    skip_cosine: bool = False,
):
    cfg_path = config or CONFIG_MAP[model]
    args = [
        "--config", cfg_path,
        "--ckpt-glob", ckpt_glob,
        "--m1-adapter", m1_adapter,
        "--eval-file", eval_file,
        "--out-tag", out_tag,
    ]
    if qids_file is not None:
        args += ["--qids-file", qids_file]
    if skip_cosine:
        args += ["--skip-cosine"]

    call = ckpt_eval.spawn(args=args)
    print(f"작업 제출 완료 (function call id: {call.object_id})")
    print("진행 상황: https://modal.com/apps → selectiveqa-ckpt-eval → App Logs")
