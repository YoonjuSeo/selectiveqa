# -*- coding: utf-8 -*-
"""
modal_infer.py — 기존 run_inference.py를 수정 없이 Modal GPU에서 실행하는 래퍼.

프로젝트 루트에 두고 실행:

  스모크 (10건만):
    modal run modal_infer.py --condition M1 --adapter-dir results/qlora_adapter_r05_s42 --out-tag M2_r05_s42 --eval-file eval_full_v2.jsonl --limit 10

  본 추론 (M2-r05, 시드별):
    modal run --detach modal_infer.py --condition M1 --adapter-dir results/qlora_adapter_r05_s42 --out-tag M2_r05_s42 --eval-file eval_full_v2.jsonl

  M0 베이스라인 (필요 시):
    modal run --detach modal_infer.py --condition M0 --eval-file eval_full_v2.jsonl

결과 회수:
    modal volume get selectiveqa-results preds_M2_r05_s42.jsonl .\\results
"""
import modal

app = modal.App("selectiveqa-infer")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .pip_install(
        "transformers==4.57.1",
        "peft==0.20.0",
        "accelerate==1.14.0",
        "bitsandbytes==0.50.0",
    )
    .add_local_file("config.yaml", "/root/proj/config.yaml")
    .add_local_file("config_qwen3.yaml", "/root/proj/config_qwen3.yaml")
    .add_local_dir("src", "/root/proj/src")
    .add_local_dir("data/processed", "/root/proj/data/processed")
)

# 학습과 동일한 Volume 재사용:
#  - results: 어댑터 읽기(qlora_adapter_r05_s*) + preds 쓰기 → 한 곳에서 해결
#  - hf-cache: EXAONE 7.8B 캐시 → 모델 로딩 수 초
results_vol = modal.Volume.from_name("selectiveqa-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    gpu="L4",
    image=image,
    volumes={
        "/root/proj/results": results_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    timeout=6 * 60 * 60,
)
def infer(condition: str, eval_file: str, adapter_dir: str | None,
          out_tag: str | None, seed: int | None, limit: int | None, fresh: bool, config: str):
    import os
    import subprocess
    import sys

    os.chdir("/root/proj")

    cmd = [
        sys.executable, "src/inference/run_inference.py",
        "--config", config,
        "--condition", condition,
        "--eval-file", eval_file,
    ]
    if adapter_dir:
        cmd += ["--adapter-dir", adapter_dir]
    if out_tag:
        cmd += ["--out-tag", out_tag]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if fresh:
        cmd += ["--fresh"]

    print("실행:", " ".join(cmd))
    try:
        result = subprocess.run(cmd)
    finally:
        results_vol.commit()   # 비정상 종료여도 그 시점까지의 preds를 Volume에 보존 (resume 대비)

    if result.returncode != 0:
        raise RuntimeError(f"추론 스크립트 비정상 종료 (code={result.returncode})")
    print("✓ 추론 완료. 'modal volume get selectiveqa-results preds_<태그>.jsonl .\\results' 로 회수하세요.")


@app.local_entrypoint()
def main(
    condition: str = "M1",
    eval_file: str = "eval_full_v2.jsonl",
    adapter_dir: str = None,
    out_tag: str = None,
    seed: int = None,
    limit: int = None,
    fresh: bool = False,
    wait: bool = False,
    config: str = "config.yaml",  
):
    if wait:
        # 스모크용: 로그를 터미널에서 직접 보며 완료까지 대기 (detach 불필요)
        infer.remote(condition=condition, eval_file=eval_file,
                     adapter_dir=adapter_dir, out_tag=out_tag,
                     seed=seed, limit=limit, fresh=fresh, config=config)   
    else:
        # 본 실행용: 제출 후 즉시 종료 (반드시 --detach와 함께)
        call = infer.spawn(condition=condition, eval_file=eval_file,
                           adapter_dir=adapter_dir, out_tag=out_tag,
                           seed=seed, limit=limit, fresh=fresh,
                           config=config)   
        print(f"작업 제출 완료 (function call id: {call.object_id})")
        print("진행 상황: https://modal.com/apps → selectiveqa-infer → App Logs")