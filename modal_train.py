# -*- coding: utf-8 -*-
"""
modal_train.py — 기존 train_qlora.py를 수정 없이 Modal GPU에서 실행하는 래퍼.

프로젝트 루트에 두고 실행:

  스모크 (200건, 1에폭):
    modal run modal_train.py --train-file train_smoke.jsonl --epochs 1 --tag _smoke

  본 학습 (시드별):
    modal run modal_train.py --train-file train_mix_r05.jsonl --tag _r05 --seed 42

결과 회수:
    modal volume get selectiveqa-results / ./results_from_modal
"""
import modal

app = modal.App("selectiveqa-train")

# ── 이미지: requirements 설치 + 프로젝트 파일 포함 ──────────────────────
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    # 코드/설정/데이터를 컨테이너의 /root/proj 아래에 동일 구조로 복사
    .add_local_file("config.yaml", "/root/proj/config.yaml")
    .add_local_file("config_qwen3.yaml", "/root/proj/config_qwen3.yaml")
    .add_local_file("config_llama.yaml", "/root/proj/config_llama.yaml")
    .add_local_dir("src", "/root/proj/src")
    .add_local_dir("data/processed", "/root/proj/data/processed")
)

# ── Volume: 결과(어댑터) 영속화 + HF 모델 캐시 ──────────────────────────
results_vol = modal.Volume.from_name("selectiveqa-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(
    gpu="L4",                      # bf16 필요 → T4 불가. 부족하면 "A10G" 또는 "A100-40GB"
    image=image,
    volumes={
        "/root/proj/results": results_vol,        # config의 results_dir와 일치
        "/root/.cache/huggingface": hf_cache_vol,  # 7.8B 모델 재다운로드 방지
    },
    secrets=[modal.Secret.from_name("huggingface")], 
    timeout=6 * 60 * 60,           # 6시간 (본 학습 대비. 스모크는 금방 끝남)
)
def train(train_file: str, tag: str, seed: int | None, epochs: float | None, config: str):
    import os
    import subprocess
    import sys

    os.chdir("/root/proj")  # config.yaml의 상대경로가 그대로 동작하도록

    cmd = [
        sys.executable, "src/training/train_qlora.py",
        "--config", config,
        "--train-file", train_file,
        "--tag", tag,
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]

    print("실행:", " ".join(cmd))
    result = subprocess.run(cmd)

    results_vol.commit()  # 어댑터 저장분을 Volume에 반영
    if result.returncode != 0:
        raise RuntimeError(f"학습 스크립트 비정상 종료 (code={result.returncode})")
    print("✓ 학습 완료. 'modal volume get selectiveqa-results / ./results_from_modal' 로 회수하세요.")


@app.local_entrypoint()
def main(
    train_file: str = "train_smoke.jsonl",
    tag: str = "",
    seed: int = None,
    epochs: float = None,
    config: str = "config.yaml", 
):
    call = train.spawn(train_file=train_file, tag=tag, seed=seed, epochs=epochs, config=config)
    print(f"작업 제출 완료 (function call id: {call.object_id})")
    print("진행 상황: https://modal.com/apps → selectiveqa-train → App Logs")