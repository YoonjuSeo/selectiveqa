# -*- coding: utf-8 -*-
"""
modal_train_pinned.py — train_qlora.py 를 "원 실험과 같은 라이브러리 버전"으로 고정한 이미지에서 실행.

배경 (2026-10-02): modal_train.py 의 이미지는 requirements.txt(최소 버전만 지정)로 빌드되어
transformers 5.x 가 설치되었다. EXAONE-3.5 의 원격 코드(modeling_exaone.py)는 5.x 와 호환되지 않아
(DynamicCache.from_legacy_cache 제거 등) 학습 loss 가 원 실험의 약 8배로 나왔다.
이 래퍼는 원 실험 기록(results/env_versions.txt)의 버전으로 고정한다. 기존 modal_train.py 는 수정하지 않는다.

사용 (레포 루트, --detach 필수):
  modal run --detach modal_train_pinned.py --train-file train.jsonl          --tag _pin     --seed 42 --config config_abl_exaone_outproj.yaml
  modal run --detach modal_train_pinned.py --train-file train_mix_r05.jsonl --tag _r05_pin --seed 42 --config config_abl_exaone_outproj.yaml
"""
import modal

app = modal.App("selectiveqa-train-pinned")

PINNED = {  # results/env_versions.txt (원 실험 학습 환경)
    "torch": "2.8.0",
    "transformers": "4.57.1",
    "peft": "0.20.0",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.0",
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*[f"{k}=={v}" for k, v in PINNED.items()],
                 "pyyaml", "numpy", "tqdm", "safetensors", "sentencepiece")
    .add_local_file("config.yaml", "/root/proj/config.yaml")
    .add_local_file("config_abl_exaone_outproj.yaml", "/root/proj/config_abl_exaone_outproj.yaml")
    .add_local_file("config_klue_exaone_outproj.yaml", "/root/proj/config_klue_exaone_outproj.yaml")  # 2026-10-04 KLUE M1 재학습
    .add_local_dir("src", "/root/proj/src")
    .add_local_dir("data/processed", "/root/proj/data/processed")
)

results_vol = modal.Volume.from_name("selectiveqa-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)


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
def train(train_file: str, tag: str, seed: int | None, epochs: float | None, config: str):
    import importlib.metadata as md
    import os
    import subprocess
    import sys

    # 버전 확인: 하나라도 다르면 학습하지 않고 중단
    actual = {k: md.version(k) for k in PINNED}
    print("라이브러리 버전:", actual)
    bad = {k: (v, actual[k]) for k, v in PINNED.items() if not actual[k].startswith(v)}
    if bad:
        raise RuntimeError(f"버전 불일치(기대, 실제): {bad}")

    os.chdir("/root/proj")
    cmd = [sys.executable, "src/training/train_qlora.py",
           "--config", config, "--train-file", train_file, "--tag", tag]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    print("실행:", " ".join(cmd))
    result = subprocess.run(cmd)
    results_vol.commit()
    if result.returncode != 0:
        raise RuntimeError(f"학습 스크립트 비정상 종료 (code={result.returncode})")
    print("✓ 학습 완료.")


@app.local_entrypoint()
def main(train_file: str = "train.jsonl", tag: str = "", seed: int = None,
         epochs: float = None, config: str = "config.yaml"):
    call = train.spawn(train_file=train_file, tag=tag, seed=seed, epochs=epochs, config=config)
    print(f"작업 제출 완료 (function call id: {call.object_id})")
    print("진행 상황: modal.com → selectiveqa-train-pinned → App Logs")
