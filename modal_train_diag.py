# -*- coding: utf-8 -*-
"""
modal_train_diag.py — train_qlora_diag.py(Tier1 진단 재학습)를 Modal GPU에서
실행하는 래퍼. modal_train.py와 동일한 이미지/Volume 구성을 재사용한다.

프로젝트 루트에서 실행 (--detach 필수):

  1~2단계 (EXAONE M2-r05, seed42, optimizer step마다 loss 분리 기록[사전등록 3절] +
           20스텝마다 어댑터 스냅샷):
    modal run --detach modal_train_diag.py --model exaone --train-file train_mix_r05.jsonl \
        --tag _r05_diag --seed 42 --ckpt-interval 20

  M1 진단 재학습 (사전등록 4.3절 ρ_content·G 기준선용 — 콜백 포함 M1 재학습):
    modal run --detach modal_train_diag.py --model exaone --train-file train.jsonl \
        --tag _m1_diag --seed 42 --ckpt-interval 20

  3단계 (그래디언트 클리핑 개입, 위와 동일하되 max-grad-norm 추가):
    modal run --detach modal_train_diag.py --model exaone --train-file train_mix_r05.jsonl \
        --tag _r05_diag_clip1 --seed 42 --ckpt-interval 20 --max-grad-norm 1.0

결과 회수 (exaone은 results_dir가 루트이므로 그대로):
    modal volume get selectiveqa-results tier1_loss_log_r05_diag_s42.jsonl ./results/tier1
    modal volume get selectiveqa-results qlora_adapter_r05_diag_s42 ./results/qlora_adapter_r05_diag_s42
    (스텝별 스냅샷은 qlora_adapter_r05_diag_s42_step20, _step40 ... 형태로 별도 폴더)

로컬에서 문서 첨부용 구간 확인(GPU 불필요, --check-spans만 실행):
    python src/training/train_qlora_diag.py --config config.yaml --check-spans 5
"""
import modal

app = modal.App("selectiveqa-train-diag")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("config.yaml", "/root/proj/config.yaml")
    .add_local_file("config_qwen3.yaml", "/root/proj/config_qwen3.yaml")
    .add_local_file("config_llama.yaml", "/root/proj/config_llama.yaml")
    .add_local_file("config_kanana.yaml", "/root/proj/config_kanana.yaml")
    .add_local_file("config_ax31.yaml", "/root/proj/config_ax31.yaml")
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
def train_diag(args: list):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    os.chdir("/root/proj")

    # 학습 시작 전, 이미지에 번들링된 스크립트의 스키마 버전을 먼저 확인한다.
    # 모델 로딩(수십 초~분)까지 기다릴 필요 없이 몇 초 안에 구버전 재배포 사고를
    # 잡아낼 수 있다 — 2026-09-24 M2-r05 진단 재학습에서 이걸 놓쳐 2시간짜리 실행을
    # 통째로 버린 적이 있어서 추가함.
    sys.path.insert(0, "src")
    from training.train_qlora_diag import SCHEMA_VERSION
    print(f"[스키마 확인] 번들링된 train_qlora_diag.py SCHEMA_VERSION={SCHEMA_VERSION}")

    cmd = [sys.executable, "src/training/train_qlora_diag.py"] + args
    print("실행:", " ".join(cmd))
    try:
        result = subprocess.run(cmd)
    finally:
        results_vol.commit()
    if result.returncode != 0:
        raise RuntimeError(f"진단 학습 스크립트 비정상 종료 (code={result.returncode})")

    # 학습이 정상 종료돼도, 실제로 저장된 로그가 새 스키마인지 다시 한번 확인한다
    # (스크립트 버전 확인만으로는 못 잡는, 예컨대 캐시된 이전 실행 파일 재사용 같은
    # 경우까지 대비).
    def arg_val(flag, default):
        return args[args.index(flag) + 1] if flag in args else default

    tag = arg_val("--tag", "_r05_diag")
    seed = arg_val("--seed", "42")
    results_dir = Path("results")
    # exaone 외 모델은 results_dir가 results/<model>일 수 있으나, 이 스크립트는
    # cfg["paths"]["results_dir"] 그대로 쓰므로 config에 맞춰 경로가 정해진다.
    import yaml
    cfg_path = arg_val("--config", "config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    log_path = Path(cfg["paths"]["results_dir"]) / f"tier1_loss_log{tag}_s{seed}.jsonl"

    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            first_line = f.readline()
        first_record = json.loads(first_line) if first_line.strip() else {}
        if first_record.get("schema_version") != SCHEMA_VERSION or "tok_ans_field" not in first_record:
            raise RuntimeError(
                f"학습은 끝났지만 저장된 로그({log_path})가 신규 스키마가 아닙니다 "
                f"(schema_version={first_record.get('schema_version')!r}). "
                "Q2/Q4 계산에 못 쓰는 로그이니 재배포 상태를 먼저 확인하세요."
            )
        print(f"[스키마 확인] 저장된 로그({log_path})가 신규 스키마 정상 확인됨 "
              f"— schema_version={first_record['schema_version']}, "
              f"tok_ans_field 키 존재")
    else:
        print(f"[경고] 예상 로그 경로를 찾지 못함: {log_path} (results_dir 설정을 확인하세요)")

    print("✓ 완료. `modal volume get selectiveqa-results <파일/폴더명> ./results` 로 회수하세요.")


@app.local_entrypoint()
def main(
    model: str = "exaone",
    config: str = None,
    train_file: str = "train_mix_r05.jsonl",
    tag: str = "_r05_diag",
    seed: int = 42,
    epochs: float = None,
    ckpt_interval: int = 20,
    max_grad_norm: float = None,
):
    cfg_path = config or CONFIG_MAP[model]
    args = [
        "--config", cfg_path,
        "--train-file", train_file,
        "--tag", tag,
        "--seed", str(seed),
        "--ckpt-interval", str(ckpt_interval),
    ]
    if epochs is not None:
        args += ["--epochs", str(epochs)]
    if max_grad_norm is not None:
        args += ["--max-grad-norm", str(max_grad_norm)]

    call = train_diag.spawn(args=args)
    print(f"작업 제출 완료 (function call id: {call.object_id})")
    print("진행 상황: https://modal.com/apps → selectiveqa-train-diag → App Logs")
