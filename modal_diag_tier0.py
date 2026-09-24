# -*- coding: utf-8 -*-
"""
modal_diag_tier0.py — Tier 0 진단(a·c·e)을 Modal GPU에서 실행하는 래퍼.
기존 modal_inference.py와 동일한 이미지/Volume을 재사용한다.

프로젝트 루트에 두고 실행 (반드시 --detach 로 본 실행):

  (a) UA 타깃 NLL — 4모델 × {M0, M1} 조합을 각각 실행
    modal run --detach modal_diag_tier0.py --task nll --model exaone --condition M0
    modal run --detach modal_diag_tier0.py --task nll --model exaone --condition M1 --seed 42
    modal run --detach modal_diag_tier0.py --task nll --model qwen3  --condition M0
    modal run --detach modal_diag_tier0.py --task nll --model qwen3  --condition M1 --seed 42
    modal run --detach modal_diag_tier0.py --task nll --model llama  --condition M0
    modal run --detach modal_diag_tier0.py --task nll --model llama  --condition M1 --seed 42
    modal run --detach modal_diag_tier0.py --task nll --model kanana --condition M0
    modal run --detach modal_diag_tier0.py --task nll --model kanana --condition M1 --seed 42
    # 결과: results/diag_nll_{model}_{condition}[_s{seed}].json

  (c) 어댑터 보간 (EXAONE, seed42, M1→M2-r05)
    modal run --detach modal_diag_tier0.py --task interp --model exaone \
        --m1-adapter results/qlora_adapter_s42 --m2-adapter results/qlora_adapter_r05_s42 \
        --out-tag exaone_s42
    # qids-file을 지정하려면 --qids-file results/d_broke_qids_s42.json (선택)

  (e) 프롬프트 개입 (EXAONE, seed42; M2로 본 실행 + M1로 대조군)
    modal run --detach modal_diag_tier0.py --task prompt_ablation --model exaone \
        --adapter results/qlora_adapter_r05_s42 --condition M2 --out-tag exaone_M2_s42
    modal run --detach modal_diag_tier0.py --task prompt_ablation --model exaone \
        --adapter results/qlora_adapter_s42 --condition M1 --out-tag exaone_M1_s42

결과 회수:
    modal volume get selectiveqa-results diag_nll_exaone_M0.json ./results
    (또는 results/ 디렉터리 통째로: modal volume get selectiveqa-results / ./results_from_modal --force)

주의:
  - --task interp 는 어댑터 두 개를 병합해 임시 어댑터를 5개(λ별) 만들고 각각 추론하므로
    다른 task보다 오래 걸린다(qids 200건 × 5λ ≈ M1 단독 추론의 ~5배).
  - config 파일은 --model 값에 따라 자동 매핑된다(아래 CONFIG_MAP). 다른 config를 쓰려면
    --config 로 직접 지정.
"""
import modal

app = modal.App("selectiveqa-diag-tier0")

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
    volumes={"/root/proj/results": results_vol, "/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=6 * 60 * 60,
)
def run_task(task: str, args: list):
    import os
    import subprocess
    import sys

    os.chdir("/root/proj")
    script_map = {
        "nll": "src/analysis/diag_nll.py",
        "interp": "src/analysis/diag_interp.py",
        "prompt_ablation": "src/analysis/diag_prompt_ablation.py",
    }
    cmd = [sys.executable, script_map[task]] + args
    print("실행:", " ".join(cmd))
    try:
        result = subprocess.run(cmd)
    finally:
        results_vol.commit()
    if result.returncode != 0:
        raise RuntimeError(f"진단 스크립트 비정상 종료 (code={result.returncode})")
    print("✓ 완료. `modal volume get selectiveqa-results <출력파일> ./results` 로 회수하세요.")


@app.local_entrypoint()
def main(
    task: str,                 # nll | interp | prompt_ablation
    model: str = "exaone",     # exaone | qwen3 | llama | kanana
    config: str = None,
    condition: str = "M0",     # nll/prompt_ablation 용: M0|M1|M2
    seed: int = None,
    adapter_dir: str = None,
    m1_adapter: str = None,
    m2_adapter: str = None,
    qids_file: str = None,
    eval_file: str = "eval_full_v2.jsonl",
    n_samples: int = 150,
    out_tag: str = None,
    lambdas: str = "0.0,0.25,0.5,0.75,1.0",
    wait: bool = False,
):
    cfg_path = config or CONFIG_MAP[model]
    args = ["--config", cfg_path, "--eval-file", eval_file]

    if task == "nll":
        args += ["--condition", condition, "--n-samples", str(n_samples)]
        tag = out_tag or f"{model}_{condition}" + (f"_s{seed}" if seed else "")
        args += ["--out-tag", tag]
        if condition == "M1":
            if not adapter_dir:
                raise ValueError("--condition M1 에는 --adapter-dir 필요")
            args += ["--adapter-dir", adapter_dir]

    elif task == "interp":
        if not (m1_adapter and m2_adapter):
            raise ValueError("--task interp 에는 --m1-adapter, --m2-adapter 필요")
        args += ["--m1-adapter", m1_adapter, "--m2-adapter", m2_adapter, "--lambdas", lambdas]
        if qids_file:
            args += ["--qids-file", qids_file]
        args += ["--out-tag", out_tag or f"{model}_interp"]

    elif task == "prompt_ablation":
        if not adapter_dir:
            raise ValueError("--task prompt_ablation 에는 --adapter-dir 필요")
        args += ["--condition", condition, "--adapter-dir", adapter_dir,
                 "--n-samples", str(n_samples), "--out-tag", out_tag or f"{model}_{condition}_promptabl"]
    else:
        raise ValueError(f"알 수 없는 task: {task}")

    if wait:
        run_task.remote(task=task, args=args)
    else:
        call = run_task.spawn(task=task, args=args)
        print(f"작업 제출 완료 (function call id: {call.object_id})")
        print("진행 상황: https://modal.com/apps → selectiveqa-diag-tier0 → App Logs")
