# -*- coding: utf-8 -*-
"""
modal_envcheck.py — 같은 EXAONE M0 NLL 을 두 라이브러리 환경에서 계산해 비교한다. (학습 없음)

  v4: Tier 0 / 추론 이미지와 같은 고정 버전 (transformers 4.57.1, peft 0.20.0 ...)
  v5: 현재 학습 이미지와 같은 정의 (requirements.txt 그대로 → 지금 캐시된 transformers 5.18)

사용 (레포 루트):  modal run modal_envcheck.py
기준값: Tier 0-a (4.57.1 환경) EXAONE M0  UA NLL 0.0995 / Answerable NLL 1.3145 (각 150건)
"""
import modal

app = modal.App("selectiveqa-envcheck")

_common = dict(python_version="3.12")
image_v5 = (  # modal_train.py 와 같은 정의 → 같은 캐시 이미지
    modal.Image.debian_slim(**_common)
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("config.yaml", "/root/proj/config.yaml")
    .add_local_dir("src", "/root/proj/src")
    .add_local_dir("data/processed", "/root/proj/data/processed")
)
image_v4 = (  # modal_diag_tier0.py / modal_inference.py 와 같은 고정 버전
    modal.Image.debian_slim(**_common)
    .pip_install_from_requirements("requirements.txt")
    .pip_install("transformers==4.57.1", "peft==0.20.0", "accelerate==1.14.0", "bitsandbytes==0.50.0")
    .add_local_file("config.yaml", "/root/proj/config.yaml")
    .add_local_dir("src", "/root/proj/src")
    .add_local_dir("data/processed", "/root/proj/data/processed")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
N = "50"   # UA/Answerable 각 50건 (기준값과 같은 앞쪽 문항부터)


def _run(tag):
    import importlib.metadata as md, json, os, subprocess, sys
    os.chdir("/root/proj")
    print(tag, {p: md.version(p) for p in ["transformers", "peft", "torch", "bitsandbytes"]})
    subprocess.run([sys.executable, "src/analysis/diag_nll.py", "--config", "config.yaml",
                    "--condition", "M0", "--n-samples", N, "--out-tag", f"envcheck_{tag}"], check=True)
    r = json.load(open(f"results/diag_nll_envcheck_{tag}.json", encoding="utf-8"))
    return tag, r["ua_nll_mean"], r["ans_nll_mean"], r["ans_nll_per_example"][:5]


@app.function(image=image_v5, gpu="L4", timeout=3600, volumes={"/root/.cache/huggingface": hf_cache})
def run_v5():
    return _run("v5")


@app.function(image=image_v4, gpu="L4", timeout=3600, volumes={"/root/.cache/huggingface": hf_cache})
def run_v4():
    return _run("v4")


@app.local_entrypoint()
def main():
    calls = [run_v4.spawn(), run_v5.spawn()]
    print("\n===== 결과 (기준: Tier 0-a 4.57.1 환경 UA 0.0995 / Answerable 1.3145, 150건) =====")
    for c in calls:
        tag, ua, ans, first5 = c.get()
        print(f"{tag}: UA NLL {ua:.4f} · Answerable NLL {ans:.4f} · 앞 5건 {[round(x, 3) for x in first5]}")
