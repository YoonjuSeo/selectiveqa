# -*- coding: utf-8 -*-
"""
modal_inference_cb.py — closed-book 통제 실험(사전지식 점검) 전용 Modal 래퍼.
사전 계획: docs/prereg_closedbook_261106.md

왜 modal_inference.py를 그대로 쓰지 않는가:
  modal_inference.py의 이미지는 requirements.txt(최소 버전)로 torch를 설치해 torch가 고정되지 않는다.
  이 래퍼는 modal_train_pinned.py와 **같은 PINNED 버전·같은 이미지 정의**를 써서 다섯 모델이 모두
  results/env_versions.txt(원 실험 환경)와 같은 라이브러리에서 추론되도록 하고, 실행 전에 버전을
  검사해 하나라도 다르면 추론하지 않고 중단한다. run_inference.py는 수정 없이 그대로 호출한다.

절차 (레포 루트에서, 모델마다):
  0) 버전 검사만:           modal run modal_inference_cb.py --config config_qwen3.yaml --check-only
  1) 재현 관문 (GPU, ~5분): modal run modal_inference_cb.py --config config_qwen3.yaml --gate
       같은 이미지에서 **정상 프롬프트** M0 앞 100건을 돌려 원 preds_M0_v2.jsonl과 비교한다.
       통과 기준(사전 계획): answer 일치 ≥ 98%, answerable 판단 일치 ≥ 99%. 미통과면 본 실험 금지.
  2) 본 실험 (--detach):    modal run --detach modal_inference_cb.py --config config_qwen3.yaml --mode force
       → results/<results_dir>/preds_M0_cb_force.jsonl (+ env_closedbook_force.json 버전 기록)
  3) 탐색적(선택):          modal run --detach modal_inference_cb.py --config config_qwen3.yaml --mode abstain
  4) M1 어댑터로도 보려면:  --condition M1 --adapter-dir results/<dir>/qlora_adapter_s42 --out-tag M1_cb_force_s42

회수:
  modal volume get selectiveqa-results <results_dir>/preds_M0_cb_force.jsonl .\\results\\<results_dir>
  modal volume get selectiveqa-results <results_dir>/env_closedbook_force.json .\\results\\<results_dir>
분석(로컬 CPU): python src/analysis/closedbook_analysis.py --config config_qwen3.yaml
"""
import modal

app = modal.App("selectiveqa-infer-cb")

PINNED = {  # results/env_versions.txt — modal_train_pinned.py와 동일하게 유지할 것
    "torch": "2.8.0",
    "transformers": "4.57.1",
    "peft": "0.20.0",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.0",
}
CONFIGS = [  # 다섯 모델 + 변형 설정 전부 포함 (어느 config를 넘겨도 이미지 안에 있도록)
    "config.yaml", "config_abl_exaone_outproj.yaml", "config_abl_exaone_qkv_pin.yaml",
    "config_qwen3.yaml", "config_llama.yaml", "config_kanana.yaml", "config_ax31.yaml",
    "config_klue_exaone.yaml", "config_klue_exaone_outproj.yaml", "config_klue_kanana.yaml",
]

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    *[f"{k}=={v}" for k, v in PINNED.items()],
    "pyyaml", "numpy", "tqdm", "safetensors", "sentencepiece",   # modal_train_pinned.py와 같은 목록·순서 → 같은 캐시 레이어
)
for _c in CONFIGS:
    image = image.add_local_file(_c, f"/root/proj/{_c}")
image = (image
         .add_local_dir("src", "/root/proj/src")
         .add_local_dir("data/processed", "/root/proj/data/processed"))

results_vol = modal.Volume.from_name("selectiveqa-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

GATE_LIMIT = 100          # 재현 관문: 앞 100건 (BOS 점검 2단계와 동일)
# 관문 기준(G1 answerable 일치 >=0.99, G2 정답 여부 일치 >=0.97, G3 신뢰도 평균 차 <=0.01)은
# src/analysis/gate_check.py에 있다 (사전 계획 3절, 2026-11-06 개정).


def _check_versions():
    """PINNED와 설치 버전이 하나라도 다르면 예외. 반환: 실제 버전 dict."""
    import importlib.metadata as md
    actual = {k: md.version(k) for k in PINNED}
    bad = {k: (v, PINNED[k]) for k, v in actual.items() if not v.startswith(PINNED[k])}
    print("라이브러리 버전:", actual)
    if bad:
        raise RuntimeError(f"버전 불일치 → 중단 (실제, 기대): {bad}")
    return actual


def _env_record(cfg_path, extra):
    import importlib.metadata as md, json, platform, subprocess, sys, datetime, yaml
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    rec = {
        "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "versions": {k: md.version(k) for k in PINNED},
        "base_model": cfg["model"]["base_model"], "revision": cfg["model"].get("revision"),
        "enable_thinking": cfg["model"].get("enable_thinking"),
        "max_new_tokens": cfg["inference"]["max_new_tokens"],
        "decoding": "greedy (do_sample=False)",
        **extra,
    }
    try:
        rec["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                           capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return rec


@app.function(gpu="L4", image=image,
              volumes={"/root/proj/results": results_vol, "/root/.cache/huggingface": hf_cache_vol},
              secrets=[modal.Secret.from_name("huggingface")], timeout=6 * 60 * 60)
def infer_cb(config: str, mode: str, condition: str, adapter_dir: str | None, out_tag: str | None,
             eval_file: str, limit: int | None, fresh: bool, dedupe_bos: bool):
    import json, os, subprocess, sys, yaml
    os.chdir("/root/proj")
    _check_versions()
    cfg = yaml.safe_load(open(config, encoding="utf-8"))
    res_dir = cfg["paths"]["results_dir"]
    tag = out_tag or f"{condition}_cb_{mode}"
    cmd = [sys.executable, "src/inference/run_inference.py", "--config", config, "--condition", condition,
           "--eval-file", eval_file, "--closed-book", mode, "--out-tag", tag]
    if adapter_dir: cmd += ["--adapter-dir", adapter_dir]
    if limit is not None: cmd += ["--limit", str(limit)]
    if fresh: cmd += ["--fresh"]
    if dedupe_bos: cmd += ["--dedupe-bos"]
    print("실행:", " ".join(cmd))
    os.makedirs(res_dir, exist_ok=True)
    with open(f"{res_dir}/env_closedbook_{mode}.json", "w", encoding="utf-8") as f:
        json.dump(_env_record(config, {"condition": condition, "mode": mode, "eval_file": eval_file,
                                       "adapter_dir": adapter_dir, "out_tag": tag, "dedupe_bos": dedupe_bos,
                                       "cmd": cmd}), f, ensure_ascii=False, indent=2)
    try:
        result = subprocess.run(cmd)
    finally:
        results_vol.commit()
    if result.returncode != 0:
        raise RuntimeError(f"추론 비정상 종료 (code={result.returncode})")
    print(f"✓ 완료: {res_dir}/preds_{tag}.jsonl")


@app.function(gpu="L4", image=image,
              volumes={"/root/proj/results": results_vol, "/root/.cache/huggingface": hf_cache_vol},
              secrets=[modal.Secret.from_name("huggingface")], timeout=2 * 60 * 60)
def gate_fn(config: str, eval_file: str, ref_file: str, dedupe_bos: bool):
    """재현 관문: 같은 이미지에서 정상 프롬프트 M0 앞 GATE_LIMIT건 → 원 예측과 일치율 비교."""
    import json, os, subprocess, sys, yaml
    os.chdir("/root/proj")
    _check_versions()
    cfg = yaml.safe_load(open(config, encoding="utf-8"))
    res_dir = cfg["paths"]["results_dir"]
    tag = "M0_cbgate"
    cmd = [sys.executable, "src/inference/run_inference.py", "--config", config, "--condition", "M0",
           "--eval-file", eval_file, "--limit", str(GATE_LIMIT), "--out-tag", tag, "--fresh"]
    if dedupe_bos: cmd += ["--dedupe-bos"]
    print("실행:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    finally:
        results_vol.commit()
    ref = f"{res_dir}/{ref_file}"
    if not os.path.exists(ref):
        raise RuntimeError(f"원 예측 파일이 Volume에 없음: {ref} — 'modal volume put selectiveqa-results "
                           f"results/{ref} {ref}' 로 먼저 올리세요")
    sys.path.insert(0, "/root/proj/src/analysis")
    from gate_check import gate_metrics, _load
    g = gate_metrics(_load(ref), _load(f"{res_dir}/preds_{tag}.jsonl"), cfg["eval"]["numeric_tolerance"])
    g["ref"] = ref
    verdict = g["pass"]
    rec = _env_record(config, {"gate": g})
    with open(f"{res_dir}/env_closedbook_gate.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    results_vol.commit()
    print(f"재현 관문: 공통 {g['n']}건 · answerable 일치 {g['answerable_match']:.3f} · 정답 여부 일치 {g['correct_agree']:.3f} "
          f"· 신뢰도 평균 차 {g['mean_abs_conf_diff']:.4f} (참고: 문자열 일치 {g['answer_exact_match']:.3f}, 최대 차 {g['max_abs_conf_diff']:.4f})")
    print("→ 통과: closed-book 본 실험 진행 가능" if verdict else "→ 미통과: 환경 차이 의심. 본 실험 금지(사전 계획 3절)")
    return rec["gate"]


@app.function(image=image, timeout=600)
def check_only_fn():
    return _check_versions()


@app.local_entrypoint()
def main(config: str = "config.yaml", mode: str = "force", condition: str = "M0",
         adapter_dir: str = None, out_tag: str = None, eval_file: str = "eval_full_v2.jsonl",
         limit: int = None, fresh: bool = False, dedupe_bos: bool = False,
         gate_ref: str = "preds_M0_v2.jsonl", check_only: bool = False, gate: bool = False,
         wait: bool = False):
    # modal run 옵션명: --check-only, --gate, --wait, --dedupe-bos 등 (하이픈)
    if check_only:
        print(check_only_fn.remote()); return
    if gate:
        print(gate_fn.remote(config=config, eval_file=eval_file, ref_file=gate_ref, dedupe_bos=dedupe_bos)); return
    kw = dict(config=config, mode=mode, condition=condition, adapter_dir=adapter_dir, out_tag=out_tag,
              eval_file=eval_file, limit=limit, fresh=fresh, dedupe_bos=dedupe_bos)
    if wait:
        infer_cb.remote(**kw)
    else:
        call = infer_cb.spawn(**kw)
        print(f"작업 제출 완료 (function call id: {call.object_id}) — --detach와 함께 실행했는지 확인")
