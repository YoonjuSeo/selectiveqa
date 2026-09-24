# -*- coding: utf-8 -*-
"""
diag_nll.py — Tier 0(a): UA 타깃 vs Answerable 타깃의 per-token NLL을
M0(zero-shot)·M1(파인튜닝) 각각에서 측정해 "EXAONE에게 UA 타깃이 유독
낯선가(사전분포 거리)"를 4모델 비교로 검정한다.

가설: EXAONE M0에서 UA 타깃의 NLL이 다른 세 모델보다 유독 높다면(=canonical
UA JSON이 EXAONE의 사전학습 분포에서 특히 먼 지점이라면), 이것이 붕괴의
원인 후보 중 하나("EXAONE의 사전학습/정렬 분포 특이성")를 구체화하는
정량적 근거가 된다.

측정: 평가셋에서 gold_answerable=False인 문항 267건(hard UA) 중 일부를
샘플링해, 프롬프트+정답 UA JSON을 모델에 흘려(teacher forcing) 타깃
토큰들의 평균 NLL을 계산한다. 비교 기준으로 gold_answerable=True 문항의
정답 JSON에 대해서도 동일하게 측정(정상적인 학습 타깃과의 상대 거리).

사용법 (Modal):
  modal run modal_diag_tier0.py --task nll --model exaone --condition M0
  modal run modal_diag_tier0.py --task nll --model exaone --condition M1 --seed 42
  # model: exaone|qwen3|llama|kanana, condition: M0|M1

출력: results/diag_nll_{model}_{condition}.json
  {"ua_nll_mean":.., "ua_nll_per_example":[...], "ans_nll_mean":.., "ans_nll_per_example":[...],
   "gap": ua_nll_mean - ans_nll_mean, "n_ua":.., "n_ans":..}
  gap이 클수록(양수) UA 타깃이 answerable 타깃보다 이 모델에게 "더 낯설다"는 뜻.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from inference.prompts import build_messages, build_target, apply_template


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(cfg, condition, adapter_dir=None):
    model_name = cfg["model"]["base_model"]
    revision = cfg["model"].get("revision")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, revision=revision,
    )
    if condition == "M1":
        model = PeftModel.from_pretrained(model, adapter_dir)
        print(f"어댑터 로드: {adapter_dir}")
    model.eval()
    return model, tokenizer


@torch.no_grad()
def target_nll(model, tokenizer, context, question, target_json_str, enable_thinking=None):
    """target_json_str + eos 를 teacher forcing으로 흘려 타깃 부분의 평균 NLL(nats/token)을 계산."""
    messages = build_messages(context, question)
    prompt = apply_template(tokenizer, messages, enable_thinking)
    target_text = target_json_str + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([prompt_ids + target_ids]).to(model.device)
    labels = torch.tensor([[-100] * len(prompt_ids) + target_ids]).to(model.device)

    out = model(input_ids=input_ids, labels=labels)
    # HF의 기본 loss는 non-masked 토큰 평균 CE(=NLL, nats). 그대로 사용.
    return float(out.loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--condition", choices=["M0", "M1"], required=True)
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument("--eval-file", default="eval_full_v2.jsonl")
    ap.add_argument("--n-samples", type=int, default=150, help="UA/Answerable 각각 샘플 수")
    ap.add_argument("--out-tag", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = load_model(cfg, args.condition, args.adapter_dir)
    enable_thinking = cfg["model"].get("enable_thinking")

    eval_path = Path(cfg["paths"]["processed_dir"]) / args.eval_file
    rows = [json.loads(l) for l in open(eval_path, encoding="utf-8")]

    ua_rows = [r for r in rows if not r.get("answerable", True)][: args.n_samples]
    ans_rows = [r for r in rows if r.get("answerable", True)][: args.n_samples]

    from inference.prompts import UA_TARGETS
    import random
    random.seed(cfg["seed"])

    ua_nlls, ans_nlls = [], []
    for r in ua_rows:
        target = json.dumps(UA_TARGETS["null"], ensure_ascii=False)
        nll = target_nll(model, tokenizer, r["context"], r["question"], target, enable_thinking)
        ua_nlls.append(nll)

    for r in ans_rows:
        target = build_target(r["gold_answer"], answerable=True, ua_style="null")
        nll = target_nll(model, tokenizer, r["context"], r["question"], target, enable_thinking)
        ans_nlls.append(nll)

    import statistics as st
    result = {
        "model": cfg["model"]["base_model"],
        "condition": args.condition,
        "n_ua": len(ua_nlls), "n_ans": len(ans_nlls),
        "ua_nll_mean": st.mean(ua_nlls), "ua_nll_median": st.median(ua_nlls),
        "ans_nll_mean": st.mean(ans_nlls), "ans_nll_median": st.median(ans_nlls),
        "gap_mean": st.mean(ua_nlls) - st.mean(ans_nlls),
        "ua_nll_per_example": ua_nlls, "ans_nll_per_example": ans_nlls,
    }

    out_path = Path(cfg["paths"]["results_dir"]) / f"diag_nll_{args.out_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_path}")
    print(f"UA NLL mean={result['ua_nll_mean']:.4f}  Answerable NLL mean={result['ans_nll_mean']:.4f}  "
          f"gap={result['gap_mean']:+.4f}")


if __name__ == "__main__":
    main()
