# -*- coding: utf-8 -*-
"""
diag_interp.py — Tier 0(c): M1 어댑터와 M2 어댑터를 선형 보간(λ)해
붕괴가 λ축에서 급격한 상전이로 나타나는지, 완만한 변화인지를 확인한다.

merged(λ) = M1 + λ * (M2 - M1)   (LoRA A/B 각 행렬을 텐서 단위로 직접 보간)

λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0} 각각에서 병합 어댑터를 만들어 (b)에서
"M1 correct → M2 broke"로 확인된 문항 집합(/tmp 산출물 d_token_divergence_raw.json
참조, 또는 --qids-file로 직접 지정) 위주로 소규모 추론을 돌려 EM과 평균
답변 길이를 본다.

사용법 (Modal):
  modal run modal_diag_tier0.py --task interp --model exaone \
      --m1-adapter results/qlora_adapter_s42 \
      --m2-adapter results/qlora_adapter_r05_s42 \
      --qids-file results/d_broke_qids_s42.json   # (d) 분석에서 만든 붕괴 문항 id 목록

출력: results/diag_interp_{tag}.json — λ별 {em, mean_ntok, n}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import torch
import yaml
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from inference.prompts import build_messages, parse_model_output, apply_template
from evaluation.evaluate import is_correct


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_interpolated_adapter(m1_dir, m2_dir, lam, out_dir):
    """m1_dir/adapter_model.safetensors 와 m2_dir/adapter_model.safetensors 를
    텐서 단위로 선형 보간해 out_dir 에 새 어댑터를 만든다 (config/tokenizer는 m1에서 복사)."""
    import shutil
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["adapter_config.json", "chat_template.jinja",
                  "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "vocab.json", "merges.txt"]:
        src = Path(m1_dir) / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)

    w1 = load_file(Path(m1_dir) / "adapter_model.safetensors")
    w2 = load_file(Path(m2_dir) / "adapter_model.safetensors")
    merged = {}
    for k in w1:
        if k not in w2:
            merged[k] = w1[k]
            continue
        merged[k] = (w1[k].float() + lam * (w2[k].float() - w1[k].float())).to(w1[k].dtype)
    save_file(merged, out_dir / "adapter_model.safetensors")
    return out_dir


def load_base(cfg):
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
    return model, tokenizer


@torch.no_grad()
def predict_one(model, tokenizer, context, question, max_new_tokens, enable_thinking=None):
    messages = build_messages(context, question)
    prompt = apply_template(tokenizer, messages, enable_thinking)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id)
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return gen_text, len(gen_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--m1-adapter", required=True)
    ap.add_argument("--m2-adapter", required=True)
    ap.add_argument("--eval-file", default="eval_full_v2.jsonl")
    ap.add_argument("--qids-file", default=None,
                    help="JSON 리스트: 평가할 question_id 부분집합 (없으면 Answerable 문항 앞 200건)")
    ap.add_argument("--lambdas", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--tmp-adapter-dir", default="/tmp/interp_adapter")
    args = ap.parse_args()

    cfg = load_config(args.config)
    lambdas = [float(x) for x in args.lambdas.split(",")]

    eval_path = Path(cfg["paths"]["processed_dir"]) / args.eval_file
    rows = {r["question_id"]: r for r in
            (json.loads(l) for l in open(eval_path, encoding="utf-8"))}
    if args.qids_file:
        qids = json.load(open(args.qids_file))
    else:
        qids = [qid for qid, r in rows.items() if r.get("answerable", True)][:200]

    tol = cfg["eval"]["numeric_tolerance"]
    results = {}
    base_model, tokenizer = load_base(cfg)
    enable_thinking = cfg["model"].get("enable_thinking")

    for lam in lambdas:
        adapter_path = make_interpolated_adapter(
            args.m1_adapter, args.m2_adapter, lam, f"{args.tmp_adapter_dir}_lam{lam}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

        n_correct, ntoks = 0, []
        for qid in qids:
            r = rows[qid]
            gen_text, n_gen = predict_one(model, tokenizer, r["context"], r["question"],
                                          cfg["inference"]["max_new_tokens"], enable_thinking)
            parsed = parse_model_output(gen_text)
            correct = (parsed["parse_ok"] and parsed["answerable"] is True and
                      is_correct(parsed["answer"], r["gold_answer"], r["type"], tol))
            n_correct += int(correct)
            ntoks.append(n_gen)

        results[str(lam)] = {
            "em": n_correct / len(qids), "mean_ntok": sum(ntoks) / len(ntoks), "n": len(qids),
        }
        print(f"λ={lam}: EM={results[str(lam)]['em']:.3f}  mean_ntok={results[str(lam)]['mean_ntok']:.1f}")

        # 병합 모델을 base에서 unload 해서 다음 λ 준비 (peft가 base를 공유하도록)
        base_model = model.unload()

    out_path = Path(cfg["paths"]["results_dir"]) / f"diag_interp_{args.out_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
