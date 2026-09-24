# -*- coding: utf-8 -*-
"""
diag_prompt_ablation.py — Tier 0(e): M2(붕괴형) EXAONE에 시스템 프롬프트를
변형해 추론, 붕괴(EM 손실·답변 길이 팽창)가 프롬프트 개입만으로 완화되는지
확인한다. 4.7절 (A) ablation의 발견("무응답의 표면 형식은 SFT 타깃보다
프롬프트 지시에 지배된다")을 뒤집어, "그렇다면 프롬프트를 바꾸면 붕괴도
바뀌는가"를 직접 검정한다.

변형 프롬프트 3종 (SYSTEM_PROMPT_VARIANTS에서 수정):
  base       : 원본 시스템 프롬프트 (대조군)
  no_ua_ex   : 무응답 예시({"answerable": false, ...}) 문장을 시스템 프롬프트에서 제거
               → M2가 학습한 무응답 형식 힌트를 추론 시점에서 지웠을 때 EM이 회복되는지
  terse      : "다른 설명 없이 간결하게 답하라"를 추가
               → M2의 장황한 서술형 회귀(4.7절 "M0 스타일로 회귀")를 프롬프트로 강제 억제할 수 있는지
  strict_json: JSON 스키마를 더 엄격하게 반복 강조
               → 형식 붕괴(9~12%p)가 프롬프트 강조만으로 줄어드는지

주의: 학습(M2)과 추론의 프롬프트가 달라지므로 이 결과는 "붕괴의 원인 규명"이
아니라 "붕괴의 완화 가능성(실무적 우회로)" 탐색이다. 원본 M1(붕괴 없음)에도
동일 변형을 돌려 대조군으로 삼는다.

사용법 (Modal):
  modal run modal_diag_tier0.py --task prompt_ablation --model exaone \
      --adapter results/qlora_adapter_r05_s42 --condition M2
  modal run modal_diag_tier0.py --task prompt_ablation --model exaone \
      --adapter results/qlora_adapter_s42 --condition M1   # 대조군

출력: results/diag_promptablation_{tag}.json — variant별 {em, mean_ntok, n}
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

from evaluation.evaluate import is_correct
from inference import prompts as prompts_mod
from inference.prompts import parse_model_output

ORIGINAL_SYSTEM_PROMPT = prompts_mod.SYSTEM_PROMPT

SYSTEM_PROMPT_VARIANTS = {
    "base": ORIGINAL_SYSTEM_PROMPT,
    "no_ua_ex": (
        "당신은 금융 문서 질의응답 시스템입니다. "
        "주어진 지문에 근거해서만 답하세요. "
        "반드시 아래 JSON 형식 한 줄로만 출력하세요.\n"
        '{"answerable": true, "answer": "<답>", "evidence_span": "<지문 내 근거 문장>"}'
        # 무응답 예시 문장 두 줄을 의도적으로 제거
    ),
    "terse": ORIGINAL_SYSTEM_PROMPT + "\n다른 설명 없이 위 JSON 형식만 출력하세요. 서술형 문장을 덧붙이지 마세요.",
    "strict_json": (
        "당신은 금융 문서 질의응답 시스템입니다. "
        "주어진 지문에 근거해서만 답하세요. "
        "출력은 반드시 아래 두 형식 중 하나와 정확히 일치하는 JSON 한 줄이어야 하며, "
        "그 외의 어떤 텍스트도 포함해서는 안 됩니다.\n"
        '형식 1(답할 수 있음): {"answerable": true, "answer": "<답>", "evidence_span": "<지문 내 근거 문장>"}\n'
        '형식 2(답할 수 없음): {"answerable": false, "answer": null, "evidence_span": null}'
    ),
}


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
    if condition in ("M1", "M2"):
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def predict_with_system_prompt(model, tokenizer, system_prompt, context, question,
                                max_new_tokens, enable_thinking=None):
    user = f"[지문]\n{context}\n\n[질문]\n{question}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]
    prompt = prompts_mod.apply_template(tokenizer, messages, enable_thinking)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id)
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return gen_text, len(gen_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--condition", choices=["M1", "M2"], required=True)
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--eval-file", default="eval_full_v2.jsonl")
    ap.add_argument("--n-samples", type=int, default=200, help="Answerable 문항 샘플 수")
    ap.add_argument("--out-tag", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = load_model(cfg, args.condition, args.adapter_dir)
    enable_thinking = cfg["model"].get("enable_thinking")
    tol = cfg["eval"]["numeric_tolerance"]

    eval_path = Path(cfg["paths"]["processed_dir"]) / args.eval_file
    rows = [json.loads(l) for l in open(eval_path, encoding="utf-8")]
    rows = [r for r in rows if r.get("answerable", True)][: args.n_samples]

    results = {}
    for variant_name, sys_prompt in SYSTEM_PROMPT_VARIANTS.items():
        n_correct, ntoks = 0, []
        for r in rows:
            gen_text, n_gen = predict_with_system_prompt(
                model, tokenizer, sys_prompt, r["context"], r["question"],
                cfg["inference"]["max_new_tokens"], enable_thinking)
            parsed = parse_model_output(gen_text)
            correct = (parsed["parse_ok"] and parsed["answerable"] is True and
                      is_correct(parsed["answer"], r["gold_answer"], r["type"], tol))
            n_correct += int(correct)
            ntoks.append(n_gen)
        results[variant_name] = {
            "em": n_correct / len(rows), "mean_ntok": sum(ntoks) / len(ntoks), "n": len(rows),
        }
        print(f"[{args.condition}] variant={variant_name:12s} EM={results[variant_name]['em']:.3f} "
              f"mean_ntok={results[variant_name]['mean_ntok']:.1f}")

    out_path = Path(cfg["paths"]["results_dir"]) / f"diag_promptablation_{args.out_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
