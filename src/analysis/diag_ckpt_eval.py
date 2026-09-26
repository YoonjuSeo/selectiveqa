# -*- coding: utf-8 -*-
"""
diag_ckpt_eval.py — Tier 1 Q5(선택, 사전등록 5절-5): M2-r05 진단 재학습의 스텝별
어댑터 체크포인트(step20, step40, ... train_qlora_diag.py의 StepAdapterSaver 산출물)에서
"형식 회귀(응답 길이 폭증)가 학습 중 언제 나타나는가"를 측정한다.

각 체크포인트에서:
  ① Answerable 200문항(diag_interp.py의 λ 스윕과 동일 부분집합 — eval_full_v2 앞 200건)
     추론 → EM, mean_ntok
  ② (옵션) 기존 M1 어댑터와의 LoRA delta-W(B@A) 코사인 유사도 — Tier0(b)와 동일 방법
     (safetensors를 key 단위로 스트리밍해 OOM 회피, 전체 키를 이어붙인 하나의 벡터로 집계)

t_fmt = mean_ntok가 M1 대비 ×1.5를 처음 넘는 step (사전등록 4.3절)

사용법 (Modal):
  modal run --detach modal_ckpt_eval.py --model exaone \
      --ckpt-glob "qlora_adapter_r05_diag_s42_step*" \
      --m1-adapter results/qlora_adapter_s42 --out-tag r05_diag_s42

출력: results/diag_ckpt_eval_{tag}.json — {"20": {"em":..,"mean_ntok":..,"n":..,"cosine_vs_m1":..}, ...}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import re

import torch
import yaml
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from inference.prompts import build_messages, parse_model_output, apply_template
from evaluation.evaluate import is_correct


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_cosine_vs_m1(m1_dir, ckpt_dir):
    """M1과 체크포인트 어댑터의 LoRA delta-W(B@A)를 key 단위로 스트리밍하며
    전체를 이어붙인 하나의 벡터로 취급한 코사인 유사도를 계산한다(Tier0-b와 동일 방법,
    전체 델타 행렬을 한 번에 메모리에 올리지 않음)."""
    f1 = safe_open(Path(m1_dir) / "adapter_model.safetensors", framework="pt")
    f2 = safe_open(Path(ckpt_dir) / "adapter_model.safetensors", framework="pt")
    keys1 = set(f1.keys())
    keys2 = set(f2.keys())

    # lora_A/lora_B 쌍을 (레이어, 프로젝션) 단위로 묶는다
    def group_pairs(keys):
        pairs = {}
        for k in keys:
            if ".lora_A." in k:
                base = k.replace(".lora_A.", ".lora_?.")
                pairs.setdefault(base, {})["A"] = k
            elif ".lora_B." in k:
                base = k.replace(".lora_B.", ".lora_?.")
                pairs.setdefault(base, {})["B"] = k
        return {k: v for k, v in pairs.items() if "A" in v and "B" in v}

    p1 = group_pairs(keys1)
    p2 = group_pairs(keys2)
    common = sorted(set(p1) & set(p2))
    if not common:
        return None

    dot, norm1_sq, norm2_sq = 0.0, 0.0, 0.0
    for base in common:
        a1 = f1.get_tensor(p1[base]["A"]).float()
        b1 = f1.get_tensor(p1[base]["B"]).float()
        a2 = f2.get_tensor(p2[base]["A"]).float()
        b2 = f2.get_tensor(p2[base]["B"]).float()
        d1 = (b1 @ a1).flatten()
        d2 = (b2 @ a2).flatten()
        dot += torch.dot(d1, d2).item()
        norm1_sq += torch.dot(d1, d1).item()
        norm2_sq += torch.dot(d2, d2).item()
        del a1, b1, a2, b2, d1, d2

    if norm1_sq <= 0 or norm2_sq <= 0:
        return None
    return dot / ((norm1_sq ** 0.5) * (norm2_sq ** 0.5))


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


def step_num(ckpt_dir_name):
    m = re.search(r"_step(\d+)$", ckpt_dir_name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt-glob", required=True,
                    help="results_dir 기준 체크포인트 디렉토리 glob, 예: qlora_adapter_r05_diag_s42_step*")
    ap.add_argument("--m1-adapter", required=True, help="비교 기준 M1 어댑터 경로 (코사인·mean_ntok 기준선)")
    ap.add_argument("--eval-file", default="eval_full_v2.jsonl")
    ap.add_argument("--qids-file", default=None,
                    help="JSON 리스트: 평가할 question_id 부분집합 (없으면 Answerable 문항 앞 200건, λ 스윕과 동일)")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--skip-cosine", action="store_true", help="코사인 계산 생략(속도 우선)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    res_dir = Path(cfg["paths"]["results_dir"])

    eval_path = Path(cfg["paths"]["processed_dir"]) / args.eval_file
    rows = {r["question_id"]: r for r in
            (json.loads(l) for l in open(eval_path, encoding="utf-8"))}
    if args.qids_file:
        qids = json.load(open(args.qids_file))
    else:
        qids = [qid for qid, r in rows.items() if r.get("answerable", True)][:200]
    tol = cfg["eval"]["numeric_tolerance"]

    ckpt_dirs = sorted(res_dir.glob(args.ckpt_glob), key=lambda p: step_num(p.name) or 0)
    if not ckpt_dirs:
        raise SystemExit(f"체크포인트를 찾지 못함: {res_dir / args.ckpt_glob}")
    print(f"체크포인트 {len(ckpt_dirs)}개 발견: {[p.name for p in ckpt_dirs]}")

    out_path = res_dir / f"diag_ckpt_eval_{args.out_tag}.json"
    results = {}
    if out_path.exists():
        try:
            results = json.load(open(out_path, encoding="utf-8"))
            print(f"기존 결과 로드: {out_path} (완료된 step: {list(results.keys())})")
        except Exception as e:
            print(f"기존 결과 파일 로드 실패({e}), 처음부터 시작")

    base_model, tokenizer = load_base(cfg)
    enable_thinking = cfg["model"].get("enable_thinking")

    for ckpt_dir in ckpt_dirs:
        step = step_num(ckpt_dir.name)
        key = str(step)
        if key in results:
            print(f"step {step}: 이미 완료, 스킵")
            continue

        model = PeftModel.from_pretrained(base_model, ckpt_dir)
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

        cosine = None
        if not args.skip_cosine:
            cosine = compute_cosine_vs_m1(args.m1_adapter, ckpt_dir)

        results[key] = {
            "em": n_correct / len(qids), "mean_ntok": sum(ntoks) / len(ntoks),
            "n": len(qids), "cosine_vs_m1": cosine,
        }
        print(f"step {step}: EM={results[key]['em']:.3f} mean_ntok={results[key]['mean_ntok']:.1f} "
              f"cosine_vs_m1={cosine}")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"저장(누적): {out_path}")

        base_model = model.unload()

    print(f"전체 완료: {out_path}")


if __name__ == "__main__":
    main()
