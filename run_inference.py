# -*- coding: utf-8 -*-
"""
run_inference.py — M0(베이스) / M1(QLoRA) 조건으로 평가셋을 추론하고
답 시퀀스의 신뢰도(평균 로그확률, 최소 토큰 확률)를 추출한다.

사용법:
  python run_inference.py --condition M0
  python run_inference.py --condition M1

출력: results/preds_M0.jsonl, results/preds_M1.jsonl
  {"question_id", "type", "prediction", "answerable_pred", "parse_ok",
   "confidence", "min_token_prob", "conf_scope", "gold_answer"}

신뢰도 산정 원칙:
  - greedy 디코딩 고정 (샘플링 금지: 조건 간 비교가 흔들림)
  - 가능하면 생성 JSON 중 "answer" 값에 해당하는 토큰 구간만으로 계산
    ({"answerable": 같은 고정 토큰은 확률이 높아 전체 평균을 상향 편향시킴)
  - answer 구간 탐지 실패 시 생성 전체 평균으로 폴백하고 conf_scope에 기록
"""

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompts import build_messages, parse_model_output


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_resume_state(out_path):
    """기존 출력 파일에서 완료된 레코드를 복구한다 (spot 중단 대비 resume).

    - 파일이 없으면 빈 상태 반환.
    - 마지막 줄이 중단으로 잘려 있으면(파싱 실패) 그 줄만 버린다.
    - 유효 레코드만으로 파일을 다시 써서 이후 append가 안전하도록 정리.
    반환: (완료 question_id 집합, 기존 폴백 건수, 기존 레코드 수)
    """
    if not out_path.exists():
        return set(), 0, 0
    valid, dropped = [], 0
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                valid.append(json.loads(line))
            except json.JSONDecodeError:
                dropped += 1          # 중단 시점의 불완전한 마지막 줄
    with open(out_path, "w", encoding="utf-8") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    done = {r["question_id"] for r in valid}
    n_fb = sum(1 for r in valid if r.get("conf_scope") != "answer_span")
    if dropped:
        print(f"[resume] 불완전한 레코드 {dropped}건 폐기 (중단 시점 잘린 줄)")
    return done, n_fb, len(valid)


def load_model(cfg, condition, adapter_dir=None):
    model_name = cfg["model"]["base_model"]
    revision = cfg["model"].get("revision")   # 원격 코드·가중치 리비전 고정 (재현성)
    if revision:
        print(f"모델 리비전 고정: {revision}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, revision=revision,
    )
    if condition == "M1":
        path = adapter_dir or cfg["model"]["adapter_dir"]
        model = PeftModel.from_pretrained(model, path)
        print(f"어댑터 로드: {path}")
    model.eval()
    return model, tokenizer


def token_char_offsets(tokenizer, gen_ids):
    """생성 토큰 i가 생성 텍스트의 어느 문자 구간 [start, end)에 해당하는지 계산.

    누적 디코딩 방식(O(n^2)이지만 n<=128이라 무시 가능). BPE 병합으로 인한
    미세한 불일치는 폴백 로직이 흡수한다.
    """
    offsets, prev_len = [], 0
    for i in range(1, len(gen_ids) + 1):
        text = tokenizer.decode(gen_ids[:i], skip_special_tokens=True)
        offsets.append((prev_len, len(text)))
        prev_len = len(text)
    return offsets


def step_dist_stats(scores):
    """각 생성 스텝의 출력 분포 엔트로피(nat)와 top1−top2 확률 마진을 계산.

    [탐색적 지표 — 기제 분석용] 파인튜닝이 confidence를 올리는 이유가
    '분포 첨예화(sharpening)'인지 보기 위한 로그. 사전등록 판정에는 쓰지 않는다.
    """
    ents, margins = [], []
    for s in scores:
        logp = torch.log_softmax(s[0].float(), dim=-1)
        p = logp.exp()
        ents.append(float(-(p * logp).sum()))
        top2 = torch.topk(p, 2).values
        margins.append(float(top2[0] - top2[1]))
    return ents, margins


def answer_token_range(tokenizer, gen_ids, gen_text, answer_value):
    """생성 텍스트에서 answer 값의 문자 구간을 찾아 토큰 인덱스 범위로 변환."""
    if answer_value is None:
        return None
    answer_str = str(answer_value)
    char_start = gen_text.find(answer_str)
    if char_start == -1:
        return None
    char_end = char_start + len(answer_str)
    offsets = token_char_offsets(tokenizer, gen_ids)
    idxs = [i for i, (s, e) in enumerate(offsets) if e > char_start and s < char_end]
    return (idxs[0], idxs[-1] + 1) if idxs else None


@torch.no_grad()
def predict_one(model, tokenizer, row, max_new_tokens):
    messages = build_messages(row["context"], row["question"])
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,                 # greedy 고정
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    # 생성 토큰별 log P (프롬프트 제외)
    trans = model.compute_transition_scores(
        out.sequences, out.scores, normalize_logits=True
    )[0]  # shape: (gen_len,)
    gen_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
    step_ents, step_margins = step_dist_stats(out.scores)  # 탐색적: 분포 통계

    # eos/pad 이후 잘라내기
    keep = []
    for tid, lp, en, mg in zip(gen_ids.tolist(), trans.tolist(),
                               step_ents, step_margins):
        if tid in (tokenizer.eos_token_id, tokenizer.pad_token_id):
            break
        keep.append((tid, lp, en, mg))
    if not keep:
        return {"prediction": "", "answerable_pred": None, "parse_ok": False,
                "confidence": 0.0, "min_token_prob": 0.0, "conf_scope": "empty",
                "n_gen_tokens": 0, "n_answer_tokens": 0,
                "entropy_mean": None, "margin_mean": None,
                "entropy_full": None, "margin_full": None}
    gen_ids = [t for t, _, _, _ in keep]
    logps = [lp for _, lp, _, _ in keep]
    ents = [en for _, _, en, _ in keep]
    margins = [mg for _, _, _, mg in keep]

    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    parsed = parse_model_output(gen_text)

    # 신뢰도: answer 값 토큰 구간 우선, 실패 시 전체 폴백
    rng = answer_token_range(tokenizer, gen_ids, gen_text, parsed["answer"]) \
        if parsed["parse_ok"] else None
    if rng:
        span_logps, scope = logps[rng[0]:rng[1]], "answer_span"
        span_ents, span_margins = ents[rng[0]:rng[1]], margins[rng[0]:rng[1]]
    else:
        span_logps, scope = logps, "full_sequence"
        span_ents, span_margins = ents, margins

    confidence = math.exp(sum(span_logps) / len(span_logps))
    min_token_prob = math.exp(min(span_logps))

    return {
        "prediction": parsed["answer"],
        "answerable_pred": parsed["answerable"],
        "parse_ok": parsed["parse_ok"],
        "confidence": confidence,
        "min_token_prob": min_token_prob,
        "conf_scope": scope,
        # ---- 이하 탐색적 로그 (기제·측정 인공물 분석용, 판정에 미사용) ----
        "n_gen_tokens": len(logps),
        "n_answer_tokens": len(span_logps),          # 유형별 답 토큰 수 → 측정 인공물 점검
        "entropy_mean": sum(span_ents) / len(span_ents),      # answer 구간 평균 엔트로피
        "margin_mean": sum(span_margins) / len(span_margins),  # answer 구간 평균 top1−top2
        "entropy_full": sum(ents) / len(ents),
        "margin_full": sum(margins) / len(margins),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["M0", "M1"], required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=None,
                    help="M1 전용: 학습 시드. {adapter_dir}_s{seed} 어댑터를 로드하고 "
                         "preds_M1_s{seed}.jsonl 로 저장 (본 실험 3시드 프로토콜)")
    ap.add_argument("--limit", type=int, default=None, help="디버깅용: 앞 N건만 추론")
    ap.add_argument("--fresh", action="store_true",
                    help="기존 출력을 무시하고 처음부터 다시 추론 (기본은 이어서 실행)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])

    adapter_dir, tag = None, args.condition
    if args.condition == "M1" and args.seed is not None:
        adapter_dir = f'{cfg["model"]["adapter_dir"]}_s{args.seed}'
        tag = f"M1_s{args.seed}"
    model, tokenizer = load_model(cfg, args.condition, adapter_dir)

    eval_path = Path(cfg["paths"]["processed_dir"]) / "eval.jsonl"
    rows = [json.loads(l) for l in open(eval_path, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(cfg["paths"]["results_dir"]) / f"preds_{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- resume: 기존 출력이 있으면 완료분을 건너뛰고 이어서 실행 (spot 중단 대비)
    if args.fresh:
        done, n_fallback, n_done = set(), 0, 0
        out_path.unlink(missing_ok=True)
    else:
        done, n_fallback, n_done = load_resume_state(out_path)
        if n_done:
            print(f"[resume] 기존 완료 {n_done}건 발견 → 건너뛰고 이어서 추론 "
                  f"(처음부터 다시 하려면 --fresh)")
    todo = [r for r in rows if r["question_id"] not in done]
    if not todo:
        print(f"이미 전건 완료: {out_path} ({n_done}건). 재실행하려면 --fresh 사용.")
        return

    with open(out_path, "a", encoding="utf-8") as f:
        for row in tqdm(todo, desc=f"추론 {args.condition}",
                        initial=n_done, total=n_done + len(todo)):
            pred = predict_one(model, tokenizer, row, cfg["inference"]["max_new_tokens"])
            if pred["conf_scope"] != "answer_span":
                n_fallback += 1
            record = {
                "question_id": row["question_id"],
                "type": row["type"],
                "orig_type": row.get("orig_type"),          # unanswerable의 원유형
                "gold_answerable": row.get("answerable", True),  # 정답측 응답가능 여부
                "gold_answer": row["gold_answer"],
                **pred,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()                    # 중단 시 손실을 최대 1건으로 제한

    n_total = n_done + len(todo)
    print(f"저장: {out_path} ({n_total}건, 이번 실행 {len(todo)}건)")
    print(f"answer_span 탐지 실패(전체 평균 폴백): {n_fallback}건 "
          f"({100 * n_fallback / n_total:.1f}%) — 10%를 넘으면 파싱 로직 점검 필요")


if __name__ == "__main__":
    main()