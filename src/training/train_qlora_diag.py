# -*- coding: utf-8 -*-
"""
train_qlora_diag.py — Tier 1(1~2단계): EXAONE M2 재학습 시 UA/Answerable loss를
"사전등록: Tier 1 학습 중 UA/Answerable loss 분리(진단 f)" 3절 스펙에 맞춰
optimizer step 단위로 분리 기록하고, 스텝 단위로 어댑터 체크포인트를 저장한다.

기존 train_qlora.py와 동일한 학습 파이프라인(모델/LoRA/데이터 처리, gradient 경로)을
그대로 쓴다. 학습에 쓰는 loss(파라미터 업데이트)는 기존과 동일하게 배치 평균으로
계산하며, 아래 추가 사항은 모두 "읽기 전용" 로깅이다:

  1) UASplitTrainer: compute_loss에서 배치 내 각 예제의 per-example loss를 계산하는
     한편, reduction="none" 토큰별 loss를 별도로 UA/Answerable/필드(field)/내용(content)
     구간으로 나눠 누적한다(사전등록 3절 로그 키 전량). optimizer step마다(=1개
     logging 이벤트마다) 누적값을 JSONL 한 줄로 flush한다.
       - field 구간: `{"answerable": true, ...}` 의 JSON 구조·키·`answerable` 값 토큰
       - content 구간: `answer`/`evidence_span` 값(문자열) 토큰
     구간 경계는 build_target()과 동일한 직렬화 규칙으로 문자 오프셋을 먼저 계산한 뒤,
     fast tokenizer의 offset_mapping으로 토큰 인덱스에 매핑한다(3절: "토큰 구간 정의는
     학습 시작 전에 코드로 고정"). offset_mapping을 지원하지 않는 토크나이저이거나
     build_target()의 직렬화와 재구성 문자열이 어긋나는 예제는 안전하게 전부 field로
     처리한다(경고 출력, 학습은 중단하지 않음).

  2) StepAdapterSaver: N스텝마다 LoRA 어댑터만 가볍게 저장하고, 저장 시점마다
     NaN/Inf 파라미터 점검을 수행한다(사전등록 2.2절: "Qwen3 s42 발산 재발 감시용").

  3) --check-spans: 학습 없이 데이터셋 앞 N건에 대해 field/content 문자 구간과
     디코딩된 토큰을 출력만 하고 종료한다. 사전등록 3절의 "예제 5건을 손으로 확인"
     절차용 — 여기 출력을 그대로 문서에 첨부하면 된다.

원본 train_qlora.py는 건드리지 않음(검증된 M1/M2 판정용 파이프라인 보존).

사용법 (Modal, 1~2단계 / M2-r05):
  modal run --detach modal_train_diag.py --model exaone --train-file train_mix_r05.jsonl \
      --tag _r05_diag --seed 42 --ckpt-interval 20

M1 진단 재학습(ρ_content, G 기준선용 — 사전등록 4.3절 "M1 재학습" 결정 반영):
  modal run --detach modal_train_diag.py --model exaone --train-file train.jsonl \
      --tag _m1_diag --seed 42 --ckpt-interval 20

구간 확인(문서 첨부용, 로컬 CPU에서도 가능):
  python src/training/train_qlora_diag.py --config config.yaml --check-spans 5

출력:
  results/tier1_loss_log{tag}_s{seed}.jsonl
      — optimizer step마다 1줄: {step, epoch, n_ua, n_ans, tok_ua, tok_ans,
        sum_loss_ua, sum_loss_ans, tok_ans_field, sum_loss_ans_field,
        tok_ans_content, sum_loss_ans_content, loss_total, grad_norm, example_ids}
  results/qlora_adapter{tag}_s{seed}_step{N}/  — 스텝별 어댑터 스냅샷(LoRA만, 가벼움)
  results/qlora_adapter{tag}_s{seed}/          — 최종 어댑터(진단용, 기존 판정용과 별개)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import random

import torch
import torch.nn as nn
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from inference.prompts import build_messages, build_target, apply_template


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_target_with_spans(gold_answer, evidence_span=None, answerable=True, ua_style="null"):
    """build_target()과 바이트 단위로 동일한 문자열을 재구성하면서, 그 과정에서
    'answer'/'evidence_span' 값(content) 구간의 (start, end) 문자 오프셋을 함께 반환한다.

    UA 예제(answerable=False)는 field/content 구분이 필요 없으므로 spans=[]를 반환한다.
    재구성 문자열이 build_target()의 실제 출력과 한 글자라도 다르면(직렬화 방식이
    예상과 어긋나는 경우에 대한 안전장치) spans=[]로 되돌리고 전부 field로 취급한다.
    """
    canonical = build_target(gold_answer, evidence_span=evidence_span,
                             answerable=answerable, ua_style=ua_style)
    if not answerable:
        return canonical, []

    ev = evidence_span if evidence_span else gold_answer
    answer_json = json.dumps(gold_answer, ensure_ascii=False)
    ev_json = json.dumps(ev, ensure_ascii=False)
    part1 = '{"answerable": true, "answer": '
    part2 = answer_json
    part3 = ', "evidence_span": '
    part4 = ev_json
    part5 = '}'
    reconstructed = part1 + part2 + part3 + part4 + part5

    if reconstructed != canonical:
        return canonical, []

    start_answer = len(part1)
    end_answer = start_answer + len(part2)
    start_ev = end_answer + len(part3)
    end_ev = start_ev + len(part4)
    return canonical, [(start_answer, end_answer), (start_ev, end_ev)]


class SFTDatasetLabeled(Dataset):
    """SFTDataset과 동일하지만 각 예제의 answerable 여부·예제 ID·field/content
    토큰 마스크를 함께 반환한다."""

    def __init__(self, jsonl_path, tokenizer, max_len, enable_thinking=None, ua_style="null"):
        self.tokenizer = tokenizer
        self.ua_style = ua_style
        self.max_len = max_len
        self.enable_thinking = enable_thinking
        self.rows = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))

        # offset_mapping 지원 여부를 한 번만 확인 (slow tokenizer 대비 안전장치)
        try:
            probe = self.tokenizer("테스트", add_special_tokens=False, return_offsets_mapping=True)
            self.has_offsets = "offset_mapping" in probe
        except Exception:
            self.has_offsets = False
        if not self.has_offsets:
            print("[경고] 이 토크나이저는 offset_mapping을 지원하지 않습니다. "
                  "field/content 구간 분리를 생략하고 전부 field로 기록합니다.")

    def __len__(self):
        return len(self.rows)

    def _build_target_and_content_flags(self, row, answerable):
        target_json, content_spans = build_target_with_spans(
            row["gold_answer"], evidence_span=row.get("evidence_span"),
            answerable=answerable, ua_style=self.ua_style,
        )
        target_text = target_json + self.tokenizer.eos_token

        if self.has_offsets:
            enc = self.tokenizer(target_text, add_special_tokens=False, return_offsets_mapping=True)
            target_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            if content_spans:
                content_flags = [
                    1 if any(o[1] > s and o[0] < e for (s, e) in content_spans) else 0
                    for o in offsets
                ]
            else:
                content_flags = [0] * len(target_ids)
        else:
            target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]
            content_flags = [0] * len(target_ids)

        return target_ids, content_flags

    def __getitem__(self, idx):
        row = self.rows[idx]
        answerable = bool(row.get("answerable", True))
        example_id = row.get("id", row.get("qid", idx))
        messages = build_messages(row["context"], row["question"])
        prompt_text = apply_template(self.tokenizer, messages, self.enable_thinking)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        target_ids, content_flags = self._build_target_and_content_flags(row, answerable)

        input_ids = (prompt_ids + target_ids)[: self.max_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_len]
        content_mask = ([0] * len(prompt_ids) + content_flags)[: self.max_len]
        if all(l == -100 for l in labels):
            n = len(labels)
            labels[-len(target_ids):] = target_ids[: n]
            content_mask[-len(content_flags):] = content_flags[: n]

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "content_mask": torch.tensor(content_mask, dtype=torch.float),
            "answerable": answerable,
            "example_id": example_id,
        }


def collate_labeled(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)

    def pad(seq, value):
        return torch.cat([seq, torch.full((max_len - len(seq),), value, dtype=seq.dtype)])

    return {
        "input_ids": torch.stack([pad(b["input_ids"], pad_id) for b in batch]),
        "labels": torch.stack([pad(b["labels"], -100) for b in batch]),
        "attention_mask": torch.stack([pad(b["attention_mask"], 0) for b in batch]),
        "content_mask": torch.stack([pad(b["content_mask"], 0.0) for b in batch]),
        "answerable": [b["answerable"] for b in batch],     # 텐서화하지 않고 그대로 전달
        "example_ids": [b["example_id"] for b in batch],
    }


class UASplitTrainer(Trainer):
    """배치별 per-example/per-token loss를 UA/Answerable(field/content)로 나눠
    누적하고, optimizer step마다 콜백(LossFlushCallback)이 읽어갈 수 있도록
    self.loss_tracker에 쌓아둔다. 학습에 쓰는 loss(반환값)는 기존과 동일한
    배치-평균(per-example-mean) 방식이며, 아래 누적 집계는 로깅 전용이다."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_tracker()

    def _reset_tracker(self):
        self.loss_tracker = {
            "n_ua": 0, "n_ans": 0,
            "tok_ua": 0.0, "tok_ans": 0.0,
            "sum_loss_ua": 0.0, "sum_loss_ans": 0.0,
            "tok_ans_field": 0.0, "sum_loss_ans_field": 0.0,
            "tok_ans_content": 0.0, "sum_loss_ans_content": 0.0,
            "example_ids": [],
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        answerable = inputs.pop("answerable")
        example_ids = inputs.pop("example_ids")
        content_mask = inputs.pop("content_mask")

        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.logits
        labels = inputs["labels"]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_content = content_mask[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        tok_mask = (shift_labels != -100).float()
        denom = tok_mask.sum(dim=1).clamp(min=1)
        per_example_loss = (per_token_loss * tok_mask).sum(dim=1) / denom
        loss = per_example_loss.mean()   # ★ 학습에 실제로 쓰이는 loss — 기존과 동일한 정의

        field_mask = tok_mask * (1.0 - shift_content)
        content_mask_eff = tok_mask * shift_content

        t = self.loss_tracker
        for i, is_ans in enumerate(answerable):
            n_tok = tok_mask[i].sum().item()
            s_loss = (per_token_loss[i] * tok_mask[i]).sum().item()
            t["example_ids"].append(example_ids[i])
            if is_ans:
                t["n_ans"] += 1
                t["tok_ans"] += n_tok
                t["sum_loss_ans"] += s_loss
                f_tok = field_mask[i].sum().item()
                c_tok = content_mask_eff[i].sum().item()
                t["tok_ans_field"] += f_tok
                t["sum_loss_ans_field"] += (per_token_loss[i] * field_mask[i]).sum().item()
                t["tok_ans_content"] += c_tok
                t["sum_loss_ans_content"] += (per_token_loss[i] * content_mask_eff[i]).sum().item()
            else:
                t["n_ua"] += 1
                t["tok_ua"] += n_tok
                t["sum_loss_ua"] += s_loss

        return (loss, outputs) if return_outputs else loss


class LossFlushCallback(TrainerCallback):
    """logging_steps(=1로 설정, optimizer step마다) 주기로 UASplitTrainer.loss_tracker를
    읽어 사전등록 3절 스키마 그대로 JSONL에 기록하고 비운다. loss_total/grad_norm은
    Trainer가 on_log에 넘겨주는 logs dict에서 그대로 가져온다(같은 step 값)."""

    def __init__(self, trainer_ref_holder, out_path):
        self.trainer_ref_holder = trainer_ref_holder  # dict; ["trainer"]에 나중에 채워짐
        self.out_path = out_path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        open(out_path, "w", encoding="utf-8").close()  # 새로 시작

    def on_log(self, args, state, control, logs=None, **kwargs):
        trainer = self.trainer_ref_holder.get("trainer")
        if trainer is None:
            return
        logs = logs or {}
        t = trainer.loss_tracker

        ua_mean = (t["sum_loss_ua"] / t["tok_ua"]) if t["tok_ua"] > 0 else None
        ans_mean = (t["sum_loss_ans"] / t["tok_ans"]) if t["tok_ans"] > 0 else None

        record = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch is not None else None,
            "n_ua": t["n_ua"], "n_ans": t["n_ans"],
            "tok_ua": t["tok_ua"], "tok_ans": t["tok_ans"],
            "sum_loss_ua": t["sum_loss_ua"], "sum_loss_ans": t["sum_loss_ans"],
            "tok_ans_field": t["tok_ans_field"], "sum_loss_ans_field": t["sum_loss_ans_field"],
            "tok_ans_content": t["tok_ans_content"], "sum_loss_ans_content": t["sum_loss_ans_content"],
            "loss_total": logs.get("loss"),
            "grad_norm": logs.get("grad_norm"),
            "example_ids": t["example_ids"],
        }
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        gap = (ans_mean - ua_mean) if (ans_mean is not None and ua_mean is not None) else None
        print(f"[loss-split] step={state.global_step} ua_mean={ua_mean} ans_mean={ans_mean} "
              f"gap={gap} loss_total={record['loss_total']} grad_norm={record['grad_norm']} "
              f"(n_ua={t['n_ua']}, n_ans={t['n_ans']})")

        trainer._reset_tracker()


def check_adapter_nan_inf(model, tag=""):
    """사전등록 2.2절: 저장 시점마다 NaN/Inf 점검(읽기 전용, Qwen3 s42류 발산 재발 감시)."""
    bad = []
    for name, p in model.named_parameters():
        if p.requires_grad and (torch.isnan(p).any() or torch.isinf(p).any()):
            bad.append(name)
    if bad:
        shown = bad[:5]
        more = "..." if len(bad) > 5 else ""
        print(f"[경고{tag}] NaN/Inf 파라미터 감지 ({len(bad)}개): {shown}{more}")
    return bad


class StepAdapterSaver(TrainerCallback):
    """N스텝마다 LoRA 어댑터만 저장(가중치 궤적 사후분석용) + NaN/Inf 점검."""

    def __init__(self, model, base_dir, interval):
        self.model = model
        self.base_dir = base_dir
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        if self.interval <= 0:
            return
        if state.global_step > 0 and state.global_step % self.interval == 0:
            path = f"{self.base_dir}_step{state.global_step}"
            self.model.save_pretrained(path)
            check_adapter_nan_inf(self.model, tag=f"(step {state.global_step})")
            print(f"[체크포인트] 스텝 {state.global_step} 어댑터 저장: {path}")


def dump_span_check(dataset, n=5):
    """사전등록 3절 '예제 5건을 손으로 확인' 절차용: field/content 구간을 사람이
    읽을 수 있는 형태로 출력하고 종료한다(학습 없음)."""
    tok = dataset.tokenizer
    print(f"offset_mapping 지원 여부: {dataset.has_offsets}")
    for i in range(min(n, len(dataset))):
        row = dataset.rows[i]
        answerable = bool(row.get("answerable", True))
        target_ids, content_flags = dataset._build_target_and_content_flags(row, answerable)
        pieces = []
        for tid, flag in zip(target_ids, content_flags):
            piece = tok.decode([tid])
            pieces.append(f"[{piece}]{'*' if flag else ''}")
        print(f"\n=== 예제 {i} (id={row.get('id', row.get('qid', i))}, answerable={answerable}) ===")
        print("토큰 나열 (*=content, 무표시=field):")
        print(" ".join(pieces))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--train-file", default="train_mix_r05.jsonl")
    ap.add_argument("--tag", default="_r05_diag")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--ckpt-interval", type=int, default=20,
                    help="N스텝마다 어댑터 스냅샷 저장 (0이면 비활성화)")
    ap.add_argument("--max-grad-norm", type=float, default=None,
                    help="명시적 그래디언트 클리핑(Tier1-3단계용). 미지정 시 HF 기본값(1.0)")
    ap.add_argument("--check-spans", type=int, default=0,
                    help="N>0이면 학습 없이 앞 N건의 field/content 토큰 구간만 출력하고 종료")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    print(f"학습 시드: {seed}")

    model_name = cfg["model"]["base_model"]
    revision = cfg["model"].get("revision")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.check_spans > 0:
        train_path = Path(cfg["paths"]["processed_dir"]) / args.train_file
        ua_style = cfg["train"].get("ua_target", "null")
        dataset = SFTDatasetLabeled(train_path, tokenizer, cfg["train"]["max_seq_len"],
                                    enable_thinking=cfg["model"].get("enable_thinking"),
                                    ua_style=ua_style)
        dump_span_check(dataset, n=args.check_spans)
        return

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, revision=revision,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora = LoraConfig(
        r=cfg["train"]["lora_r"], lora_alpha=cfg["train"]["lora_alpha"],
        lora_dropout=cfg["train"]["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_path = Path(cfg["paths"]["processed_dir"]) / args.train_file
    ua_style = cfg["train"].get("ua_target", "null")
    print(f"무응답 타깃 형식: {ua_style}")
    dataset = SFTDatasetLabeled(train_path, tokenizer, cfg["train"]["max_seq_len"],
                                enable_thinking=cfg["model"].get("enable_thinking"),
                                ua_style=ua_style)
    print(f"학습 데이터: {len(dataset)}건 "
          f"(UA={sum(1 for r in dataset.rows if not r.get('answerable', True))}건)")

    targs_kwargs = dict(
        output_dir=str(Path(cfg["paths"]["results_dir"]) / f"train_ckpt_diag_s{seed}"),
        num_train_epochs=args.epochs if args.epochs is not None else cfg["train"]["epochs"],
        learning_rate=float(cfg["train"]["lr"]),
        per_device_train_batch_size=cfg["train"]["batch_size"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        bf16=True,
        logging_steps=1,          # ★ 사전등록 3절: "매 optimizer step마다" 로깅
        save_strategy="no",
        report_to="none",
        seed=seed,
        remove_unused_columns=False,  # "answerable"/"content_mask"/"example_ids" 필드가
                                       # model.forward 시그니처에 없어 Trainer가 자동 제거하는
                                       # 것을 방지 (compute_loss에서 직접 씀)
    )
    if args.max_grad_norm is not None:
        targs_kwargs["max_grad_norm"] = args.max_grad_norm
        print(f"그래디언트 클리핑 명시: max_grad_norm={args.max_grad_norm}")
    else:
        print("그래디언트 클리핑: HF 기본값(max_grad_norm=1.0) 유지 (사전등록 2.3절: 원 실험 값 그대로)")
    targs = TrainingArguments(**targs_kwargs)

    adapter_dir = f'{cfg["model"]["adapter_dir"]}{args.tag}_s{seed}'
    loss_log_path = str(Path(cfg["paths"]["results_dir"]) / f"tier1_loss_log{args.tag}_s{seed}.jsonl")

    trainer_holder = {}
    callbacks = [
        LossFlushCallback(trainer_holder, loss_log_path),
        StepAdapterSaver(model, adapter_dir, args.ckpt_interval),
    ]

    trainer = UASplitTrainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=lambda b: collate_labeled(b, tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    trainer_holder["trainer"] = trainer

    trainer.train()

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    check_adapter_nan_inf(model, tag="(최종)")
    print(f"어댑터 저장 완료: {adapter_dir}")
    print(f"loss 로그 저장 완료: {loss_log_path}")


if __name__ == "__main__":
    main()
