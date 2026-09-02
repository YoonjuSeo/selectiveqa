# -*- coding: utf-8 -*-
"""
train_qlora.py — M1 조건: 응답가능 질문만으로 QLoRA 파인튜닝.

사용법:
  python train_qlora.py                # config.yaml 기준 학습, 어댑터 저장

프롬프트 토큰은 라벨 -100으로 마스킹하여 정답 JSON 부분만 학습한다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/ 를 import 경로에 추가

import json
import random

import torch
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

class EpochAdapterSaver(TrainerCallback):
    """에폭 종료마다 LoRA 어댑터만 별도 저장한다.

    [탐색적 분석 대비] 학습 진행량에 따른 confidence 팽창(용량-반응)을
    재학습 없이 사후 분석할 수 있도록 중간 어댑터를 남긴다.
    최종 판정용 어댑터({adapter_dir}_s{seed})와는 별개이며,
    저장 경로: {adapter_dir}_s{seed}_ep{N}. 어댑터만 저장하므로 용량 부담이 작다.
    """

    def __init__(self, model, base_dir):
        self.model = model
        self.base_dir = base_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        ep = int(round(state.epoch))
        path = f"{self.base_dir}_ep{ep}"
        self.model.save_pretrained(path)
        print(f"[체크포인트] 에폭 {ep} 어댑터 저장: {path}")


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class SFTDataset(Dataset):
    """프롬프트(마스킹) + 정답 JSON(학습 대상) 형태의 causal LM 데이터셋."""

    def __init__(self, jsonl_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.rows = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        messages = build_messages(row["context"], row["question"])
        prompt_text = apply_template(self.tokenizer, messages, self.enable_thinking)
        target_text = build_target(row["gold_answer"],
                                   answerable=row.get("answerable", True)
                                   ) + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + target_ids)[: self.max_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_len]
        # 지문이 길어 정답이 통째로 잘리면 학습 신호가 없으므로 경고성 처리:
        if all(l == -100 for l in labels):
            labels[-len(target_ids):] = target_ids[: len(labels)]

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)

    def pad(seq, value):
        return torch.cat([seq, torch.full((max_len - len(seq),), value, dtype=seq.dtype)])

    return {
        "input_ids": torch.stack([pad(b["input_ids"], pad_id) for b in batch]),
        "labels": torch.stack([pad(b["labels"], -100) for b in batch]),
        "attention_mask": torch.stack([pad(b["attention_mask"], 0) for b in batch]),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=None,
                    help="학습 시드 (본 실험: 42/43/44 각각 1회씩 실행. 미지정 시 config seed)")
    ap.add_argument("--train-file", default="train.jsonl",
                    help="학습 데이터 파일명 (M2: train_mix_r05/r10/r30.jsonl)")
    ap.add_argument("--tag", default="",
                    help="어댑터 이름 접미 (M2: _r05 등. 미지정 시 M1 호환)")
    ap.add_argument("--epochs", type=float, default=None,
                    help="에폭 수 재정의 (스모크: 1 등). 미지정 시 config 값 사용")
    
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    print(f"학습 시드: {seed}")

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
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora = LoraConfig(
        r=cfg["train"]["lora_r"],
        lora_alpha=cfg["train"]["lora_alpha"],
        lora_dropout=cfg["train"]["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_path = Path(cfg["paths"]["processed_dir"]) / args.train_file
    dataset = SFTDataset(train_path, tokenizer, cfg["train"]["max_seq_len"],
                     enable_thinking=cfg["model"].get("enable_thinking"))
    print(f"학습 데이터: {len(dataset)}건")

    targs = TrainingArguments(
        output_dir=str(Path(cfg["paths"]["results_dir"]) / f"train_ckpt_s{seed}"),
        num_train_epochs=args.epochs if args.epochs is not None else cfg["train"]["epochs"],
        learning_rate=float(cfg["train"]["lr"]),
        per_device_train_batch_size=cfg["train"]["batch_size"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        bf16=True,
        logging_steps=20,
        save_strategy="no",          # 파일럿: 최종 어댑터만 저장
        report_to="none",
        seed=seed,
    )

    adapter_dir = f'{cfg["model"]["adapter_dir"]}{args.tag}_s{seed}'   # 시드별 분리 저장
    callbacks = []
    if cfg["train"].get("save_epoch_adapters", True):
        callbacks.append(EpochAdapterSaver(model, adapter_dir))

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    trainer.train()

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"어댑터 저장 완료: {adapter_dir}")


if __name__ == "__main__":
    main()