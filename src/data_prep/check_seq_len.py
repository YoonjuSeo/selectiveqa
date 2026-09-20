# -*- coding: utf-8 -*-
"""
check_seq_len.py — 학습 전 토큰 길이 검사 (Phase 2 사전등록 §4 '지문 길이').

실제 토크나이저로 (시스템 프롬프트 + 지문 + 질문 + 타깃 JSON) 길이를 재어
max_seq_len 을 넘는 문항 수를 보고한다. 토크나이저만 내려받으므로 로컬 CPU 에서
돌아간다 (모델 가중치 불필요). 학습·추론과 같은 prompts.build_messages /
apply_template 을 써서 바이트 단위로 동일한 입력을 잰다.

사용:
  python src/data_prep/check_seq_len.py --config config_klue_exaone.yaml
  python src/data_prep/check_seq_len.py --config config_klue_kanana.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import yaml

from inference.prompts import apply_template, build_messages, build_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--files", nargs="*",
                    default=["klue_train.jsonl", "klue_train_mix_r05.jsonl", "klue_eval_full.jsonl"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    max_len = cfg["train"]["max_seq_len"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["base_model"],
                                        revision=cfg["model"].get("revision"),
                                        trust_remote_code=True)
    enable_thinking = cfg["model"].get("enable_thinking")   # Qwen3 전용, 없으면 None

    proc = Path(cfg["paths"]["processed_dir"])
    worst = 0
    for fn in args.files:
        p = proc / fn
        if not p.exists():
            print(f"[건너뜀] {p} 없음"); continue
        lens = []
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            prompt = apply_template(tok, build_messages(r["context"], r["question"]),
                                    enable_thinking=enable_thinking)
            gold = r["gold_answer"]
            gold0 = gold[0] if isinstance(gold, list) and gold else (gold or "")
            target = build_target(gold0, answerable=r.get("answerable", True))
            n = len(tok(prompt + target, add_special_tokens=False)["input_ids"])
            lens.append(n)
        lens.sort()
        over = sum(n > max_len for n in lens)
        worst = max(worst, lens[-1])
        print(f"{fn:28s} n={len(lens):5d}  중앙 {lens[len(lens)//2]:5d}  p90 {lens[int(.9*len(lens))]:5d}  "
              f"최대 {lens[-1]:5d}  |  {max_len} 초과 {over}건 ({100*over/len(lens):.2f}%)")
    print()
    if worst <= max_len:
        print(f"✓ 전 문항이 max_seq_len {max_len} 이내 — 절단 없음. 학습 진행.")
    else:
        print(f"✗ 초과 문항 존재. 사전등록 §4 대로 건수를 보고하고, 필요 시 "
              f"prepare_klue.py --max-context-chars 로 제한한 뒤 재구축.")


if __name__ == "__main__":
    main()
