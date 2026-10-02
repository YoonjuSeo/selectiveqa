# -*- coding: utf-8 -*-
"""
lora_cosine.py — LoRA 업데이트 ΔW = (α/r)·B·A 의 모델 간 코사인 유사도 (Tier 0-b 재현용)

ΔW 를 직접 만들지 않고 r×r 행렬만으로 정확히 계산한다:
  <B1A1, B2A2>_F = sum( (B1ᵀB2) ∘ (A1A2ᵀ) ),   ||BA||_F² = sum( (BᵀB) ∘ (AAᵀ) )
α/r 은 모든 어댑터에서 같으므로(r16/α32) 코사인에서 약분된다.

사용 (레포 루트):
  # A.X: M1↔M2(같은 시드) 본비교 + 같은 조건·다른 시드 노이즈 기준선
  python lora_cosine.py --root results/ax31

  # 방법 검증: EXAONE 에 돌려 Tier 0-b 값(M1↔M2 0.026, 노이즈 0.157~0.329)이 재현되는지 확인
  python lora_cosine.py --root results

출력: 전역 코사인(전체 모듈을 이어 붙인 ΔW 기준)과 모듈별 코사인의 평균.
"""
import argparse
import itertools
import json
import struct
from pathlib import Path

try:
    import torch
    from safetensors.torch import load_file as _load

    def load(p):
        return {k: v.double() for k, v in _load(str(p)).items()}
except ImportError:  # torch 가 없으면 numpy 로 직접 파싱 (F32/F16/BF16)
    import numpy as np

    def load(p):
        b = Path(p).read_bytes()
        n = struct.unpack("<Q", b[:8])[0]
        hdr = json.loads(b[8:8 + n])
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            raw = b[8 + n + s: 8 + n + e]
            if v["dtype"] == "BF16":
                a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                a = np.frombuffer(raw, dtype={"F32": np.float32, "F16": np.float16}[v["dtype"]])
            out[k] = a.astype(np.float64).reshape(v["shape"])
        return out


def modules(sd):
    """{모듈명: (A, B)} — peft 키 '<모듈>.lora_A.weight' / '<모듈>.lora_B.weight'."""
    mods = {}
    for k, v in sd.items():
        for tag, idx in ((".lora_A.weight", 0), (".lora_B.weight", 1)):
            if k.endswith(tag):
                mods.setdefault(k[: -len(tag)], [None, None])[idx] = v
    bad = [m for m, ab in mods.items() if ab[0] is None or ab[1] is None]
    if bad or not mods:
        raise SystemExit(f"LoRA A/B 짝을 찾지 못함: {bad[:3] or '키 없음'}")
    return mods


def inner(a1, b1, a2, b2):
    return float(((b1.T @ b2) * (a1 @ a2.T)).sum())


def cosine(p1, p2):
    m1, m2 = modules(load(p1)), modules(load(p2))
    if set(m1) != set(m2):
        raise SystemExit(f"모듈 구성이 다름: {p1} vs {p2}")
    dot = n1 = n2 = 0.0
    per = []
    for name in sorted(m1):
        (a1, b1), (a2, b2) = m1[name], m2[name]
        d, x, y = inner(a1, b1, a2, b2), inner(a1, b1, a1, b1), inner(a2, b2, a2, b2)
        dot, n1, n2 = dot + d, n1 + x, n2 + y
        per.append(d / (x * y) ** 0.5)
    return dot / (n1 * n2) ** 0.5, sum(per) / len(per), len(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/ax31")
    ap.add_argument("--m1", default="qlora_adapter_s{seed}")
    ap.add_argument("--m2", default="qlora_adapter_r05_s{seed}")
    ap.add_argument("--seeds", default="42,43,44")
    args = ap.parse_args()
    seeds = args.seeds.split(",")
    path = lambda pat, s: Path(args.root) / pat.format(seed=s) / "adapter_model.safetensors"

    print(f"[{args.root}]  (전역 코사인 / 모듈별 평균)")
    print("본비교 M1↔M2 (같은 시드)")
    for s in seeds:
        g, m, n = cosine(path(args.m1, s), path(args.m2, s))
        print(f"  s{s}: {g:.4f} / {m:.4f}   (모듈 {n}개)")
    for label, pat in (("노이즈 기준선 M1↔M1 (다른 시드)", args.m1),
                       ("노이즈 기준선 M2↔M2 (다른 시드)", args.m2)):
        print(label)
        for a, b in itertools.combinations(seeds, 2):
            g, m, _ = cosine(path(pat, a), path(pat, b))
            print(f"  s{a}↔s{b}: {g:.4f} / {m:.4f}")


if __name__ == "__main__":
    main()