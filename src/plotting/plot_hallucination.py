# -*- coding: utf-8 -*-
"""
plot_hallucination.py — 그림 5: 환각 vs 정답 confidence 분리 (selective prediction).

좌: M1의 응답가능 정답 confidence와 응답불가능 환각 confidence의 겹친 히스토그램
    + 정답 90% 보존 임계값 수직선과 환각 차단율 주석.
우: coverage-risk 트레이드오프 — 임계값을 옮길 때 (정답 보존율, 환각 차단율) 곡선.

사용법: python plot_hallucination.py   (main_code 폴더에서, 3시드 자동 처리)
출력: results/hallucination_separation.png
"""

import glob
import json
import unicodedata
import re

import matplotlib.pyplot as plt
import numpy as np


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    return re.sub(r"[\s\.\,\!\?]+$", "", s)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def split_confs(rows):
    hall = [r["confidence"] for r in rows
            if not r["gold_answerable"] and r["answerable_pred"] is not False]
    corr = [r["confidence"] for r in rows
            if r["gold_answerable"] and norm(r["prediction"]) == norm(r["gold_answer"])]
    return np.array(hall), np.array(corr)


def main():
    files = sorted(glob.glob("results/preds_M1_s*.jsonl"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)

    # ---- 좌: 대표 시드(첫 파일) 히스토그램
    rows = load(files[0])
    tag = files[0].split("preds_")[1].split(".jsonl")[0]
    hall, corr = split_confs(rows)
    bins = np.linspace(0, 1, 31)
    ax1.hist(corr, bins=bins, density=True, alpha=0.55, color="#4C72B0",
             label=f"Correct answers (n={len(corr)})")
    ax1.hist(hall, bins=bins, density=True, alpha=0.55, color="#C44E52",
             label=f"Hallucinations (n={len(hall)})")
    thr = np.percentile(corr, 10)          # 정답 90% 보존 임계값
    blocked = (hall < thr).mean()
    ax1.axvline(thr, color="k", ls="--", lw=1.2)
    ax1.annotate(f"threshold = {thr:.2f}\nkeeps 90% of correct\nblocks {blocked:.0%} of hallucinations",
                 xy=(thr, ax1.get_ylim()[1] * 0.55), xytext=(0.06, 0.62),
                 textcoords="axes fraction", fontsize=9,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    ax1.set_xlabel("Confidence"); ax1.set_ylabel("Density")
    ax1.set_title(f"M1 confidence distributions ({tag})")
    ax1.legend(fontsize=9, loc="upper left")

    # ---- 우: coverage-blocking 곡선 (3시드)
    for f in files:
        h, c = split_confs(load(f))
        ts = np.quantile(c, np.linspace(0.0, 0.5, 51))   # 정답 보존 100%~50% 구간
        keep = [(c >= t).mean() for t in ts]
        block = [(h < t).mean() for t in ts]
        s = f.split("preds_")[1].split(".jsonl")[0]
        ax2.plot(keep, block, marker="", lw=1.6, label=s)
    ax2.set_xlabel("Correct answers kept (coverage)")
    ax2.set_ylabel("Hallucinations blocked")
    ax2.set_xlim(0.5, 1.005); ax2.set_ylim(0, 1.02)
    ax2.axvline(0.9, color="gray", ls=":", lw=1)
    ax2.set_title("Threshold trade-off (3 seeds)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    fig.suptitle("Abstention behavior is erased, but the confidence signal survives",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = "results/hallucination_separation.png"
    fig.savefig(out, bbox_inches="tight")
    print("저장:", out)


if __name__ == "__main__":
    main()