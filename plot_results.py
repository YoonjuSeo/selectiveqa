# -*- coding: utf-8 -*-
"""
plot_results.py (본 실험판) — 보고용 그림 3+1종 생성.

  1) delta_ece_by_type.png    : 유형별 ΔECE 막대(시드 평균) + 대표 시드 95% CI
  2) residual_gap.png         : 유형별 잔존 과신(gap) M0 vs M1
  3) reliability_diagrams.png : 유형×조건 reliability diagram (대표 시드)
  4) unanswerable.png         : 자발적 무응답률 + 환각률/환각 신뢰도 (H4)

사용법:
  python plot_results.py     # metrics_main.json, preds_*.jsonl 필요

한글 라벨을 쓰려면 KOREAN_FONT 에 시스템 폰트명(예: "NanumGothic")을 지정.
기본은 영문 라벨이라 폰트 없이도 동작합니다.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

KOREAN_FONT = None  # 예: "NanumGothic"
if KOREAN_FONT:
    matplotlib.rcParams["font.family"] = KOREAN_FONT
    matplotlib.rcParams["axes.unicode_minus"] = False

TYPE_LABELS = {
    "text_span": "Text span",
    "table_lookup": "Table lookup",
    "numeric_reasoning": "Numeric",
    "yes_no": "Yes/No",
}


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_items(metrics):
    """metrics_main.json 의 per_seed 항목을 (tag, point, ci) 리스트로."""
    return [(tag, v["point"], v["ci"]) for tag, v in metrics["per_seed"].items()]


def plot_delta_ece(metrics, types, out_path):
    items = seed_items(metrics)
    rep_tag, rep_point, rep_ci = items[0]  # CI 막대는 대표(첫) 시드 기준

    labels, means, err_lo, err_hi = [], [], [], []
    for t in types:
        vals = [p["per_type"][t]["delta_ece"] for _, p, _ in items if t in p["per_type"]]
        if not vals:
            continue
        mean = float(np.mean(vals))
        ci = rep_ci.get(f"delta_ece:{t}", {})
        rep = rep_point["per_type"][t]["delta_ece"]
        labels.append(TYPE_LABELS.get(t, t))
        means.append(mean)
        err_lo.append(max(0.0, rep - ci.get("lo", rep)))
        err_hi.append(max(0.0, ci.get("hi", rep) - rep))

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    colors = ["#4878A8" if d <= 0 else "#C0504D" for d in means]
    ax.bar(x, means, yerr=[err_lo, err_hi], capsize=4, color=colors)
    # 시드별 점 오버레이 (변동성 시각화)
    for i, t in enumerate([t for t in types if TYPE_LABELS.get(t, t) in labels]):
        ys = [p["per_type"][t]["delta_ece"] for _, p, _ in items if t in p["per_type"]]
        ax.scatter([i] * len(ys), ys, color="black", s=12, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("ΔECE = ECE(M1) − ECE(M0)")
    ax.set_title(f"ΔECE by type (bar = seed mean, dots = seeds, CI = {rep_tag})")
    contrast_vals = [p.get("contrast") for _, p, _ in items if p.get("contrast") is not None]
    if contrast_vals:
        cc = rep_ci.get("contrast", {})
        ax.text(0.02, 0.95,
                f"contrast mean = {np.mean(contrast_vals):.3f} "
                f"({rep_tag} CI [{cc.get('lo', float('nan')):.3f}, {cc.get('hi', float('nan')):.3f}])",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="#F0F0F0"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"저장: {out_path}")


def plot_residual_gap(metrics, types, out_path):
    items = seed_items(metrics)
    labels = [TYPE_LABELS.get(t, t) for t in types]
    g0 = [np.mean([p["per_type"][t]["gap_M0"] for _, p, _ in items]) for t in types]
    g1 = [np.mean([p["per_type"][t]["gap_M1"] for _, p, _ in items]) for t in types]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, g0, w, label="M0 (base)", color="#9AAFC8")
    ax.bar(x + w / 2, g1, w, label="M1 (fine-tuned)", color="#C0504D")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("gap = mean confidence − accuracy")
    ax.set_title("Residual overconfidence by type (seed mean)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"저장: {out_path}")


def reliability_data(rows, n_bins):
    conf = np.array([r["confidence"] for r in rows])
    corr = np.array([r["correct"] for r in rows], dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, weights = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf <= hi) if lo == 0 else (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        centers.append(conf[mask].mean())
        accs.append(corr[mask].mean())
        weights.append(mask.sum())
    return centers, accs, weights


def plot_reliability(cfg, out_path):
    """M0와 대표 M1 시드의 응답가능 문항으로 유형×조건 diagram."""
    import evaluate  # 채점 로직 재사용

    res_dir = Path(cfg["paths"]["results_dir"])
    types = cfg["data"]["types"]
    n_bins = cfg["eval"]["n_bins"]
    tol = cfg["eval"]["numeric_tolerance"]

    m1_tag, m1_path = evaluate.find_m1_files(res_dir)[0]
    paths = {"M0": res_dir / "preds_M0.jsonl", m1_tag: m1_path}
    preds = {}
    for cond, path in paths.items():
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        rows = [r for r in rows if r.get("gold_answerable", True)]
        for r in rows:
            r["correct"] = int(evaluate.is_correct(
                r["prediction"], r["gold_answer"], r["type"], tol))
        preds[cond] = rows

    conds = list(preds)
    fig, axes = plt.subplots(2, len(types), figsize=(4 * len(types), 7.5),
                             sharex=True, sharey=True)
    for row_i, cond in enumerate(conds):
        for col_i, t in enumerate(types):
            ax = axes[row_i][col_i]
            sub = [r for r in preds[cond] if r["type"] == t]
            ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
            if sub:
                centers, accs, weights = reliability_data(sub, n_bins)
                sizes = 200 * np.array(weights) / max(weights)
                ax.scatter(centers, accs, s=sizes, color="#4878A8", alpha=0.8)
                ax.plot(centers, accs, color="#4878A8", linewidth=1)
            ax.set_title(f"{cond} · {TYPE_LABELS.get(t, t)} (n={len(sub)})", fontsize=10)
            if row_i == 1:
                ax.set_xlabel("Confidence")
            if col_i == 0:
                ax.set_ylabel("Accuracy")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    fig.suptitle("Reliability diagrams, answerable only (point size ∝ bin count)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"저장: {out_path}")


def plot_unanswerable(metrics, out_path):
    """H4 그림: 자발적 무응답률(응답가능) + 환각률·환각 신뢰도(응답불가능)."""
    items = seed_items(metrics)
    with_unans = [p for _, p, _ in items if "unans" in p]
    if not with_unans:
        print("[안내] 무응답 지표 없음 — unanswerable.png 생략")
        return
    abst0 = np.mean([p["abstain_rate_M0"] for _, p, _ in items])
    abst1 = np.mean([p["abstain_rate_M1"] for _, p, _ in items])
    hall0 = np.mean([p["unans"]["hall_rate_M0"] for p in with_unans])
    hall1 = np.mean([p["unans"]["hall_rate_M1"] for p in with_unans])
    hc0 = np.nanmean([p["unans"]["hall_conf_M0"] for p in with_unans])
    hc1 = np.nanmean([p["unans"]["hall_conf_M1"] for p in with_unans])

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    panels = [
        ("Spontaneous abstention\n(answerable Qs)", abst0, abst1),
        ("Hallucination rate\n(unanswerable Qs)", hall0, hall1),
        ("Hallucination confidence\n(unanswerable Qs)", hc0, hc1),
    ]
    for ax, (title, v0, v1) in zip(axes, panels):
        ax.bar([0, 1], [v0, v1], color=["#9AAFC8", "#C0504D"], width=0.55)
        ax.set_xticks([0, 1], ["M0", "M1"])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1)
        for x, v in ((0, v0), (1, v1)):
            ax.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    fig.suptitle("H4: abstention collapse & confident hallucination (seed mean)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"저장: {out_path}")


def main():
    cfg = load_config()
    res_dir = Path(cfg["paths"]["results_dir"])
    with open(res_dir / "metrics_main.json", encoding="utf-8") as f:
        metrics = json.load(f)

    types = cfg["data"]["types"]
    plot_delta_ece(metrics, types, res_dir / "delta_ece_by_type.png")
    plot_residual_gap(metrics, types, res_dir / "residual_gap.png")
    plot_reliability(cfg, res_dir / "reliability_diagrams.png")
    plot_unanswerable(metrics, res_dir / "unanswerable.png")


if __name__ == "__main__":
    main()
