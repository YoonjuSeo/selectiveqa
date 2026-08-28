# -*- coding: utf-8 -*-
"""
evaluate_followup.py — 후속실험(사전등록 260817) H1~H5 판정 + risk-coverage.

사용법 (프로젝트 루트에서):
  python src/evaluation/evaluate_followup.py
  python src/evaluation/evaluate_followup.py --n-boot 1000        # 빠른 예비 실행
  python src/evaluation/evaluate_followup.py --h4-signal margin_full

입력:  results/preds_M0_v2.jsonl, preds_M1_v2_s{seed}.jsonl, preds_M2_r05_s{seed}.jsonl
출력:  results/metrics_followup_r05.json, results/risk_coverage_r05.png + 콘솔 판정

easy/hard UA 구분 (check_fields 검증 결과로 고정):
  easy = question_id가 'unans_' 시작 (300건) / hard = '-hd-' 포함 (267건, -er/-ts)

사전등록 판정 기준 (설계서 §3):
  H1  easy·hard 각각: M2 환각률 95% CI 상한 < M1 CI 하한, 3/3 시드
  H2  응답가능 전체 EM 차이(M2−M1) 95% CI 하한 > −0.03, 3/3 시드
  H3  M1: hard UA 환각 vs 응답가능 정답 분리 AUROC CI 하한 > 0.85, 3/3 시드
H4  M2: UA(easy+hard) 내 무응답 결정 예측 AUROC CI 하한 > 0.75, 3/3 시드
      ※ 부칙 확정(260828): 주 신호 = m1_conf (동일 시드 M1 answer-span
        confidence의 역수). 사전등록 원문의 '분기 이전 로그확률'의 문자적
        구현은 greedy 결정과 동치(순환)여서 채택 불가 — 변경 기록 참조.
        민감도: entropy_full·margin_full 병행 보고.
  H5  M2의 전 유형 ΔECE(대 M0) 95% CI 상한 < 0, 3/3 시드
      (사전등록 원문 확인 완료 — 가정 구현과 일치)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import re
import string
import unicodedata
from collections import defaultdict

import numpy as np
import yaml

# ---------------------------------------------------------------- 설정/채점 (evaluate.py와 동일 정의)
_PUNCT = set(string.punctuation) | {"·", "…", "「", "」", "『", "』", "%"}
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize(text):
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return "".join(ch for ch in text if not ch.isspace() and ch not in _PUNCT)


def extract_numbers(text):
    if text is None:
        return []
    return [float(m.replace(",", "")) for m in _NUM_RE.findall(str(text))]


def is_correct(pred, gold, qtype, tol):
    if qtype == "numeric_reasoning":
        gold_nums, pred_nums = extract_numbers(gold), extract_numbers(pred)
        if gold_nums and pred_nums:
            return any(abs(p - g) <= tol * max(1.0, abs(g))
                       for g in gold_nums for p in pred_nums)
        return normalize(pred) == normalize(gold)
    if qtype == "yes_no":
        from evaluation.polarity_v2 import polarity_v2 as polarity
        gp, pp = polarity(gold), polarity(pred)
        if gp is not None and pp is not None:
            return gp == pp
        return normalize(pred) == normalize(gold)
    return normalize(pred) == normalize(gold)


def ece(conf, corr, n_bins):
    conf = np.asarray(conf, float)
    corr = np.asarray(corr, float)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    v, n = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            v += (m.sum() / n) * abs(corr[m].mean() - conf[m].mean())
    return float(v)


def auroc(scores, labels):
    """labels(0/1)에 대해 scores가 1을 상위에 두는 정도 (Mann-Whitney)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allc = np.concatenate([pos, neg])
    order = np.argsort(allc, kind="mergesort")
    ranks = np.empty(len(allc), float)
    sv, i = allc[order], 0
    while i < len(allc):
        j = i
        while j + 1 < len(allc) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


# ---------------------------------------------------------------- 로드/조인
def load_preds(path):
    return {r["question_id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def load_exclusions(res_dir, names):
    """excluded_gold_v2*.json 에서 제외 question_id 집합을 유연하게 수집."""
    ids = set()
    for name in names:
        p = Path(res_dir) / name
        if not p.exists():
            continue
        data = json.load(open(p, encoding="utf-8"))
        items = data.values() if isinstance(data, dict) else data
        for it in items:
            if isinstance(it, str):
                ids.add(it)
            elif isinstance(it, dict) and "question_id" in it:
                ids.add(it["question_id"])
            elif isinstance(it, list):
                ids.update(x for x in it if isinstance(x, str))
        print(f"[제외] {name}: 누적 {len(ids)}건")
    return ids


def ua_kind(qid):
    if qid.startswith("unans_"):
        return "easy"
    if "-hd-" in qid:
        return "hard"
    return None


def is_abstain(rec):
    return rec.get("answerable_pred") is False


def build_rows(m0, m1, m2, tol, excluded):
    """세 조건을 question_id로 조인한 공통 행."""
    common = sorted((set(m0) & set(m1) & set(m2)) - excluded)
    rows = []
    for qid in common:
        a, b, c = m0[qid], m1[qid], m2[qid]
        ga = bool(a.get("gold_answerable", True))
        row = {
            "qid": qid, "type": a["type"], "gold_answerable": ga,
            "ua": ua_kind(qid) if not ga else None,
            "hd_method": ("er" if qid.endswith("-hd-er") else
                          "ts" if qid.endswith("-hd-ts") else None),
        }
        for tag, r in (("M0", a), ("M1", b), ("M2", c)):
            row[f"conf_{tag}"] = float(r["confidence"])
            row[f"abstain_{tag}"] = is_abstain(r)
            row[f"entropy_full_{tag}"] = r.get("entropy_full")
            row[f"margin_full_{tag}"] = r.get("margin_full")
            row[f"minprob_{tag}"] = r.get("min_token_prob")
            if ga:
                row[f"correct_{tag}"] = int(is_correct(
                    r["prediction"], r["gold_answer"], r["type"], tol))
        rows.append(row)
    return rows


# ---------------------------------------------------------------- 시드별 통계 (한 리샘플에 대해)
def h4_score(row, cond, signal):
    """무응답 예측 신호. 값이 클수록 '무응답일 것'을 예측하도록 부호 통일."""
    if signal == "entropy_full":
        v = row[f"entropy_full_{cond}"]
        return v if v is not None else np.nan          # 엔트로피↑ = 불확실↑
    if signal == "margin_full":
        v = row[f"margin_full_{cond}"]
        return -v if v is not None else np.nan         # 마진↓ = 불확실↑
    if signal == "one_minus_conf":
        return 1.0 - row[f"conf_{cond}"]
    if signal == "m1_conf":                            # ← 추가
        return 1.0 - row["conf_M1"]                    # M1 잔존 신호↓ = 무응답 예측↑ (부칙 확정 신호)
    raise ValueError(signal)


def scalars_one_sample(rows, n_bins, types, signal):
    """판정에 필요한 스칼라들을 한 표본에서 계산."""
    ans = [r for r in rows if r["gold_answerable"]]
    easy = [r for r in rows if r["ua"] == "easy"]
    hard = [r for r in rows if r["ua"] == "hard"]
    out = {}

    # H1: 조건별 환각률 (easy/hard)
    for kind, sub in (("easy", easy), ("hard", hard)):
        for c in ("M1", "M2"):
            out[f"hall_{kind}_{c}"] = float(np.mean(
                [not r[f"abstain_{c}"] for r in sub])) if sub else np.nan

    # H2: 응답가능 EM 차이 (무응답은 오답 처리 — EM 정의상 자동으로 0)
    em1 = float(np.mean([r["correct_M1"] and not r["abstain_M1"] for r in ans]))
    em2 = float(np.mean([r["correct_M2"] and not r["abstain_M2"] for r in ans]))
    out["em_M1"], out["em_M2"], out["d_em"] = em1, em2, em2 - em1

    # H3: M1 — hard UA 환각 vs 응답가능 정답 분리 AUROC (confidence 기준)
    pos = [r["conf_M1"] for r in ans if r["correct_M1"] and not r["abstain_M1"]]
    neg = [r["conf_M1"] for r in hard if not r["abstain_M1"]]
    out["auroc_h3"] = auroc(pos + neg, [1] * len(pos) + [0] * len(neg))

    # H4: M2 — UA 내 무응답 결정 예측 AUROC (신호 기반)
    ua_all = easy + hard
    sc = [h4_score(r, "M2", signal) for r in ua_all]
    lb = [int(r["abstain_M2"]) for r in ua_all]
    pair = [(s, l) for s, l in zip(sc, lb) if not np.isnan(s)]
    out["auroc_h4"] = auroc([p[0] for p in pair], [p[1] for p in pair]) if pair else np.nan

    # H5: 전 유형 ΔECE (M2 − M0), 응답가능 문항 (무응답 문항은 스코프 상이 → 제외)
    for t in types:
        sub = [r for r in ans if r["type"] == t]
        if not sub:
            continue
        e0 = ece([r["conf_M0"] for r in sub], [r["correct_M0"] for r in sub], n_bins)
        e2 = ece([r["conf_M2"] for r in sub],
                 [r["correct_M2"] and not r["abstain_M2"] for r in sub], n_bins)
        out[f"d_ece_{t}"] = e2 - e0
    return out


def bootstrap(rows, n_bins, types, signal, n_boot, seed):
    """층화(4유형 + easy + hard) paired bootstrap — 세 조건 동일 인덱스."""
    rng = np.random.default_rng(seed)
    strata = {t: [r for r in rows if r["gold_answerable"] and r["type"] == t]
              for t in types}
    strata["easy"] = [r for r in rows if r["ua"] == "easy"]
    strata["hard"] = [r for r in rows if r["ua"] == "hard"]
    samples = defaultdict(list)
    for _ in range(n_boot):
        rs = []
        for sub in strata.values():
            if sub:
                idx = rng.integers(0, len(sub), len(sub))
                rs.extend(sub[i] for i in idx)
        for k, v in scalars_one_sample(rs, n_bins, types, signal).items():
            samples[k].append(v)
    ci = {}
    for k, vals in samples.items():
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if vals:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            ci[k] = {"lo": float(lo), "hi": float(hi)}
    return ci


# ---------------------------------------------------------------- risk-coverage
def risk_coverage(rows, cond, use_filter):
    """coverage = 응답 유지 비율, risk = 유지분 내 (응답가능 오답 + UA 확답) 비율.

    use_filter=True: confidence 임계값 τ 스윕 (모델 무응답 + 필터 제거를 모두 비응답 처리).
    use_filter=False: 모델의 무응답 행동만 반영한 단일 운영점.
    """
    def point(keep_mask):
        n = len(rows)
        kept = [r for r, k in zip(rows, keep_mask) if k]
        if not kept:
            return None
        bad = sum((not r["gold_answerable"]) or (not r[f"correct_{cond}"])
                  for r in kept)
        return {"coverage": len(kept) / n, "risk": bad / len(kept)}

    base_keep = [not r[f"abstain_{cond}"] for r in rows]
    if not use_filter:
        return [point(base_keep)]
    taus = np.unique([r[f"conf_{cond}"] for r in rows])
    curve = []
    for tau in np.concatenate([[0.0], taus]):
        keep = [bk and r[f"conf_{cond}"] >= tau for r, bk in zip(rows, base_keep)]
        pt = point(keep)
        if pt:
            pt["tau"] = float(tau)
            curve.append(pt)
    return curve


def operating_points(curve, cov_target=0.9, risk_target=0.05):
    """사전 지정 운영점 2개: coverage≥0.9에서의 최소 risk / risk≤0.05에서의 최대 coverage."""
    at_cov = [p for p in curve if p["coverage"] >= cov_target]
    at_risk = [p for p in curve if p["risk"] <= risk_target]
    return {
        "risk_at_cov0.9": min((p["risk"] for p in at_cov), default=None),
        "cov_at_risk0.05": max((p["coverage"] for p in at_risk), default=None),
    }


# ---------------------------------------------------------------- 판정
def judge_seed(point, ci, margin=-0.03, h3_floor=0.85, h4_floor=0.75, types=()):
    j = {}
    j["H1_easy"] = ci["hall_easy_M2"]["hi"] < ci["hall_easy_M1"]["lo"]
    j["H1_hard"] = ci["hall_hard_M2"]["hi"] < ci["hall_hard_M1"]["lo"]
    j["H1"] = j["H1_easy"] and j["H1_hard"]
    j["H2"] = ci["d_em"]["lo"] > margin
    j["H3"] = ci["auroc_h3"]["lo"] > h3_floor
    j["H4"] = ci["auroc_h4"]["lo"] > h4_floor
    j["H5"] = all(ci.get(f"d_ece_{t}", {}).get("hi", 1) < 0 for t in types)  # [가정 기준]
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--m0", default="preds_M0_v2.jsonl")
    ap.add_argument("--m1-glob", default="preds_M1_v2_s*.jsonl")
    ap.add_argument("--m2-glob", default="preds_M2_r05_s*.jsonl")
    ap.add_argument("--tag", default="r05")
    ap.add_argument("--n-boot", type=int, default=None, help="미지정 시 config 값(10000)")
    ap.add_argument("--h4-signal", default="entropy_full",
                    choices=["entropy_full", "margin_full", "one_minus_conf", "m1_conf"])
    ap.add_argument("--exclude-files", nargs="*",
                    default=["excluded_gold_v2.json", "excluded_gold_v2_manual.json"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    types = cfg["data"]["types"]
    n_bins = cfg["eval"]["n_bins"]
    n_boot = args.n_boot or cfg["eval"]["n_bootstrap"]
    tol = cfg["eval"]["numeric_tolerance"]
    res_dir = Path(cfg["paths"]["results_dir"])

    excluded = load_exclusions(res_dir, args.exclude_files)
    m0 = load_preds(res_dir / args.m0)

    def seed_of(p):
        m = re.search(r"_s(\d+)\.jsonl$", p.name)
        return m.group(1) if m else p.stem

    m1_files = {seed_of(p): p for p in sorted(res_dir.glob(args.m1_glob))}
    m2_files = {seed_of(p): p for p in sorted(res_dir.glob(args.m2_glob))}
    seeds = sorted(set(m1_files) & set(m2_files))
    if not seeds:
        raise SystemExit("M1/M2 시드 쌍을 찾지 못했습니다. glob 패턴 확인.")
    print(f"시드 쌍: {seeds} · bootstrap {n_boot}회 · H4 신호 = {args.h4_signal}")
    print("[부칙 확정]")
    #print("※ H4 신호 산식과 H5 판정 기준은 사전등록 부칙/원문과 대조 후 확정할 것.\n")

    results, judges = {}, {}
    rc_store = {}
    for s in seeds:
        rows = build_rows(m0, load_preds(m1_files[s]), load_preds(m2_files[s]),
                          tol, excluded)
        n_easy = sum(r["ua"] == "easy" for r in rows)
        n_hard = sum(r["ua"] == "hard" for r in rows)
        point = scalars_one_sample(rows, n_bins, types, args.h4_signal)
        ci = bootstrap(rows, n_bins, types, args.h4_signal, n_boot, cfg["seed"])
        j = judge_seed(point, ci, types=types)
        results[s] = {"point": point, "ci": ci, "judge": j,
                      "n": {"rows": len(rows), "easy": n_easy, "hard": n_hard}}
        judges[s] = j

        # risk-coverage: M1+필터 곡선 vs M2 단일점 vs M2+필터 곡선
        rc_store[s] = {
            "M1_filter": risk_coverage(rows, "M1", True),
            "M2_point": risk_coverage(rows, "M2", False),
            "M2_filter": risk_coverage(rows, "M2", True),
        }

        def ci_s(k):
            c = ci.get(k, {})
            return f"[{c.get('lo', float('nan')):.3f}, {c.get('hi', float('nan')):.3f}]"

        print(f"======== 시드 {s} (n={len(rows)}, easy={n_easy}, hard={n_hard}) ========")
        print(f"H1 환각률 easy: M1 {point['hall_easy_M1']:.3f} {ci_s('hall_easy_M1')}"
              f" vs M2 {point['hall_easy_M2']:.3f} {ci_s('hall_easy_M2')}"
              f" → {'충족' if j['H1_easy'] else '미충족'}")
        print(f"H1 환각률 hard: M1 {point['hall_hard_M1']:.3f} {ci_s('hall_hard_M1')}"
              f" vs M2 {point['hall_hard_M2']:.3f} {ci_s('hall_hard_M2')}"
              f" → {'충족' if j['H1_hard'] else '미충족'}")
        print(f"H2 EM: M1 {point['em_M1']:.3f} → M2 {point['em_M2']:.3f}"
              f" (Δ {point['d_em']:+.3f} {ci_s('d_em')}, 마진 −0.03)"
              f" → {'충족' if j['H2'] else '미충족'}")
        print(f"H3 M1 hard-UA 분리 AUROC: {point['auroc_h3']:.3f} {ci_s('auroc_h3')}"
              f" (하한>0.85) → {'충족' if j['H3'] else '미충족'}")
        print(f"H4 M2 무응답 정렬 AUROC: {point['auroc_h4']:.3f} {ci_s('auroc_h4')}"
              f" (하한>0.75, 신호={args.h4_signal}) → {'충족' if j['H4'] else '미충족'}")
        dstr = " · ".join(f"{t} {point.get(f'd_ece_{t}', float('nan')):+.3f}"
                          f" {ci_s(f'd_ece_{t}')}" for t in types)
        print(f"H5 ΔECE(M2−M0): {dstr} → {'충족' if j['H5'] else '미충족'} [원문 일치]")
        op1 = operating_points(rc_store[s]["M1_filter"])
        m2p = rc_store[s]["M2_point"][0]
        print(f"운영점 — M1+필터: risk@cov0.9={op1['risk_at_cov0.9']:.3f}"
              f" / cov@risk0.05={op1['cov_at_risk0.05'] if op1['cov_at_risk0.05'] is not None else float('nan'):.3f}"
              f" · M2 그대로: cov={m2p['coverage']:.3f}, risk={m2p['risk']:.3f}\n")

    print("======== 시드 집계 ========")
    summary = {}
    for h in ("H1", "H2", "H3", "H4", "H5"):
        n_pass = sum(judges[s][h] for s in seeds)
        verdict = ("강건 채택" if n_pass == len(seeds) >= 3 else
                   "조건부 채택" if len(seeds) >= 3 and n_pass >= 2 else
                   f"{n_pass}/{len(seeds)}")
        summary[h] = {"pass": int(n_pass), "total": len(seeds), "verdict": verdict}
        print(f"{h}: {n_pass}/{len(seeds)} → {verdict}")

    # 그림: risk-coverage (시드 42 대표 + 전 시드 반투명)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for i, s in enumerate(seeds):
            a = 1.0 if i == 0 else 0.35
            c1 = rc_store[s]["M1_filter"]
            c2 = rc_store[s]["M2_filter"]
            ax.plot([p["coverage"] for p in c1], [p["risk"] for p in c1],
                    color="tab:blue", alpha=a,
                    label="M1 + threshold filter" if i == 0 else None)
            ax.plot([p["coverage"] for p in c2], [p["risk"] for p in c2],
                    color="tab:green", alpha=a,
                    label="M2 + threshold filter" if i == 0 else None)
            m2p = rc_store[s]["M2_point"][0]
            ax.scatter([m2p["coverage"]], [m2p["risk"]], color="tab:red", alpha=a,
                       zorder=5, label="M2 (as-is)" if i == 0 else None)
        ax.axvline(0.9, ls=":", c="gray")
        ax.axhline(0.05, ls=":", c="gray")
        ax.set_xlabel("coverage")
        ax.set_ylabel("risk (error + hallucination among answered)")
        ax.set_title(f"Risk–coverage ({args.tag})")
        ax.legend()
        fig.tight_layout()
        fig_path = res_dir / f"risk_coverage_{args.tag}.png"
        fig.savefig(fig_path, dpi=150)
        print(f"그림 저장: {fig_path}")
    except Exception as e:  # matplotlib 미설치 등
        print(f"[그림 생략] {e}")

    out = {"per_seed": results, "seed_summary": summary,
           "risk_coverage": rc_store,
           "settings": {"h4_signal": args.h4_signal, "n_boot": n_boot,
                        "excluded": sorted(excluded),
                        "ua_rule": "easy=unans_* / hard=*-hd-{er,ts}"}}
    out_path = res_dir / f"metrics_followup_{args.tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()