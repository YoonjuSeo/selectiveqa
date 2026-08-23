# -*- coding: utf-8 -*-
"""
evaluate.py (본 실험판) — 채점(EM/F1) + 보정 지표(ECE/ΔECE/Δconf) + 무응답 지표(H4)
+ paired bootstrap CI + H1~H4 사전등록 판정 + 다중 시드 집계.

사용법:
  python evaluate.py                    # preds_M0.jsonl 과 preds_M1*.jsonl 자동 탐색
  python evaluate.py --m1 preds_M1_s42.jsonl   # 특정 시드만

입력:  results/preds_M0.jsonl, results/preds_M1_s{seed}.jsonl (또는 preds_M1.jsonl)
출력:  results/metrics_main.json + 콘솔 요약

사전 판정 기준 (설계서 §2 — 분석 전 고정):
  H1  전 유형에서 Δconf 95% CI 하한 > 0
  H2  contrast CI가 0 제외 그리고 |contrast| ≥ min_contrast(0.03)
  H3  숫자·표 유형 모두 ΔECE 95% CI 하한 > 0
  H4a 자발적 무응답률 차이(M1−M0) 95% CI 상한 < 0
  H4b 환각률·환각 신뢰도 차이(M1−M0) 모두 95% CI 하한 > 0
시드 집계: 3/3 충족 = 강건 채택, 2/3 = 조건부 채택, 그 외 = 미충족 (설계서 §7)

──────────────────────────────────────────────────────────────────────
[탐색적 지표 — 부록 보고용. H1~H4 판정 지표·기준은 변경하지 않음]
  기제 분석 (지도교수 피드백 2 "왜 팽창하는가"):
    - delta_conf_bothwrong : M0·M1 모두 오답인 문항에서의 Δconf (+CI).
      정확도 상승으로 설명 불가능한 팽창의 직접 증거.
    - entropy/margin        : answer 구간 평균 분포 엔트로피·top1−top2 마진의
      조건 간 변화 (run_inference.py가 로그한 경우). 분포 첨예화 증거.
  요인 분석 (피드백 3 "유형별 차이는 무엇 때문인가"):
    - delta_em, delta_gap   : ΔECE를 정확도 이득(Δacc)과 팽창(Δconf)으로 분해.
    - brier, auroc, aece    : ECE의 빈/정확도 수준 의존성에 대한 보조 보정 지표.
    - n_answer_tokens       : 유형별 답 토큰 수 (측정 인공물 점검).
  측정 민감도 (피드백 1 "측정 방식" 방어):
    - contrast_no_fallback  : 폴백 표본 제외 후 contrast (설계서 §9 명시 분석).
    - contrast_minprob      : min_token_prob 기반 confidence로 재계산한 contrast.
──────────────────────────────────────────────────────────────────────
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/ 를 import 경로에 추가
import argparse
import json
import re
import string
import unicodedata
from collections import defaultdict

import numpy as np
import yaml

from inference.prompts import parse_model_output 

ANSWERABLE_TYPES_KEY = "types"          # config의 4개 응답가능 유형
CONTRAST_POS = ("table_lookup", "numeric_reasoning")
CONTRAST_NEG = "text_span"


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- 채점 유틸 (파일럿과 동일)
_PUNCT = set(string.punctuation) | {"·", "…", "「", "」", "『", "』", "%"}
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


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
        from evaluation.polarity_v2 import polarity_v2 as polarity  # 최장 일치 우선 (단계 0 패치)

        gp, pp = polarity(gold), polarity(pred)
        if gp is not None and pp is not None:
            return gp == pp
        return normalize(pred) == normalize(gold)


    return normalize(pred) == normalize(gold)


def char_f1(pred, gold):
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return float(p == g)
    common = 0
    gold_chars = defaultdict(int)
    for ch in g:
        gold_chars[ch] += 1
    for ch in p:
        if gold_chars[ch] > 0:
            common += 1
            gold_chars[ch] -= 1
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def ece(confidences, corrects, n_bins):
    confidences = np.asarray(confidences, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(confidences)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        value += (mask.sum() / total) * abs(corrects[mask].mean() - confidences[mask].mean())
    return float(value)


# ------------------------------------------------- 탐색적 보정 지표 (부록용)
def adaptive_ece(confidences, corrects, n_bins):
    """등질량(equal-mass) 빈 ECE. 고정 빈 ECE의 빈 배치 민감성에 대한 보조 지표."""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(corrects, dtype=float)
    n = len(conf)
    if n == 0:
        return float("nan")
    order = np.argsort(conf)
    value = 0.0
    for chunk in np.array_split(order, min(n_bins, n)):
        if len(chunk) == 0:
            continue
        value += (len(chunk) / n) * abs(corr[chunk].mean() - conf[chunk].mean())
    return float(value)


def brier(confidences, corrects):
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(corrects, dtype=float)
    return float(np.mean((conf - corr) ** 2)) if len(conf) else float("nan")


def auroc(confidences, corrects):
    """confidence가 정답/오답을 변별하는 정도 (rank 기반 Mann-Whitney AUROC).

    팽창이 '수준'만 올리고 변별력은 유지하는지(기제 서사) 보는 용도.
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(corrects, dtype=int)
    pos, neg = conf[corr == 1], conf[corr == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allc = np.concatenate([pos, neg])
    order = np.argsort(allc, kind="mergesort")
    ranks = np.empty(len(allc), dtype=float)
    i = 0
    sorted_vals = allc[order]
    while i < len(allc):                       # 동률은 평균 순위
        j = i
        while j + 1 < len(allc) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------- 데이터 로드/조인
def find_m1_files(res_dir, explicit=None):
    """preds_M1_s*.jsonl 을 시드별로 수집. 없으면 preds_M1.jsonl 폴백."""
    if explicit:
        p = Path(explicit)
        return [(p.stem.replace("preds_", ""), p)]
    files = sorted(Path(res_dir).glob("preds_M1_s*.jsonl"))
    if files:
        return [(f.stem.replace("preds_", ""), f) for f in files]
    single = Path(res_dir) / "preds_M1.jsonl"
    if single.exists():
        return [("M1", single)]
    raise SystemExit("M1 예측 파일이 없습니다: preds_M1_s*.jsonl 또는 preds_M1.jsonl")


def load_preds(path):
    return {r["question_id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def is_abstain(rec):
    """무응답 판정: answerable_pred == False 인 경우만 (None=파싱실패는 별도 집계)."""
    return rec.get("answerable_pred") is False


def join_pair(m0, m1, tol):
    """M0/M1을 question_id로 조인. 응답가능/불가능 공통 스키마로 변환."""
    common = sorted(set(m0) & set(m1))
    if len(common) < max(len(m0), len(m1)):
        print(f"[경고] 공통 질문 {len(common)}건만 사용 (M0={len(m0)}, M1={len(m1)})")
    joined = []
    for qid in common:
        a, b = m0[qid], m1[qid]
        gold_ans = a.get("gold_answerable", True)
        row = {
            "question_id": qid,
            "type": a["type"],
            "gold_answerable": bool(gold_ans),
            "conf": {"M0": a["confidence"], "M1": b["confidence"]},
            "abstain": {"M0": is_abstain(a), "M1": is_abstain(b)},
            "parse_ok": {"M0": bool(a.get("parse_ok")), "M1": bool(b.get("parse_ok"))},
            "fallback": {"M0": a.get("conf_scope") != "answer_span",
                         "M1": b.get("conf_scope") != "answer_span"},
            # ---- 탐색적 필드 (구버전 예측 파일에는 없을 수 있음 → None 허용)
            "conf_min": {"M0": a.get("min_token_prob"), "M1": b.get("min_token_prob")},
            "n_ans_tok": {"M0": a.get("n_answer_tokens"), "M1": b.get("n_answer_tokens")},
            "entropy": {"M0": a.get("entropy_mean"), "M1": b.get("entropy_mean")},
            "margin": {"M0": a.get("margin_mean"), "M1": b.get("margin_mean")},
        }
        if gold_ans:
            row["correct"] = {
                "M0": int(is_correct(a["prediction"], a["gold_answer"], a["type"], tol)),
                "M1": int(is_correct(b["prediction"], b["gold_answer"], b["type"], tol)),
            }
            row["f1"] = {"M0": char_f1(a["prediction"], a["gold_answer"]),
                         "M1": char_f1(b["prediction"], b["gold_answer"])}
        joined.append(row)
    return joined


# ---------------------------------------------------------------- 지표 계산
def _bothwrong_delta_conf(sub):
    """M0·M1 모두 오답인 문항에서의 Δconf. 해당 문항이 없으면 (nan, 0)."""
    bw = [r for r in sub if r["correct"]["M0"] == 0 and r["correct"]["M1"] == 0]
    if not bw:
        return float("nan"), 0
    d = float(np.mean([r["conf"]["M1"] - r["conf"]["M0"] for r in bw]))
    return d, len(bw)


def compute_stats(rows, types, n_bins, full=False):
    """한 (리샘플된) 표본에 대해 지표를 계산한다.

    full=False : 부트스트랩용 경량 모드 — 판정 지표 + CI 추적 탐색 지표만.
    full=True  : 점추정용 — brier/auroc/aece/토큰수/엔트로피 등 부록 지표 포함.

    반환 dict 키:
      per_type[t]: n, em/f1/ece/conf (M0,M1), delta_ece, delta_conf, delta_em,
                   delta_gap, delta_conf_bothwrong (+ full 모드 부록 지표)
      contrast, delta_conf_bothwrong(전체), abstain_rate/delta,
      unans: abstain_acc, hall_rate, hall_conf (+delta)
    """
    ans = [r for r in rows if r["gold_answerable"]]
    unans = [r for r in rows if not r["gold_answerable"]]
    out = {"per_type": {}}

    for t in types:
        sub = [r for r in ans if r["type"] == t]
        if not sub:
            continue
        m = {"n": len(sub)}
        for c in ("M0", "M1"):
            m[f"em_{c}"] = float(np.mean([r["correct"][c] for r in sub]))
            m[f"f1_{c}"] = float(np.mean([r["f1"][c] for r in sub]))
            m[f"conf_{c}"] = float(np.mean([r["conf"][c] for r in sub]))
            m[f"ece_{c}"] = ece([r["conf"][c] for r in sub],
                                [r["correct"][c] for r in sub], n_bins)
            m[f"gap_{c}"] = m[f"conf_{c}"] - m[f"em_{c}"]
        m["delta_ece"] = m["ece_M1"] - m["ece_M0"]
        m["delta_conf"] = m["conf_M1"] - m["conf_M0"]
        # ---- 탐색적: ΔECE 분해용 (피드백 3) — Δacc(정확도 이득)와 Δgap(순팽창)
        m["delta_em"] = m["em_M1"] - m["em_M0"]
        m["delta_gap"] = m["gap_M1"] - m["gap_M0"]      # = delta_conf − delta_em
        # ---- 탐색적: 오답 조건부 팽창 (피드백 2) — 유형별
        m["delta_conf_bothwrong"], m["n_bothwrong"] = _bothwrong_delta_conf(sub)

        if full:  # ---- 부록 지표 (부트스트랩에서는 생략해 속도 유지)
            for c in ("M0", "M1"):
                confs = [r["conf"][c] for r in sub]
                corrs = [r["correct"][c] for r in sub]
                m[f"brier_{c}"] = brier(confs, corrs)
                m[f"auroc_{c}"] = auroc(confs, corrs)
                m[f"aece_{c}"] = adaptive_ece(confs, corrs, n_bins)
                toks = [r["n_ans_tok"][c] for r in sub if r["n_ans_tok"][c] is not None]
                m[f"n_ans_tok_{c}"] = float(np.mean(toks)) if toks else None
                ents = [r["entropy"][c] for r in sub if r["entropy"][c] is not None]
                m[f"entropy_{c}"] = float(np.mean(ents)) if ents else None
                mgs = [r["margin"][c] for r in sub if r["margin"][c] is not None]
                m[f"margin_{c}"] = float(np.mean(mgs)) if mgs else None
            m["delta_aece"] = m["aece_M1"] - m["aece_M0"]
            m["delta_brier"] = m["brier_M1"] - m["brier_M0"]
        out["per_type"][t] = m

    # ---- 탐색적: 오답 조건부 팽창 — 응답가능 전체 (주 CI 추적 대상)
    if ans:
        out["delta_conf_bothwrong"], out["n_bothwrong"] = _bothwrong_delta_conf(ans)

    pt = out["per_type"]
    if all(t in pt for t in CONTRAST_POS) and CONTRAST_NEG in pt:
        out["contrast"] = float(np.mean([pt[t]["delta_ece"] for t in CONTRAST_POS])
                                - pt[CONTRAST_NEG]["delta_ece"])

    # H4a: 응답가능 질문에 대한 자발적 무응답률
    if ans:
        for c in ("M0", "M1"):
            out[f"abstain_rate_{c}"] = float(np.mean([r["abstain"][c] for r in ans]))
        out["delta_abstain"] = out["abstain_rate_M1"] - out["abstain_rate_M0"]

    # H4b: 응답 불가능 질문 — 무응답 정확도 / 환각률 / 환각 신뢰도
    if unans:
        u = {"n": len(unans)}
        for c in ("M0", "M1"):
            abst = [r["abstain"][c] for r in unans]
            u[f"abstain_acc_{c}"] = float(np.mean(abst))
            u[f"hall_rate_{c}"] = 1.0 - u[f"abstain_acc_{c}"]
            hall_conf = [r["conf"][c] for r in unans if not r["abstain"][c]]
            u[f"hall_conf_{c}"] = float(np.mean(hall_conf)) if hall_conf else float("nan")
        u["delta_hall_rate"] = u["hall_rate_M1"] - u["hall_rate_M0"]
        u["delta_hall_conf"] = u["hall_conf_M1"] - u["hall_conf_M0"]
        out["unans"] = u
    return out


def flatten_for_ci(stats):
    """부트스트랩에서 CI를 추적할 스칼라들만 뽑는다."""
    flat = {}
    for t, m in stats["per_type"].items():
        flat[f"delta_ece:{t}"] = m["delta_ece"]
        flat[f"delta_conf:{t}"] = m["delta_conf"]
        flat[f"delta_conf_bothwrong:{t}"] = m["delta_conf_bothwrong"]  # 탐색적
    if "delta_conf_bothwrong" in stats:
        flat["delta_conf_bothwrong"] = stats["delta_conf_bothwrong"]   # 탐색적(전체)
    if "contrast" in stats:
        flat["contrast"] = stats["contrast"]
    if "delta_abstain" in stats:
        flat["delta_abstain"] = stats["delta_abstain"]
    if "unans" in stats:
        flat["delta_hall_rate"] = stats["unans"]["delta_hall_rate"]
        if not np.isnan(stats["unans"]["delta_hall_conf"]):
            flat["delta_hall_conf"] = stats["unans"]["delta_hall_conf"]
    return flat


def bootstrap_ci(rows, types, n_bins, n_boot, seed, alpha=0.05):
    """질문 단위 paired bootstrap: 층 = 4개 응답가능 유형 + unanswerable."""
    rng = np.random.default_rng(seed)
    strata = {t: [r for r in rows if r["gold_answerable"] and r["type"] == t] for t in types}
    strata["unanswerable"] = [r for r in rows if not r["gold_answerable"]]
    samples = defaultdict(list)
    for _ in range(n_boot):
        resampled = []
        for sub in strata.values():
            if not sub:
                continue
            idx = rng.integers(0, len(sub), len(sub))
            resampled.extend(sub[i] for i in idx)
        for key, val in flatten_for_ci(compute_stats(resampled, types, n_bins)).items():
            samples[key].append(val)
    ci = {}
    for key, vals in samples.items():
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            continue
        lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        ci[key] = {"lo": float(lo), "hi": float(hi)}
    return ci


# ---------------------------------------------------------------- H1~H4 판정
def judge(point, ci, types, min_contrast):
    j = {}
    # H1: 전 유형 Δconf CI 하한 > 0
    lows = [ci.get(f"delta_conf:{t}", {}).get("lo") for t in types]
    j["H1"] = all(lo is not None and lo > 0 for lo in lows)
    # H2: contrast CI 0 제외 & |contrast| ≥ 기준
    c, cc = point.get("contrast"), ci.get("contrast", {})
    j["H2"] = (c is not None and cc
               and (cc["lo"] > 0 or cc["hi"] < 0) and abs(c) >= min_contrast)
    # H3: 숫자·표 모두 ΔECE CI 하한 > 0
    j["H3"] = all(ci.get(f"delta_ece:{t}", {}).get("lo", -1) > 0 for t in CONTRAST_POS)
    # H4a: Δ자발적무응답 CI 상한 < 0
    j["H4a"] = ci.get("delta_abstain", {}).get("hi", 1) < 0
    # H4b: 환각률·환각신뢰도 Δ CI 하한 > 0 (둘 다)
    j["H4b"] = (ci.get("delta_hall_rate", {}).get("lo", -1) > 0
                and ci.get("delta_hall_conf", {}).get("lo", -1) > 0)
    return j


def sensitivity_report(rows, types, n_bins):
    """[탐색적] 측정 방식 민감도 — 점추정만 (피드백 1 방어, 설계서 §9).

    ① contrast_no_fallback : 두 조건 모두 answer_span으로 신뢰도를 잰 문항만.
    ② contrast_minprob     : confidence 대신 min_token_prob로 ECE·contrast 재계산.
    본판정 contrast와 부호·크기가 유지되면 결과가 측정 인공물이 아님을 방어.
    """
    def contrast_from(sub_rows, conf_key):
        pt = {}
        for t in list(CONTRAST_POS) + [CONTRAST_NEG]:
            sub = [r for r in sub_rows if r["gold_answerable"] and r["type"] == t
                   and r[conf_key]["M0"] is not None and r[conf_key]["M1"] is not None]
            if not sub:
                return None, {}
            d = {}
            for c in ("M0", "M1"):
                d[c] = ece([r[conf_key][c] for r in sub],
                           [r["correct"][c] for r in sub], n_bins)
            pt[t] = {"delta_ece": d["M1"] - d["M0"], "n": len(sub)}
        val = float(np.mean([pt[t]["delta_ece"] for t in CONTRAST_POS])
                    - pt[CONTRAST_NEG]["delta_ece"])
        return val, pt

    rep = {}
    no_fb = [r for r in rows if not r["fallback"]["M0"] and not r["fallback"]["M1"]]
    val, pt = contrast_from(no_fb, "conf")
    rep["contrast_no_fallback"] = {"value": val, "per_type": pt,
                                   "n_rows": len(no_fb)}
    val, pt = contrast_from(rows, "conf_min")
    rep["contrast_minprob"] = {"value": val, "per_type": pt}
    return rep


def quality_report(rows):
    """폴백률·파싱실패율을 조건별로 (설계서 §9 타당성 통제)."""
    rep = {}
    for c in ("M0", "M1"):
        rep[f"fallback_rate_{c}"] = float(np.mean([r["fallback"][c] for r in rows]))
        rep[f"parse_fail_rate_{c}"] = float(np.mean([not r["parse_ok"][c] for r in rows]))
    return rep


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--m1", default=None, help="특정 M1 예측 파일만 평가")
    args = ap.parse_args()

    cfg = load_config(args.config)
    types = cfg["data"][ANSWERABLE_TYPES_KEY]
    n_bins = cfg["eval"]["n_bins"]
    n_boot = cfg["eval"]["n_bootstrap"]
    tol = cfg["eval"]["numeric_tolerance"]
    min_contrast = cfg["eval"].get("min_contrast", 0.03)
    res_dir = Path(cfg["paths"]["results_dir"])

    m0 = load_preds(res_dir / "preds_M0.jsonl")
    m1_files = find_m1_files(res_dir, args.m1)
    print(f"M1 예측 파일 {len(m1_files)}개: {[t for t, _ in m1_files]}")

    all_results, all_judges = {}, {}
    for tag, path in m1_files:
        rows = join_pair(m0, load_preds(path), tol)
        point = compute_stats(rows, types, n_bins, full=True)
        ci = bootstrap_ci(rows, types, n_bins, n_boot, cfg["seed"])
        j = judge(point, ci, types, min_contrast)
        sens = sensitivity_report(rows, types, n_bins)
        all_results[tag] = {"point": point, "ci": ci,
                            "quality": quality_report(rows),
                            "sensitivity": sens, "judge": j}
        all_judges[tag] = j

        # ---- 콘솔 요약 (시드별)
        print(f"\n======== {tag} ========")
        print(f"{'유형':<20}{'n':>5}{'EM M0':>8}{'EM M1':>8}{'ΔECE':>8}"
              f"{'Δconf':>8}  ΔECE 95% CI")
        for t in types:
            if t not in point["per_type"]:
                continue
            m = point["per_type"][t]
            c = ci.get(f"delta_ece:{t}", {})
            print(f"{t:<20}{m['n']:>5}{m['em_M0']:>8.3f}{m['em_M1']:>8.3f}"
                  f"{m['delta_ece']:>8.3f}{m['delta_conf']:>8.3f}"
                  f"  [{c.get('lo', float('nan')):.3f}, {c.get('hi', float('nan')):.3f}]")
        if "contrast" in point:
            cc = ci.get("contrast", {})
            print(f"contrast = {point['contrast']:.3f} "
                  f"[{cc.get('lo'):.3f}, {cc.get('hi'):.3f}]")
        if "delta_abstain" in point:
            ca = ci.get("delta_abstain", {})
            print(f"자발적 무응답률: M0 {point['abstain_rate_M0']:.3f} → "
                  f"M1 {point['abstain_rate_M1']:.3f} "
                  f"(Δ {point['delta_abstain']:+.3f} "
                  f"[{ca.get('lo', float('nan')):.3f}, {ca.get('hi', float('nan')):.3f}])")
        if "unans" in point:
            u = point["unans"]
            ch, cf = ci.get("delta_hall_rate", {}), ci.get("delta_hall_conf", {})
            print(f"[응답불가능 n={u['n']}] 환각률 M0 {u['hall_rate_M0']:.3f} → "
                  f"M1 {u['hall_rate_M1']:.3f} "
                  f"[{ch.get('lo', float('nan')):.3f}, {ch.get('hi', float('nan')):.3f}] / "
                  f"환각 신뢰도 M0 {u['hall_conf_M0']:.3f} → M1 {u['hall_conf_M1']:.3f} "
                  f"[{cf.get('lo', float('nan')):.3f}, {cf.get('hi', float('nan')):.3f}]")
        q = all_results[tag]["quality"]
        print(f"폴백률 M0 {q['fallback_rate_M0']:.1%} / M1 {q['fallback_rate_M1']:.1%}"
              f" · 파싱실패율 M0 {q['parse_fail_rate_M0']:.1%} / M1 {q['parse_fail_rate_M1']:.1%}"
              f"  (10% 초과 또는 조건 간 5%p 초과 차이 시 점검 — 설계서 §5·§9)")
        print("판정:", ", ".join(f"{h}={'충족' if v else '미충족'}" for h, v in j.items()))

        # ---------------- 탐색적 지표 (부록 — 판정에 미사용) ----------------
        print("\n-- 탐색적: ΔECE 분해 + 기제 + 측정 (부록) --")
        print(f"{'유형':<20}{'Δacc':>7}{'Δconf':>7}{'Δgap':>7}"
              f"{'Δcf|둘다오답':>13}{'nBW':>5}{'ΔaECE':>8}{'AUROC M0→M1':>13}{'답토큰':>7}")
        for t in types:
            if t not in point["per_type"]:
                continue
            m = point["per_type"][t]
            bw_ci = ci.get(f"delta_conf_bothwrong:{t}", {})
            au = (f"{m.get('auroc_M0', float('nan')):.2f}→{m.get('auroc_M1', float('nan')):.2f}"
                  if m.get("auroc_M0") is not None else "-")
            tok = (f"{m['n_ans_tok_M0']:.0f}/{m['n_ans_tok_M1']:.0f}"
                   if m.get("n_ans_tok_M0") is not None else "-")
            print(f"{t:<20}{m['delta_em']:>7.3f}{m['delta_conf']:>7.3f}"
                  f"{m['delta_gap']:>7.3f}{m['delta_conf_bothwrong']:>13.3f}"
                  f"{m['n_bothwrong']:>5}{m.get('delta_aece', float('nan')):>8.3f}"
                  f"{au:>13}{tok:>7}")
        bw = ci.get("delta_conf_bothwrong", {})
        print(f"오답 조건부 Δconf(전체): {point.get('delta_conf_bothwrong', float('nan')):.3f} "
              f"[{bw.get('lo', float('nan')):.3f}, {bw.get('hi', float('nan')):.3f}] "
              f"(n={point.get('n_bothwrong', 0)}) — CI 하한 > 0 이면 정확도와 무관한 팽창 증거")
        ent_ok = any(point["per_type"][t].get("entropy_M0") is not None
                     for t in types if t in point["per_type"])
        if ent_ok:
            print("분포 통계(answer 구간 평균): "
                  + " · ".join(
                      f"{t}: H {point['per_type'][t]['entropy_M0']:.2f}→"
                      f"{point['per_type'][t]['entropy_M1']:.2f}, "
                      f"margin {point['per_type'][t]['margin_M0']:.2f}→"
                      f"{point['per_type'][t]['margin_M1']:.2f}"
                      for t in types
                      if t in point["per_type"]
                      and point["per_type"][t].get("entropy_M0") is not None))
        else:
            print("분포 통계 없음 (구버전 예측 파일 — 패치된 run_inference.py로 추론 시 자동 포함)")
        nf, mp = sens["contrast_no_fallback"], sens["contrast_minprob"]
        nf_s = f"{nf['value']:.3f} (n={nf['n_rows']})" if nf["value"] is not None else "산출 불가"
        mp_s = f"{mp['value']:.3f}" if mp["value"] is not None else "산출 불가 (min_token_prob 없음)"
        print(f"측정 민감도 — contrast(폴백 제외): {nf_s} · contrast(min_token_prob): {mp_s}"
              f"  ← 본판정 contrast와 부호 일치 여부 확인")

    # ---- 시드 집계 (설계서 §7: 3/3 강건, 2/3 조건부)
    n_seeds = len(all_judges)
    summary = {}
    print(f"\n======== 시드 집계 ({n_seeds}개) ========")
    for h in ("H1", "H2", "H3", "H4a", "H4b"):
        n_pass = sum(1 for j in all_judges.values() if j[h])
        if n_pass == n_seeds and n_seeds >= 3:
            verdict = "강건 채택"
        elif n_seeds >= 3 and n_pass >= n_seeds - 1 and n_pass >= 2:
            verdict = "조건부 채택"
        elif n_seeds < 3:
            verdict = f"{n_pass}/{n_seeds} 충족 (시드 3개 미만 — 등급화 불가)"
        else:
            verdict = "미충족"
        summary[h] = {"pass": n_pass, "total": n_seeds, "verdict": verdict}
        print(f"{h}: {n_pass}/{n_seeds} → {verdict}")

    out = {"per_seed": all_results, "seed_summary": summary,
           "criteria": {"min_contrast": min_contrast, "n_bootstrap": n_boot,
                        "contrast_def": "mean ΔECE(table_lookup, numeric_reasoning) − ΔECE(text_span)"}}
    out_path = res_dir / "metrics_main.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
