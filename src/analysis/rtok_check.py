import json
def stats(p):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    ans = [r for r in rows if not r["question_id"].startswith("unans_") and "-hd-" not in r["question_id"]]
    f = lambda g: sum(r.get("n_gen_tokens", 0) for r in g) / len(g)
    return f(rows), f(ans), sum(r.get("answerable_pred") is False for r in ans) / len(ans)
d = "results/exaone_outproj/"
for s in (42, 43, 44):
    a1, b1, c1 = stats(d + f"preds_M1_pin_s{s}.jsonl")
    a2, b2, c2 = stats(d + f"preds_M2_r05_pin_s{s}.jsonl")
    print(f"s{s}  Answerable 토큰 {b1:.1f}->{b2:.1f} (x{b2/b1:.2f}) | 전체 {a1:.1f}->{a2:.1f} (x{a2/a1:.2f}) | 오무응답 {c1:.1%}->{c2:.1%}")
