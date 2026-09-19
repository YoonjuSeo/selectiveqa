import json, glob, os
rows = []
for tag in ['r01', 'r05', 'r05_filled', 'r10', 'r30']:
    p = f'results/metrics_followup_{tag}.json'
    if not os.path.exists(p):
        continue
    d = json.load(open(p, encoding='utf-8'))['per_seed']
    for s in sorted(d):
        pt = d[s]['point']
        rows.append((tag, s, pt['d_em'], 1 - pt['hall_easy_M2'], 1 - pt['hall_hard_M2']))
print('cond        seed   dEM     easyUA  hardUA')
for tag, s, de, ea, ha in rows:
    print(f'{tag:11s} {s}   {de:+.3f}   {ea:.3f}   {ha:.3f}')
print()
import statistics as st
for tag in dict.fromkeys(r[0] for r in rows):
    v = [r[2] for r in rows if r[0] == tag]
    print(f'{tag:11s} mean dEM {st.mean(v):+.3f}  sd {st.pstdev(v):.3f}')
