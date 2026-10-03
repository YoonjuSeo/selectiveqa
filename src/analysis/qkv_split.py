import sys; sys.path.insert(0, "src/analysis")
from lora_cosine import load, modules, inner
r = "results/exaone_outproj/"
m1 = modules(load(r + "qlora_adapter_pin_s42/adapter_model.safetensors"))
m2 = modules(load(r + "qlora_adapter_r05_pin_s42/adapter_model.safetensors"))
for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
    cs = []
    for n in m1:
        if n.endswith(proj):
            (a1, b1), (a2, b2) = m1[n], m2[n]
            cs.append(inner(a1, b1, a2, b2) / (inner(a1, b1, a1, b1) * inner(a2, b2, a2, b2)) ** 0.5)
    print(f"{proj:9s} {sum(cs)/len(cs):.4f}  (n={len(cs)})")
