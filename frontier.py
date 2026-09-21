#!/usr/bin/env python3
"""汇总今天所有实验 -> 成本/准确率前沿，找最优方案。

延迟取四个任务的等权平均（每任务 2500 条，权重相同），
这样"选项数越多越贵"这件事才会体现在一个可比的数字里。
"""
import json, glob
from pathlib import Path
import numpy as np

DS = ["sst2", "ag_news", "emotion", "banking77"]
OUT = Path("out")
L = lambda p: json.loads((OUT / p).read_text()) if (OUT / p).exists() else None

heads = L("heads.json") or {}
qwen, qp = L("qwen.json"), L("qwen_probe.json")
jev, jl = L("jev.json"), L("jev_latency.json")
dist = L("distill.json")
meters = {e: L(f"meter_{e}.json") for e in ["static", "minilm", "modernbert", "mbert-embed"]}

def cross(model):
    r = {}
    for ds in DS:
        pat = f"out/cross_{ds}_k*.json" if model == "base" else f"out/xc_{model}_{ds}_k*_t1_n*.json"
        hits = glob.glob(pat)
        if model == "base" and not hits:
            hits = glob.glob(f"out/xc_base_{ds}_k*_t1_n*.json")
        if hits:
            d = json.load(open(max(hits, key=lambda h: json.load(open(h)).get("n", 0))))
            r[ds] = (d.get("acc_argmax", d.get("acc_mean_template")), d["lat_p50_ms"], d.get("n"))
    return r

rows = []
def add(name, accs, lats, ram, note=""):
    have = [d for d in DS if accs.get(d) is not None]
    if len(have) < 4:
        note = (note + f" [仅 {len(have)}/4 任务]").strip()
    rows.append({"name": name, "acc": np.mean([accs[d] for d in have]),
                 "lat": np.mean([lats[d] for d in have]) if lats else None,
                 "ram": ram, "note": note,
                 "per": {d: accs.get(d) for d in DS}})

# 双塔 + Fly
for e, m in meters.items():
    if not m or e not in heads:
        continue
    h = heads[e]
    for mode, tag in [("fly_zero", "零样本"), ("fly_few", "few-shot 2000 标注")]:
        lat = m["lat_p50_ms"] + h["head_latency"]["zero" if mode == "fly_zero" else "few"]["p50_ms"]
        add(f"{e} + Fly（{tag}）",
            {d: h["per_dataset"][d][mode] for d in DS},
            {d: lat for d in DS}, m["model_rss_mb"])

# 交叉编码
for model in ["base", "large"]:
    c = cross(model)
    if c:
        add(f"交叉编码 {model}（零样本）",
            {d: c[d][0] for d in c}, {d: c[d][1] for d in c}, None,
            "banking77 用双塔裁 top-20")

# 蒸馏
if dist:
    m = meters["mbert-embed"]; h = heads["mbert-embed"]
    add("交叉编码蒸馏 -> 双塔 Fly（零人工标注）",
        {d: dist[d]["distilled_self"] for d in dist},
        {d: m["lat_p50_ms"] + h["head_latency"]["few"]["p50_ms"] for d in DS},
        m["model_rss_mb"], "线下付一次交叉编码的钱")

if qwen:
    add("Qwen2.5-0.5B 生成", {d: qwen["per_dataset"][d]["acc"] for d in DS},
        {d: qp["lat_p50_ms"] for d in DS}, qp["model_rss_mb"])
if jev:
    add("typesafe/jev（云端）", {d: jev["per_dataset"][d]["acc"] for d in DS},
        {d: jl["lat_p50_ms"] for d in DS}, None, "延迟含网络往返")

rows.sort(key=lambda r: -r["acc"])
print(f"{'方案':42s} {'宏平均':>7s} {'平均延迟':>9s} {'RAM':>8s}  备注")
print("-" * 100)
for r in rows:
    lat = f"{r['lat']:.1f}ms" if r["lat"] else "—"
    ram = f"{r['ram']:.0f}MB" if r["ram"] else "云端"
    print(f"{r['name']:42s} {r['acc']*100:6.1f}% {lat:>9s} {ram:>8s}  {r['note']}")
json.dump(rows, open(OUT / "frontier.json", "w"), ensure_ascii=False, indent=1, default=float)
