#!/usr/bin/env python3
"""把 out/usecases.json 变成可以直接贴进 README 的表格（中英双语）。"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

R = json.load(open("out/usecases.json"))
LANG = sys.argv[1] if len(sys.argv) > 1 else "zh"
ORDER = ["cross", "laya", "fused", "jev"]
NAME = {"zh": {"cross": "交叉编码器 149M", "laya": "Laya 421M", "fused": "**融合（本地）**",
               "jev": "`typesafe/jev`（云端付费）"},
        "en": {"cross": "Cross-encoder 149M", "laya": "Laya 421M", "fused": "**Fused (local)**",
               "jev": "`typesafe/jev` (paid, cloud)"}}[LANG]
H = {"zh": ("用例", "原语", "选项数", "多数类基线", "条数"),
     "en": ("Use case", "Primitive", "Options", "Majority", "n")}[LANG]

stacks = [s for s in ORDER if any(s in r["stacks"] for r in R.values())]

print(f"| {H[0]} | {H[1]} | {H[2]} | {H[3]} | " + " | ".join(NAME[s] for s in stacks) + " |")
print("|" + "---|" * (4 + len(stacks)))
for k, r in R.items():
    row = [r["jev_usecase"] if LANG == "zh" else k, r["primitive"], str(r["n_options"]),
           f"{r['majority_baseline']*100:.1f} %"]
    accs = {s: r["stacks"][s]["acc"] for s in stacks if s in r["stacks"]}
    best = max(accs.values()) if accs else 0
    for s in stacks:
        if s not in r["stacks"]: row.append("—"); continue
        a = r["stacks"][s]["acc"]
        cell = f"{a*100:.1f} %"
        row.append(f"**{cell}**" if a == best else cell)
    print("| " + " | ".join(row) + " |")

macro = {s: np.mean([r["stacks"][s]["acc"] for r in R.values() if s in r["stacks"]])
         for s in stacks}
mb = np.mean([r["majority_baseline"] for r in R.values()])
row = ["**macro**", "", "", f"**{mb*100:.1f} %**"]
for s in stacks:
    row.append(f"**{macro[s]*100:.1f} %**" if macro[s] == max(macro.values()) else f"{macro[s]*100:.1f} %")
print("| " + " | ".join(row) + " |")

lat = {"zh": "\n**延迟 p50 (ms，batch=1)**", "en": "\n**p50 latency (ms, batch=1)**"}[LANG]
print(lat)
print(f"| {H[0]} | " + " | ".join(NAME[s] for s in stacks) + " |")
print("|" + "---|" * (1 + len(stacks)))
for k, r in R.items():
    row = [r["jev_usecase"] if LANG == "zh" else k]
    for s in stacks:
        row.append(f"{r['stacks'][s]['lat_p50_ms']:.0f}" if s in r["stacks"] else "—")
    print("| " + " | ".join(row) + " |")

fails = {s: sum(r["stacks"][s]["fails"] for r in R.values() if s in r["stacks"]) for s in stacks}
print(("\n失败（解析不出合法选项，按答错计）：" if LANG == "zh" else
       "\nFailures (no valid option parsed, counted as wrong): ")
      + ", ".join(f"{NAME[s]} {fails[s]}" for s in stacks))

w = {k: r.get("fused_calibration", {}).get("weights", {}) for k, r in R.items()}
if any(w.values()):
    print(("\n融合权重（100 条校准样本上自动学出）：" if LANG == "zh" else
           "\nFusion weights (learned from 100 calibration labels): "))
    for k, v in w.items():
        if v: print(f"  {k}: " + ", ".join(f"{a}={b:.2f}" for a, b in v.items()))
