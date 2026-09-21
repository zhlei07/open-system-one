#!/usr/bin/env python3
"""端到端验证 Ensemble 组件能复现离线实验的结论。

离线结论（用缓存 logits 算的）：
  ag_news  交叉编码器 83.5% → 融合 92.9%，dev 选中权重 0.85（Laya 强，该重用）
  emotion  交叉编码器 75.9% → 融合 75.7%，dev 选中权重 0.05（Laya 弱，该关掉）

组件如果是对的，应该自己学出这两个方向相反的权重。
"""
from __future__ import annotations
import json, sys, time
import numpy as np
from data import build
from ensemble import Ensemble

DS = sys.argv[1] if len(sys.argv) > 1 else "ag_news"
N_CAL, N_EVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 100, 200

b = build(); ev = b["splits"]["eval"]; opts = b["options"][DS]
rows = [r for r in ev if r["dataset"] == DS]
cal, tst = rows[:N_CAL], rows[N_CAL:N_CAL + N_EVAL]
print(f"{DS}: {len(opts)} 选项，校准 {len(cal)} 条，评测 {len(tst)} 条")

def run(e, tag):
    t0 = time.perf_counter()
    ok = sum(e.decide(r["text"], opts).choice == opts[r["label"]] for r in tst)
    dt = (time.perf_counter() - t0) / len(tst) * 1000
    print(f"  {tag:26s} {ok/len(tst)*100:5.1f}%   {dt:6.1f} ms/条")
    return ok / len(tst)

solo = Ensemble(backends=("cross",))
a = run(solo, "交叉编码器单打")

duo = Ensemble(backends=("cross", "laya"))
print(f"  (laya 设备 = {duo.backends['laya'].device})")
info = duo.calibrate([(r["text"], opts, r["label"]) for r in cal])
c = run(duo, "融合（组件自己学的权重）")

json.dump({"dataset": DS, "n_cal": len(cal), "n_eval": len(tst),
           "cross_only": a, "fused": c, "gain_pp": (c - a) * 100,
           "learned": info}, open(f"out/verify_{DS}.json", "w"), ensure_ascii=False, indent=1)
print(f"\n  增益 {(c-a)*100:+.1f} pp   学到的权重 {info['weights']}")
