#!/usr/bin/env python3
"""按任务融合需要多少标注？只是选一个标量权重，应该很便宜。

对每个标注预算 n，从 dev 里随机抽 n 条选权重，在 eval 上评分，重复 20 次看均值和方差。
"""
from __future__ import annotations
import json, glob
import numpy as np
from pathlib import Path
exec(open("local_fuse.py").read().split("base_c = per_split")[0])   # 复用数据加载与温度标定

ws = np.arange(0, 1.0001, 0.05)
rng = np.random.default_rng(20260921)
BUDGETS = [10, 25, 50, 100, 200, 500, None]      # None = 用全部 dev

print(f"\n{'标注/任务':>9s} {'macro':>8s} {'±std':>7s} {'对比交叉编码器':>14s}")
base = np.mean([ok_of(data[d]["pc_t"][data[d]["evl"]], data[d]["y"][data[d]["evl"]]).mean() for d in DS])
rows = {}
for nb in BUDGETS:
    trials = []
    for _ in range(20 if nb else 1):
        accs = []
        for d in DS:
            D = data[d]; dv, ev = D["dev"], D["evl"]
            pcd, pld, yd = D["pc_t"][dv], D["pl_t"][dv], D["y"][dv]
            if nb:
                sel = rng.choice(len(yd), size=min(nb, len(yd)), replace=False)
                pcd, pld, yd = pcd[sel], pld[sel], yd[sel]
            wd = ws[int(np.argmax([ok_of(x*pld + (1-x)*pcd, yd).mean() for x in ws]))]
            accs.append(ok_of(wd*D["pl_t"][ev] + (1-wd)*D["pc_t"][ev], D["y"][ev]).mean())
        trials.append(np.mean(accs))
    m, s = float(np.mean(trials)), float(np.std(trials))
    rows[str(nb)] = {"macro": m, "std": s}
    print(f"{str(nb or '全部'):>9s} {m*100:7.2f}% {s*100:6.2f} {(m-base)*100:+13.2f} pp")

json.dump({"base_cross": float(base), "budgets": rows}, open("out/weight_budget.json","w"), indent=1)
