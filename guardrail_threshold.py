#!/usr/bin/env python3
"""Agent 护栏的正确对标方式：置信度门控，不是二分类。

我第一版把"域外拦截"做成了二选一（"能处理" vs "超出范围"），两个模型都接近或低于
多数类基线。那是设计错了——这两个选项本身没有可判别的语义，模型从一句普通用户请求里
读不出"这个助手的能力边界在哪"。

Jev 被宣传的用法是 confidence-gated routing：在**所有域内意图**上打分，
如果最高分也很低，就判域外。这才是在考校准过的概率，而不是考分类。

所以这里：150 个域内意图上打分 -> 取 max prob -> 阈值判域外。
主指标用 AUC（不依赖阈值），再报按校准折选出的阈值下的准确率。
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from pathlib import Path

from usecases import build_usecases
from ensemble import Ensemble, _BACKENDS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
STACKS = (sys.argv[2] if len(sys.argv) > 2 else "cross,laya").split(",")
N_CAL = 100

uc = build_usecases(max(N, 500))
routing, guard = uc["routing"], uc["guardrail"]
OPTS = routing.options                      # 150 个域内意图
texts, is_oos = guard.texts[:N], np.array(guard.gold[:N])
print(f"{len(texts)} 条（域内 {(is_oos==0).sum()} / 域外 {(is_oos==1).sum()}），"
      f"在 {len(OPTS)} 个域内意图上打分", flush=True)

def auc(score, pos):
    """score 越大越可能是域外。pos = 真域外。"""
    a, b = score[pos == 1], score[pos == 0]
    if not len(a) or not len(b): return float("nan")
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())

res = {}
for name in STACKS:
    e = Ensemble(backends={name: _BACKENDS[name]()})
    mx, lat = [], []
    t0 = time.perf_counter()
    for i, t in enumerate(texts):
        t1 = time.perf_counter()
        mx.append(float(e.decide(t, OPTS).probs.max()))
        lat.append((time.perf_counter() - t1) * 1000)
        if i and i % 100 == 0:
            print(f"  {name} {i}/{len(texts)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    mx = np.array(mx)
    oos_score = -mx                                    # 最高分越低越像域外
    a = auc(oos_score, is_oos)
    # 阈值在校准折上选，评测折只看一次
    cal, evl = slice(0, N_CAL), slice(N_CAL, None)
    grid = np.unique(np.quantile(mx[cal], np.linspace(0.02, 0.98, 60)))
    accs = [((mx[cal] < th).astype(int) == is_oos[cal]).mean() for th in grid]
    th = float(grid[int(np.argmax(accs))])
    acc = float(((mx[evl] < th).astype(int) == is_oos[evl]).mean())
    maj = float(max((is_oos[evl] == 0).mean(), (is_oos[evl] == 1).mean()))
    res[name] = {"auc": a, "threshold": th, "acc": acc, "majority": maj,
                 "lat_p50_ms": round(float(np.percentile(lat, 50)), 1),
                 "mean_maxprob_in": float(mx[is_oos == 0].mean()),
                 "mean_maxprob_oos": float(mx[is_oos == 1].mean())}
    print(f"  {name:6s} AUC={a:.3f}  阈值={th:.3f}  准确率={acc*100:.1f}% "
          f"(基线 {maj*100:.1f}%)  域内均值 {res[name]['mean_maxprob_in']:.3f} "
          f"vs 域外 {res[name]['mean_maxprob_oos']:.3f}", flush=True)

Path("out/guardrail_threshold.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
