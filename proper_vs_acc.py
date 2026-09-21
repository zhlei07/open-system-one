#!/usr/bin/env python3
"""对照实验：ES 目标用"准确率" vs 用"严格恰当评分规则"。

问题：我们一整天都在报概率和 ECE，但 ES 的目标函数是
      0.7*均值准确率 + 0.3*最差准确率 —— 对校准零奖励。
      也就是说那些概率从来没有被优化过。

这次把目标换成对数分（温度就地最优化），测两件事：
  1. 校准（ECE）改善多少
  2. argmax 准确率付出多少代价

纪律不变：fit 前 1500 建质心 -> fit 后 500 当 ES 目标 -> dev 标最终温度 -> eval 只碰一次。
"""
from __future__ import annotations
import json
import numpy as np
import fly_head as fh
from data import CACHE, build

DS = ["sst2", "ag_news", "emotion", "banking77"]
ENC = "mbert-embed"
N_PROTO = 1500

b = build(); z = np.load(CACHE / f"emb/{ENC}.npz")
ev = b["splits"]["eval"]
ev_idx = {d: [i for i, r in enumerate(ev) if r["dataset"] == d] for d in DS}
ev_y = np.array([r["label"] for r in ev])
fit_y = {d: np.array([r["label"] for r in b["splits"]["fit"] if r["dataset"] == d]) for d in DS}
dev_y = {d: np.array([r["label"] for r in b["splits"]["dev"] if r["dataset"] == d]) for d in DS}

banks = {d: fh.FeatureBank(b["options"][d], z[f"fit_{d}"][:N_PROTO], fit_y[d][:N_PROTO],
                           z[f"desc_{d}"]) for d in DS}
es_set = {d: (banks[d].few(z[f"fit_{d}"][N_PROTO:]), fit_y[d][N_PROTO:]) for d in DS}

results = {}
for mode in ["acc", "proper"]:
    w, _ = fh.search_weights(es_set, len(fh.FEW_FEATURES), objective_mode=mode, generations=80)
    per = {}
    for d in DS:
        q, y = z["eval"][ev_idx[d]], ev_y[ev_idx[d]]
        f_eval = banks[d].few(q)
        # 最终温度仍在 dev 上拟合（不能用 ES 集，否则等于把它变成训练集）
        t, _ = fh.best_temperature(fh.scores(banks[d].few(z[f"dev_{d}"]), w), dev_y[d], fine=True)
        p = fh.probabilities(f_eval, w, t)
        per[d] = {"acc": fh.accuracy(f_eval, w, y), "ece": fh.ece(p, y), "temp": round(t, 2),
                  "nll": round(fh._nll(fh.scores(f_eval, w), y, t), 4)}
    results[mode] = {"per": per,
                     "macro_acc": float(np.mean([per[d]["acc"] for d in DS])),
                     "mean_ece": float(np.mean([per[d]["ece"] for d in DS])),
                     "mean_nll": float(np.mean([per[d]["nll"] for d in DS])),
                     "weights": {n: round(float(v), 4) for n, v in zip(fh.FEW_FEATURES, w)}}
    print(f"[{mode:6s}] 宏平均准确率 {results[mode]['macro_acc']*100:.1f}%  "
          f"平均 ECE {results[mode]['mean_ece']:.3f}  平均 NLL {results[mode]['mean_nll']:.3f}", flush=True)

print(f"\n{'任务':11s} {'acc目标: 准确率/ECE':>24s}   {'proper目标: 准确率/ECE':>24s}")
for d in DS:
    a, p = results["acc"]["per"][d], results["proper"]["per"][d]
    print(f"{d:11s} {a['acc']*100:10.1f}% / {a['ece']:.3f}      "
          f"{p['acc']*100:10.1f}% / {p['ece']:.3f}")
json.dump(results, open("out/proper_vs_acc.json", "w"), ensure_ascii=False, indent=1)
