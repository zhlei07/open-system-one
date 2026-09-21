#!/usr/bin/env python3
"""Jev ↔ Laya 逐条路由：取长补短到底有没有，有多少。

协议（防止在测试集上选参数）：
  - 每个任务的 2500 条按 i 的奇偶对半分：dev 定温度和阈值，eval 只看一次。
  - 温度**全局**标定（不按任务），因为路由器的卖点是"对没见过的任务也生效"，
    按任务标定等于偷偷告诉它这是什么任务。
  - 三条基线都报：单模型、oracle 上界、手写按任务规则（我之前那个不诚实的做法）。
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

OUT = Path("out")

def load(fn):
    rows = {}
    for line in (OUT / fn).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("probs"):
            rows[r["i"]] = r
    return rows

jev = load("jev_preds_enriched.jsonl")
laya = load("laya_preds_english.jsonl")
common = sorted(set(jev) & set(laya))
print(f"对齐 {len(common)} 条")

ds   = np.array([jev[i]["dataset"] for i in common])
gold = np.array([jev[i]["gold"] for i in common])

def mat(src):
    """把每条的 probs 补齐成等宽矩阵（按任务分别处理，选项数不同）。"""
    out = {}
    for d in np.unique(ds):
        idx = np.where(ds == d)[0]
        w = max(len(src[common[k]]["probs"]) for k in idx)
        m = np.zeros((len(idx), w), np.float64)
        for r, k in enumerate(idx):
            p = src[common[k]]["probs"]
            m[r, :len(p)] = p
        out[d] = (idx, m)
    return out

JM, LM = mat(jev), mat(laya)

def temp_apply(p, t):
    """概率 -> log -> 除温度 -> 重新归一。t>1 变钝，t<1 变尖。"""
    with np.errstate(divide="ignore"):
        lg = np.log(np.clip(p, 1e-9, None)) / t
    lg -= lg.max(1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(1, keepdims=True)

dev = np.array([i % 2 == 0 for i in common])          # 偶数 i = dev
evl = ~dev

def nll_global(M, t, mask):
    tot, n = 0.0, 0
    for d, (idx, m) in M.items():
        sel = mask[idx]
        if not sel.any():
            continue
        p = temp_apply(m[sel], t)
        g = gold[idx][sel]
        tot += -np.log(np.clip(p[np.arange(len(g)), g], 1e-9, None)).sum()
        n += len(g)
    return tot / n

grid = np.geomspace(0.3, 20, 60)
t_jev  = float(grid[np.argmin([nll_global(JM, t, dev) for t in grid])])
t_laya = float(grid[np.argmin([nll_global(LM, t, dev) for t in grid])])
print(f"全局温度：jev={t_jev:.2f}  laya={t_laya:.2f}")

# 拼回逐条的 (预测, 置信度)
N = len(common)
def flat(M, t):
    pred = np.zeros(N, int); conf = np.zeros(N)
    for d, (idx, m) in M.items():
        p = temp_apply(m, t)
        pred[idx] = p.argmax(1); conf[idx] = p.max(1)
    return pred, conf

pj, cj = flat(JM, t_jev)
pl, cl = flat(LM, t_laya)
okj, okl = (pj == gold), (pl == gold)

def acc(mask, ok): return ok[mask].mean()
def macro(mask, ok):
    return np.mean([ok[mask & (ds == d)].mean() for d in np.unique(ds)])

print("\n=== eval 折（只看一次）===")
print(f"  Jev 单打        macro {macro(evl, okj)*100:.2f}%")
print(f"  Laya 单打       macro {macro(evl, okl)*100:.2f}%")
oracle = okj | okl
print(f"  oracle 上界     macro {macro(evl, oracle)*100:.2f}%   ← 两个里只要有一个对就算对")

# --- 路由器：Laya 够自信就用 Laya，否则退回 Jev ---
best = max(((macro(dev, np.where(cl >= th, okl, okj)), th) for th in np.arange(0.30, 1.001, 0.01)))
th = best[1]
routed = np.where(cl >= th, okl, okj)
share = (cl[evl] >= th).mean()
print(f"\n  置信度路由 (dev 选出阈值 {th:.2f})")
print(f"     eval macro   {macro(evl, routed)*100:.2f}%   Laya 接手 {share*100:.1f}% 的条目")
print(f"     dev  macro   {best[0]*100:.2f}%  （选阈值用的那折，仅供对照）")

# --- 我之前那个不诚实的手写按任务规则 ---
hand = np.where(np.isin(ds, ["sst2", "ag_news"]), okl, okj)
print(f"\n  手写按任务规则   eval macro {macro(evl, hand)*100:.2f}%  ← 规则是看了测试分数定的，不可信")

print("\n=== 分任务：路由器把活分给了谁 ===")
print(f"{'任务':10s} {'Jev':>7s} {'Laya':>7s} {'路由':>7s} {'Laya接手':>9s}")
for d in ["sst2", "ag_news", "emotion", "banking77"]:
    m = evl & (ds == d)
    print(f"{d:10s} {acc(m,okj)*100:6.1f}% {acc(m,okl)*100:6.1f}% {acc(m,routed)*100:6.1f}% "
          f"{(cl[m] >= th).mean()*100:8.1f}%")

json.dump({"t_jev": t_jev, "t_laya": t_laya, "threshold": th,
           "eval_macro": {"jev": macro(evl, okj), "laya": macro(evl, okl),
                          "routed": macro(evl, routed), "oracle": macro(evl, oracle),
                          "hand_rule_dishonest": macro(evl, hand)},
           "laya_share_eval": float(share)},
          open(OUT / "router_probe.json", "w"), indent=1)
