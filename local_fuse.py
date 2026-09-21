#!/usr/bin/env python3
"""纯本地免费栈：把 Laya 拉进来到底能给 open-system-one 加多少。

只比本地方案，Jev 完全不参与——它是付费对手，不是我们会 ship 的组件。

对比的两条本地零样本链路：
  A. 交叉编码器 149M (NLI, 裸标签)   —— 我们现在的零样本主力，macro 78.7%
  B. Laya 421M                        —— macro 73.2%，但 ag_news 上强 9.5pp

三种合并方式，全部用 dev/eval 对半分，参数只在 dev 上选：
  1. 概率融合（加权平均）    —— 不需要知道谁更强
  2. 置信度路由              —— 已知置信度信号是坏的，作为对照
  3. oracle 上界             —— 互补性的物理上限
"""
from __future__ import annotations
import json, glob
import numpy as np
from pathlib import Path

DS = ["sst2", "ag_news", "emotion", "banking77"]
OUT = Path("out")

laya = {r["i"]: r for r in map(json.loads, (OUT/"laya_preds_english.jsonl").read_text().splitlines())
        if r.get("probs")}

def softmax(lg):
    lg = lg - np.nanmax(np.where(np.isfinite(lg), lg, -np.inf), 1, keepdims=True)
    e = np.exp(np.where(np.isfinite(lg), lg, -np.inf))
    return e / e.sum(1, keepdims=True)

def temp(p, t):
    with np.errstate(divide="ignore"):
        lg = np.log(np.clip(p, 1e-12, None)) / t
    return softmax(lg)

data = {}
for d in DS:
    z = np.load(glob.glob(f"cache/cross_{d}_k*.npz")[0])
    idx, y = z["idx"], z["y"]
    pc = softmax(z["logits"].astype(np.float64))
    keep = [k for k, i in enumerate(idx) if int(i) in laya]
    idx, y, pc = idx[keep], y[keep], pc[keep]
    pl = np.zeros_like(pc)
    for r, i in enumerate(idx):
        q = laya[int(i)]["probs"]
        pl[r, :len(q)] = q
    pl = pl / np.clip(pl.sum(1, keepdims=True), 1e-12, None)
    dev = (idx % 2 == 0)
    data[d] = dict(y=y, pc=pc, pl=pl, dev=dev, evl=~dev)
    print(f"{d:10s} n={len(y)}  dev={dev.sum()}  eval={(~dev).sum()}")

def ok_of(p, y):
    """p 和 y 都已经切好片，长度一致。"""
    return p.argmax(1) == y

def per_split(fn, split):
    """fn(pc, pl, y) -> 正确与否的布尔数组；返回四任务 macro。"""
    out = []
    for d in DS:
        D = data[d]; m = D[split]
        out.append(fn(D["pc_t"][m], D["pl_t"][m], D["y"][m]).mean())
    return float(np.mean(out))

# --- 全局温度：在 dev 上按 NLL 选，两条链路各一个 ---
def fit_t(key):
    grid = np.geomspace(0.2, 20, 70); best = (1e9, 1.0)
    for t in grid:
        tot = n = 0
        for d in DS:
            D = data[d]; m = D["dev"]
            p = temp(D[key][m], t); y = D["y"][m]
            tot += -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).sum(); n += len(y)
        best = min(best, (tot/n, float(t)))
    return best[1]

tc, tl = fit_t("pc"), fit_t("pl")
print(f"\n全局温度：交叉编码器 {tc:.2f}   Laya {tl:.2f}")
for d in DS:
    data[d]["pc_t"] = temp(data[d]["pc"], tc)
    data[d]["pl_t"] = temp(data[d]["pl"], tl)

base_c = per_split(lambda pc, pl, y: ok_of(pc, y), "evl")
base_l = per_split(lambda pc, pl, y: ok_of(pl, y), "evl")
orc    = per_split(lambda pc, pl, y: ok_of(pc, y) | ok_of(pl, y), "evl")

# --- 1. 概率融合：权重在 dev 上选 ---
fuse_fn = lambda w: (lambda pc, pl, y: ok_of(w*pl + (1-w)*pc, y))
ws = np.arange(0, 1.0001, 0.05)
w = float(ws[int(np.argmax([per_split(fuse_fn(w_), "dev") for w_ in ws]))])
fuse = per_split(fuse_fn(w), "evl")

# --- 2. 置信度路由（对照）---
route_fn = lambda th: (lambda pc, pl, y: np.where(pl.max(1) >= th, ok_of(pl, y), ok_of(pc, y)))
ths = np.arange(0.3, 1.001, 0.01)
th = float(ths[int(np.argmax([per_split(route_fn(t), "dev") for t in ths]))])
rt = per_split(route_fn(th), "evl")

print(f"\n=== eval 折（只看一次），纯本地 macro ===")
print(f"  交叉编码器 149M 单打      {base_c*100:.2f}%")
print(f"  Laya 421M 单打            {base_l*100:.2f}%")
print(f"  概率融合 (dev 选 w={w:.2f})  {fuse*100:.2f}%   ← 对比现状 {(fuse-base_c)*100:+.2f} pp")
print(f"  置信度路由 (dev 选 {th:.2f})  {rt*100:.2f}%   ← 对比现状 {(rt-base_c)*100:+.2f} pp")
print(f"  oracle 上界               {orc*100:.2f}%")

print(f"\n=== 分任务（eval 折）===")
print(f"{'任务':10s} {'交叉编码器':>10s} {'Laya':>8s} {'融合':>8s} {'该用谁':>10s}")
per = {}
for d in DS:
    D = data[d]; m = D["evl"]
    pc, pl, y = D["pc_t"][m], D["pl_t"][m], D["y"][m]
    a = ok_of(pc, y).mean(); b = ok_of(pl, y).mean()
    f = ok_of(w*pl + (1-w)*pc, y).mean()
    per[d] = dict(cross=a, laya=b, fuse=f)
    print(f"{d:10s} {a*100:9.1f}% {b*100:7.1f}% {f*100:7.1f}% {'融合' if f>=max(a,b) else ('Laya' if b>a else '交叉编码器'):>10s}")

json.dump({"weight": w, "threshold": th, "temps": {"cross": tc, "laya": tl},
           "eval_macro": {"cross": base_c, "laya": base_l, "fuse": fuse,
                          "route": rt, "oracle": orc},
           "per_dataset_eval": per},
          open(OUT/"local_fuse.json", "w"), indent=1)

# --- 3. 按任务选权重：在"有 dev 标注"的场景下完全合法 ---
print(f"\n=== 按任务在 dev 上各选各的权重（eval 只看一次）===")
print(f"{'任务':10s} {'dev选中w':>9s} {'交叉编码器':>10s} {'Laya':>8s} {'全局融合':>9s} {'按任务融合':>11s}")
gain = []
per_w = {}
for d in DS:
    D = data[d]
    dv, ev = D["dev"], D["evl"]
    pcd, pld, yd = D["pc_t"][dv], D["pl_t"][dv], D["y"][dv]
    pce, ple, ye = D["pc_t"][ev], D["pl_t"][ev], D["y"][ev]
    wd = float(ws[int(np.argmax([ok_of(x*pld + (1-x)*pcd, yd).mean() for x in ws]))])
    per_w[d] = wd
    a = ok_of(pce, ye).mean(); b = ok_of(ple, ye).mean()
    g = ok_of(w*ple + (1-w)*pce, ye).mean()
    t_ = ok_of(wd*ple + (1-wd)*pce, ye).mean()
    gain.append(t_)
    print(f"{d:10s} {wd:9.2f} {a*100:9.1f}% {b*100:7.1f}% {g*100:8.1f}% {t_*100:10.1f}%")
pt = float(np.mean(gain))
print(f"\n  按任务融合 macro = {pt*100:.2f}%   ← 对比交叉编码器单打 {(pt-base_c)*100:+.2f} pp")
print(f"  （全局权重只有 {(fuse-base_c)*100:+.2f} pp，oracle 上界 {(orc-base_c)*100:+.2f} pp）")

j = json.load(open(OUT/"local_fuse.json"))
j["per_task_weights"] = per_w
j["eval_macro"]["fuse_per_task"] = pt
json.dump(j, open(OUT/"local_fuse.json", "w"), indent=1)
