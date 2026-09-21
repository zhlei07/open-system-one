#!/usr/bin/env python3
"""后处理与融合：在已存盘的 logit 上榨干不需要再跑前向的那部分增益。

四个层次，代价全部为零（只是对已有矩阵做线性代数）：

1. **raw**      直接 argmax 交叉编码 logit
2. **center**   列中心化：减掉"每个选项在整批查询上的平均分"。
                修的是选项先验偏差——有的假设句天生更容易被判为 entailment，
                它会无差别地吸走样本。注意这是 **transductive**：需要一批查询
                才能估计那个均值，单条请求做不到。所以单独标注，不和逐条方案混报。
3. **prior**    单条可用的替代方案：用 fit 集（**无标签**，只用文本）预先估好
                每个选项的均值，线上直接减。零样本不破，且逐条可用。
4. **fuse**     Fly 头融合：交叉编码 logit + 双塔的 6 个特征 = 8 个特征，
                (1+λ)-ES 留一搜权重。这是这个实验真正的产物——
                把"贵但准"的交叉编码和"便宜但糙"的双塔当成两个特征，让打分函数决定信谁。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import fly_head as fh
from data import CACHE, build

OUT = Path("out")
DATASETS = ["sst2", "ag_news", "emotion", "banking77"]
BI = "mbert-embed"


def load_cross(ds, prefer=None):
    """同时支持两种存盘格式：
       cross_*.npz  (n, k)     —— v1，单模板
       xc_*.npz     (n, k, T)  —— v2，多模板
    多模板的返回 (n, k, T)，单模板补一个长度 1 的模板维，下游统一处理。
    prefer: 文件名里必须包含的子串（用来指定 base/large 或某个 n）。
    """
    hits = sorted(CACHE.glob(f"xc_*_{ds}_k*.npz")) + sorted(CACHE.glob(f"cross_{ds}_k*.npz"))
    if prefer:
        hits = [h for h in hits if prefer in h.name] or hits
    if not hits:
        return None
    # 优先取样本量最大的那一份
    def n_of(h):
        d = np.load(h); return len(d["y"])
    d = np.load(max(hits, key=n_of))
    lg = d["logits"]
    if lg.ndim == 2:
        lg = lg[..., None]
    return lg, d["y"], d["idx"]


def softmax_rows(s):
    m = np.where(np.isfinite(s), s, -np.inf)
    m = m - np.nanmax(np.where(np.isfinite(m), m, -np.inf), axis=1, keepdims=True)
    e = np.where(np.isfinite(m), np.exp(m), 0.0)
    return e / np.maximum(e.sum(1, keepdims=True), 1e-12)


def main():
    b = build()
    z = np.load(CACHE / f"emb/{BI}.npz")
    jev = json.load(open(OUT / "jev.json"))
    ev = b["splits"]["eval"]
    fit_y = {d: np.array([r["label"] for r in b["splits"]["fit"] if r["dataset"] == d])
             for d in DATASETS}

    rows, results = [], {}
    # ---- 先把每个数据集的特征张量备好，供 LODO 搜权重 ----
    feats, ys = {}, {}
    for ds in DATASETS:
        got = load_cross(ds)
        if not got:
            continue
        logits, y, idx = got                               # (n, k, T)
        bank = fh.FeatureBank(b["options"][ds], None, None, z[f"desc_{ds}"])
        bi = bank.zero(z["eval"][idx])                     # (n, k, 6)
        finite = np.isfinite(logits)
        safe = np.where(finite, logits, 0.0)
        # 裁剪掉的候选给一个很低的常数分，而不是 -inf——线性组合里 -inf 会把整行毒掉
        floor = np.where(finite.any(-1), 0.0, -20.0).astype(np.float32)[..., None]
        cnt = np.maximum(finite.sum(0, keepdims=True), 1)
        col = safe - (safe * finite).sum(0, keepdims=True) / cnt     # 列中心化，逐模板
        # 特征 = 每个模板的原始 logit + 每个模板的中心化 logit + 裁剪标记 + 双塔 6 个
        f = np.concatenate([safe / 5.0, col / 5.0, floor, bi], axis=-1).astype(np.float32)
        feats[ds], ys[ds] = f, y
        results[ds] = {}
        # 单模板 = 第 0 个模板；多模板 = 各模板均值（等权集成）
        raw1 = np.where(finite[..., 0], logits[..., 0], -1e9)
        rawT = np.where(finite.any(-1), safe.mean(-1), -1e9)
        colT = np.where(finite.any(-1), col.mean(-1), -1e9)
        results[ds]["raw"] = float((raw1.argmax(1) == y).mean())
        results[ds]["tmpl_mean"] = float((rawT.argmax(1) == y).mean())
        results[ds]["center"] = float((colT.argmax(1) == y).mean())
        results[ds]["n_templates"] = int(logits.shape[-1])
        results[ds]["n"] = int(len(y))

    if not feats:
        print("还没有 cross_*.npz，先跑 cross_full.py")
        return

    dim = next(iter(feats.values())).shape[-1]
    for ds in feats:
        # LODO：权重在**其他**数据集上搜，保证目标任务仍是零样本
        src = {d: (feats[d], ys[d]) for d in feats if d != ds}
        if src:
            w, _ = fh.search_weights(src, dim, generations=60)
            results[ds]["fuse"] = fh.accuracy(feats[ds], w, ys[ds])
        # 作弊上界：本集上搜
        wo, _ = fh.search_weights({ds: (feats[ds], ys[ds])}, dim, generations=60)
        results[ds]["fuse_oracle"] = fh.accuracy(feats[ds], wo, ys[ds])

    names = [d for d in DATASETS if d in results]
    print(f"{'方案':28s} " + " ".join(f"{d:>10s}" for d in names) + "   宏平均")
    for key, label in [("raw", "交叉编码 argmax (单模板)"),
                       ("tmpl_mean", "+ 模板等权集成"),
                       ("center", "+ 列中心化 (transductive)"),
                       ("fuse", "+ Fly 融合双塔 (LODO)"), ("fuse_oracle", "+ Fly 融合 (作弊上界)")]:
        vals = [results[d].get(key) for d in names]
        if any(v is None for v in vals):
            continue
        print(f"{label:28s} " + " ".join(f"{v*100:9.1f}%" for v in vals) +
              f"  {np.mean(vals)*100:6.1f}%")
    jv = [jev["per_dataset"][d]["acc"] for d in names]
    print(f"{'typesafe/jev':28s} " + " ".join(f"{v*100:9.1f}%" for v in jv) +
          f"  {np.mean(jv)*100:6.1f}%")
    json.dump(results, open(OUT / "fuse.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
