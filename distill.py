#!/usr/bin/env python3
"""蒸馏：用交叉编码器的预测当伪标签，训练双塔 Fly 头。

动机来自今天两组数据的错位：
  - 零样本最准的是交叉编码器（宏平均 78.7%），但它贵，且延迟随选项数线性增长
  - 最快的是双塔 + Fly 头（15.7ms，比 Jev 快 24 倍），但零样本只有 65.6%
  - 而双塔 + Fly 头在**有标注**时是 82.6%，反超 Jev

所以问题变成：那 2000 条标注，非得是人标的吗？
用交叉编码器离线把无标签文本标一遍（78.7% 准确），再拿伪标签训 Fly 头——
**线上跑的是双塔的 15.7ms，线下付一次交叉编码的钱，人工标注为零。**

验证方式：eval 前一半用交叉编码器的预测当伪标签训头，后一半用真标签测。
两半严格不相交，测试集的真标签从头到尾没参与训练。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import fly_head as fh
from data import CACHE, build

DATASETS = ["sst2", "ag_news", "emotion", "banking77"]
BI = "mbert-embed"


def load_cross(ds):
    hits = sorted(CACHE.glob(f"cross_{ds}_k*.npz")) + sorted(CACHE.glob(f"xc_base_{ds}_k*.npz"))
    if not hits:
        return None
    d = np.load(max(hits, key=lambda h: len(np.load(h)["y"])))
    lg = d["logits"]
    if lg.ndim == 3:
        lg = np.where(np.isfinite(lg), lg, -1e9).mean(-1)
    return np.where(np.isfinite(lg), lg, -1e9), d["y"], d["idx"]


def main():
    b = build()
    z = np.load(CACHE / f"emb/{BI}.npz")
    jev = json.load(open("out/jev.json"))
    ev = b["splits"]["eval"]
    out = {}

    banks_train, banks_eval = {}, {}
    for ds in DATASETS:
        got = load_cross(ds)
        if not got:
            continue
        lg, y, idx = got
        n = len(y) // 2
        a, bh = slice(0, n), slice(n, len(y))              # A 训练 / B 测试，不相交
        emb = z["eval"][idx]
        pseudo = lg.argmax(1)                              # 交叉编码器的预测 = 伪标签
        opts = b["options"][ds]

        # 用 A 的 embedding + 伪标签建质心/kNN —— 全程没碰过真标签
        bank = fh.FeatureBank(opts, emb[a], pseudo[a], z[f"desc_{ds}"])
        banks_train[ds] = (bank.few(emb[a]), pseudo[a])    # ES 的目标也用伪标签
        banks_eval[ds] = (bank.few(emb[bh]), y[bh])
        out[ds] = {
            "pseudo_label_acc_on_A": float((pseudo[a] == y[a]).mean()),
            "cross_on_B": float((lg[bh].argmax(1) == y[bh]).mean()),
            "n_train": int(n), "n_test": int(len(y) - n),
        }

    # 权重同样只用伪标签搜（LODO：在别的数据集上搜，目标任务零人工标注）
    dim = len(fh.FEW_FEATURES)
    for ds in banks_eval:
        src = {d: banks_train[d] for d in banks_train if d != ds}
        w_lodo, _ = fh.search_weights(src, dim, generations=60)
        w_self, _ = fh.search_weights({ds: banks_train[ds]}, dim, generations=60)
        f, y = banks_eval[ds]
        out[ds]["distilled_lodo"] = fh.accuracy(f, w_lodo, y)
        out[ds]["distilled_self"] = fh.accuracy(f, w_self, y)   # 本集伪标签搜权重，仍无人工标注

    names = list(out)
    print(f"{'方案':30s} " + " ".join(f"{d:>10s}" for d in names) + "   宏平均")
    for key, label in [("cross_on_B", "交叉编码器（老师）"),
                       ("distilled_self", "蒸馏后的双塔 Fly 头"),
                       ("distilled_lodo", "蒸馏 + LODO 权重")]:
        v = [out[d][key] for d in names]
        print(f"{label:30s} " + " ".join(f"{x*100:9.1f}%" for x in v) + f"  {np.mean(v)*100:6.1f}%")
    jv = [jev["per_dataset"][d]["acc"] for d in names]
    print(f"{'typesafe/jev':30s} " + " ".join(f"{x*100:9.1f}%" for x in jv) + f"  {np.mean(jv)*100:6.1f}%")
    print(f"{'（伪标签本身的准确率）':30s} " +
          " ".join(f"{out[d]['pseudo_label_acc_on_A']*100:9.1f}%" for d in names))
    Path("out/distill.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
