#!/usr/bin/env python3
"""离线拟合列均值：裁剪器一层 + 交叉编码器一层，都改成归纳式。

两层原本都是直推式的（对当前这批查询求列均值），单条请求时整项退化成 0：

  裁剪器   fly_head 的 t0c —— 已公布的 recall@20 97.8% 依赖它，
           单条时会掉回裸余弦的 96.8%。
  交叉编码 还没有这一层，本脚本加上。

池子取 **fit 切分的文本，标签一律不读**，不碰 eval。
拟合产物写到 cache/colcenter_{bi,cross}_<ds>.json，线上直接 load。

    python fit_colcenter.py banking77 --pool 500
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from colcenter import GATE, ColumnCenter, gated
from data import CACHE, build

REPO = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
TEMPLATE = "This example is about {}."     # 与 cache/cross_*.npz 逐位一致，已核验
BI = "mbert-embed"
K = 20


def recall(cand, y):
    return float((cand == y[:, None]).any(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds", nargs="?", default="banking77")
    ap.add_argument("--pool", type=int, default=500)
    ap.add_argument("--k", type=int, default=K)
    a = ap.parse_args()
    ds, k = a.ds, a.k

    b = build()
    opts = b["options"][ds]
    if not gated(len(opts)):
        print(f"{ds} 只有 {len(opts)} 个选项 < 门控 {GATE}，不该开列中心化。")
        return
    z = np.load(CACHE / f"emb/{BI}.npz")
    ev = b["splits"]["eval"]
    idx = [i for i, r in enumerate(ev) if r["dataset"] == ds]
    y = np.array([ev[i]["label"] for i in idx])
    desc0 = z[f"desc_{ds}"][0]                       # 模板 0 = 裸标签名
    pool_emb = z[f"fit_{ds}"][: a.pool]              # fit 切分，只取向量，标签不读
    pool_txt = [r["text"] for r in b["splits"]["fit"] if r["dataset"] == ds][: a.pool]
    print(f"{ds}: {len(opts)} 选项 | eval {len(y)} 条 | 无标签池 {len(pool_txt)} 条(fit 切分)")

    # ---------- 第 1 层：裁剪器 ----------
    cc_bi = ColumnCenter(len(opts)).fit(pool_emb @ desc0.T)
    t0_ev = z["eval"][idx] @ desc0.T
    trans = t0_ev + (t0_ev - t0_ev.mean(0, keepdims=True))     # 现状：直推式 2*t0 - 批均值
    variants = {
        "裸余弦（单条时的现状）":      t0_ev,
        "直推式 t0+t0c（已公布口径）": trans,
        "归纳式 t0+(t0-池均值)":       t0_ev + cc_bi.transform(t0_ev),
        "归纳式 纯 t0-池均值":         cc_bi.transform(t0_ev),
    }
    print(f"\n裁剪器 recall@{k}")
    cands = {}
    for name, s in variants.items():
        cands[name] = np.argsort(-s, axis=1)[:, :k]
        print(f"  {name:26s} {recall(cands[name], y)*100:5.2f}%")
    pick = "归纳式 t0+(t0-池均值)"
    cand_ev = cands[pick]
    cc_bi.save(CACHE / f"colcenter_bi_{ds}.json", opts)

    # ---------- 第 2 层：交叉编码器 ----------
    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()
    torch.set_grad_enabled(False)
    ent = next(i for i, l in model.config.id2label.items() if "entail" in l.lower())

    def score(texts, cand, warm=None):
        """(n, len(opts)) 的 logit，未打分处 NaN。warm 提供已缓存的值就跳过重算。"""
        out = np.full((len(texts), len(opts)), np.nan, np.float32)
        todo = 0
        t0 = time.perf_counter()
        for r, text in enumerate(texts):
            c = cand[r]
            if warm is not None:
                have = np.isfinite(warm[r][c])
                out[r][c[have]] = warm[r][c[have]]
                c = c[~have]
            if len(c) == 0:
                continue
            todo += len(c)
            enc = tok([text] * len(c), [TEMPLATE.format(opts[int(j)]) for j in c],
                      return_tensors="pt", padding=True, truncation=True, max_length=256)
            out[r][c] = model(**enc).logits[:, ent].float().numpy()
            if r and r % 250 == 0:
                print(f"    {r}/{len(texts)}  新算 {todo} 对  "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
        print(f"    完成：新算 {todo} 对，{time.perf_counter()-t0:.0f}s", flush=True)
        return out

    cand_pool = np.argsort(-(pool_emb @ desc0.T + cc_bi.transform(pool_emb @ desc0.T)),
                           axis=1)[:, :k]
    print("\n给无标签池打分（拟合交叉编码器的列均值）")
    cc_cross = ColumnCenter(len(opts)).fit(score(pool_txt, cand_pool))
    cc_cross.save(CACHE / f"colcenter_cross_{ds}.json", opts)
    print(f"  {cc_cross}")

    # eval：复用旧缓存里命中的部分，只补缺的
    old = np.load(CACHE / f"cross_{ds}_k{k}.npz")
    assert (old["y"] == y).all(), "缓存的 y 和当前切分不一致"
    warm = np.where(np.isfinite(old["logits"]), old["logits"], np.nan)
    print("\n给 eval 打分（新候选集，复用旧缓存）")
    lg = score([ev[i]["text"] for i in idx], cand_ev, warm=warm)

    mask = np.zeros_like(lg, bool)
    np.put_along_axis(mask, cand_ev, True, axis=1)
    base = float((np.where(mask, lg, -1e9).argmax(1) == y).mean())
    cent = float((np.where(mask, lg - cc_cross.mean, -1e9).argmax(1) == y).mean())
    published = float((np.where(np.isfinite(old["logits"]), old["logits"], -1e9)
                       .argmax(1) == y).mean())

    res = {"dataset": ds, "n": len(y), "k": k, "n_pool": len(pool_txt),
           "n_options": len(opts), "gate": GATE, "template": TEMPLATE,
           "recall": {n: recall(c, y) for n, c in cands.items()},
           "acc_published_transductive_prefilter": published,
           "acc_inductive_prefilter": base,
           "acc_inductive_prefilter_plus_centering": cent,
           "delta_centering_pp": (cent - base) * 100,
           "delta_vs_published_pp": (cent - published) * 100}
    Path(f"out/colcenter_{ds}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n已公布（直推式裁剪 + 无中心化）      {published*100:5.2f}%")
    print(f"归纳式裁剪，无中心化                {base*100:5.2f}%  ({(base-published)*100:+.2f})")
    print(f"归纳式裁剪 + 归纳式列中心化          {cent*100:5.2f}%  ({(cent-published)*100:+.2f})")
    print(f"-> out/colcenter_{ds}.json")


if __name__ == "__main__":
    main()
