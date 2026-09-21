#!/usr/bin/env python3
"""交叉编码器 v2：多模板 + 可换骨干 + 候选裁剪。

和 v1 的区别：
- 一次前向同时算 **k 个候选 × T 个模板**，logit 存成 (n, k, T) 三维。
  模板集成的权重留给 fuse.py 用 LODO 搜——**不拿目标任务的标签挑模板**，
  否则"零样本"就名存实亡了。
- 骨干可换（base 149M / large 395M）。

用法:
    python cross_v2.py --model base  --ds sst2,ag_news --n 1000 --topk 0
    python cross_v2.py --model large --ds banking77   --n 1000 --topk 20 --templates 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import fly_head as fh
from data import CACHE, build

MODELS = {
    "base":  ("MoritzLaurer/ModernBERT-base-zeroshot-v2.0", 149),
    "large": ("MoritzLaurer/ModernBERT-large-zeroshot-v2.0", 395),
}
# 模板必须**任务无关**——一旦为某个数据集定制，零样本的前提就破了。
TEMPLATES = [
    "This example is about {}.",
    "This text expresses {}.",
    "The correct category for this text is {}.",
]
BI = "mbert-embed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base", choices=list(MODELS))
    ap.add_argument("--ds", default="sst2,ag_news,emotion,banking77")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--topk", type=int, default=0, help="0 = 全部选项")
    ap.add_argument("--templates", type=int, default=len(TEMPLATES))
    ap.add_argument("--enriched", action="store_true",
                    help="假设句直接用 descriptions.py 的丰富描述，而不是模板套标签名")
    a = ap.parse_args()

    repo, params = MODELS[a.model]
    tmpls = TEMPLATES[:a.templates]
    b = build()
    z = np.load(CACHE / f"emb/{BI}.npz")
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
    torch.set_grad_enabled(False)
    ent = next(i for i, l in model.config.id2label.items() if "entail" in l.lower())
    print(f"{repo} ({params}M)  模板 {len(tmpls)} 个  entail_idx={ent}", flush=True)

    ev = b["splits"]["eval"]
    for ds in a.ds.split(","):
        opts = b["options"][ds]
        if a.enriched:
            from descriptions import describe
            # 丰富描述本身就是一个完整的句子，不再套模板——套了反而变成
            # "This example is about The customer wants to activate a new card.." 这种病句
            desc_text = describe(ds, opts, True)
        idx = [i for i, r in enumerate(ev) if r["dataset"] == ds][:a.n]
        y = np.array([ev[i]["label"] for i in idx])
        k = min(a.topk, len(opts)) if a.topk else len(opts)

        cand = None
        if k < len(opts):                      # 双塔裁候选
            bank = fh.FeatureBank(opts, None, None, z[f"desc_{ds}"])
            w = np.zeros(len(fh.ZERO_FEATURES), np.float32); w[0] = w[3] = 1.0
            cand = np.argsort(-(bank.zero(z["eval"][idx]) @ w), axis=1)[:, :k]

        logits = np.full((len(idx), len(opts), len(tmpls)), -np.inf, np.float32)
        lat, t0 = [], time.perf_counter()
        for row, i in enumerate(idx):
            c = cand[row] if cand is not None else np.arange(len(opts))
            text = ev[i]["text"]
            pairs_a, pairs_b = [], []
            for t in tmpls:                     # k×T 个 pair 一次前向
                for j in c:
                    pairs_a.append(text)
                    # 丰富描述本身是完整句子，不再套模板（套了会变成病句）
                    pairs_b.append(desc_text[j] if a.enriched else t.format(opts[j]))
            t1 = time.perf_counter()
            enc = tok(pairs_a, pairs_b, return_tensors="pt", padding=True,
                      truncation=True, max_length=256)
            out = model(**enc).logits[:, ent].float().numpy()
            lat.append((time.perf_counter() - t1) * 1000)
            logits[row][np.ix_(c, range(len(tmpls)))] = out.reshape(len(tmpls), len(c)).T
            if row and row % 250 == 0:
                acc = (np.where(np.isfinite(logits[:row]), logits[:row], -1e9).mean(-1)
                       .argmax(1) == y[:row]).mean()
                print(f"  {ds} {row}/{len(idx)} acc={acc*100:.1f}% "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)

        mean_logit = np.where(np.isfinite(logits), logits, -1e9).mean(-1)
        acc = float((mean_logit.argmax(1) == y).mean())
        lat.sort()
        tag = f"{a.model}_{ds}_k{k}_t{len(tmpls)}_n{len(idx)}" + ("_rich" if a.enriched else "")
        np.savez_compressed(CACHE / f"xc_{tag}.npz", logits=logits, y=y,
                            idx=np.array(idx), cand=cand if cand is not None else np.zeros(0))
        meta = {"model": a.model, "params_m": params, "dataset": ds, "n": len(idx),
                "enriched": bool(a.enriched),
                "k": k, "n_options": len(opts), "templates": len(tmpls),
                "acc_mean_template": acc,
                "lat_p50_ms": round(lat[len(lat) // 2], 1),
                "lat_p95_ms": round(lat[int(len(lat) * .95)], 1),
                "wall_s": round(time.perf_counter() - t0, 1)}
        Path("out").mkdir(exist_ok=True)
        (Path("out") / f"xc_{tag}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        print(f"[{a.model}/{ds}] k={k} T={len(tmpls)} acc={acc*100:.1f}% "
              f"p50={meta['lat_p50_ms']}ms {meta['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
