#!/usr/bin/env python3
"""把 static + Fly 头打包成自包含的 HF 模型仓库。

决策头本身只有 8 个浮点数，但它需要三样离线算好的东西才能推理：
类质心、用于 kNN 特征的 fit 向量、选项描述向量。全部一起打包，
这样使用者只要 numpy + model2vec 就能跑，不需要 clone 本仓库。
"""
import json, shutil
from pathlib import Path
import numpy as np
from data import CACHE, build

DS = ["sst2", "ag_news", "emotion", "banking77"]
N_PROTO = 1500
OUT = Path("hf/model")

b = build()
z = np.load(CACHE / "emb/static.npz")
heads = json.load(open("out/heads.json"))
w = heads["static"]["weights_few"]
feature_names = list(w)
weights = [w[k] for k in feature_names]

meta = {"encoder": "minishlab/potion-base-8M", "dim": int(z["eval"].shape[1]),
        "feature_names": feature_names, "weights": weights, "n_prototypes": N_PROTO,
        "datasets": {}}

for ds in DS:
    fit_rows = [r for r in b["splits"]["fit"] if r["dataset"] == ds]
    y = np.array([r["label"] for r in fit_rows])[:N_PROTO]
    emb = z[f"fit_{ds}"][:N_PROTO]
    opts = b["options"][ds]
    cent = np.stack([emb[y == c].mean(0) if (y == c).any() else np.zeros(emb.shape[1])
                     for c in range(len(opts))]).astype(np.float32)
    cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-9)
    counts = np.array([(y == c).sum() for c in range(len(opts))], np.float32)
    lp = np.log(np.maximum(counts, 1) / counts.sum()); lp -= lp.mean()
    np.savez_compressed(OUT / "heads" / f"{ds}.npz",
                        fit_emb=emb.astype(np.float32), fit_lab=y.astype(np.int32),
                        centroids=cent, desc_emb=z[f"desc_{ds}"][0].astype(np.float32),
                        global_mean=emb.mean(0).astype(np.float32),
                        log_prior=lp.astype(np.float32))
    meta["datasets"][ds] = {"options": opts, "n_options": len(opts),
                            "accuracy_2500_eval": heads["static"]["per_dataset"][ds]["fly_few"]}

(OUT / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
print("权重（8 个浮点数）:", json.dumps(dict(zip(feature_names, [round(x,4) for x in weights]))))
for f in sorted((OUT / "heads").glob("*.npz")):
    print(f"  {f.name:16s} {f.stat().st_size/1024:.0f} KB")
