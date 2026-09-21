#!/usr/bin/env python3
"""测 laya-multilingual 的中文能力，用英文 checkpoint 作对照。

数据：CLUE/TNEWS 今日头条新闻标题分类，15 类。
对照的意义：Laya 文档称英文 checkpoint 在非拉丁文字上会崩溃
（高棠语 0.000 准确率 / 0.952 置信）—— 如果中文也如此，
说明多语种 checkpoint 不是可选项而是必需项。
"""
from __future__ import annotations
import json, os, sys, time
os.environ.setdefault("USE_TF", "0")
from pathlib import Path
import numpy as np, laya
from datasets import load_dataset

N = int(os.environ.get("N", 600))
CKPT = sys.argv[1] if len(sys.argv) > 1 else "multilingual"

# TNEWS 的标签是数字代码，映射成人能读的中文类目
NAMES = {"100":"民生故事","101":"文化","102":"娱乐","103":"体育","104":"财经",
         "106":"房产","107":"汽车","108":"教育","109":"科技","110":"军事",
         "112":"旅游","113":"国际","114":"股票","115":"农业","116":"电子竞技"}

d = load_dataset("clue/clue", "tnews")["validation"]
codes = d.features["label"].names
opts = [NAMES[c] for c in codes]
rows = d.shuffle(seed=42).select(range(N))

agent = laya.load("convaiinnovations/laya", **({} if CKPT == "english" else {"subfolder": CKPT}))
agent.cfg["head_max_len"] = 512; agent.cfg["max_len"] = 1024

crit = {o: o for o in opts}
correct, lat, confs, hits = 0, [], [], []
t0 = time.perf_counter()
for i, r in enumerate(rows):
    q = {"label": {"type":"choice","instructions":"这条新闻标题属于哪个类别？","criteria":crit}}
    t1 = time.perf_counter()
    res = agent.predict({"body": r["sentence"]}, q)["answers"]["label"]
    lat.append((time.perf_counter()-t1)*1000)
    ok = opts.index(res["choice"]) == r["label"]
    correct += ok; hits.append(ok); confs.append(res.get("confidence", 0.0))
    if i and i % 200 == 0:
        print(f"  {i}/{N} acc={correct/i*100:.1f}%", flush=True)

acc = correct/N
out = {"checkpoint": CKPT, "dataset": "CLUE/TNEWS", "n": N, "n_options": len(opts),
       "acc": acc, "random_baseline": 1/len(opts),
       "mean_confidence": float(np.mean(confs)),
       "lat_p50_ms": round(float(np.median(lat)),1),
       "wall_s": round(time.perf_counter()-t0,1)}
Path(f"out/laya_zh_{CKPT}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"[{CKPT}] 中文 TNEWS {acc*100:.1f}%（随机 {100/len(opts):.1f}%）"
      f"  平均置信 {out['mean_confidence']:.3f}  p50={out['lat_p50_ms']}ms")
