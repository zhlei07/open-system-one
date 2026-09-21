#!/usr/bin/env python3
"""Laya 的 CPU 延迟 —— 修正一处我们自己发布错了的数字。

laya.load() 不传 device 时会自动挑设备：cuda > mps > cpu。在 M5 Pro 上它选了 **mps**，
也就是 Apple GPU。我们 README 里把那组 19–86 ms 写成了"laptop CPU"，是错的：
本仓库其他所有链路（static / MiniLM / ModernBERT / 交叉编码器）都是纯 CPU、batch=1。

这个脚本在**同样的 CPU、batch=1** 口径下重测 Laya，让延迟那一列可比。
准确率只作抽样自检——设备不该改变预测，只用来确认 CPU 路径没走歪。
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
os.environ.setdefault("USE_TF", "0")
import numpy as np
import laya
from data import build

DS = ["sst2", "ag_news", "emotion", "banking77"]
PER = int(os.environ.get("LAYA_CPU_PER", 250))     # 每任务抽样条数
DEVICE = os.environ.get("LAYA_CPU_DEVICE", "cpu")

b = build(); ev = b["splits"]["eval"]
agent = laya.load("convaiinnovations/laya", device=DEVICE)
print(f"device={agent.device}", flush=True)
agent.cfg["head_max_len"] = 512
agent.cfg["max_len"] = 1024

def ask(text, opts):
    """和 laya_bench.py 完全相同的调用形状，否则延迟不可比。"""
    q = {"label": {"type": "choice",
                   "instructions": "Which option best describes this text?",
                   "criteria": {o: o for o in opts}}}
    res = agent.predict({"body": text}, q)["answers"]["label"]
    return opts.index(res["choice"])

out = {"device": str(agent.device), "per_dataset_n": PER,
       "head_max_len": agent.cfg["head_max_len"], "max_len": agent.cfg["max_len"],
       "per_dataset": {}}

for ds in DS:
    rows = [r for r in ev if r["dataset"] == ds][:PER]
    opts = b["options"][ds]
    lat, ok = [], 0
    # 预热，别把首次 kernel 编译算进 p50
    ask(rows[0]["text"], opts)
    for r in rows:
        t = time.perf_counter()
        try:
            pred = ask(r["text"], opts)
        except Exception as e:
            print("ERR", repr(e)[:140], flush=True); pred = None
        lat.append((time.perf_counter() - t) * 1000)
        ok += int(pred == r["label"])
    out["per_dataset"][ds] = {
        "n": len(rows), "acc": round(ok / len(rows), 4),
        "lat_p50_ms": round(float(np.percentile(lat, 50)), 1),
        "lat_p95_ms": round(float(np.percentile(lat, 95)), 1),
        "n_options": len(opts),
    }
    print(ds, out["per_dataset"][ds], flush=True)

Path(f"out/laya_cpu.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out["per_dataset"], indent=1))
