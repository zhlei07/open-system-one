#!/usr/bin/env python3
"""把我们的候选裁剪装到 Laya 上，攻它最大的弱点。

Laya 文档自述：77 个选项共享固定 head_max_len，每个只分到 3-4 token，
文本变得无法区分 —— 并建议"拆成两步、由粗到细的层次化选择"，但未实现。

我们实现了：双塔零样本裁到 top-k，只把这 k 个选项交给 Laya。
如果 Laya 的 banking77 从 54.3% 被救起来，说明这个补丁对它有效。
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("USE_TF", "0")
import numpy as np, laya
import fly_head as fh
from data import CACHE, build

DS = sys.argv[1] if len(sys.argv) > 1 else "banking77"
TOPK = int(sys.argv[2]) if len(sys.argv) > 2 else 20
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2500

b = build(); ev = b["splits"]["eval"]
z = np.load(CACHE / "emb/mbert-embed.npz")
idx = [i for i, r in enumerate(ev) if r["dataset"] == DS][:N]
y = np.array([ev[i]["label"] for i in idx])
opts = b["options"][DS]

# 双塔裁候选：cos + 列中心化 cos，零样本，不用标注
bank = fh.FeatureBank(opts, None, None, z[f"desc_{DS}"])
w = np.zeros(len(fh.ZERO_FEATURES), np.float32); w[0] = w[3] = 1.0
cand = np.argsort(-(bank.zero(z["eval"][idx]) @ w), axis=1)[:, :TOPK]
ceiling = float((cand == y[:, None]).any(1).mean())
print(f"{DS}: 双塔裁到 top-{TOPK}，天花板 recall = {ceiling*100:.1f}%", flush=True)

agent = laya.load("convaiinnovations/laya", device=os.environ.get("LAYA_DEVICE") or None)
agent.cfg["head_max_len"] = 512; agent.cfg["max_len"] = 1024
# 设备必须落盘：laya.load() 不传 device 时会自动选 cuda>mps>cpu，
# 在 Mac 上它默认走 mps，和本仓库其他纯 CPU 链路不可比。
print(f"device={agent.device}", flush=True)

correct = 0; lat = []
t0 = time.perf_counter()
for row, i in enumerate(idx):
    c = cand[row]
    q = {"label": {"type": "choice", "instructions": "Which option best describes this text?",
                   "criteria": {opts[j]: opts[j] for j in c}}}
    t1 = time.perf_counter()
    res = agent.predict({"body": ev[i]["text"]}, q)["answers"]["label"]
    lat.append((time.perf_counter() - t1) * 1000)
    correct += int(opts.index(res["choice"]) == y[row])
    if row and row % 500 == 0:
        print(f"  {row}/{len(idx)} acc={correct/row*100:.1f}% ({time.perf_counter()-t0:.0f}s)", flush=True)

acc = correct / len(idx)
out = {"dataset": DS, "topk": TOPK, "n": len(idx), "ceiling_recall": ceiling,
       "device": str(agent.device),
       "acc": acc, "lat_p50_ms": round(float(np.median(lat)), 1),
       "wall_s": round(time.perf_counter() - t0, 1)}
Path(f"out/laya_shortlist_{DS}_k{TOPK}{'_cpu' if agent.device.type == 'cpu' else ''}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"[laya+裁剪{TOPK}] {DS} {acc*100:.1f}%  p50={out['lat_p50_ms']}ms  {out['wall_s']}s")
