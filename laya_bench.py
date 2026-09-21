#!/usr/bin/env python3
"""把 Laya 接进同一套 10,000 条基准。

为什么值得做：目前全网没有人把 Laya 和 Jev 放在**同一批样本**上测过。
Laya 的 README 引用的 Jev 数字是第三方发布的（不同样本、不同 prompt、72 vs 77 标签），
而我们手上有 Jev 在这 10,000 条上的逐条预测。补上 Laya 这一列，
就是第一份三方同条件对比。

口径与其他链路完全一致：
- 选项用裸标签名（和 jev_preds_bare / cross_bare 相同）
- 逐条预测存盘，失败算错、单独报失败率
- 高选项数任务按 Laya 官方建议抬高 head_max_len
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("USE_TF", "0")
import numpy as np
import laya
from data import build

DS = ["sst2", "ag_news", "emotion", "banking77"]
CKPT = os.environ.get("LAYA_CKPT", "")          # "" = English root, 或 "typed-decisions"
LIMIT = int(os.environ.get("LAYA_LIMIT", 0)) or None

b = build(); ev = b["splits"]["eval"]
if LIMIT:
    ev = ev[:LIMIT]

t0 = time.perf_counter()
agent = laya.load("convaiinnovations/laya", device=os.environ.get("LAYA_DEVICE") or None,
                  **({"subfolder": CKPT} if CKPT else {}))
print(f"加载 {CKPT or 'english'} 用时 {time.perf_counter()-t0:.1f}s", flush=True)

# 77 个选项在默认 192 token 预算下每个只分到 3-4 token —— 官方明确说这是它 banking77
# 掉到 0.425 的原因，并建议抬高预算。不抬就是在测一个已知的配置错误。
agent.cfg["head_max_len"] = 512
agent.cfg["max_len"] = 1024
print(f"head_max_len={agent.cfg['head_max_len']} max_len={agent.cfg['max_len']}", flush=True)
# 设备自动选择（cuda>mps>cpu）会悄悄改变延迟口径，必须显式记录。
print(f"device={agent.device}", flush=True)

per = {d: [0, 0] for d in DS}
lat = {d: [] for d in DS}
rows, errors = [], 0
t0 = time.perf_counter()
for i, r in enumerate(ev):
    ds = r["dataset"]; opts = b["options"][ds]
    q = {"label": {"type": "choice",
                   "instructions": "Which option best describes this text?",
                   "criteria": {o: o for o in opts}}}
    t1 = time.perf_counter()
    try:
        res = agent.predict({"body": r["text"]}, q)["answers"]["label"]
        pred = opts.index(res["choice"]); probs = res.get("probabilities", {})
    except Exception as e:
        errors += 1; pred, probs = None, {}
        if errors <= 3:
            print("ERR", repr(e)[:140], flush=True)
    lat[ds].append((time.perf_counter() - t1) * 1000)
    per[ds][1] += 1
    if pred is not None:
        per[ds][0] += int(pred == r["label"])
    rows.append({"i": i, "dataset": ds, "gold": r["label"], "pred": pred,
                 "probs": [round(float(probs.get(o, 0.0)), 4) for o in opts] if probs else None})
    if i and i % 500 == 0:
        done = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
        print(f"  {i}/{len(ev)} acc={done[0]/max(done[1],1)*100:.1f}% "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

tag = CKPT or "english"
Path(f"out/laya_preds_{tag}.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
out = {"checkpoint": tag, "n": len(ev), "errors": errors, "device": str(agent.device),
       "head_max_len": agent.cfg["head_max_len"], "max_len": agent.cfg["max_len"],
       "wall_s": round(time.perf_counter()-t0, 1),
       "per_dataset": {d: {"acc": per[d][0]/per[d][1] if per[d][1] else None, "n": per[d][1],
                           "lat_p50_ms": round(float(np.median(lat[d])), 1) if lat[d] else None}
                       for d in DS},
       "overall_acc": sum(v[0] for v in per.values())/max(1, sum(v[1] for v in per.values()))}
Path(f"out/laya_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out["per_dataset"], indent=1))
print(f"[laya/{tag}] 总体 {out['overall_acc']*100:.1f}%  失败 {errors}  {out['wall_s']}s")
