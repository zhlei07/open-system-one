#!/usr/bin/env python3
"""Jev 的公平性重跑：用**和我们方案完全相同的丰富描述**当 criteria。

之前只给了裸标签名。Cloudflare 官方示例里 criteria 是完整句子，
所以裸标签名很可能低估了 Jev。这一跑的结果不管是涨是跌都必须写进报告——
只在对自己有利的条件下测对手，是这类对比最常见也最致命的问题。
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import fly_head as fh
from data import build
from descriptions import describe
from jev_adapter import JevClient

DATASETS = ["sst2", "ag_news", "emotion", "banking77"]
CONC = 8


def main():
    b = build()
    ev = b["splits"]["eval"]
    crit = {ds: dict(zip(b["options"][ds], describe(ds, b["options"][ds], True)))
            for ds in DATASETS}
    clients = [JevClient() for _ in range(CONC)]
    results = [None] * len(ev)
    errors = []

    def work(arg):
        i, r = arg
        c = clients[i % CONC]
        try:
            pred, probs = c.predict(r["text"], b["options"][r["dataset"]],
                                    criteria=crit[r["dataset"]])
        except Exception as e:
            errors.append(f"{i}: {e}")
            return
        results[i] = (pred, probs)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        for n, _ in enumerate(ex.map(work, enumerate(ev))):
            if n and n % 2000 == 0:
                print(f"  {n}/{len(ev)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    per = {d: [0, 0] for d in DATASETS}
    probs_by, y_by = {d: [] for d in DATASETS}, {d: [] for d in DATASETS}
    for r, row in zip(results, ev):
        d = row["dataset"]; per[d][1] += 1
        if r is None:
            continue
        per[d][0] += int(r[0] == row["label"])
        probs_by[d].append(r[1]); y_by[d].append(row["label"])

    tokens = sum(c.total_input_tokens for c in clients)
    out = {"criteria": "enriched", "n": len(ev), "errors": len(errors),
           "error_sample": errors[:5], "wall_s": round(time.perf_counter()-t0, 1),
           "per_dataset": {d: {"acc": v[0]/v[1] if v[1] else None, "n": v[1]} for d, v in per.items()},
           "overall_acc": sum(v[0] for v in per.values())/max(1, sum(v[1] for v in per.values())),
           "input_tokens": tokens, "cost_usd": round(tokens/1e6*0.042, 4)}
    for d in DATASETS:
        if probs_by[d]:
            out["per_dataset"][d]["ece"] = fh.ece(np.array(probs_by[d]), np.array(y_by[d]))
    Path("out/jev_enriched.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

    bare = json.load(open("out/jev.json"))
    print(f"\n{'任务':11s} {'裸标签名':>9s} {'丰富描述':>9s} {'Δ':>7s}")
    for d in DATASETS:
        a0, a1 = bare["per_dataset"][d]["acc"], out["per_dataset"][d]["acc"]
        print(f"{d:11s} {a0*100:8.1f}% {a1*100:8.1f}% {(a1-a0)*100:+6.1f}")
    print(f"{'总体':11s} {bare['overall_acc']*100:8.1f}% {out['overall_acc']*100:8.1f}% "
          f"{(out['overall_acc']-bare['overall_acc'])*100:+6.1f}")
    print(f"成本 ${out['cost_usd']}  失败 {len(errors)}  {out['wall_s']}s")


if __name__ == "__main__":
    main()
