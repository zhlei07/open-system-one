#!/usr/bin/env python3
"""CLINC 路由：只换裁剪器，Laya 不动。

用例里的 84.5% 是 8M 静态嵌入裁到前 20 名之后的数，天花板 92.0%。
这里在同一 400 条上再跑一遍，并把裁剪器换成 ModernBERT 嵌入。
打分都是裸标签余弦，和 ensemble._prefilter 相同，不列中心化。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import laya

from encoders import STEncoder, StaticEncoder
from usecases import build_usecases

TOPK = 20


def shortlist(q, d, k):
    return np.argsort(-(q @ d.T), axis=1)[:, :k]


def run(agent, texts, opts, y, cand):
    ok = 0
    lat = []
    for row, text in enumerate(texts):
        c = cand[row]
        q = {"label": {"type": "choice",
                       "instructions": "Which option best describes this text?",
                       "criteria": {opts[j]: opts[j] for j in c}}}
        t0 = time.perf_counter()
        res = agent.predict({"body": text}, q)["answers"]["label"]
        lat.append((time.perf_counter() - t0) * 1000)
        ok += int(opts.index(res["choice"]) == int(y[row]))
    lat.sort()
    return {"acc": ok / len(y),
            "ceiling_recall": float((cand == y[:, None]).any(1).mean()),
            "lat_p50_ms": round(lat[len(lat) // 2], 1)}


def main():
    u = build_usecases(500)["routing"]
    texts, opts = u.texts[100:], u.options
    y = np.array(u.gold[100:])
    print(f"CLINC {len(opts)} 类  {len(y)} 条", flush=True)

    agent = laya.load("convaiinnovations/laya", device=os.environ.get("LAYA_DEVICE") or None)
    agent.cfg["head_max_len"] = 512
    agent.cfg["max_len"] = 1024
    print(f"device={agent.device}", flush=True)

    static = StaticEncoder().load()
    mbert = STEncoder("mbert-embed", "nomic-ai/modernbert-embed-base", 149.0,
                      prefix="search_query: ").load()
    out = {"n": len(y), "k": TOPK, "device": str(agent.device)}
    for name, enc in (("static", static), ("mbert-embed", mbert)):
        cand = shortlist(enc.encode(texts), enc.encode(opts), TOPK)
        out[name] = run(agent, texts, opts, y, cand)
        print(f"  {name:12s}  acc={out[name]['acc']*100:.1f}%  "
              f"R@{TOPK}={out[name]['ceiling_recall']*100:.1f}%  "
              f"p50={out[name]['lat_p50_ms']}ms", flush=True)
    Path("out/clinc_prefilter.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
