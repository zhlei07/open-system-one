#!/usr/bin/env python3
"""在「最火的 5 类 Jev 用例」上横向对比我们的本地方案。

口径（和主基准保持一致，否则数字没法拼在一起看）：
  - 每个用例前 100 条做校准（定温度 + 融合权重），其余只看一次
  - batch=1，纯 CPU（Laya 会自己选 mps，脚本里显式记录设备）
  - 选项多于 20 时用 8M 静态嵌入裁候选，裁剪前后都记
  - 解析失败按答错计，单独报失败数
  - 多数类基线一并报——2 选项的任务不报这个就看不出模型有没有在干活

跑：  .venv/bin/python usecase_bench.py [每用例条数] [stacks]
      stacks 逗号分隔，默认 cross,laya,fused；加 jev 会花钱（约 $0.01/千条）
"""
from __future__ import annotations

import json, os, sys, time
from collections import Counter
from pathlib import Path

import numpy as np

from usecases import build_usecases

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
STACKS = (sys.argv[2] if len(sys.argv) > 2 else "cross,laya,fused").split(",")
N_CAL = 100
OUT = Path("out"); OUT.mkdir(exist_ok=True)

cases = build_usecases(N)
print(f"用例 {len(cases)} 个，每个 {N} 条（校准 {N_CAL} / 评测 {N - N_CAL}）\n")

results: dict[str, dict] = {}

def evaluate(fn, u, idx, tag):
    """fn(text, options) -> (选中下标 or None)。None 记作失败并算错。"""
    ok = fails = 0; lat = []
    for i in idx:
        t0 = time.perf_counter()
        try:
            p = fn(u.texts[i], u.options)
        except Exception as e:
            if fails == 0: print(f"      首个异常 {tag}: {repr(e)[:110]}", flush=True)
            p = None
        lat.append((time.perf_counter() - t0) * 1000)
        if p is None: fails += 1
        else: ok += int(p == u.gold[i])
    return {"acc": ok / len(idx), "fails": fails,
            "lat_p50_ms": round(float(np.percentile(lat, 50)), 1),
            "lat_p95_ms": round(float(np.percentile(lat, 95)), 1)}

for key, u in cases.items():
    cal, evl = list(range(N_CAL)), list(range(N_CAL, len(u)))
    maj = Counter(u.gold[i] for i in evl).most_common(1)[0][1] / len(evl)
    r = {"jev_usecase": u.jev_usecase, "primitive": u.primitive, "source": u.source,
         "n_options": u.n_options, "n_eval": len(evl), "n_cal": len(cal),
         "majority_baseline": round(maj, 4), "random_baseline": round(1 / u.n_options, 4),
         "stacks": {}}
    print(f"=== {key}  ({u.jev_usecase})  {u.n_options} 选项  "
          f"多数类基线 {maj*100:.1f}%", flush=True)

    ens: dict[str, object] = {}
    for name in ("cross", "laya"):
        if name in STACKS or "fused" in STACKS:
            from ensemble import Ensemble
            ens.setdefault(name, Ensemble(backends=(name,)))

    for name in ("cross", "laya"):
        if name not in STACKS: continue
        e = ens[name]
        r["stacks"][name] = evaluate(
            lambda t, o, e=e: int(np.argmax(e.decide(t, o).probs)), u, evl, name)
        print(f"  {name:8s} {r['stacks'][name]['acc']*100:5.1f}%  "
              f"p50={r['stacks'][name]['lat_p50_ms']:.0f}ms  失败 {r['stacks'][name]['fails']}", flush=True)

    if "fused" in STACKS:
        from ensemble import Ensemble
        f = Ensemble(backends=("cross", "laya"))
        info = f.calibrate([(u.texts[i], u.options, u.gold[i]) for i in cal], verbose=False)
        r["fused_calibration"] = info
        r["stacks"]["fused"] = evaluate(
            lambda t, o: int(np.argmax(f.decide(t, o).probs)), u, evl, "fused")
        print(f"  {'fused':8s} {r['stacks']['fused']['acc']*100:5.1f}%  "
              f"p50={r['stacks']['fused']['lat_p50_ms']:.0f}ms  "
              f"权重 { {k: round(v,2) for k,v in info['weights'].items()} }", flush=True)

    if "jev" in STACKS:
        from jev_adapter import JevClient
        jc = JevClient()
        def _jev(t, o):
            res = jc.predict(t, o)
            return o.index(res) if isinstance(res, str) and res in o else None
        r["stacks"]["jev"] = evaluate(_jev, u, evl, "jev")
        print(f"  {'jev':8s} {r['stacks']['jev']['acc']*100:5.1f}%  "
              f"p50={r['stacks']['jev']['lat_p50_ms']:.0f}ms  失败 {r['stacks']['jev']['fails']}", flush=True)

    results[key] = r
    (OUT / "usecases.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(flush=True)

print("=== 汇总（macro）===")
for s in STACKS:
    vals = [r["stacks"][s]["acc"] for r in results.values() if s in r["stacks"]]
    if vals: print(f"  {s:8s} {np.mean(vals)*100:.2f}%")
