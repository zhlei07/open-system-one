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

from descriptions import describe_usecase
from usecases import build_usecases

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
STACKS = (sys.argv[2] if len(sys.argv) > 2 else "cross,laya,fused").split(",")
ONLY = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None   # 只跑某几个用例
RICH = os.environ.get("RICH") == "1"       # 用丰富选项描述而不是裸标签
N_CAL = 100
OUT = Path("out"); OUT.mkdir(exist_ok=True)

cases = build_usecases(N)
if ONLY:
    cases = {k: v for k, v in cases.items() if k in ONLY}

# 后端只装载一次，cross 单打 / laya 单打 / 融合三条链路共享同一批权重。
from ensemble import PREFILTERS, Ensemble, _BACKENDS
_pool: dict[str, object] = {}
for _b in ("cross", "laya"):
    if _b in STACKS or "fused" in STACKS:
        _pool[_b] = _BACKENDS[_b]()
        print(f"装载后端 {_b}"
              + (f"（设备 {_pool[_b].device}）" if hasattr(_pool[_b], "device") else ""), flush=True)
print(f"用例 {len(cases)} 个，每个 {N} 条（校准 {N_CAL} / 评测 {N - N_CAL}）\n")

results: dict[str, dict] = {}
TAG = "_rich" if RICH else ""
_prev = OUT / f"usecases{TAG}.json"
if _prev.exists():
    results.update(json.loads(_prev.read_text()))   # 合并，别把没重跑的用例抹掉

preds: dict[str, dict] = {}
_pp = OUT / f"usecase_preds{TAG}.json"
if _pp.exists():
    preds.update(json.loads(_pp.read_text()))

def evaluate(fn, u, idx, tag):
    """fn(text, options) -> (选中下标 or None)。None 记作失败并算错。

    逐条预测必须落盘。只累加 ok 的话，事后要画混淆矩阵（比如判断 Jev 在护栏上
    41.8% 是不是选项下标反了）就得重跑一遍——Jev 那一列是要花钱的。
    """
    ok = fails = 0; lat = []; got = []
    for i in idx:
        t0 = time.perf_counter()
        try:
            p = fn(u.texts[i], u.options)
        except Exception as e:
            if fails == 0: print(f"      首个异常 {tag}: {repr(e)[:110]}", flush=True)
            p = None
        lat.append((time.perf_counter() - t0) * 1000)
        got.append(None if p is None else int(p))
        if p is None: fails += 1
        else: ok += int(p == u.gold[i])
    return ({"acc": ok / len(idx), "fails": fails,
             "lat_p50_ms": round(float(np.percentile(lat, 50)), 1),
             "lat_p95_ms": round(float(np.percentile(lat, 95)), 1)}, got)


def attach_center(e, key, u):
    """高基数用例加载离线拟合的列均值（cache/colcenter_cross_<用例>.json）。

    选项数不到 colcenter.GATE 的用例根本没有这个文件——门控是按选项个数定的，
    不按用例名，所以"有没有文件"本身就是门控的结果，不需要在这里再判一次。
    """
    f = Path("cache") / f"colcenter_cross_{key}.json"
    if f.exists() and "cross" in getattr(e, "backends", {}):
        e.load_center(f, u.options)
        return True
    return False


def record(key, name, u, idx, got):
    """把 (下标, 金标, 预测) 三元组存下来，混淆矩阵事后随便画。"""
    d = preds.setdefault(key, {"idx": list(idx), "gold": [u.gold[i] for i in idx],
                               "options": u.options, "pred": {}})
    d["pred"][name] = got
    _pp.write_text(json.dumps(preds, ensure_ascii=False))

for key, u in cases.items():
    cal, evl = list(range(N_CAL)), list(range(N_CAL, len(u)))
    maj = Counter(u.gold[i] for i in evl).most_common(1)[0][1] / len(evl)
    r = {"jev_usecase": u.jev_usecase, "primitive": u.primitive, "source": u.source,
         "n_options": u.n_options, "n_eval": len(evl), "n_cal": len(cal),
         "majority_baseline": round(maj, 4), "random_baseline": round(1 / u.n_options, 4),
         "prefilters": dict(PREFILTERS), "stacks": {}}
    print(f"=== {key}  ({u.jev_usecase})  {u.n_options} 选项  "
          f"多数类基线 {maj*100:.1f}%", flush=True)

    hyp = describe_usecase(key, u.options, RICH)
    if RICH and hyp is None:
        print(f"    （{key} 无手写描述，退回裸标签）", flush=True)
    hyp = None if (not RICH or hyp == list(u.options)) else hyp
    r["enriched"] = bool(hyp)

    for name in ("cross", "laya"):
        if name not in STACKS: continue
        e = Ensemble(backends={name: _pool[name]})
        if attach_center(e, key, u):
            print(f"    {name}: 已加载列均值 cache/colcenter_cross_{key}.json", flush=True)
        r["stacks"][name], _got = evaluate(
            lambda t, o, e=e: int(np.argmax(e.decide(t, o, hyp).probs)), u, evl, name)
        record(key, name, u, evl, _got)
        print(f"  {name:8s} {r['stacks'][name]['acc']*100:5.1f}%  "
              f"p50={r['stacks'][name]['lat_p50_ms']:.0f}ms  失败 {r['stacks'][name]['fails']}", flush=True)

    if "fused" in STACKS:
        f = Ensemble(backends={k: _pool[k] for k in ("cross", "laya")})
        attach_center(f, key, u)
        info = f.calibrate([(u.texts[i], u.options, u.gold[i]) for i in cal],
                           hypotheses=hyp, verbose=False)
        r["fused_calibration"] = info
        r["stacks"]["fused"], _got = evaluate(
            lambda t, o: int(np.argmax(f.decide(t, o, hyp).probs)), u, evl, "fused")
        record(key, "fused", u, evl, _got)
        print(f"  {'fused':8s} {r['stacks']['fused']['acc']*100:5.1f}%  "
              f"p50={r['stacks']['fused']['lat_p50_ms']:.0f}ms  "
              f"权重 { {k: round(v,2) for k,v in info['weights'].items()} }", flush=True)

    if "jev" in STACKS:
        from jev_adapter import JevClient
        jc = JevClient()
        def _jev(t, o):
            # predict() 返回 (下标, 概率列表)，不是字符串。按字符串处理会把每条都算成失败。
            idx, _ = jc.predict(t, o)
            return int(idx)
        r["stacks"]["jev"], _got = evaluate(_jev, u, evl, "jev")
        record(key, "jev", u, evl, _got)
        print(f"  {'jev':8s} {r['stacks']['jev']['acc']*100:5.1f}%  "
              f"p50={r['stacks']['jev']['lat_p50_ms']:.0f}ms  失败 {r['stacks']['jev']['fails']}", flush=True)

    # 合并到已有记录：r 是新建的，直接赋值会把之前测过的 stacks 一起抹掉。
    if key in results:
        merged = dict(results[key]); merged.setdefault("stacks", {})
        merged["stacks"].update(r.pop("stacks"))
        merged.update(r); merged["stacks"] = merged["stacks"]
        results[key] = merged
    else:
        results[key] = r
    (OUT / f"usecases{TAG}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(flush=True)

print("=== 汇总（macro）===")
for s in sorted({k for r in results.values() for k in r["stacks"]}):
    vals = [r["stacks"][s]["acc"] for r in results.values() if s in r["stacks"]]
    if vals: print(f"  {s:8s} {np.mean(vals)*100:.2f}%")
