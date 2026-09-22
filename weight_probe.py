#!/usr/bin/env python3
"""把融合权重规则的选择问题变成一次离线实验。

背景：退回单后端的规则在 5 个用例上触发 4 次、选错 2 次，融合的宏平均因此
掉到 70.15%，低于单用 Laya 的 71.05%。要判断换哪条规则更好，不该靠合成数据
猜——但也不该每换一条规则就把两个模型重跑一遍。

所以这里**只跑一次模型**，把 cross / laya 在校准折和 eval 上的概率全部落盘，
之后任意规则都在 numpy 里秒级评估。产物 cache/weight_probe.npz。

    python weight_probe.py          # 跑模型并落盘，然后评估所有规则
    python weight_probe.py --eval   # 只用已落盘的数据评估（不碰模型）
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from colcenter import gated
from ensemble import PREFILTERS, Ensemble, _BACKENDS, _temper
from usecases import build_usecases

N, N_CAL = 500, 100
CACHE = Path("cache/weight_probe.npz")
GRID = np.arange(0, 1.0001, 0.05)


def collect():
    cases = build_usecases(N)
    pool = {b: _BACKENDS[b]() for b in ("cross", "laya")}
    store = {}
    for key, u in cases.items():
        e = Ensemble(backends={b: pool[b] for b in ("cross", "laya")})
        f = Path("cache") / f"colcenter_cross_{key}.json"
        if f.exists():
            e.load_center(f, u.options)
            print(f"  {key}: 已加载列均值", flush=True)
        raw = {b: np.zeros((len(u), u.n_options)) for b in ("cross", "laya")}
        t0 = time.perf_counter()
        for i in range(len(u)):
            r = e._raw(u.texts[i], u.options, None)
            for b in raw:
                raw[b][i] = r[b]
            if i and i % 200 == 0:
                print(f"    {key} {i}/{len(u)} ({time.perf_counter()-t0:.0f}s)", flush=True)
        store[f"{key}__cross"] = raw["cross"]
        store[f"{key}__laya"] = raw["laya"]
        store[f"{key}__y"] = np.array(u.gold)
        print(f"  {key}: 完成 {time.perf_counter()-t0:.0f}s", flush=True)
    np.savez_compressed(CACHE, **store)
    return store


# ---------------- 规则 ----------------

def rule_argmax(accs, grid, se):
    return float(grid[int(np.argmax(accs))])


def rule_snap_single(accs, grid, se):
    """现状：混合不显著胜过最好的单后端就退回单后端。"""
    best = accs.max()
    singles = {0.0: accs[0], 1.0: accs[-1]}
    top = max(singles, key=singles.get)
    return top if singles[top] >= best - se else float(grid[int(np.argmax(accs))])


def rule_plateau_mid(accs, grid, se):
    """与最优点不可分辨的那段连续平台的中点。"""
    ok = accs >= accs.max() - se
    i = int(np.argmax(accs)); lo = hi = i
    while lo > 0 and ok[lo - 1]: lo -= 1
    while hi < len(grid) - 1 and ok[hi + 1]: hi += 1
    return float((grid[lo] + grid[hi]) / 2)


def rule_shrink_half(accs, grid, se):
    """argmax 与 0.5 的中点：固定幅度地朝"不拟合"收缩。"""
    return float((grid[int(np.argmax(accs))] + 0.5) / 2)


def rule_fixed_half(accs, grid, se):
    """不拟合，永远 50/50。"""
    return 0.5


RULES = {"argmax（旧旧）": rule_argmax, "退回单后端（现状）": rule_snap_single,
         "平台中点": rule_plateau_mid, "朝0.5收缩一半": rule_shrink_half,
         "固定50/50": rule_fixed_half}


def evaluate(store):
    keys = sorted({k.split("__")[0] for k in store})
    rows = {r: [] for r in RULES}
    rows["纯cross"], rows["纯laya"], rows["事后最优w"] = [], [], []
    print(f"\n{'用例':<12}" + "".join(f"{r:>16}" for r in
          list(RULES) + ["纯cross", "纯laya", "事后最优w"]))
    print("-" * (12 + 16 * 8))
    for key in keys:
        A, C = store[f"{key}__cross"], store[f"{key}__laya"]
        y = store[f"{key}__y"]
        cal, evl = slice(0, N_CAL), slice(N_CAL, None)
        # 温度按各自在校准折上的 NLL 定，和 Ensemble.calibrate 一致
        T = {}
        for name, P in (("cross", A), ("laya", C)):
            tg = np.geomspace(0.2, 20, 70)
            nll = [-np.log(np.clip(_temper(P[cal], t)[np.arange(N_CAL), y[cal]], 1e-12, None)).mean()
                   for t in tg]
            T[name] = _temper(P, float(tg[int(np.argmin(nll))]))
        def acc_at(w, sl):
            return float(((w * T["laya"][sl] + (1 - w) * T["cross"][sl]).argmax(1) == y[sl]).mean())
        cal_accs = np.array([acc_at(w, cal) for w in GRID])
        ev_accs = np.array([acc_at(w, evl) for w in GRID])
        best = cal_accs.max()
        se = float(np.sqrt(max(best * (1 - best), 1e-9) / N_CAL))
        line = f"{key:<12}"
        for rname, fn in RULES.items():
            w = fn(cal_accs, GRID, se)
            a = acc_at(w, evl)
            rows[rname].append(a)
            line += f"{a*100:11.1f}(w{w:.2f})"
        for nm, v in (("纯cross", acc_at(0.0, evl)), ("纯laya", acc_at(1.0, evl)),
                      ("事后最优w", ev_accs.max())):
            rows[nm].append(v); line += f"{v*100:16.1f}"
        print(line)
    print("-" * (12 + 16 * 8))
    print(f"{'宏平均':<12}" + "".join(f"{np.mean(rows[r])*100:16.2f}" for r in
          list(RULES) + ["纯cross", "纯laya", "事后最优w"]))
    Path("out/weight_probe.json").write_text(json.dumps(
        {r: [round(v, 4) for v in rows[r]] for r in rows}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    evaluate(dict(np.load(CACHE)) if a.eval else collect())
