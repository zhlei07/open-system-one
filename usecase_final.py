#!/usr/bin/env python3
"""五个用例的最终测试：全新样本，一次性。

构造方式：build_usecases(900) 去掉 build_usecases(500) 里出现过的全部文本，
取剩下的前 400 条。**按文本做差集，不按下标切片**——guardrail 的组成随 n 变化
（half = n//2 之后整体重洗），切片会拿到一批混着旧样本的数据。

校准折仍用旧的前 100 条（只用来定温度）。那 100 条从来没进过任何 eval，
而且保持不变，这样"唯一变的是评测样本"这句话才成立。

融合权重不拟合（等权），门控阈值 GATE=20 原样应用，均不重新选型。
Jev 不在此脚本内：本次无凭证，那一列留空并标注"未在此样本上测量"。

    python usecase_final.py [每用例条数=400]
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ensemble import PREFILTERS, Ensemble, _BACKENDS
from usecases import build_usecases

N_FRESH = int(sys.argv[1]) if len(sys.argv) > 1 else 400
N_CAL = 100
OUT = Path("out")


def main():
    old = build_usecases(500)
    big = build_usecases(500 + N_FRESH * 2)      # 取宽一点，差集之后再截断
    pool = {b: _BACKENDS[b]() for b in ("cross", "laya")}
    print(f"后端已装载（laya 设备 {getattr(pool['laya'], 'device', '?')}）\n")

    results, preds = {}, {}
    for key, u_old in old.items():
        u_big = big[key]
        seen = set(u_old.texts)
        fresh = [(t, g) for t, g in zip(u_big.texts, u_big.gold) if t not in seen]
        # 有的数据集测试集本身就小（tweet_eval/offensive 扣掉旧 500 条只剩 353），
        # 凑不够就用实际条数，并把 n 如实记进结果——不为了对齐而硬凑。
        MIN = 200
        assert len(fresh) >= MIN, f"{key} 全新样本只有 {len(fresh)} 条，少于 {MIN}，不足以评测"
        if len(fresh) < N_FRESH:
            print(f"  注意：{key} 只有 {len(fresh)} 条全新样本（上限 {N_FRESH}），按实际条数评测")
        fresh = fresh[:N_FRESH]
        ftxt = [t for t, _ in fresh]
        fy = np.array([g for _, g in fresh])
        assert not (set(ftxt) & seen), f"{key} 全新样本与旧样本有重叠"
        maj = Counter(fy.tolist()).most_common(1)[0][1] / len(fy)
        print(f"=== {key}  {u_old.n_options} 选项  全新 {len(fy)} 条  "
              f"多数类基线 {maj*100:.1f}%（旧样本 {len(u_old.texts)} 条全部排除）", flush=True)

        cal = [(u_old.texts[i], u_old.options, u_old.gold[i]) for i in range(N_CAL)]
        r = {"n_options": u_old.n_options, "n_fresh": len(fy),
             "majority_baseline": round(maj, 4), "prefilters": dict(PREFILTERS),
             "jev": None, "jev_note": "未在此样本上测量（本次无 Jev 凭证）", "stacks": {}}
        pr = {"gold": fy.tolist(), "options": u_old.options, "pred": {}}

        def bench(e, name):
            ok, lat, got = 0, [], []
            for i, t in enumerate(ftxt):
                t0 = time.perf_counter()
                p = int(np.argmax(e.decide(t, u_old.options).probs))
                lat.append((time.perf_counter() - t0) * 1000)
                got.append(p); ok += int(p == fy[i])
            r["stacks"][name] = {"acc": ok / len(fy),
                                 "lat_p50_ms": round(float(np.percentile(lat, 50)), 1),
                                 "lat_p95_ms": round(float(np.percentile(lat, 95)), 1)}
            pr["pred"][name] = got
            print(f"  {name:8s} {ok/len(fy)*100:5.1f}%  p50={r['stacks'][name]['lat_p50_ms']:.0f}ms",
                  flush=True)

        for name in ("cross", "laya"):
            e = Ensemble(backends={name: pool[name]})
            f = Path("cache") / f"colcenter_cross_{key}.json"
            if f.exists():
                e.load_center(f, u_old.options)
                print(f"    {name}: 已加载列均值（在 CLINC train 上拟合）", flush=True)
            bench(e, name)

        fu = Ensemble(backends={k: pool[k] for k in ("cross", "laya")})
        f = Path("cache") / f"colcenter_cross_{key}.json"
        if f.exists():
            fu.load_center(f, u_old.options)
        r["fused_calibration"] = fu.calibrate(cal, verbose=False)
        bench(fu, "fused")

        results[key] = r
        preds[key] = pr
        (OUT / "usecases_final.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
        (OUT / "usecases_final_preds.json").write_text(json.dumps(preds, ensure_ascii=False))
        print(flush=True)

    print("=== 汇总（macro，全新样本）===")
    for s in ("cross", "laya", "fused"):
        v = [r["stacks"][s]["acc"] for r in results.values() if s in r["stacks"]]
        print(f"  {s:8s} {np.mean(v)*100:.2f}%")
    print("  jev      未在此样本上测量")


if __name__ == "__main__":
    main()
