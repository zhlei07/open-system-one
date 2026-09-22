#!/usr/bin/env python3
"""CLINC-150：验证"按选项数门控的列中心化"这个假说的第二个证据。

BANKING77（77 选项）上列中心化是 +2.3pp，SST-2/AG News/Emotion（2/4/6 选项）上
是 -0.2/+0.0/-3.3。假说是：修的是 hubness，选项越多越值钱。
如果假说为真，150 个选项的 CLINC 上应该看到**不小于** BANKING77 的增益。
如果只有噪声，那这个门控就是拟合了 4 个任务里的 1 个，趁早知道。

同时把 #2 一起测了：裁剪器 static(8M) vs mbert-embed(149M)。
两件事共用一次交叉编码器打分——候选取两个裁剪器 top-20 的并集，各自再掩码还原。

列均值只从 **CLINC train split 的无标签样本**估计，不碰 eval。
这样它是归纳式的：线下估一次，线上逐条推理，不需要看到一批查询。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

K = 20
N_POOL = 500
OUT = Path("out/clinc_center.json")


def main():
    from datasets import load_dataset
    from encoders import STEncoder, StaticEncoder
    from ensemble import TEMPLATE, _Cross
    from usecases import SEED, _humanize, build_usecases

    u = build_usecases(500)["routing"]
    opts = u.options
    ev_texts, ev_y = u.texts[100:], np.array(u.gold[100:])

    # 无标签池：train split 的域内话术。不取 eval 的任何一条。
    clinc = load_dataset("clinc/clinc_oos", "plus", split="train")
    oos_id = clinc.features["intent"].names.index("oos")
    pool = clinc.filter(lambda r: r["intent"] != oos_id).shuffle(seed=SEED + 77)
    pool_texts = [r["text"] for r in pool.select(range(N_POOL))]
    assert not (set(pool_texts) & set(ev_texts)), "池子和 eval 有重叠"
    print(f"CLINC {len(opts)} 选项 | eval {len(ev_y)} 条 | 无标签池 {len(pool_texts)} 条(train)")

    encs = {"static": StaticEncoder().load(),
            "mbert": STEncoder("mbert-embed", "nomic-ai/modernbert-embed-base", 149.0,
                               prefix="search_query: ").load()}
    all_texts = pool_texts + ev_texts
    cands = {}
    for name, e in encs.items():
        sim = e.encode(all_texts) @ e.encode(opts).T
        cands[name] = np.argsort(-sim, axis=1)[:, :K]
        print(f"  裁剪器 {name:7s} 已算候选")

    cross = _Cross()
    n, m = len(all_texts), len(opts)
    logits = np.full((n, m), np.nan, np.float32)
    lat = []
    t0 = time.perf_counter()
    for i, text in enumerate(all_texts):
        uni = np.union1d(cands["static"][i], cands["mbert"][i])
        hyps = [TEMPLATE.format(opts[j]) for j in uni]
        t1 = time.perf_counter()
        enc = cross.tok([text] * len(uni), hyps, return_tensors="pt",
                        padding=True, truncation=True, max_length=256)
        logits[i, uni] = cross.model(**enc).logits[:, cross.ent].float().numpy()
        lat.append((time.perf_counter() - t1) * 1000)
        if (i + 1) % 100 == 0:
            print(f"  打分 {i+1}/{n}  ({time.perf_counter()-t0:.0f}s, 并集均值 "
                  f"{np.isfinite(logits[:i+1]).sum(1).mean():.1f} 个选项)", flush=True)

    np.savez_compressed("cache/clinc_cross_union.npz", logits=logits,
                        static=cands["static"], mbert=cands["mbert"],
                        y=ev_y, n_pool=N_POOL)

    res = {"n_eval": len(ev_y), "n_pool": N_POOL, "k": K, "n_options": m,
           "lat_p50_ms": round(float(np.median(lat)), 1), "arms": {}}
    print(f"\n{'裁剪器':<9}{'R@20':>8}{'基线':>9}{'+列中心化':>11}{'Δ':>9}")
    print("-" * 47)
    for name in ("static", "mbert"):
        c = cands[name]
        mask = np.zeros((n, m), bool)
        np.put_along_axis(mask, c, True, axis=1)
        s = np.where(mask, logits, np.nan)
        pool_s, ev_s = s[:N_POOL], s[N_POOL:]
        col = np.nanmean(pool_s, axis=0)
        col = np.where(np.isnan(col), np.nanmean(pool_s), col)   # 池中未出现的列退回全局均值
        base = float((np.where(mask[N_POOL:], ev_s, -1e9).argmax(1) == ev_y).mean())
        cent = float((np.where(mask[N_POOL:], ev_s - col[None, :], -1e9).argmax(1) == ev_y).mean())
        rec = float((c[N_POOL:] == ev_y[:, None]).any(1).mean())
        res["arms"][name] = {"recall_at_20": rec, "acc": base, "acc_centered": cent,
                             "delta_pp": (cent - base) * 100,
                             "col_seen": int((~np.isnan(np.nanmean(pool_s, axis=0))).sum())}
        print(f"{name:<9}{rec*100:7.1f}%{base*100:8.1f}%{cent*100:10.1f}%{(cent-base)*100:+8.2f}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n-> {OUT}  (p50={res['lat_p50_ms']}ms)")


if __name__ == "__main__":
    main()
