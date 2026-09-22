#!/usr/bin/env python3
"""BANKING77：同一份前 12 名里，独立打分 vs 两两对比。

对照已经在 cache/cross_banking77_k20.npz 里：
独立打分 R@1 = 68.8%，R@2 = 79.7%，R@12 = 94.9%。
Jev 是 77.8%。差的那一段几乎全是"第二名其实是对的"。

本脚本不训练、不换骨干。候选是双塔的前 12 名（前 20 名的前缀，
和那份缓存是同一次排序）。两种打分用同一个交叉编码器：

- 独立："This example is about {label}."
- 对比："This text is about {A} rather than {B}."，两个方向都跑，
  一个选项的分数是它赢过其余选项的蕴含 logit 之和。

只看前 200 条。对比不比独立高 3 个点，就停，不跑满 2500。

    python pairwise_probe.py --n 200
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import CACHE, build

REPO = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
INDEP = "This example is about {}."
PAIR = "This text is about {} rather than {}."
K = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=K)
    a = ap.parse_args()

    b = build()
    opts = b["options"]["banking77"]
    ev = b["splits"]["eval"]
    d = np.load(CACHE / "cross_banking77_k20.npz")
    # cand 按双塔分数从高到低排，前 k 名就是要对比的那一组。
    cand = d["cand"][:a.n, :a.k]
    y = d["y"][:a.n]
    idx = d["idx"][:a.n]
    cached = np.where(np.isfinite(d["logits"][:a.n]), d["logits"][:a.n], -1e9)
    ceiling = float((cand == y[:, None]).any(1).mean())

    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()
    torch.set_grad_enabled(False)
    ent = next(i for i, lab in model.config.id2label.items() if "entail" in lab.lower())
    print(f"device=cpu  threads={torch.get_num_threads()}  entail_idx={ent}", flush=True)
    print(f"n={a.n}  k={a.k}  候选天花板 R@{a.k}={ceiling*100:.1f}%  "
          f"缓存独立打分（前20）={(cached.argmax(1)==y).mean()*100:.1f}%", flush=True)

    ind_ok = pair_ok = flip_ok = 0
    both_right = ind_only = pair_only = 0
    lat = []
    t0 = time.perf_counter()
    k = a.k
    for row in range(a.n):
        c = cand[row]
        text = ev[int(idx[row])]["text"]
        labels = [opts[int(j)] for j in c]
        prem, hyp = [], []
        for lab in labels:
            prem.append(text)
            hyp.append(INDEP.format(lab))
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                prem.append(text)
                hyp.append(PAIR.format(labels[i], labels[j]))
        t1 = time.perf_counter()
        enc = tok(prem, hyp, return_tensors="pt", padding=True,
                  truncation=True, max_length=128)
        logit = model(**enc).logits[:, ent].float().numpy()
        lat.append((time.perf_counter() - t1) * 1000)

        s = logit[:k]
        win = np.zeros((k, k), np.float32)
        win[np.nonzero(np.ones((k, k)) - np.eye(k))] = logit[k:]
        pick_i = int(s.argmax())
        pick_p = int(win.sum(1).argmax())
        order = np.argsort(-s)
        a0, b0 = int(order[0]), int(order[1])
        pick_f = b0 if win[b0, a0] > win[a0, b0] else a0
        gold = int(y[row])
        ok_i = int(c[pick_i] == gold)
        ok_p = int(c[pick_p] == gold)
        ok_f = int(c[pick_f] == gold)
        ind_ok += ok_i
        pair_ok += ok_p
        flip_ok += ok_f
        both_right += ok_i and ok_p
        ind_only += ok_i and not ok_p
        pair_only += ok_p and not ok_i
        if (row + 1) % 25 == 0:
            done = row + 1
            print(f"  {done}/{a.n}  独立 {ind_ok/done*100:.1f}%  "
                  f"两两 {pair_ok/done*100:.1f}%  "
                  f"只翻前两名 {flip_ok/done*100:.1f}%  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    n = a.n
    lat.sort()
    res = {
        "dataset": "banking77", "n": n, "k": k, "device": "cpu",
        "threads": torch.get_num_threads(),
        "ceiling_recall": ceiling,
        "acc_cached_independent_top20": float((cached.argmax(1) == y).mean()),
        "acc_independent_top12": ind_ok / n,
        "acc_pairwise_top12": pair_ok / n,
        "acc_flip_top2": flip_ok / n,
        "delta_pairwise_pp": (pair_ok - ind_ok) / n * 100,
        "delta_flip_pp": (flip_ok - ind_ok) / n * 100,
        "ind_only": ind_only, "pair_only": pair_only, "both_right": both_right,
        "lat_p50_ms": round(lat[len(lat) // 2], 1),
        "wall_s": round(time.perf_counter() - t0, 1),
        "hypothesis_indep": INDEP, "hypothesis_pair": PAIR,
    }
    out = Path("out") / f"pairwise_banking77_k{k}_n{n}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    if res["delta_pairwise_pp"] < 3 and res["delta_flip_pp"] < 3:
        print("停：两种对比都没有比独立打分高 3 个点。", flush=True)
    else:
        print("过线：可以跑满 2500。", flush=True)


if __name__ == "__main__":
    main()
