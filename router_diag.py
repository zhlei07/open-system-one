#!/usr/bin/env python3
"""路由失败的诊断：置信度到底能不能指示'谁该接这一条'。"""
import json
import numpy as np
from pathlib import Path
OUT = Path("out")

def load(fn):
    return {r["i"]: r for r in map(json.loads, (OUT/fn).read_text().splitlines()) if r.get("probs")}

jev, laya = load("jev_preds_enriched.jsonl"), load("laya_preds_english.jsonl")
common = sorted(set(jev) & set(laya))
ds   = np.array([jev[i]["dataset"] for i in common])
gold = np.array([jev[i]["gold"] for i in common])
cj = np.array([max(jev[i]["probs"]) for i in common])
cl = np.array([max(laya[i]["probs"]) for i in common])
pj = np.array([int(np.argmax(jev[i]["probs"])) for i in common])
pl = np.array([int(np.argmax(laya[i]["probs"])) for i in common])
okj, okl = pj == gold, pl == gold

print(f"{'任务':10s} {'Jev均信':>8s} {'Laya均信':>9s} {'Jev对':>7s} {'Laya对':>7s}")
for d in ["sst2","ag_news","emotion","banking77"]:
    m = ds == d
    print(f"{d:10s} {cj[m].mean():8.3f} {cl[m].mean():9.3f} {okj[m].mean()*100:6.1f}% {okl[m].mean()*100:6.1f}%")

print("\n=== 意见不一致时谁对 ===")
dis = pj != pl
print(f"分歧率 {dis.mean()*100:.1f}%  ({dis.sum()} 条)")
for d in ["sst2","ag_news","emotion","banking77"]:
    m = dis & (ds == d)
    both_wrong = (~okj[m] & ~okl[m]).mean()
    print(f"  {d:10s} 分歧 {m.sum():5d} 条 → Jev对 {okj[m].mean()*100:5.1f}%  "
          f"Laya对 {okl[m].mean()*100:5.1f}%  都错 {both_wrong*100:5.1f}%")

print("\n=== 置信度能不能分辨'自己这条对不对' (AUC-like) ===")
for nm, c, ok in [("Jev", cj, okj), ("Laya", cl, okl)]:
    for d in ["sst2","ag_news","emotion","banking77"]:
        m = ds == d
        cc, oo = c[m], ok[m]
        if oo.all() or not oo.any(): continue
        # 随机取一对(对的,错的)，对的那条置信度更高的概率
        auc = (cc[oo][:, None] > cc[~oo][None, :]).mean()
        print(f"  {nm:5s} {d:10s} AUC={auc:.3f}")
    print()

print("=== 分歧条目上，Laya 更自信时它对吗 ===")
m = dis
more_conf = cl[m] > cj[m]
print(f"  Laya 更自信的分歧条目 {more_conf.sum()} 条 → Laya 对 {okl[m][more_conf].mean()*100:.1f}%  Jev 对 {okj[m][more_conf].mean()*100:.1f}%")
print(f"  Jev  更自信的分歧条目 {(~more_conf).sum()} 条 → Laya 对 {okl[m][~more_conf].mean()*100:.1f}%  Jev 对 {okj[m][~more_conf].mean()*100:.1f}%")
