#!/usr/bin/env python3
"""用今天做的交叉编码器给候选素材做分类和排序。

为什么值得：深读一个 Reddit 帖子要 ~1.8 万 token。30 个就是 54 万。
先在本地零样本分类，只精读高价值的，成本降一个数量级。

这就是「结构化判断模型」的典型用法：短文本 -> 给定选项里选一个 + 置信度。
和 Jev 的 API 形状一致，只是跑在自己机器上。
"""
import json, sys, time
import numpy as np, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
LABELS = {
 "pain":    "The writer complains that health or fitness data is collected but never interpreted or explained.",
 "rivals":  "The writer recommends or compares specific health apps or smart scale brands, or discusses their prices.",
 "doubt":   "The writer argues about whether body fat or body composition measurements are accurate or trustworthy.",
 "buy":     "The writer is asking what product to buy or where to buy it.",
 "price":   "The writer complains that app subscriptions cost too much or objects to paying monthly.",
 "noise":   "The text is about something else entirely.",
}
ITEMS = json.load(open(sys.argv[1]))

tok = AutoTokenizer.from_pretrained(REPO)
m = AutoModelForSequenceClassification.from_pretrained(REPO).eval()
torch.set_grad_enabled(False)
ENT = next(i for i, l in m.config.id2label.items() if "entail" in l.lower())
keys, hyps = list(LABELS), list(LABELS.values())

t0 = time.perf_counter(); out = []
for it in ITEMS:
    text = f"{it['title']}. {it.get('snippet','')}"[:700]
    enc = tok([text]*len(hyps), hyps, return_tensors="pt", padding=True, truncation=True, max_length=256)
    s = m(**enc).logits[:, ENT].float().numpy()
    p = np.exp(s - s.max()); p /= p.sum()
    order = np.argsort(-p)
    out.append({**it, "label": keys[order[0]], "conf": float(p[order[0]]),
                "second": keys[order[1]], "scores": {k: round(float(v),3) for k,v in zip(keys,p)}})
dt = time.perf_counter()-t0

PRIORITY = {"pain":3.0, "rivals":2.5, "price":2.0, "doubt":1.8, "buy":1.5, "noise":0.0}
for o in out: o["value"] = round(PRIORITY[o["label"]] * o["conf"], 3)
out.sort(key=lambda x: -x["value"])

print(f"{len(ITEMS)} 条，{dt:.1f}s，{dt/len(ITEMS)*1000:.0f}ms/条\n")
print(f"{'价值':>5} {'类别':7} {'置信':>5}  {'社区':22} 标题")
print("-"*118)
for o in out:
    print(f"{o['value']:5.2f} {o['label']:7} {o['conf']*100:4.0f}%  {o['sub']:22} {o['title'][:60]}")
json.dump(out, open("out/triage.json","w"), ensure_ascii=False, indent=1)
