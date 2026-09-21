"""双塔零样本：裸标签名 vs 丰富描述。几秒钟的事，因为只要重编码 89 条描述。"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import fly_head as fh
from data import CACHE, build
from descriptions import describe

DS = ["sst2", "ag_news", "emotion", "banking77"]
b = build(); z = np.load(CACHE / "emb/mbert-embed.npz")
m = SentenceTransformer("nomic-ai/modernbert-embed-base", device="cpu").eval()
ev = b["splits"]["eval"]
idx = {d: [i for i, r in enumerate(ev) if r["dataset"] == d] for d in DS}
y = {d: np.array([ev[i]["label"] for i in idx[d]]) for d in DS}
w = np.zeros(len(fh.ZERO_FEATURES), np.float32); w[0] = w[3] = 1.0   # cos + 列中心化

out = {}
for d in DS:
    opts = b["options"][d]
    rich = describe(d, opts, True)
    # 丰富描述本身是完整句子，不再套模板；三层模板维度用同一份描述填满以复用 FeatureBank
    e = m.encode(["search_query: " + t for t in rich], normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    desc_rich = np.stack([e, e, e])
    q = z["eval"][idx[d]]
    a_bare = fh.accuracy(fh.FeatureBank(opts, None, None, z[f"desc_{d}"]).zero(q), w, y[d])
    a_rich = fh.accuracy(fh.FeatureBank(opts, None, None, desc_rich).zero(q), w, y[d])
    out[d] = {"bare": a_bare, "rich": a_rich}
    print(f"{d:11s} 裸 {a_bare*100:5.1f}%   丰富 {a_rich*100:5.1f}%   {(a_rich-a_bare)*100:+5.1f}")
mb = np.mean([out[d]["bare"] for d in DS]); mr = np.mean([out[d]["rich"] for d in DS])
print(f"{'宏平均':11s} 裸 {mb*100:5.1f}%   丰富 {mr*100:5.1f}%   {(mr-mb)*100:+5.1f}")
json.dump(out, open("out/fair_bi.json", "w"), indent=1)
