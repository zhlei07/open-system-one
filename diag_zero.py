"""诊断零样本还有多少可挖的空间。

三个问题：
1. nomic 的 prefix 我用错了吗？（query 和 document 该用不同前缀）
2. 权重搜索的天花板在哪？（留一迁移 vs 每个数据集单独搜的"作弊上界"）
3. 差距集中在哪个任务？
"""
import json
import numpy as np
from data import CACHE, build
import fly_head as fh

DATASETS = ["sst2", "ag_news", "emotion", "banking77"]
b = build()
z = np.load(CACHE / "emb/mbert-embed.npz")
jev = json.load(open("out/jev.json"))

def rows(split, ds):
    return [t for t in b["splits"][split] if t["dataset"] == ds]

# ---- 用正确的非对称 prefix 重编码选项描述 ----
# nomic-embed 的约定：查询用 search_query:，被检索的文档用 search_document:。
# 原来两边都用 search_query:，这在检索任务里是已知会掉分的用法。
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("nomic-ai/modernbert-embed-base", device="cpu").eval()

def enc(texts, prefix):
    return m.encode([prefix + t for t in texts], normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

desc_doc = {}
for ds in DATASETS:
    opts = b["options"][ds]
    desc_doc[ds] = np.stack([enc([t.format(o) for o in opts], "search_document: ")
                             for t in fh.TEMPLATES])

def build_banks(desc_src):
    out = {}
    for ds in DATASETS:
        d = desc_src[ds] if isinstance(desc_src, dict) else z[f"desc_{ds}"]
        out[ds] = fh.FeatureBank(b["options"][ds], None, None, d)
    return out

ev = b["splits"]["eval"]
ev_idx = {ds: [i for i, r in enumerate(ev) if r["dataset"] == ds] for ds in DATASETS}
ev_y = np.array([r["label"] for r in ev])
fit_y = {ds: np.array([r["label"] for r in rows("fit", ds)]) for ds in DATASETS}

print(f"{'方案':34s} " + " ".join(f"{d:>10s}" for d in DATASETS) + "   总体")

def evaluate(tag, banks, mode):
    """mode: cos / lodo / oracle"""
    accs, ns = [], []
    per = {}
    for ds in DATASETS:
        q = z["eval"][ev_idx[ds]]
        y = ev_y[ev_idx[ds]]
        f = banks[ds].zero(q)
        if mode == "cos":
            w = np.zeros(len(fh.ZERO_FEATURES), np.float32); w[0] = 1.0
        elif mode == "lodo":
            src = {d: (banks[d].zero(z[f"fit_{d}"]), fit_y[d]) for d in DATASETS if d != ds}
            w, _ = fh.search_weights(src, len(fh.ZERO_FEATURES), generations=60)
        else:  # oracle：直接在目标数据集的 fit 上搜 —— 这是作弊，只用来量天花板
            w, _ = fh.search_weights({ds: (banks[ds].zero(z[f"fit_{ds}"]), fit_y[ds])},
                                     len(fh.ZERO_FEATURES), generations=60)
        a = fh.accuracy(f, w, y)
        per[ds] = a; accs.append(a); ns.append(len(y))
    overall = float(np.average(accs, weights=ns))
    print(f"{tag:34s} " + " ".join(f"{per[d]*100:9.1f}%" for d in DATASETS) + f"  {overall*100:6.1f}%")
    return overall

banks_q = build_banks(None)        # 原方案：选项也用 search_query:
banks_d = build_banks(desc_doc)    # 修正：选项用 search_document:

r = {}
r["cos_q"]      = evaluate("原 prefix, 纯余弦", banks_q, "cos")
r["lodo_q"]     = evaluate("原 prefix, LODO 搜权重（报告值）", banks_q, "lodo")
r["cos_d"]      = evaluate("修正 prefix, 纯余弦", banks_d, "cos")
r["lodo_d"]     = evaluate("修正 prefix, LODO 搜权重", banks_d, "lodo")
r["oracle_d"]   = evaluate("修正 prefix, 每集单独搜（作弊上界）", banks_d, "oracle")
print(f"{'typesafe/jev':34s} " + " ".join(f"{jev['per_dataset'][d]['acc']*100:9.1f}%" for d in DATASETS)
      + f"  {jev['overall_acc']*100:6.1f}%")
json.dump(r, open("out/diag_zero.json", "w"), indent=1)
