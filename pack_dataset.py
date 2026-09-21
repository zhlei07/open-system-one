#!/usr/bin/env python3
"""打包可发布的对比数据集：逐条预测，不含数据集正文。

发的是什么：每条任务的**切分索引 + 正确答案 + 六条链路各自的预测**。
不发文本——源数据集各有各的许可，`data.py` 用固定种子能逐条重建。

为什么逐条预测才是这份数据的价值：有了它，第三方不用花钱调 Jev，
就能做显著性检验、找出各方案分歧的具体样本、验证我们的汇总数字。
"""
import glob, json
from pathlib import Path
import numpy as np
import fly_head as fh
from data import CACHE, build

DS = ["sst2", "ag_news", "emotion", "banking77"]
OUT = Path("hf/dataset"); OUT.mkdir(parents=True, exist_ok=True)
b = build(); ev = b["splits"]["eval"]


def jev(cond):
    p = Path(f"out/jev_preds_{cond}.jsonl")
    if not p.exists():
        return {}
    return {j["i"]: j for j in map(json.loads, p.open())}


def cross_pred(rich):
    """交叉编码器逐条预测：从存盘的 logit 矩阵直接取 argmax。"""
    out = {}
    for ds in DS:
        hits = [h for h in glob.glob(f"cache/cross_{ds}_k*.npz") + glob.glob(f"cache/xc_base_{ds}_k*_t1_n2500*.npz")
                if ("_rich" in h) == rich]
        if not hits:
            continue
        d = np.load(hits[0]); lg = d["logits"]
        if lg.ndim == 3:
            lg = np.where(np.isfinite(lg), lg, -1e9).mean(-1)
        pred = np.where(np.isfinite(lg), lg, -1e9).argmax(1)
        for row, i in enumerate(d["idx"]):
            out[int(i)] = int(pred[row])
    return out


def bi_pred():
    """双塔 few-shot（2000 标注）逐条预测，用发布的 8 个权重。"""
    z = np.load(CACHE / "emb/mbert-embed.npz")
    heads = json.load(open("out/heads.json"))
    w = np.array([heads["mbert-embed"]["weights_few"][k] for k in fh.FEW_FEATURES], np.float32)
    out = {}
    for ds in DS:
        fit_rows = [r for r in b["splits"]["fit"] if r["dataset"] == ds]
        y = np.array([r["label"] for r in fit_rows])[:1500]
        bank = fh.FeatureBank(b["options"][ds], z[f"fit_{ds}"][:1500], y, z[f"desc_{ds}"])
        idx = [i for i, r in enumerate(ev) if r["dataset"] == ds]
        pred = fh.scores(bank.few(z["eval"][idx]), w).argmax(1)
        for row, i in enumerate(idx):
            out[i] = int(pred[row])
    return out


jb, jr = jev("bare"), jev("enriched")
cb, cr, bi = cross_pred(False), cross_pred(True), bi_pred()

with (OUT / "predictions.jsonl").open("w") as f:
    for i, r in enumerate(ev):
        f.write(json.dumps({
            "i": i, "dataset": r["dataset"], "n_options": r["n_options"], "gold": r["label"],
            "jev_bare": jb.get(i, {}).get("pred"), "jev_bare_probs": jb.get(i, {}).get("probs"),
            "jev_enriched": jr.get(i, {}).get("pred"), "jev_enriched_probs": jr.get(i, {}).get("probs"),
            "cross_bare": cb.get(i), "cross_enriched": cr.get(i),
            "bi_fewshot": bi.get(i),
        }) + "\n")

(OUT / "options.json").write_text(json.dumps(b["options"], ensure_ascii=False, indent=1))
(OUT / "split_indices.json").write_text(json.dumps(
    {s: [{"dataset": t["dataset"], "label": t["label"]} for t in b["splits"][s]]
     for s in ("fit", "dev", "eval")}, ensure_ascii=False))

n = sum(1 for _ in (OUT / "predictions.jsonl").open())
print(f"predictions.jsonl  {n} 行  {(OUT/'predictions.jsonl').stat().st_size/1024:.0f} KB")
for k in ["jev_bare", "jev_enriched", "cross_bare", "cross_enriched", "bi_fewshot"]:
    got = [json.loads(l) for l in (OUT / "predictions.jsonl").open()]
    ok = [g for g in got if g[k] is not None]
    acc = sum(g[k] == g["gold"] for g in ok) / max(1, len(ok))
    print(f"  {k:16s} 覆盖 {len(ok):5d}/{n}  准确率 {acc*100:.1f}%")
