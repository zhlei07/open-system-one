#!/usr/bin/env python3
"""四个基准数据集的最终测试：全新留出集，一次性。

和主基准完全同一套口径，唯一区别是数据换成 cache/holdout.json（idx[5500:]，
fit/dev/eval 都没碰过，且已按文本去重）。

两个在 eval 上选出来的超参，这里**原样应用、不重新选**：
  - 门控阈值 GATE=20（选项数 >= 20 才开列中心化）
  - 两份列均值，均在 fit 切分上拟合，从未见过 eval，更没见过留出集

Jev 不在此脚本内。本次没有 Jev 凭证，Jev 那一列必须留空并标注
"未在此样本上测量"——绝不把另一批数据上的旧数字并排放。

    python final_test.py            # 全部四个
    python final_test.py banking77
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from colcenter import GATE, ColumnCenter, gated
from data import CACHE, build

REPO = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
TEMPLATE = "This example is about {}."      # 与已公布的 cross_*.npz 逐位一致
K = 20


def main():
    want = sys.argv[1:] or ["sst2", "ag_news", "emotion", "banking77"]
    h = json.loads((CACHE / "holdout.json").read_text())
    b = build()
    z = np.load(CACHE / "emb/mbert-embed.npz")

    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()
    torch.set_grad_enabled(False)
    ent = next(i for i, l in model.config.id2label.items() if "entail" in l.lower())
    print(f"{REPO}  entail_idx={ent}  threads={torch.get_num_threads()}  门控 GATE={GATE}\n")

    res = {}
    for ds in want:
        rows = [r for r in h["rows"] if r["dataset"] == ds]
        opts = h["options"][ds]
        assert opts == b["options"][ds], f"{ds} 选项文字与主基准不一致"
        texts = [r["text"] for r in rows]
        y = np.array([r["label"] for r in rows])
        on = gated(len(opts))
        print(f"=== {ds}  {len(opts)} 选项  {len(y)} 条（全新留出）  列中心化={'开' if on else '关'}")

        # 候选裁剪：只有选项多于 K 才裁，和主基准一致
        if len(opts) > K:
            from encoders import STEncoder
            enc = STEncoder("mbert-embed", "nomic-ai/modernbert-embed-base", 149.0,
                            prefix="search_query: ").load()
            t0 = enc.encode(texts) @ z[f"desc_{ds}"][0].T
            cc_bi = ColumnCenter.load(CACHE / f"colcenter_bi_{ds}.json")   # fit 切分拟合
            cand = np.argsort(-(t0 + cc_bi.transform(t0)), axis=1)[:, :K]
            recall = float((cand == y[:, None]).any(1).mean())
            print(f"  裁剪 top-{K}  recall={recall*100:.2f}%（裁剪器列均值来自 fit 切分）")
        else:
            cand = np.tile(np.arange(len(opts)), (len(y), 1))
            recall = 1.0

        cc = ColumnCenter.load(CACHE / f"colcenter_cross_{ds}.json") if on else None
        lg = np.full((len(y), len(opts)), np.nan, np.float32)
        lat, t_start = [], time.perf_counter()
        for r, text in enumerate(texts):
            c = cand[r]
            t1 = time.perf_counter()
            e = tok([text] * len(c), [TEMPLATE.format(opts[int(j)]) for j in c],
                    return_tensors="pt", padding=True, truncation=True, max_length=256)
            lg[r, c] = model(**e).logits[:, ent].float().numpy()
            lat.append((time.perf_counter() - t1) * 1000)
            if r and r % 500 == 0:
                a = (np.where(np.isfinite(lg[:r]), lg[:r], -1e9).argmax(1) == y[:r]).mean()
                print(f"    {r}/{len(y)} acc={a*100:.1f}% ({time.perf_counter()-t_start:.0f}s)", flush=True)

        mask = np.isfinite(lg)
        plain = float((np.where(mask, lg, -1e9).argmax(1) == y).mean())
        gatedacc = plain if cc is None else float(
            (np.where(mask, lg - cc.mean, -1e9).argmax(1) == y).mean())
        lat.sort()
        res[ds] = {"n": int(len(y)), "n_options": len(opts), "recall_at_k": recall,
                   "centering_on": bool(on),
                   "acc_no_centering": plain, "acc_gated": gatedacc,
                   "delta_pp": (gatedacc - plain) * 100,
                   "lat_p50_ms": round(lat[len(lat) // 2], 1),
                   "wall_s": round(time.perf_counter() - t_start, 1)}
        print(f"  无列中心化 {plain*100:.2f}%   门控后 {gatedacc*100:.2f}%   "
              f"({(gatedacc-plain)*100:+.2f} pp)   p50={res[ds]['lat_p50_ms']}ms\n", flush=True)
        Path("out/final_test.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))

    if len(res) == 4:
        print(f"宏平均  无列中心化 {np.mean([v['acc_no_centering'] for v in res.values()])*100:.2f}%"
              f"   门控后 {np.mean([v['acc_gated'] for v in res.values()])*100:.2f}%")


if __name__ == "__main__":
    main()
