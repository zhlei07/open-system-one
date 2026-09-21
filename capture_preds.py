#!/usr/bin/env python3
"""重跑 Jev 并**保存逐条预测**，用于发布可复核的对比数据集。

之前只存了汇总准确率，这让第三方无法做显著性检验、也无法看具体分歧案例。
逐条预测才是这份数据集真正的价值——别人不用花钱调 Jev 就能分析它在哪些
样本上赢、在哪些上输。
"""
import json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from data import build
from descriptions import describe
from jev_adapter import JevClient

DS = ["sst2", "ag_news", "emotion", "banking77"]
CONC = 8
b = build(); ev = b["splits"]["eval"]
rich = {d: dict(zip(b["options"][d], describe(d, b["options"][d], True))) for d in DS}

for cond in ["bare", "enriched"]:
    dest = Path(f"out/jev_preds_{cond}.jsonl")
    if dest.exists():
        print(f"{cond} 已存在，跳过"); continue
    clients = [JevClient() for _ in range(CONC)]
    res = [None] * len(ev)

    def work(a):
        i, r = a
        try:
            p, probs = clients[i % CONC].predict(
                r["text"], b["options"][r["dataset"]],
                criteria=rich[r["dataset"]] if cond == "enriched" else None)
            res[i] = (p, [round(x, 4) for x in probs])
        except Exception:
            res[i] = None

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        for n, _ in enumerate(ex.map(work, enumerate(ev))):
            if n and n % 2500 == 0:
                print(f"  {cond} {n}/{len(ev)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    with dest.open("w") as f:
        for i, (r, row) in enumerate(zip(res, ev)):
            f.write(json.dumps({"i": i, "dataset": row["dataset"], "gold": row["label"],
                                "pred": r[0] if r else None,
                                "probs": r[1] if r else None}) + "\n")
    ok = [r for r in res if r]
    acc = sum(r[0] == row["label"] for r, row in zip(res, ev) if r) / len(ok)
    tok = sum(c.total_input_tokens for c in clients)
    print(f"[{cond}] {acc*100:.1f}%  成功 {len(ok)}/{len(ev)}  "
          f"${tok/1e6*0.042:.4f}  {time.perf_counter()-t0:.0f}s", flush=True)
