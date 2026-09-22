#!/usr/bin/env python3
"""全新留出集：从来没有被任何选型、调参或评估碰过的数据。

为什么需要它。本轮有两个决定是**在 eval 上选出来的**：

  1. 门控阈值（选项数 >= 20）—— 看过四个任务的逐任务结果之后才定的。
  2. 融合权重规则 —— 拿 5 条规则在 5 个用例的 eval 上比宏平均，选了"固定 50/50"。

这是直接在测试集上选模型。拿同一批 eval 报最终成绩不成立，必须换数据。

data.py 把每个数据集的 train+test+validation 合池、用 SEED 打散，只取了前
5500 条（fit 2000 + dev 1000 + eval 2500）。idx[5500:] 一条都没用过。
这里用同一个种子重放同一次打散，取紧接着的 n 条。

纪律：data.py 里的 rng 是**跨数据集共享**的，每个数据集按 SPECS 顺序消费一次
shuffle。所以必须按同样的顺序重放，否则拿到的是另一份切分——那样构造出来的
"留出集"可能和 eval 相交，比不换数据还糟。

    python holdout.py            # 构建并落盘 cache/holdout.json
    python holdout.py --verify   # 只验证与 fit/dev/eval 不相交
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from data import CACHE, N_DEV, N_EVAL, N_FIT, SEED, SPECS, Task, _humanize, build

NEED = N_FIT + N_DEV + N_EVAL
OUT = CACHE / "holdout.json"


def build_holdout(n: int = 2500, force: bool = False) -> dict:
    if OUT.exists() and not force:
        return json.loads(OUT.read_text())
    from datasets import concatenate_datasets, load_dataset

    rng = random.Random(SEED)                 # 必须和 data.build() 用同一个种子
    bundle = {"n_per_dataset": n, "rows": [], "options": {}}
    for name, (hf_id, text_col, override) in SPECS.items():
        ds = load_dataset(hf_id)
        parts = [ds[s] for s in ("train", "test", "validation") if s in ds]
        pool = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        if override:
            label_names = override
        elif hasattr(pool.features["label"], "names"):
            label_names = pool.features["label"].names
        else:
            m = dict(zip(pool["label"], pool["label_text"]))
            label_names = [m[i] for i in sorted(m)]
        options = [_humanize(x) for x in label_names]
        bundle["options"][name] = options

        idx = list(range(len(pool)))
        rng.shuffle(idx)                      # 顺序消费，和 data.build() 一致
        # 下标不相交还不够：源数据集里有重复文本，同一句话会出现在不同下标上
        # （实测 10,000 条留出里有 7 条这样撞上 fit/dev/eval）。按文本再去一次重，
        # 所以这里多取一截再截断，而不是直接切 n 条。
        used = {r["text"] for r in build()["splits"]["eval"]
                if r["dataset"] == name}
        for sp in ("fit", "dev"):
            used |= {r["text"] for r in build()["splits"][sp] if r["dataset"] == name}
        kept, dropped = [], 0
        for row in pool.select(idx[NEED:]):
            txt = " ".join(str(row[text_col]).split())
            if not txt:
                continue
            if txt in used:
                dropped += 1
                continue
            kept.append(asdict(Task(name, txt, int(row["label"]), len(options))))
            if len(kept) == n:
                break
        assert len(kept) == n, f"{name} 去重后不足 {n} 条（只有 {len(kept)}）"
        bundle["rows"].extend(kept)
        print(f"{name:10s} 池 {len(pool):6d}  已用 {NEED}  -> 全新留出 {n}"
              f"（按文本去重丢掉 {dropped} 条）", flush=True)
    OUT.write_text(json.dumps(bundle))
    return bundle


def verify() -> None:
    """留出集的每一条文本都不能出现在 fit/dev/eval 里。"""
    h, b = json.loads(OUT.read_text()), build()
    used = {}
    for split, rows in b["splits"].items():
        for r in rows:
            used.setdefault(r["dataset"], set()).add(r["text"])
    print(f"{'数据集':<11}{'留出':>7}{'与fit/dev/eval重叠':>20}{'选项数一致':>12}")
    bad = 0
    for ds in b["options"]:
        hs = [r for r in h["rows"] if r["dataset"] == ds]
        ov = sum(1 for r in hs if r["text"] in used.get(ds, ()))
        bad += ov
        same = h["options"][ds] == b["options"][ds]
        print(f"{ds:<11}{len(hs):7d}{ov:20d}{str(same):>12}")
        assert same, f"{ds} 的选项文字和主基准不一致——那就不是同一个任务了"
    print(f"\n{'通过：留出集与 fit/dev/eval 完全不相交' if bad == 0 else f'失败：{bad} 条重叠'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not a.verify:
        build_holdout(a.n, a.force)
    verify()
