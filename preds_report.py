#!/usr/bin/env python3
"""从 out/usecase_preds.json 做事后诊断。不重跑、不花钱。

写这个是因为一个具体的洞：护栏上 Jev 是 41.8%，在平衡二分类上**低于**瞎猜。
要判断这是"选项下标反了"还是"模型真的读不出能力边界"，需要逐条预测——
而 usecase_bench 以前只累加 ok，不落盘预测，想查就得重跑一遍 Jev（要花钱）。

    python preds_report.py            # 全部用例
    python preds_report.py guardrail  # 只看一个
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

P = Path("out/usecase_preds.json")


def binary_flip(gold, pred):
    """二分类专用：预测整体取反之后准不准。明显更高就是下标接反了。"""
    ok = np.array([g == p for g, p in zip(gold, pred) if p is not None])
    inv = np.array([g == 1 - p for g, p in zip(gold, pred) if p is not None])
    return ok.mean(), inv.mean()


def main():
    if not P.exists():
        print(f"{P} 不存在——先跑一次 usecase_bench.py（新版才落盘逐条预测）")
        return
    data = json.loads(P.read_text())
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for key, d in data.items():
        if only and key != only:
            continue
        gold, opts = d["gold"], d["options"]
        k = len(opts)
        print(f"\n=== {key}  {k} 选项  {len(gold)} 条 ===")
        cnt = np.bincount(gold, minlength=k)
        print(f"金标分布: 最多的类 {cnt.max()}/{len(gold)} = {cnt.max()/len(gold)*100:.1f}%"
              f"（多数类基线）, 非空类 {int((cnt>0).sum())}/{k}")
        for name, pred in d["pred"].items():
            good = [(g, p) for g, p in zip(gold, pred) if p is not None]
            acc = np.mean([g == p for g, p in good])
            nfail = sum(p is None for p in pred)
            line = f"  {name:8s} acc={acc*100:5.2f}%  失败={nfail}"
            pc = np.bincount([p for p in pred if p is not None], minlength=k)
            line += f"  预测用到 {int((pc>0).sum())}/{k} 个类"
            if pc.max() / max(len(good), 1) > 0.9 and k > 1:
                line += f"  ⚠ 塌缩到单一类 {opts[int(pc.argmax())]!r}"
            print(line)
            if k == 2:
                a, inv = binary_flip(gold, pred)
                flag = "  ⚠ 取反后显著更高 -> 疑似选项下标接反" if inv - a > 0.05 else ""
                print(f"           取反后 acc={inv*100:5.2f}%{flag}")
                tp = sum(1 for g, p in good if g == 1 and p == 1)
                fp = sum(1 for g, p in good if g == 0 and p == 1)
                fn = sum(1 for g, p in good if g == 1 and p == 0)
                tn = sum(1 for g, p in good if g == 0 and p == 0)
                print(f"           混淆矩阵  [金标0]: 预测0={tn} 预测1={fp}   "
                      f"[金标1]: 预测0={fn} 预测1={tp}")
        names = list(d["pred"])
        if len(names) > 1:
            print("  两两一致率 / 都对 / 都错:")
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = d["pred"][names[i]], d["pred"][names[j]]
                    same = np.mean([x == y for x, y in zip(a, b)])
                    both = np.mean([x == g and y == g for x, y, g in zip(a, b, gold)])
                    neither = np.mean([x != g and y != g for x, y, g in zip(a, b, gold)])
                    print(f"    {names[i]:7s} vs {names[j]:7s}  一致 {same*100:5.1f}%  "
                          f"都对 {both*100:5.1f}%  都错 {neither*100:5.1f}%")


if __name__ == "__main__":
    main()
