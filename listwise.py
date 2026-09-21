#!/usr/bin/env python3
"""列表式打分：把候选**放在一起**比，而不是逐个独立打分。

为什么要这条：banking77 的诊断显示，交叉编码器精排后
  R@1 = 69.0%，R@2 = 78.1%，R@5 = 89.5%
也就是"知道答案在哪两三个里，但分不出第一名"。根因是 NLI 逐个提问
——"这段文本蕴含 card_arrival 吗？"——判断的是**相容**，而路由要的是**最相容**。
选项互斥且高度相似时，这两个目标会分叉，而逐个打分的结构上让选项之间从不照面。

Jev 的 API 形状（所有 criteria 一次提交、返回归一化概率）天生在做选项间比较，
这很可能就是它在 banking77 上领先 9 个点的原因。

做法：取交叉编码器的 top-k 候选，编号 A/B/C/D/E 放进同一个 prompt，
**取下一个 token 的 logits 在这 k 个字母上的分布**——一次前向，不生成。
不用自由生成是因为那要解析、会失败，而且慢得多；受限读 logits 既快又不可能出非法输出
（顺带也复刻了 Jev"类型安全"的那一半）。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import CACHE, build

LETTERS = "ABCDEFGH"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ds", default="banking77")
    ap.add_argument("--cross", default="cache/cross_banking77_k20.npz")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--n", type=int, default=2500)
    a = ap.parse_args()

    b = build()
    d = np.load(a.cross)
    logits_x, y, idx = d["logits"], d["y"], d["idx"]
    logits_x, y, idx = logits_x[:a.n], y[:a.n], idx[:a.n]
    if logits_x.ndim == 3:
        logits_x = logits_x.mean(-1)
    opts = b["options"][a.ds]
    ev = b["splits"]["eval"]

    cand = np.argsort(-np.where(np.isfinite(logits_x), logits_x, -1e9), axis=1)[:, :a.topk]
    ceiling = float((cand == y[:, None]).any(1).mean())
    print(f"候选来自交叉编码器 top-{a.topk}，天花板 R@{a.topk} = {ceiling*100:.1f}%", flush=True)

    tok = AutoTokenizer.from_pretrained(a.repo)
    model = AutoModelForCausalLM.from_pretrained(a.repo, dtype=torch.float32).eval()
    torch.set_grad_enabled(False)
    # 每个字母的首 token id —— 受限读 logits，不可能产生非法输出
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS[:a.topk]]

    correct = 0
    lat = []
    t0 = time.perf_counter()
    scores = np.zeros((len(idx), a.topk), np.float32)
    for row, i in enumerate(idx):
        c = cand[row]
        listing = "\n".join(f"{LETTERS[j]}. {opts[o]}" for j, o in enumerate(c))
        msgs = [
            {"role": "system", "content": "You pick the single best matching option. Reply with one letter."},
            {"role": "user", "content": f"Text: {ev[i]['text']}\n\nOptions:\n{listing}\n\nBest option letter:"},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        t1 = time.perf_counter()
        enc = tok(prompt, return_tensors="pt")
        out = model(**enc).logits[0, -1]                    # 一次前向，读下一个 token 的分布
        s = out[letter_ids].float().numpy()
        lat.append((time.perf_counter() - t1) * 1000)
        scores[row] = s
        correct += int(c[int(s.argmax())] == y[row])
        if row and row % 500 == 0:
            print(f"  {row}/{len(idx)} acc={correct/row*100:.1f}% "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    acc = correct / len(idx)
    # 和交叉编码器分数做简单融合（各自先在候选内 z-标准化，再等权相加）
    xs = np.take_along_axis(logits_x, cand, axis=1)
    def z(v):
        return (v - v.mean(1, keepdims=True)) / np.maximum(v.std(1, keepdims=True), 1e-6)
    fused = z(xs) + z(scores)
    acc_fused = float((np.take_along_axis(cand, fused.argmax(1)[:, None], 1)[:, 0] == y).mean())
    lat.sort()
    res = {"repo": a.repo, "dataset": a.ds, "topk": a.topk, "n": len(idx),
           "ceiling_recall": ceiling, "acc_listwise": acc, "acc_fused_with_cross": acc_fused,
           "acc_cross_alone": float((np.where(np.isfinite(logits_x), logits_x, -1e9).argmax(1) == y).mean()),
           "lat_p50_ms": round(lat[len(lat)//2], 1), "lat_p95_ms": round(lat[int(len(lat)*.95)], 1),
           "wall_s": round(time.perf_counter()-t0, 1)}
    Path("out").mkdir(exist_ok=True)
    tag = a.repo.split("/")[-1]
    (Path("out")/f"listwise_{a.ds}_{tag}_k{a.topk}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[列表式] {acc*100:.1f}%   与交叉编码融合 {acc_fused*100:.1f}%   "
          f"（交叉编码单独 {res['acc_cross_alone']*100:.1f}%，天花板 {ceiling*100:.1f}%）  "
          f"p50={res['lat_p50_ms']}ms", flush=True)


if __name__ == "__main__":
    main()
