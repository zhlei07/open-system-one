#!/usr/bin/env python3
"""召回天花板：精排之前，正确答案在不在候选里。

不训练、不改打分。两套口径分开报，因为它们不是同一个裁剪器：

- 主基准（laya_shortlist / cross_v2）：mbert-embed，模板 0 的余弦
  加上列中心化余弦。BANKING77 上这套的 recall@20 以前报过 98%。
- 用例（ensemble._prefilter）：8M 静态嵌入，对裸标签做余弦，
  选项多于 20 才裁。CLINC 的 84.5% 是裁完之后的数，天花板没量过。

用法:
    python diag_recall.py banking
    python diag_recall.py clinc
"""
from __future__ import annotations

import sys

import numpy as np

import fly_head as fh
from data import CACHE, build

KS = (1, 5, 8, 12, 20)


def _report(title: str, scores: np.ndarray, y: np.ndarray) -> None:
    order = np.argsort(-scores, axis=1)
    ranks = (order == y[:, None]).argmax(1) + 1
    parts = [f"R@{k}={float((ranks <= k).mean()) * 100:.1f}%" for k in KS]
    print(f"  {title:22s}  n={len(y):4d}  " + "  ".join(parts)
          + f"  中位名次={int(np.median(ranks))}", flush=True)


def _combo(desc: np.ndarray, q: np.ndarray, options: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """返回 (纯余弦, 余弦+列中心化)。desc 是 (模板, 类, 维)，模板 0 是裸标签。"""
    bank = fh.FeatureBank(options, None, None, desc)
    feat = bank.zero(q)
    w = np.zeros(len(fh.ZERO_FEATURES), np.float32)
    w[0] = w[3] = 1.0
    return feat[..., 0], feat @ w


def banking() -> None:
    b = build()
    ev = b["splits"]["eval"]
    idx = [i for i, r in enumerate(ev) if r["dataset"] == "banking77"]
    y = np.array([ev[i]["label"] for i in idx])
    opts = b["options"]["banking77"]
    print(f"BANKING77 eval  {len(opts)} 类  {len(y)} 条  （与主基准同一条切分）")
    for name in ("static", "mbert-embed"):
        z = np.load(CACHE / "emb" / f"{name}.npz")
        cos, combo = _combo(z[f"desc_banking77"], z["eval"][idx], opts)
        print(f"[{name}]")
        _report("裸标签余弦", cos, y)
        _report("余弦 + 列中心化", combo, y)


def clinc() -> None:
    from encoders import STEncoder, StaticEncoder
    from usecases import build_usecases

    # 和 usecase_bench 的默认一样：500 条，前 100 条校准，后 400 条才是报出去的准确率。
    u = build_usecases(500)["routing"]
    y = np.array(u.gold[100:])
    texts = u.texts[100:]
    print(f"CLINC 路由评测集  {u.n_options} 类  {len(y)} 条  "
          f"（seed 与 usecases.py 相同，不含前 100 条校准）")

    static = StaticEncoder().load()
    q = static.encode(texts)
    d = static.encode(u.options)
    _report("static 裸标签余弦", q @ d.T, y)

    enc = STEncoder("mbert-embed", "nomic-ai/modernbert-embed-base", 149.0,
                    prefix="search_query: ").load()
    # 与 run.py 相同：查询和标签都走这个编码器自己的前缀。
    q = enc.encode(texts)
    d = enc.encode(u.options)
    _report("mbert 裸标签余弦", q @ d.T, y)
    # 列中心化需要 (1, k, dim) 的 desc，这里只有一个模板。
    desc = d[None, :, :]
    _, combo = _combo(np.concatenate([desc, desc, desc], 0), q, u.options)
    _report("mbert 余弦+列中心化", combo, y)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "banking"
    if which == "banking":
        banking()
    elif which == "clinc":
        clinc()
    else:
        banking()
        clinc()
