#!/usr/bin/env python3
"""归纳式列中心化：修 hubness，但不需要看到一整批查询。

`fly_head._colcenter` 是**直推式**的——列均值对当前这批查询求。单条请求时
这批只有一行，均值就是它自己，整项恒等于 0：线上逐条推理拿不到任何收益。
这里把列均值改成**离线**从一池无标签文本估一次，线上只做一次减法。

BANKING77 实测（cross-encoder，前 20 候选，2500 条 eval）：

    池子大小   直推式    归纳式
       100     +2.51    +1.57
       250     +2.50    +2.24
       500     +2.53    +2.54   <- 500 条就追平直推式
      1000     +2.60    +2.24

门控：只在选项数 >= GATE 时开。Δ = 开 - 关，实测：

      2 选项  SST-2       -0.18 pp
      4 选项  AG News     +0.04 pp
      6 选项  Emotion     -3.34 pp
     77 选项  BANKING77   +2.30 pp
    150 选项  CLINC       +3.25 ~ +5.50 pp

修的是 hubness——有些标签词天生和一切都像（banking77 的 "card payment"、
CLINC 的 "change speed"），不减掉它会无差别地吸走样本。选项越多，它能吸走的
越多，所以门控看的是**选项个数**，不是数据集名字：后者会让"零样本"名存实亡。
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

GATE = 20          # 选项数达到这个值才开。见模块头部的逐任务实测。
MIN_OBS = 5        # 一列在池子里被打分少于这么多次，就退回全局均值


def gated(n_options: int, gate: int = GATE) -> bool:
    """该不该对这个选项集开列中心化。只看选项个数。"""
    return n_options >= gate


class ColumnCenter:
    """一个选项集的列均值。用无标签文本拟合，之后逐条推理时减掉。

        cc = ColumnCenter(len(options)).fit(pool_scores)   # (n_pool, k)，未打分处填 NaN
        s  = cc.transform(scores)                          # (k,) 或 (n, k)

    池子里没被打分过（或打分次数太少）的列退回全局均值——不这么做的话，
    一个罕见标签会因为均值噪声被系统性地抬高或压低。
    """

    def __init__(self, k: int, min_obs: int = MIN_OBS):
        self.k, self.min_obs = k, min_obs
        self.mean = np.zeros(k, np.float32)
        self.n_obs = np.zeros(k, np.int32)
        self.fitted = False

    def fit(self, pool_scores: np.ndarray) -> "ColumnCenter":
        s = np.asarray(pool_scores, np.float64)
        if s.ndim != 2 or s.shape[1] != self.k:
            raise ValueError(f"pool_scores 应为 (n_pool, {self.k})，收到 {s.shape}")
        s = np.where(np.isfinite(s), s, np.nan)
        self.n_obs = (~np.isnan(s)).sum(0).astype(np.int32)
        if not self.n_obs.any():
            raise ValueError("池子里没有任何一个选项被打过分")
        with warnings.catch_warnings():        # 全 NaN 的列是预期内的，下面会退回全局均值
            warnings.simplefilter("ignore", RuntimeWarning)
            col = np.nanmean(s, axis=0)
            glob = float(np.nanmean(s))
        self.mean = np.where(self.n_obs >= self.min_obs, np.nan_to_num(col, nan=glob),
                             glob).astype(np.float32)
        self.fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.asarray(scores)
        return np.asarray(scores) - self.mean

    # ---------- 落盘：离线拟合一次，线上直接加载 ----------

    def save(self, path: str | Path, options: list[str] | None = None) -> None:
        Path(path).write_text(json.dumps(
            {"k": self.k, "min_obs": self.min_obs, "mean": self.mean.tolist(),
             "n_obs": self.n_obs.tolist(), "options": options}, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "ColumnCenter":
        d = json.loads(Path(path).read_text())
        cc = cls(int(d["k"]), int(d.get("min_obs", MIN_OBS)))
        cc.mean = np.array(d["mean"], np.float32)
        cc.n_obs = np.array(d["n_obs"], np.int32)
        cc.fitted = True
        return cc

    def __repr__(self) -> str:
        return (f"ColumnCenter(k={self.k}, fitted={self.fitted}, "
                f"列观测数 min={int(self.n_obs.min()) if self.fitted else 0}"
                f"/median={int(np.median(self.n_obs)) if self.fitted else 0})")
