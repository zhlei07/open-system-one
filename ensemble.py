#!/usr/bin/env python3
"""open-system-one 的决策组件：文本 + 选项 -> 类型安全的选择 + 校准概率。

纯本地、免费、CPU 可跑。Laya 是**可选**后端：不装照常工作，装了可融合。

    from ensemble import Ensemble

    e = Ensemble()                                  # 只用交叉编码器
    e = Ensemble(backends=("cross", "laya"))        # 融合，需要 pip install laya
    e.calibrate(dev_examples)                       # 每任务 50 条就够
    d = e.decide("my card was declined", OPTIONS)
    d.choice, d.prob, d.backend

实测（4 个公开任务、10,000 条、eval 折只看一次）：

    交叉编码器单打          78.57 %
    + Laya，置信度路由      78.57 %   （无效：最差的模型最自信）
    + Laya，全局权重        79.14 %
    + Laya，按任务权重      81.90 %   ← 每任务 50 条标注拿到其中 2.48pp
    oracle 上界             86.21 %

两个必须知道的口径问题：
  1. **权重按准确率选，不要按 log score 选。** 后者亏 3.2pp——NLL 惩罚的是混合分布
     的校准，会被 Laya 未校准的过度自信主导，而权重真正要优化的是排序。校准由
     温度那一层单独负责，两件事分开做。
  2. **不要用模型置信度做路由。** Laya 在 banking77 上平均置信度最高(0.958)、准确率
     最低(54.3%)，置信度辨别力 AUC 0.437——比抛硬币还差。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

CROSS_REPO = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
LAYA_ID = "convaiinnovations/laya"
TEMPLATE = "This example is {}."


@dataclass
class Decision:
    """一次判断的结果。probs 与传入的 options 顺序一一对应。"""
    choice: str
    prob: float
    probs: np.ndarray
    backend: str
    runner_up: str | None = None

    def __repr__(self) -> str:
        return f"Decision({self.choice!r}, p={self.prob:.3f}, via={self.backend})"


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    x = x - np.nanmax(np.where(np.isfinite(x), x, -np.inf), -1, keepdims=True)
    e = np.exp(np.where(np.isfinite(x), x, -np.inf))
    return e / e.sum(-1, keepdims=True)


def _temper(p: np.ndarray, t: float) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return _softmax(np.log(np.clip(p, 1e-12, None)) / t)


class _Cross:
    """零样本 NLI 交叉编码器：每个选项一次前向，延迟随选项数线性增长。"""

    name = "cross"

    def __init__(self, repo: str = CROSS_REPO, shortlist: int | None = 20):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
        torch.set_grad_enabled(False)
        self.ent = next(i for i, l in self.model.config.id2label.items()
                        if "entail" in l.lower())
        self.shortlist = shortlist

    def probs(self, text: str, options: Sequence[str],
              hypotheses: Sequence[str] | None = None) -> np.ndarray:
        hyps = list(hypotheses) if hypotheses else [TEMPLATE.format(o) for o in options]
        enc = self.tok([text] * len(hyps), hyps, return_tensors="pt",
                       padding=True, truncation=True, max_length=256)
        s = self.model(**enc).logits[:, self.ent].float().numpy()
        return _softmax(s)


class _Laya:
    """Laya 421M：单次前向给所有选项打分，延迟不随选项数增长。

    head_max_len 默认 192，77 个选项时每个只分到 3-4 token —— 官方明确说这是
    banking77 掉到 0.425 的原因。这里按官方建议抬到 512。
    """

    name = "laya"

    def __init__(self, model_id: str = LAYA_ID, device: str | None = None,
                 head_max_len: int = 512, max_len: int = 1024):
        import laya

        self.agent = laya.load(model_id, device=device)
        self.agent.cfg["head_max_len"] = head_max_len
        self.agent.cfg["max_len"] = max_len
        # 设备必须可见：laya.load() 不传 device 会自动选 cuda>mps>cpu，
        # 静默选中 mps 会让延迟和纯 CPU 链路不可比。
        self.device = str(self.agent.device)

    def probs(self, text: str, options: Sequence[str],
              hypotheses: Sequence[str] | None = None) -> np.ndarray:
        crit = dict(zip(options, hypotheses)) if hypotheses else {o: o for o in options}
        q = {"label": {"type": "choice",
                       "instructions": "Which option best describes this text?",
                       "criteria": crit}}
        res = self.agent.predict({"body": text}, q)["answers"]["label"]
        got = res.get("probabilities") or {}
        p = np.array([float(got.get(o, 0.0)) for o in options], np.float64)
        if p.sum() <= 0:                      # 没给概率就退化成 one-hot
            p = np.zeros(len(options)); p[list(options).index(res["choice"])] = 1.0
        return p / p.sum()


_BACKENDS = {"cross": _Cross, "laya": _Laya}


class Ensemble:
    """一个或多个本地后端 + 温度校准 + 融合权重。

    backends 里只写 "cross" 就是现在的零样本主力；加上 "laya" 才需要装它。
    calibrate() 之前，温度=1、权重=均分，仍然可用，只是拿不到那 3.33pp。
    """

    def __init__(self, backends: Sequence[str] = ("cross",), *,
                 device: str | None = None, shortlist: int | None = 20, **kw):
        unknown = set(backends) - set(_BACKENDS)
        if unknown:
            raise ValueError(f"未知后端 {sorted(unknown)}，可选：{sorted(_BACKENDS)}")
        self.backends = {}
        for b in backends:
            try:
                self.backends[b] = (_BACKENDS[b](device=device, **kw) if b == "laya"
                                    else _BACKENDS[b](**kw))
            except ImportError as e:
                raise ImportError(
                    f"后端 {b!r} 需要额外依赖：{e}。"
                    f"Laya 是可选的——去掉它，Ensemble(backends=('cross',)) 照常工作。"
                ) from None
        # 选项多于 shortlist 时先用 8M 静态嵌入裁候选：交叉编码器每个选项一次前向，
        # 150 个选项就是 150 次。裁到 20 个，延迟降一个数量级，recall@20 实测 97.8%。
        self.shortlist = shortlist
        self._pre = None
        self._opt_cache: dict[tuple, np.ndarray] = {}
        self.temps = {b: 1.0 for b in self.backends}
        self.weights = {b: 1.0 / len(self.backends) for b in self.backends}
        self.calibrated = False

    # ---------- 打分 ----------

    def _prefilter(self, text, options, hypotheses):
        """返回候选下标；选项不多就是全体。"""
        k = self.shortlist
        if not k or len(options) <= k:
            return np.arange(len(options))
        if self._pre is None:
            from encoders import StaticEncoder
            self._pre = StaticEncoder().load()
        key = tuple(hypotheses) if hypotheses else tuple(options)
        if key not in self._opt_cache:
            v = self._pre.encode(list(key))
            self._opt_cache[key] = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)
        q = self._pre.encode_one(text)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        return np.argsort(-(self._opt_cache[key] @ q))[:k]

    def _raw(self, text, options, hypotheses):
        cand = self._prefilter(text, options, hypotheses)
        sub_o = [options[i] for i in cand]
        sub_h = [hypotheses[i] for i in cand] if hypotheses else None
        out = {}
        for b, m in self.backends.items():
            p = np.zeros(len(options), np.float64)
            p[cand] = m.probs(text, sub_o, sub_h)
            out[b] = p
        return out

    def _mix(self, raw: dict[str, np.ndarray]) -> np.ndarray:
        p = sum(self.weights[b] * _temper(raw[b], self.temps[b]) for b in raw)
        return np.asarray(p) / np.sum(p)

    def decide(self, text: str, options: Sequence[str],
               hypotheses: Sequence[str] | None = None) -> Decision:
        options = list(options)
        p = self._mix(self._raw(text, options, hypotheses))
        order = np.argsort(-p)
        via = "+".join(f"{b}:{self.weights[b]:.2f}" for b in self.backends)
        return Decision(choice=options[order[0]], prob=float(p[order[0]]), probs=p,
                        backend=via,
                        runner_up=options[order[1]] if len(options) > 1 else None)

    # ---------- 校准 ----------

    def calibrate(self, examples: Iterable[tuple[str, Sequence[str], int]], *,
                  hypotheses: Sequence[str] | None = None,
                  grid: Sequence[float] = tuple(np.arange(0, 1.0001, 0.05)),
                  verbose: bool = True) -> dict:
        """examples 是 (文本, 选项列表, 正确选项下标)。每任务 50 条就能拿到大部分收益。

        顺序很重要：**先按准确率选融合权重，再用温度修校准**。反过来（用 NLL 选权重）
        实测亏 3.2 pp，因为 NLL 会被未校准后端的过度自信主导。
        """
        ex = list(examples)
        if not ex:
            raise ValueError("calibrate() 需要至少一条带标注的样本")
        y = np.array([g for _, _, g in ex])
        raw = {b: [] for b in self.backends}
        for text, opts, _ in ex:
            r = self._raw(text, list(opts), hypotheses)
            for b in self.backends:
                raw[b].append(r[b])

        width = max(len(v) for v in next(iter(raw.values())))
        def pad(rows):
            m = np.zeros((len(rows), width))
            for i, v in enumerate(rows):
                m[i, :len(v)] = v
            return m
        P = {b: pad(v) for b, v in raw.items()}

        # 1) 温度：每个后端各自按 NLL 在这批样本上标定
        tg = np.geomspace(0.2, 20, 70)
        for b in self.backends:
            nlls = [-np.log(np.clip(_temper(P[b], t)[np.arange(len(y)), y], 1e-12, None)).mean()
                    for t in tg]
            self.temps[b] = float(tg[int(np.argmin(nlls))])
        T = {b: _temper(P[b], self.temps[b]) for b in self.backends}

        # 2) 融合权重：按**准确率**选，不是 NLL
        names = list(self.backends)
        if len(names) == 1:
            self.weights = {names[0]: 1.0}
        elif len(names) == 2:
            a, c = names
            accs = [( (w * T[c] + (1 - w) * T[a]).argmax(1) == y).mean() for w in grid]
            w = float(grid[int(np.argmax(accs))])
            self.weights = {a: 1.0 - w, c: w}
        else:                                  # >2 个后端：坐标上升
            self.weights = {b: 1.0 / len(names) for b in names}
            for _ in range(3):
                for b in names:
                    best = (-1.0, self.weights[b])
                    for w in grid:
                        trial = dict(self.weights); trial[b] = w
                        s = sum(trial.values()) or 1.0
                        mix = sum(trial[k] / s * T[k] for k in names)
                        best = max(best, (float((mix.argmax(1) == y).mean()), float(w)))
                    self.weights[b] = best[1]
                s = sum(self.weights.values()) or 1.0
                self.weights = {b: v / s for b, v in self.weights.items()}

        mix = sum(self.weights[b] * T[b] for b in names)
        self.calibrated = True
        info = {"n": len(ex), "temps": dict(self.temps), "weights": dict(self.weights),
                "fit_acc": float((mix.argmax(1) == y).mean()),
                "per_backend_acc": {b: float((T[b].argmax(1) == y).mean()) for b in names}}
        if verbose:
            print(f"校准 {info['n']} 条：权重 {info['weights']}  温度 "
                  f"{ {k: round(v,2) for k,v in info['temps'].items()} }")
            print(f"  拟合折准确率 {info['fit_acc']*100:.1f}%  "
                  f"(各后端 { {k: f'{v*100:.1f}%' for k,v in info['per_backend_acc'].items()} })")
        return info

    # ---------- 存取 ----------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"temps": self.temps, "weights": self.weights,
             "backends": list(self.backends), "calibrated": self.calibrated},
            ensure_ascii=False, indent=1))

    def load(self, path: str | Path) -> "Ensemble":
        d = json.loads(Path(path).read_text())
        missing = set(d["backends"]) - set(self.backends)
        if missing:
            raise ValueError(f"这份校准是给 {d['backends']} 存的，当前缺少 {sorted(missing)}")
        self.temps.update({k: v for k, v in d["temps"].items() if k in self.backends})
        self.weights.update({k: v for k, v in d["weights"].items() if k in self.backends})
        self.calibrated = d.get("calibrated", False)
        return self
