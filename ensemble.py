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

from colcenter import ColumnCenter, gated

CROSS_REPO = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
# 裁剪器按后端拆开。实测：把裁剪器从 8M 静态嵌入换成 149M ModernBERT，
# CLINC-150 上 recall@20 从 92.0% 升到 97.5%，但两个后端的反应相反——
#   laya  84.5% -> 87.8%  （+3.3，一次前向读全部候选，候选质量直接决定 token 预算）
#   cross 59.0% -> 60.5%  （+1.5；开了列中心化之后反而低 0.8，见 out/clinc_center.json）
# 召回多进来的候选既是答案也是新干扰项，对逐选项打分的 cross 不划算。
PREFILTERS = {"cross": "static", "laya": "mbert-embed"}
_ENCODERS = {"static": lambda: __import__("encoders").StaticEncoder(),
             "mbert-embed": lambda: __import__("encoders").STEncoder(
                 "mbert-embed", "nomic-ai/modernbert-embed-base", 149.0,
                 prefix="search_query: ")}
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

    def logits(self, text: str, options: Sequence[str],
               hypotheses: Sequence[str] | None = None) -> np.ndarray:
        """蕴含 logit，未归一化。列中心化必须减在这一层——softmax 之后再减，
        减掉的是概率不是分数，行内 argmax 会被归一化常数带偏。"""
        hyps = list(hypotheses) if hypotheses else [TEMPLATE.format(o) for o in options]
        enc = self.tok([text] * len(hyps), hyps, return_tensors="pt",
                       padding=True, truncation=True, max_length=256)
        # no_grad 必须在这里，不能只靠 __init__ 里那次 set_grad_enabled(False)——
        # 后者是**线程局部**的，只对装载模型的线程生效。多线程服务端里每个请求
        # 开新线程，梯度又是打开的，张量带 requires_grad，.numpy() 直接抛
        # "Can't call numpy() on Tensor that requires grad"。
        with self._torch.no_grad():
            return self.model(**enc).logits[:, self.ent].float().numpy()

    def probs(self, text: str, options: Sequence[str],
              hypotheses: Sequence[str] | None = None,
              instructions: str | None = None) -> np.ndarray:
        # instructions 对 NLI 交叉编码器无意义（提问语义已经在假设句里），
        # 保留参数只为让两个后端签名一致，调用方不必分情况处理。
        return _softmax(self.logits(text, options, hypotheses))

    def noul(self, text: str, statements: Sequence[str]) -> np.ndarray:
        """每句话**各自**判真假，返回 P(蕴含)。这是 Noul 原语，不是 Choice。

        和 probs() 的关键差别：softmax 做在模型自己的三分类（蕴含/中立/矛盾）上，
        **不跨选项归一化**。跨选项归一化会把"选项之间谁更好"耦合进来，
        而 Noul 问的是"这一句单独看是不是真的"——正是 NLI 模型的原生题型。
        """
        enc = self.tok([text] * len(statements), list(statements), return_tensors="pt",
                       padding=True, truncation=True, max_length=256)
        with self._torch.no_grad():
            lg = self.model(**enc).logits.float().numpy()          # (n, 3)
        e = np.exp(lg - lg.max(axis=-1, keepdims=True))
        return (e / e.sum(axis=-1, keepdims=True))[:, self.ent]


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

    # 提问词必须可换。写死成"哪个选项最能描述这段文本"在邮件路由上是对的，
    # 但换到别的任务（比如"蛇该往哪走"）就是在问一个无关的问题，然后怪模型答不对。
    # 这类静默失效不报错、有输出、概率也正常，比崩溃危险得多。
    INSTRUCTIONS = "Which option best describes this text?"

    def probs(self, text: str, options: Sequence[str],
              hypotheses: Sequence[str] | None = None,
              instructions: str | None = None) -> np.ndarray:
        crit = dict(zip(options, hypotheses)) if hypotheses else {o: o for o in options}
        q = {"label": {"type": "choice",
                       "instructions": instructions or self.INSTRUCTIONS,
                       "criteria": crit}}
        res = self.agent.predict({"body": text}, q)["answers"]["label"]
        got = res.get("probabilities") or {}
        p = np.array([float(got.get(o, 0.0)) for o in options], np.float64)
        if p.sum() <= 0:                      # 没给概率就退化成 one-hot
            p = np.zeros(len(options)); p[list(options).index(res["choice"])] = 1.0
        return p / p.sum()

    def noul(self, text: str, statements: Sequence[str]) -> np.ndarray:
        """Noul 原语：每句话各自返回"是真的"的概率。

        Laya 一次前向能答多个问题，所以这里把 n 句话打包成 n 个 noul 问题
        一次发出去——比循环调 n 次快得多，也是它被宣传的批量优势所在。
        """
        q = {f"s{i}": {"type": "noul", "instructions": st}
             for i, st in enumerate(statements)}
        ans = self.agent.predict({"body": text}, q)["answers"]
        return np.array([float(ans[f"s{i}"].get("noul", 0.0))
                         for i in range(len(statements))], np.float64)


_BACKENDS = {"cross": _Cross, "laya": _Laya}


class Ensemble:
    """一个或多个本地后端 + 温度校准 + 融合权重。

    backends 里只写 "cross" 就是现在的零样本主力；加上 "laya" 才需要装它。
    calibrate() 之前，温度=1、权重=均分，仍然可用，只是拿不到那 3.33pp。
    """

    def __init__(self, backends: "Sequence[str] | dict" = ("cross",), *,
                 device: str | None = None, shortlist: int | None = 20,
                 prefilters: dict[str, str] | None = None, **kw):
        # backends 可以是名字序列（自己造），也可以是 {名字: 实例}（复用已装载的）。
        # 后者很重要：一次评测里要跑 cross 单打、laya 单打、二者融合，
        # 每次新建就是把 149M+421M 反复装载，纯属浪费。
        if isinstance(backends, dict):
            self.backends = dict(backends)
        else:
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
        self.prefilters = dict(PREFILTERS) | dict(prefilters or {})
        self._pre: dict[str, object] = {}
        self._opt_cache: dict[tuple, np.ndarray] = {}
        self._centers: dict[tuple, ColumnCenter] = {}
        self.temps = {b: 1.0 for b in self.backends}
        self.weights = {b: 1.0 / len(self.backends) for b in self.backends}
        self.calibrated = False

    # ---------- 打分 ----------

    def _encoder(self, name: str):
        if name not in self._pre:
            if name not in _ENCODERS:
                raise ValueError(f"未知裁剪器 {name!r}，可选：{sorted(_ENCODERS)}")
            self._pre[name] = _ENCODERS[name]().load()
        return self._pre[name]

    def _prefilter(self, text, options, hypotheses, enc_name: str):
        """返回候选下标；选项不多就是全体。enc_name 决定用哪个编码器裁。"""
        k = self.shortlist
        if not k or len(options) <= k:
            return np.arange(len(options))
        enc = self._encoder(enc_name)
        key = (enc_name, tuple(hypotheses) if hypotheses else tuple(options))
        if key not in self._opt_cache:
            v = enc.encode(list(key[1]))
            self._opt_cache[key] = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)
        q = enc.encode_one(text)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        return np.argsort(-(self._opt_cache[key] @ q))[:k]

    # ---------- 归纳式列中心化 ----------

    def fit_center(self, pool_texts: Sequence[str], options: Sequence[str],
                   hypotheses: Sequence[str] | None = None) -> ColumnCenter | None:
        """用一池**无标签**文本拟合交叉编码器的列均值。选项数不到门控就不拟合。

        直推式列中心化要对一批查询求均值，单条请求时整项退化成 0。离线拟合一次
        之后，线上逐条推理只多一次减法。500 条就够（见 colcenter.py）。
        """
        options = list(options)
        if not gated(len(options)) or "cross" not in self.backends:
            return None
        m = self.backends["cross"]
        en = self.prefilters.get("cross", "static")
        rows = np.full((len(pool_texts), len(options)), np.nan, np.float32)
        for r, t in enumerate(pool_texts):
            c = self._prefilter(t, options, hypotheses, en)
            sub_h = [hypotheses[i] for i in c] if hypotheses else None
            rows[r, c] = m.logits(t, [options[i] for i in c], sub_h)
        cc = ColumnCenter(len(options)).fit(rows)
        self._centers[tuple(options)] = cc
        return cc

    def load_center(self, path, options: Sequence[str]) -> ColumnCenter:
        cc = ColumnCenter.load(path)
        if cc.k != len(options):
            raise ValueError(f"列均值是 {cc.k} 个选项拟合的，当前有 {len(options)} 个")
        self._centers[tuple(options)] = cc
        return cc

    def _center_for(self, options) -> ColumnCenter | None:
        return self._centers.get(tuple(options)) if gated(len(options)) else None

    def _raw(self, text, options, hypotheses):
        cc = self._center_for(options)
        out, seen = {}, {}
        for b, m in self.backends.items():
            en = self.prefilters.get(b, "static")
            if en not in seen:                      # 两个后端共用同一裁剪器时只编码一次
                seen[en] = self._prefilter(text, options, hypotheses, en)
            cand = seen[en]
            sub_o = [options[i] for i in cand]
            sub_h = [hypotheses[i] for i in cand] if hypotheses else None
            p = np.zeros(len(options), np.float64)
            if cc is not None and hasattr(m, "logits"):
                p[cand] = _softmax(m.logits(text, sub_o, sub_h) - cc.mean[cand])
            else:
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
                  fit_weights: bool = False, verbose: bool = True) -> dict:
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

        # 2) 融合权重：**不拟合**，等权。
        #
        # 这不是偷懒，是量出来的。weight_probe.py 在 5 个用例上把 cross/laya 的概率
        # 全部落盘，再离线比了 5 条规则（out/weight_probe.json）：
        #
        #     固定 50/50          72.45%   <- 最好
        #     朝 0.5 收缩一半      72.30%
        #     平台中点            71.95%
        #     argmax             71.85%
        #     退回单后端          70.15%   <- 最差，曾是本仓库的实现
        #     纯 laya            71.05%
        #     事后最优 w（上界）    73.10%
        #
        # 拟合得越多越差，严格单调。原因是权重和温度冗余：每个后端已经各自按 NLL
        # 定过温度，温度里已经编码了"这个后端有多可靠"。routing 上纯 cross 64.5%、
        # 纯 laya 87.8%，差 23 个点，等权混合仍然是 87.8%——权重根本不是瓶颈。
        # 再用 100 条校准折去估一个已经被估过的量，只加方差不加信号。
        #
        # 反过来，融合本身是有价值的：guardrail 上两个后端各自 52.5% / 53.0%，
        # 等权混合 57.0%（两者"都错"只占 25.2%，互补空间大）。旧规则 snap 到单后端，
        # 把这 4 个点扔了。想复现这个比较：python weight_probe.py --eval
        names = list(self.backends)
        self.weights = {b: 1.0 / len(names) for b in names}
        grid_acc = None
        if len(names) == 2 and fit_weights:          # 仅供诊断/复现旧行为
            a, c = names
            g = np.asarray(grid, float)
            grid_acc = np.array([((w * T[c] + (1 - w) * T[a]).argmax(1) == y).mean() for w in g])
            w = float(g[int(np.argmax(grid_acc))])
            self.weights = {a: 1.0 - w, c: w}

        mix = sum(self.weights[b] * T[b] for b in names)
        self.calibrated = True
        info = {"n": len(ex), "temps": dict(self.temps), "weights": dict(self.weights),
                "fit_acc": float((mix.argmax(1) == y).mean()),
                "per_backend_acc": {b: float((T[b].argmax(1) == y).mean()) for b in names}}
        info["fit_weights"] = bool(fit_weights)
        if grid_acc is not None:
            # 整条曲线落盘：事后判断权重规则好不好，只有 fit_acc 是不够的。
            info["grid"] = [round(float(g), 4) for g in grid]
            info["grid_acc"] = [round(float(v), 4) for v in grid_acc]
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
