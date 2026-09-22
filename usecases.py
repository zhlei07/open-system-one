#!/usr/bin/env python3
"""网上最火的 5 类 Jev 用例，换成可复现的公开数据集。

选择依据不是我们觉得哪个重要，是从两份 awesome-jev 清单和多家测评文章里交叉比对
出现频率最高的几类（来源见 README）。实时控制类（Doom / 象棋 / 交易）同样很火，
但那是延迟演示不是分类任务，用文本分类器对标不公平，所以单独排除并注明。

每个用例都标了 Jev 的原语类型（Choice / Noul / Score）和选项数——选项数是我们
已知架构差异最大的一根轴，2 到 151 的跨度正是这套基准的价值所在。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

N_DEFAULT = 500
SEED = 20260922


@dataclass
class UseCase:
    key: str
    jev_usecase: str          # 对应的 Jev 用例
    primitive: str            # Choice / Noul / Score
    source: str               # 数据集
    options: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    gold: list[int] = field(default_factory=list)

    @property
    def n_options(self) -> int: return len(self.options)
    def __len__(self) -> int: return len(self.texts)


def _humanize(s: str) -> str:
    """intent 名 restaurant_reviews -> 'restaurant reviews'。零样本模型要读得懂标签。"""
    return re.sub(r"[_\-]+", " ", s).strip().lower()


def _take(ds, n, seed=SEED):
    return ds.shuffle(seed=seed).select(range(min(n, len(ds))))


def build_usecases(n: int = N_DEFAULT) -> dict[str, UseCase]:
    from datasets import load_dataset
    out: dict[str, UseCase] = {}

    # 1. 内容审核 —— cloudraft 把它列为最重要的一类；Mastra / Discord bot 都是这个
    d = _take(load_dataset("cardiffnlp/tweet_eval", "offensive", split="test"), n)
    out["moderation"] = UseCase(
        "moderation", "内容审核 / 安全门", "Noul", "cardiffnlp/tweet_eval:offensive",
        options=["an acceptable message", "an offensive or abusive message"],
        texts=[r["text"] for r in d], gold=[int(r["label"]) for r in d])

    # 2. 工单路由 —— 151 个意图，Jev 被宣传最多的落地场景
    clinc = load_dataset("clinc/clinc_oos", "plus", split="test")
    names = clinc.features["intent"].names
    oos_id = names.index("oos")
    ins = clinc.filter(lambda r: r["intent"] != oos_id)
    d = _take(ins, n)
    opts = [_humanize(x) for i, x in enumerate(names) if i != oos_id]
    remap = {old: new for new, old in enumerate(i for i in range(len(names)) if i != oos_id)}
    out["routing"] = UseCase(
        "routing", "工单 / 邮件路由", "Choice", "clinc/clinc_oos:plus (in-scope)",
        options=opts, texts=[r["text"] for r in d],
        gold=[remap[int(r["intent"])] for r in d])

    # 3. Agent 护栏 —— "这个请求在不在能力范围内"，就是 tool-call 校验的公开代理
    oos = clinc.filter(lambda r: r["intent"] == oos_id)
    half = n // 2
    a, b = _take(ins, half, SEED + 1), _take(oos, half, SEED + 2)
    # 两半必须打散后再拼：直接拼接会让"前 100 条做校准"取到清一色域内样本，
    # 校准集退化成单一类别，温度和融合权重都白标定。这个坑踩过一次。
    gt = [(t["text"], 0) for t in a] + [(t["text"], 1) for t in b]
    _rng = __import__("random").Random(SEED)
    _rng.shuffle(gt)
    out["guardrail"] = UseCase(
        "guardrail", "Agent 护栏 / 域外拦截", "Noul", "clinc/clinc_oos:plus (in vs oos)",
        options=["a request the assistant is designed to handle",
                 "a request outside the assistant's scope"],
        texts=[t for t, _ in gt], gold=[g for _, g in gt])

    # 4. 优先级 / 质量评分 —— 序数，Jev 自己承认这是它最弱的一类
    d = _take(load_dataset("SetFit/sst5", split="test"), n)
    out["scoring"] = UseCase(
        "scoring", "优先级 / 质量评分", "Score", "SetFit/sst5",
        options=["very negative", "negative", "neutral", "positive", "very positive"],
        texts=[r["text"] for r in d], gold=[int(r["label"]) for r in d])

    # 5. 批量打标 —— 1kpapers / 简历筛选那一类，14 类本体标注
    d = _take(load_dataset("fancyzhx/dbpedia_14", split="test"), n)
    dn = load_dataset("fancyzhx/dbpedia_14", split="test").features["label"].names
    out["labeling"] = UseCase(
        "labeling", "批量打标 / 筛选", "Choice", "fancyzhx/dbpedia_14",
        options=[_humanize(re.sub(r"(?<!^)(?=[A-Z])", " ", x)) for x in dn],
        texts=[(r["title"] + ". " + r["content"])[:600] for r in d],
        gold=[int(r["label"]) for r in d])

    return out


if __name__ == "__main__":
    for k, u in build_usecases(int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 200).items():
        print(f"{k:11s} {u.primitive:7s} {u.n_options:4d} 选项  {len(u):5d} 条   {u.source}")
        print(f"            选项样例: {u.options[:4]}")
        print(f"            文本样例: {u.texts[0][:80]!r}  -> {u.options[u.gold[0]]!r}")
