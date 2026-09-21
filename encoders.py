"""前置编码器：把文本变成向量。四条链路。

| 名字        | 参数量 | 类型                          | 为什么选它 |
|-------------|--------|-------------------------------|------------|
| static      | 8M     | model2vec 静态嵌入（查表+平均）| 没有 attention，纯查表。这是"编码器能有多便宜"的下界 |
| minilm      | 22M    | 6 层 Transformer 句向量       | 边缘部署的事实标准 |
| modernbert  | 149M   | ModernBERT-base 原始 MLM      | 用户点名要的。注意：没做过句向量训练，mean-pool 是我们外加的 |
| mbert-embed | 149M   | nomic modernbert-embed-base   | 同尺寸但做过对比学习。没有它，上一行会得出"大模型更差"的错误结论 |

`modernbert` 和 `mbert-embed` 参数量一模一样、架构一模一样，差别只有"有没有
为句向量训练过"。把两条都跑，才能把"规模"和"训练目标"这两个变量拆开。
"""
from __future__ import annotations

import numpy as np

CPU = "cpu"


class Encoder:
    name = "?"
    params_m = 0.0

    def load(self): raise NotImplementedError
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray: raise NotImplementedError
    def encode_one(self, text: str) -> np.ndarray: return self.encode([text], batch_size=1)[0]


class StaticEncoder(Encoder):
    """model2vec：词表蒸馏成静态向量，推理 = 查表 + 平均。没有矩阵乘法的前向。"""
    name, params_m = "static", 7.6

    def load(self):
        from model2vec import StaticModel
        self.m = StaticModel.from_pretrained("minishlab/potion-base-8M")
        self.dim = self.m.dim
        return self

    def encode(self, texts, batch_size=64):
        v = np.asarray(self.m.encode(texts, show_progress_bar=False), dtype=np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(n, 1e-9)


class STEncoder(Encoder):
    """sentence-transformers 通路。prefix 是 nomic 那类模型要求的任务前缀，
    不加的话它的分数会莫名其妙掉几个点——这类"不报错只是变差"的坑最贵。"""
    def __init__(self, name, repo, params_m, prefix=""):
        self.name, self.repo, self.params_m, self.prefix = name, repo, params_m, prefix

    def load(self):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_grad_enabled(False)
        self.m = SentenceTransformer(self.repo, device=CPU)
        self.m.eval()
        self.dim = self.m.get_sentence_embedding_dimension()
        return self

    def encode(self, texts, batch_size=64):
        if self.prefix:
            texts = [self.prefix + t for t in texts]
        return self.m.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def all_encoders() -> dict[str, Encoder]:
    return {
        "static":      StaticEncoder(),
        "minilm":      STEncoder("minilm", "sentence-transformers/all-MiniLM-L6-v2", 22.7),
        "modernbert":  STEncoder("modernbert", "answerdotai/ModernBERT-base", 149.0),
        "mbert-embed": STEncoder("mbert-embed", "nomic-ai/modernbert-embed-base", 149.0,
                                 prefix="search_query: "),
    }
