"""Fly 决策头：把 palletizer 的打分器原样搬到文本决策上。

移植对应表：

| palletizer            | 这里                          |
|-----------------------|-------------------------------|
| 候选落位（运行时生成）| 候选选项（请求时才给，同 Jev）|
| 8 个归一化几何特征    | 8 个归一化语义特征            |
| `w·featurize(f)`      | `w·phi(query, option)`        |
| argmax 选落位         | argmax 选选项                 |
| (1+λ)-ES 搜 8 个权重  | 同一套 ES，同样的 σ 重启      |
| 0.7*均值 + 0.3*最差   | 0.7*四个任务均值 + 0.3*最差任务|

**为什么决策头必须是"给候选打分"而不是固定分类头**：固定头的输出维度在训练时
就焊死了，换一组选项就得重训。Jev 的 API 是请求时才给 `criteria`，所以它不可能
是固定头。打分式的头天然支持任意候选集——这一点和码垛里"候选点随已放箱子动态生成"
是同一个结构。

两种模式：
  few-shot  —— 允许看 2000 条带标签样本（类质心 / kNN）。这是"你有历史数据"的场景。
  zero-shot —— 只有选项文字描述，和 Jev / Qwen 拿到的信息完全一样。**这一列才是
               回答"能不能平替 Jev"的那一列**，few-shot 赢了不算数。
"""
from __future__ import annotations

import random

import numpy as np

# ---- 特征名：与 palletizer.scoring.FEATURE_NAMES 同样的角色 ----
FEW_FEATURES = ("proto", "desc", "knn1", "knnk", "margin", "prior", "white", "tight")
ZERO_FEATURES = ("t0", "t1", "t2", "t0c", "t1c", "t2c")

# zero-shot 的三个模板。多模板是有信息量的：同一个标签换个说法，
# 编码器给出的相似度分布不一样，ES 负责学怎么加权。
TEMPLATES = (
    "{}",
    "This text is about {}.",
    "The correct category is: {}.",
)


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (n,d) b (m,d)，都已 L2 归一化 -> (n,m)。"""
    return a @ b.T


def _colcenter(s: np.ndarray) -> np.ndarray:
    """按查询维度中心化：减掉"这个选项对所有查询的平均相似度"。

    修的是 hubness——有些标签词天生和一切都像（比如 banking77 的 "card payment"），
    不中心化的话它会无差别地吸走大量样本。这是零样本分类最常见的系统性偏差，
    和码垛里"低位优先会无差别地吸走所有候选"是同一类问题：
    一个特征的尺度失衡会吃掉其他所有特征的话语权。
    """
    return s - s.mean(axis=0, keepdims=True)


class FeatureBank:
    """预计算某个编码器在某个数据集上的 (n_query, n_option, n_feature) 特征张量。

    预计算之后 ES 的每次评估就只是一次张量点乘——8 个权重搜 80 代
    在 numpy 里是秒级的，不需要 palletizer 那样开 18 进程。
    """

    def __init__(self, options: list[str], fit_emb: np.ndarray | None, fit_lab: np.ndarray | None,
                 desc_emb: np.ndarray):
        self.options = options
        self.k = len(options)
        self.desc_emb = desc_emb              # (n_template, k, d)
        self.fit_emb, self.fit_lab = fit_emb, fit_lab
        if fit_emb is not None:
            self.centroids = np.stack([
                fit_emb[fit_lab == c].mean(axis=0) if (fit_lab == c).any() else np.zeros(fit_emb.shape[1])
                for c in range(self.k)
            ]).astype(np.float32)
            n = np.linalg.norm(self.centroids, axis=1, keepdims=True)
            self.centroids /= np.maximum(n, 1e-9)
            self.global_mean = fit_emb.mean(axis=0)
            counts = np.array([(fit_lab == c).sum() for c in range(self.k)], dtype=np.float32)
            self.log_prior = np.log(np.maximum(counts, 1) / counts.sum())
            self.log_prior -= self.log_prior.mean()
            self.by_class = [np.where(fit_lab == c)[0] for c in range(self.k)]

    # ---------- zero-shot ----------
    def zero(self, q: np.ndarray) -> np.ndarray:
        """(n, k, 6)：3 个模板 × {原始 cos, 列中心化 cos}。"""
        raw = [_cos(q, self.desc_emb[t]) for t in range(len(TEMPLATES))]
        feats = raw + [_colcenter(s) for s in raw]
        return np.stack(feats, axis=-1).astype(np.float32)

    # ---------- few-shot ----------
    def few(self, q: np.ndarray, knn_k: int = 5) -> np.ndarray:
        """(n, k, 8)。每一列都取"越大越好"的方向，和 scoring.featurize 的纪律一致。"""
        n = q.shape[0]
        proto = _cos(q, self.centroids)                       # 类质心相似度
        desc = _cos(q, self.desc_emb[0])                      # 标签文字相似度
        sim = _cos(q, self.fit_emb)                           # (n, n_fit)

        knn1 = np.zeros((n, self.k), np.float32)
        knnk = np.zeros((n, self.k), np.float32)
        tight = np.zeros((n, self.k), np.float32)
        for c, idx in enumerate(self.by_class):
            if len(idx) == 0:
                continue
            s = sim[:, idx]
            kk = min(knn_k, s.shape[1])
            top = np.partition(s, -kk, axis=1)[:, -kk:]
            knn1[:, c] = top.max(axis=1)
            knnk[:, c] = top.mean(axis=1)
            tight[:, c] = -s.std(axis=1)                      # 类内越紧越可信

        margin = proto - proto.mean(axis=1, keepdims=True)    # 跨选项中心化
        qc = q - self.global_mean
        cc = self.centroids - self.global_mean
        qc /= np.maximum(np.linalg.norm(qc, axis=1, keepdims=True), 1e-9)
        cc /= np.maximum(np.linalg.norm(cc, axis=1, keepdims=True), 1e-9)
        white = _cos(qc, cc)                                  # 去掉共同成分后的相似度
        prior = np.tile(self.log_prior / 5.0, (n, 1))         # /5 让它和 cos 同量级

        return np.stack([proto, desc, knn1, knnk, margin, prior, white, tight],
                        axis=-1).astype(np.float32)


# ---------------- 打分 + 搜权重 ----------------

def scores(feat: np.ndarray, w: np.ndarray) -> np.ndarray:
    """feat (n,k,f) · w (f,) -> (n,k)。这就是 make_scorer 那一行。"""
    return feat @ w


def accuracy(feat: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
    return float((scores(feat, w).argmax(axis=1) == y).mean())


def objective(accs: list[float]) -> float:
    """0.7*均值 + 0.3*最差。原封不动抄 optimize.py 第 3 条纪律：
    只优化均值会得到"平均不错但在 banking77 上崩盘"的权重。"""
    return 0.7 * (sum(accs) / len(accs)) + 0.3 * min(accs)


def search_weights(banks: dict[str, tuple[np.ndarray, np.ndarray]], dim: int,
                   generations: int = 80, lam: int = 14, restarts: int = 24,
                   seed: int = 7, verbose: bool = False) -> tuple[np.ndarray, list[dict]]:
    """(1+λ)-ES + σ 重启。参数与 optimize.py 完全一致。

    banks: {数据集名: (特征张量, 标签)}，每个数据集算一个"任务"，
    对应 palletizer 里的一条到达流。
    """
    rng = random.Random(seed)
    names = list(banks)

    def evaluate(w):
        return objective([accuracy(banks[n][0], w, banks[n][1]) for n in names])

    starts = [np.array([rng.uniform(0, 2) if i < dim * 0.75 else rng.uniform(-0.5, 1.0)
                        for i in range(dim)], np.float32) for _ in range(restarts)]
    scored = [(evaluate(w), w) for w in starts]
    best_obj, best_w = max(scored, key=lambda t: t[0])

    sigma, stale, history = 0.45, 0, []
    for gen in range(generations):
        cands = [best_w + np.array([rng.gauss(0, sigma) for _ in range(dim)], np.float32)
                 for _ in range(lam)]
        res = [(evaluate(w), w) for w in cands]
        obj, w = max(res, key=lambda t: t[0])
        if obj > best_obj:
            best_obj, best_w = obj, w
            sigma, stale = min(0.6, sigma * 1.15), 0
        else:
            sigma = max(0.06, sigma * 0.88)
            stale += 1
            if stale >= 8:                       # σ 重启，见 optimize.py 的注释
                sigma, stale = 0.40, 0
        history.append({"gen": gen, "sigma": round(sigma, 4), "obj": round(best_obj, 5)})
        if verbose and gen % 20 == 0:
            print(f"    gen {gen:3d} σ={sigma:.3f} obj={best_obj:.4f}")
    return best_w, history


def fit_temperature(feat: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
    """温度标定：Jev 卖点是"带校准概率"，光有 argmax 不算复现。
    在 dev 上最小化 NLL，网格搜一维，不值得上优化器。"""
    s = scores(feat, w)
    best_t, best_nll = 1.0, float("inf")
    for t in np.geomspace(0.5, 200, 60):
        z = s * t
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        nll = -logp[np.arange(len(y)), y].mean()
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t


def probabilities(feat: np.ndarray, w: np.ndarray, t: float) -> np.ndarray:
    z = scores(feat, w) * t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error。Jev 宣称概率是校准过的，这是能验证它的指标。"""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)
