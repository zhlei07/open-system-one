"""Fly decision head — standalone inference. Needs only numpy + model2vec.

    pip install numpy model2vec
    python inference.py banking77 "my card hasn't arrived yet"

The head itself is the eight floats in config.json. Everything else in heads/*.npz
is offline-computed state (class centroids, 1500 fit vectors for the kNN features,
option-description vectors) — no gradients, no training loop, no GPU.
"""
import json
import sys
from pathlib import Path

import numpy as np
from model2vec import StaticModel

HERE = Path(__file__).parent
CFG = json.loads((HERE / "config.json").read_text())
KNN_K = 5


def _norm(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def features(q, h):
    """(1, d) -> (1, n_options, 8). Same eight features used to search the weights."""
    cent, fit, lab = h["centroids"], h["fit_emb"], h["fit_lab"]
    k = cent.shape[0]
    proto = q @ cent.T
    desc = q @ h["desc_emb"].T
    sim = q @ fit.T
    knn1 = np.zeros((1, k), np.float32)
    knnk = np.zeros((1, k), np.float32)
    tight = np.zeros((1, k), np.float32)
    for c in range(k):
        idx = np.where(lab == c)[0]
        if len(idx) == 0:
            continue
        s = sim[:, idx]
        kk = min(KNN_K, s.shape[1])
        top = np.partition(s, -kk, axis=1)[:, -kk:]
        knn1[:, c], knnk[:, c], tight[:, c] = top.max(1), top.mean(1), -s.std(1)
    margin = proto - proto.mean(1, keepdims=True)
    qc, cc = _norm(q - h["global_mean"]), _norm(cent - h["global_mean"])
    white = qc @ cc.T
    prior = np.tile(h["log_prior"] / 5.0, (1, 1))
    return np.stack([proto, desc, knn1, knnk, margin, prior, white, tight], -1)


class FlyHead:
    def __init__(self, dataset):
        self.ds = dataset
        self.opts = CFG["datasets"][dataset]["options"]
        self.w = np.array(CFG["weights"], np.float32)
        self.h = dict(np.load(HERE / "heads" / f"{dataset}.npz"))
        self.enc = StaticModel.from_pretrained(CFG["encoder"])

    def __call__(self, text):
        q = _norm(np.asarray(self.enc.encode([text]), np.float32))
        s = (features(q, self.h) @ self.w)[0]
        p = np.exp(s - s.max()); p /= p.sum()
        order = np.argsort(-s)
        return [(self.opts[i], float(p[i])) for i in order]


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "banking77"
    text = " ".join(sys.argv[2:]) or "my card hasn't arrived yet"
    for name, prob in FlyHead(ds)(text)[:5]:
        print(f"  {prob:6.1%}  {name}")
