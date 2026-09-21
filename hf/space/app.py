"""open-system-one — side-by-side demo.

Two ways to make a typed decision without a generative LLM, with real measured latency:

  1. Static embedding + an 8-float head  — 0.1 ms, needs ~2k labelled examples
  2. Cross-encoder NLI                   — 50-600 ms, zero-shot, any options you type

The second one is the Jev-shaped use case: you supply the options at request time.
"""
import json
import time

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from model2vec import StaticModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_REPO = "dylantom2012/fly-head-potion-8m"
XENC = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
HYPOTHESIS = "This example is about {}."

local = snapshot_download(MODEL_REPO)
CFG = json.loads(open(f"{local}/config.json").read())
STATIC = StaticModel.from_pretrained(CFG["encoder"])
W = np.array(CFG["weights"], np.float32)
HEADS = {d: dict(np.load(f"{local}/heads/{d}.npz")) for d in CFG["datasets"]}

tok = AutoTokenizer.from_pretrained(XENC)
xmodel = AutoModelForSequenceClassification.from_pretrained(XENC).eval()
torch.set_grad_enabled(False)
ENT = next(i for i, l in xmodel.config.id2label.items() if "entail" in l.lower())


def _n(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def fly(text, ds):
    h = HEADS[ds]; opts = CFG["datasets"][ds]["options"]
    t0 = time.perf_counter()
    q = _n(np.asarray(STATIC.encode([text]), np.float32))
    cent, fit, lab = h["centroids"], h["fit_emb"], h["fit_lab"]
    k = cent.shape[0]
    proto, desc, sim = q @ cent.T, q @ h["desc_emb"].T, q @ fit.T
    knn1 = np.zeros((1, k), np.float32); knnk = np.zeros((1, k), np.float32)
    tight = np.zeros((1, k), np.float32)
    for c in range(k):
        idx = np.where(lab == c)[0]
        if not len(idx):
            continue
        s = sim[:, idx]; kk = min(5, s.shape[1])
        top = np.partition(s, -kk, axis=1)[:, -kk:]
        knn1[:, c], knnk[:, c], tight[:, c] = top.max(1), top.mean(1), -s.std(1)
    qc, cc = _n(q - h["global_mean"]), _n(cent - h["global_mean"])
    f = np.stack([proto, desc, knn1, knnk, proto - proto.mean(1, keepdims=True),
                  np.tile(h["log_prior"] / 5.0, (1, 1)), qc @ cc.T, tight], -1)
    s = (f @ W)[0]
    ms = (time.perf_counter() - t0) * 1000
    p = np.exp(s - s.max()); p /= p.sum()
    return {opts[i]: float(p[i]) for i in np.argsort(-s)[:5]}, ms


def cross(text, options):
    opts = [o.strip() for o in options.split("\n") if o.strip()]
    if len(opts) < 2:
        return {"need at least 2 options": 1.0}, 0.0
    t0 = time.perf_counter()
    enc = tok([text] * len(opts), [HYPOTHESIS.format(o) for o in opts],
              return_tensors="pt", padding=True, truncation=True, max_length=256)
    s = xmodel(**enc).logits[:, ENT].float().numpy()
    ms = (time.perf_counter() - t0) * 1000
    p = np.exp(s - s.max()); p /= p.sum()
    return {opts[i]: float(p[i]) for i in np.argsort(-s)[:5]}, ms


def run_bench(text, ds):
    a, ta = fly(text, ds)
    b, tb = cross(text, "\n".join(CFG["datasets"][ds]["options"]))
    return a, f"**{ta:.2f} ms** · 82 MB · 8-float head · needs ~2k labels", \
           b, f"**{tb:.0f} ms** · 149M cross-encoder · zero-shot"


DESC = """# open-system-one

Typed decisions without a generative model. Two open CPU-only stacks, with the latency
actually measured in your browser session.

On a 10,000-item benchmark these score **78.2 %** (static + 8-float head, 0.1 ms) and
**78.7 %** (cross-encoder, zero-shot) macro — against **79.3 %** for the hosted
`typesafe/jev`. Full numbers, the eight ideas that failed, and every raw output:
[github.com/zhlei07/open-system-one](https://github.com/zhlei07/open-system-one)

*Latency here is higher than the benchmark: free HF CPU hardware, cold caches.*
"""

with gr.Blocks(title="open-system-one") as demo:
    gr.Markdown(DESC)
    with gr.Tab("Benchmark tasks"):
        with gr.Row():
            t1 = gr.Textbox(label="Text", value="I still haven't received the card I ordered", lines=3)
            d1 = gr.Dropdown(list(CFG["datasets"]), value="banking77", label="Task")
        go1 = gr.Button("Decide", variant="primary")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Static embedding + 8-float head")
                o1 = gr.Label(num_top_classes=5); m1 = gr.Markdown()
            with gr.Column():
                gr.Markdown("### Cross-encoder (zero-shot)")
                o2 = gr.Label(num_top_classes=5); m2 = gr.Markdown()
        go1.click(run_bench, [t1, d1], [o1, m1, o2, m2])
    with gr.Tab("Your own options"):
        gr.Markdown("Zero-shot: type any options, one per line. No labels, no training — "
                    "this is the Jev-shaped use case.")
        t2 = gr.Textbox(label="Text", value="The deploy failed twice and customers are seeing 500s.", lines=3)
        op = gr.Textbox(label="Options (one per line)",
                        value="billing question\ntechnical outage\nsales enquiry\naccount access", lines=5)
        go2 = gr.Button("Decide", variant="primary")
        o3 = gr.Label(num_top_classes=5); m3 = gr.Markdown()
        go2.click(lambda t, o: (lambda r: (r[0], f"**{r[1]:.0f} ms** · zero-shot"))(cross(t, o)),
                  [t2, op], [o3, m3])

demo.launch()
