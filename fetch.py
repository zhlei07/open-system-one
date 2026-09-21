"""预下载：4 个数据集 + 4 个编码器 + Qwen2.5-0.5B-Instruct。"""
import os
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from huggingface_hub import snapshot_download
from datasets import load_dataset

DATASETS = ["PolyAI/banking77", "fancyzhx/ag_news", "stanfordnlp/sst2", "dair-ai/emotion"]
MODELS = [
    "minishlab/potion-base-8M",              # static embedding (model2vec)
    "sentence-transformers/all-MiniLM-L6-v2",
    "answerdotai/ModernBERT-base",           # 原始 MLM，未做句向量训练
    "nomic-ai/modernbert-embed-base",        # ModernBERT 的句向量版（公平对照）
    "Qwen/Qwen2.5-0.5B-Instruct",
]
for d in DATASETS:
    try:
        load_dataset(d)
        print("dataset OK", d, flush=True)
    except Exception as e:
        print("dataset FAIL", d, repr(e)[:200], flush=True)
for m in MODELS:
    try:
        p = snapshot_download(m, ignore_patterns=["*.onnx", "*.msgpack", "*.h5", "*openvino*"])
        print("model OK", m, flush=True)
    except Exception as e:
        print("model FAIL", m, repr(e)[:200], flush=True)
