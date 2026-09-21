# 数据集来源与许可

本仓库**不重新分发任何数据集正文**，只提供切分索引与构建脚本（`data.py`）。
运行 `python data.py` 会从 Hugging Face Hub 下载原数据集并在本地重建 10,000 条任务集。

| 数据集 | HF 仓库 | 许可 |
|---|---|---|
| SST-2 | `stanfordnlp/sst2` | 见仓库卡片 |
| AG News | `fancyzhx/ag_news` | 见仓库卡片 |
| Emotion | `dair-ai/emotion` | 见仓库卡片（非商业使用条款，注意） |
| BANKING77 | `mteb/banking77`（`PolyAI/banking77` 的 parquet 镜像） | CC-BY-4.0 |

切分用固定随机种子 `SEED = 20260921` 生成，任何人重跑 `data.py` 都会得到
逐条一致的 fit / dev / eval，因此结果可以逐条复现。

模型：
- `minishlab/potion-base-8M`、`sentence-transformers/all-MiniLM-L6-v2`、
  `answerdotai/ModernBERT-base`、`nomic-ai/modernbert-embed-base`、
  `MoritzLaurer/ModernBERT-{base,large}-zeroshot-v2.0`、`Qwen/Qwen2.5-{0.5B,1.5B}-Instruct`
  —— 各自遵循其 HF 仓库的许可。
- `typesafe/jev` 经 Cloudflare Workers AI 调用，遵循 TypeSafe 的条款，本仓库不包含其任何权重或输出以外的内容。
