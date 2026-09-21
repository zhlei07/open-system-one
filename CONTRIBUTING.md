# 复现与贡献

## 复现全部结果

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  torch transformers sentence-transformers model2vec datasets scikit-learn psutil

.venv/bin/python data.py            # 重建 10,000 条任务集（固定种子，逐条一致）
./encode_isolated.sh                # 四个双塔编码器，**一个模型一个进程**
.venv/bin/python run.py heads       # Fly 打分头：搜权重 + 准确率
.venv/bin/python run.py qwen qwenprobe
.venv/bin/python cross_v2.py --model base --ds sst2,emotion,ag_news --n 2500 --templates 1
.venv/bin/python cross_v2.py --model base --ds banking77 --n 2500 --templates 1 --topk 20
.venv/bin/python fuse.py && .venv/bin/python distill.py && .venv/bin/python frontier.py
```

Jev 那几列需要 Cloudflare 凭证（见 `.env.example`）：

```bash
export CF_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=...
.venv/bin/python run.py jev jevlat   # 裸标签名条件
.venv/bin/python fair_jev.py         # 丰富描述条件
```

10,000 条 Jev 调用约 $0.3（$0.042 / 百万 input token，output 免费）。

## 测量纪律

这几条是踩过坑之后定下来的，改动时请一并遵守，否则数字会**悄悄**变假：

1. **RAM 必须一个模型一个进程。** 同进程顺序加载多个模型，后加载的会复用前一个
   释放的内存池，RSS 增量甚至是负数。
2. **延迟用 batch=1。** 在线决策攒不了批，报批处理吞吐是自欺。
3. **不要在计时的同时跑别的重活。** 我们有一次让 10k 并发 HTTP 和本地编码重叠，
   把 ModernBERT 的 p50 从 15.7ms 拖到 29.7ms —— 接近一倍的误差，而且不报错。
4. **解析失败计入错误，并单独报失败率。** 静默丢弃会让生成式方案凭空涨几个点。
5. **权重用训练目标选，不用验证分数选。** 拿验证分数挑超参等于把验证集变成训练集。
6. **零样本就要真零样本。** 模板/权重一旦用目标任务的标签挑过，就不再是零样本；
   本仓库统一用留一数据集（LODO）来搜，并且把"用目标任务标签搜"的结果
   单独标成「作弊上界」。

## 已知局限

见 `REPORT.md` 末尾的「局限」一节。最重要的两条：单随机种子、
Jev 的延迟以本机到 Cloudflare 的网络往返为主。

## 关于产物文件名

`cache/cross_*.npz`（base 骨干、单模板）由早期脚本生成，功能上等价于
`cross_v2.py --model base --templates 1` 的输出（`cache/xc_base_*.npz`）。
分析脚本（`fuse.py` / `distill.py` / `frontier.py`）两种命名都认。
从零复现时统一用 `cross_v2.py` 即可。
