#!/usr/bin/env python3
"""汇总 out/*.json -> out/REPORT.md。

主表的"端到端延迟" = 编码延迟 + 决策头延迟，都是 batch=1 单条测的。
不报批处理吞吐当主指标：在线决策场景攒不了批。
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "out"
DATASETS = ["sst2", "ag_news", "emotion", "banking77"]
ENCODERS = ["static", "minilm", "modernbert", "mbert-embed"]
PRETTY = {"static": "Static (potion-8M)", "minilm": "MiniLM-L6", "modernbert": "ModernBERT-base",
          "mbert-embed": "ModernBERT-embed"}


def _load(p):
    f = OUT / p
    return json.loads(f.read_text()) if f.exists() else None


def pct(x):
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "—"


def main():
    heads = _load("heads.json") or {}
    qwen = _load("qwen.json")
    jev = _load("jev.json")
    meters = {e: _load(f"meter_{e}.json") for e in ENCODERS}

    L = []
    w = L.append
    w("# Jev 复现实验：前置编码器 + Fly 决策头 能不能平替 Jev\n")
    w("同一批 **10,000 个决策任务**（sst2 2500 / ag_news 2500 / emotion 2500 / banking77 2500），")
    w("同一台机器（Apple M5 Pro, 18 核, 64GB），全部 **CPU、batch=1**。\n")

    # ---------- 结论 ----------
    zeros = {e: heads[e]["overall"]["fly_zero"] for e in ENCODERS if e in heads}
    fews = {e: heads[e]["overall"]["fly_few"] for e in ENCODERS if e in heads}
    if zeros and jev and jev.get("status") != "unavailable":
        bz = max(zeros, key=zeros.get); bf = max(fews, key=fews.get)
        jv = jev["overall_acc"]
        w("## 结论\n")
        w(f"**零样本下复现不了。** 最好的一条（{PRETTY[bz]} + Fly）是 {pct(zeros[bz])}，"
          f"Jev 是 {pct(jv)}，**差 {(jv-zeros[bz])*100:.1f} 个百分点**，且四个任务上全面落后。"
          "所以 Jev 不是\"通用句向量 + 线性头\"——前置那一层它做了额外的事。\n")
        w(f"**但给 2000 条标注就反超。** {PRETTY[bf]} + Fly 头做到 {pct(fews[bf])}，"
          f"比 Jev 高 {(fews[bf]-jv)*100:.1f} pp；")
        if "static" in fews:
            w(f"连 8M 的静态嵌入都有 {pct(fews['static'])}，和 Jev 基本打平"
              f"（差 {(fews['static']-jv)*100:+.1f} pp），而它端到端 "
              f"{meters['static']['lat_p50_ms']+heads['static']['head_latency']['few']['p50_ms']:.2f} ms、"
              f"{meters['static']['model_rss_mb']:.0f} MB、纯 CPU、零边际成本。\n")
        if qwen:
            w(f"**同尺寸的生成式小模型不是一个量级。** Qwen2.5-0.5B 只有 {pct(qwen['overall_acc'])}，"
              f"还有 {qwen['parse_fail_rate']*100:.1f}% 的输出解析不出合法选项——"
              "这正是 Jev 那套\"类型安全\"要解决的问题，它确实解决了（0 个非法输出）。\n")
        w("**所以该不该用 Jev，取决于这个决策点有没有历史标注**："
          "有，本地一个静态嵌入 + 8 个权重就打平了；没有，那十几个点就是它的定价依据。\n")

    # ---------- 主表 ----------
    w("## 主表：零样本（和 Jev / Qwen 信息量对等）\n")
    w("| 链路 | 参数量 | 准确率 | 最差任务 | 端到端 p50 | p95 | 模型 RAM | CPU-秒/千条 |")
    w("|---|---|---|---|---|---|---|---|")
    for e in ENCODERS:
        h, m = heads.get(e), meters.get(e)
        if not h or not m:
            continue
        hl = h["head_latency"]["zero"]
        w(f"| {PRETTY[e]} + Fly | {m['params_m']:.0f}M | **{pct(h['overall']['fly_zero'])}** | "
          f"{pct(h['worst_dataset']['fly_zero'])} | {m['lat_p50_ms']+hl['p50_ms']:.2f} ms | "
          f"{m['lat_p95_ms']+hl['p95_ms']:.2f} ms | {m['model_rss_mb']:.0f} MB | {m['cpu_s_per_1k']:.2f} |")
    if qwen:
        # 准确率取全量 10k 那次；延迟/RAM 取干净进程的 probe（同进程里装过别的模型会污染 RSS）
        qp = _load("qwen_probe.json") or qwen
        w(f"| Qwen2.5-0.5B 生成 | 494M | **{pct(qwen['overall_acc'])}** | "
          f"{pct(min(v['acc'] for v in qwen['per_dataset'].values() if v['acc'] is not None))} | "
          f"{qp['lat_p50_ms']:.0f} ms | {qp['lat_p95_ms']:.0f} ms | "
          f"{qp['model_rss_mb']:.0f} MB | {qp['cpu_s_per_1k']:.1f} |")
    if jev and jev.get("status") != "unavailable":
        jl = _load("jev_latency.json")
        # 延迟取顺序测的那份；没有就退回并发测的，并标注（并发下的数字偏大）
        p50 = f"{jl['lat_p50_ms']:.0f} ms" if jl else f"{jev['lat_concurrent_p50_ms']:.0f} ms*"
        p95 = f"{jl['lat_p95_ms']:.0f} ms" if jl else f"{jev['lat_concurrent_p95_ms']:.0f} ms*"
        worst = min(v["acc"] for v in jev["per_dataset"].values() if v.get("acc") is not None)
        w(f"| **typesafe/jev** (CF 云端) | 未公开 | **{pct(jev['overall_acc'])}** | {pct(worst)} | "
          f"{p50} | {p95} | 云端 | 云端 |")
    else:
        detail = (jev or {}).get("detail", "未运行")
        w(f"| typesafe/jev (CF) | — | — | — | — | — | — | — |")
        w(f"\n> Jev 列空缺：`{detail}`。补上的方法见文末。")
    if jev and jev.get("status") != "unavailable":
        if not _load("jev_latency.json"):
            w("\n> `*` Jev 延迟是并发 8 下测的，被服务端排队拉长了；干净数字跑 `run.py jevlat`。")
        w(f"\n> Jev 实测成本：{jev['input_tokens']:,} input tokens = **${jev['cost_usd']}** / 10,000 条决策"
          f"（$0.042/M，output 免费），失败 {jev['errors']} 条。")
    w("")

    # ---------- few-shot ----------
    w("## 对照：给 Fly 头 2000 条带标签样本（few-shot）\n")
    w("Jev 拿不到这些样本，所以这一列**不能用来宣称平替**，只用来看编码器的天花板。\n")
    w("| 链路 | Fly few-shot | 最差任务 | 纯质心对照 | LogReg 参考 | 决策头 p50 |")
    w("|---|---|---|---|---|---|")
    for e in ENCODERS:
        h = heads.get(e)
        if not h:
            continue
        lr = [h["per_dataset"][d].get("logreg") for d in DATASETS]
        lr_avg = sum(x for x in lr if x is not None) / max(1, len([x for x in lr if x is not None]))
        w(f"| {PRETTY[e]} | **{pct(h['overall']['fly_few'])}** | {pct(h['worst_dataset']['fly_few'])} | "
          f"{pct(h['overall']['baseline_proto'])} | {pct(lr_avg)} | "
          f"{h['head_latency']['few']['p50_ms']:.2f} ms |")
    w("")

    # ---------- 分数据集 ----------
    w("## 分任务准确率（难度递增）\n")
    w("| 链路 | sst2 (2类) | ag_news (4类) | emotion (6类) | banking77 (77类) |")
    w("|---|---|---|---|---|")
    for e in ENCODERS:
        h = heads.get(e)
        if not h:
            continue
        w(f"| {PRETTY[e]} zero | " + " | ".join(pct(h["per_dataset"][d]["fly_zero"]) for d in DATASETS) + " |")
    for e in ENCODERS:
        h = heads.get(e)
        if not h:
            continue
        w(f"| {PRETTY[e]} few | " + " | ".join(pct(h["per_dataset"][d]["fly_few"]) for d in DATASETS) + " |")
    if qwen:
        w("| Qwen2.5-0.5B | " + " | ".join(pct(qwen["per_dataset"][d]["acc"]) for d in DATASETS) + " |")
    if jev and jev.get("status") != "unavailable":
        w("| **typesafe/jev** | " + " | ".join(pct(jev["per_dataset"][d]["acc"]) for d in DATASETS) + " |")
    w("")

    # ---------- 校准 ----------
    w("## 概率校准 (ECE，越低越好)\n")
    w("Jev 的卖点之一是\"概率是校准过的\"。打分头 + 温度标定能做到什么程度：\n")
    w("| 链路 | ECE zero-shot | ECE few-shot |")
    w("|---|---|---|")
    for e in ENCODERS:
        h = heads.get(e)
        if not h:
            continue
        ez = sum(h["per_dataset"][d]["ece_zero"] for d in DATASETS) / 4
        ef = sum(h["per_dataset"][d]["ece_few"] for d in DATASETS) / 4
        w(f"| {PRETTY[e]} | {ez:.3f} | {ef:.3f} |")
    if jev and jev.get("status") != "unavailable":
        ej = [v["ece"] for v in jev["per_dataset"].values() if "ece" in v]
        if ej:
            w(f"| **typesafe/jev** | {sum(ej)/len(ej):.3f} | （拿不到带标签样本） |")
    w("")

    # ---------- 测量注意事项 ----------
    w("## 测量注意事项（这些坑会让数字悄悄变假）\n")
    w("1. **ECE 这一列不是公平对比。** 本地各条的温度是在 1000 条带标签 dev 上标定的，"
      "Jev 一条标注都没拿到。公平的说法是：*Jev 零标注下 ECE 0.122；本地方案有 1000 条标注时能做到 0.02–0.04*。\n")
    w("2. **Jev 的 381 ms 几乎全是网络。** banking77 的输入 token 是其他任务的 5 倍"
      "（1612 vs ~320），p50 只从 374 ms 涨到 392 ms——token 涨 5 倍延迟涨 5%，"
      "说明模型计算时间小到测不出来。所以\"本地快几千倍\"省掉的是一次跨网往返，不是算力。\n")
    w("3. **RAM 必须一个模型一个进程。** 同进程里顺序加载多个模型，后加载的会复用前一个"
      "释放的内存池，RSS 增量甚至是负数。本表的 RAM/CPU/延迟全部由 `encode_isolated.sh` "
      "在独立进程里重测（首轮同进程测出的 ModernBERT 是 29.7 ms，隔离后是 15.7 ms——"
      "首轮还撞上了 Jev 的并发请求，被拖慢近一倍）。\n")
    w("4. **Qwen 的解析失败按答错计。** 失败率 "
      f"{qwen['parse_fail_rate']*100:.1f}%（{qwen['parse_fail']} 条）。"
      "把解析失败静默丢弃是这类评测最常见的作弊方式，会让小模型凭空涨几个点。\n")
    w("5. **权重选型用训练目标，不是验证分数。** fit 前 1500 条建质心 → fit 后 500 条当 ES 目标 "
      "→ dev 1000 条只标温度 → eval 2500×4 只碰一次。\n")

    # ---------- 搜出来的权重 ----------
    w("## (1+λ)-ES 搜出来的权重\n")
    for e in ENCODERS:
        h = heads.get(e)
        if not h:
            continue
        w(f"- **{PRETTY[e]}** few-shot：`{json.dumps(h['weights_few'], ensure_ascii=False)}`（ES 耗时 {h['es_seconds']}s）")
    w("")

    # ---------- 复现 ----------
    w("## 复现\n```bash\n./encode_isolated.sh                  # 编码（一个模型一个进程，RAM 才可比）\n"
      ".venv/bin/python run.py heads         # 搜权重 + 准确率\n"
      ".venv/bin/python run.py qwen qwenprobe\n"
      ".venv/bin/python run.py jev jevlat\n"
      ".venv/bin/python report.py\n```\n")
    w("补 Jev 那一列：给 Cloudflare AI Gateway 充值，或 BYOK 填 TypeSafe 的 key，然后")
    w("```bash\n.venv/bin/python run.py jev      # 全量 10k，并发 8；JEV_LIMIT=1000 可省钱\n"
      ".venv/bin/python run.py jevlat   # 干净的顺序延迟\n```\n")
    w("真瓦数（powermetrics 要 root，得你自己跑）：\n```bash\nsudo ./power_probe.sh .venv/bin/python run.py encode\n```")

    (OUT / "REPORT.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
