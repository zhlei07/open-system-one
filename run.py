#!/usr/bin/env python3
"""主实验：同一批 10,000 个任务，四条前置编码器 + Fly 决策头 vs Qwen2.5-0.5B。

    10,000 tasks (sst2 / ag_news / emotion / banking77 各 2500)
         ├── static      ─┐
         ├── minilm      ─┤
         ├── modernbert  ─┼─ vector ─→ Fly 打分头 (w·φ, 8 权重由 (1+λ)-ES 搜) ─→ Decision
         ├── mbert-embed ─┘
         ├── Qwen2.5-0.5B-Instruct ────────────── 生成 ────────────────────→ Decision
         └── typesafe/jev (CF) ─────────────── 结构化判断 ──────────────────→ Decision

分阶段跑，中间结果落盘，任一段都能单独重跑：
    python run.py encode      # 编码 + 计时（最慢）
    python run.py heads       # 搜权重 + 出准确率
    python run.py qwen        # LLM 基线
    python run.py report      # 汇总
    python run.py all
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fly_head as fh
from data import CACHE, build
from encoders import all_encoders
from measure import measured

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
EMB = CACHE / "emb"
EMB.mkdir(exist_ok=True)

DATASETS = ["sst2", "ag_news", "emotion", "banking77"]
N_PROTO = 1500      # fit 里用来建类质心/kNN 的部分
# 剩下 500 条作为 ES 的目标函数。权重用**训练目标**选，dev 只做温度标定，
# eval 从头到尾只碰一次 —— 抄 select_best.py 的选型纪律。


def _rows(bundle, split, ds=None):
    return [t for t in bundle["splits"][split] if ds is None or t["dataset"] == ds]


# ---------------------------------------------------------------- 编码
def stage_encode(bundle, only=None, force=False):
    """注意：RAM 这一列只有**一个模型一个进程**才可比——同进程里顺序加载多个模型，
    后加载的会复用前一个释放的内存池，RSS 增量甚至可能是负数。
    所以正式测量走 `encode_isolated.sh`，它给每个编码器起一个干净的进程。"""
    for name, enc in all_encoders().items():
        if only and name not in only:
            continue
        dest = EMB / f"{name}.npz"
        if dest.exists() and not force:
            print(f"[encode] {name} 已存在，跳过")
            continue
        print(f"[encode] {name} ...", flush=True)
        with measured(name) as m:
            with m.loading():
                enc.load()

            # 选项描述：3 个模板 × 每个数据集的选项。离线成本，不计入服务延迟。
            desc = {}
            for ds in DATASETS:
                opts = bundle["options"][ds]
                desc[ds] = np.stack([enc.encode([t.format(o) for o in opts])
                                     for t in fh.TEMPLATES])

            # fit / dev 批量编码：离线训练成本，单独记
            t0 = time.perf_counter()
            fit = {ds: enc.encode([r["text"] for r in _rows(bundle, "fit", ds)]) for ds in DATASETS}
            dev = {ds: enc.encode([r["text"] for r in _rows(bundle, "dev", ds)]) for ds in DATASETS}
            offline_s = time.perf_counter() - t0

            # eval：**一条一条**编码，这才是在线服务的真实延迟
            ev = _rows(bundle, "eval")
            vecs = np.zeros((len(ev), enc.dim), np.float32)
            with m.run_window():
                for i, r in enumerate(ev):
                    with m.timed():
                        vecs[i] = enc.encode_one(r["text"])

        s = m.summary()
        s["offline_encode_s"] = round(offline_s, 1)
        s["params_m"] = enc.params_m
        s["dim"] = enc.dim
        np.savez_compressed(dest, eval=vecs,
                            **{f"fit_{d}": fit[d] for d in DATASETS},
                            **{f"dev_{d}": dev[d] for d in DATASETS},
                            **{f"desc_{d}": desc[d] for d in DATASETS})
        (OUT / f"meter_{name}.json").write_text(json.dumps(s, ensure_ascii=False, indent=1))
        print(f"  p50={s['lat_p50_ms']}ms p95={s['lat_p95_ms']}ms "
              f"RAM={s['model_rss_mb']}MB CPU={s['cpu_s_per_1k']}s/千条", flush=True)
        del enc


# ---------------------------------------------------------------- 决策头
def _banks(bundle, z, name):
    """每个数据集建一个 FeatureBank（用 fit 的前 N_PROTO 条建质心/kNN）。"""
    banks, labels = {}, {}
    for ds in DATASETS:
        fit_rows = _rows(bundle, "fit", ds)
        y = np.array([r["label"] for r in fit_rows])
        banks[ds] = fh.FeatureBank(bundle["options"][ds],
                                   z[f"fit_{ds}"][:N_PROTO], y[:N_PROTO], z[f"desc_{ds}"])
        labels[ds] = y
    return banks, labels


def stage_heads(bundle, only=None):
    results = {}
    for name in all_encoders():
        if only and name not in only:
            continue
        p = EMB / f"{name}.npz"
        if not p.exists():
            print(f"[heads] {name} 没有 embedding，跳过")
            continue
        print(f"[heads] {name} ...", flush=True)
        z = np.load(p)
        banks, fit_y = _banks(bundle, z, name)

        ev_rows = _rows(bundle, "eval")
        ev_idx = {ds: [i for i, r in enumerate(ev_rows) if r["dataset"] == ds] for ds in DATASETS}
        ev_vec, ev_y = z["eval"], np.array([r["label"] for r in ev_rows])
        dev_y = {ds: np.array([r["label"] for r in _rows(bundle, "dev", ds)]) for ds in DATASETS}

        # ---- few-shot：ES 目标跑在 fit 的后 500 条（建质心时没用过） ----
        es_feat = {ds: (banks[ds].few(z[f"fit_{ds}"][N_PROTO:]), fit_y[ds][N_PROTO:])
                   for ds in DATASETS}
        t0 = time.perf_counter()
        w_few, hist_few = fh.search_weights(es_feat, len(fh.FEW_FEATURES), verbose=True)
        es_s = time.perf_counter() - t0

        # ---- zero-shot：留一个数据集出来搜权重（LODO），保证目标任务真零样本 ----
        w_zero = {}
        for held in DATASETS:
            src = {ds: (banks[ds].zero(z[f"fit_{ds}"]), fit_y[ds])
                   for ds in DATASETS if ds != held}
            w_zero[held], _ = fh.search_weights(src, len(fh.ZERO_FEATURES), generations=60)

        # ---- 最终评测（eval 只碰这一次） ----
        per_ds = {}
        probs_all, ece_all = {}, {}
        for ds in DATASETS:
            idx = ev_idx[ds]
            q, y = ev_vec[idx], ev_y[idx]
            b = banks[ds]
            f_few, f_zero = b.few(q), b.zero(q)

            # 温度标定用 dev
            dq = z[f"dev_{ds}"]
            t_few = fh.fit_temperature(b.few(dq), w_few, dev_y[ds])
            t_zero = fh.fit_temperature(b.zero(dq), w_zero[ds], dev_y[ds])

            # 对照组：不搜权重，纯 argmax 余弦（= palletizer 的 score_baseline）
            w_cos = np.zeros(len(fh.ZERO_FEATURES), np.float32); w_cos[0] = 1.0
            w_proto = np.zeros(len(fh.FEW_FEATURES), np.float32); w_proto[0] = 1.0

            pf = fh.probabilities(f_few, w_few, t_few)
            pz = fh.probabilities(f_zero, w_zero[ds], t_zero)
            per_ds[ds] = {
                "n": len(idx),
                "fly_few": fh.accuracy(f_few, w_few, y),
                "fly_zero": fh.accuracy(f_zero, w_zero[ds], y),
                "baseline_cos": fh.accuracy(f_zero, w_cos, y),
                "baseline_proto": fh.accuracy(f_few, w_proto, y),
                "ece_few": fh.ece(pf, y),
                "ece_zero": fh.ece(pz, y),
                "temp_few": t_few, "temp_zero": t_zero,
            }
            # 参考列：embedding + 多分类逻辑回归（few-shot 上界）
            try:
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(max_iter=2000, C=10.0)
                lr.fit(z[f"fit_{ds}"], fit_y[ds])
                per_ds[ds]["logreg"] = float((lr.predict(q) == y).mean())
            except Exception as e:
                per_ds[ds]["logreg"] = None

        # ---- 决策头本身的延迟（单条，含 kNN 对 1500 条 fit 向量） ----
        head_lat = {}
        for mode in ("zero", "few"):
            ts = []
            for ds in DATASETS:
                b = banks[ds]
                w = w_zero[ds] if mode == "zero" else w_few
                for i in ev_idx[ds][:250]:
                    q = ev_vec[i:i + 1]
                    t0 = time.perf_counter()
                    f = b.zero(q) if mode == "zero" else b.few(q)
                    int((f @ w).argmax())
                    ts.append((time.perf_counter() - t0) * 1000)
            ts.sort()
            head_lat[mode] = {"p50_ms": round(ts[len(ts)//2], 4),
                              "p95_ms": round(ts[int(len(ts)*0.95)], 4)}

        macro = {k: float(np.mean([per_ds[d][k] for d in DATASETS]))
                 for k in ("fly_few", "fly_zero", "baseline_cos", "baseline_proto")}
        overall = {k: float(np.average([per_ds[d][k] for d in DATASETS],
                                       weights=[per_ds[d]["n"] for d in DATASETS]))
                   for k in ("fly_few", "fly_zero", "baseline_cos", "baseline_proto")}
        results[name] = {
            "per_dataset": per_ds, "macro": macro, "overall": overall,
            "worst_dataset": {k: min(per_ds[d][k] for d in DATASETS) for k in macro},
            "weights_few": {n: round(float(v), 4) for n, v in zip(fh.FEW_FEATURES, w_few)},
            "weights_zero": {h: {n: round(float(v), 4) for n, v in zip(fh.ZERO_FEATURES, w)}
                             for h, w in w_zero.items()},
            "es_seconds": round(es_s, 1),
            "head_latency": head_lat,
            "es_history_tail": hist_few[-3:],
        }
        print(f"  fly-zero {overall['fly_zero']*100:.1f}%  fly-few {overall['fly_few']*100:.1f}%  "
              f"(对照 cos {overall['baseline_cos']*100:.1f}%)", flush=True)

    prev = json.loads((OUT / "heads.json").read_text()) if (OUT / "heads.json").exists() else {}
    prev.update(results)
    (OUT / "heads.json").write_text(json.dumps(prev, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- Qwen
def stage_qwen(bundle, limit=None):
    from qwen_baseline import QwenClassifier
    ev = _rows(bundle, "eval")
    if limit:
        ev = ev[:limit]
    clf = QwenClassifier()
    with measured("qwen0.5b") as m:
        with m.loading():
            clf.load()
        correct = {d: [0, 0] for d in DATASETS}
        with m.run_window():
            for i, r in enumerate(ev):
                with m.timed():
                    pred = clf.predict(r["text"], bundle["options"][r["dataset"]])
                c = correct[r["dataset"]]
                c[0] += int(pred == r["label"]); c[1] += 1
                if i % 500 == 0:
                    done = sum(v[0] for v in correct.values()), sum(v[1] for v in correct.values())
                    print(f"  {i}/{len(ev)} acc={done[0]/max(done[1],1)*100:.1f}%", flush=True)
    s = m.summary()
    s["per_dataset"] = {d: {"acc": c[0] / c[1] if c[1] else None, "n": c[1]}
                        for d, c in correct.items()}
    s["overall_acc"] = sum(c[0] for c in correct.values()) / sum(c[1] for c in correct.values())
    s["macro_acc"] = float(np.mean([v["acc"] for v in s["per_dataset"].values() if v["acc"] is not None]))
    s["parse_fail"] = clf.parse_fail
    s["parse_fail_rate"] = clf.parse_fail / len(ev)
    (OUT / "qwen.json").write_text(json.dumps(s, ensure_ascii=False, indent=1))
    print(f"  Qwen 总体 {s['overall_acc']*100:.1f}%  p50={s['lat_p50_ms']}ms "
          f"解析失败率 {s['parse_fail_rate']*100:.2f}%")


def stage_qwen_probe(bundle, n=400):
    """干净进程里重测 Qwen 的 RAM 与延迟。

    准确率仍用全量 10k 那次的结果——准确率不受同机负载影响，
    但 RSS 增量和 per-item 延迟会。这两类指标必须分开取数。
    """
    from qwen_baseline import QwenClassifier
    ev = _rows(bundle, "eval")
    sample = []
    for d in DATASETS:
        sample += [r for r in ev if r["dataset"] == d][:n // len(DATASETS)]
    clf = QwenClassifier()
    with measured("qwen_probe") as m:
        with m.loading():
            clf.load()
        by_ds = {d: [] for d in DATASETS}
        with m.run_window():
            for r in sample:
                t0 = time.perf_counter()
                with m.timed():
                    clf.predict(r["text"], bundle["options"][r["dataset"]])
                by_ds[r["dataset"]].append((time.perf_counter() - t0) * 1000)
    out = m.summary()
    out["per_dataset_p50_ms"] = {d: round(sorted(v)[len(v)//2], 1) for d, v in by_ds.items() if v}
    (OUT / "qwen_probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[qwenprobe] p50={out['lat_p50_ms']}ms p95={out['lat_p95_ms']}ms "
          f"RAM={out['model_rss_mb']}MB CPU={out['cpu_s_per_1k']}s/千条 "
          f"分任务 {out['per_dataset_p50_ms']}")


# ---------------------------------------------------------------- Jev
def stage_jev(bundle, limit=None, concurrency=8):
    """准确率跑并发（省时间，对准确率无影响），延迟单独用顺序小样本测。

    两件事必须分开量：并发下的 per-call 延迟会被服务端排队和限流拉长，
    拿它当"Jev 有多快"是不诚实的；而顺序跑 10,000 条要一个多小时。
    所以准确率用并发全量，延迟用 `run.py jevlat` 的顺序样本。
    """
    from concurrent.futures import ThreadPoolExecutor
    from jev_adapter import JevClient

    probe = JevClient()
    ok, msg = probe.available()
    if not ok:
        (OUT / "jev.json").write_text(json.dumps(
            {"status": "unavailable", "detail": msg}, ensure_ascii=False, indent=1))
        print(f"[jev] 不可用：{msg}")
        return

    ev = _rows(bundle, "eval")
    if limit:
        ev = ev[:limit]
    print(f"[jev] {len(ev)} 条，并发 {concurrency}", flush=True)

    clients = [JevClient() for _ in range(concurrency)]
    results = [None] * len(ev)
    errors = []

    def work(args):
        i, r = args
        c = clients[i % concurrency]
        t0 = time.perf_counter()
        try:
            pred, probs = c.predict(r["text"], bundle["options"][r["dataset"]])
        except Exception as e:
            errors.append(f"{i}: {e}")
            return
        results[i] = (pred, probs, (time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for n, _ in enumerate(ex.map(work, enumerate(ev))):
            if n % 1000 == 0 and n:
                done = [r for r in results if r]
                print(f"  {n}/{len(ev)}  已用 {time.perf_counter()-t0:.0f}s", flush=True)
    wall = time.perf_counter() - t0

    per_ds = {d: {"correct": 0, "n": 0} for d in DATASETS}
    lat, probs_by_ds, y_by_ds = [], {d: [] for d in DATASETS}, {d: [] for d in DATASETS}
    for r, row in zip(results, ev):
        d = row["dataset"]
        per_ds[d]["n"] += 1
        if r is None:                       # 请求失败算答错，不静默丢弃
            continue
        pred, probs, ms = r
        per_ds[d]["correct"] += int(pred == row["label"])
        lat.append(ms)
        probs_by_ds[d].append(probs)
        y_by_ds[d].append(row["label"])

    tokens = sum(c.total_input_tokens for c in clients)
    lat.sort()
    out = {
        "n": len(ev),
        "concurrency": concurrency,
        "wall_s": round(wall, 1),
        "errors": len(errors), "error_sample": errors[:5],
        "lat_concurrent_p50_ms": round(lat[len(lat)//2], 1) if lat else None,
        "lat_concurrent_p95_ms": round(lat[int(len(lat)*0.95)], 1) if lat else None,
        "per_dataset": {d: {"acc": v["correct"] / v["n"] if v["n"] else None, "n": v["n"]}
                        for d, v in per_ds.items()},
        "overall_acc": sum(v["correct"] for v in per_ds.values()) / max(1, sum(v["n"] for v in per_ds.values())),
        "macro_acc": float(np.mean([v["correct"] / v["n"] for v in per_ds.values() if v["n"]])),
        "input_tokens": tokens,
        "cost_usd": round(tokens / 1e6 * 0.042, 4),
    }
    for d in DATASETS:                      # Jev 宣称概率校准过——这里验一下
        if probs_by_ds[d]:
            out["per_dataset"][d]["ece"] = fh.ece(np.array(probs_by_ds[d]),
                                                  np.array(y_by_ds[d]))
    (OUT / "jev.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"  Jev 总体 {out['overall_acc']*100:.1f}%  {wall:.0f}s  "
          f"${out['cost_usd']}  失败 {len(errors)}", flush=True)


def stage_jev_latency(bundle, n=300):
    """顺序打 n 条，测干净的 per-call 延迟。跑之前确认本机没有别的重活。"""
    from jev_adapter import JevClient
    c = JevClient()
    ev = _rows(bundle, "eval")
    sample = []
    for d in DATASETS:                      # 每个数据集等量，避免被 banking77 的长 prompt 带偏
        sample += [r for r in ev if r["dataset"] == d][:n // len(DATASETS)]
    by_ds = {d: [] for d in DATASETS}
    for r in sample:
        t0 = time.perf_counter()
        try:
            c.predict(r["text"], bundle["options"][r["dataset"]])
        except Exception:
            continue
        by_ds[r["dataset"]].append((time.perf_counter() - t0) * 1000)
    allv = sorted(v for vs in by_ds.values() for v in vs)
    out = {
        "n": len(allv), "mode": "sequential",
        "lat_p50_ms": round(allv[len(allv)//2], 1),
        "lat_p95_ms": round(allv[int(len(allv)*0.95)], 1),
        "per_dataset_p50_ms": {d: round(sorted(v)[len(v)//2], 1) for d, v in by_ds.items() if v},
    }
    (OUT / "jev_latency.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[jevlat] 顺序 p50={out['lat_p50_ms']}ms p95={out['lat_p95_ms']}ms "
          f"分任务 {out['per_dataset_p50_ms']}")


if __name__ == "__main__":
    cmds = sys.argv[1:] or ["all"]
    only = os.environ.get("ONLY", "").split(",") if os.environ.get("ONLY") else None
    b = build()
    if "all" in cmds or "encode" in cmds:
        stage_encode(b, only, force=bool(os.environ.get("FORCE_ENCODE")))
    if "all" in cmds or "heads" in cmds:
        stage_heads(b, only)
    if "all" in cmds or "qwen" in cmds:
        stage_qwen(b, limit=int(os.environ.get("QWEN_LIMIT", 0)) or None)
    if "all" in cmds or "jev" in cmds:
        stage_jev(b, limit=int(os.environ.get("JEV_LIMIT", 0)) or None,
                  concurrency=int(os.environ.get("JEV_CONC", 8)))
    if "qwenprobe" in cmds:
        stage_qwen_probe(b)
    if "jevlat" in cmds:
        stage_jev_latency(b)
