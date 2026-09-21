"""测量：延迟 / RAM / CPU / 功耗代理。

一条纪律：**per-item 计时用 batch=1**。
批处理能把 ModernBERT 的吞吐拉到接近 MiniLM，但在线决策场景里请求是一条条到的
（和 palletizer 的在线到达流是同一个道理：能攒批的算法在现实里直接作废）。
所以主表用 batch=1 的 p50/p95，批处理吞吐单独列，不混在一起。

功耗：powermetrics 要 root，这里拿不到。用 **CPU-秒** 当代理并明确标注——
它抓得住"靠烧 18 个核换来的快"这种假便宜，抓不住 GPU/ANE 的能耗。
要真瓦数就跑 ./power_probe.sh，它把 powermetrics 套在一次 run 外面。
"""
from __future__ import annotations

import resource
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict

import psutil

_PROC = psutil.Process()


def _cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


class _RSSSampler(threading.Thread):
    """峰值 RSS。用采样而不是 ru_maxrss，因为 ru_maxrss 是整个进程生命周期的高水位，
    分不开"加载编码器"和"跑推理"这两段。"""
    def __init__(self, hz: float = 20.0):
        super().__init__(daemon=True)
        self.interval = 1.0 / hz
        self.peak = 0
        self._done = threading.Event()

    def run(self):
        while not self._done.is_set():
            try:
                self.peak = max(self.peak, _PROC.memory_info().rss)
            except psutil.Error:
                break
            self._done.wait(self.interval)

    def stop(self) -> int:
        self._done.set()
        self.join(timeout=1.0)
        return self.peak


@dataclass
class Meter:
    """一次 run 的全部计量。latencies 单位 ms。"""
    label: str
    latencies_ms: list[float] = field(default_factory=list)
    load_wall_s: float = 0.0
    load_rss_delta_mb: float = 0.0
    run_cpu_s: float = 0.0
    run_wall_s: float = 0.0
    peak_rss_mb: float = 0.0
    baseline_rss_mb: float = 0.0

    def summary(self) -> dict:
        lat = sorted(self.latencies_ms)
        n = len(lat) or 1
        def pct(p):
            return lat[min(len(lat) - 1, int(p * len(lat)))] if lat else 0.0
        return {
            "label": self.label,
            "n": len(lat),
            "lat_mean_ms": round(statistics.fmean(lat), 3) if lat else 0.0,
            "lat_p50_ms": round(pct(0.50), 3),
            "lat_p95_ms": round(pct(0.95), 3),
            "lat_p99_ms": round(pct(0.99), 3),
            "throughput_item_s": round(len(lat) / self.run_wall_s, 2) if self.run_wall_s else 0.0,
            # CPU-秒/千条：这一列是"功耗代理"。并行加速不会让它变小。
            "cpu_s_per_1k": round(self.run_cpu_s / n * 1000, 3),
            "cpu_efficiency": round(self.run_cpu_s / self.run_wall_s, 2) if self.run_wall_s else 0.0,
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "model_rss_mb": round(self.load_rss_delta_mb, 1),
            "load_wall_s": round(self.load_wall_s, 2),
        }


@contextmanager
def measured(label: str):
    """用法:
        with measured("minilm") as m:
            with m.loading():  model = load()
            for x in items:
                with m.timed():  predict(x)
    """
    m = Meter(label)
    m.baseline_rss_mb = _PROC.memory_info().rss / 2**20
    sampler = _RSSSampler()
    sampler.start()

    def loading():
        @contextmanager
        def _cm():
            r0, t0 = _PROC.memory_info().rss, time.perf_counter()
            yield
            m.load_wall_s = time.perf_counter() - t0
            m.load_rss_delta_mb = (_PROC.memory_info().rss - r0) / 2**20
        return _cm()

    def timed():
        @contextmanager
        def _cm():
            t0 = time.perf_counter()
            yield
            m.latencies_ms.append((time.perf_counter() - t0) * 1000)
        return _cm()

    def run_window():
        @contextmanager
        def _cm():
            c0, w0 = _cpu_seconds(), time.perf_counter()
            yield
            m.run_cpu_s = _cpu_seconds() - c0
            m.run_wall_s = time.perf_counter() - w0
        return _cm()

    m.loading, m.timed, m.run_window = loading, timed, run_window
    try:
        yield m
    finally:
        m.peak_rss_mb = sampler.stop() / 2**20
