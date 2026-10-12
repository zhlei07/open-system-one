#!/usr/bin/env python3
"""The same-row comparison proposed on NandhaKishorM/laya#102 (xinpingzh, 2026-10-12), run on our machines.

Protocol, as he froze it:
- 500 rows of each official test split, the same rows for every arm. We use `benchmark.py`'s seed-0
  sample from Laya's research/benchmarks/tournament, so the rows are those of #950's 500-row table.
- #950's question: state {"message": text}, labels with underscores turned to spaces, no description.
- top-20 retrieval, tournament groups of at most 16 (`predict_tournament`'s default cut).
- latency per request = query encoding + ranking + every decision round; models and label vectors
  are warmed beforehand, and each arm is warmed with its own call shape.
- BGE-small vs MiniLM is chosen on development rows only (`select`); Nomic is a separate reference.

Arms are interleaved row by row in rotating order, so heat and background load fall on all of them.

    python laya102_bench.py select --device mps --out out/laya102/selection.json
    python laya102_bench.py run --dataset banking77 --device mps \
        --selection out/laya102/selection.json --out out/laya102/banking77_a_mps
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

K = 20
GROUP = 16
N_TEST = 500
N_DEV = 600

# Laya research/benchmarks/tournament/benchmark.py pins these.
REVISIONS = {
    "mteb/banking77": "18072d2685ea682290f7b8924d94c62acc19c0b2",
    "clinc/clinc_oos": "155b9c710419136e17307b80d0a13e68cd46b4ec",
    "SetFit/amazon_massive_intent_en-US": "f7672a018e8ceb37fc0184dcfbb7e665155ffea6",
}
SOURCES = {"banking77": "mteb/banking77", "clinc150": "clinc/clinc_oos", "massive": "SetFit/amazon_massive_intent_en-US"}

# key -> (repo, {variant: (query prefix, label prefix)})
EMBEDDERS = {
    "bge-small": ("BAAI/bge-small-en-v1.5", {
        "none": ("", ""),
        "instruct": ("Represent this sentence for searching relevant passages: ", ""),
    }),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", {"none": ("", "")}),
    "nomic": ("nomic-ai/modernbert-embed-base", {
        "sym": ("search_query: ", "search_query: "),      # what open-system-one published
        "asym": ("search_query: ", "search_document: "),  # the model card's retrieval convention
    }),
}
SMALL = ("bge-small", "minilm")
# "bare": the message alone. "library": what laya.shortlist embeds, instructions + "\n" + JSON state.
QUERY_TEXTS = ("bare", "library")


# ---------------------------------------------------------------- data (as in benchmark.py)
def fetch(repo, filename):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=REVISIONS[repo]))


def banking77(split):
    import pyarrow.parquet as pq

    def read(name):
        return pq.read_table(fetch("mteb/banking77", "data/%s-00000-of-00001.parquet" % name)).to_pylist()

    names = dict((r["label"], r["label_text"]) for r in read("test"))
    return [names[i] for i in sorted(names)], [(r["text"], r["label_text"]) for r in read(split)]


def clinc150(split):
    import pyarrow.parquet as pq

    path = fetch("clinc/clinc_oos", "plus/%s-00000-of-00001.parquet" % split)
    names = json.loads(pq.read_schema(path).metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
    rows = [(r["text"], names[r["intent"]]) for r in pq.read_table(path).to_pylist()]
    return [n for n in names if n != "oos"], [(text, gold) for text, gold in rows if gold != "oos"]


def massive(split):
    def read(name):
        path = fetch("SetFit/amazon_massive_intent_en-US", "%s.jsonl" % name)
        return [json.loads(line) for line in path.read_text().splitlines()]

    names = dict((r["label"], r["label_text"]) for r in read("train"))
    return [names[i] for i in sorted(names)], [(r["text"], r["label_text"]) for r in read(split)]


DATASETS = {
    "banking77": (banking77, "Which banking intent does `message` express?", "train"),
    "clinc150": (clinc150, "Which intent does `message` express?", "validation"),
    "massive": (massive, "Which intent does `message` express?", "validation"),
}


def load_rows(dataset, split, limit, seed=0):
    """benchmark.py's `--limit` sample: random.Random(seed).sample(rows, limit)."""
    load, instructions, _dev = DATASETS[dataset]
    names, rows = load(split)
    total = len(rows)
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    labels = [n.replace("_", " ") for n in names]
    return labels, [(text, gold.replace("_", " ")) for text, gold in rows], instructions, total


# ---------------------------------------------------------------- embedders
def snapshot(repo, repo_type="model"):
    from huggingface_hub import snapshot_download
    try:
        path = snapshot_download(repo, repo_type=repo_type, local_files_only=True, allow_patterns=["*.json"])
        return Path(path).name
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        return "unknown (%s)" % type(exc).__name__


class Embedder:
    def __init__(self, key, variant, query_text, device):
        from sentence_transformers import SentenceTransformer

        self.key, self.variant, self.query_text = key, variant, query_text
        self.repo = EMBEDDERS[key][0]
        self.qp, self.lp = EMBEDDERS[key][1][variant]
        self.model = SentenceTransformer(self.repo, device=device)
        self.model.eval()
        self.device = str(self.model.device)
        self.revision = snapshot(self.repo)

    @property
    def name(self):
        return "%s/%s/%s" % (self.key, self.variant, self.query_text)

    def encode(self, texts, prefix, batch_size):
        return self.model.encode([prefix + t for t in texts], batch_size=batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    def query_string(self, text, instructions):
        if self.query_text == "bare":
            return text
        return "%s\n%s" % (instructions, json.dumps({"message": text}, ensure_ascii=False))

    def set_labels(self, labels):
        self.label_vecs = self.encode(labels, self.lp, 64)

    def rank(self, text, instructions):
        """Cosine against the warmed label vectors, ties to the earlier label (laya.shortlist._rank)."""
        q = self.encode([self.query_string(text, instructions)], self.qp, 1)[0]
        return np.argsort(-(self.label_vecs @ q), kind="mergesort")

    def rank_batch(self, texts, instructions):
        q = self.encode([self.query_string(t, instructions) for t in texts], self.qp, 64)
        return np.argsort(-(q @ self.label_vecs.T), axis=1, kind="mergesort")


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def mcnemar_exact(gained, lost):
    n = gained + lost
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(gained, lost) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(len(s) * q))], 1)


def thermal_level():
    try:
        out = subprocess.run(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"],
                             capture_output=True, text=True, timeout=5).stdout.split()
        return int(out[-1])
    except Exception:  # noqa: BLE001 - not on macOS / not available
        return None


def machine_info():
    """OS and architecture only: result files get published, the machine's model does not."""
    return {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}


# ---------------------------------------------------------------- select (development rows only)
def cmd_select(args):
    results, per_variant = {}, {}
    for dataset in DATASETS:
        dev_split = DATASETS[dataset][2]
        labels, rows, instructions, total = load_rows(dataset, dev_split, N_DEV)
        gold = np.array([labels.index(g) for _t, g in rows])
        results[dataset] = {"dev_split": dev_split, "dev_rows": len(rows), "pool": total}
        for key, (_repo, variants) in EMBEDDERS.items():
            for variant in variants:
                emb = None
                for query_text in QUERY_TEXTS:
                    if emb is None:
                        emb = Embedder(key, variant, query_text, args.device)
                        emb.set_labels(labels)
                    emb.query_text = query_text
                    order = emb.rank_batch([t for t, _g in rows], instructions)
                    pos = (order == gold[:, None]).argmax(1)
                    rec = {"recall@%d" % k: round(float(np.mean(pos < k)), 4) for k in (1, 5, 10, 16, 20, 32, 48)}
                    results[dataset][emb.name] = rec
                    per_variant.setdefault(emb.name, []).append(rec)
                    print(dataset, emb.name, json.dumps(rec), flush=True)
    mean = {name: {m: round(float(np.mean([r[m] for r in recs])), 4) for m in recs[0]}
            for name, recs in per_variant.items()}

    def best(keys):
        cands = [n for n in mean if n.split("/")[0] in keys]
        return max(cands, key=lambda n: (mean[n]["recall@20"], mean[n]["recall@1"]))

    chosen = {"small": best(SMALL), "nomic": best(("nomic",))}
    out = {"rule": "highest mean dev recall@20 over the three datasets, ties by recall@1",
           "chosen": chosen, "mean": mean, "per_dataset": results, "device": args.device,
           "revisions": {EMBEDDERS[k][0]: snapshot(EMBEDDERS[k][0]) for k in EMBEDDERS}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print("chosen", chosen, flush=True)


# ---------------------------------------------------------------- manifest (the rows, without their text)
def cmd_manifest(args):
    """Each sampled row's position in the loaded test rows, sha256 of its text, and gold label.

    random.sample picks positions from the RNG alone, so sampling range(n) gives the rows' positions.
    For CLINC150 the positions count in-scope rows only, in file order.
    """
    out = {"sample": "random.Random(%d).sample(rows, %d) over the loaded test rows" % (args.seed, N_TEST),
           "datasets": {}}
    for dataset in DATASETS:
        labels, rows, _instructions, total = load_rows(dataset, "test", N_TEST, args.seed)
        pos = random.Random(args.seed).sample(range(total), N_TEST)
        _names, all_rows = DATASETS[dataset][0]("test")
        assert [(all_rows[j][0], all_rows[j][1].replace("_", " ")) for j in pos] == rows, dataset
        repo = SOURCES[dataset]
        out["datasets"][dataset] = {"source": repo, "revision": REVISIONS[repo], "pool_rows": total, "rows": [{"i": i, "pos": j, "sha256": hashlib.sha256(text.encode()).hexdigest(), "gold": gold}
                     for i, (j, (text, gold)) in enumerate(zip(pos, rows))]}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print({d: len(v["rows"]) for d, v in out["datasets"].items()})


# ---------------------------------------------------------------- run (test rows)
def tournament_groups(labels, size):
    """predict_tournament's cut: criteria order, near-equal groups of at most `size`."""
    parts = -(-len(labels) // size)
    return [labels[i * len(labels) // parts:(i + 1) * len(labels) // parts] for i in range(parts)]


def cmd_run(args):
    import torch
    import laya
    from laya.common import build_head
    from laya.evals import ece

    dataset = args.dataset
    labels, rows, instructions, total = load_rows(dataset, "test", N_TEST, args.seed)
    if args.limit:
        rows = rows[:args.limit]
    _l, dev_rows, _i, _t = load_rows(dataset, DATASETS[dataset][2], N_DEV)
    warm_rows = [t for t, _g in dev_rows[:args.warm]]
    index = {label: i for i, label in enumerate(labels)}
    question = {"type": "choice", "instructions": instructions, "criteria": {label: None for label in labels}}

    agent = laya.load("english", device=args.device)
    wide = {"head_max_len": 512, "max_len": 1024}
    selection = json.loads(Path(args.selection).read_text())
    roles = dict(selection["chosen"])  # small, nomic
    if args.roles:  # e.g. minilm=minilm/none/bare,minilm-lib=minilm/none/library
        roles = dict(item.split("=", 1) for item in args.roles.split(","))
    embedders = {}
    for role, spec in roles.items():
        key, variant, query_text = spec.split("/")
        emb = Embedder(key, variant, query_text, args.device)
        emb.set_labels(labels)
        embedders[role] = emb

    def state(text):
        return {"message": text}

    def single(budget):
        def run(text):
            a = agent.predict(state(text), {"q": question}, **budget)["answers"]["q"]
            return a["choice"], a.get("answer_confidence"), 1, {}
        return run

    def tournament(text):
        out = laya.predict_tournament(agent, state(text), {"q": question}, group_size=GROUP)
        t, a = out["tournament"]["q"], out["answers"]["q"]
        return a["choice"], a.get("answer_confidence"), t["rounds"] + 1, {"finalists": t["labels"]}

    def top1(emb):
        def run(text):
            order = emb.rank(text, instructions)
            return labels[order[0]], None, 0, {"gold_rank_src": order[:K].tolist()}
        return run

    def shortlist(emb, then_tournament):
        def run(text):
            order = emb.rank(text, instructions)
            top = [labels[i] for i in order[:K]]  # rank order, as predict_shortlist passes them
            q = dict(question, criteria={label: None for label in top})
            if then_tournament:
                out = laya.predict_tournament(agent, state(text), {"q": q}, group_size=GROUP)
                t, a = out["tournament"]["q"], out["answers"]["q"]
                extra = {"top": order[:K].tolist(), "finalists": t["labels"]}
                return a["choice"], a.get("answer_confidence"), t["rounds"] + 1, extra
            a = agent.predict(state(text), {"q": q})["answers"]["q"]
            return a["choice"], a.get("answer_confidence"), 1, {"top": order[:K].tolist()}
        return run

    arms = {} if args.embedder_arms_only else {"single": single({}), "wide": single(wide), "tournament": tournament}
    for role, emb in embedders.items():
        arms[role + ":top1"] = top1(emb)
        arms[role + ":shortlist"] = shortlist(emb, False)
        arms[role + ":shortlist+tournament"] = shortlist(emb, True)
    names = list(arms)

    # The library path and ours must pick the same 20 (checked on warm-up rows, never timed).
    for emb in embedders.values():
        if emb.query_text != "library":
            continue

        def embed_fn(texts, emb=emb):
            return np.concatenate([emb.encode(texts[:1], emb.qp, 1), emb.encode(list(texts[1:]), emb.lp, 64)])
        for text in warm_rows:
            lib = laya.shortlist_choice(state(text), question["criteria"], embed_fn, k=K, instructions=instructions)
            ours = [labels[i] for i in emb.rank(text, instructions)[:K]]
            assert lib == ours, (emb.name, lib, ours)

    for name in names:  # each arm warmed with its own call shape
        for text in warm_rows:
            arms[name](text)

    log = Path(args.out + ".rows.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    records, thermal = [], []
    started = time.time()
    with log.open("w") as fh:
        for i, (text, gold) in enumerate(rows):
            rot = i % len(names)
            rec = {"i": i, "gold": gold, "arms": {}}
            for name in names[rot:] + names[:rot]:
                t0 = time.perf_counter()
                choice, conf, calls, extra = arms[name](text)
                ms = (time.perf_counter() - t0) * 1000
                rec["arms"][name] = {"choice": choice, "ok": choice == gold, "ms": round(ms, 2),
                                     "conf": conf, "calls": calls, **extra}
            records.append(rec)
            fh.write(json.dumps(rec) + "\n")
            if i % 50 == 0 or i == len(rows) - 1:
                level = thermal_level()
                thermal.append([i, level])
                done = i + 1
                eta = (time.time() - started) / done * (len(rows) - done)
                accs = " ".join("%s=%.3f" % (n.replace("tournament", "tour"),
                                               np.mean([r["arms"][n]["ok"] for r in records])) for n in names)
                print("%s %s %d/%d thermal=%s eta=%.0fs %s" % (dataset, args.device, done, len(rows), level,
                                                              eta, accs), flush=True)

    # ---- summary
    gi = [index[r["gold"]] for r in records]
    summary = {}
    for name in names:
        res = [r["arms"][name] for r in records]
        ok = [x["ok"] for x in res]
        ms = [x["ms"] for x in res]
        entry = {"accuracy": round(float(np.mean(ok)), 4), "correct": int(sum(ok)), "ci95": wilson(sum(ok), len(ok)),
                 "p50_ms": pct(ms, 0.5), "p95_ms": pct(ms, 0.95), "mean_ms": round(float(np.mean(ms)), 1),
                 "calls_per_row": round(float(np.mean([x["calls"] for x in res])), 3)}
        confs = [(x["conf"], x["ok"]) for x in res if x["conf"] is not None]
        if confs:
            entry["ece"] = round(ece([c for c, _ in confs], [o for _, o in confs]), 4)
        if name.endswith("top1"):
            ranks = [x["gold_rank_src"].index(g) if g in x["gold_rank_src"] else K
                     for x, g in zip(res, gi)]
            entry["recall@%d" % K] = round(float(np.mean([p < K for p in ranks])), 4)
        if "top" in res[0]:
            miss = [g not in x["top"] for x, g in zip(res, gi)]
            entry["errors"] = {"retrieval_miss": int(sum(miss))}
            if "finalists" in res[0]:
                elim = [(not m) and r["gold"] not in x["finalists"] for x, m, r in zip(res, miss, records)]
                entry["errors"]["elimination_miss"] = int(sum(elim))
                entry["errors"]["final_error"] = int(sum((not x["ok"]) and not m and not e
                                                         for x, m, e in zip(res, miss, elim)))
            else:
                entry["errors"]["decision_error"] = int(sum((not x["ok"]) and not m for x, m in zip(res, miss)))
        elif "finalists" in res[0]:
            elim = [r["gold"] not in x["finalists"] for x, r in zip(res, records)]
            entry["errors"] = {"elimination_miss": int(sum(elim)),
                               "final_error": int(sum((not x["ok"]) and not e for x, e in zip(res, elim)))}
        summary[name] = entry

    pairs = [("tournament", "single"), ("wide", "single"), ("tournament", "wide")]
    for role in roles:
        pairs += [(role + ":shortlist", role + ":top1"), (role + ":shortlist+tournament", role + ":top1"),
                  (role + ":shortlist", "tournament"), (role + ":shortlist+tournament", "tournament"),
                  (role + ":shortlist+tournament", role + ":shortlist"), ("tournament", role + ":top1")]
    pairs += [("nomic:shortlist", "small:shortlist"), ("nomic:shortlist+tournament", "small:shortlist+tournament")]
    pairs = [(a, b) for a, b in pairs if a in arms and b in arms]
    paired = {}
    for a, b in pairs:
        gained = sum(r["arms"][a]["ok"] and not r["arms"][b]["ok"] for r in records)
        lost = sum(r["arms"][b]["ok"] and not r["arms"][a]["ok"] for r in records)
        paired["%s vs %s" % (a, b)] = {"delta": round((gained - lost) / len(records), 4), "gained": gained,
                                       "lost": lost, "mcnemar_p": mcnemar_exact(gained, lost)}

    # ---- what the option budget kept (laya.common.build_head, the head the model reads)
    def cut_stats(label_set, head):
        q = {"t": "choice", "ins": instructions, "crit": {label: None for label in label_set}}
        _ids, _markers, stats = build_head(agent.tok, q, head)
        return stats

    head = int(agent.cfg.get("head_max_len", 192))
    budget = {"single": cut_stats(labels, head), "wide": cut_stats(labels, wide["head_max_len"]),
              "tournament_groups": [cut_stats(g, head) for g in tournament_groups(labels, GROUP)]}
    for role in roles:
        stats = [cut_stats([labels[i] for i in r["arms"][role + ":shortlist"]["top"]], head) for r in records]
        budget[role + ":shortlist"] = {"rows_cut": sum(s.get("tokens_per_option") is not None for s in stats),
                                       "min_distinct": min(s.get("options_distinct", K) for s in stats)}

    report = {
        "dataset": dataset, "split": "test", "pool_rows": total, "rows": len(records), "seed": args.seed,
        "labels": len(labels), "k": K, "group_size": GROUP, "device": str(agent.device),
        "embedder_devices": {r: e.device for r, e in embedders.items()},
        "torch_threads": torch.get_num_threads(), "laya": laya.__version__, "torch": torch.__version__,
        "laya_checkpoint_snapshot": snapshot("convaiinnovations/laya"),
        "agent_cfg": {k: agent.cfg.get(k) for k in ("max_len", "head_max_len")},
        "embedders": {r: {"name": e.name, "repo": e.repo, "revision": e.revision, "query_prefix": e.qp,
                          "label_prefix": e.lp} for r, e in embedders.items()},
        "selection_rule": selection["rule"], "roles": roles, "roles_from": "--roles" if args.roles else "selection",
        "dataset_revisions": REVISIONS,
        "machine": machine_info(), "thermal_pressure": thermal, "warm_rows_per_arm": len(warm_rows),
        "arm_order": "rotating per row", "wall_s": round(time.time() - started, 1),
        "arms": summary, "paired": paired, "option_budget": budget,
    }
    Path(args.out + ".json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({n: (s["accuracy"], s["p50_ms"], s["p95_ms"]) for n, s in summary.items()}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--device", default="cpu")
    s.add_argument("--out", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    r.add_argument("--device", required=True)
    r.add_argument("--selection", required=True)
    r.add_argument("--out", required=True, help="path prefix: writes <out>.json and <out>.rows.jsonl")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--warm", type=int, default=3)
    r.add_argument("--limit", type=int, help="smoke test: only the first N of the 500 rows")
    r.add_argument("--roles", help="replace the selection: name=embedder/variant/query_text,...")
    r.add_argument("--embedder-arms-only", action="store_true",
                   help="skip single / wide / tournament (they do not depend on the embedder)")
    args = ap.parse_args()
    {"select": cmd_select, "manifest": cmd_manifest, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
