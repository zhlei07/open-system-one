#!/usr/bin/env python3
"""Turn the laya102_bench.py result files into one Markdown report.

    python laya102_report.py out/laya102 <laya repo>/research/benchmarks/tournament/results.json

Result files are named <dataset>_<machine>_<device>; the machines are called A and B.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from laya102_bench import mcnemar_exact

DATASETS = ("banking77", "clinc150", "massive")
ARMS = ("single", "wide", "tournament", "small:top1", "small:shortlist", "small:shortlist+tournament",
        "nomic:top1", "nomic:shortlist", "nomic:shortlist+tournament")
RUNS = (("a", "mps"), ("b", "cpu"), ("a", "cpu"), ("b", "mps"))
# notifyutil com.apple.system.thermalpressurelevel
LEVELS = {0: "nominal", 1: "moderate", 2: "heavy", 3: "trapping", 4: "sleeping"}
NAMES = {"a": "A", "b": "B"}
MIN_SPLIT_ROWS = 20  # a thermal level gets its own latency column only with at least this many rows


def row_levels(samples, n):
    """Thermal level for each row, from the samples taken after rows 0, 50, 100, ... .

    A row gets a level only when the samples on both sides of it agree; rows in a
    50-row stretch where the level changed get None and are left out of the split.
    """
    levels = []
    for j in range(n):
        after = next((lvl for i, lvl in samples if i >= j), None)
        before = next((lvl for i, lvl in reversed(samples) if i < j), after)
        levels.append(after if before == after else None)
    return levels


def ms_pct(values, q):
    """Same percentile rule as laya102_bench.pct."""
    s = sorted(values)
    return round(s[min(len(s) - 1, int(len(s) * q))], 1) if s else None


def load_run(folder: Path, stem):
    path = folder / f"{stem}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()), [json.loads(line) for line in (folder / f"{stem}.rows.jsonl").open()]


def paired(a, b):
    """a, b = (rows, arm) on the same rows; '+delta (gained/lost, exact McNemar p)'."""
    (rows_a, arm_a), (rows_b, arm_b) = a, b
    assert len(rows_a) == len(rows_b) and all(x["i"] == y["i"] and x["gold"] == y["gold"]
                                              for x, y in zip(rows_a, rows_b)), (arm_a, arm_b)
    gained = sum(x["arms"][arm_a]["ok"] and not y["arms"][arm_b]["ok"] for x, y in zip(rows_a, rows_b))
    lost = sum(y["arms"][arm_b]["ok"] and not x["arms"][arm_a]["ok"] for x, y in zip(rows_a, rows_b))
    p = mcnemar_exact(gained, lost)
    return "%+.1f (%d/%d, p%s)" % (100 * (gained - lost) / len(rows_a), gained, lost,
                                   "<0.001" if p < 0.001 else "=%.3f" % p)


def load(folder: Path):
    found = {}
    for ds in DATASETS:
        for machine, device in RUNS:
            path = folder / f"{ds}_{machine}_{device}.json"
            if path.exists():
                rep = json.loads(path.read_text())
                rows = [json.loads(line) for line in (folder / f"{ds}_{machine}_{device}.rows.jsonl").open()]
                found[(ds, machine, device)] = (rep, rows)
    return found


def run_name(machine, device):
    return "%s %s" % (NAMES[machine], device.upper())


def pct(x):
    return "—" if x is None else "%.1f" % (100 * x)


def main():
    folder, published = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
    runs = load(folder)
    primary = {ds: runs.get((ds, "a", "mps")) for ds in DATASETS}
    any_rep = next(r for r, _ in runs.values())
    sel = json.loads((folder / "selection.json").read_text())
    out = []
    w = out.append

    w("# Laya #102 protocol, run on our machines\n")
    w("Rows: `benchmark.py`'s seed-0 sample, 500 per official test split (the rows of #950's 500-row table). "
      "Question as in #950: state `{\"message\": text}`, labels with spaces, no descriptions. "
      "top-%d retrieval, tournament groups of at most %d. Latency per request = query encoding + ranking + "
      "every decision round; models and label vectors warmed; each arm warmed on its own call shape on dev rows; "
      "arms interleaved per row in rotating order.\n" % (any_rep["k"], any_rep["group_size"]))
    w("Laya %s (checkpoint snapshot `%s`), torch %s, defaults max_len=%s head_max_len=%s; `wide` = 512/1024.\n" % (
        any_rep["laya"], any_rep["laya_checkpoint_snapshot"][:10], any_rep["torch"],
        any_rep["agent_cfg"]["max_len"], any_rep["agent_cfg"]["head_max_len"]))
    w("Embedders, chosen on dev rows only (%s): small = `%s`, Nomic reference = `%s`.\n" % (
        sel["rule"], sel["chosen"]["small"], sel["chosen"]["nomic"]))
    w("Machines: A and B, two Apple-silicon laptops. MPS = torch on the Apple GPU, CPU = torch on the CPU.\n")

    w("\n## Development-row selection (mean over the three datasets)\n")
    w("| embedder / prefix / query text | recall@1 | recall@20 |\n|---|---:|---:|")
    for name, m in sorted(sel["mean"].items(), key=lambda kv: -kv[1]["recall@20"]):
        w("| `%s` | %s | %s |" % (name, pct(m["recall@1"]), pct(m["recall@20"])))
    w("\n`library` = the query text `laya.shortlist` embeds (instructions + newline + JSON state); "
      "`bare` = the message alone.\n")

    w("\n## Accuracy, %s (95 %% Wilson interval)\n" % run_name("a", "mps"))
    w("| arm | " + " | ".join(DATASETS) + " |\n|---|" + "---:|" * len(DATASETS))
    for arm in ARMS:
        cells = []
        for ds in DATASETS:
            if primary[ds] is None:
                cells.append("—")
                continue
            a = primary[ds][0]["arms"][arm]
            cells.append("%s [%s–%s]" % (pct(a["accuracy"]), pct(a["ci95"][0]), pct(a["ci95"][1])))
        w("| `%s` | %s |" % (arm, " | ".join(cells)))

    w("\n## Same rows, #950's published 500-row table (Laya 0.3.27, MPS) vs ours\n")
    w("| dataset | arm | #950 | ours (%s) |\n|---|---|---:|---:|" % run_name("a", "mps"))
    for ds in DATASETS:
        pub = published.get("sample", {}).get(ds, {}).get("strategies", {})
        for arm in ("single", "wide", "tournament"):
            if arm in pub and primary[ds] is not None:
                w("| %s | `%s` | %s | %s |" % (ds, arm, pct(pub[arm].get("accuracy")),
                                               pct(primary[ds][0]["arms"][arm]["accuracy"])))
        if "shortlist" in pub:
            w("| %s | `shortlist` (embed_fn_from_agent) | %s | — |" % (ds, pct(pub["shortlist"].get("accuracy"))))

    w("\n## Errors, %s (rows out of 500)\n" % run_name("a", "mps"))
    w("| dataset | arm | retrieval miss | elimination miss | final / decision error |\n|---|---|---:|---:|---:|")
    for ds in DATASETS:
        if primary[ds] is None:
            continue
        for arm in ARMS:
            e = primary[ds][0]["arms"][arm].get("errors")
            if e:
                w("| %s | `%s` | %s | %s | %s |" % (ds, arm, e.get("retrieval_miss", "—"),
                                                  e.get("elimination_miss", "—"),
                                                  e.get("final_error", e.get("decision_error", "—"))))

    w("\n## Paired comparisons, %s (delta in points; gained / lost rows; exact McNemar p)\n"
      % run_name("a", "mps"))
    keys = ["tournament vs single", "wide vs single", "small:shortlist vs small:top1",
            "small:shortlist+tournament vs small:top1", "small:shortlist vs tournament",
            "small:shortlist+tournament vs tournament", "small:shortlist+tournament vs small:shortlist",
            "tournament vs small:top1", "nomic:shortlist vs small:shortlist",
            "nomic:shortlist+tournament vs small:shortlist+tournament"]
    w("| comparison | " + " | ".join(DATASETS) + " |\n|---|" + "---|" * len(DATASETS))
    for key in keys:
        cells = []
        for ds in DATASETS:
            p = primary[ds][0]["paired"].get(key) if primary[ds] else None
            cells.append("—" if p is None else "%+.1f (%d/%d, p%s)" % (
                100 * p["delta"], p["gained"], p["lost"],
                "<0.001" if p["mcnemar_p"] < 0.001 else "=%.3f" % p["mcnemar_p"]))
        w("| `%s` | %s |" % (key, " | ".join(cells)))

    w("\n## Latency, ms (p50 / p95)\n")
    cols = [(ds, m, d) for ds in DATASETS for m, d in RUNS if (ds, m, d) in runs]
    w("| arm | " + " | ".join("%s %s" % (ds, run_name(m, d)) for ds, m, d in cols) + " |")
    w("|---|" + "---:|" * len(cols))
    for arm in ARMS:
        cells = []
        for c in cols:
            a = runs[c][0]["arms"][arm]
            cells.append("%s / %s" % (a["p50_ms"], a["p95_ms"]))
        w("| `%s` | %s |" % (arm, " | ".join(cells)))
    w("")
    split = {}
    for c in cols:
        rep, rows = runs[c]
        levels = row_levels(rep["thermal_pressure"], len(rows))
        counts = {lvl: levels.count(lvl) for lvl in sorted(set(levels), key=lambda x: (x is None, x))}
        w("- %s %s: torch threads %s, wall %.0f s, thermal pressure by row: %s" % (
            c[0], run_name(c[1], c[2]), rep["torch_threads"],
            rep["wall_s"], ", ".join("%s %d" % ("between samples" if lvl is None else LEVELS.get(lvl, lvl), n)
                                for lvl, n in counts.items())))
        known = [lvl for lvl, n in counts.items() if lvl is not None and n >= MIN_SPLIT_ROWS]
        if len(known) > 1:
            split[c] = (rows, levels, known)

    if split:
        w("\n### Latency by thermal pressure, ms (p50 / p95)\n")
        w("macOS thermal pressure is sampled after every 50th row; a row is counted at a level only when the "
          "samples on both sides of it agree; levels with fewer than %d rows are not shown. Arms run "
          "interleaved, so every arm sees the same heat; accuracy does not depend on it.\n" % MIN_SPLIT_ROWS)
        heads = [(c, lvl) for c, (_rows, _levels, known) in split.items() for lvl in known]
        w("| arm | " + " | ".join("%s %s, %s (%d rows)" % (c[0], run_name(c[1], c[2]), LEVELS.get(lvl, lvl),
                                                          split[c][1].count(lvl)) for c, lvl in heads) + " |")
        w("|---|" + "---:|" * len(heads))
        for arm in ARMS:
            cells = []
            for c, lvl in heads:
                rows, levels, _known = split[c]
                ms = [r["arms"][arm]["ms"] for r, x in zip(rows, levels) if x == lvl]
                cells.append("%s / %s" % (ms_pct(ms, 0.5), ms_pct(ms, 0.95)))
            w("| `%s` | %s |" % (arm, " | ".join(cells)))

    w("\n## Device agreement (same rows, same arm: rows whose answer differs from %s)\n"
      % run_name("a", "mps"))
    w("| run | " + " | ".join("`%s`" % a for a in ARMS) + " |\n|---|" + "---:|" * len(ARMS))
    for (ds, m, d), (rep, rows) in sorted(runs.items()):
        if (m, d) == ("a", "mps") or primary[ds] is None:
            continue
        base = primary[ds][1]
        cells = []
        for arm in ARMS:
            diff = sum(x["arms"][arm]["choice"] != y["arms"][arm]["choice"] for x, y in zip(rows, base))
            cells.append("%d (acc %s)" % (diff, pct(rep["arms"][arm]["accuracy"])))
        w("| %s %s | %s |" % (ds, run_name(m, d), " | ".join(cells)))

    extra = {ds: load_run(folder, f"{ds}_a_mps_extra") for ds in DATASETS}
    if any(extra.values()):
        roles = next(x for x in extra.values() if x)[0]["roles"]
        w("\n## Add-on: other embedder configurations, %s (k=20, same rows)\n" % run_name("a", "mps"))
        w("Embedder arms only; `single` / `wide` / `tournament` do not depend on the embedder and are the "
          "rows above. Roles: %s. `-lib` = the query text `laya.shortlist` embeds (instructions + newline + "
          "JSON state); Nomic Q/D = `search_query: ` on the message, `search_document: ` on the labels.\n"
          % ", ".join("`%s` = `%s`" % kv for kv in roles.items()))
        w("| arm | " + " | ".join(DATASETS) + " |\n|---|" + "---:|" * len(DATASETS))
        for role in roles:
            for arm in (role + ":top1", role + ":shortlist", role + ":shortlist+tournament"):
                cells = []
                for ds in DATASETS:
                    a = extra[ds][0]["arms"][arm] if extra[ds] else None
                    cells.append("—" if a is None else "%s [%s–%s]" % (
                        pct(a["accuracy"]), pct(a["ci95"][0]), pct(a["ci95"][1])))
                w("| `%s` | %s |" % (arm, " | ".join(cells)))
            w("| %s recall@20 | %s |" % (role, " | ".join(
                pct(extra[ds][0]["arms"][role + ":top1"]["recall@20"]) if extra[ds] else "—" for ds in DATASETS)))

        w("\nPaired on the same rows (delta in points; gained / lost rows; exact McNemar p). `tournament` and "
          "`small:*` come from the main %s run.\n" % run_name("a", "mps"))
        comps = []
        for role in roles:
            comps += [(role + ":shortlist", "x", role + ":top1", "x"),
                      (role + ":shortlist+tournament", "x", role + ":shortlist", "x"),
                      (role + ":shortlist", "x", "tournament", "p"),
                      (role + ":shortlist+tournament", "x", "tournament", "p"),
                      (role + ":shortlist", "x", "small:shortlist", "p")]
            if role.endswith("-lib") and role[:-4] in roles:
                comps.append((role + ":shortlist", "x", role[:-4] + ":shortlist", "x"))
        w("| comparison | " + " | ".join(DATASETS) + " |\n|---|" + "---|" * len(DATASETS))
        for arm_a, src_a, arm_b, src_b in comps:
            cells = []
            for ds in DATASETS:
                if extra[ds] is None or primary[ds] is None:
                    cells.append("—")
                    continue
                pick = {"x": extra[ds][1], "p": primary[ds][1]}
                cells.append(paired((pick[src_a], arm_a), (pick[src_b], arm_b)))
            w("| `%s vs %s` | %s |" % (arm_a, arm_b, " | ".join(cells)))

    paths = {ds: load_run(folder, f"paths_{ds}_a_mps") for ds in DATASETS}
    if any(paths.values()):
        # laya102_paths.py writes each row's arms at the top level; give them the main runs' shape
        paths = {ds: x and (x[0], [{"i": r["i"], "gold": r["gold"],
                                    "arms": {k: v for k, v in r.items() if k not in ("i", "gold")}} for r in x[1]])
                 for ds, x in paths.items()}
        rep = next(x for x in paths.values() if x)[0]
        arms = list(rep["arms"])
        w("\n## Add-on: the shortlist paths Laya documents, Laya %s, %s (k=%d, same rows)\n" % (
            rep["laya"], run_name("a", rep["device"]), rep["k"]))
        w("`laya102_paths.py`, Laya %s (commit `%s`, checkpoint `%s`), every call through Laya's own functions, "
          "every embedder wrapped in `laya.cached_embed_fn`. Embedders: `agent` = `laya.embed_fn_from_agent(agent)` "
          "(the README example; the MCP `laya_shortlist` tool makes the same call); `bge` = `%s`, no prefix; "
          "`minilm` = `%s`. Query text: `default` = `laya.predict_shortlist` as documented (instructions + "
          "newline + JSON state); `state` = `laya.shortlist_choice` on the JSON state; `message` = "
          "`laya.shortlist_choice` on the message string. Every shortlist then gets one `agent.predict` with "
          "#950's full question; `single` = no shortlist. Cell = accuracy [95 %% Wilson] (recall@20).\n" % (
              rep["laya"], rep["laya_commit"][:7], rep["laya_checkpoint"][:7],
              rep["models"]["bge"], rep["models"]["minilm"]))
        w("| arm | " + " | ".join(DATASETS) + " |\n|---|" + "---:|" * len(DATASETS))
        for arm in arms:
            cells = []
            for ds in DATASETS:
                a = paths[ds][0]["arms"][arm] if paths[ds] else None
                cells.append("—" if a is None else "%s [%s–%s]%s" % (
                    pct(a["accuracy"]), pct(a["ci95"][0]), pct(a["ci95"][1]),
                    " (%s)" % pct(a["recall@%d" % rep["k"]]) if "recall@%d" % rep["k"] in a else ""))
            w("| `%s` | %s |" % (arm, " | ".join(cells)))
        w("")
        for ds in DATASETS:
            if paths[ds] is None or primary[ds] is None:
                continue
            same = sum(x["arms"]["single"]["ok"] == y["arms"]["single"]["ok"]
                       for x, y in zip(paths[ds][1], primary[ds][1]))
            pub = published.get("sample", {}).get(ds, {}).get("strategies", {}).get("shortlist", {})
            w("- %s: `single` scores the same as the main Laya %s run on %d of %d rows; `agent:default` %s, "
              "#950's published `shortlist` %s" % (ds, primary[ds][0]["laya"], same, len(paths[ds][1]),
                                                    pct(paths[ds][0]["arms"]["agent:default"]["accuracy"]),
                                                    pct(pub.get("accuracy"))))

        w("\nPaired on the same rows (delta in points; gained / lost rows; exact McNemar p). `tournament` and "
          "`small:shortlist` (BGE-small with its retrieval instruction on the bare message) come from the main "
          "%s run, Laya %s.\n" % (run_name("a", "mps"), any_rep["laya"]))
        comps = [("bge:default", "x", "agent:default", "x"), ("agent:default", "x", "single", "x"),
                 ("agent:message", "x", "single", "x"), ("agent:message", "x", "agent:default", "x"),
                 ("bge:message", "x", "bge:default", "x"), ("bge:state", "x", "bge:default", "x"),
                 ("bge:message", "x", "bge:state", "x"), ("minilm:message", "x", "minilm:default", "x"),
                 ("agent:default", "x", "tournament", "p"), ("bge:message", "x", "tournament", "p"),
                 ("bge:message", "x", "small:shortlist", "p")]
        w("| comparison | " + " | ".join(DATASETS) + " |\n|---|" + "---|" * len(DATASETS))
        for arm_a, src_a, arm_b, src_b in comps:
            cells = []
            for ds in DATASETS:
                if paths[ds] is None or primary[ds] is None:
                    cells.append("—")
                    continue
                pick = {"x": paths[ds][1], "p": primary[ds][1]}
                cells.append(paired((pick[src_a], arm_a), (pick[src_b], arm_b)))
            w("| `%s vs %s` | %s |" % (arm_a, arm_b, " | ".join(cells)))

    cc_path = folder / "colcenter.json"
    if cc_path.exists():
        cc = json.loads(cc_path.read_text())
        w("\n## Column-centering (open-system-one's shortlist score), %s\n" % run_name(cc["machine"].lower(), cc["device"]))
        w("Score = %s (`laya_shortlist.py`: cosine + column-centered cosine), %d development rows per dataset, "
          "labels not read. Plain → centered; paired delta in points (gained / lost rows, exact McNemar p). "
          "The plain top-%d lists match the main runs on every row.\n" % (cc["score"], cc["dev_rows"], cc["k"]))
        w("| dataset | embedder | recall@20 | top-1 | shortlist → Laya | top-1 Δ | shortlist Δ |\n"
          "|---|---|---:|---:|---:|---|---|")
        for ds, roles in cc["datasets"].items():
            for role, r in roles.items():
                assert r["plain_top20_matches_stored"] == 1.0, (ds, role)
                cells = ["%s → %s" % (pct(r[m][0]), pct(r[m][1])) for m in ("recall@20", "top1", "shortlist")]
                deltas = ["%+.1f (%d/%d, p=%.2f)" % (100 * d["delta"], d["gained"], d["lost"], d["p"])
                          for d in (r["top1_paired"], r["shortlist_paired"])]
                w("| %s | `%s` | %s |" % (ds, r["spec"], " | ".join(cells + deltas)))

    w("\n## What the option budget kept (laya.common.build_head)\n")
    for ds in DATASETS:
        if primary[ds] is None:
            continue
        b = primary[ds][0]["option_budget"]
        groups = b["tournament_groups"]
        w("- %s: one question at the default head %s; `wide` %s; tournament groups %s; shortlists cut on "
          "%d (small) / %d (nomic) of 500 rows" % (
              ds, b["single"], b["wide"], [g["options"] for g in groups] + (
                  ["cut"] if any(g["tokens_per_option"] for g in groups) else ["none cut"]),
              b["small:shortlist"]["rows_cut"], b["nomic:shortlist"]["rows_cut"]))
    print("\n".join(out))


if __name__ == "__main__":
    main()
