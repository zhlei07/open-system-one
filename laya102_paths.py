#!/usr/bin/env python3
"""Which shortlist do Laya users get? #950's 500 rows per dataset, through Laya's own functions.

Embedders (each wrapped in laya.cached_embed_fn, as the README suggests):
  agent    laya.embed_fn_from_agent(agent)   -- the README's example
  bge      BAAI/bge-small-en-v1.5, no prefix  -- sentence-transformers defaults
  minilm   sentence-transformers/all-MiniLM-L6-v2
Query text:
  default  laya.predict_shortlist as documented: instructions + "\\n" + JSON state
  state    laya.shortlist_choice(state, criteria, embed_fn, k)       -- JSON state, no instructions
  message  laya.shortlist_choice(message, criteria, embed_fn, k)     -- the message string alone
Every shortlist then gets one agent.predict with #950's full question. `single` = no shortlist.

    PYTHONPATH=<laya checkout> python laya102_paths.py <dataset> <out prefix> [rows, for a smoke test]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import laya
import laya102_bench as b

K = 20
DATASET, OUT = sys.argv[1], sys.argv[2]
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else None
MODELS = {"bge": "BAAI/bge-small-en-v1.5", "minilm": "sentence-transformers/all-MiniLM-L6-v2"}
QUERIES = ("default", "state", "message")


def st_embed_fn(repo):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(repo, device="mps")
    model.eval()
    return lambda texts: model.encode(list(texts), batch_size=64, convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False)


def main():
    labels, rows, instructions, _total = b.load_rows(DATASET, "test", b.N_TEST)
    rows = rows[:LIMIT]
    question = {"type": "choice", "instructions": instructions, "criteria": {lab: None for lab in labels}}
    agent = laya.load("english", device="mps")
    embedders = {"agent": laya.cached_embed_fn(laya.embed_fn_from_agent(agent))}
    embedders.update({k: laya.cached_embed_fn(st_embed_fn(repo)) for k, repo in MODELS.items()})

    def decide(state, kept):
        q = dict(question, criteria={lab: None for lab in kept})
        return agent.predict(state, {"q": q})["answers"]["q"]["choice"]

    records, started = [], time.time()
    log = Path(OUT + ".rows.jsonl").open("w")
    for i, (text, gold) in enumerate(rows):
        state = {"message": text}
        rec = {"i": i, "gold": gold,
               "single": {"ok": agent.predict(state, {"q": question})["answers"]["q"]["choice"] == gold}}
        for name, ef in embedders.items():
            out = laya.predict_shortlist(agent, state, {"q": question}, embed_fn=ef, k=K)
            kept = out["shortlist"]["q"]["labels"]
            if i < 3:  # the documented call ranks exactly like shortlist_choice with the instructions
                assert kept == laya.shortlist_choice(state, question["criteria"], ef, k=K,
                                                     instructions=instructions), (name, i)
            rec[name + ":default"] = {"hit": gold in kept, "ok": out["answers"]["q"]["choice"] == gold}
            for query, source in (("state", state), ("message", text)):
                kept = laya.shortlist_choice(source, question["criteria"], ef, k=K)
                rec[name + ":" + query] = {"hit": gold in kept, "ok": decide(state, kept) == gold}
        records.append(rec)
        log.write(json.dumps(rec) + "\n")
        if i % 100 == 0 or i == len(rows) - 1:
            print(DATASET, i + 1, "%.0fs" % (time.time() - started), flush=True)
    log.close()

    arms = [k for k in records[0] if k not in ("i", "gold")]
    summary = {}
    for arm in arms:
        ok = [r[arm]["ok"] for r in records]
        entry = {"accuracy": round(float(np.mean(ok)), 4), "ci95": b.wilson(sum(ok), len(ok))}
        if "hit" in records[0][arm]:
            entry["recall@%d" % K] = round(float(np.mean([r[arm]["hit"] for r in records])), 4)
        summary[arm] = entry
    sha = subprocess.run(["git", "-C", str(Path(laya.__file__).parent.parent), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    report = {"dataset": DATASET, "rows": len(records), "k": K, "laya": laya.__version__, "laya_commit": sha,
              "laya_checkpoint": getattr(agent, "revision", None), "device": str(agent.device),
              "models": MODELS, "wall_s": round(time.time() - started, 1), "arms": summary}
    Path(OUT + ".json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v["accuracy"] for k, v in summary.items()}), flush=True)


if __name__ == "__main__":
    main()
