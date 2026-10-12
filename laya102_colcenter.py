#!/usr/bin/env python3
"""Does our column-centering (the Fly head's cos + centered-cos score) help on #950's 500 rows?

The same score `laya_shortlist.py` uses (w[t0] = w[t0c] = 1): cos + (cos - mu), where mu is each label's
mean cosine over the unlabeled development-split texts (inductive; labels are never read), so a single
request needs one subtraction. Plain-cosine results come from the laya102_bench.py runs on machine A, MPS;
this script checks it re-derives their top-20 exactly, then asks Laya on the centered top-20.

    python laya102_colcenter.py      # writes out/laya102/colcenter.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import laya
import laya102_bench as b

RUNS = Path(__file__).parent / "out/laya102"
SPECS = {"small": ("bge-small/instruct/bare", "a_mps"), "nomic": ("nomic/sym/bare", "a_mps"),
         "minilm": ("minilm/none/bare", "a_mps_extra")}
K = 20


def pair(a, c):
    gained, lost = int((a & ~c).sum()), int((c & ~a).sum())
    return {"delta": round((gained - lost) / len(a), 4), "gained": gained, "lost": lost,
            "p": b.mcnemar_exact(gained, lost)}


def main():
    agent = laya.load("english", device="mps")
    result = {}
    for ds in ("banking77", "clinc150", "massive"):
        labels, rows, instructions, _total = b.load_rows(ds, "test", b.N_TEST)
        _l, dev_rows, _i, _t = b.load_rows(ds, b.DATASETS[ds][2], b.N_DEV)
        gold = np.array([labels.index(g) for _t2, g in rows])
        question = {"type": "choice", "instructions": instructions, "criteria": {lab: None for lab in labels}}
        for role, (spec, run) in SPECS.items():
            stored = [json.loads(line) for line in (RUNS / f"{ds}_{run}.rows.jsonl").open()]
            key, variant, query_text = spec.split("/")
            emb = b.Embedder(key, variant, query_text, "mps")
            emb.set_labels(labels)
            q = emb.encode([t for t, _g in rows], emb.qp, 64)
            pool = emb.encode([t for t, _g in dev_rows], emb.qp, 64)
            s = q @ emb.label_vecs.T
            mu = (pool @ emb.label_vecs.T).mean(axis=0)
            plain = np.argsort(-s, axis=1, kind="mergesort")
            cent = np.argsort(-(s + (s - mu)), axis=1, kind="mergesort")
            same_top = np.mean([plain[i, :K].tolist() == stored[i]["arms"][role + ":shortlist"]["top"]
                                for i in range(len(rows))])
            ok_plain = np.array([r["arms"][role + ":shortlist"]["ok"] for r in stored])
            ok_cent = []
            for i, (text, g) in enumerate(rows):
                qd = dict(question, criteria={labels[j]: None for j in cent[i, :K]})
                ok_cent.append(agent.predict({"message": text}, {"q": qd})["answers"]["q"]["choice"] == g)
            ok_cent = np.array(ok_cent)
            top1_plain, top1_cent = plain[:, 0] == gold, cent[:, 0] == gold
            res = {"spec": spec, "plain_top20_matches_stored": round(float(same_top), 4),
                   "top1": [round(float(top1_plain.mean()), 4), round(float(top1_cent.mean()), 4)],
                   "recall@20": [round(float((plain[:, :K] == gold[:, None]).any(1).mean()), 4),
                                 round(float((cent[:, :K] == gold[:, None]).any(1).mean()), 4)],
                   "shortlist": [round(float(ok_plain.mean()), 4), round(float(ok_cent.mean()), 4)],
                   "top1_paired": pair(top1_cent, top1_plain), "shortlist_paired": pair(ok_cent, ok_plain)}
            result.setdefault(ds, {})[role] = res
            print(ds, role, json.dumps(res), flush=True)
    out = {"score": "cos + (cos - mu), mu fitted on the unlabeled development-split texts",
           "device": "mps", "machine": "A", "k": K, "dev_rows": b.N_DEV, "datasets": result}
    (RUNS / "colcenter.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
