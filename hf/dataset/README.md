---
license: mit
task_categories:
  - text-classification
  - zero-shot-classification
language:
  - en
tags:
  - benchmark
  - model-comparison
  - jev
  - typesafe
  - cross-encoder
  - per-example-predictions
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files: predictions.jsonl
---

# open-system-one-bench — per-item predictions, 10,000 decisions, 5 stacks

Per-example predictions for 10,000 classification/routing decisions, from six
different stacks **including `typesafe/jev` and `convaiinnovations/laya`**.
As far as we know this is the only public per-item output from Jev on a standard
benchmark, and the only place the two are measured on **identical items**.

With this file you can, **without spending a cent on API calls**:

- run significance tests (McNemar) — is 78.7 % vs 79.3 % actually a difference?
- find the specific examples where Jev wins and the open stacks lose
- verify every aggregate number in the [benchmark repo](https://github.com/zhlei07/open-system-one)

## What's here

`predictions.jsonl` — one line per evaluation item:

```json
{"i": 0, "dataset": "banking77", "n_options": 77, "gold": 11,
 "jev_bare": 11, "jev_bare_probs": [0.0, 0.87, ...],
 "jev_enriched": 11, "jev_enriched_probs": [...],
 "cross_bare": 11, "cross_enriched": 34, "bi_fewshot": 11,
 "laya": 11, "laya_probs": [...]}
```

| column | stack | coverage | accuracy |
|---|---|---|---|
| `jev_bare` | `typesafe/jev`, bare label names as criteria | 9,976 | 78.7 % |
| `jev_enriched` | `typesafe/jev`, one-sentence descriptions | 9,976 | 79.5 % |
| `cross_bare` | ModernBERT-base-zeroshot cross-encoder, bare labels | 10,000 | 78.7 % |
| `cross_enriched` | same, enriched descriptions | 10,000 | 74.4 % |
| `bi_fewshot` | ModernBERT-embed + 8-float head, 2,000 labels/task | 10,000 | 82.6 % |
| `laya` | `convaiinnovations/laya` (421M, English root, `head_max_len=512`) | 10,000 | 73.2 % |

`options.json` — the option list per dataset, in label-index order.
`split_indices.json` — dataset + gold label for every row of the fit / dev / eval splits.

## Rebuilding the text

**No dataset text is redistributed** — the four source datasets carry their own
licences. `data.py` in the [repo](https://github.com/zhlei07/open-system-one) rebuilds
the exact same 10,000 items from a fixed seed (`SEED = 20260921`), so row `i` here
corresponds to row `i` there, deterministically.

Sources: `stanfordnlp/sst2`, `fancyzhx/ag_news`, `dair-ai/emotion` (note: its card
carries a non-commercial clause), `mteb/banking77` (CC-BY-4.0).

## Two things to know before you quote a number

1. **24 Jev requests timed out** and are `null` in both Jev columns. The accuracies
   above are computed over successful calls only. The repo's original run counted
   failures as wrong, which gives 78.8 % / 79.3 % instead of 78.7 % / 79.5 %.
   **Both conventions are defensible — just don't mix them.** Decide which you want
   and apply it to every column.
2. **`bi_fewshot` used 2,000 labelled examples per task; Jev used none.** They are not
   solving the same problem and the comparison is not a win. It is included because
   "what does this cost you once you *do* have labels" is the question most production
   teams actually face.

## Jev vs Laya, on identical items

Both are "System One" decision models; only Laya is open. Same 10,000 items, same
bare label names, same harness:

| | sst2 | ag_news | emotion | banking77 | macro |
|---|---|---|---|---|---|
| `typesafe/jev` (bare) | 89.8 % | 88.3 % | 58.6 % | **78.4 %** | **78.8 %** |
| `laya` (bare) | 86.5 % | **93.9 %** | 58.2 % | 54.3 % | 73.2 % |

Laya wins AG News by 5.6 pp and loses BANKING77 by 24.1 pp. The BANKING77 gap is a
token-budget artefact its own docs explain: 77 options share one `head_max_len`, so
each label gets 3–4 tokens. Adding a zero-shot bi-encoder shortlist to top-20 lifts it
to 60.8 % while halving latency — see the
[repo](https://github.com/zhlei07/open-system-one).

Latency is not comparable in kind: Jev's 381 ms is mostly network from our location,
Laya's is local compute.

**Correction.** We first described Laya's 19–86 ms as CPU. It was not — `laya.load()`
auto-selects `cuda > mps > cpu` and chose **Apple MPS**, while every other stack here is
pure CPU, batch = 1. Re-measured like-for-like on CPU (250 items/task): 59.5 ms (SST-2),
78.0 ms (AG News), 61.9 ms (Emotion), **309.0 ms** (BANKING77) — about 3× higher. On
BANKING77 that leaves Laya at 309 ms against Jev's 381 ms while scoring 54.3 % against
77.8 %. The shortlist gains value on CPU rather than losing it: 309.0 ms → 169.9 ms.

## Provenance

Built in [`zhlei07/open-system-one`](https://github.com/zhlei07/open-system-one).
The repository owner does not program; the direction and the judgement calls were his,
every experiment was designed and run by Claude Code (Opus 5) in one session. Raw logs
and all intermediate outputs are committed so these numbers can be checked rather than
trusted. If we have misconfigured anything, please open an issue.
