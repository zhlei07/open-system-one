# open-system-one

**An independent, reproducible benchmark of TypeSafe's [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) against open, CPU-only alternatives.**

This is *not* an open-source implementation of Jev, and it does not claim to beat it.
It is a measurement: 10,000 classification/routing decisions, one machine, one set of
option descriptions, run through six different stacks — and an honest account of
which ideas worked, which didn't, and what each one costs in latency, RAM and CPU.

**Artifacts** — [model (8 floats)](https://huggingface.co/dylantom2012/fly-head-potion-8m) ·
[interactive results](https://huggingface.co/spaces/dylantom2012/open-system-one) ·
[per-item predictions](https://huggingface.co/datasets/dylantom2012/open-system-one-bench)

中文版：[README.zh.md](README.zh.md) · 详细报告：[REPORT.md](out/REPORT.md)

---

## What Jev is, and the hypothesis we tested

Jev is a "System One Model": it doesn't generate text. You send it a `state` plus a set
of typed questions whose options (`criteria`) are supplied **at request time**, and it
returns a choice with calibrated probabilities. TypeSafe reports 40–200× faster and
40–400× cheaper than frontier LLMs on classification.

The community's first guess — and ours — was:

> Jev ≈ a fast text encoder + a very small decision head.

The API shape supports it: because `criteria` arrive per request, Jev cannot be a
fixed-output classifier head. It has to score *(state, option)* pairs and rank them.

So we built exactly that, from open parts, and measured the gap.

## The benchmark

10,000 tasks, 2,500 from each of four public datasets, chosen to span a difficulty range
rather than to flatter any single method:

| dataset | options | what it tests |
|---|---|---|
| SST-2 | 2 | sentiment — lexical cues are strong |
| AG News | 4 | topic — abstract, terse category names |
| Emotion | 6 | fine-grained affect — label names carry little meaning |
| BANKING77 | 77 | intent routing — many mutually confusable options |

Splits are rebuilt deterministically from a fixed seed (`data.py`), so every row is
reproducible. No dataset text is redistributed — see [DATA_LICENSES.md](DATA_LICENSES.md).

All local numbers: **Apple M5 Pro, CPU only, batch = 1**, each model measured in its own
process. Jev is called through Cloudflare Workers AI (`typesafe/jev`).

## Results

Every stack was run twice: once with **bare label names** as the options
(`card_arrival`, `Sci/Tech`) and once with **enriched one-sentence descriptions**
([`descriptions.py`](descriptions.py)) — the *same* descriptions for every stack.
Both conditions are reported, because neither is neutral (see below).

| stack | sst2 (2) | ag_news (4) | emotion (6) | banking77 (77) | macro |
|---|---|---|---|---|---|
| typesafe/jev — bare label names | 89.8 % | 88.3 % | 58.6 % | 78.4 % | **78.8 %** |
| typesafe/jev — enriched descriptions | 91.6 % | 88.6 % | 59.0 % | 77.8 % | **79.3 %** |
| Cross-encoder base — bare label names | 85.5 % | 84.4 % | 76.2 % | 68.8 % | **78.7 %** |
| Cross-encoder base — enriched descriptions | 84.5 % | 84.6 % | 71.7 % | 56.9 % | **74.4 %** |
| Bi-encoder + head — bare label names | 81.1 % | 66.3 % | 43.6 % | 63.2 % | **63.6 %** |
| Bi-encoder + head — enriched descriptions | 82.8 % | 80.0 % | 50.8 % | 65.8 % | **69.9 %** |

**Best result per stack, zero-shot, no labels on either side:**

| | best macro | best condition | p50 latency | RAM |
|---|---|---|---|---|
| `typesafe/jev` (cloud) | **79.3 %** | enriched | 381 ms † | cloud |
| Cross-encoder, ModernBERT-base-zeroshot (149M) | **78.7 %** | bare | 50–616 ms * | — |
| Bi-encoder + 8-float head | 69.9 % | enriched | 15.7 ms | 490 MB |
| Qwen2.5-0.5B, generative | 49.6 % | bare | 112 ms | 2382 MB |

**A 149M open model on a CPU lands 0.6 pp behind Jev.**

\* scales linearly with option count: 50 ms at 2 options, 616 ms at 77 (shortlisted to 20).
† dominated by network round-trip; see Limitations.

### There is no neutral set of option descriptions

The single most useful thing we learned. The same edit — replacing terse labels with
one-sentence descriptions — moves the three architectures in **different directions**:

| reaction to enriched descriptions | sst2 | ag_news | emotion | banking77 | mean |
|---|---|---|---|---|---|
| `typesafe/jev` | +1.8 | +0.4 | +0.4 | −0.5 | **+0.5** |
| Bi-encoder (cosine) | +1.7 | **+13.7** | +7.2 | +2.5 | **+6.3** |
| Cross-encoder (NLI) | −1.0 | +0.2 | −4.5 | **−11.9** | **−4.3** |

- **Cosine similarity cannot expand a label's meaning.** `World / Sports / Business /
  Sci-Tech` carries almost no signal for a dot product, so the bi-encoder gains +13.7 pp
  on AG News once you spell the categories out.
- **The cross-encoder and Jev already do that expansion internally**, so they gain
  nothing — and the cross-encoder is actively hurt by long hypotheses.
- **The BANKING77 collapse (−11.9 pp) is our fault, and instructive**: 70+ of our 77
  descriptions begin with *"The customer …"*. That shared prefix makes the options
  near-identical as NLI hypotheses. With 2 options a shared prefix is harmless; with 77
  it is fatal.

So "give every system the same text" is **not** the same as "be fair". Verbaliser design
is a per-architecture tuning knob, and we publish `descriptions.py` precisely so someone
can write better ones and re-run.

### With 2,000 labels per task — **not comparable to Jev**

Jev receives no labels, so this is a different problem, not a win.

| stack | macro acc | p50 latency | RAM |
|---|---|---|---|
| ModernBERT-embed + linear head | **82.6 %** | 15.8 ms | 490 MB |
| MiniLM-L6 + linear head | 79.5 % | 2.8 ms | 452 MB |
| **potion-base-8M (static embedding) + linear head** | **78.2 %** | **0.1 ms** | **82 MB** |

Per task, ModernBERT-embed + head vs Jev: **BANKING77 88.8 % vs 77.8 % (+11.0)**,
AG News +0.9, Emotion +7.2, SST-2 −5.8.

The last row is the result we find most interesting: an 8M-parameter static embedding
(pure lookup — no attention, no matrix multiply in the forward pass) plus a decision head
whose entire parameter count is **eight floats**, reaching the accuracy of a cloud
decision model at 0.1 ms and 82 MB. The head trains in 0.1 s.

## What did not work

Eight ideas we tested and would otherwise have guessed were wins. These cost us a day;
they are the part of this repo most likely to save you one.

| idea | result |
|---|---|
| Template ensembling (3 NLI hypotheses), base backbone | **0.0 pp** — the three templates' errors are highly correlated |
| Evolutionary search over decision-head weights, honest LODO transfer | **−1.5 pp** — negative transfer; only helps when the underlying signal is weak |
| Scaling the cross-encoder 149M → 395M | **+2.8 / −3.2 / +0.2 pp** across three tasks — no net gain for 3× the cost |
| Listwise re-ranking with Qwen2.5-1.5B over the top-5 | **−1.0 pp** vs the cross-encoder alone |
| Column-centering the NLI scores | **0.0 pp** macro — helps BANKING77 +2.6, hurts Emotion −3.1 |
| Asymmetric `search_query:`/`search_document:` prefixes | **−1.8 pp** — classification-by-similarity is not retrieval |
| Distilling the cross-encoder into the bi-encoder head | **−5.4 pp** vs the teacher; collapses on Emotion (−18.8 pp) |
| Raw `ModernBERT-base` (MLM) as a sentence encoder | **35.0 %**, worse than an 8M static embedding — training objective dominates scale |

The one thing that *did* work was architectural: scoring *(text, option)* **jointly**
(cross-encoder) instead of embedding them separately and comparing cosines. That single
change is worth **+12 to +24 pp** depending on the task, and it is the reason a 149M
open model lands within 0.1 pp of Jev.

## How this was built

**The repository owner does not program.** The contribution was a hunch and a set of
judgement calls:

> *"The hyped Jev model is probably just a fast text encoder in front of a small
> decision head. Can we reproduce it with a static embedding, MiniLM, and ModernBERT?"*

Every experiment, measurement and analysis here was designed and run by **Claude Code
(Opus 5)** in a single working session — including the eight ideas that failed and two
measurement bugs it introduced and then caught (a latency run contaminated by concurrent
HTTP traffic, and a results file it overwrote).

That is not the same as "the AI did it all". Three interventions from the owner changed
the outcome, and none of them required writing a line of code:

1. **"Drop Qwen."** The agent had started ensembling a generative model into a stack
   whose whole premise was cheap local parts, for a gain that turned out to be negative.
2. **"Test under genuinely fair conditions. We need the real numbers before we decide."**
   This forced the enriched-description re-run — which produced the most interesting
   finding in this repo.
3. **"Are you sure the task hasn't drifted?"** It had. That question pulled a day's work
   back onto its original question.

All raw outputs are committed, so the numbers can be **checked rather than trusted**.
We think that is how a benchmark should be published: not "believe our table", but
"here is every file we produced, find our mistakes".

## Where this goes next

The gap in the zero-shot setting is small and the latency advantage is real. Neither is
the end of the line. Things we have **not** tried, roughly in order of expected value —
PRs welcome:

- **Quantisation / ONNX export** of the cross-encoder — expected 3–5× lower latency at
  the same accuracy. This is the single cheapest win available.
- **Stronger cross-encoder backbones**: `deberta-v3-large-zeroshot`, `bge-reranker-v2`.
  Our 149M → 395M scaling test was a wash, but those are different model families.
- **Hybrid shortlisting** (BM25 + dense) for many-option tasks. On BANKING77 our dense
  shortlist already reaches 98.1 % recall@20; the bottleneck is the re-ranker, not recall.
- **Listwise re-ranking with a stronger model** than the 1.5B we tried, which failed.
- **Fine-tuning a cross-encoder on diverse classification data.** This is the honest path
  to actually beating Jev zero-shot, and it is a training project, not a prompt tweak.
- **Batched / GPU serving**, which changes the cost picture entirely for offline work.
- **More datasets and more languages.** Four English datasets is a narrow slice.

## Cost

Jev, measured: 6,764,108 input tokens for 10,000 decisions = **$0.284**
($0.042 per million input tokens, output free), 0 failures, 0 invalid outputs.

## Reproduce

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version:

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python \
  torch transformers sentence-transformers model2vec datasets scikit-learn psutil
.venv/bin/python data.py
./encode_isolated.sh
.venv/bin/python run.py heads
.venv/bin/python cross_v2.py --model base --ds sst2,emotion,ag_news --n 2500 --templates 1
.venv/bin/python cross_v2.py --model base --ds banking77 --n 2500 --templates 1 --topk 20
.venv/bin/python frontier.py
```

Jev columns need Cloudflare credentials — see [.env.example](.env.example).

## Limitations

1. **Jev's latency is almost entirely network.** 2 options and 77 options both take
   ~381 ms from our location (input tokens differ 5×, latency differs 5%). Jev's own
   compute time is too small for us to measure. "Local is 7× faster" means we skipped a
   round trip, not that we out-computed it.
2. **Single seed, single machine, one run per configuration.** No confidence intervals.
   Sub-sample checks moved results by up to ±2 pp.
3. **Calibration (ECE) is not compared fairly.** Our temperatures are fitted on 1,000
   labeled dev examples; Jev gets none.
4. **We are not TypeSafe.** Jev is a black box to us; we only observe its API.
   If we have mis-configured it, please open an issue — we will re-run and correct.

## License

Code: MIT. Datasets and models retain their own licenses — see
[DATA_LICENSES.md](DATA_LICENSES.md).
