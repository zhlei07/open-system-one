# open-system-one

**An independent, reproducible benchmark of TypeSafe's [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) against open, CPU-only alternatives.**

This is *not* an open-source implementation of Jev, and it does not claim to beat it.
It is a measurement: 10,000 classification/routing decisions, one machine, one set of
option descriptions, run through six different stacks — and an honest account of
which ideas worked, which didn't, and what each one costs in latency, RAM and CPU.

**Write-up** — [the full story, with the eight approaches that failed](https://huggingface.co/blog/dylantom2012/i-benchmarked-jev-against-open-cpu-only-stacks-it)

**Artifacts** — [model (8 floats)](https://huggingface.co/dylantom2012/fly-head-potion-8m) ·
[per-item predictions](https://huggingface.co/datasets/dylantom2012/open-system-one-bench) ·
[results explorer](https://huggingface.co/spaces/dylantom2012/open-system-one) ·
[live demo](https://huggingface.co/spaces/dylantom2012/open-system-one-demo)

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

## Update — Laya, and a fourth architecture

A week after Jev, [Laya](https://huggingface.co/convaiinnovations/laya) shipped: an
Apache-2.0 System One model, ModernBERT-large + a decision head, 421M params. It answers
every option in **one forward pass** by scoring each option at its own `[MASK]` token —
the thing our cross-encoder cannot do, and the reason our latency scales with option count.

We ran it through the same 10,000 items, same bare label names, same measurement discipline.
**As far as we know this is the first time Laya and Jev have been measured on identical items.**
(Laya's own README compares against third-party published Jev numbers on different samples
and 72 vs 77 labels; our Jev column is directly measured.)

| stack | sst2 | ag_news | emotion | banking77 | macro | p50 latency |
|---|---|---|---|---|---|---|
| `typesafe/jev` (enriched) | 91.6 % | 88.6 % | 59.0 % | 77.8 % | **79.3 %** | 381 ms |
| Cross-encoder 149M (bare) | 85.5 % | 84.4 % | **76.2 %** | 68.8 % | **78.7 %** | 50–616 ms |
| **Laya 421M (bare)** | 86.5 % | **93.9 %** | 58.2 % | 54.3 % | **73.2 %** | 19–86 ms *(MPS)* / **60–309 ms** *(CPU)* |
| bi-encoder + head, 2k labels | 85.8 % | 89.5 % | 66.2 % | **88.8 %** | **82.6 %** | 15.8 ms |

Three things fall out of this:

**Laya wins AG News outright** — 93.9 %, 5.3 pp above Jev. Its own README claims 0.950 there;
we measure 0.939. Its published Emotion figure (0.595) matches our 0.582. **Its self-reported
numbers hold up**, which is worth saying plainly.

**Its high-cardinality weakness is real, and we partially fixed it.** Laya's docs explain the
failure honestly: 77 options share a fixed `head_max_len`, so each label gets 3–4 tokens and
the texts stop being distinguishable. They recommend *"split large option sets into a two-step
coarse-to-fine hierarchical choice"* — but do not implement it. We had independently built
exactly that for our own cross-encoder, so we bolted it on:

| BANKING77, 77 options | accuracy | p50 (MPS) | p50 (CPU) |
|---|---|---|---|
| Laya, all 77 options | 54.3 % | 86.3 ms | 309.0 ms |
| **Laya + our zero-shot bi-encoder shortlist to top-20** | **60.8 %** | **39.8 ms** | **169.9 ms** |
| *(shortlist ceiling, recall@20)* | *97.8 %* | — | — |

*(CPU column measured on a 250-item sample, `laya_cpu.py` / `LAYA_DEVICE=cpu`; it scores
53.6 % and 62.0 % on that sample, consistent with the full-set 54.3 % / 60.8 %.)*

**+6.5 pp and latency cut in half.** Still 17 pp behind Jev on this task, so the token-budget
problem is mitigated rather than solved — but the fix costs one cheap bi-encoder pass and is
the only thing here that made both accuracy and latency better at once.

**Correction — Laya's original latencies were Apple GPU, not CPU.** We first published these
as CPU numbers. They were not. `laya.load()` auto-selects `cuda > mps > cpu` and silently
chose **mps** on this Mac, while every other stack in this repo is pure CPU, batch=1. Re-measured
like-for-like, Laya costs 59.5 ms (SST-2) / 78.0 ms (AG News) / 61.9 ms (Emotion) / **309.0 ms**
(BANKING77) on CPU — roughly 3× its MPS figures. That mostly removes Laya's apparent latency win
over Jev on the hardest task (309 ms vs 381 ms) while it still scores 54.3 % against Jev's 77.8 %.
It also makes the shortlist worth more, not less: on CPU it cuts 309 ms to 170 ms.

The lesson generalises past this one number — we had strict discipline about RAM (one model per
process) but none about **device**. Anything auto-selected (device, dtype, thread count, batch)
has to be written into the result file, not remembered. `laya_bench.py` and `laya_shortlist.py`
now record `device` and accept `LAYA_DEVICE`.

**The three architectures fail in three different ways.** Bi-encoder: no interaction between
text and option, so abstract labels kill it. Cross-encoder: one forward pass per option, so
latency scales with cardinality. Single-pass option markers: all options share one token
budget, so cardinality destroys accuracy instead of speed. There is no free lunch; pick the
failure mode you can live with.

## Update — the decision head's objective was wrong

Laya is trained with RLCD against strictly proper scoring rules. That prompted us to look at
our own objective, which was `0.7 × mean accuracy + 0.3 × worst-task accuracy` — a rule that
gives **zero credit for reporting honest probabilities**. We swapped it for the log score,
with the temperature optimised inside each evaluation.

| ES objective | macro accuracy | mean ECE | mean NLL |
|---|---|---|---|
| accuracy (before) | 82.6 % | 0.018 | 0.490 |
| **strictly proper (log score)** | **83.6 %** | 0.018 | **0.462** |

The result is not what we expected. **Calibration did not move** (ECE 0.018 either way) —
temperature scaling on a held-out dev split was already doing that job, so the missing
calibration term had not actually been costing us anything. What improved was **accuracy**,
by 1.0 pp, biggest on SST-2 (+1.8) and Emotion (+2.0).

The explanation is about optimisation, not calibration: accuracy is a step function, so the
(1+λ)-ES searches a landscape of plateaus. The log score is smooth and rewards margin, which
gives an evolution strategy something like a gradient to follow. **We went looking for better
probabilities and found a better search signal instead.**

## Update — Chinese, and a confidence trap

Laya ships a multilingual checkpoint (mmBERT-base, 322M, 100+ languages). Its own benchmarks
cover MASSIVE and XNLI; we tested something it does not: **CLUE/TNEWS**, Chinese news-headline
classification, 15 classes, 600 items.

| checkpoint | accuracy | mean confidence | overconfidence gap | p50 |
|---|---|---|---|---|
| `laya-multilingual` | **39.2 %** | 0.756 | +0.365 | 14.3 ms |
| `laya` (English root) | 22.5 % | **0.968** | **+0.743** | 42.5 ms |
| *random* | *6.7 %* | — | — | — |

The English checkpoint on Chinese is the dangerous case: **22.5 % right while 96.8 % confident.**
This reproduces, in Chinese, the failure Laya documents for Khmer — *"the model stays confident
while being wrong, so confidence gating cannot save you."* If you serve mixed-language traffic,
route on script before the forward pass; do not rely on confidence thresholds to catch it.

The multilingual checkpoint is usable — 6× random at 14 ms — but it is **also overconfident by
0.365**. Fit your own temperature on your own data before trusting any probability it gives you.

## Update — column-centering, gated by option count

We published "column-centering the NLI scores: **0.0 pp** macro" as a dead end. That verdict
was correct *on the macro average* and wrong as a conclusion. The per-task numbers were
+2.6 on BANKING77 and −3.1 on Emotion; averaging them to zero threw away the signal.

Centering fixes **hubness** — some label words are similar to everything (`card payment`
in BANKING77, `change speed` in CLINC) and absorb queries indiscriminately. Hubness grows
with the number of options. So the gate looks at **option count only**, never the dataset
name:

| options | task | Δ from centering |
|---|---|---|
| 2 | SST-2 | −0.18 pp |
| 4 | AG News | +0.04 pp |
| 6 | Emotion | −3.34 pp |
| **77** | **BANKING77** | **+2.30 pp** |
| **150** | **CLINC routing** | **+3.25 to +5.50 pp** |

Monotone in option count, positive on both high-cardinality tasks, ≤0 on all three
low-cardinality ones. CLINC is independent evidence: it is not one of the four benchmark
datasets and played no part in choosing the gate.

**Making it inductive.** `fly_head`'s column-centering is *transductive*: it takes the
column mean over the current batch of queries. With a single request that batch is one row,
the mean is the score itself, and the term is identically zero — no gain online. We refit
the column means **offline from an unlabeled pool** (the `fit` split's texts, labels never
read), then subtract them per request. 500 unlabeled queries match the transductive result:

| pool | transductive | inductive |
|---|---|---|
| 100 | +2.51 | +1.57 |
| 250 | +2.50 | +2.24 |
| **500** | **+2.53** | **+2.54** |

The same applies one layer up, to the bi-encoder prefilter. Its published `recall@20` of
97.8 % also relied on a batch mean; per request it degrades to the bare-cosine 96.8 %.
Refit offline it recovers fully (97.9 %).

**Held-out test.** The gate threshold and the fusion rule below were both chosen after
looking at eval results, so we re-ran on data nothing had touched: `idx[5500:]` from each
pool (`fit`/`dev`/`eval` use `idx[:5500]`), deduplicated by text against all three splits.
2,500 fresh items per dataset. The column means were **not refit** — the ones fitted on the
`fit` split were applied as-is.

| task | options | gate | no centering | gated | Δ |
|---|---|---|---|---|---|
| sst2 | 2 | off | 83.72 % | 83.72 % | +0.00 |
| ag_news | 4 | off | 84.08 % | 84.08 % | +0.00 |
| emotion | 6 | off | 77.56 % | 77.56 % | +0.00 |
| banking77 | 77 | **on** | 68.52 % | **70.76 %** | **+2.24** |
| macro | | | 78.47 % | **79.03 %** | +0.56 |

+2.24 pp on data never seen, against +2.28 pp on eval. The three exact zeros are the
control: they prove the gate is the identity on low-cardinality tasks and that nothing
else changed.

**What this is not.** The threshold (20) was picked after seeing per-task results on four
datasets, of which one crosses it. Two high-cardinality tasks support the mechanism; that
is evidence, not proof. Jev was **not** measured on the held-out set — no comparison there.

## Update — fitting the fusion weight made things worse

Picking the two-backend mixing weight by accuracy on a 100-item calibration fold is
overfitting, and the more you fit the worse it gets. We dumped both backends' probabilities
for all five use cases and compared five rules offline (`weight_probe.py`, `out/weight_probe.json`):

| rule | macro |
|---|---|
| **fixed 50/50 — no fitting** | **72.45 %** |
| shrink argmax halfway to 0.5 | 72.30 % |
| midpoint of the within-1-SE plateau | 71.95 % |
| argmax of the calibration fold | 71.85 % |
| fall back to the best single backend | 70.15 % |
| Laya alone | 71.05 % |
| best weight in hindsight (ceiling) | 73.10 % |

Monotone. The weight is redundant with temperature: each backend is already calibrated by
NLL, and temperature encodes how much to trust it. On CLINC routing, cross scores 64.5 %
and Laya 87.8 % — 23 points apart — and an equal mix still scores 87.8 %. Fitting a weight
on 100 samples adds variance and no signal.

Fusion itself is worth having: on the guardrail case the two backends score 52.5 % and
53.0 % alone and 57.0 % mixed (they are wrong together only 25.2 % of the time). The old
rule fell back to a single backend and threw those 4 points away. `Ensemble.calibrate()`
now fits temperature only; `fit_weights=True` restores the old behaviour for comparison.

## Update — enriched option descriptions, tested on the use cases

We published "+13.7 pp on AG News from writing label descriptions as sentences" and never
applied it to the five use cases, all of which still ran on bare labels. DBpedia's classes
(`Mean Of Transportation`, `Natural Place`, `Written Work`) looked like the same disease,
and that case was our largest gap to Jev (−11.0 pp). It was not the same disease.

| use case | options | cross bare → rich | Laya bare → rich | fused bare → rich |
|---|---|---|---|---|
| moderation | 2 | 81.2 → 78.5 (−2.8) | 83.8 → 83.0 (−0.8) | 84.2 → 83.0 (−1.2) |
| guardrail | 2 | 52.5 → **57.5 (+5.0)** | 53.0 → 56.0 (+3.0) | 57.0 → 57.0 (0.0) |
| scoring (SST-5) | 5 | 41.2 → 33.2 (−8.0) | 44.2 → 40.2 (−4.0) | 45.8 → **47.8 (+2.0)** |
| labeling (DBpedia) | 14 | 75.5 → **81.0 (+5.5)** | 86.5 → 81.2 (−5.2) | 87.5 → 85.2 (−2.2) |
| macro | | 62.6 → 62.6 (−0.1) | 66.9 → 65.1 (−1.8) | 68.6 → 68.2 (−0.4) |

The +13.7 pp was a **bi-encoder** result: a dot product cannot expand a label's meaning, so
writing it as a sentence adds real signal. The same table already records **−4.3 pp** for the
cross-encoder. The use cases run on a cross-encoder and Laya, so extrapolating from the
bi-encoder number was the wrong reading of our own data.

What the four cases do show cleanly is that **the gain tracks how much the bare label leaves
unsaid**: guardrail +5.0 (the bare option never said what the assistant actually does),
DBpedia on the cross-encoder +5.5 (ontology terms are abstract), moderation −2.8 (the bare
labels were already unambiguous), SST-5 −8.0 (`very negative` is already the plainest
possible phrasing; a longer sentence is pure noise).

**The one result worth keeping.** On the guardrail case all three local stacks previously sat
on the majority baseline (51.2 %) and we described the task as "both sides are about
coin-flipping". With descriptions that say what the assistant handles, the cross-encoder
reaches 57.5 %, 6.3 points above the baseline. The model was not failing to read a capability
boundary; we had never told it where the boundary was.

Descriptions live in `descriptions.py::USECASE_ENRICHED`; run with `RICH=1`. CLINC routing has
no hand-written descriptions — 150 intents is too many to write well, and templated ones are
worse than the bare labels. Jev was measured on bare labels only, so its column is not
comparable under this condition.

## Update — the shortlist on official test rows (Laya #102)

On [NandhaKishorM/laya#102](https://github.com/NandhaKishorM/laya/issues/102), xinpingzh proposed a
same-row comparison of the shortlist above against Laya's own fix for large option sets, the tournament
from #950: 500 fixed rows per official test split (BANKING77, CLINC150 in-scope, MASSIVE en-US), top-20
retrieval, tournament groups of at most 16, latency including query encoding and every decision round.
We ran that protocol ourselves on #950's 500-row sample (`benchmark.py`'s seed-0 rows), with #950's question
format, Laya 0.4.4, on two Apple-silicon laptops (MPS and CPU); the embedder was chosen on development rows only.

The rows check out first: Laya 0.4.4 reproduces all nine `single` / `wide` / `tournament` accuracies of
#950's published 500-row table, and MPS and CPU give the same answer on every row and arm but one.

| 500 official test rows, MPS | BANKING77 (77) | CLINC150 (150) | MASSIVE (60) | p50 |
|---|---|---|---|---|
| Laya, all options, default budget | 41.6 % | 60.4 % | 54.2 % | 41–89 ms |
| Laya, all options, 512/1024 budget *(our 54.3 % setting)* | 55.4 % | 68.0 % | 62.2 % | 46–92 ms |
| Laya tournament, groups ≤ 16 | 58.4 % | **89.4 %** | 59.0 % | 70–92 ms |
| BGE-small top-1, no Laya | **66.2 %** | 70.4 % | 58.4 % | 6 ms |
| **BGE-small top-20 → Laya once** | 63.0 % | 86.8 % | **64.4 %** | 29–34 ms |
| BGE-small top-20 → tournament | 62.2 % | 86.4 % | 64.0 % | 50–57 ms |

- **The shortlist holds up on official test rows.** Against all options at the 512/1024 budget — the
  comparison behind our 54.3 % → 60.8 % — it is +7.6 / +18.8 / +2.2 pp.
- **Against the tournament** it is +4.6 pp on BANKING77 (54 rows gained / 31 lost, exact McNemar p = 0.017)
  and +5.4 on MASSIVE (p = 0.011), −2.6 on CLINC150 (p = 0.09), at about half the latency: 34 vs 70 ms on
  MPS, 105 vs 304 ms on CPU (BANKING77).
- **A tournament after the shortlist adds nothing**: −0.8 / −0.4 / −0.4 pp, all p > 0.6. At k = 20 the option
  budget cuts none of the shortlists, so there is no truncation left to relieve.
- **On BANKING77 the embedder alone is as good as any Laya arm**: top-1 66.2 % against 63.0 % for the
  shortlist (p = 0.14). Where the labels are near-synonyms, Laya adds nothing after a good retriever.
- **The retriever barely matters, its query text does.** BGE-small, MiniLM and Nomic land within 1 pp of each
  other on the BANKING77 and CLINC150 shortlists. But `laya.shortlist` embeds `instructions + "\n" +
  JSON(state)`, not the message: that leaves the BANKING77 shortlist at 62.8 % (62.6 % with the bare message)
  while MiniLM's top-1 drops from 59.6 % to 48.0 %. A top-1 baseline computed that way makes Laya look
  12 points better than it is.
- **Our column-centering does not move these numbers.** Scoring the shortlist with cosine + centered cosine
  (means fitted on unlabeled development text) raises recall@20 by 0.2–2.0 pp in all nine embedder × dataset
  cells, but top-1 and shortlist accuracy change by −1.6 to +0.8 pp, none significant. On these rows the gain
  comes from the shortlist structure, not from our score.

**What the documented path gives.** Laya's README example and its MCP `laya_shortlist` tool build the shortlist
with the decision model's own encoder (`laya.embed_fn_from_agent`). On the same rows, through Laya 0.4.6's own
functions (`laya102_paths.py`, same checkpoint):

| 500 official test rows, MPS, top-20 → Laya once | BANKING77 | CLINC150 | MASSIVE |
|---|---|---|---|
| no shortlist | 41.6 % | 60.4 % | 54.2 % |
| Laya's encoder, documented query (`predict_shortlist`) | 22.4 % | 18.8 % | 18.4 % |
| Laya's encoder, message only | 30.4 % | 28.4 % | 30.2 % |
| BGE-small, documented query | 63.0 % | 76.8 % | 57.8 % |
| **BGE-small, message only** (`shortlist_choice(message, ...)`) | 63.0 % | **87.2 %** | **65.2 %** |

- **Laya's own encoder keeps the gold label among the 20 on 20–30 % of rows** (recall@20 29.6 / 19.8 / 27.8 %),
  so its shortlist answers below no shortlist at all, even with the message alone. This path is #950's
  published `shortlist` row, reproduced exactly; the `embed_fn_from_agent` docstring already says a dedicated
  bi-encoder will usually shortlist better.
- **The sentence embedder is the step that matters**: BGE-small in the same documented call is
  +40.6 / +58.0 / +39.4 pp (p < 0.001).
- **Embedding the message alone** then adds +10.4 pp on CLINC150 and +7.4 on MASSIVE (p < 0.001) and 0.0 on
  BANKING77, where the JSON state alone is 3.2 pp better (p = 0.011). It needs no API change:
  `laya.shortlist_choice` embeds a string state as it is.
- BGE-small with no prefix lands within 1 pp of our development-chosen setting (its retrieval instruction on the
  query) on all three datasets. Without an external embedder, Laya's tournament (58.4 / 89.4 / 59.0 %) is the
  better built-in route.

Code: `laya102_bench.py`, `laya102_colcenter.py`, `laya102_paths.py`, `laya102_report.py`. Results, per-row
files and the full report: [`out/laya102/`](out/laya102/) ([REPORT.md](out/laya102/REPORT.md)).

## What did not work

Eleven ideas we tested and would otherwise have guessed were wins. These cost us a day;
they are the part of this repo most likely to save you one.

| idea | result |
|---|---|
| Template ensembling (3 NLI hypotheses), base backbone | **0.0 pp** — the three templates' errors are highly correlated |
| Evolutionary search over decision-head weights, honest LODO transfer | **−1.5 pp** — negative transfer; only helps when the underlying signal is weak |
| Scaling the cross-encoder 149M → 395M | **+2.8 / −3.2 / +0.2 pp** across three tasks — no net gain for 3× the cost |
| Listwise re-ranking with Qwen2.5-1.5B over the top-5 | **−1.0 pp** vs the cross-encoder alone |
| Column-centering the NLI scores, applied to every task | **0.0 pp** macro — helps BANKING77 +2.6, hurts Emotion −3.1. **Superseded**: gated by option count it is worth +2.24 pp on held-out BANKING77 — see the update above |
| Asymmetric `search_query:`/`search_document:` prefixes | **−1.8 pp** — classification-by-similarity is not retrieval |
| Distilling the cross-encoder into the bi-encoder head | **−5.4 pp** vs the teacher; collapses on Emotion (−18.8 pp) |
| Raw `ModernBERT-base` (MLM) as a sentence encoder | **35.0 %**, worse than an 8M static embedding — training objective dominates scale |
| Pairwise contrast prompting (`A rather than B`) over the top-12 | **−3.5 pp** vs independent scoring; flipping only the top-2 is −7.5. The model is reliable on single-label entailment and out of distribution on contrast sentences |
| Fitting the two-backend fusion weight on a 100-item calibration fold | **−2.3 pp** vs fixed 50/50 — see the update above |
| Upgrading the cross-encoder's candidate prefilter from 8M static to 149M ModernBERT | **+1.5 pp** on CLINC despite recall@20 rising 92.0 → 97.5; with centering on it is **−0.8 pp**. Newly recalled candidates are also new distractors. (Worth +3.3 pp for the *Laya* backend — the same change has opposite signs on the two architectures) |

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
