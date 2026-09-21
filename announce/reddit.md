# r/LocalLLaMA

**Title:**

    I benchmarked the hyped Jev decision model against open CPU-only stacks. A 149M
    cross-encoder lands 0.6pp behind it, and an 8-float head hits 78.2% at 0.1ms.

**Body:**

TypeSafe's Jev came out last week: no text generation, you send it state + options and
it returns a typed choice with calibrated probabilities. Options are supplied per
request, which means it can't be a fixed classifier head — it has to score
(state, option) pairs. So I tried to rebuild that from open parts and measured the gap
instead of guessing.

**Setup.** 10,000 decisions: SST-2 (2 options), AG News (4), Emotion (6), BANKING77
(77). One machine, Apple M5 Pro, **CPU only, batch=1**, each model in its own process
(RSS is meaningless otherwise — a second model loaded in the same process can show a
*negative* delta). Deterministic splits from a fixed seed.

**Zero-shot, nobody gets labels:**

| stack | macro | p50 | RAM |
|---|---|---|---|
| typesafe/jev (hosted) | 79.3% | 381 ms | cloud |
| cross-encoder ModernBERT-base-zeroshot 149M | **78.7%** | 50–616 ms | — |
| bi-encoder + 8-float head | 69.9% | 15.7 ms | 490 MB |
| Qwen2.5-0.5B generative | 49.6% | 112 ms | 2382 MB |

The generative baseline is the one to look at if you're tempted to use a small
instruct model as a classifier: 49.6%, and **4.0% of its outputs couldn't be parsed
into a legal option at all**. On 77-way routing it got 14.2%.

**With ~2k labels per task** (different problem, separate table — Jev gets none):

| stack | macro | p50 | RAM |
|---|---|---|---|
| ModernBERT-embed + linear head | 82.6% | 15.8 ms | 490 MB |
| MiniLM-L6 + linear head | 79.5% | 2.8 ms | 452 MB |
| **potion-base-8M static + linear head** | **78.2%** | **0.1 ms** | **82 MB** |

That last row is a static embedding — lookup table, no attention, no matmul in the
forward pass — with a decision layer of eight floats, trained in 0.1 seconds. On
BANKING77 it gets 88.8% vs Jev's 77.8%.

**Eight things that didn't work**, which is most of what I learned:

- template ensembling (3 NLI hypotheses): **0.0pp** — the errors are correlated
- evolutionary search over head weights, honest leave-one-dataset-out: **−1.5pp**
- scaling the cross-encoder 149M → 395M: **+2.8 / −3.2 / +0.2** across three tasks
- listwise reranking with Qwen2.5-1.5B over the top-5: **−1.0pp**
- column-centering NLI scores: 0.0pp macro
- asymmetric `search_query:`/`search_document:` prefixes: **−1.8pp**
- distilling the cross-encoder into the bi-encoder head: **−5.4pp** vs teacher
- raw ModernBERT-base (MLM) as a sentence encoder: **35.0%**, worse than the 8M static
  model — training objective beats scale, decisively

The only thing that worked was architectural: joint (text, option) scoring instead of
two separate embeddings. +12 to +24pp depending on the task.

**Caveats I'd rather state than be corrected on:** Jev's 381ms is almost entirely
network (2 options and 77 options both take ~381ms), so "local is faster" means I
skipped a round trip. Single seed, no confidence intervals. And there is no neutral
set of option descriptions — the same rewrite moved Jev +0.5pp, the bi-encoder +6.3pp
and the cross-encoder −4.3pp.

Code, raw outputs, per-item predictions and the failures:
https://github.com/zhlei07/open-system-one

Model (8 floats): https://huggingface.co/dylantom2012/fly-head-potion-8m
Demo: https://huggingface.co/spaces/dylantom2012/open-system-one

Direction was mine; every experiment was designed and run by Claude Code in one
session, including two measurement bugs it introduced and then caught. Everything is
committed so you can check rather than trust — if I've misconfigured Jev, tell me and
I'll re-run.
