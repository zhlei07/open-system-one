# Hacker News

**Title** (80 char limit — the number does the work, no "we beat X"):

    Show HN: A text classifier whose decision layer is eight floats (78.2%, 0.1ms)

**Alternative titles**, if the first feels too narrow:

    Show HN: Benchmarking TypeSafe's Jev against open CPU-only stacks (10k decisions)
    Show HN: Eight floats, 0.1ms, 82MB – matching a hosted decision model's accuracy

**First comment** (post immediately after submitting — HN rewards the author
explaining scope and limits before anyone has to ask):

---

Author here. Background: TypeSafe shipped Jev a week ago — a model that doesn't
generate text, it returns a typed choice with calibrated probabilities, with the
options supplied per request. The obvious guess was "that's a fast encoder plus a tiny
decision head." I wanted to know how much of it you can rebuild from open parts, so I
measured instead of speculating.

10,000 decisions across SST-2, AG News, Emotion and BANKING77 (2→4→6→77 options), one
machine, CPU only, batch=1, every model measured in its own process.

Zero-shot, no labels on either side:

    typesafe/jev (hosted)        79.3%   381ms
    cross-encoder, 149M, CPU     78.7%   50-616ms
    bi-encoder + 8-float head    69.9%   15.7ms
    Qwen2.5-0.5B, generative     49.6%   112ms  (4.0% unparseable outputs)

So an open 149M model on a laptop CPU lands 0.6pp behind. That surprised me.

With ~2k labels per task the picture flips: a bi-encoder plus a linear head hits 82.6%
at 15.8ms, and — the part I did not expect — an 8M *static* embedding (a lookup table,
no attention, no matmul in the forward pass) plus a decision layer that is literally
eight floats reaches 78.2% at 0.1ms in 82MB. The head trains in 0.1 seconds. On
BANKING77, 77-way intent routing, it gets 88.8% vs Jev's 77.8%.

Those two regimes are different problems and I've kept them in separate tables. Jev's
pitch is "works with zero labels", and that is real: give it nothing and it beats every
zero-shot open stack I tried on the hardest task. The counter-argument is that most
production decision points accumulate labels by themselves — support tickets get
resolved, routes get corrected — so the question isn't which is better, it's when you
switch.

Two things I'd flag before anyone quotes the latency numbers at me:

1. Jev's 381ms is almost entirely network. Two options and 77 options both take ~381ms
   (input tokens differ 5x, latency differs 5%). Its own compute is too small for me to
   measure. "Local is faster" means I skipped a round trip, not that I out-computed it.
2. There is no neutral set of option descriptions. Replacing terse labels with
   one-sentence descriptions moved Jev +0.5pp, the bi-encoder +6.3pp, and the
   cross-encoder **-4.3pp** — same text, three directions. On BANKING77 the
   cross-encoder dropped 11.9pp because 70 of my 77 descriptions began with "The
   customer…", which makes them near-identical as NLI hypotheses. That one's my bug and
   it's in the README.

I also published the eight things that didn't work, which is most of what I actually
learned: template ensembling (0.0pp), evolutionary weight search under honest
leave-one-dataset-out transfer (-1.5pp), scaling the cross-encoder 149M→395M (net
zero across three tasks), listwise reranking with a 1.5B model (-1.0pp), distillation
(-5.4pp vs the teacher). The one thing that did work was architectural: scoring
(text, option) jointly instead of comparing two separate embeddings, worth +12 to +24pp.

Full disclosure: I supplied the direction; every experiment was designed and run by
Claude Code in one session, including two measurement bugs it introduced and then
caught (a latency run contaminated by concurrent HTTP, and a results file it
overwrote). All raw outputs are committed, so please check the numbers rather than
trust them — if I've misconfigured Jev, open an issue and I'll re-run.

Repo: https://github.com/zhlei07/open-system-one
Model: https://huggingface.co/dylantom2012/fly-head-potion-8m
Demo: https://huggingface.co/spaces/dylantom2012/open-system-one
