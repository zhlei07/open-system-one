# Lobste.rs

**Tags:** `ai`, `ml`, `performance`, `show`

**Title:**

    Benchmarking TypeSafe's Jev against open CPU-only stacks: 10,000 decisions

**Comment:**

Jev shipped last week — non-generative, returns a typed choice with calibrated
probabilities, options supplied per request. Since `criteria` arrive per call it can't
be a fixed classifier head; it has to score (state, option) pairs. I wanted to know how
much of that is reproducible from open parts.

10,000 decisions across SST-2 / AG News / Emotion / BANKING77 (2→4→6→77 options).
One machine, CPU only, batch=1, each model in its own process. Zero-shot:

    typesafe/jev (hosted)       79.3%   381ms
    cross-encoder, 149M, CPU    78.7%   50-616ms
    bi-encoder + 8-float head   69.9%   15.7ms
    Qwen2.5-0.5B generative     49.6%   112ms, 4.0% unparseable

Methodology notes, since this is the part that took the longest to get right:

- RAM has to be measured one model per process. Load a second model in the same process
  and the RSS delta can come out *negative* — the allocator reuses the first one's
  freed pages.
- A latency run that overlapped with concurrent HTTP traffic read 29.7ms instead of
  15.7ms. Nearly 2x, silently.
- Jev's 381ms is essentially all network: 2 options and 77 options both take ~381ms
  while input tokens differ 5x.
- There is no neutral set of option descriptions. The same rewrite moved Jev +0.5pp,
  the bi-encoder +6.3pp and the cross-encoder -4.3pp. On BANKING77 the cross-encoder
  lost 11.9pp because 70 of my 77 descriptions began "The customer…" — shared prefix,
  near-identical NLI hypotheses.

Eight things that didn't work are published too, which is most of what I learned.
Raw outputs are all committed: https://github.com/zhlei07/open-system-one
