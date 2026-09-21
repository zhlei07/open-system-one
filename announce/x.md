# X / Twitter

**Thread — 7 posts. Lead with the artifact, not the comparison.**

1/
The entire trained decision layer of this text classifier:

{"proto": 3.72, "desc": 0.44, "knn1": 3.00, "knnk": 4.13,
 "margin": 2.50, "prior": 0.43, "white": 1.58, "tight": 0.69}

Eight floats. 78.2% macro on 10k decisions. 0.1 ms. 82 MB. Trains in 0.1 s.

2/
It sits on an 8M static embedding — a lookup table. No attention. No matmul in the
forward pass.

For reference, typesafe/jev (hosted, commercial) scores 78.8% on the same 10,000 items
at 381 ms per call.

3/
Caveat that matters: the 8-float head needs ~2,000 labelled examples per task. Jev
needs zero. Those are different problems and I keep them in different tables.

Zero-shot, nobody gets labels:

jev            79.3%
cross-encoder  78.7%  ← 149M, open, CPU
bi-encoder     69.9%
Qwen2.5-0.5B   49.6%

4/
The surprise for me: 0.6pp. A 149M open model on a laptop CPU is 0.6 points behind a
hosted commercial decision model.

The thing that bought those points wasn't scale or prompting. It was scoring
(text, option) *jointly* instead of comparing two embeddings. +12 to +24pp.

5/
What didn't work, all published:

· template ensembling — 0.0pp
· evolutionary weight search (honest LODO) — −1.5pp
· 149M → 395M backbone — net zero
· listwise reranking w/ 1.5B — −1.0pp
· distillation — −5.4pp vs teacher

Scale and prompt tricks bought nothing. Architecture bought everything.

6/
Finding I didn't expect: there is no neutral set of option descriptions.

Same rewrite, three directions:
jev           +0.5
bi-encoder    +6.3
cross-encoder −4.3

On 77-way routing the cross-encoder lost 11.9pp because 70 of my descriptions started
with "The customer…". Shared prefix → NLI hypotheses become indistinguishable. My bug,
documented.

7/
Everything is public — code, raw outputs, per-item predictions, the failures.

Go break it:
→ github.com/zhlei07/open-system-one
→ huggingface.co/spaces/dylantom2012/open-system-one

Direction from me, every experiment run by Claude Code in one session. Check the
numbers, don't trust them.
