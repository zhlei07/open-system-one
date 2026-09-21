# 首发执行单 · 2026-09-21 周一

闹钟已设置：**19:50 / 20:00 / 20:10 / 20:30 / 22:00**（macOS 通知 + 语音）

链接：
- Repo  https://github.com/zhlei07/open-system-one
- Model https://huggingface.co/dylantom2012/fly-head-potion-8m
- Space https://huggingface.co/spaces/dylantom2012/open-system-one
- Data  https://huggingface.co/datasets/dylantom2012/open-system-one-bench

---

## ⏰ 20:00 — 第 1 步：Hacker News

标签页已填好表单，**你只需要点 submit**。如果表单空了，重新填：

**title**（78 字符，上限 80）
```
Show HN: A text classifier whose decision layer is eight floats (78.2%, 0.1ms)
```

**url**
```
https://github.com/zhlei07/open-system-one
```

**text** — 留空。

### 发完立刻贴这条首评（最关键，头 30 分钟决定排名）

```
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

Full disclosure, and it's the part I'd most like feedback on: **I don't program.**
The hunch was mine, every experiment was designed and run by Claude Code in one
session — including two measurement bugs it introduced and then caught (a latency run
contaminated by concurrent HTTP traffic, and a results file it overwrote).

That's not "the AI did it all", though. Three interventions changed the outcome and
none of them needed a line of code:

- *"Drop Qwen."* It had started ensembling a 1.5B generative model into a stack whose
  entire premise was cheap local parts, for a gain that turned out to be negative.
- *"Test under genuinely fair conditions — we need the real numbers before we decide."*
  That forced the enriched-description re-run, which produced the most interesting
  result in the repo.
- *"Are you sure the task hasn't drifted?"* It had.

If there's a transferable lesson here it's that the bottleneck wasn't code, it was
knowing when to stop something that was going nowhere. All raw outputs are committed,
so please check the numbers rather than trust them — if I've misconfigured Jev, open
an issue and I'll re-run.

Repo: https://github.com/zhlei07/open-system-one
Model: https://huggingface.co/dylantom2012/fly-head-potion-8m
Demo: https://huggingface.co/spaces/dylantom2012/open-system-one
```

---

## ⏰ 20:10 — 第 2 步：X thread（7 条）

逐条发，每条回复上一条。**最后一条发完后，把 HN 帖子链接补在第 7 条的回复里。**

```
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

I don't write code. I gave Claude Code one hunch and spent the day telling it when it
was going nowhere — dropping a dead-end ensemble, forcing a fair re-test, asking "has
this drifted?" once. That turned out to be the whole job.

Check the numbers, don't trust them.
```

---

## ⏰ 20:30 — 回帖检查

去 HN 看评论。三条纪律：

1. **答得快** —— 早期互动是 HN 排名的主要变量
2. **对方说得对就立刻认** —— 这份工作的可信度建立在局限性写得诚实上，为弱点辩护赔得更多
3. **永远不要求点赞** —— HN 会检测并惩罚

### 三个必然被问到的问题，答案

**"你拿 few-shot 比 zero-shot"**
> Correct, and they're in separate tables for exactly that reason. The headline
> comparison is zero-shot vs zero-shot: 78.7 vs 79.3.

**"你的延迟是本地算力 vs 一次网络往返"**
> Also correct, and it's the first item in Limitations. Jev's own compute is too small
> for us to measure — 2 options and 77 options both take ~381ms.

**"单种子、没有误差棒"**
> True. Sub-sample checks moved results by up to ±2pp, which is stated. The per-item
> predictions are published so anyone can run McNemar themselves.

**"Jev 你配错了"**
> 谢谢 + 要具体配置 + 承诺重跑。我们有现成的 harness，重跑一次 $0.30、10 分钟。

---

## ⏰ 22:00 — 第 3 步：Reddit r/LocalLLaMA

⚠️ **先确认账号 karma**。零 karma 的新账号会被自动删帖——如果 karma 很低，今晚跳过，改成次日。

**标题**
```
I benchmarked the hyped Jev decision model against open CPU-only stacks. A 149M cross-encoder lands 0.6pp behind it, and an 8-float head hits 78.2% at 0.1ms.
```

**正文**
```
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

**On how this was built**, since it's relevant to whether you should trust it: I don't
program. The hunch was mine; Claude Code designed and ran every experiment in one
session, including two measurement bugs it introduced and then caught. My actual
contribution was three course corrections — killing a dead-end generative ensemble,
demanding a genuinely fair re-test, and asking once whether the task had drifted (it
had). Everything is committed so you can check rather than trust. If I've
misconfigured Jev, tell me and I'll re-run.
```

---

## 次日早上 — LinkedIn

```
I don't write code. Last week I read about Jev — the model that doesn't generate text,
it just returns a typed decision — and had one thought: isn't that a fast text encoder
with a small decision head on top?

Instead of speculating, I handed the question to Claude Code and let it run experiments
for a day. 10,000 decisions, four public datasets, CPU only.

Zero-shot, neither side given labels:
· typesafe/jev — 79.3%
· an open 149M cross-encoder on a laptop CPU — 78.7%

0.6 points apart.

And the part I keep coming back to: an 8M static embedding plus a decision layer of
eight floating-point numbers reaches 78.2% at 0.1 milliseconds in 82 MB. It trains in a
tenth of a second.

What I actually did, over a full day, was three things — none of which required
programming:
1. Told it to drop a dead-end experiment it had wandered into
2. Insisted the comparison be re-run under genuinely fair conditions
3. Asked once whether the task had drifted from the original question (it had)

That last one is the part I'd tell anyone working with these tools. The bottleneck
isn't writing code any more. It's judgement about when to stop.

Everything is public, including the eight approaches that failed and every raw output
file, so the numbers can be checked rather than trusted:
github.com/zhlei07/open-system-one

#AI #MachineLearning #OpenSource
```

---

## 次日 — 短视频

竖版（TikTok / Shorts / Reels）和横版（YouTube）各一条，55 秒：

```
~/projects/moneyprintprobo/storage/tasks/4b03359b-2858-4d88-bf0a-cdff3d34239b/tiktok/final.mp4
~/projects/moneyprintprobo/storage/tasks/4b03359b-2858-4d88-bf0a-cdff3d34239b/youtube/final.mp4
```

文案在同目录的 `post.md`。
