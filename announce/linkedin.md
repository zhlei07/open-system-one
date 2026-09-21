# LinkedIn

**Post** (no hashtag spam; LinkedIn rewards a plain, specific story)

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
