# Launch sequence — Monday 20:00 SGT (08:00 US Eastern)

Why this window: HN's front page is decided in the first 30–60 minutes, and US Eastern
morning is when the most people are online to vote. Posting at 04:40 ET, as we nearly
did, buries the submission.

| # | when | platform | file | notes |
|---|---|---|---|---|
| 1 | 20:00 | **Hacker News** — Show HN | `hn.md` | Post, then immediately add the first comment. Do not ask anyone to vote. |
| 2 | 20:10 | **X / Twitter** | `x.md` | 7-post thread. Quote the HN link in the last post once it exists. |
| 3 | 22:00–23:00 | **r/LocalLLaMA** | `reddit.md` | Wait 2h+ after HN — simultaneous cross-posting reads as spam. |
| 4 | next morning | **LinkedIn** | `linkedin.md` | Different audience, different rhythm; no rush. |
| 5 | next morning | **Lobste.rs** | `lobsters.md` | Needs an invite. Skip if you don't have an account. |
| 6 | any time | **TikTok / YT Shorts** | rendered `final.mp4` + `post.md` | 9:16 for TikTok/Shorts/Reels, 16:9 for YouTube. |

## Rules for the first two hours

- **Answer every comment, fast.** Early engagement is most of what decides HN ranking.
- **Concede immediately when someone is right.** The credibility of this whole thing
  rests on the limitations section; defending a weak point destroys more than it saves.
- **Never ask for upvotes**, not even indirectly. HN detects and penalises it.
- If someone says Jev was misconfigured: thank them, ask for the exact config, offer to
  re-run. We have the harness; a re-run costs $0.30 and 10 minutes.

## The three questions that will definitely come up

**"You compared few-shot against zero-shot."**
Correct, and they're in separate tables for exactly that reason. The headline
comparison is zero-shot-vs-zero-shot: 78.7 vs 79.3.

**"Your latency comparison is local compute vs a network round trip."**
Also correct, and it's the first item in Limitations. Jev's own compute is too small
for us to measure — 2 options and 77 options both take ~381ms.

**"Single seed, no error bars."**
True. Sub-sample checks moved results by up to ±2pp, which is stated. The per-item
predictions are published so anyone can run McNemar themselves.
