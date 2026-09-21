---
license: mit
library_name: numpy
tags:
  - text-classification
  - zero-shot-classification
  - intent-classification
  - static-embeddings
  - cpu
  - efficient-inference
datasets:
  - PolyAI/banking77
  - fancyzhx/ag_news
  - stanfordnlp/sst2
  - dair-ai/emotion
base_model: minishlab/potion-base-8M
pipeline_tag: text-classification
---

# Fly head — a text classifier whose decision layer is eight floats

```json
{"proto": 3.7233, "desc": 0.4448, "knn1": 3.0027, "knnk": 4.1251,
 "margin": 2.4977, "prior": 0.4307, "white": 1.5815, "tight": 0.6906}
```

That is the entire trained decision layer. It sits on top of
[`minishlab/potion-base-8M`](https://huggingface.co/minishlab/potion-base-8M), a static
embedding — **no attention, no matrix multiply in the forward pass, just a lookup table
and an average.**

Measured on 10,000 held-out examples across four datasets:

| | value |
|---|---|
| Macro accuracy | **78.2 %** |
| p50 end-to-end latency (CPU, batch = 1) | **0.1 ms** |
| Peak RSS | **82 MB** |
| Time to train the head | **0.1 s** |

For reference, `typesafe/jev` — a hosted commercial decision model — scores **78.8 %**
on the same 10,000 items at 381 ms per call. Jev needs **no labels**; this head needs
~2,000 per task. That difference is the whole story, and it is discussed honestly in
the [benchmark repository](https://github.com/zhlei07/open-system-one).

## Per-task accuracy

| dataset | options | this head | typesafe/jev |
|---|---|---|---|
| SST-2 | 2 | 78.3 % | 89.8 % |
| AG News | 4 | 87.7 % | 88.3 % |
| Emotion | 6 | 67.3 % | 58.6 % |
| BANKING77 | 77 | 79.4 % | 78.4 % |

## Use it

```bash
pip install numpy model2vec
python inference.py banking77 "I still haven't received the card I ordered"
#   45.3%  card arrival
#    7.6%  pending card payment
#    6.8%  card delivery estimate
```

```python
from inference import FlyHead
head = FlyHead("banking77")
head("my contactless stopped working")   # -> [(option, probability), ...]
```

## How it works

The head scores every *(query, option)* pair with eight normalised features and takes
the argmax — the same shape as a ranking function, which is why it supports any option
set rather than a fixed output layer:

| feature | meaning |
|---|---|
| `proto` | cosine to the class centroid |
| `desc` | cosine to the option's name/description |
| `knn1` / `knnk` | nearest / mean top-5 neighbour similarity within the class |
| `margin` | centroid cosine, centred across options |
| `prior` | class log-prior |
| `white` | cosine after subtracting the global mean (kills hubness) |
| `tight` | negative within-class spread — how trustworthy this class's cluster is |

The eight weights were found with a **(1+λ) evolution strategy**, λ=14 with σ-restarts,
maximising `0.7 × mean accuracy + 0.3 × worst-task accuracy` across the four tasks.
Weights are selected on the *training* objective, never on validation score.

`heads/<dataset>.npz` holds the offline state the features need: class centroids,
1,500 fit vectors for the kNN features, option-description vectors, the global mean and
the class log-priors. About 1.4 MB per task.

## Limitations

- **It needs labelled examples** (~2,000 per task). For a genuinely cold start, use a
  zero-shot cross-encoder instead — the benchmark repo measures that path too (78.7 %
  macro, 50–616 ms, CPU).
- **Adding a new option requires recomputing that class's centroid**, which needs
  examples of it. Nothing needs retraining, but you do need data.
- **Single seed, single machine.** No confidence intervals. See the repo's limitations.
- Trained and evaluated on four English datasets. Nothing here has been tested in
  another language.

## Citation / provenance

Built and measured in [`zhlei07/open-system-one`](https://github.com/zhlei07/open-system-one).
The direction came from the repository owner; every experiment was designed and run by
Claude Code (Opus 5) in a single session, including the ideas that failed. All raw
outputs are published so the numbers can be checked rather than trusted.

Code: MIT. The base encoder and the datasets retain their own licences.
