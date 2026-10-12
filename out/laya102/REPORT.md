# Laya #102 protocol, run on our machines

Rows: `benchmark.py`'s seed-0 sample, 500 per official test split (the rows of #950's 500-row table). Question as in #950: state `{"message": text}`, labels with spaces, no descriptions. top-20 retrieval, tournament groups of at most 16. Latency per request = query encoding + ranking + every decision round; models and label vectors warmed; each arm warmed on its own call shape on dev rows; arms interleaved per row in rotating order.

Laya 0.4.4 (checkpoint snapshot `55cf4c4ebb`), torch 2.14.1, defaults max_len=512 head_max_len=192; `wide` = 512/1024.

Embedders, chosen on dev rows only (highest mean dev recall@20 over the three datasets, ties by recall@1): small = `bge-small/instruct/bare`, Nomic reference = `nomic/sym/bare`.

Machines: A and B, two Apple-silicon laptops. MPS = torch on the Apple GPU, CPU = torch on the CPU.


## Development-row selection (mean over the three datasets)

| embedder / prefix / query text | recall@1 | recall@20 |
|---|---:|---:|
| `bge-small/instruct/bare` | 65.0 | 96.6 |
| `bge-small/none/bare` | 65.7 | 96.1 |
| `minilm/none/bare` | 60.1 | 94.7 |
| `nomic/sym/bare` | 61.6 | 94.5 |
| `nomic/asym/bare` | 59.7 | 93.5 |
| `minilm/none/library` | 37.8 | 89.2 |
| `bge-small/instruct/library` | 32.1 | 88.5 |
| `bge-small/none/library` | 29.3 | 88.2 |
| `nomic/sym/library` | 41.7 | 86.7 |
| `nomic/asym/library` | 29.0 | 82.2 |

`library` = the query text `laya.shortlist` embeds (instructions + newline + JSON state); `bare` = the message alone.


## Accuracy, A MPS (95 % Wilson interval)

| arm | banking77 | clinc150 | massive |
|---|---:|---:|---:|
| `single` | 41.6 [37.4–46.0] | 60.4 [56.0–64.6] | 54.2 [49.8–58.5] |
| `wide` | 55.4 [51.0–59.7] | 68.0 [63.8–71.9] | 62.2 [57.9–66.3] |
| `tournament` | 58.4 [54.0–62.6] | 89.4 [86.4–91.8] | 59.0 [54.6–63.2] |
| `small:top1` | 66.2 [61.9–70.2] | 70.4 [66.3–74.2] | 58.4 [54.0–62.6] |
| `small:shortlist` | 63.0 [58.7–67.1] | 86.8 [83.5–89.5] | 64.4 [60.1–68.5] |
| `small:shortlist+tournament` | 62.2 [57.9–66.3] | 86.4 [83.1–89.1] | 64.0 [59.7–68.1] |
| `nomic:top1` | 64.4 [60.1–68.5] | 71.4 [67.3–75.2] | 50.8 [46.4–55.2] |
| `nomic:shortlist` | 63.4 [59.1–67.5] | 87.8 [84.6–90.4] | 60.2 [55.9–64.4] |
| `nomic:shortlist+tournament` | 62.8 [58.5–66.9] | 86.6 [83.3–89.3] | 63.2 [58.9–67.3] |

## Same rows, #950's published 500-row table (Laya 0.3.27, MPS) vs ours

| dataset | arm | #950 | ours (A MPS) |
|---|---|---:|---:|
| banking77 | `single` | 41.6 | 41.6 |
| banking77 | `wide` | 55.4 | 55.4 |
| banking77 | `tournament` | 58.4 | 58.4 |
| banking77 | `shortlist` (embed_fn_from_agent) | 22.4 | — |
| clinc150 | `single` | 60.4 | 60.4 |
| clinc150 | `wide` | 68.0 | 68.0 |
| clinc150 | `tournament` | 89.4 | 89.4 |
| clinc150 | `shortlist` (embed_fn_from_agent) | 18.8 | — |
| massive | `single` | 54.2 | 54.2 |
| massive | `wide` | 62.2 | 62.2 |
| massive | `tournament` | 59.0 | 59.0 |
| massive | `shortlist` (embed_fn_from_agent) | 18.4 | — |

## Errors, A MPS (rows out of 500)

| dataset | arm | retrieval miss | elimination miss | final / decision error |
|---|---|---:|---:|---:|
| banking77 | `tournament` | — | 112 | 96 |
| banking77 | `small:shortlist` | 9 | — | 176 |
| banking77 | `small:shortlist+tournament` | 9 | 162 | 18 |
| banking77 | `nomic:shortlist` | 16 | — | 167 |
| banking77 | `nomic:shortlist+tournament` | 16 | 157 | 13 |
| clinc150 | `tournament` | — | 14 | 39 |
| clinc150 | `small:shortlist` | 21 | — | 45 |
| clinc150 | `small:shortlist+tournament` | 21 | 38 | 9 |
| clinc150 | `nomic:shortlist` | 14 | — | 47 |
| clinc150 | `nomic:shortlist+tournament` | 14 | 39 | 14 |
| massive | `tournament` | — | 112 | 93 |
| massive | `small:shortlist` | 26 | — | 152 |
| massive | `small:shortlist+tournament` | 26 | 135 | 19 |
| massive | `nomic:shortlist` | 58 | — | 141 |
| massive | `nomic:shortlist+tournament` | 58 | 105 | 21 |

## Paired comparisons, A MPS (delta in points; gained / lost rows; exact McNemar p)

| comparison | banking77 | clinc150 | massive |
|---|---|---|---|
| `tournament vs single` | +16.8 (126/42, p<0.001) | +29.0 (161/16, p<0.001) | +4.8 (85/61, p=0.057) |
| `wide vs single` | +13.8 (96/27, p<0.001) | +7.6 (53/15, p<0.001) | +8.0 (58/18, p<0.001) |
| `small:shortlist vs small:top1` | -3.2 (43/59, p=0.137) | +16.4 (96/14, p<0.001) | +6.0 (68/38, p=0.005) |
| `small:shortlist+tournament vs small:top1` | -4.0 (43/63, p=0.065) | +16.0 (95/15, p<0.001) | +5.6 (72/44, p=0.012) |
| `small:shortlist vs tournament` | +4.6 (54/31, p=0.017) | -2.6 (19/32, p=0.092) | +5.4 (66/39, p=0.011) |
| `small:shortlist+tournament vs tournament` | +3.8 (47/28, p=0.037) | -3.0 (16/31, p=0.040) | +5.0 (66/41, p=0.020) |
| `small:shortlist+tournament vs small:shortlist` | -0.8 (24/28, p=0.678) | -0.4 (12/14, p=0.845) | -0.4 (32/34, p=0.902) |
| `tournament vs small:top1` | -7.8 (40/79, p<0.001) | +19.0 (117/22, p<0.001) | +0.6 (77/74, p=0.871) |
| `nomic:shortlist vs small:shortlist` | +0.4 (22/20, p=0.878) | +1.0 (21/16, p=0.511) | -4.2 (23/44, p=0.014) |
| `nomic:shortlist+tournament vs small:shortlist+tournament` | +0.6 (23/20, p=0.761) | +0.2 (18/17, p=1.000) | -0.8 (30/34, p=0.708) |

## Latency, ms (p50 / p95)

| arm | banking77 A MPS | banking77 B CPU | banking77 A CPU | banking77 B MPS | clinc150 A MPS | clinc150 B CPU | massive A MPS | massive B CPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `single` | 53.1 / 66.8 | 168.2 / 223.8 | 191.3 / 207.8 | 56.2 / 80.9 | 88.6 / 109.2 | 229.9 / 235.7 | 40.6 / 50.1 | 134.5 / 137.4 |
| `wide` | 61.9 / 79.3 | 193.6 / 303.7 | 227.2 / 244.8 | 62.0 / 94.9 | 91.9 / 113.9 | 242.9 / 253.8 | 45.8 / 55.6 | 140.1 / 149.9 |
| `tournament` | 70.3 / 86.4 | 303.8 / 449.4 | 361.5 / 405.9 | 113.0 / 177.0 | 91.7 / 112.5 | 360.5 / 402.1 | 72.1 / 90.9 | 224.4 / 238.3 |
| `small:top1` | 5.7 / 6.8 | 8.1 / 10.9 | 5.5 / 6.5 | 7.3 / 8.2 | 6.1 / 7.3 | 7.7 / 8.2 | 5.7 / 6.5 | 7.7 / 8.1 |
| `small:shortlist` | 33.7 / 40.9 | 105.1 / 130.9 | 97.5 / 112.4 | 37.4 / 50.0 | 32.1 / 40.2 | 88.0 / 96.6 | 29.0 / 35.3 | 95.2 / 97.7 |
| `small:shortlist+tournament` | 57.2 / 69.8 | 174.9 / 215.1 | 175.0 / 199.2 | 60.8 / 82.7 | 55.4 / 68.2 | 155.7 / 165.3 | 49.9 / 59.6 | 160.8 / 167.1 |
| `nomic:top1` | 8.2 / 9.5 | 22.8 / 28.5 | 15.3 / 17.5 | 10.4 / 12.9 | 8.4 / 9.8 | 22.4 / 24.1 | 7.7 / 9.2 | 22.1 / 23.5 |
| `nomic:shortlist` | 36.5 / 44.5 | 119.5 / 140.6 | 109.0 / 127.3 | 41.5 / 52.9 | 34.5 / 42.5 | 102.8 / 111.5 | 31.3 / 37.8 | 109.4 / 112.7 |
| `nomic:shortlist+tournament` | 59.9 / 74.5 | 189.7 / 229.7 | 192.5 / 211.7 | 64.5 / 87.0 | 57.6 / 70.1 | 171.2 / 180.1 | 52.3 / 63.6 | 175.4 / 182.9 |

- banking77 A MPS: torch threads 6, wall 201 s, thermal pressure by row: nominal 401, heavy 49, between samples 50
- banking77 B CPU: torch threads 10, wall 681 s, thermal pressure by row: nominal 401, between samples 99
- banking77 A CPU: torch threads 6, wall 697 s, thermal pressure by row: nominal 200, moderate 149, heavy 1, between samples 150
- banking77 B MPS: torch threads 10, wall 244 s, thermal pressure by row: nominal 151, heavy 249, between samples 100
- clinc150 A MPS: torch threads 6, wall 241 s, thermal pressure by row: heavy 500
- clinc150 B CPU: torch threads 10, wall 696 s, thermal pressure by row: nominal 151, moderate 150, heavy 99, between samples 100
- massive A MPS: torch threads 6, wall 174 s, thermal pressure by row: moderate 301, heavy 149, between samples 50
- massive B CPU: torch threads 10, wall 534 s, thermal pressure by row: nominal 49, moderate 300, heavy 51, between samples 100

### Latency by thermal pressure, ms (p50 / p95)

macOS thermal pressure is sampled after every 50th row; a row is counted at a level only when the samples on both sides of it agree; levels with fewer than 20 rows are not shown. Arms run interleaved, so every arm sees the same heat; accuracy does not depend on it.

| arm | banking77 A MPS, nominal (401 rows) | banking77 A MPS, heavy (49 rows) | banking77 A CPU, nominal (200 rows) | banking77 A CPU, moderate (149 rows) | banking77 B MPS, nominal (151 rows) | banking77 B MPS, heavy (249 rows) | clinc150 B CPU, nominal (151 rows) | clinc150 B CPU, moderate (150 rows) | clinc150 B CPU, heavy (99 rows) | massive A MPS, moderate (301 rows) | massive A MPS, heavy (149 rows) | massive B CPU, nominal (49 rows) | massive B CPU, moderate (300 rows) | massive B CPU, heavy (51 rows) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `single` | 52.3 / 62.3 | 65.4 / 75.6 | 190.9 / 206.6 | 191.6 / 207.6 | 55.3 / 62.9 | 60.6 / 85.3 | 230.2 / 239.3 | 229.9 / 234.6 | 229.8 / 232.4 | 40.4 / 43.2 | 47.7 / 53.5 | 134.3 / 137.3 | 134.5 / 137.3 | 134.5 / 137.5 |
| `wide` | 61.5 / 69.7 | 78.8 / 86.2 | 226.8 / 241.6 | 227.4 / 241.8 | 61.7 / 69.8 | 68.2 / 100.3 | 243.9 / 267.5 | 242.8 / 245.1 | 242.5 / 245.3 | 43.9 / 46.7 | 50.9 / 58.6 | 139.5 / 149.6 | 139.6 / 149.8 | 148.4 / 150.4 |
| `tournament` | 69.7 / 83.7 | 81.8 / 90.7 | 361.3 / 405.9 | 358.5 / 391.9 | 111.6 / 130.5 | 124.7 / 184.5 | 365.7 / 407.3 | 360.1 / 389.6 | 359.1 / 378.7 | 71.3 / 77.7 | 84.0 / 93.0 | 224.2 / 245.3 | 224.3 / 237.9 | 225.6 / 240.8 |
| `small:top1` | 5.6 / 6.6 | 6.2 / 7.2 | 5.4 / 6.5 | 5.5 / 6.4 | 7.2 / 7.7 | 7.4 / 8.3 | 7.7 / 8.3 | 7.7 / 8.3 | 7.7 / 8.2 | 5.7 / 5.9 | 6.1 / 6.6 | 7.6 / 8.2 | 7.7 / 8.2 | 7.7 / 8.1 |
| `small:shortlist` | 32.7 / 37.9 | 38.9 / 43.9 | 97.2 / 111.8 | 97.6 / 111.8 | 35.9 / 43.2 | 39.0 / 51.5 | 88.2 / 96.9 | 88.2 / 96.4 | 87.7 / 96.0 | 28.8 / 30.4 | 32.9 / 36.8 | 95.0 / 98.3 | 95.2 / 97.9 | 95.5 / 97.2 |
| `small:shortlist+tournament` | 56.2 / 65.2 | 67.2 / 72.6 | 169.8 / 198.5 | 182.2 / 194.9 | 60.4 / 69.5 | 63.8 / 85.3 | 155.8 / 165.3 | 155.5 / 165.5 | 155.4 / 164.5 | 49.1 / 52.0 | 55.4 / 62.5 | 160.8 / 166.9 | 160.7 / 167.1 | 161.3 / 167.3 |
| `nomic:top1` | 8.1 / 9.1 | 8.9 / 10.2 | 15.3 / 16.6 | 15.3 / 16.4 | 10.3 / 12.5 | 10.8 / 13.2 | 22.3 / 24.2 | 22.4 / 23.9 | 22.3 / 24.1 | 7.5 / 8.0 | 8.4 / 9.6 | 22.3 / 23.7 | 22.1 / 23.5 | 22.2 / 23.6 |
| `nomic:shortlist` | 35.6 / 40.9 | 41.8 / 46.3 | 108.4 / 123.5 | 109.4 / 125.8 | 40.8 / 46.8 | 42.4 / 54.9 | 103.2 / 111.9 | 102.9 / 111.1 | 102.2 / 110.9 | 30.8 / 34.1 | 35.3 / 39.8 | 109.3 / 112.9 | 109.3 / 112.6 | 109.9 / 112.7 |
| `nomic:shortlist+tournament` | 58.7 / 70.3 | 70.6 / 80.1 | 192.4 / 210.9 | 192.6 / 210.1 | 63.8 / 78.0 | 67.0 / 88.0 | 171.2 / 179.8 | 171.0 / 180.5 | 173.0 / 179.8 | 51.4 / 56.0 | 58.2 / 65.9 | 174.7 / 188.9 | 175.6 / 183.7 | 176.5 / 182.9 |

## Device agreement (same rows, same arm: rows whose answer differs from A MPS)

| run | `single` | `wide` | `tournament` | `small:top1` | `small:shortlist` | `small:shortlist+tournament` | `nomic:top1` | `nomic:shortlist` | `nomic:shortlist+tournament` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| banking77 A CPU | 0 (acc 41.6) | 0 (acc 55.4) | 1 (acc 58.4) | 0 (acc 66.2) | 0 (acc 63.0) | 0 (acc 62.2) | 0 (acc 64.4) | 0 (acc 63.4) | 0 (acc 62.8) |
| banking77 B CPU | 0 (acc 41.6) | 0 (acc 55.4) | 1 (acc 58.4) | 0 (acc 66.2) | 0 (acc 63.0) | 0 (acc 62.2) | 0 (acc 64.4) | 0 (acc 63.4) | 0 (acc 62.8) |
| banking77 B MPS | 0 (acc 41.6) | 0 (acc 55.4) | 1 (acc 58.4) | 0 (acc 66.2) | 0 (acc 63.0) | 0 (acc 62.2) | 0 (acc 64.4) | 0 (acc 63.4) | 0 (acc 62.8) |
| clinc150 B CPU | 0 (acc 60.4) | 0 (acc 68.0) | 0 (acc 89.4) | 0 (acc 70.4) | 0 (acc 86.8) | 0 (acc 86.4) | 0 (acc 71.4) | 0 (acc 87.8) | 0 (acc 86.6) |
| massive B CPU | 0 (acc 54.2) | 0 (acc 62.2) | 0 (acc 59.0) | 0 (acc 58.4) | 0 (acc 64.4) | 0 (acc 64.0) | 0 (acc 50.8) | 0 (acc 60.2) | 0 (acc 63.2) |

## Add-on: other embedder configurations, A MPS (k=20, same rows)

Embedder arms only; `single` / `wide` / `tournament` do not depend on the embedder and are the rows above. Roles: `minilm` = `minilm/none/bare`, `minilm-lib` = `minilm/none/library`, `nomic-qd` = `nomic/asym/bare`, `nomic-qd-lib` = `nomic/asym/library`. `-lib` = the query text `laya.shortlist` embeds (instructions + newline + JSON state); Nomic Q/D = `search_query: ` on the message, `search_document: ` on the labels.

| arm | banking77 | clinc150 | massive |
|---|---:|---:|---:|
| `minilm:top1` | 59.6 [55.2–63.8] | 70.6 [66.5–74.4] | 51.6 [47.2–56.0] |
| `minilm:shortlist` | 62.6 [58.3–66.7] | 86.6 [83.3–89.3] | 60.6 [56.2–64.8] |
| `minilm:shortlist+tournament` | 62.4 [58.1–66.5] | 87.6 [84.4–90.2] | 62.0 [57.7–66.1] |
| minilm recall@20 | 97.0 | 96.4 | 92.4 |
| `minilm-lib:top1` | 48.0 [43.6–52.4] | 34.0 [30.0–38.3] | 27.6 [23.9–31.7] |
| `minilm-lib:shortlist` | 62.8 [58.5–66.9] | 79.0 [75.2–82.3] | 60.2 [55.9–64.4] |
| `minilm-lib:shortlist+tournament` | 63.2 [58.9–67.3] | 79.2 [75.4–82.5] | 59.0 [54.6–63.2] |
| minilm-lib recall@20 | 94.2 | 86.0 | 84.6 |
| `nomic-qd:top1` | 63.4 [59.1–67.5] | 67.8 [63.6–71.8] | 45.0 [40.7–49.4] |
| `nomic-qd:shortlist` | 62.4 [58.1–66.5] | 86.4 [83.1–89.1] | 55.2 [50.8–59.5] |
| `nomic-qd:shortlist+tournament` | 61.8 [57.5–66.0] | 86.8 [83.5–89.5] | 58.0 [53.6–62.3] |
| nomic-qd recall@20 | 97.0 | 96.8 | 86.4 |
| `nomic-qd-lib:top1` | 43.2 [38.9–47.6] | 21.4 [18.0–25.2] | 28.2 [24.4–32.3] |
| `nomic-qd-lib:shortlist` | 62.0 [57.7–66.1] | 72.2 [68.1–75.9] | 53.4 [49.0–57.7] |
| `nomic-qd-lib:shortlist+tournament` | 61.2 [56.9–65.4] | 72.4 [68.3–76.1] | 57.4 [53.0–61.7] |
| nomic-qd-lib recall@20 | 88.6 | 77.0 | 79.8 |

Paired on the same rows (delta in points; gained / lost rows; exact McNemar p). `tournament` and `small:*` come from the main A MPS run.

| comparison | banking77 | clinc150 | massive |
|---|---|---|---|
| `minilm:shortlist vs minilm:top1` | +3.0 (59/44, p=0.167) | +16.0 (92/12, p<0.001) | +9.0 (74/29, p<0.001) |
| `minilm:shortlist+tournament vs minilm:shortlist` | -0.2 (23/24, p=1.000) | +1.0 (14/9, p=0.405) | +1.4 (32/25, p=0.427) |
| `minilm:shortlist vs tournament` | +4.2 (53/32, p=0.029) | -2.8 (19/33, p=0.070) | +1.6 (49/41, p=0.461) |
| `minilm:shortlist+tournament vs tournament` | +4.0 (47/27, p=0.026) | -1.8 (18/27, p=0.233) | +3.0 (62/47, p=0.180) |
| `minilm:shortlist vs small:shortlist` | -0.4 (19/21, p=0.875) | -0.2 (15/16, p=1.000) | -3.8 (21/40, p=0.020) |
| `minilm-lib:shortlist vs minilm-lib:top1` | +14.8 (110/36, p<0.001) | +45.0 (229/4, p<0.001) | +32.6 (171/8, p<0.001) |
| `minilm-lib:shortlist+tournament vs minilm-lib:shortlist` | +0.4 (27/25, p=0.890) | +0.2 (8/7, p=1.000) | -1.2 (24/30, p=0.497) |
| `minilm-lib:shortlist vs tournament` | +4.4 (54/32, p=0.023) | -10.4 (13/65, p<0.001) | +1.2 (49/43, p=0.602) |
| `minilm-lib:shortlist+tournament vs tournament` | +4.8 (51/27, p=0.009) | -10.2 (15/66, p<0.001) | +0.0 (47/47, p=1.000) |
| `minilm-lib:shortlist vs small:shortlist` | -0.2 (25/26, p=1.000) | -7.8 (15/54, p<0.001) | -4.2 (27/48, p=0.020) |
| `minilm-lib:shortlist vs minilm:shortlist` | +0.2 (28/27, p=1.000) | -7.6 (15/53, p<0.001) | -0.4 (27/29, p=0.894) |
| `nomic-qd:shortlist vs nomic-qd:top1` | -1.0 (62/67, p=0.725) | +18.6 (111/18, p<0.001) | +10.2 (82/31, p<0.001) |
| `nomic-qd:shortlist+tournament vs nomic-qd:shortlist` | -0.6 (21/24, p=0.766) | +0.4 (16/14, p=0.856) | +2.8 (41/27, p=0.114) |
| `nomic-qd:shortlist vs tournament` | +4.0 (53/33, p=0.040) | -3.0 (19/34, p=0.053) | -3.8 (45/64, p=0.084) |
| `nomic-qd:shortlist+tournament vs tournament` | +3.4 (47/30, p=0.068) | -2.6 (17/30, p=0.079) | -1.0 (50/55, p=0.697) |
| `nomic-qd:shortlist vs small:shortlist` | -0.6 (21/24, p=0.766) | -0.4 (22/24, p=0.883) | -9.2 (23/69, p<0.001) |
| `nomic-qd-lib:shortlist vs nomic-qd-lib:top1` | +18.8 (134/40, p<0.001) | +50.8 (258/4, p<0.001) | +25.2 (146/20, p<0.001) |
| `nomic-qd-lib:shortlist+tournament vs nomic-qd-lib:shortlist` | -0.8 (21/25, p=0.659) | +0.2 (9/8, p=1.000) | +4.0 (41/21, p=0.015) |
| `nomic-qd-lib:shortlist vs tournament` | +3.6 (58/40, p=0.085) | -17.2 (14/100, p<0.001) | -5.6 (48/76, p=0.015) |
| `nomic-qd-lib:shortlist+tournament vs tournament` | +2.8 (51/37, p=0.165) | -17.0 (14/99, p<0.001) | -1.6 (58/66, p=0.530) |
| `nomic-qd-lib:shortlist vs small:shortlist` | -1.0 (34/39, p=0.640) | -14.6 (19/92, p<0.001) | -11.0 (29/84, p<0.001) |
| `nomic-qd-lib:shortlist vs nomic-qd:shortlist` | -0.4 (31/33, p=0.901) | -14.2 (18/89, p<0.001) | -1.8 (47/56, p=0.431) |

## Column-centering (open-system-one's shortlist score), A MPS

Score = cos + (cos - mu), mu fitted on the unlabeled development-split texts (`laya_shortlist.py`: cosine + column-centered cosine), 600 development rows per dataset, labels not read. Plain → centered; paired delta in points (gained / lost rows, exact McNemar p). The plain top-20 lists match the main runs on every row.

| dataset | embedder | recall@20 | top-1 | shortlist → Laya | top-1 Δ | shortlist Δ |
|---|---|---:|---:|---:|---|---|
| banking77 | `bge-small/instruct/bare` | 98.2 → 98.6 | 66.2 → 66.4 | 63.0 → 62.8 | +0.2 (13/12, p=1.00) | -0.2 (15/16, p=1.00) |
| banking77 | `nomic/sym/bare` | 96.8 → 98.2 | 64.4 → 62.8 | 63.4 → 64.2 | -1.6 (7/15, p=0.13) | +0.8 (21/17, p=0.63) |
| banking77 | `minilm/none/bare` | 97.0 → 98.2 | 59.6 → 60.2 | 62.6 → 63.0 | +0.6 (17/14, p=0.72) | +0.4 (21/19, p=0.87) |
| clinc150 | `bge-small/instruct/bare` | 95.8 → 96.8 | 70.4 → 71.6 | 86.8 → 87.0 | +1.2 (13/7, p=0.26) | +0.2 (13/12, p=1.00) |
| clinc150 | `nomic/sym/bare` | 97.2 → 97.8 | 71.4 → 72.2 | 87.8 → 88.2 | +0.8 (11/7, p=0.48) | +0.4 (11/9, p=0.82) |
| clinc150 | `minilm/none/bare` | 96.4 → 96.6 | 70.6 → 71.2 | 86.6 → 87.0 | +0.6 (6/3, p=0.51) | +0.4 (7/5, p=0.77) |
| massive | `bge-small/instruct/bare` | 94.8 → 95.2 | 58.4 → 58.8 | 64.4 → 62.8 | +0.4 (15/13, p=0.85) | -1.6 (13/21, p=0.23) |
| massive | `nomic/sym/bare` | 88.4 → 90.4 | 50.8 → 50.8 | 60.2 → 60.2 | +0.0 (4/4, p=1.00) | +0.0 (17/17, p=1.00) |
| massive | `minilm/none/bare` | 92.4 → 93.4 | 51.6 → 51.6 | 60.6 → 61.2 | +0.0 (5/5, p=1.00) | +0.6 (22/19, p=0.76) |

## What the option budget kept (laya.common.build_head)

- banking77: one question at the default head {'options': 77, 'options_distinct': 73, 'tokens_per_option': 4}; `wide` {'options': 77, 'options_distinct': 77, 'tokens_per_option': None}; tournament groups [15, 15, 16, 15, 16, 'none cut']; shortlists cut on 0 (small) / 0 (nomic) of 500 rows
- clinc150: one question at the default head {'options': 150, 'options_distinct': 149, 'tokens_per_option': 4}; `wide` {'options': 150, 'options_distinct': 150, 'tokens_per_option': None}; tournament groups [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 'none cut']; shortlists cut on 0 (small) / 0 (nomic) of 500 rows
- massive: one question at the default head {'options': 60, 'options_distinct': 55, 'tokens_per_option': 4}; `wide` {'options': 60, 'options_distinct': 60, 'tokens_per_option': None}; tournament groups [15, 15, 15, 15, 'none cut']; shortlists cut on 0 (small) / 0 (nomic) of 500 rows
