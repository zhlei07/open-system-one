## Zero-shot, all stacks, both description conditions

| stack | sst2 (2) | ag_news (4) | emotion (6) | banking77 (77) | macro |
|---|---|---|---|---|---|
| typesafe/jev — bare label names | 89.8 % | 88.3 % | 58.6 % | 78.4 % | **78.8 %** |
| typesafe/jev — enriched descriptions | 91.6 % | 88.6 % | 59.0 % | 77.8 % | **79.3 %** |
| Cross-encoder base — bare label names | 85.5 % | 84.4 % | 76.2 % | 68.8 % | **78.7 %** |
| Cross-encoder base — enriched descriptions | 84.5 % | 84.6 % | 71.7 % | 56.9 % | **74.4 %** |
| Bi-encoder + head — bare label names | 81.1 % | 66.3 % | 43.6 % | 63.2 % | **63.6 %** |
| Bi-encoder + head — enriched descriptions | 82.8 % | 80.0 % | 50.8 % | 65.8 % | **69.9 %** |

### With 2,000 labels (not comparable to Jev)

| stack | sst2 | ag_news | emotion | banking77 | macro |
|---|---|---|---|---|---|
| ModernBERT-embed + head | 85.8 % | 89.5 % | 66.2 % | 88.8 % | **82.6 %** |
