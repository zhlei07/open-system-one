# Laya #102 protocol, run on our machines

The same-row comparison proposed on [NandhaKishorM/laya#102](https://github.com/NandhaKishorM/laya/issues/102):
500 fixed rows per official test split (BANKING77, CLINC150 in-scope, MASSIVE en-US), top-20 retrieval,
tournament groups of at most 16, latency per request including query encoding and every decision round.
The rows are `benchmark.py`'s seed-0 sample from Laya's `research/benchmarks/tournament`, i.e. the rows of
#950's 500-row table, and the question is #950's (state `{"message": text}`, labels with spaces, no
descriptions). Embedders were chosen on development rows only (`selection.json`).

**Read [`REPORT.md`](REPORT.md).**

## Files

| file | what |
|---|---|
| `selection.json` | development-row screening of BGE-small, MiniLM and Nomic (prefixes, query text) and the choice |
| `<dataset>_<machine>_<device>.json` | one run: accuracy, Wilson intervals, latency, paired tests, errors, option budget, thermal pressure |
| `<dataset>_<machine>_<device>.rows.jsonl` | the same run row by row: every arm's choice, correctness, latency, candidates (no dataset text) |
| `<dataset>_a_mps_extra.*` | embedder arms only, for MiniLM and Nomic Q/D, each with the bare message and with `laya.shortlist`'s query text |
| `colcenter.json` | the same shortlists scored with our column-centering (cosine + centered cosine) |
| `paths_<dataset>_a_mps.*` | the shortlist paths Laya documents, through Laya 0.4.6's own functions: Laya's own encoder (`embed_fn_from_agent`), BGE-small and MiniLM, each with the documented query, the JSON state and the message alone; per row, whether the gold label was kept and whether the answer was right (no dataset text) |
| `manifest.json` | the 500 rows per dataset: position in the loaded test rows, sha256 of the text, gold label (no text) |
| `REPORT.md` | all of the above as tables |

Machines: A and B, two Apple-silicon laptops. `mps` = torch on the Apple GPU, `cpu` = torch on the CPU.
Runs: A/MPS on all three datasets, A/CPU on BANKING77, B/CPU on all three, B/MPS on BANKING77.
Laya 0.4.4, torch 2.14.1; model and dataset revisions are recorded in each run's `.json`. The `paths_*` runs
(A/MPS) use Laya 0.4.6 at `c73908d` with the same checkpoint `55cf4c4`, and record Laya's commit and checkpoint.

## Reproduce

```bash
python laya102_bench.py select --device mps --out out/laya102/selection.json
python laya102_bench.py manifest --out out/laya102/manifest.json
python laya102_bench.py run --dataset banking77 --device mps --warm 5 \
    --selection out/laya102/selection.json --out out/laya102/banking77_a_mps
python laya102_bench.py run --dataset banking77 --device mps --warm 5 --embedder-arms-only \
    --roles minilm=minilm/none/bare,minilm-lib=minilm/none/library,nomic-qd=nomic/asym/bare,nomic-qd-lib=nomic/asym/library \
    --selection out/laya102/selection.json --out out/laya102/banking77_a_mps_extra
python laya102_colcenter.py
PYTHONPATH=<laya checkout at c73908d> python laya102_paths.py banking77 out/laya102/paths_banking77_a_mps
python laya102_report.py out/laya102 <laya>/research/benchmarks/tournament/results.json > out/laya102/REPORT.md
```

#950's published numbers are read from Laya's own `research/benchmarks/tournament/results.json`
(we used the file at NandhaKishorM/laya `682ac67`, release 0.4.4); it is not copied here.
