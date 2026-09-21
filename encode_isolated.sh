#!/bin/bash
# 一个编码器一个进程，RAM 列才可比。跑之前确认本机没有别的重活。
set -euo pipefail
cd "$(dirname "$0")"
for e in static minilm modernbert mbert-embed; do
  echo "=== $e ==="
  FORCE_ENCODE=1 ONLY="$e" TOKENIZERS_PARALLELISM=false .venv/bin/python run.py encode
done
