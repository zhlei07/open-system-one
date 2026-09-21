#!/bin/bash
# 真瓦数测量。powermetrics 要 root，所以这个脚本得你自己跑：
#   ! cd ~/projects/fly/jev_repro && sudo ./power_probe.sh .venv/bin/python run.py --only minilm
# 输出 out/power_<label>.txt，run.py 会在汇总时自动读取。
set -euo pipefail
LABEL="${LABEL:-run}"
mkdir -p out
powermetrics --samplers cpu_power,gpu_power -i 500 > "out/power_${LABEL}.txt" 2>/dev/null &
PM=$!
trap 'kill $PM 2>/dev/null || true' EXIT
"$@"
sleep 1
kill $PM 2>/dev/null || true
wait $PM 2>/dev/null || true
awk '/Combined Power/ {gsub(/[^0-9.]/,"",$NF); s+=$NF; n++} END {if(n) printf("平均整机功耗 %.1f mW（%d 个采样点）\n", s/n, n)}' "out/power_${LABEL}.txt"
