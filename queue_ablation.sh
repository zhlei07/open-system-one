#!/bin/bash
# 消融队列：先等全量基线跑完，再依次跑模板集成 / large 骨干。
# 消融用 n=1000（±1.5pp 足够选方案），选定配置再全量 2500 确认。
set -u
cd "$(dirname "$0")"
while pgrep -f "cross_full.py" > /dev/null; do sleep 20; done
echo "=== 基线跑完，开始消融 $(date +%H:%M) ==="
P=.venv/bin/python
# 1. 模板集成单独的贡献（同 base 骨干，只加模板）—— 最便宜、信息量最大
$P cross_v2.py --model base  --ds sst2     --n 1000 --templates 3
# 2. large + 模板集成，先在最便宜的任务上看叠加效果
$P cross_v2.py --model large --ds sst2     --n 1000 --templates 3
# 3. ag_news：缺口 -5.1pp，两条一起上
$P cross_v2.py --model large --ds ag_news  --n 1000 --templates 3
# 4. banking77：缺口 -15.7pp，large + 裁到 top-20，先不叠模板（太贵）
$P cross_v2.py --model large --ds banking77 --n 1000 --templates 1 --topk 20
# 5. emotion 已经赢 16pp，最后补一个模板集成的数据点
$P cross_v2.py --model base  --ds emotion  --n 1000 --templates 3
echo "=== 消融完成 $(date +%H:%M) ==="
