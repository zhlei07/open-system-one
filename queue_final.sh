#!/bin/bash
# 收敛确认：统一配置 = ModernBERT-large-zeroshot + **单个默认模板** + banking77 双塔裁 top-20
#
# 为什么用单模板而不是 3 模板集成：n=1000 的消融显示两者宏平均几乎相同
#   sst2    单模板 88.9%  vs  3模板 89.9%
#   ag_news 单模板 86.5%  vs  3模板 85.1%
# 而单模板便宜 3 倍。用的是 TEMPLATES[0]（所有 base 跑法一直在用的那个默认值），
# 不是回头挑出来的最优模板——回头挑就要用目标任务的标签，零样本的前提就破了。
set -u
cd "$(dirname "$0")"
P=.venv/bin/python
$P cross_v2.py --model large --ds sst2      --n 2500 --templates 1            #  ~7 min
$P cross_v2.py --model large --ds emotion   --n 2500 --templates 1            # ~25 min
$P cross_v2.py --model large --ds ag_news   --n 2500 --templates 1            # ~39 min
$P cross_v2.py --model large --ds banking77 --n 2500 --templates 1 --topk 20  # ~78 min
echo "=== 收敛确认完成 $(date +%H:%M) ==="
