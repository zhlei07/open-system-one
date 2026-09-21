#!/usr/bin/env python3
"""从落盘 JSON 生成 README 的数字区（中英双语），写到 out/tables_{en,zh}.md。

为什么要脚本生成：手抄表格必然出错——本次实验里我就把 84.5% 抄成了 84.6%。
数字只有一个来源，就是 out/*.json。
"""
import glob, json
from pathlib import Path
import numpy as np

OUT = Path("out")
DS = ["sst2", "ag_news", "emotion", "banking77"]
L = lambda p: json.loads((OUT / p).read_text()) if (OUT / p).exists() else None


def cross(rich: bool):
    """交叉编码 base 的四个任务成绩。rich=True 取丰富描述版。"""
    r = {}
    for ds in DS:
        pats = [f"out/xc_base_{ds}_k*_t1_n2500_rich.json"] if rich else \
               [f"out/cross_{ds}_k*.json", f"out/xc_base_{ds}_k*_t1_n2500.json"]
        hits = [h for p in pats for h in glob.glob(p) if ("_rich" in h) == rich]
        if hits:
            d = json.load(open(hits[0]))
            r[ds] = (d.get("acc_argmax", d.get("acc_mean_template")), d["lat_p50_ms"])
    return r


def macro(d):
    v = [x for x in d.values() if x is not None]
    return float(np.mean(v)) if len(v) == len(DS) else None


jev_b, jev_r = L("jev.json"), L("jev_enriched.json")
bi = L("fair_bi.json")
heads = L("heads.json") or {}
xb, xr = cross(False), cross(True)

rows = []
def acc(d, key=None):
    return {ds: (d[ds][0] if key is None else d[ds][key]) for ds in DS if ds in d}

sets = {
 "jev_bare":   {ds: jev_b["per_dataset"][ds]["acc"] for ds in DS} if jev_b else {},
 "jev_rich":   {ds: jev_r["per_dataset"][ds]["acc"] for ds in DS} if jev_r else {},
 "cross_bare": acc(xb), "cross_rich": acc(xr),
 "bi_bare":    {ds: bi[ds]["bare"] for ds in DS} if bi else {},
 "bi_rich":    {ds: bi[ds]["rich"] for ds in DS} if bi else {},
 "bi_few":     {ds: heads["mbert-embed"]["per_dataset"][ds]["fly_few"] for ds in DS} if heads else {},
}

def fmt(v):
    return f"{v*100:.1f} %" if v is not None else "—"

def table(lang):
    zh = lang == "zh"
    hdr = ["方案", "sst2 (2)", "ag_news (4)", "emotion (6)", "banking77 (77)", "宏平均"] if zh else \
          ["stack", "sst2 (2)", "ag_news (4)", "emotion (6)", "banking77 (77)", "macro"]
    names = {
      "jev_bare":   ("typesafe/jev — 裸标签名", "typesafe/jev — bare label names"),
      "jev_rich":   ("typesafe/jev — 丰富描述", "typesafe/jev — enriched descriptions"),
      "cross_bare": ("交叉编码 base — 裸标签名", "Cross-encoder base — bare label names"),
      "cross_rich": ("交叉编码 base — 丰富描述", "Cross-encoder base — enriched descriptions"),
      "bi_bare":    ("双塔 + Fly — 裸标签名", "Bi-encoder + head — bare label names"),
      "bi_rich":    ("双塔 + Fly — 丰富描述", "Bi-encoder + head — enriched descriptions"),
    }
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for k in ["jev_bare", "jev_rich", "cross_bare", "cross_rich", "bi_bare", "bi_rich"]:
        d = sets[k]
        if not d:
            continue
        m = macro(d)
        cells = [fmt(d.get(ds)) for ds in DS] + [f"**{fmt(m)}**" if m else "—"]
        out.append(f"| {names[k][0 if zh else 1]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


for lang, f in [("en", "tables_en.md"), ("zh", "tables_zh.md")]:
    body = ["## Zero-shot, all stacks, both description conditions\n" if lang == "en"
            else "## 零样本：全部方案 × 两种描述条件\n", table(lang), ""]
    d = sets["bi_few"]
    if d:
        body += [("### With 2,000 labels (not comparable to Jev)\n" if lang == "en"
                  else "### 给 2000 条标注（不能与 Jev 相比）\n"),
                 "| stack | " + " | ".join(DS) + " | macro |" if lang == "en"
                 else "| 方案 | " + " | ".join(DS) + " | 宏平均 |",
                 "|" + "---|" * (len(DS) + 2),
                 ("| ModernBERT-embed + head | " if lang == "en" else "| ModernBERT-embed + Fly | ")
                 + " | ".join(fmt(d[ds]) for ds in DS) + f" | **{fmt(macro(d))}** |", ""]
    (OUT / f).write_text("\n".join(body))
    print(f"--- {f} ---"); print("\n".join(body))
