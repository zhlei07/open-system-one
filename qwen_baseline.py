"""Qwen2.5-0.5B-Instruct 基线：图里最右边那条"直接让小 LLM 出答案"的链路。

这条链路和另外三条**信息量完全对等**：拿到的是同一段文本、同一组选项描述，
没有任何带标签样本。所以它的对照对象是 fly-zeroshot，不是 fly-fewshot。

自由生成 + 解析，而不是"取受限 logits"。因为受限解码要枚举 77 个选项的 logprob，
那是 77 次前向——现实里没人这么部署 0.5B。解析失败按答错计，失败率单独报：
**把解析失败悄悄丢掉是这类评测最常见的作弊方式**，会让小模型凭空涨好几个点。
"""
from __future__ import annotations

import re

import numpy as np

REPO = "Qwen/Qwen2.5-0.5B-Instruct"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


class QwenClassifier:
    def __init__(self, device: str = "cpu", max_new_tokens: int = 12):
        self.device, self.max_new_tokens = device, max_new_tokens
        self.parse_fail = 0

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(REPO)
        self.model = AutoModelForCausalLM.from_pretrained(
            REPO, dtype=torch.float32, low_cpu_mem_usage=True).to(self.device).eval()
        torch.set_grad_enabled(False)
        return self

    def _prompt(self, text: str, options: list[str]) -> str:
        opts = "\n".join(f"- {o}" for o in options)
        msgs = [
            {"role": "system",
             "content": "You are a classifier. Reply with exactly one option from the list, verbatim. No explanation."},
            {"role": "user", "content": f"Options:\n{opts}\n\nText: {text}\n\nOption:"},
        ]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def predict(self, text: str, options: list[str]) -> int:
        ids = self.tok(self._prompt(text, options), return_tensors="pt").to(self.device)
        out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens,
                                  do_sample=False, pad_token_id=self.tok.eos_token_id)
        gen = self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return self._parse(gen, options)

    def _parse(self, gen: str, options: list[str]) -> int:
        g = _norm(gen)
        norms = [_norm(o) for o in options]
        for i, o in enumerate(norms):          # 完全匹配
            if g == o:
                return i
        hits = [i for i, o in enumerate(norms) if o and o in g]
        if len(hits) == 1:
            return hits[0]
        if hits:                                # 多个命中取最长的那个（更具体）
            return max(hits, key=lambda i: len(norms[i]))
        best, best_score = -1, 0.0              # 退而求其次：词重叠
        gw = set(g.split())
        for i, o in enumerate(norms):
            ow = set(o.split())
            if ow:
                s = len(gw & ow) / len(ow)
                if s > best_score:
                    best, best_score = i, s
        if best_score > 0:
            return best
        self.parse_fail += 1
        return -1                               # 解析失败 -> 必然算错
