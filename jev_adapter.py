"""Jev 适配器：通过 Cloudflare Workers AI 的 typesafe/jev 打真模型。

请求体形状来自 CF 官方文档：state + questions{name: {type, instructions, criteria}}。
我们用 type="choice"，criteria 就是选项描述——**和另外三条链路给的选项文字一字不差**，
否则比的是提示词不是模型。

状态：账号凭证有效，但 AI Gateway 余额为 0（HTTP 402
"Insufficient balance; add money to your gateway or use BYOK"）。
充值或填 TYPESAFE_API_KEY 之后这一列就能补上，其余代码一行不用改。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

ACCOUNT = os.environ.get("CF_ACCOUNT_ID")
URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run"


class JevClient:
    def __init__(self):
        self.token = os.environ.get("CLOUDFLARE_API_TOKEN")
        self.email = os.environ.get("CLOUDFLARE_EMAIL")
        self.key = os.environ.get("CLOUDFLARE_API_KEY")
        self.last_usage = None
        self.total_input_tokens = 0

    def _headers(self) -> dict:
        if not ACCOUNT:
            raise RuntimeError("请设置 CF_ACCOUNT_ID（Cloudflare 账号 ID，见 dash 地址栏）")
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        elif self.email and self.key:
            h["X-Auth-Email"], h["X-Auth-Key"] = self.email, self.key
        else:
            raise RuntimeError("没有 CF 凭证：设 CLOUDFLARE_API_TOKEN，或 EMAIL+API_KEY")
        return h

    def available(self) -> tuple[bool, str]:
        try:
            self.predict("ping", ["a", "b"])
            return True, "ok"
        except Exception as e:
            return False, str(e)[:200]

    def predict(self, text: str, options: list[str], retries: int = 3,
                criteria: dict[str, str] | None = None) -> tuple[int, list[float]]:
        body = json.dumps({
            "model": "typesafe/jev",
            "input": {
                "state": text,
                "questions": {"label": {
                    "type": "choice",
                    "instructions": "Which option best describes this text?",
                    # criteria 为空时退回裸标签名；公平性对比时传入丰富描述
                    "criteria": criteria or {o: o for o in options},
                }},
            },
        }).encode()
        for attempt in range(retries):
            req = urllib.request.Request(URL, data=body, headers=self._headers())
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    payload = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode()[:300]
                if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {detail}") from None
        if not payload.get("success", True):
            raise RuntimeError(json.dumps(payload.get("errors"))[:300])
        # CF 把模型响应又套了一层：result.result.answers（异步任务信封）。
        # 两种形状都兼容，免得哪天网关改了封装就全盘跑不动。
        res = payload["result"]
        if "answers" not in res and isinstance(res.get("result"), dict):
            res = res["result"]
        ans = res["answers"]["label"]
        self.last_usage = res.get("usage")
        self.total_input_tokens += (self.last_usage or {}).get("input_tokens", 0)
        probs = ans.get("probabilities", {})
        return options.index(ans["choice"]), [float(probs.get(o, 0.0)) for o in options]
