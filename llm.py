"""
Generative-AI layer (optional).

Supports Google Gemini (free tier via Google AI Studio) and Groq (free tier),
called through plain HTTPS so no extra SDKs are needed. If no key is configured
or a call fails, the app automatically falls back to deterministic templates,
so it ALWAYS works - even offline / without any API key.
"""
from __future__ import annotations

import json
import os
import re

import requests

GEMINI_MODELS = [m for m in [os.getenv("GEMINI_MODEL"), "gemini-flash-latest", "gemini-3.5-flash",
                             "gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-2.5-flash"] if m]
GROQ_MODELS = [m for m in [os.getenv("GROQ_MODEL"), "llama-3.3-70b-versatile", "llama-3.1-8b-instant"] if m]


class LLM:
    def __init__(self, provider: str | None = None, api_key: str | None = None, model: str | None = None):
        self.provider = (provider or "none").lower()
        self.api_key = (api_key or "").strip()
        self.model = model
        self.last_error = None
        self.used_model = None

    @property
    def enabled(self) -> bool:
        return self.provider in ("gemini", "groq") and bool(self.api_key)

    @property
    def label(self) -> str:
        if not self.enabled:
            return "Built-in rule engine (no AI key)"
        return f"{self.provider.title()} · {self.used_model or self.model or 'auto'}"

    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, system: str = "", json_mode: bool = False, temperature: float = 0.2) -> str | None:
        if not self.enabled:
            return None
        try:
            if self.provider == "gemini":
                return self._gemini(prompt, system, json_mode, temperature)
            return self._groq(prompt, system, json_mode, temperature)
        except Exception as e:  # noqa: BLE001
            self.last_error = str(e)[:300]
            return None

    def _gemini(self, prompt, system, json_mode, temperature):
        models = [self.model] if self.model else GEMINI_MODELS
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": 4096}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        err = None
        for m in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
            resp = requests.post(url, params={"key": self.api_key}, json=body, timeout=60)
            if resp.status_code in (404, 400) and "not found" in resp.text.lower():
                err = f"{m}: model not available"
                continue
            if resp.status_code != 200:
                err = f"{m}: HTTP {resp.status_code} {resp.text[:200]}"
                if resp.status_code in (401, 403):
                    break
                continue
            data = resp.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            if text:
                self.used_model = m
                return text
            err = f"{m}: empty response"
        self.last_error = err
        return None

    def _groq(self, prompt, system, json_mode, temperature):
        models = [self.model] if self.model else GROQ_MODELS
        err = None
        for m in models:
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
            body = {"model": m, "messages": msgs, "temperature": temperature, "max_tokens": 4096}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                 headers={"Authorization": f"Bearer {self.api_key}"}, json=body, timeout=60)
            if resp.status_code != 200:
                err = f"{m}: HTTP {resp.status_code} {resp.text[:200]}"
                if resp.status_code in (401, 403):
                    break
                continue
            self.used_model = m
            return resp.json()["choices"][0]["message"]["content"]
        self.last_error = err
        return None

    def generate_json(self, prompt: str, system: str = "") -> dict | None:
        txt = self.generate(prompt, system, json_mode=True, temperature=0.0)
        if not txt:
            return None
        txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", txt, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        self.last_error = "AI returned invalid JSON"
        return None


def test_connection(llm: LLM) -> tuple[bool, str]:
    out = llm.generate("Reply with exactly: OK")
    if out:
        return True, f"Connected to {llm.label}"
    return False, llm.last_error or "No response"
