# src/services/llm.py
"""
DeepSeek LLM Client for Generator, Judge, and Controller.
Features:
- Configurable models (deepseek-chat, deepseek-flash)
- Strict temperature=0.0
- Comprehensive token & latency accounting
- Robust exponential backoff retries
"""

import os
import time
import json
import httpx
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

class LLMService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format_json: bool = False,
        max_tokens: int = 1024,
        max_retries: int = 3
    ) -> Tuple[str, Dict[str, int], float]:
        """
        Returns: (content_text, usage_dict, latency_ms)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}

        t0 = time.time()
        for attempt in range(max_retries):
            try:
                resp = self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                if resp.status_code == 200:
                    dur_ms = (time.time() - t0) * 1000.0
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
                    return content, usage, dur_ms
                elif resp.status_code == 429:
                    time.sleep(2.0 * (attempt + 1))
                else:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"LLM call returned {resp.status_code}: {resp.text}")
                    time.sleep(1.0)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {e}")
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError("LLM call exhausted all retries.")

    def close(self):
        self.client.close()
