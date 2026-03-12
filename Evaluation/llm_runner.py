"""Async LLM inference with disk-based response caching.

Wraps the OpenAI API with:
  - Deterministic disk cache (hash of model + messages -> JSON file)
  - Configurable concurrency via asyncio.Semaphore
  - Exponential-backoff retry on rate limits and server errors

The cache ensures identical prompts are never billed twice, which is critical
when iterating on scoring / analysis without re-running inference.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APIStatusError

log = logging.getLogger(__name__)

_ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
_DEFAULT_MODEL = "o4-mini"
_DEFAULT_CACHE_DIR = Path("data/results/cache")
_MAX_RETRIES = 5
_BASE_BACKOFF = 2.0
_MAX_BACKOFF = 60.0
_DEFAULT_CONCURRENCY = 10


def _cache_key(model: str, messages: list[dict]) -> str:
    blob = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class LLMRunner:
    """Async OpenAI chat-completion runner with response caching."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        cache_dir: str | Path = _DEFAULT_CACHE_DIR,
        concurrency: int = _DEFAULT_CONCURRENCY,
    ):
        load_dotenv(_ROOT_ENV)
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = AsyncOpenAI()
        self.semaphore = asyncio.Semaphore(concurrency)
        self._stats = {"cache_hits": 0, "api_calls": 0, "errors": 0}

    # ── Single call ───────────────────────────────────────────────────────────

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict:
        """Send a prompt and return the response (with caching).

        Returns
        -------
        dict with keys:
            answer            str
            prompt_tokens     int
            completion_tokens int
            cached            bool
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        key = _cache_key(self.model, messages)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            self._stats["cache_hits"] += 1
            return json.loads(cache_file.read_text())

        # Reasoning models (o1, o3, o4) need max_completion_tokens
        # (which includes internal reasoning tokens) and don't
        # support the temperature parameter.
        _is_reasoning = any(
            self.model.startswith(p) for p in ("o1", "o3", "o4")
        )
        create_kwargs: dict = {"model": self.model, "messages": messages}
        if _is_reasoning:
            create_kwargs["max_completion_tokens"] = max_tokens
        else:
            create_kwargs["max_tokens"] = max_tokens
            create_kwargs["temperature"] = temperature

        async with self.semaphore:
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    resp = await self.client.chat.completions.create(
                        **create_kwargs,
                    )
                    raw = resp.choices[0].message.content or ""
                    result = {
                        "answer": raw.strip(),
                        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                        "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                        "cached": False,
                    }
                    cache_file.write_text(json.dumps(result, indent=2))
                    self._stats["api_calls"] += 1
                    return result

                except RateLimitError as exc:
                    wait = getattr(exc, "retry_after", None)
                    if wait is None:
                        wait = min(_BASE_BACKOFF * (2 ** (attempt - 1)), _MAX_BACKOFF)
                    log.warning(
                        "Rate limited, retry in %.1fs (attempt %d/%d)",
                        wait, attempt, _MAX_RETRIES,
                    )
                    await asyncio.sleep(wait)

                except APIStatusError as exc:
                    if exc.status_code >= 500:
                        wait = min(_BASE_BACKOFF * (2 ** (attempt - 1)), _MAX_BACKOFF)
                        log.warning(
                            "Server error %d, retry in %.1fs (attempt %d/%d)",
                            exc.status_code, wait, attempt, _MAX_RETRIES,
                        )
                        await asyncio.sleep(wait)
                    else:
                        log.error("API error %d: %s", exc.status_code, exc.message)
                        self._stats["errors"] += 1
                        return {
                            "answer": "",
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "cached": False,
                            "error": str(exc),
                        }

        log.error("All %d retries exhausted", _MAX_RETRIES)
        self._stats["errors"] += 1
        return {
            "answer": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached": False,
            "error": "max_retries_exhausted",
        }

    # ── Batch call ────────────────────────────────────────────────────────────

    async def generate_batch(
        self,
        items: list[dict],
        system_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> list[dict]:
        """Run generate() concurrently for a list of items.

        Each item must have a ``user_prompt`` key.  Returns results in the same
        order as *items*, with the original item fields merged into each result.
        """
        async def _one(item: dict) -> dict:
            result = await self.generate(
                system_prompt=system_prompt,
                user_prompt=item["user_prompt"],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return {**item, **result}

        return await asyncio.gather(*[_one(it) for it in items])

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return dict(self._stats)
