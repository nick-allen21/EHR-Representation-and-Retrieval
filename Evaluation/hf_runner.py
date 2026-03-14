"""Local HuggingFace model inference with disk-based response caching.

Same external interface as LLMRunner (generate / generate_batch / stats) so
run_evaluation.py works without modification — just swap the runner class.

GPU batching strategy
---------------------
Unlike the async OpenAI runner, HF inference is synchronous and GPU-bound.
generate_batch() checks the cache for all items first, then processes all
uncached items in batches of `batch_size` through the model.  Results are
cached after each batch so partial runs are recoverable.

Supported models (tested tokenizers; GPU runs pending)
------------------------------------------------------
- meta-llama/Llama-3.1-8B-Instruct       (gated — requires HF_TOKEN)
- mistralai/Mistral-7B-Instruct-v0.3     (open)
- microsoft/Phi-3-mini-4k-instruct       (open, 4k context)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

log = logging.getLogger(__name__)

_ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
_DEFAULT_CACHE_DIR = Path("data/results/cache")
_DEFAULT_BATCH_SIZE = 4
_DEFAULT_MAX_NEW_TOKENS = 512


def _cache_key(model_id: str, messages: list[dict]) -> str:
    blob = json.dumps({"model": model_id, "messages": messages}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class HFRunner:
    """Local HuggingFace causal-LM runner with response caching.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier, e.g. 'meta-llama/Meta-Llama-3-8B-Instruct'.
    cache_dir:
        Directory for JSON response cache (same format as LLMRunner).
    batch_size:
        Number of prompts to process per GPU forward pass.
    device_map:
        Passed to ``from_pretrained`` — 'auto' distributes across all GPUs.
    torch_dtype:
        Torch dtype string; 'float16' is recommended for 7-8B models on L40S.
    """

    def __init__(
        self,
        model_id: str,
        cache_dir: str | Path = _DEFAULT_CACHE_DIR,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        device_map: str = "auto",
        torch_dtype: str = "float16",
    ):
        load_dotenv(_ROOT_ENV)
        self.model_id = model_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self._stats = {"cache_hits": 0, "api_calls": 0, "errors": 0}

        # Same attribute name as llmrunner so callers can log runner.model
        self.model = model_id

        hf_token = os.environ.get("HF_TOKEN")
        log.info(
            "Loading model %s (device_map=%s, dtype=%s) …",
            model_id, device_map, torch_dtype,
        )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            padding_side="left",  # required for decode only batch generation
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self._hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=dtype,
            token=hf_token,
        )
        self._hf_model.eval()

        self._max_model_len = getattr(
            self._hf_model.config, "max_position_embeddings", 4096
        )
        log.info(
            "Model loaded: %s (max_position_embeddings=%d)",
            model_id, self._max_model_len,
        )

    # ── Core batch inference (synchronous) 

    def _batch_generate(
        self,
        messages_batch: list[list[dict]],
        temperature: float,
        max_new_tokens: int,
    ) -> list[tuple[str, int, int]]:
        """Run one GPU forward pass and return (answer, prompt_tokens, new_tokens)."""
        import torch

        texts = [
            self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            for msgs in messages_batch
        ]

        max_input = self._max_model_len - max_new_tokens
        orig_trunc_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "left"
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input,
        ).to(self._hf_model.device)
        self.tokenizer.truncation_side = orig_trunc_side

        if inputs["input_ids"].shape[1] >= max_input:
            log.warning(
                "Input truncated to %d tokens (model max=%d, reserved %d for generation)",
                max_input, self._max_model_len, max_new_tokens,
            )

        input_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self._hf_model.generate(**inputs, **gen_kwargs)

        results = []
        for i, output in enumerate(outputs):
            new_tokens = output[input_len:]
            answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            n_prompt = int(inputs["attention_mask"][i].sum().item())
            n_new = int(
                (new_tokens != self.tokenizer.pad_token_id).sum().item()
            )
            results.append((answer, n_prompt, n_new))

        return results

    # ── Public interface (mirrors LLMRunner) ─────────────────────────────────

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    ) -> dict:
        """Generate a response for a single (system, user) prompt pair."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        key = _cache_key(self.model_id, messages)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            self._stats["cache_hits"] += 1
            return json.loads(cache_file.read_text())

        try:
            answer, prompt_tokens, completion_tokens = self._batch_generate(
                [messages], temperature, max_tokens
            )[0]
            result = {
                "answer": answer,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached": False,
            }
            cache_file.write_text(json.dumps(result, indent=2))
            self._stats["api_calls"] += 1
            return result
        except Exception as exc:
            log.error("HF inference error: %s", exc)
            self._stats["errors"] += 1
            return {
                "answer": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached": False,
                "error": str(exc),
            }

    async def generate_batch(
        self,
        items: list[dict],
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    ) -> list[dict]:
        """Batch-process items, checking cache first and batching GPU work.

        Each item must have a ``user_prompt`` key.  Returns results in the same
        order as *items*, with the original item fields merged into each result.
        """
        results: list[dict | None] = [None] * len(items)
        uncached: list[tuple[int, dict, list[dict], str]] = []

        # --- Cache check pass ---
        for i, item in enumerate(items):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["user_prompt"]},
            ]
            key = _cache_key(self.model_id, messages)
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                results[i] = {**item, **json.loads(cache_file.read_text())}
                self._stats["cache_hits"] += 1
            else:
                uncached.append((i, item, messages, key))

        if uncached:
            log.info(
                "%s: %d cached, %d to generate (batch_size=%d)",
                self.model_id,
                len(items) - len(uncached),
                len(uncached),
                self.batch_size,
            )

        # GPU inference in sub-batches 
        for batch_start in range(0, len(uncached), self.batch_size):
            batch = uncached[batch_start : batch_start + self.batch_size]
            msgs_batch = [msgs for _, _, msgs, _ in batch]

            try:
                outputs = self._batch_generate(msgs_batch, temperature, max_tokens)
            except Exception as exc:
                log.error(
                    "Batch %d–%d failed: %s",
                    batch_start + 1,
                    batch_start + len(batch),
                    exc,
                )
                outputs = [("", 0, 0)] * len(batch)
                self._stats["errors"] += len(batch)

            for (i, item, msgs, key), (answer, ptok, ctok) in zip(batch, outputs):
                result = {
                    "answer": answer,
                    "prompt_tokens": ptok,
                    "completion_tokens": ctok,
                    "cached": False,
                }
                cache_file = self.cache_dir / f"{key}.json"
                cache_file.write_text(json.dumps(result, indent=2))
                results[i] = {**item, **result}
                self._stats["api_calls"] += 1

            log.info(
                "  batch %d–%d complete",
                batch_start + 1,
                batch_start + len(batch),
            )

        return results  # type: ignore[return-value]

    #  Stats 

    @property
    def stats(self) -> dict:
        return dict(self._stats)
