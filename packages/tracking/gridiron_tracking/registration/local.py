"""Downloaded vision models on Apple Silicon. Inference never calls a hosted API.

Download once with ``python scripts/setup_local_vision.py``. Keeping the model in
the project's ignored data directory avoids duplicating multi-GB weights in a
second cache. Loading only cached files also makes missing setup fail immediately.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock

from ..config import load_env

DEFAULT_MODEL = "mlx-community/Qwen3.5-9B-4bit"
MODEL_FILES = ["*.safetensors", "*.json", "*.txt", "*.jinja"]
_lock = Lock()


def cache_dir() -> Path:
    return Path(os.environ.get("FOOTBALL_MODEL_CACHE", "data/models/hub"))


def existing_model_path(model_id: str) -> Path | None:
    path = Path(model_id).expanduser()
    if path.is_dir():
        return path
    if directory := os.environ.get("FOOTBALL_LOCAL_MODELS_DIR"):
        path = Path(directory).expanduser() / model_id.rsplit("/", 1)[-1]
        if (path / "config.json").is_file() and any(path.glob("*.safetensors")):
            return path
    return None


@lru_cache(maxsize=1)
def _load(model_id: str):
    load_env()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "install the local reader: python -m pip install -e './packages/tracking[local-vision]'"
        ) from e

    path = existing_model_path(model_id)
    if path is None:
        try:
            path = Path(snapshot_download(model_id, cache_dir=cache_dir(),
                                          allow_patterns=MODEL_FILES, local_files_only=True))
        except FileNotFoundError as e:
            raise RuntimeError(
                f"{model_id} is not downloaded; run python scripts/setup_local_vision.py "
                f"--model {model_id}"
            ) from e
    try:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
    except ImportError as e:
        raise RuntimeError(
            "install the local reader: python -m pip install -e './packages/tracking[local-vision]'"
        ) from e
    model, processor = load(str(path), trust_remote_code=False)
    import mlx.core as mx
    mx.set_cache_limit(256 * 1024**2)
    return model, processor, load_config(str(path))


@lru_cache(maxsize=1)
def _constraint_tokenizer(tokenizer):
    from llguidance.hf import from_tokenizer

    return from_tokenizer(tokenizer)


def _json_processor(processor, schema: dict):
    """Restrict each generated token to the schema, rather than repairing guesses."""
    import mlx.core as mx
    import numpy as np
    import llguidance as llg
    import llguidance.numpy as llgnp

    tokenizer = _constraint_tokenizer(processor.tokenizer)
    grammar = llg.JsonCompiler(separators=(",", ":"), whitespace_pattern="").compile(json.dumps(schema))
    matcher = llg.LLMatcher(tokenizer, grammar)
    mask = llgnp.allocate_token_bitmask(1, tokenizer.vocab_size)
    started = False

    def constrain(tokens, logits):
        nonlocal started
        if started:
            matcher.consume_token(int(tokens[-1].item()))
            if error := matcher.get_error():
                raise RuntimeError(f"local JSON decoder: {error}")
        started = True
        llgnp.fill_next_token_bitmask(matcher, mask)
        bits = np.unpackbits(mask.view(np.uint8), bitorder="little")[:logits.shape[-1]]
        allowed = np.zeros(logits.shape[-1], dtype=bool)
        allowed[:len(bits)] = bits
        return mx.where(mx.array(allowed), logits, -float("inf"))

    return constrain


def generate_json(img, system: str, prompt: str, schema: dict, *,
                  model: str = DEFAULT_MODEL, max_tokens: int = 4000) -> tuple[str, int]:
    # The API can schedule two clips concurrently. A single shared model keeps
    # their weights from filling memory twice; the lock also protects its state.
    with _lock:
        loaded, processor, config = _load(model)
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from PIL import Image

        image = Image.fromarray(img[:, :, ::-1])  # OpenCV BGR -> RGB
        if image.width > 1600:
            image = image.resize((1600, round(image.height * 1600 / image.width)))
        asked = (f"{system}\n\n{prompt}\n\nReturn only a JSON object matching this schema. "
                 f"No Markdown or explanation.\n{json.dumps(schema, separators=(',', ':'))}")
        formatted = apply_chat_template(processor, config, asked, num_images=1,
                                        enable_thinking=False)
        result = generate(loaded, processor, formatted, image=[image],
                          max_tokens=max_tokens, temperature=0.0, verbose=False,
                          prefill_step_size=512, repetition_penalty=1.1,
                          logits_processors=[_json_processor(processor, schema)])
        return result.text, result.generation_tokens
