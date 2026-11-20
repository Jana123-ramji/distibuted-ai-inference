"""CPU fallback backend - real inference with ONNX Runtime.

This is the development/testing path that works on any laptop (no GPU, no MAX,
Windows included). It is a *real* model doing *real* token-by-token generation:

    tokenizer.json          -> HuggingFace `tokenizers` (Rust) encoder/decoder
    onnx/model_int8.onnx    -> SmolLM2-135M-Instruct, int8 quantised (~137 MB)
    onnxruntime             -> CPU execution provider

The decode loop below is a normal greedy/top-k sampling loop with a KV cache:
run the model, take the logits of the last position, pick the next token, feed it
back together with the cached keys/values. Nothing is faked or pre-canned - if
the model files are missing, loading fails and the worker reports `unhealthy`.

The exact input/output names of an exported ONNX model differ between exports, so
they are *discovered* from the session at load time instead of hardcoded.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .inference_engine import InferenceEngine, InferenceError, InferenceResult

logger = logging.getLogger(__name__)

# SmolLM2-Instruct was trained with this ChatML-style prompt format.
_SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."
_CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
_STOP_TOKEN_NAMES = ("<|im_end|>", "<|endoftext|>")
# Preference order when several quantisations are present in the model folder.
_FILE_PREFERENCE = ("int8", "quantized", "uint8", "q4", "fp16", "model.onnx")


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


class FallbackEngine(InferenceEngine):
    """Small instruct model on the CPU through ONNX Runtime."""

    name = "fallback"

    def __init__(
        self,
        model_id: str,
        model_dir: str,
        max_new_tokens: int = 96,
        temperature: float | None = None,
        top_k: int | None = None,
        threads: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_dir = Path(model_dir)
        self.max_new_tokens = max_new_tokens
        # temperature 0 => greedy (deterministic). Default 0.7 gives natural text.
        self.temperature = float(os.getenv("TEMPERATURE", "0.7") if temperature is None else temperature)
        self.top_k = int(os.getenv("TOP_K", "40") if top_k is None else top_k)
        # Three workers share one laptop CPU, so keep the thread count small.
        self.threads = int(os.getenv("ORT_THREADS", "2") if threads is None else threads)

        self._session: ort.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None
        self._onnx_file: Path | None = None
        self._past_inputs: list[str] = []
        self._present_outputs: list[str] = []
        self._kv_shape: tuple[int, int] = (0, 0)  # (kv_heads, head_dim)
        self._kv_dtype = np.float32
        self._wants_position_ids = False
        self._wants_cache_branch = False
        self._stop_ids: set[int] = set()
        self._rng = np.random.default_rng()
        self._load_seconds = 0.0

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def _find_onnx_file(self) -> Path:
        """Pick the ONNX weights file, preferring small quantised variants."""
        explicit = os.getenv("MODEL_ONNX_FILE", "").strip()
        if explicit:
            candidate = self.model_dir / explicit
            if candidate.is_file():
                return candidate
            raise InferenceError(f"MODEL_ONNX_FILE={explicit} not found under {self.model_dir}")

        candidates = sorted(self.model_dir.glob("onnx/*.onnx")) + sorted(self.model_dir.glob("*.onnx"))
        if not candidates:
            raise InferenceError(
                f"No .onnx file in {self.model_dir.resolve()}. "
                "Run: python scripts/download_model.py"
            )
        for token in _FILE_PREFERENCE:
            for path in candidates:
                if token in path.name:
                    return path
        return candidates[0]

    def _kv_geometry(self, sample_input: ort.NodeArg) -> tuple[int, int]:
        """Work out (kv_heads, head_dim) for the empty initial KV cache.

        Static dimensions in the ONNX graph are used when present; otherwise the
        values come from config.json. Nothing is guessed silently.
        """
        shape = list(sample_input.shape)
        heads = shape[1] if len(shape) > 1 and isinstance(shape[1], int) else None
        head_dim = shape[3] if len(shape) > 3 and isinstance(shape[3], int) else None

        if heads is None or head_dim is None:
            config_path = self.model_dir / "config.json"
            if not config_path.is_file():
                raise InferenceError(
                    f"Cannot determine KV cache shape: {config_path} is missing "
                    "and the ONNX graph uses dynamic dimensions"
                )
            config = json.loads(config_path.read_text(encoding="utf-8"))
            attention_heads = int(config["num_attention_heads"])
            heads = heads or int(config.get("num_key_value_heads", attention_heads))
            head_dim = head_dim or int(config.get("head_dim", config["hidden_size"] // attention_heads))
        return int(heads), int(head_dim)

    def load_model(self) -> None:
        """Load tokenizer + ONNX session and run one warm-up token."""
        started = time.perf_counter()

        tokenizer_path = self.model_dir / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise InferenceError(
                f"{tokenizer_path.resolve()} is missing. Run: python scripts/download_model.py"
            )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

        self._onnx_file = self._find_onnx_file()
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3  # warnings and errors only

        logger.info(
            "[WORKER] Loading ONNX model %s (threads=%d)", self._onnx_file.name, self.threads
        )
        self._session = ort.InferenceSession(
            str(self._onnx_file), options, providers=["CPUExecutionProvider"]
        )

        # --- discover the graph interface instead of hardcoding names -----
        inputs = {i.name: i for i in self._session.get_inputs()}
        output_names = [o.name for o in self._session.get_outputs()]
        self._wants_position_ids = "position_ids" in inputs
        self._wants_cache_branch = "use_cache_branch" in inputs
        self._past_inputs = [name for name in inputs if "past" in name]

        if self._past_inputs:
            sample = inputs[self._past_inputs[0]]
            self._kv_shape = self._kv_geometry(sample)
            self._kv_dtype = np.float16 if "float16" in sample.type else np.float32
            # present.N.key <- past_key_values.N.key
            mapped = [name.replace("past_key_values", "present") for name in self._past_inputs]
            if all(name in output_names for name in mapped):
                self._present_outputs = mapped
            else:
                # Fall back to positional order (logits first, then the cache).
                self._present_outputs = output_names[1 : 1 + len(self._past_inputs)]

        for token in _STOP_TOKEN_NAMES:
            token_id = self._tokenizer.token_to_id(token)
            if token_id is not None:
                self._stop_ids.add(int(token_id))

        self._load_seconds = time.perf_counter() - started
        logger.info(
            "[WORKER] Model ready in %.1f s (file=%s, layers_cached=%d, kv_heads=%d, head_dim=%d)",
            self._load_seconds,
            self._onnx_file.name,
            len(self._past_inputs) // 2,
            self._kv_shape[0],
            self._kv_shape[1],
        )

        # Warm-up: the first run allocates buffers and is much slower, which
        # would otherwise show up as a fake "slow first request" in the stats.
        warmup_started = time.perf_counter()
        self.infer("Hello", max_new_tokens=1)
        logger.info("[WORKER] Warm-up run took %.0f ms", (time.perf_counter() - warmup_started) * 1000)

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def _empty_cache(self) -> dict[str, np.ndarray]:
        heads, head_dim = self._kv_shape
        empty = np.zeros((1, heads, 0, head_dim), dtype=self._kv_dtype)
        return {name: empty for name in self._past_inputs}

    def _sample_next(self, logits: np.ndarray) -> int:
        """Greedy when temperature == 0, otherwise top-k sampling."""
        if self.temperature <= 0:
            return int(np.argmax(logits))
        scaled = logits.astype(np.float32) / self.temperature
        if 0 < self.top_k < scaled.size:
            top_idx = np.argpartition(-scaled, self.top_k)[: self.top_k]
            probs = _softmax(scaled[top_idx])
            return int(self._rng.choice(top_idx, p=probs))
        probs = _softmax(scaled)
        return int(self._rng.choice(probs.size, p=probs))

    def infer(self, prompt: str, max_new_tokens: int | None = None) -> InferenceResult:
        if self._session is None or self._tokenizer is None:
            raise InferenceError("Fallback engine used before load_model()")

        limit = max_new_tokens or self.max_new_tokens
        formatted = _CHAT_TEMPLATE.format(system=_SYSTEM_PROMPT, user=prompt.strip())
        prompt_ids = self._tokenizer.encode(formatted).ids

        started = time.perf_counter()
        past = self._empty_cache()
        input_ids = np.array([prompt_ids], dtype=np.int64)
        past_length = 0
        generated: list[int] = []

        for step in range(limit):
            total_length = past_length + input_ids.shape[1]
            feeds: dict[str, np.ndarray] = {
                "input_ids": input_ids,
                "attention_mask": np.ones((1, total_length), dtype=np.int64),
            }
            if self._wants_position_ids:
                feeds["position_ids"] = np.arange(
                    past_length, total_length, dtype=np.int64
                ).reshape(1, -1)
            if self._wants_cache_branch:
                feeds["use_cache_branch"] = np.array([step > 0], dtype=bool)
            feeds.update(past)

            outputs = self._session.run(None, feeds)
            logits = outputs[0][0, -1, :]
            next_id = self._sample_next(logits)

            if next_id in self._stop_ids:
                break
            generated.append(next_id)

            # Reuse the returned keys/values as the cache for the next step.
            if self._past_inputs:
                output_names = [o.name for o in self._session.get_outputs()]
                past = {
                    past_name: outputs[output_names.index(present_name)]
                    for past_name, present_name in zip(self._past_inputs, self._present_outputs)
                }
                past_length = total_length
                input_ids = np.array([[next_id]], dtype=np.int64)
            else:
                # No KV cache in this export: re-feed the whole sequence.
                input_ids = np.array([prompt_ids + generated], dtype=np.int64)

        latency_ms = (time.perf_counter() - started) * 1000
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()

        return InferenceResult(
            text=text,
            backend=self.name,
            model_id=self.model_id,
            latency_ms=round(latency_ms, 1),
            input_tokens=len(prompt_ids),
            output_tokens=len(generated),
        )

    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        return self._session is not None and self._tokenizer is not None

    def describe(self) -> dict[str, str]:
        return {
            "backend": self.name,
            "model_id": self.model_id,
            "onnx_file": self._onnx_file.name if self._onnx_file else "not loaded",
            "runtime": f"onnxruntime {ort.__version__} (CPUExecutionProvider)",
            "threads": str(self.threads),
            "temperature": str(self.temperature),
            "load_seconds": f"{self._load_seconds:.1f}",
        }
