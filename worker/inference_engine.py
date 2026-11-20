"""The inference abstraction every worker uses.

    Worker  ->  InferenceEngine  ->  MAX backend        (primary)
                               \\->  ONNX CPU backend   (development fallback)

Both backends implement the same three methods, so the worker code does not
care which one is active:

    engine.load_model()
    engine.infer(prompt)
    engine.health_check()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    """Raised when a backend cannot produce a result."""


class BackendUnavailable(InferenceError):
    """Raised when a backend cannot run in this environment at all."""


@dataclass
class InferenceResult:
    """What a backend returns for one prompt."""

    text: str
    backend: str
    model_id: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


class InferenceEngine(ABC):
    """Common interface for all inference backends."""

    #: short backend name reported to the API ("max" or "fallback")
    name: str = "abstract"
    #: identifier of the loaded model
    model_id: str = "unknown"

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory. Raises on failure."""

    @abstractmethod
    def infer(self, prompt: str, max_new_tokens: int | None = None) -> InferenceResult:
        """Run real inference for `prompt`."""

    @abstractmethod
    def health_check(self) -> bool:
        """True when the engine is loaded and able to serve requests."""

    def describe(self) -> dict[str, str]:
        """Small human-readable description for GET /info."""
        return {"backend": self.name, "model_id": self.model_id}


def create_engine(
    backend: str,
    model_id: str,
    model_dir: str,
    max_new_tokens: int,
) -> InferenceEngine:
    """Pick the backend according to `INFERENCE_BACKEND`.

    * ``max``      - force MAX; fail loudly if MAX is not usable here
    * ``fallback`` - force the ONNX CPU backend
    * ``auto``     - use MAX when it is really available, otherwise ONNX CPU
    """
    from .fallback_engine import FallbackEngine  # imported lazily: heavy deps
    from .max_engine import MaxEngine

    choice = backend.strip().lower()

    if choice == "max":
        logger.info("[WORKER] INFERENCE_BACKEND=max -> using the MAX backend")
        return MaxEngine(model_id=model_id, max_new_tokens=max_new_tokens)

    if choice == "fallback":
        logger.info("[WORKER] INFERENCE_BACKEND=fallback -> using the ONNX CPU backend")
        return FallbackEngine(model_id=model_id, model_dir=model_dir, max_new_tokens=max_new_tokens)

    # auto
    available, reason = MaxEngine.availability()
    if available:
        logger.info("[WORKER] INFERENCE_BACKEND=auto -> MAX is available, using MAX")
        return MaxEngine(model_id=model_id, max_new_tokens=max_new_tokens)

    logger.info("[WORKER] INFERENCE_BACKEND=auto -> MAX unavailable (%s), using the ONNX CPU backend", reason)
    return FallbackEngine(model_id=model_id, model_dir=model_dir, max_new_tokens=max_new_tokens)
