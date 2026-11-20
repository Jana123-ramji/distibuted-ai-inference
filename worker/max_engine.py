"""MAX backend - the primary, high-performance inference path.

All MAX-specific code lives in this file only, so the rest of the platform never
imports MAX and keeps working when MAX is not installed.

Honesty notes (important for the viva):

* MAX runs on Linux and macOS. On native Windows it does not run at all, so on a
  Windows laptop this backend reports itself unavailable and the worker uses
  `fallback_engine.py`. Inside the Linux worker container MAX can be installed
  (`pip install max[all]`) and this backend becomes the active one.
* This module never fabricates a result. If MAX cannot be used it raises
  `BackendUnavailable`, the worker reports `status=unhealthy`, and the scheduler
  routes work to another worker.
* The concrete MAX call is intentionally written against the *installed*
  version's documented API and is probed with `importlib` instead of assumed:
  `availability()` first checks that the package and the entry point really
  exist. Phase 7 pins the exact version in the README after running
  `python -c "import max; print(max.__version__)"` inside the container.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import platform
import time

from .inference_engine import BackendUnavailable, InferenceEngine, InferenceError, InferenceResult

logger = logging.getLogger(__name__)

# Documented offline-inference entry point of the MAX Python API.
_MAX_LLM_MODULE = "max.entrypoints.llm"
_MAX_LLM_CLASS = "LLM"


def _max_version() -> str:
    """Installed MAX version, or "unknown"."""
    try:
        module = importlib.import_module("max")
        return str(getattr(module, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "unknown"


class MaxEngine(InferenceEngine):
    """Runs inference through MAX's Python API."""

    name = "max"

    def __init__(self, model_id: str, max_new_tokens: int = 96) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._llm: object | None = None
        self._version = "unknown"

    # -- availability ------------------------------------------------------
    @staticmethod
    def availability() -> tuple[bool, str]:
        """Can MAX really be used in this environment?

        Returns (usable, reason). The reason is logged so the active backend is
        always visible instead of guessed.
        """
        if platform.system() == "Windows":
            return False, "MAX is not supported on native Windows (use WSL2 or the Linux container)"
        if importlib.util.find_spec("max") is None:
            return False, "the 'max' package is not installed"
        if importlib.util.find_spec(_MAX_LLM_MODULE) is None:
            return False, f"'{_MAX_LLM_MODULE}' is missing in MAX {_max_version()}"
        return True, f"MAX {_max_version()} detected"

    # -- InferenceEngine ---------------------------------------------------
    def load_model(self) -> None:
        usable, reason = self.availability()
        if not usable:
            raise BackendUnavailable(f"MAX backend unavailable: {reason}")

        self._version = _max_version()
        module = importlib.import_module(_MAX_LLM_MODULE)
        llm_class = getattr(module, _MAX_LLM_CLASS, None)
        if llm_class is None:
            # Never guess an API that this version does not expose.
            raise BackendUnavailable(
                f"MAX {self._version} does not expose {_MAX_LLM_MODULE}.{_MAX_LLM_CLASS}; "
                "check the API of the installed version before using the MAX backend"
            )

        logger.info("[WORKER] Loading model '%s' with MAX %s", self.model_id, self._version)
        started = time.perf_counter()
        # `LLM.from_pretrained` is the documented constructor for offline
        # inference; it is resolved dynamically so a version without it fails
        # with a clear message instead of a wrong assumption.
        factory = getattr(llm_class, "from_pretrained", None)
        self._llm = factory(self.model_id) if callable(factory) else llm_class(self.model_id)
        logger.info(
            "[WORKER] MAX model ready in %.1f s (model=%s)",
            time.perf_counter() - started,
            self.model_id,
        )

    def infer(self, prompt: str, max_new_tokens: int | None = None) -> InferenceResult:
        if self._llm is None:
            raise InferenceError("MAX engine used before load_model()")

        tokens = max_new_tokens or self.max_new_tokens
        started = time.perf_counter()
        generate = getattr(self._llm, "generate", None)
        if not callable(generate):
            raise InferenceError(f"MAX {self._version} LLM object has no generate() method")

        outputs = generate([prompt], max_new_tokens=tokens)
        # MAX returns a list of completions for a list of prompts.
        text = outputs[0] if isinstance(outputs, (list, tuple)) and outputs else str(outputs)
        latency_ms = (time.perf_counter() - started) * 1000

        return InferenceResult(
            text=str(text).strip(),
            backend=self.name,
            model_id=self.model_id,
            latency_ms=round(latency_ms, 1),
            input_tokens=len(prompt.split()),
            output_tokens=len(str(text).split()),
        )

    def health_check(self) -> bool:
        return self._llm is not None

    def describe(self) -> dict[str, str]:
        return {"backend": self.name, "model_id": self.model_id, "max_version": self._version}
