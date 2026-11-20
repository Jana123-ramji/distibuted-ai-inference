"""Prompt preprocessing - runs in Mojo when available, in Python otherwise.

Why preprocess at all? Before the model sees a prompt we clean it and compute a
small feature vector (character histogram + statistics). This is a tight numeric
loop over every character, which is exactly the kind of CPU-bound work Mojo is
good at and CPython is slow at.

The contract between Python and Mojo is deliberately trivial so a student can
follow it:

    stdin :  the raw prompt text
    stdout:  one JSON object
             {"cleaned": str, "chars": int, "words": int,
              "avg_word_len": float, "histogram": [32 floats], "checksum": int}

`mojo/preprocess.mojo` (Phase 8) implements exactly this contract, and
`_python_preprocess` below is the reference implementation used when the compiled
Mojo binary is not present (for example on native Windows, where Mojo does not
run). Both produce the same JSON, so the platform never depends on Mojo being
available - it just gets faster when it is.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Number of buckets in the character histogram feature vector.
HISTOGRAM_BINS = 32
# The Mojo binary is a tiny CPU task; if it takes longer than this something is
# wrong and we fall back to Python instead of blocking the request.
MOJO_TIMEOUT_SECONDS = 5.0
# Prompts longer than this are truncated before inference.
MAX_PROMPT_CHARS = 4000


@dataclass
class PreprocessResult:
    """Result of the preprocessing step."""

    cleaned: str
    chars: int
    words: int
    avg_word_len: float
    histogram: list[float] = field(default_factory=list)
    checksum: int = 0
    mojo_used: bool = False
    duration_ms: float = 0.0


def _python_preprocess(prompt: str) -> PreprocessResult:
    """Reference implementation (pure Python) of the Mojo preprocessing."""
    # 1. clean: drop control characters, collapse whitespace, cap the length
    cleaned_chars = [c for c in prompt if unicodedata.category(c)[0] != "C" or c in "\n\t"]
    cleaned = " ".join("".join(cleaned_chars).split())[:MAX_PROMPT_CHARS]

    # 2. statistics
    words = cleaned.split()
    avg_word_len = (sum(len(w) for w in words) / len(words)) if words else 0.0

    # 3. character histogram (the CPU-bound part Mojo accelerates)
    counts = [0] * HISTOGRAM_BINS
    checksum = 0
    for char in cleaned:
        code = ord(char)
        counts[code % HISTOGRAM_BINS] += 1
        checksum = (checksum * 31 + code) % 1_000_000_007
    total = len(cleaned) or 1
    histogram = [round(c / total, 6) for c in counts]

    return PreprocessResult(
        cleaned=cleaned,
        chars=len(cleaned),
        words=len(words),
        avg_word_len=round(avg_word_len, 3),
        histogram=histogram,
        checksum=checksum,
        mojo_used=False,
    )


def _mojo_preprocess(prompt: str, binary: Path) -> PreprocessResult:
    """Run the compiled Mojo binary and parse its JSON output."""
    completed = subprocess.run(
        [str(binary)],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=MOJO_TIMEOUT_SECONDS,
        check=True,
    )
    payload = json.loads(completed.stdout)
    return PreprocessResult(
        cleaned=payload["cleaned"],
        chars=int(payload["chars"]),
        words=int(payload["words"]),
        avg_word_len=float(payload["avg_word_len"]),
        histogram=[float(x) for x in payload.get("histogram", [])],
        checksum=int(payload.get("checksum", 0)),
        mojo_used=True,
    )


class Preprocessor:
    """Chooses the Mojo binary when it exists, Python otherwise."""

    def __init__(self, mojo_enabled: bool, mojo_binary: str) -> None:
        self.mojo_enabled = mojo_enabled
        self.mojo_binary = Path(mojo_binary)
        # Remember a failure so we log it once instead of on every request.
        self._mojo_failed = False

    def mojo_available(self) -> bool:
        """True when the compiled Mojo binary can be used."""
        if not self.mojo_enabled or self._mojo_failed:
            return False
        return self.mojo_binary.is_file()

    def run(self, prompt: str) -> PreprocessResult:
        """Preprocess a prompt, preferring Mojo, never failing the request."""
        started = time.perf_counter()

        if self.mojo_available():
            try:
                result = _mojo_preprocess(prompt, self.mojo_binary)
                result.duration_ms = round((time.perf_counter() - started) * 1000, 3)
                return result
            except Exception as exc:  # noqa: BLE001 - degrade instead of failing
                self._mojo_failed = True
                logger.warning(
                    "[WORKER] Mojo preprocessing failed (%s: %s); using the Python "
                    "reference implementation from now on",
                    type(exc).__name__,
                    exc,
                )

        result = _python_preprocess(prompt)
        result.duration_ms = round((time.perf_counter() - started) * 1000, 3)
        return result
