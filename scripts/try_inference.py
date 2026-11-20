"""Run one prompt through the inference engine without any HTTP or Docker.

Useful to prove the model really works (and to measure CPU latency) before the
distributed parts are involved:

    python scripts/try_inference.py "Explain artificial intelligence in simple words"
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Allow running this file directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker.config import get_worker_settings  # noqa: E402
from worker.inference_engine import create_engine  # noqa: E402
from worker.preprocess import Preprocessor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s")


def main() -> int:
    prompts = sys.argv[1:] or [
        "Explain artificial intelligence in simple words",
        "What is cloud computing?",
    ]
    settings = get_worker_settings()

    engine = create_engine(
        backend=settings.inference_backend,
        model_id=settings.model_id,
        model_dir=settings.model_dir,
        max_new_tokens=settings.max_new_tokens,
    )
    preprocessor = Preprocessor(settings.mojo_enabled, settings.mojo_binary)

    engine.load_model()
    print(f"\nengine: {engine.describe()}")
    print(f"mojo binary used: {preprocessor.mojo_available()}\n")

    for prompt in prompts:
        wall = time.perf_counter()
        pre = preprocessor.run(prompt)
        result = engine.infer(pre.cleaned)
        print("=" * 78)
        print(f"PROMPT : {prompt}")
        print(f"RESULT : {result.text}")
        print(
            f"STATS  : tokens_in={result.input_tokens} tokens_out={result.output_tokens} "
            f"inference={result.latency_ms:.0f} ms preprocess={pre.duration_ms:.2f} ms "
            f"(mojo={pre.mojo_used}) total={(time.perf_counter() - wall) * 1000:.0f} ms"
        )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
