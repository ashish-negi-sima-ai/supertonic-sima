"""Hybrid Supertonic 3 runtime for SiMa Modalix."""

from .audio import save_wav, wav_bytes
from .text import AVAILABLE_LANGUAGES, AVAILABLE_VOICES, MAX_SPEED, MIN_SPEED


def __getattr__(name: str):
    # HTTP/browser tools can be exercised without loading the Modalix runtime.
    if name in {"SynthesisResult", "SupertonicModalix", "benchmark_summary"}:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = (
    "AVAILABLE_LANGUAGES",
    "AVAILABLE_VOICES",
    "MAX_SPEED",
    "MIN_SPEED",
    "SynthesisResult",
    "SupertonicModalix",
    "benchmark_summary",
    "save_wav",
    "wav_bytes",
)
