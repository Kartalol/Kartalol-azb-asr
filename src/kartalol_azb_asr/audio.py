"""Audio loading and model-input preparation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


def prepare_audio_array(
    samples: "np.ndarray", source_rate: int, target_rate: int = 16_000
) -> "np.ndarray":
    """Convert audio to mono float32 and resample with linear interpolation.

    Production file loading uses ``librosa``'s higher-quality resampler. This
    small dependency-free helper defines and tests channel/dtype/shape behavior.
    """

    import numpy as np

    array = np.asarray(samples)
    if array.ndim == 2:
        # Accept channels-first or samples-first; the shorter axis is channels.
        axis = 0 if array.shape[0] <= array.shape[1] else 1
        array = array.mean(axis=axis)
    if array.ndim != 1:
        raise ValueError("audio samples must be one- or two-dimensional")
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sampling rates must be positive")
    array = array.astype(np.float32, copy=False)
    if source_rate == target_rate or array.size == 0:
        return array
    output_length = max(1, round(array.size * target_rate / source_rate))
    source_positions = np.linspace(0.0, 1.0, array.size, endpoint=False)
    target_positions = np.linspace(0.0, 1.0, output_length, endpoint=False)
    return np.interp(target_positions, source_positions, array).astype(np.float32)


def load_audio(path: str | Path, target_rate: int = 16_000) -> tuple["np.ndarray", int]:
    """Load supported audio, convert to mono, and resample to ``target_rate``."""

    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension {audio_path.suffix!r}; expected "
            f"{', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
        )
    import numpy as np

    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - exercised in installed CLI
        raise RuntimeError("Audio loading requires librosa; install requirements.txt") from exc
    samples, _ = librosa.load(str(audio_path), sr=target_rate, mono=True)
    return np.asarray(samples, dtype=np.float32), target_rate


def processor_inputs(processor: Any, samples: "np.ndarray", sampling_rate: int) -> Any:
    """Create tensor inputs accepted by Whisper or CTC processors."""

    return processor(samples, sampling_rate=sampling_rate, return_tensors="pt")
