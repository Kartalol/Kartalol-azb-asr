"""Dataset registry and conversion to the benchmark's internal schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_id: str
    role: str
    description: str


DATASETS: dict[str, DatasetSpec] = {
    "community": DatasetSpec(
        "community",
        "yoyo-research-group/south-azerbaijani-asr",
        "community",
        "Community-collected South Azerbaijani speech",
    ),
    "external": DatasetSpec(
        "external",
        "Kartal-Ol/azb-asr-corpus",
        "external",
        "External/paper corpus index; select its published external split/config",
    ),
    "goldset": DatasetSpec(
        "goldset",
        "Kartal-Ol/AZB-ASR-Gold-Testset",
        "goldset",
        "Held-out AZB ASR GoldSet benchmark",
    ),
    "bhosai": DatasetSpec(
        "bhosai",
        "BHOSAI/PseudoLabelled_Azerbaijani_Voices",
        "external",
        "Pseudo-labelled Azerbaijani voices",
    ),
}

_AUDIO_FIELDS = ("audio", "path", "audio_filepath", "file")
_TEXT_FIELDS = ("text", "sentence", "transcription", "azb")
_SPEAKER_FIELDS = ("speaker_id", "client_id", "person_id", "speaker")
_DURATION_FIELDS = ("duration", "duration_seconds", "length")


def _first(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def standardize_record(
    row: Mapping[str, Any],
    *,
    audio_field: str | None = None,
    text_field: str | None = None,
) -> dict[str, Any]:
    """Convert a source row to ``audio/text/speaker_id/duration`` fields."""

    audio = row.get(audio_field) if audio_field else _first(row, _AUDIO_FIELDS)
    text = row.get(text_field) if text_field else _first(row, _TEXT_FIELDS)
    if audio is None:
        raise ValueError(f"No audio field found; tried: {', '.join(_AUDIO_FIELDS)}")
    if text is None:
        raise ValueError(f"No text field found; tried: {', '.join(_TEXT_FIELDS)}")
    result: dict[str, Any] = {"audio": audio, "text": str(text)}
    speaker = _first(row, _SPEAKER_FIELDS)
    duration = _first(row, _DURATION_FIELDS)
    if speaker is not None:
        result["speaker_id"] = str(speaker)
    if duration is not None:
        result["duration"] = float(duration)
    return result


def resolve_dataset(name_or_id: str) -> DatasetSpec:
    """Resolve a public alias or preserve a custom Hugging Face dataset ID."""

    key = name_or_id.lower()
    return DATASETS.get(
        key,
        DatasetSpec(name_or_id, name_or_id, "custom", "User-supplied dataset"),
    )


def ensure_training_role(spec: DatasetSpec) -> None:
    """Prevent accidental GoldSet leakage into model training."""

    if spec.role == "goldset" or "gold-test" in spec.dataset_id.lower():
        raise ValueError("GoldSet is evaluation-only and cannot be used for training")


def load_hf_dataset(
    name_or_id: str,
    *,
    split: str,
    config_name: str | None = None,
    for_training: bool = False,
    audio_field: str | None = None,
    text_field: str | None = None,
) -> Any:
    """Load a Hugging Face split and standardize its public schema."""

    spec = resolve_dataset(name_or_id)
    if for_training:
        ensure_training_role(spec)
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Dataset loading requires the datasets package") from exc
    dataset = load_dataset(spec.dataset_id, config_name, split=split)
    original_columns = dataset.column_names
    standardized = dataset.map(
        lambda row: standardize_record(
            row, audio_field=audio_field, text_field=text_field
        ),
        remove_columns=original_columns,
        desc=f"Standardizing {spec.name}",
    )
    return standardized.cast_column("audio", Audio(sampling_rate=16_000))
