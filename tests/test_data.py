import pytest

from kartalol_azb_asr.data import ensure_training_role, resolve_dataset, standardize_record


def test_standardize_manifest_schema() -> None:
    result = standardize_record(
        {
            "audio_filepath": "voice.ogg",
            "transcription": "سالام",
            "person_id": 7,
            "duration": "2.5",
        }
    )
    assert result == {
        "audio": "voice.ogg",
        "text": "سالام",
        "speaker_id": "7",
        "duration": 2.5,
    }


def test_goldset_cannot_train() -> None:
    with pytest.raises(ValueError, match="evaluation-only"):
        ensure_training_role(resolve_dataset("goldset"))
