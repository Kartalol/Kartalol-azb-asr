import numpy as np
import pytest

from kartalol_azb_asr.audio import prepare_audio_array


def test_prepare_audio_mixes_channels_and_resamples() -> None:
    stereo = np.vstack([np.ones(8), np.zeros(8)])
    result = prepare_audio_array(stereo, source_rate=8_000, target_rate=16_000)
    assert result.shape == (16,)
    assert result.dtype == np.float32
    assert result == pytest.approx(np.full(16, 0.5, dtype=np.float32))


def test_prepare_audio_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        prepare_audio_array(np.zeros((2, 2, 2)), 16_000)
