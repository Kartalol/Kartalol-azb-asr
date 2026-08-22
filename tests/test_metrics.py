import math

import pytest

from kartalol_azb_asr.metrics import alignment_counts, compute_asr_metrics


def test_metric_percentages_and_raw_dir() -> None:
    metrics = compute_asr_metrics(["a b c d"], ["a c d y"])
    assert metrics["wer"] == pytest.approx(50.0)
    assert metrics["dir"] == pytest.approx(1.0)
    assert metrics["word_substitutions"] == 0
    assert metrics["word_deletions"] == 1
    assert metrics["word_insertions"] == 1


def test_deletion_only_dir_is_infinite() -> None:
    counts = alignment_counts("a b".split(), "a".split())
    assert math.isinf(counts.deletion_insertion_ratio)
