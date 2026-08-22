"""Dependency-free WER, CER, and deletion/insertion ratio calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import inf
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ErrorCounts:
    """Levenshtein alignment counts for a sequence collection."""

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_length: int = 0

    @property
    def error_rate(self) -> float:
        return (
            (self.substitutions + self.deletions + self.insertions)
            / self.reference_length
            if self.reference_length
            else 0.0
        )

    @property
    def deletion_insertion_ratio(self) -> float:
        """Return D/I; zero errors give 0 and deletion-only errors give infinity."""

        if self.insertions:
            return self.deletions / self.insertions
        return inf if self.deletions else 0.0

    def __add__(self, other: "ErrorCounts") -> "ErrorCounts":
        return ErrorCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.reference_length + other.reference_length,
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def alignment_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> ErrorCounts:
    """Return deterministic minimum-edit alignment counts for two sequences."""

    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    costs: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)
    ]
    for i in range(1, rows):
        costs[i][0] = (i, 0, i, 0)
    for j in range(1, cols):
        costs[0][j] = (j, 0, 0, j)
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                continue
            diagonal = costs[i - 1][j - 1]
            delete = costs[i - 1][j]
            insert = costs[i][j - 1]
            candidates = (
                (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            )
            # Stable tie-break: substitution, deletion, insertion.
            costs[i][j] = min(candidates, key=lambda item: item[0])
    _, substitutions, deletions, insertions = costs[-1][-1]
    return ErrorCounts(substitutions, deletions, insertions, len(reference))


def compute_asr_metrics(
    references: Iterable[str], predictions: Iterable[str]
) -> dict[str, float | int]:
    """Compute corpus WER/CER percentages and raw word-level DIR."""

    refs = list(references)
    preds = list(predictions)
    if len(refs) != len(preds):
        raise ValueError("references and predictions must have equal length")
    word_counts = ErrorCounts()
    char_counts = ErrorCounts()
    for reference, prediction in zip(refs, preds):
        word_counts += alignment_counts(reference.split(), prediction.split())
        char_counts += alignment_counts(list(reference), list(prediction))
    return {
        "wer": word_counts.error_rate * 100.0,
        "cer": char_counts.error_rate * 100.0,
        "dir": word_counts.deletion_insertion_ratio,
        "utterances": len(refs),
        "word_substitutions": word_counts.substitutions,
        "word_deletions": word_counts.deletions,
        "word_insertions": word_counts.insertions,
        "reference_words": word_counts.reference_length,
        "reference_characters": char_counts.reference_length,
    }
