"""Independent, deterministic text and sequence metrics."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence


def conservative_normalize(text: str) -> str:
    """Apply NFC and newline normalization without reversing or rewriting Arabic text."""

    normalized = unicodedata.normalize("NFC", text)
    return normalized.replace("\r\n", "\n").replace("\r", "\n")


def edit_distance(reference: Sequence[object], hypothesis: Sequence[object]) -> int:
    """Return Levenshtein distance using O(min(n, m)) memory."""

    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for ref_item_index, ref_item in enumerate(reference, start=1):
        current = [ref_item_index]
        for hyp_item_index, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_item_index] + 1,
                    previous[hyp_item_index - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def _similarity(reference: Sequence[object], hypothesis: Sequence[object]) -> float:
    size = max(len(reference), len(hypothesis), 1)
    return 1.0 - edit_distance(reference, hypothesis) / size


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference_chars = list(conservative_normalize(reference))
    hypothesis_chars = list(conservative_normalize(hypothesis))
    return edit_distance(reference_chars, hypothesis_chars) / max(len(reference_chars), 1)


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = conservative_normalize(reference).split()
    hypothesis_words = conservative_normalize(hypothesis).split()
    return edit_distance(reference_words, hypothesis_words) / max(len(reference_words), 1)


def normalized_edit_similarity(reference: str, hypothesis: str) -> float:
    reference_text = list(conservative_normalize(reference))
    hypothesis_text = list(conservative_normalize(hypothesis))
    return _similarity(reference_text, hypothesis_text)


def _digits(text: str) -> list[str]:
    result: list[str] = []
    for character in conservative_normalize(text):
        if character.isdecimal():
            result.append(str(unicodedata.digit(character)))
    return result


def digit_exact_accuracy(reference: str, hypothesis: str) -> float:
    """Return 1 when the logical digit sequences match, including an empty pair."""

    return float(_digits(reference) == _digits(hypothesis))


def reading_order_sequence_score(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    """Score an ordered unit-ID sequence with normalized edit similarity."""

    return _similarity(list(reference), list(hypothesis))
