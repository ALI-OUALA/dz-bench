"""Independent, deterministic text and sequence metrics."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from .models import Block, BoundingBox, DocumentExtraction, TableStructure


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


def confidence_brier_score(outcomes: Sequence[int], confidences: Sequence[float]) -> float:
    if len(outcomes) != len(confidences) or not outcomes:
        raise ValueError("outcomes and confidences must have the same non-zero length")
    squared_error = sum(
        (confidence - outcome) ** 2 for outcome, confidence in zip(outcomes, confidences)
    )
    return squared_error / len(outcomes)


def expected_calibration_error(
    outcomes: Sequence[int], confidences: Sequence[float], *, bins: int = 10
) -> float:
    if len(outcomes) != len(confidences) or not outcomes:
        raise ValueError("outcomes and confidences must have the same non-zero length")
    if bins <= 0:
        raise ValueError("bins must be positive")
    total = len(outcomes)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, confidence in enumerate(confidences)
            if (lower <= confidence <= upper if index == bins - 1 else lower <= confidence < upper)
        ]
        if members:
            accuracy = sum(outcomes[position] for position in members) / len(members)
            mean_confidence = sum(confidences[position] for position in members) / len(members)
            error += len(members) / total * abs(accuracy - mean_confidence)
    return error


def error_detection_auroc(errors: Sequence[bool], uncertainty: Sequence[float]) -> float:
    """Return pairwise AUROC; 1 means every error is ranked more uncertain."""

    if len(errors) != len(uncertainty) or not errors:
        raise ValueError("errors and uncertainty must have the same non-zero length")
    positives = [score for error, score in zip(errors, uncertainty) if error]
    negatives = [score for error, score in zip(errors, uncertainty) if not error]
    if not positives or not negatives:
        raise ValueError("AUROC requires at least one error and one correct prediction")
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def bounding_box_iou(first: BoundingBox, second: BoundingBox) -> float:
    """Return intersection over union for canonical top-left pixel boxes."""

    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union else 0.0


@dataclass(frozen=True, slots=True)
class StructuredFieldScores:
    precision: float
    recall: float
    f1: float
    exact_accuracy: float
    financial_accuracy: float
    coordinate_iou: float
    hallucination_rate: float
    reference_count: int
    hypothesis_count: int
    financial_count: int


_FINANCIAL_FIELD_PARTS = ("ht", "tva", "ttc", "quantity", "unit_price", "line_total")


def structured_field_scores(
    reference: DocumentExtraction, hypothesis: DocumentExtraction
) -> StructuredFieldScores:
    """Align repeated fields by name and occurrence, then score exact normalized values."""

    reference_by_name: dict[str, list] = {}
    hypothesis_by_name: dict[str, list] = {}
    for field in reference.fields:
        reference_by_name.setdefault(field.field_name, []).append(field)
    for field in hypothesis.fields:
        hypothesis_by_name.setdefault(field.field_name, []).append(field)
    exact = 0
    financial_exact = 0
    financial_count = 0
    coordinate_scores: list[float] = []
    hallucinated = 0
    for name in set(reference_by_name) | set(hypothesis_by_name):
        expected = reference_by_name.get(name, [])
        predicted = hypothesis_by_name.get(name, [])
        hallucinated += max(0, len(predicted) - len(expected))
        for index, reference_field in enumerate(expected):
            is_financial = any(part in name for part in _FINANCIAL_FIELD_PARTS)
            financial_count += int(is_financial)
            if index >= len(predicted):
                continue
            hypothesis_field = predicted[index]
            matches = reference_field.normalized_value == hypothesis_field.normalized_value
            exact += int(matches)
            financial_exact += int(matches and is_financial)
            if matches and reference_field.bbox is not None and hypothesis_field.bbox is not None:
                coordinate_scores.append(
                    bounding_box_iou(reference_field.bbox, hypothesis_field.bbox)
                )
    precision = exact / max(len(hypothesis.fields), 1)
    recall = exact / max(len(reference.fields), 1)
    return StructuredFieldScores(
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / max(precision + recall, 1e-12),
        exact_accuracy=recall,
        financial_accuracy=financial_exact / max(financial_count, 1),
        coordinate_iou=sum(coordinate_scores) / max(len(coordinate_scores), 1),
        hallucination_rate=hallucinated / max(len(hypothesis.fields), 1),
        reference_count=len(reference.fields),
        hypothesis_count=len(hypothesis.fields),
        financial_count=financial_count,
    )


@dataclass(frozen=True, slots=True)
class BlockMatch:
    reference: Block
    hypothesis: Block | None
    iou: float


def greedy_block_matches(
    reference: Sequence[Block], hypothesis: Sequence[Block], minimum_iou: float = 0.1
) -> list[BlockMatch]:
    """Match each reference block to at most one spatially overlapping prediction."""

    if not 0 <= minimum_iou <= 1:
        raise ValueError("minimum_iou must be between 0 and 1")
    remaining = set(range(len(hypothesis)))
    matches: list[BlockMatch] = []
    for reference_block in reference:
        if not remaining:
            matches.append(BlockMatch(reference_block, None, 0.0))
            continue
        best_index = max(
            remaining,
            key=lambda index: bounding_box_iou(reference_block.bbox, hypothesis[index].bbox),
        )
        best_iou = bounding_box_iou(reference_block.bbox, hypothesis[best_index].bbox)
        if best_iou < minimum_iou:
            matches.append(BlockMatch(reference_block, None, 0.0))
            continue
        remaining.remove(best_index)
        matches.append(BlockMatch(reference_block, hypothesis[best_index], best_iou))
    return matches


@dataclass(frozen=True, slots=True)
class LayoutScores:
    block_count: int
    correct_type: int
    bbox_iou_sum: float
    matched_count: int


def layout_scores(reference: Sequence[Block], hypothesis: Sequence[Block]) -> LayoutScores:
    matches = greedy_block_matches(reference, hypothesis)
    return LayoutScores(
        block_count=max(len(reference), len(hypothesis)),
        correct_type=sum(
            match.hypothesis is not None
            and match.reference.block_type == match.hypothesis.block_type
            for match in matches
        ),
        bbox_iou_sum=sum(match.iou for match in matches),
        matched_count=sum(match.hypothesis is not None for match in matches),
    )


def _block_text(block: Block) -> str:
    if block.equation_text is not None:
        return conservative_normalize(block.equation_text)
    return conservative_normalize("\n".join(line.normalized_text for line in block.lines))


def equation_text_page_score(
    reference: Sequence[Block], hypothesis: Sequence[Block]
) -> tuple[float | None, int]:
    reference_equations = [block for block in reference if block.block_type == "equation"]
    if not reference_equations:
        return None, 0
    hypothesis_equations = [block for block in hypothesis if block.block_type == "equation"]
    matches = greedy_block_matches(reference_equations, hypothesis_equations)
    total = sum(
        normalized_edit_similarity(_block_text(match.reference), _block_text(match.hypothesis))
        if match.hypothesis is not None
        else 0.0
        for match in matches
    )
    return total / len(reference_equations), len(reference_equations)


def table_structure_similarity(
    reference: TableStructure, hypothesis: TableStructure | None
) -> float:
    """Score dimensions, merged-cell geometry, and cell adjacency without using cell text."""

    if hypothesis is None:
        return 0.0
    reference_cells = {
        (cell.row_index, cell.column_index, cell.row_span, cell.column_span)
        for cell in reference.cells
    }
    hypothesis_cells = {
        (cell.row_index, cell.column_index, cell.row_span, cell.column_span)
        for cell in hypothesis.cells
    }
    cell_f1 = (
        2
        * len(reference_cells & hypothesis_cells)
        / max(len(reference_cells) + len(hypothesis_cells), 1)
    )
    dimensions = float(
        reference.rows == hypothesis.rows and reference.columns == hypothesis.columns
    )
    return (dimensions + cell_f1) / 2


def table_structure_page_score(
    reference: Sequence[Block], hypothesis: Sequence[Block]
) -> tuple[float | None, int]:
    reference_tables = [block for block in reference if block.block_type == "table" and block.table]
    if not reference_tables:
        return None, 0
    hypothesis_tables = [block for block in hypothesis if block.block_type == "table"]
    matches = greedy_block_matches(reference_tables, hypothesis_tables)
    total = 0.0
    for match in matches:
        reference_table = match.reference.table
        if (
            reference_table is not None
            and match.hypothesis is not None
            and match.hypothesis.table is not None
        ):
            total += table_structure_similarity(reference_table, match.hypothesis.table)
    return total / len(reference_tables), len(reference_tables)
