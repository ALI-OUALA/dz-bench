import pytest

from dz_bench.metrics import (
    bounding_box_iou,
    character_error_rate,
    conservative_normalize,
    digit_exact_accuracy,
    normalized_edit_similarity,
    reading_order_sequence_score,
    word_error_rate,
)


def test_hand_computed_text_metrics() -> None:
    assert character_error_rate("abc", "adc") == 1 / 3
    assert word_error_rate("one two three", "one three") == 1 / 3
    assert normalized_edit_similarity("abc", "adc") == pytest.approx(2 / 3)


def test_normalization_preserves_logical_arabic_order() -> None:
    text = "سَلام\r\nعليكم"
    assert conservative_normalize(text) == "سَلام\nعليكم"
    assert conservative_normalize("\u202eabc") == "\u202eabc"


def test_digits_and_reading_order() -> None:
    assert digit_exact_accuracy("المجموع ١٢٣", "Total 123") == 1.0
    assert digit_exact_accuracy("الرقم ١٢٣", "الرقم ١٣٢") == 0.0
    assert reading_order_sequence_score(["a", "b", "c"], ["a", "c"]) == pytest.approx(2 / 3)


def test_bounding_box_iou_is_exact_for_identical_and_disjoint_boxes() -> None:
    from dz_bench.models import BoundingBox

    box = BoundingBox(x=0, y=0, width=10, height=10)
    assert bounding_box_iou(box, box) == 1.0
    assert bounding_box_iou(box, BoundingBox(x=20, y=20, width=5, height=5)) == 0.0


def test_greedy_block_matches_tie_breaks_deterministically() -> None:
    from dz_bench.metrics import greedy_block_matches
    from dz_bench.models import Block, BoundingBox, Confidence, Provenance

    reference = Block(
        block_id="ref",
        block_type="paragraph",
        bbox=BoundingBox(x=0, y=0, width=10, height=10),
        reading_order_index=1,
        lines=[],
        provenance=Provenance(kind="human_annotation", source="annotator"),
        confidence=Confidence(score=1.0),
    )
    # Both hypotheses have exactly the same IoU (1.0) with the reference.
    hypothesis1 = Block(
        block_id="hyp1",
        block_type="paragraph",
        bbox=BoundingBox(x=0, y=0, width=10, height=10),
        reading_order_index=1,
        lines=[],
        provenance=Provenance(kind="human_annotation", source="annotator"),
        confidence=Confidence(score=1.0),
    )
    hypothesis2 = Block(
        block_id="hyp2",
        block_type="paragraph",
        bbox=BoundingBox(x=0, y=0, width=10, height=10),
        reading_order_index=2,
        lines=[],
        provenance=Provenance(kind="human_annotation", source="annotator"),
        confidence=Confidence(score=1.0),
    )

    matches = greedy_block_matches([reference], [hypothesis1, hypothesis2])
    assert len(matches) == 1
    # Because of the -index tie breaker, it should always pick the earliest one (hypothesis1).
    assert matches[0].hypothesis is not None
    assert matches[0].hypothesis.block_id == "hyp1"

    # Swap the order in the list, it should still pick the first one in the new list (hypothesis2).
    matches_swapped = greedy_block_matches([reference], [hypothesis2, hypothesis1])
    assert len(matches_swapped) == 1
    assert matches_swapped[0].hypothesis is not None
    assert matches_swapped[0].hypothesis.block_id == "hyp2"
