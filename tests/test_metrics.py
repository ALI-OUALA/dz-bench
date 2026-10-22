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


def test_greedy_block_matches_determinism() -> None:
    from dz_bench.metrics import greedy_block_matches

    # Provide only required fields so validation succeeds.
    from dz_bench.models import Block, BoundingBox, Confidence, Provenance

    base_kwargs = {
        "block_id": "test",
        "block_type": "paragraph",
        "reading_order_index": 0,
        "confidence": Confidence(score=1.0),
        "provenance": Provenance(kind="system_prediction", source="test"),
        "lines": [],
    }

    ref = Block(bbox=BoundingBox(x=0, y=0, width=10, height=10), **base_kwargs)
    # Both hypotheses overlap equally with the reference.
    hyp1 = Block(bbox=BoundingBox(x=5, y=5, width=10, height=10), **base_kwargs)
    hyp2 = Block(bbox=BoundingBox(x=5, y=5, width=10, height=10), **base_kwargs)

    # In a tie, it should pick the one with the lowest index (so hyp1, which is at index 0)
    matches = greedy_block_matches([ref], [hyp1, hyp2], minimum_iou=0.0)
    assert len(matches) == 1
    assert matches[0].hypothesis is hyp1


def test_structured_field_scores_determinism() -> None:
    from dz_bench.metrics import structured_field_scores
    from dz_bench.models import Confidence, DocumentExtraction, Provenance, StructuredField

    base_field = {
        "field_id": "test",
        "value_type": "string",
        "confidence": Confidence(score=1.0),
        "provenance": Provenance(kind="system_prediction", source="test"),
    }

    # Two identical documents that have fields in a different set order
    ref = DocumentExtraction(
        document_id="doc1",
        schema_name="invoice",
        schema_version="1.0.0",
        fields=[
            StructuredField(field_name="a", value="1", normalized_value="1", **base_field),
            StructuredField(field_name="b", value="2", normalized_value="2", **base_field),
        ],
    )
    hyp = DocumentExtraction(
        document_id="doc1",
        schema_name="invoice",
        schema_version="1.0.0",
        fields=[
            StructuredField(field_name="a", value="1", normalized_value="1", **base_field),
            StructuredField(field_name="b", value="2", normalized_value="2", **base_field),
        ],
    )
    scores = structured_field_scores(ref, hyp)
    assert scores.precision == 1.0
    assert scores.recall == 1.0
