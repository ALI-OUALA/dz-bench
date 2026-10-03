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


def test_structured_field_scores_are_deterministic() -> None:
    import random

    from dz_bench.metrics import structured_field_scores
    from dz_bench.models import BoundingBox, DocumentExtraction, StructuredField

    # Create fields with bounding boxes and values
    fields1 = [
        StructuredField(
            field_id=f"id_{i}",
            field_name=f"field_{i}",
            value=f"value_{i}",
            normalized_value=f"value_{i}",
            value_type="string",
            confidence={"score": 1.0}, # type: ignore
            provenance={"kind": "system_prediction", "source": "test"}, # type: ignore
            bbox=BoundingBox(x=i * 0.1, y=i * 0.1, width=10.0, height=10.0),
        )
        for i in range(20)
    ]
    fields2 = [
        StructuredField(
            field_id=f"id_{i}",
            field_name=f"field_{i}",
            value=f"value_{i}" if i % 2 == 0 else f"other_{i}",
            normalized_value=f"value_{i}" if i % 2 == 0 else f"other_{i}",
            value_type="string",
            confidence={"score": 1.0}, # type: ignore
            provenance={"kind": "system_prediction", "source": "test"}, # type: ignore
            bbox=BoundingBox(x=i * 0.1, y=i * 0.1, width=9.0, height=11.0),
        )
        for i in range(25)
    ]

    reference = DocumentExtraction(
        document_id="doc1",
        schema_name="schema1",
        schema_version="1.0.0",
        fields=fields1,
        validations=[],
    )

    # Store initial scores
    initial_scores = structured_field_scores(
        reference,
        DocumentExtraction(
            document_id="doc1",
            schema_name="schema1",
            schema_version="1.0.0",
            fields=fields2,
            validations=[],
        ),
    )

    # Randomly shuffle field names and see if scores are identical
    for _ in range(5):
        shuffled_fields1 = fields1.copy()
        shuffled_fields2 = fields2.copy()
        random.shuffle(shuffled_fields1)
        random.shuffle(shuffled_fields2)

        shuffled_ref = DocumentExtraction(
            document_id="doc1",
            schema_name="schema1",
            schema_version="1.0.0",
            fields=shuffled_fields1,
            validations=[],
        )
        shuffled_hyp = DocumentExtraction(
            document_id="doc1",
            schema_name="schema1",
            schema_version="1.0.0",
            fields=shuffled_fields2,
            validations=[],
        )

        shuffled_scores = structured_field_scores(shuffled_ref, shuffled_hyp)
        assert initial_scores == shuffled_scores
