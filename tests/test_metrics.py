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


def test_structured_field_scores_determinism() -> None:
    from dz_bench.metrics import structured_field_scores
    from dz_bench.models import (
        BoundingBox,
        Confidence,
        DocumentExtraction,
        Provenance,
        StructuredField,
    )

    # Create dummy Extractions that would cause a float precision issue
    # We create multiple fields that have IoU scores summing up non-deterministically.
    # Floating point summation can be affected by the order of addition.

    # We add 3 boxes:
    # Box A: IoU 0.1
    # Box B: IoU 0.10000000000000001
    # Box C: IoU 0.1
    ref = DocumentExtraction(
        document_id="doc1",
        schema_name="s",
        schema_version="1.0.0",
        fields=[
            StructuredField(
                field_id="id1",
                value_type="string",
                confidence=Confidence(score=1.0),
                provenance=Provenance(kind="system_prediction", source="test"),
                field_name="a",
                value="1",
                normalized_value="1",
                bbox=BoundingBox(x=0, y=0, width=10, height=10),
            ),
            StructuredField(
                field_id="id1",
                value_type="string",
                confidence=Confidence(score=1.0),
                provenance=Provenance(kind="system_prediction", source="test"),
                field_name="b",
                value="1",
                normalized_value="1",
                bbox=BoundingBox(x=0, y=0, width=10, height=10),
            ),
            StructuredField(
                field_id="id1",
                value_type="string",
                confidence=Confidence(score=1.0),
                provenance=Provenance(kind="system_prediction", source="test"),
                field_name="c",
                value="1",
                normalized_value="1",
                bbox=BoundingBox(x=0, y=0, width=10, height=10),
            ),
        ],
        validations=[],
    )
    # Give hypothesis boxes that yield different IoUs for different keys
    hyp = DocumentExtraction(
        document_id="doc1",
        schema_name="s",
        schema_version="1.0.0",
        fields=[
            StructuredField(
                field_id="id1",
                value_type="string",
                confidence=Confidence(score=1.0),
                provenance=Provenance(kind="system_prediction", source="test"),
                field_name="a",
                value="1",
                normalized_value="1",
                bbox=BoundingBox(x=0, y=0, width=2, height=5),
            ),  # IoU 10/100 = 0.1
            StructuredField(
                field_id="id1",
                value_type="string",
                confidence=Confidence(score=1.0),
                provenance=Provenance(kind="system_prediction", source="test"),
                field_name="b",
                value="1",
                normalized_value="1",
                bbox=BoundingBox(x=0, y=0, width=10, height=5),
            ),  # IoU 50/100 = 0.5
            StructuredField(
                field_id="id1",
                value_type="string",
                confidence=Confidence(score=1.0),
                provenance=Provenance(kind="system_prediction", source="test"),
                field_name="c",
                value="1",
                normalized_value="1",
                bbox=BoundingBox(x=0, y=0, width=8, height=5),
            ),  # IoU 40/100 = 0.4
        ],
        validations=[],
    )

    # To test the fix actually fixes order, we can check that it works under different orders.
    # But since we fixed it using `sorted()`, we can just ensure that `structured_field_scores`
    # returns exactly
    # the same score regardless of the order the fields were inserted.

    # We will test order consistency explicitly, although python 3.7+ preserves insertion order,
    # the original code `set() | set()` uses sets which don't preserve insertion order
    # and order depends on string hashes.
    scores1 = structured_field_scores(ref, hyp)

    # Swap names to ensure the sets might iterate differently if not sorted
    ref2 = DocumentExtraction(
        document_id="doc1",
        schema_name="s",
        schema_version="1.0.0",
        fields=[ref.fields[2], ref.fields[0], ref.fields[1]],
        validations=[],
    )
    hyp2 = DocumentExtraction(
        document_id="doc1",
        schema_name="s",
        schema_version="1.0.0",
        fields=[hyp.fields[2], hyp.fields[0], hyp.fields[1]],
        validations=[],
    )
    scores2 = structured_field_scores(ref2, hyp2)

    assert scores1.coordinate_iou == scores2.coordinate_iou
