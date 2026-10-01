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
    from dz_bench.models import DocumentExtraction, StructuredField

    reference = DocumentExtraction(
        document_id="doc1",
        schema_name="schema1",
        schema_version="1.0.0",
        fields=[
            StructuredField(
                field_id="id1",
                field_name="z_field",
                value="1",
                normalized_value="1",
                value_type="string",
                confidence={"score": 1.0},
                provenance={"kind": "system_prediction", "source": "test"},
            ),
            StructuredField(
                field_id="id1",
                field_name="a_field",
                value="2",
                normalized_value="2",
                value_type="string",
                confidence={"score": 1.0},
                provenance={"kind": "system_prediction", "source": "test"},
            ),
            StructuredField(
                field_id="id1",
                field_name="m_field",
                value="3",
                normalized_value="3",
                value_type="string",
                confidence={"score": 1.0},
                provenance={"kind": "system_prediction", "source": "test"},
            ),
        ],
        validations=[],
    )

    hypothesis = DocumentExtraction(
        document_id="doc1",
        schema_name="schema1",
        schema_version="1.0.0",
        fields=[
            StructuredField(
                field_id="id1",
                field_name="z_field",
                value="1",
                normalized_value="1",
                value_type="string",
                confidence={"score": 1.0},
                provenance={"kind": "system_prediction", "source": "test"},
            ),
            StructuredField(
                field_id="id1",
                field_name="a_field",
                value="2",
                normalized_value="2",
                value_type="string",
                confidence={"score": 1.0},
                provenance={"kind": "system_prediction", "source": "test"},
            ),
            StructuredField(
                field_id="id1",
                field_name="m_field",
                value="3",
                normalized_value="3",
                value_type="string",
                confidence={"score": 1.0},
                provenance={"kind": "system_prediction", "source": "test"},
            ),
        ],
        validations=[],
    )

    from dz_bench.metrics import structured_field_scores

    # Run multiple times to ensure deterministic execution order does not raise exceptions
    for _ in range(10):
        scores = structured_field_scores(reference, hypothesis)
        assert scores.exact_accuracy == 1.0
