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


def test_deterministic_greedy_block_matches() -> None:
    from dz_bench.metrics import greedy_block_matches
    from dz_bench.models import Block, BoundingBox, Confidence, Provenance

    # Same identical IOUs, we should pick the smaller index because of -index tiebreaker
    ref = [
        Block(
            block_id="r1",
            block_type="paragraph",
            reading_order_index=0,
            confidence=Confidence(score=1.0),
            provenance=Provenance(kind="synthetic_generator", source="test"),
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            lines=[],
        ),
    ]

    hyp = [
        Block(
            block_id="h1",
            block_type="paragraph",
            reading_order_index=0,
            confidence=Confidence(score=1.0),
            provenance=Provenance(kind="system_prediction", source="test"),
            bbox=BoundingBox(x=0, y=0, width=5, height=5),
            lines=[],
        ),
        Block(
            block_id="h2",
            block_type="paragraph",
            reading_order_index=1,
            confidence=Confidence(score=1.0),
            provenance=Provenance(kind="system_prediction", source="test"),
            bbox=BoundingBox(x=5, y=5, width=5, height=5),
            lines=[],
        ),
    ]

    matches = greedy_block_matches(ref, hyp)
    assert len(matches) == 1
    assert matches[0].hypothesis is not None
    # Because of deterministic tiebreaking, index 0 (h1) should always win over index 1 (h2)
    assert matches[0].hypothesis.block_id == "h1"


def test_deterministic_structured_field_scores() -> None:
    from dz_bench.metrics import structured_field_scores
    from dz_bench.models import (
        BoundingBox,
        Confidence,
        DocumentExtraction,
        Provenance,
        StructuredField,
    )

    fields_ref = []
    fields_hyp = []

    for i in range(100):
        name = f"field_{i}"
        box_iou = i * 0.001
        field1 = StructuredField(
            field_id=str(i),
            field_name=name,
            value="a",
            normalized_value="a",
            value_type="string",
            confidence=Confidence(score=1.0),
            provenance=Provenance(kind="synthetic_generator", source="test"),
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
        )
        field2 = StructuredField(
            field_id=str(i),
            field_name=name,
            value="a",
            normalized_value="a",
            value_type="string",
            confidence=Confidence(score=1.0),
            provenance=Provenance(kind="synthetic_generator", source="test"),
            bbox=BoundingBox(x=0, y=0, width=10, height=10 + box_iou),
        )
        fields_ref.append(field1)
        fields_hyp.append(field2)

    ref = DocumentExtraction(
        document_id="doc1",
        schema_name="schema",
        schema_version="1.0.0",
        fields=fields_ref,
        validations=[],
    )
    hyp = DocumentExtraction(
        document_id="doc1",
        schema_name="schema",
        schema_version="1.0.0",
        fields=fields_hyp,
        validations=[],
    )

    # If non-deterministic, exact floating point value would vary by PYTHONHASHSEED
    res = structured_field_scores(ref, hyp)
    assert res.coordinate_iou > 0.99
