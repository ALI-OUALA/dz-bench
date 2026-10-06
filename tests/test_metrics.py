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


def test_deterministic_structured_field_scores() -> None:
    from dz_bench.metrics import structured_field_scores
    from dz_bench.models import (
        BoundingBox,
        Confidence,
        DocumentExtraction,
        Provenance,
        StructuredField,
    )

    conf = Confidence(score=1.0)
    prov = Provenance(kind="system_prediction", source="test")
    ref = DocumentExtraction(
        document_id="doc1",
        schema_name="s",
        schema_version="1.0.0",
        fields=[
            StructuredField(
                field_id=str(i),
                field_name=str(i),
                value="1",
                normalized_value="1",
                value_type="string",
                confidence=conf,
                provenance=prov,
                bbox=BoundingBox(x=0, y=0, width=10, height=10),
            )
            for i in range(100)
        ],
        validations=[],
    )
    hyp = DocumentExtraction(
        document_id="doc1",
        schema_name="s",
        schema_version="1.0.0",
        fields=[
            StructuredField(
                field_id=str(i),
                field_name=str(i),
                value="1",
                normalized_value="1",
                value_type="string",
                confidence=conf,
                provenance=prov,
                bbox=BoundingBox(x=0, y=0, width=10, height=10),
            )
            for i in range(100)
        ],
        validations=[],
    )
    scores1 = structured_field_scores(ref, hyp)
    scores2 = structured_field_scores(ref, hyp)
    assert scores1.coordinate_iou == scores2.coordinate_iou


def test_deterministic_greedy_block_matches() -> None:
    from dz_bench.metrics import greedy_block_matches
    from dz_bench.models import Block, BoundingBox, Confidence, Provenance

    conf = Confidence(score=1.0)
    prov = Provenance(kind="system_prediction", source="test")
    ref = [
        Block(
            block_id="b1",
            block_type="paragraph",
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            reading_order_index=0,
            confidence=conf,
            provenance=prov,
        )
    ]
    hyp = [
        Block(
            block_id=str(i),
            block_type="paragraph",
            bbox=BoundingBox(x=10, y=10, width=10, height=10),
            reading_order_index=i,
            confidence=conf,
            provenance=prov,
        )
        for i in range(100)
    ]
    matches1 = greedy_block_matches(ref, hyp, minimum_iou=0.0)
    matches2 = greedy_block_matches(ref, hyp, minimum_iou=0.0)
    assert matches1[0].hypothesis is matches2[0].hypothesis
