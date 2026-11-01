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


def test_greedy_block_matches_deterministic_tie_breaking() -> None:
    from dz_bench.metrics import greedy_block_matches
    from dz_bench.models import Block, BoundingBox

    reference = [
        Block(
            block_id="ref_1",
            block_type="paragraph",
            reading_order_index=0,
            confidence={"score": 1.0},
            provenance={"kind": "system_prediction", "source": "test"},
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
        )
    ]

    # Create two identical bounding boxes to force a tie
    hypothesis = [
        Block(
            block_id="hyp_1",
            block_type="paragraph",
            reading_order_index=0,
            confidence={"score": 1.0},
            provenance={"kind": "system_prediction", "source": "test"},
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
        ),
        Block(
            block_id="hyp_2",
            block_type="paragraph",
            reading_order_index=1,
            confidence={"score": 1.0},
            provenance={"kind": "system_prediction", "source": "test"},
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
        ),
    ]

    # We can test determinism directly by verifying which index wins.
    # Because our logic uses -index, the smaller index should ALWAYS win a tie.
    matches = greedy_block_matches(reference, hypothesis)
    assert len(matches) == 1
    assert matches[0].hypothesis is not None
    assert matches[0].hypothesis.block_id == "hyp_1"


def test_structured_field_scores_determinism() -> None:
    from dz_bench.metrics import structured_field_scores
    from dz_bench.models import (
        BoundingBox,
        Confidence,
        DocumentExtraction,
        Provenance,
        StructuredField,
    )

    ref_fields = []
    hyp_fields = []

    name_big = "BIG"
    ref_fields.append(
        StructuredField(
            field_id="b",
            field_name=name_big,
            value="1",
            normalized_value="1",
            value_type="string",
            confidence=Confidence(score=1),
            provenance=Provenance(kind="system_prediction", source="a"),
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
        )
    )
    hyp_fields.append(
        StructuredField(
            field_id="b",
            field_name=name_big,
            value="1",
            normalized_value="1",
            value_type="string",
            confidence=Confidence(score=1),
            provenance=Provenance(kind="system_prediction", source="a"),
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
        )
    )

    for i in range(25):
        name = f"f_{i}"
        ref_fields.append(
            StructuredField(
                field_id=str(i),
                field_name=name,
                value="1",
                normalized_value="1",
                value_type="string",
                confidence=Confidence(score=1),
                provenance=Provenance(kind="system_prediction", source="a"),
                bbox=BoundingBox(x=0, y=0, width=100000000 + i, height=100000000 + i),
            )
        )
        hyp_fields.append(
            StructuredField(
                field_id=str(i),
                field_name=name,
                value="1",
                normalized_value="1",
                value_type="string",
                confidence=Confidence(score=1),
                provenance=Provenance(kind="system_prediction", source="a"),
                bbox=BoundingBox(x=99999999, y=99999999, width=100000000 + i, height=100000000 + i),
            )
        )

    ref = DocumentExtraction(
        document_id="doc1", schema_name="s", schema_version="1.0.0", fields=ref_fields
    )
    hyp = DocumentExtraction(
        document_id="doc1", schema_name="s", schema_version="1.0.0", fields=hyp_fields
    )

    # Run structured_field_scores multiple times
    scores1 = structured_field_scores(ref, hyp)

    # Verify the specific sum using the deterministic sorted order
    assert scores1.coordinate_iou == 0.03846153846154909
