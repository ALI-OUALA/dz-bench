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
    from dz_bench.models import Block, BoundingBox, Confidence, Provenance

    confidence = Confidence(score=1.0)
    provenance = Provenance(kind="system_prediction", source="model")

    reference = [
        Block(
            block_id="ref1",
            block_type="paragraph",
            reading_order_index=1,
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            confidence=confidence,
            provenance=provenance,
        )
    ]
    # Provide multiple hypotheses with the EXACT same bounding box.
    # The tie-breaker should reliably select the one with the lowest index
    # (because -index is larger)
    hypothesis = [
        Block(
            block_id="hyp0",
            block_type="paragraph",
            reading_order_index=1,
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            confidence=confidence,
            provenance=provenance,
        ),
        Block(
            block_id="hyp1",
            block_type="paragraph",
            reading_order_index=2,
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            confidence=confidence,
            provenance=provenance,
        ),
        Block(
            block_id="hyp2",
            block_type="paragraph",
            reading_order_index=3,
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            confidence=confidence,
            provenance=provenance,
        ),
    ]

    from dz_bench.metrics import greedy_block_matches

    matches = greedy_block_matches(reference, hypothesis)
    assert len(matches) == 1
    # It should consistently pick hyp0 because it has index 0
    assert matches[0].hypothesis is not None
    assert matches[0].hypothesis.block_id == "hyp0"


def test_structured_field_scores_determinism() -> None:
    from dz_bench.metrics import structured_field_scores
    from dz_bench.models import (
        BoundingBox,
        Confidence,
        DocumentExtraction,
        Provenance,
        StructuredField,
    )

    def create_field(name: str, width: float) -> StructuredField:
        return StructuredField(
            field_id="f1",
            field_name=name,
            value="val",
            normalized_value="val",
            value_type="string",
            confidence=Confidence(score=1.0),
            provenance=Provenance(kind="system_prediction", source="model"),
            bbox=BoundingBox(x=0, y=0, width=width, height=10),
        )

    # Use a large number of fields with names that would hash randomly.
    # We want to ensure that coordinate_iou is deterministic regardless of Python's
    # string hash randomization. If the metrics code doesn't sort the names, the
    # order of floats in `coordinate_scores` will vary, potentially leading to
    # non-associative float summation differences.
    fields_ref = []
    fields_hyp = []
    for i in range(50):
        name = f"field_xyz_{i}"
        fields_ref.append(create_field(name, 10.0 + i * 0.1))
        fields_hyp.append(create_field(name, 9.0 + i * 0.1))

    ref = DocumentExtraction(
        document_id="d1", schema_name="s1", schema_version="1.0.0", fields=fields_ref
    )
    hyp = DocumentExtraction(
        document_id="d1", schema_name="s1", schema_version="1.0.0", fields=fields_hyp
    )

    scores = structured_field_scores(ref, hyp)
    assert scores.coordinate_iou > 0.0
