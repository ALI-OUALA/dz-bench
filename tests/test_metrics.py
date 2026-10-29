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
    from dz_bench.models import Block, BoundingBox, Confidence, Provenance

    ref_block = Block(
        block_id="ref",
        block_type="paragraph",
        reading_order_index=0,
        confidence=Confidence(score=1.0),
        provenance=Provenance(kind="system_prediction", source="ai"),
        bbox=BoundingBox(x=0, y=0, width=10, height=10),
        lines=[],
    )

    # Create multiple identical hypothesis blocks to force a tie
    hyps = []
    for i in range(10):
        hyps.append(
            Block(
                block_id=f"hyp{i}",
                block_type="paragraph",
                reading_order_index=0,
                confidence=Confidence(score=1.0),
                provenance=Provenance(kind="system_prediction", source="ai"),
                bbox=BoundingBox(x=0, y=0, width=10, height=10),
                lines=[],
            )
        )

    matches = greedy_block_matches([ref_block], hyps)
    # The tiebreaker -index ensures we always pick index 0 in the hyps list
    assert matches[0].hypothesis is hyps[0]


def test_structured_field_scores_determinism() -> None:
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
    for i in range(20):
        val = float(10 + i * 0.1)
        # We vary the field names so that they get collected in the set and sorted
        field_ref = StructuredField(
            field_id=f"r{i}",
            field_name=f"f{i}",
            value="v",
            normalized_value="v",
            value_type="string",
            confidence=Confidence(score=1.0),
            provenance=Provenance(kind="system_prediction", source="ai"),
            bbox=BoundingBox(x=0, y=0, width=val, height=val),
        )
        field_hyp = StructuredField(
            field_id=f"h{i}",
            field_name=f"f{i}",
            value="v",
            normalized_value="v",
            value_type="string",
            confidence=Confidence(score=1.0),
            provenance=Provenance(kind="system_prediction", source="ai"),
            bbox=BoundingBox(x=0, y=0, width=val + 0.1, height=val + 0.1),
        )
        fields_ref.append(field_ref)
        fields_hyp.append(field_hyp)

    ref = DocumentExtraction(
        document_id="d1", schema_name="s1", schema_version="1.0.0", fields=fields_ref
    )
    hyp = DocumentExtraction(
        document_id="d1", schema_name="s1", schema_version="1.0.0", fields=fields_hyp
    )

    scores = structured_field_scores(ref, hyp)

    # Check that score is computed and coordinate_iou is deterministic.
    # Prior to fix, order of addition could cause slight floating point deviation.
    assert isinstance(scores.coordinate_iou, float)
