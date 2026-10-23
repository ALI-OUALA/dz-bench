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


def test_greedy_block_matches_tie_breaking() -> None:
    from dz_bench.metrics import greedy_block_matches
    from dz_bench.models import Block, BoundingBox, Confidence, Provenance

    prov = Provenance(kind="system_prediction", source="test")
    conf = Confidence(score=1.0)

    reference = [
        Block(
            block_id="ref1",
            block_type="paragraph",
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            reading_order_index=0,
            confidence=conf,
            provenance=prov,
        )
    ]

    hypothesis = [
        Block(
            block_id="hyp1",
            block_type="paragraph",
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            reading_order_index=0,
            confidence=conf,
            provenance=prov,
        ),
        Block(
            block_id="hyp2",
            block_type="paragraph",
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            reading_order_index=1,
            confidence=conf,
            provenance=prov,
        ),
    ]

    # hyp1 and hyp2 have identical bounding boxes, so IoU is equal.
    # The tiebreaker should favor the smaller index, meaning hyp1 is chosen.
    matches = greedy_block_matches(reference, hypothesis)
    assert len(matches) == 1
    assert matches[0].hypothesis is not None
    assert matches[0].hypothesis.block_id == "hyp1"


def test_structured_field_scores_determinism() -> None:
    from dz_bench.metrics import structured_field_scores
    from dz_bench.models import (
        BoundingBox,
        Confidence,
        DocumentExtraction,
        Provenance,
        StructuredField,
    )

    prov = Provenance(kind="human_annotation", source="test")
    conf = Confidence(score=1.0)

    # We use field names that when sorted in different ways could theoretically cause differences
    # if iterating randomly (due to floating point summation order).
    field_a = StructuredField(
        field_id="1",
        field_name="field_a",
        value="A",
        normalized_value="A",
        value_type="string",
        confidence=conf,
        provenance=prov,
        bbox=BoundingBox(x=0, y=0, width=10, height=10),
    )
    field_b = StructuredField(
        field_id="2",
        field_name="field_b",
        value="B",
        normalized_value="B",
        value_type="string",
        confidence=conf,
        provenance=prov,
        bbox=BoundingBox(x=10, y=10, width=10, height=10),
    )

    doc1 = DocumentExtraction(
        document_id="doc1", schema_name="schema", schema_version="1.0.0", fields=[field_b, field_a]
    )

    doc2 = DocumentExtraction(
        document_id="doc1", schema_name="schema", schema_version="1.0.0", fields=[field_a, field_b]
    )

    # Order of fields in the doc shouldn't matter; what matters is they are
    # aggregated deterministically. This test primarily ensures no crash/errors
    # and behavior remains consistent.
    score1 = structured_field_scores(doc1, doc1)
    score2 = structured_field_scores(doc2, doc2)
    assert score1.f1 == 1.0
    assert score1 == score2
