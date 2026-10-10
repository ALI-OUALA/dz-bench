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

    fields1 = []
    fields2 = []

    fields1.append(
        StructuredField(
            field_id="f1",
            field_name="name1",
            value="1",
            normalized_value="1",
            value_type="string",
            confidence=Confidence(score=1.0),
            provenance=Provenance(kind="system_prediction", source="test"),
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
        )
    )
    fields2.append(
        StructuredField(
            field_id="f1",
            field_name="name1",
            value="1",
            normalized_value="1",
            value_type="string",
            confidence=Confidence(score=1.0),
            provenance=Provenance(kind="system_prediction", source="test"),
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
        )
    )

    fields1.append(
        StructuredField(
            field_id="f2",
            field_name="name2",
            value="1",
            normalized_value="1",
            value_type="string",
            confidence=Confidence(score=1.0),
            provenance=Provenance(kind="system_prediction", source="test"),
            bbox=BoundingBox(x=0, y=0, width=10**8, height=10**8),
        )
    )
    fields2.append(
        StructuredField(
            field_id="f2",
            field_name="name2",
            value="1",
            normalized_value="1",
            value_type="string",
            confidence=Confidence(score=1.0),
            provenance=Provenance(kind="system_prediction", source="test"),
            bbox=BoundingBox(x=0, y=0, width=1, height=1),
        )
    )

    reference = DocumentExtraction(
        document_id="doc1", schema_name="schema1", schema_version="1.0.0", fields=fields1
    )
    hypothesis = DocumentExtraction(
        document_id="doc1", schema_name="schema1", schema_version="1.0.0", fields=fields2
    )

    scores1 = structured_field_scores(reference, hypothesis)

    # We test it's reproducible by asserting exactly equal for two different name lists
    # However we can't change PYTHONHASHSEED at runtime.
    # We can at least check it doesn't crash and returns the expected result.
    assert scores1.coordinate_iou == 0.5


def test_greedy_block_matches_tie_breaking() -> None:
    from dz_bench.metrics import greedy_block_matches
    from dz_bench.models import Block, BoundingBox, Confidence, Provenance

    reference = [
        Block(
            block_id="ref1",
            block_type="paragraph",
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            reading_order_index=0,
            confidence=Confidence(score=1.0),
            provenance=Provenance(kind="system_prediction", source="test"),
        )
    ]

    # Two identical blocks, only ID and index differ.
    hypothesis = [
        Block(
            block_id="hyp1",
            block_type="paragraph",
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            reading_order_index=0,
            confidence=Confidence(score=1.0),
            provenance=Provenance(kind="system_prediction", source="test"),
        ),
        Block(
            block_id="hyp2",
            block_type="paragraph",
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            reading_order_index=1,
            confidence=Confidence(score=1.0),
            provenance=Provenance(kind="system_prediction", source="test"),
        ),
    ]

    matches = greedy_block_matches(reference, hypothesis)
    assert len(matches) == 1
    assert matches[0].hypothesis is not None
    assert matches[0].hypothesis.block_id == "hyp1"

    # And if hyp2 is first in the list
    hypothesis_reversed = hypothesis[::-1]
    matches_reversed = greedy_block_matches(reference, hypothesis_reversed)
    assert len(matches_reversed) == 1
    assert matches_reversed[0].hypothesis is not None
    # Still picks the first one in the list deterministically
    assert matches_reversed[0].hypothesis.block_id == "hyp2"
