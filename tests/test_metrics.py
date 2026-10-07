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

    ref_fields = []
    hyp_fields = []

    # We create many DIFFERENT fields so they are grouped by different names.
    # Iterating over set(names) might cause floating point non-associativity issues.
    for i in range(100):
        name = f"field_{i}"
        val = "val"

        box = BoundingBox(x=0, y=0, width=10, height=10)
        ref_fields.append(
            StructuredField(
                field_id=f"ref_{i}",
                field_name=name,
                value=val,
                normalized_value=val,
                value_type="string",
                confidence=Confidence(score=1.0),
                provenance=Provenance(kind="system_prediction", source="model"),
                bbox=box,
            )
        )

        if i % 3 == 0:
            w = 10.0
        elif i % 3 == 1:
            w = 0.0001
        else:
            w = 10000.0

        box2 = BoundingBox(x=0, y=0, width=w, height=10)
        hyp_fields.append(
            StructuredField(
                field_id=f"hyp_{i}",
                field_name=name,
                value=val,
                normalized_value=val,
                value_type="string",
                confidence=Confidence(score=1.0),
                provenance=Provenance(kind="system_prediction", source="model"),
                bbox=box2,
            )
        )

    ref = DocumentExtraction(
        document_id="d1", schema_name="s1", schema_version="1.0.0", fields=ref_fields
    )
    hyp = DocumentExtraction(
        document_id="d1", schema_name="s1", schema_version="1.0.0", fields=hyp_fields
    )

    # Ensure reversing the order of fields in the list gives the EXACT SAME float value
    # because of sorting names instead of using a set.
    score_normal = structured_field_scores(ref, hyp)

    ref.fields.reverse()
    hyp.fields.reverse()
    score_reversed = structured_field_scores(ref, hyp)

    assert score_normal.coordinate_iou == score_reversed.coordinate_iou


def test_greedy_block_matches_determinism() -> None:
    from dz_bench.metrics import greedy_block_matches
    from dz_bench.models import Block, BoundingBox, Confidence, Provenance

    refs = []
    hyps = []

    b1 = Block(
        block_id="r1",
        block_type="paragraph",
        bbox=BoundingBox(x=0, y=0, width=10, height=10),
        reading_order_index=1,
        confidence=Confidence(score=1.0),
        provenance=Provenance(kind="system_prediction", source="model"),
        lines=[],
    )
    refs.append(b1)

    for i in range(10):
        # same exact box for all hyps, meaning they all have the same IoU (1.0)
        # Without tie-breaking logic, max() will return whichever comes first in the set()
        b2 = Block(
            block_id=f"h{i}",
            block_type="paragraph",
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            reading_order_index=i,
            confidence=Confidence(score=1.0),
            provenance=Provenance(kind="system_prediction", source="model"),
            lines=[],
        )
        hyps.append(b2)

    matches = greedy_block_matches(refs, hyps)
    # The tie breaker favors the element with the smallest index (since `-index` is maximized),
    # meaning `h0` should always be matched.
    assert matches[0].hypothesis is not None
    assert matches[0].hypothesis.block_id == "h0"
