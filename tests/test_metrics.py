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


def test_greedy_block_matches_tie_breaker() -> None:
    from dz_bench.metrics import greedy_block_matches
    from dz_bench.models import Block, BoundingBox, Confidence, Provenance

    bbox = BoundingBox(x=0, y=0, width=10, height=10)
    conf = Confidence(score=1.0)
    prov = Provenance(kind="synthetic_generator", source="test")

    reference = [
        Block(
            block_id="r1",
            block_type="paragraph",
            bbox=bbox,
            reading_order_index=1,
            confidence=conf,
            provenance=prov,
        )
    ]
    hypothesis = [
        Block(
            block_id="h1",
            block_type="paragraph",
            bbox=bbox,
            reading_order_index=1,
            confidence=conf,
            provenance=prov,
        ),
        Block(
            block_id="h2",
            block_type="paragraph",
            bbox=bbox,
            reading_order_index=2,
            confidence=conf,
            provenance=prov,
        ),
    ]

    matches = greedy_block_matches(reference, hypothesis)
    assert len(matches) == 1
    assert matches[0].hypothesis is hypothesis[0]


def test_structured_field_scores_deterministic_order() -> None:
    from dz_bench.metrics import structured_field_scores
    from dz_bench.models import (
        BoundingBox,
        Confidence,
        DocumentExtraction,
        Provenance,
        StructuredField,
    )

    bbox_ref = BoundingBox(x=0, y=0, width=10, height=10)
    bbox_hyp1 = BoundingBox(x=2, y=2, width=8, height=8)
    bbox_hyp2 = BoundingBox(x=5, y=5, width=5, height=5)
    conf = Confidence(score=1.0)
    prov = Provenance(kind="synthetic_generator", source="test")

    def make_field(name: str, bbox: BoundingBox) -> StructuredField:
        return StructuredField(
            field_id=f"id_{name}",
            field_name=name,
            value="val",
            normalized_value="val",
            value_type="string",
            confidence=conf,
            provenance=prov,
            bbox=bbox,
        )

    ref = DocumentExtraction(
        document_id="doc1",
        schema_name="schema1",
        schema_version="1.0.0",
        fields=[make_field("A", bbox_ref), make_field("B", bbox_ref)],
    )

    hyp1 = DocumentExtraction(
        document_id="doc1",
        schema_name="schema1",
        schema_version="1.0.0",
        fields=[make_field("A", bbox_hyp1), make_field("B", bbox_hyp2)],
    )

    hyp2 = DocumentExtraction(
        document_id="doc1",
        schema_name="schema1",
        schema_version="1.0.0",
        fields=[make_field("B", bbox_hyp2), make_field("A", bbox_hyp1)],
    )

    scores1 = structured_field_scores(ref, hyp1)
    scores2 = structured_field_scores(ref, hyp2)

    assert scores1 == scores2
