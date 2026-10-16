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


def test_greedy_block_matches_tie_breaker_is_deterministic() -> None:
    from dz_bench.metrics import greedy_block_matches
    from dz_bench.models import Block, BoundingBox, Confidence, Provenance

    def create_block(
        id_val: str, x: float = 0, y: float = 0, width: float = 10, height: float = 10
    ) -> Block:
        return Block(
            block_id=id_val,
            block_type="paragraph",
            bbox=BoundingBox(x=x, y=y, width=width, height=height),
            reading_order_index=0,
            confidence=Confidence(score=1.0),
            provenance=Provenance(kind="system_prediction", source="test"),
            lines=[],
        )

    # One reference block
    reference = [create_block("ref")]
    # Two identical hypothesis blocks in the same location
    hypothesis = [create_block("hyp0"), create_block("hyp1")]

    # greedy_block_matches should always pick the lower index (0) in case of an exact tie.
    matches = greedy_block_matches(reference, hypothesis)
    assert len(matches) == 1
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

    def create_field(id_val: str, name: str, val: str, width: float) -> StructuredField:
        return StructuredField(
            field_id=id_val,
            field_name=name,
            value=val,
            normalized_value=val,
            value_type="string",
            bbox=BoundingBox(x=0, y=0, width=width, height=10),
            confidence=Confidence(score=1.0),
            provenance=Provenance(kind="system_prediction", source="test"),
        )

    # Creating many fields where the sets of field names will be large.
    # Python set iteration order depends on hash seed. We can simulate permutations
    # by shuffling the order of fields in the document and observing if the coordinate_iou changes.
    fields_ref = [create_field(f"id{i}", f"test{i}", "A", 10.0) for i in range(500)]
    fields_hyp = [
        create_field(f"id{i}", f"test{i}", "A", 10.0 + (i % 10) * 0.1) for i in range(500)
    ]

    doc_ref = DocumentExtraction(
        document_id="doc1",
        schema_name="doc",
        schema_version="1.0.0",
        fields=fields_ref,
    )
    doc_hyp = DocumentExtraction(
        document_id="doc1",
        schema_name="doc",
        schema_version="1.0.0",
        fields=fields_hyp,
    )

    base_score = structured_field_scores(doc_ref, doc_hyp).coordinate_iou

    import random

    random.seed(42)

    # Check that reversing the fields does not alter the output coordinate_iou score
    doc_ref_rev = DocumentExtraction(
        document_id="doc1",
        schema_name="doc",
        schema_version="1.0.0",
        fields=fields_ref[::-1],
    )
    doc_hyp_rev = DocumentExtraction(
        document_id="doc1",
        schema_name="doc",
        schema_version="1.0.0",
        fields=fields_hyp[::-1],
    )
    rev_score = structured_field_scores(doc_ref_rev, doc_hyp_rev).coordinate_iou

    assert base_score == rev_score
