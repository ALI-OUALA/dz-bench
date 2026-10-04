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
    import random

    from dz_bench.metrics import structured_field_scores
    from dz_bench.models import (
        BoundingBox,
        Confidence,
        DocumentExtraction,
        Provenance,
        StructuredField,
    )

    def _field(name: str, value: str, *, width: float) -> StructuredField:
        return StructuredField(
            field_id=f"field-{name}",
            field_name=name,
            value=value,
            normalized_value=value,
            value_type="string",
            confidence=Confidence(score=1.0, calibrated=False),
            page_id="page-1",
            bbox=BoundingBox(x=0, y=0, width=width, height=10),
            provenance=Provenance(kind="human_annotation", source="test"),
        )

    # Use a fixed random seed to ensure determinism in the test setup itself
    random.seed(42)
    fields1 = []
    fields2 = []

    # Generate a wide variety of scales to exaggerate floating-point summation differences
    # if iteration order is non-deterministic (dependent on set hash seed).
    for i in range(100):
        # We need IoU to vary drastically
        w1 = random.random() * (10 ** random.randint(-15, 15))
        w2 = random.random() * (10 ** random.randint(-15, 15))
        fields1.append(_field(f"name_{i}", "val", width=w1))
        fields2.append(_field(f"name_{i}", "val", width=w2))

    # Add extreme values designed to trigger floating point accumulation discrepancies
    # depending on addition order (e.g. 1.0 + 1e16 - 1e16 vs 1e16 - 1e16 + 1.0)
    # The coordinate_iou summation uses Python's `sum()` which accumulates in order.
    # While Python's `sum()` on floats uses math.fsum-like Neumaier summation in modern
    # CPython versions, the exactness is not perfectly guaranteed across all
    # distributions without sorting.
    fields1.append(_field("extreme_1", "val", width=1e16))
    fields2.append(_field("extreme_1", "val", width=1e16))
    fields1.append(_field("extreme_2", "val", width=1e-16))
    fields2.append(_field("extreme_2", "val", width=1e-16))

    doc1 = DocumentExtraction(
        document_id="d1", schema_name="s", schema_version="1.0.0", fields=fields1
    )
    doc2 = DocumentExtraction(
        document_id="d1", schema_name="s", schema_version="1.0.0", fields=fields2
    )

    # We shuffle the fields to simulate order differences if hash iteration were used,
    # though since we fixed `sorted()` in the function, it should be robust.
    random.shuffle(fields1)
    random.shuffle(fields2)
    doc1_shuffled = DocumentExtraction(
        document_id="d1", schema_name="s", schema_version="1.0.0", fields=fields1
    )
    doc2_shuffled = DocumentExtraction(
        document_id="d1", schema_name="s", schema_version="1.0.0", fields=fields2
    )

    scores1 = structured_field_scores(doc1, doc2)
    scores2 = structured_field_scores(doc1_shuffled, doc2_shuffled)

    assert scores1.coordinate_iou == scores2.coordinate_iou
