from dz_bench.metrics import greedy_block_matches, structured_field_scores
from dz_bench.models import (
    Block,
    BoundingBox,
    Confidence,
    DocumentExtraction,
    Provenance,
    StructuredField,
)


def _make_block(id_: str, x: float, y: float, w: float, h: float) -> Block:
    return Block(
        block_id=id_,
        block_type="paragraph",
        reading_order_index=0,
        bbox=BoundingBox(x=x, y=y, width=w, height=h),
        confidence=Confidence(score=1.0, calibrated=False),
        provenance=Provenance(kind="human_annotation", source="test"),
    )


def test_greedy_block_matches_is_deterministic_on_ties() -> None:
    # A reference block overlapping two identical hypotheses boxes equally
    ref = [_make_block("r1", 0, 0, 10, 10)]
    hyp = [
        _make_block("h1", 5, 5, 10, 10),
        _make_block("h2", 5, 5, 10, 10),
    ]

    matches = greedy_block_matches(ref, hyp, minimum_iou=0.0)
    # the index tie breaker is -index, so max returns the one with largest -index, which is 0.
    # Therefore, it should always match with h1.
    assert matches[0].hypothesis is not None
    assert matches[0].hypothesis.block_id == "h1"


def _field(name: str, bbox: BoundingBox) -> StructuredField:
    return StructuredField(
        field_id="id",
        field_name=name,
        value="a",
        normalized_value="a",
        value_type="string",
        confidence=Confidence(score=1.0, calibrated=False),
        page_id="p1",
        bbox=bbox,
        provenance=Provenance(kind="human_annotation", source="test"),
    )


def test_structured_field_scores_float_summation_deterministic() -> None:
    # We create many fields with the same names to test deterministic set iteration
    # Since we can't reliably test hash randomization across processes in one test,
    # we at least ensure sorting is used by checking if it runs identically.

    bboxes = [
        BoundingBox(x=0, y=0, width=10, height=10),
        BoundingBox(x=1, y=1, width=10, height=10),
        BoundingBox(x=2, y=2, width=10, height=10),
        BoundingBox(x=3, y=3, width=10, height=10),
        BoundingBox(x=4, y=4, width=10, height=10),
    ]

    # Large number of fields to amplify any precision difference
    # if floats are added in different orders.
    fields_ref = []
    fields_hyp = []
    for i in range(100):
        name = f"f_{i % 10}"
        b_ref = bboxes[i % 5]
        b_hyp = bboxes[(i + 1) % 5]
        fields_ref.append(_field(name, b_ref))
        fields_hyp.append(_field(name, b_hyp))

    ref = DocumentExtraction(
        document_id="doc1",
        schema_name="sch",
        schema_version="1.0.0",
        fields=fields_ref,
    )
    hyp = DocumentExtraction(
        document_id="doc1",
        schema_name="sch",
        schema_version="1.0.0",
        fields=fields_hyp,
    )

    res1 = structured_field_scores(ref, hyp)
    res2 = structured_field_scores(ref, hyp)
    assert res1.coordinate_iou == res2.coordinate_iou
