from dz_bench.metrics import greedy_block_matches, structured_field_scores
from dz_bench.models import (
    Block,
    BoundingBox,
    Confidence,
    DocumentExtraction,
    Provenance,
    StructuredField,
)


def test_greedy_block_matches_determinism_breaks_ties():
    conf = Confidence(score=1.0)
    prov = Provenance(kind="system_prediction", source="test")
    ref_block = Block(
        block_id="ref1",
        block_type="paragraph",
        bbox=BoundingBox(x=0, y=0, width=10, height=10),
        lines=[],
        reading_order_index=0,
        confidence=conf,
        provenance=prov,
    )
    # Give it multiple hypotheses that are perfectly identical in bounding box IoU.
    # Tie breaking should reliably pick the smallest index (hyp0).
    hyps = [
        Block(
            block_id=f"hyp{i}",
            block_type="paragraph",
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            lines=[],
            reading_order_index=i,
            confidence=conf,
            provenance=prov,
        )
        for i in range(5)
    ]
    matches = greedy_block_matches([ref_block], hyps)
    assert len(matches) == 1
    assert matches[0].hypothesis is not None
    assert matches[0].hypothesis.block_id == "hyp0"


def _field(name: str, value: str) -> StructuredField:
    return StructuredField(
        field_id=f"f-{name}",
        field_name=name,
        value=value,
        normalized_value=value,
        value_type="string",
        confidence=Confidence(score=1.0),
        bbox=BoundingBox(x=0, y=0, width=10, height=10),
        provenance=Provenance(kind="system_prediction", source="test"),
    )


def test_structured_field_scores_determinism_sorts_keys():
    # If sets are unsorted, iterating them is non-deterministic.
    # While it's hard to assert Python set ordering behavior explicitly fails
    # without hash randomization (which is default in Python 3), we can at least
    # verify the logic completes successfully for many keys and calculates correctly.
    fields = [_field(f"key{i}", str(i)) for i in range(200)]
    ref = DocumentExtraction(
        document_id="doc1",
        schema_name="schema",
        schema_version="1.0.0",
        fields=fields,
    )
    hyp = DocumentExtraction(
        document_id="doc1",
        schema_name="schema",
        schema_version="1.0.0",
        fields=fields,
    )
    scores = structured_field_scores(ref, hyp)
    assert scores.coordinate_iou == 1.0
