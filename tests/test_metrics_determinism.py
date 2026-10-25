from dz_bench.metrics import (
    greedy_block_matches,
    structured_field_scores,
)
from dz_bench.models import (
    Block,
    BoundingBox,
    Confidence,
    DocumentExtraction,
    Provenance,
    StructuredField,
)


def test_greedy_block_matches_determinism():
    conf = Confidence(score=1.0)
    prov = Provenance(kind="human_annotation", source="test")
    ref = [
        Block(
            block_id="r1",
            block_type="paragraph",
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            reading_order_index=0,
            confidence=conf,
            provenance=prov,
            lines=[],
        ),
    ]
    hyp = [
        Block(
            block_id="h1",
            block_type="paragraph",
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            reading_order_index=0,
            confidence=conf,
            provenance=prov,
            lines=[],
        ),
        Block(
            block_id="h2",
            block_type="paragraph",
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            reading_order_index=1,
            confidence=conf,
            provenance=prov,
            lines=[],
        ),
    ]

    matches = greedy_block_matches(ref, hyp)
    assert len(matches) == 1
    assert matches[0].hypothesis.block_id == "h1"


def test_structured_field_scores_determinism():
    prov = Provenance(kind="human_annotation", source="test")
    fields1 = [
        StructuredField(
            field_id="f1",
            field_name="b",
            value="1",
            normalized_value="1",
            value_type="string",
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            confidence=Confidence(score=1.0),
            provenance=prov,
        ),
        StructuredField(
            field_id="f2",
            field_name="a",
            value="1",
            normalized_value="1",
            value_type="string",
            bbox=BoundingBox(x=10, y=10, width=10, height=10),
            confidence=Confidence(score=1.0),
            provenance=prov,
        ),
    ]
    doc = DocumentExtraction(
        schema_name="s", schema_version="1.0.0", document_id="d1", fields=fields1, validations=[]
    )
    scores = structured_field_scores(doc, doc)
    assert scores.f1 == 1.0
