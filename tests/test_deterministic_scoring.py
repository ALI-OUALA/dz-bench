from dz_bench.metrics import structured_field_scores
from dz_bench.models import BoundingBox, Confidence, DocumentExtraction, Provenance, StructuredField


def test_structured_field_scores_determinism():
    fields_ref = []
    fields_hyp = []

    names = ["C", "A", "B"]

    for i, name in enumerate(names):
        bbox1 = BoundingBox(x=float(i), y=0.0, width=10.0, height=10.0)
        bbox2 = BoundingBox(x=float(i) + 0.1 * i, y=0.0, width=10.0, height=10.0)

        fields_ref.append(
            StructuredField(
                field_id=f"ref_{name}",
                field_name=name,
                value=name,
                value_type="string",
                normalized_value=name,
                bbox=bbox1,
                confidence=Confidence(score=1.0),
                provenance=Provenance(kind="human_annotation", source="me"),
            )
        )
        fields_hyp.append(
            StructuredField(
                field_id=f"hyp_{name}",
                field_name=name,
                value=name,
                value_type="string",
                normalized_value=name,
                bbox=bbox2,
                confidence=Confidence(score=1.0),
                provenance=Provenance(kind="human_annotation", source="me"),
            )
        )

    ref = DocumentExtraction(
        document_id="doc1", schema_name="s", schema_version="1.0.0", fields=fields_ref
    )
    hyp = DocumentExtraction(
        document_id="doc1", schema_name="s", schema_version="1.0.0", fields=fields_hyp
    )

    score = structured_field_scores(ref, hyp)
    assert score.coordinate_iou is not None
