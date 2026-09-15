import pytest

from dz_bench.metrics import structured_field_scores
from dz_bench.models import (
    BoundingBox,
    Checksum,
    Confidence,
    CoordinateSystem,
    DatasetRevision,
    DocumentExtraction,
    GroundTruth,
    GroundTruthDocument,
    Manifest,
    ManifestDocument,
    ManifestPage,
    ManifestSource,
    PageContent,
    Predictions,
    PredictionSample,
    Provenance,
    RunMetadata,
    StructuredField,
    SystemMetadata,
)
from dz_bench.scoring import score


def create_field(id_str, name, val, bbox_x):
    return StructuredField(
        field_id=id_str,
        field_name=name,
        value=val,
        normalized_value=val,
        value_type="string",
        confidence=Confidence(score=0.9),
        provenance=Provenance(kind="system_prediction", source="test"),
        bbox=BoundingBox(x=bbox_x, y=0, width=10, height=10),
    )


def test_structured_field_scores_determinism():
    """Ensure floating point summation isn't affected by set iteration order."""
    names = [f"name_{i}" for i in range(10000)]
    fields_ref = [create_field(f"r_{i}", n, "v", float(i % 100)) for i, n in enumerate(names)]
    fields_hyp = [
        create_field(f"h_{i}", n, "v", float(i % 100) + 0.1 * (i % 3)) for i, n in enumerate(names)
    ]

    ref = DocumentExtraction(
        document_id="doc1",
        schema_name="schema1",
        schema_version="1.0.0",
        fields=fields_ref,
    )
    hyp = DocumentExtraction(
        document_id="doc1",
        schema_name="schema1",
        schema_version="1.0.0",
        fields=fields_hyp,
    )

    score1 = structured_field_scores(ref, hyp)
    for _ in range(10):
        score2 = structured_field_scores(ref, hyp)
        assert score1.coordinate_iou == pytest.approx(score2.coordinate_iou)
        assert score1.coordinate_iou == score2.coordinate_iou


def test_scoring_determinism():
    """Ensure scoring performance_values aren't affected by dict key iteration order."""

    docs = []
    gt_docs = []
    samples = []
    extractions = []

    # Create 100 documents to ensure set iteration order would normally be unstable
    for i in range(100):
        doc_id = f"doc_{i}"
        page_id = f"page_{i}"

        checksum = Checksum(algorithm="sha256", value="a" * 64)

        manifest_page = ManifestPage(
            page_id=page_id,
            page_index=0,
            checksum=checksum,
            width=1000,
            height=1000,
            source_kind="image",
        )
        docs.append(
            ManifestDocument(
                document_id=doc_id,
                split="dev",
                category="cat1",
                checksum=checksum,
                source=ManifestSource(
                    kind="synthetic",
                    title="t",
                    license_status="unknown",
                    redistribution="synthetic",
                ),
                pages=[manifest_page],
            )
        )

        page_content = PageContent(
            page_id=page_id,
            page_index=0,
            checksum=checksum,
            width=1000,
            height=1000,
            provenance=Provenance(kind="system_prediction", source="test"),
        )
        gt_docs.append(GroundTruthDocument(document_id=doc_id, pages=[page_content]))
        samples.append(
            PredictionSample(
                document_id=doc_id, page_id=page_id, status="success", page=page_content
            )
        )

        extractions.append(
            DocumentExtraction(
                document_id=doc_id,
                schema_name="s",
                schema_version="1.0.0",
                fields=[
                    create_field(f"f1_{i}", "f1", "val1", float(i % 10)),
                    create_field(f"f2_{i}", "f2", "val2", float(i % 10)),
                ],
            )
        )

    dataset_rev = DatasetRevision(dataset_id="test", revision="1")
    coord_sys = CoordinateSystem()
    manifest = Manifest(dataset_revision=dataset_rev, coordinate_system=coord_sys, documents=docs)
    gt = GroundTruth(
        dataset_revision=dataset_rev,
        coordinate_system=coord_sys,
        annotation_provenance=Provenance(kind="human_annotation", source="t"),
        documents=gt_docs,
    )

    predictions = Predictions(
        dataset_revision=dataset_rev,
        coordinate_system=coord_sys,
        system=SystemMetadata(
            name="sys",
            version="1",
            adapter_name="ad",
            adapter_version="1",
            execution_provider="cpu",
        ),
        run=RunMetadata(run_id="run1"),
        samples=samples,
        document_extractions=extractions,
    )

    for i, gt_doc in enumerate(gt.documents):
        gt_doc.extractions = [extractions[i]]

    report1 = score(manifest, gt, predictions)
    for _ in range(3):
        report2 = score(manifest, gt, predictions)
        for metric_name in report1.metrics:
            assert report1.metrics[metric_name].micro == report2.metrics[metric_name].micro
            assert report1.metrics[metric_name].macro == report2.metrics[metric_name].macro
