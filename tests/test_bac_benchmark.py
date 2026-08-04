import hashlib
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dz_bench.cli import app
from dz_bench.metrics import equation_text_page_score, layout_scores, table_structure_page_score
from dz_bench.models import Predictions, PredictionSample, RunMetadata, SystemMetadata
from dz_bench.raster import write_bac_images
from dz_bench.scoring import score
from dz_bench.synthetic import generate_bac_corpus


def _test_font() -> Path | None:
    candidates = [
        os.environ.get("DZ_BENCH_TEST_FONT"),
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    return next(
        (Path(candidate) for candidate in candidates if candidate and Path(candidate).is_file()),
        None,
    )


def _predictions(corpus, *, measured: bool = False) -> Predictions:
    return Predictions(
        dataset_revision=corpus.manifest.dataset_revision,
        coordinate_system=corpus.manifest.coordinate_system,
        system=SystemMetadata(
            name="synthetic-reference",
            version="0.2.0",
            adapter_name="public-fixture",
            adapter_version="1.0.0",
            execution_provider="cpu",
        ),
        run=RunMetadata(run_id="bac-synthetic-test"),
        samples=[
            PredictionSample(
                document_id=document.document_id,
                page_id=page.page_id,
                status="success",
                page=page,
                runtime_ms=12.5 if measured else None,
                peak_memory_mb=42.0 if measured else None,
            )
            for document in corpus.ground_truth.documents
            for page in document.pages
        ],
    )


def test_bac_synthetic_is_deterministic_and_records_only() -> None:
    first = generate_bac_corpus(seed=9)
    second = generate_bac_corpus(seed=9)

    assert first.manifest.model_dump(mode="json") == second.manifest.model_dump(mode="json")
    assert first.ground_truth.model_dump(mode="json") == second.ground_truth.model_dump(mode="json")
    assert first.records == second.records
    assert {
        page.source_kind for document in first.manifest.documents for page in document.pages
    } == {"synthetic_record"}
    assert all(record["asset_format"] == "record-only" for record in first.records)


def test_bac_synthetic_covers_taxonomy_and_structures() -> None:
    corpus = generate_bac_corpus(seed=3)
    tags = {
        tag
        for document in corpus.manifest.documents
        for page in document.pages
        for tag in page.tags
    }
    assert {"arabic", "french", "mixed", "mathematics", "physics"} <= tags
    assert {"equations", "tables", "diagrams", "reading-order-bidi"} <= tags
    assert {
        "scan-quality-clean",
        "scan-quality-compressed",
        "scan-quality-skewed",
        "scan-quality-photographed",
    } <= tags

    blocks = [
        block
        for document in corpus.ground_truth.documents
        for page in document.pages
        for block in page.blocks
    ]
    assert any(block.block_type == "equation" and block.equation_text for block in blocks)
    tables = [block.table for block in blocks if block.block_type == "table"]
    assert tables and all(table is not None and table.cells for table in tables)
    assert any(block.block_type == "diagram" for block in blocks)


def test_bac_scoring_reports_layout_equation_table_and_performance() -> None:
    corpus = generate_bac_corpus(seed=5)
    report = score(corpus.manifest, corpus.ground_truth, _predictions(corpus, measured=True))

    for metric_name in (
        "block_type_accuracy",
        "layout_bbox_iou",
        "layout_match_f1",
        "equation_text_similarity",
        "table_structure_similarity",
    ):
        assert report.metrics[metric_name].micro == 1.0
    assert report.metrics["runtime_ms"].micro == 12.5
    assert report.metrics["peak_memory_mb"].micro == 42.0
    assert report.metrics["runtime_ms"].sample_count == report.summary.total_pages


def test_bac_cli_writes_contract_and_records(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        ["bac-synthetic", "--output-dir", str(tmp_path), "--seed", "7"],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "ground-truth.json").is_file()
    assert (tmp_path / "records.json").is_file()


def test_structured_metrics_penalize_layout_equation_and_table_mismatches() -> None:
    corpus = generate_bac_corpus(seed=13)
    page = next(
        page
        for document in corpus.ground_truth.documents
        for page in document.pages
        if "mathematics"
        in next(
            item for item in corpus.manifest.documents if item.document_id == document.document_id
        )
        .pages[0]
        .tags
        and any(block.block_type == "equation" for block in page.blocks)
    )
    hypothesis = page.model_copy(deep=True)
    hypothesis.blocks[0].block_type = "paragraph"
    equation = next(block for block in hypothesis.blocks if block.block_type == "equation")
    equation.equation_text = "f(x) = x + 1"
    table = next(block for block in hypothesis.blocks if block.block_type == "table")
    assert table.table is not None
    table.table = table.table.model_copy(update={"rows": table.table.rows + 1}, deep=True)

    layout = layout_scores(page.blocks, hypothesis.blocks)
    equation_score, _ = equation_text_page_score(page.blocks, hypothesis.blocks)
    table_score, _ = table_structure_page_score(page.blocks, hypothesis.blocks)
    assert layout.correct_type < layout.block_count
    assert equation_score is not None and equation_score < 1.0
    assert table_score == 0.5


def test_bac_images_write_real_png_checksums_and_preserve_geometry(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    pytest.importorskip("arabic_reshaper")
    pytest.importorskip("bidi")
    font = _test_font()
    if font is None:
        pytest.skip("no test font available; the project intentionally bundles none")

    outputs = write_bac_images(tmp_path / "first", font, seed=19)
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    ground_truth = json.loads(outputs["ground_truth"].read_text(encoding="utf-8"))
    records = json.loads(outputs["records"].read_text(encoding="utf-8"))
    record_by_page = {record["page_id"]: record for record in records}
    truth_by_page = {
        page["page_id"]: page
        for document in ground_truth["documents"]
        for page in document["pages"]
    }

    assert manifest["dataset_revision"]["dataset_id"] == "bac-synthetic-images"
    assert manifest["dataset_revision"]["revision"] == "0.3.0"
    assert len(records) == 8
    assert len({record["scan_quality"] for record in records}) == 4
    for document in manifest["documents"]:
        page = document["pages"][0]
        record = record_by_page[page["page_id"]]
        truth_page = truth_by_page[page["page_id"]]
        image_path = Path(record["image_path"])
        assert not image_path.is_absolute()
        image_bytes = (tmp_path / "first" / image_path).read_bytes()
        checksum = hashlib.sha256(image_bytes).hexdigest()
        assert checksum == page["checksum"]["value"] == truth_page["checksum"]["value"]
        assert page["source_kind"] == "image"
        assert "raster-png" in page["tags"]
        assert record["asset_format"] == "png"
        assert record["image_sha256"] == checksum
        assert record["image_size_bytes"] == len(image_bytes)
        assert record["width"] == truth_page["width"] == 1200
        assert record["height"] == truth_page["height"] == 1600
        assert (tmp_path / "first" / image_path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    original = generate_bac_corpus(seed=19)
    original_boxes = {
        block.block_id: block.bbox.model_dump(mode="json")
        for document in original.ground_truth.documents
        for page in document.pages
        for block in page.blocks
    }
    rendered_boxes = {
        block["block_id"]: block["bbox"]
        for page in truth_by_page.values()
        for block in page["blocks"]
    }
    assert rendered_boxes == original_boxes

    second = write_bac_images(tmp_path / "second", font, seed=19)
    second_manifest = json.loads(second["manifest"].read_text(encoding="utf-8"))
    assert [
        page["checksum"]["value"]
        for document in manifest["documents"]
        for page in document["pages"]
    ] == [
        page["checksum"]["value"]
        for document in second_manifest["documents"]
        for page in document["pages"]
    ]


def test_reference_only_bac_records_have_unique_ids() -> None:
    path = Path("datasets/bac/manifests/bac-reference-only-v0.1.json")
    records = json.loads(path.read_text(encoding="utf-8"))["reference_sources"]
    identifiers = [record["reference_id"] for record in records]
    assert len(identifiers) == len(set(identifiers))


def test_bac_images_cli_requires_font_and_writes_bundle(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    pytest.importorskip("arabic_reshaper")
    pytest.importorskip("bidi")
    font = _test_font()
    if font is None:
        pytest.skip("no test font available; the project intentionally bundles none")

    result = CliRunner().invoke(
        app,
        ["bac-images", "--font", str(font), "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "ground-truth.json").is_file()
    assert (tmp_path / "records.json").is_file()
    assert len(list((tmp_path / "images").glob("*.png"))) == 8
