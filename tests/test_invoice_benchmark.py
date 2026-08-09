from pathlib import Path

import pytest

from dz_bench.invoice_synthetic import generate_invoice_corpus
from dz_bench.io import load_asset_index, load_ground_truth, load_manifest


def _font() -> Path | None:
    for path in (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if path.is_file():
            return path
    return None


def test_invoice_corpus_has_complete_original_structured_ground_truth() -> None:
    corpus = generate_invoice_corpus(seed=23)

    assert len(corpus.manifest.documents) == 4
    extraction = corpus.ground_truth.documents[0].extractions[0]
    names = {field.field_name for field in extraction.fields}
    assert {
        "supplier_company",
        "buyer_company",
        "invoice_date",
        "invoice_number",
        "nif",
        "nis",
        "rc",
        "line_items[0].description",
        "line_items[0].quantity",
        "line_items[0].unit_price_ht",
        "line_items[0].line_total_ht",
        "total_ht",
        "tva_rate",
        "tva_amount",
        "total_ttc",
        "currency",
    } <= names
    assert all(field.page_id and field.bbox for field in extraction.fields)
    assert all(document.category == "invoice-dz" for document in corpus.manifest.documents)


def test_invoice_png_bundle_is_deterministic_and_contract_valid(tmp_path: Path) -> None:
    font = _font()
    if font is None:
        pytest.skip("no external test font available")
    from dz_bench.raster import write_invoice_images

    first = write_invoice_images(tmp_path / "first", font, seed=23)
    second = write_invoice_images(tmp_path / "second", font, seed=23)

    manifest = load_manifest(first["manifest"])
    truth = load_ground_truth(first["ground_truth"])
    assets = load_asset_index(first["assets"])
    assert len(assets.assets) == 4
    assert truth.documents[0].extractions
    assert [asset.checksum for asset in assets.assets] == [
        asset.checksum for asset in load_asset_index(second["assets"]).assets
    ]
    assert manifest.dataset_revision.dataset_id == "invoice-dz-synthetic-images"
