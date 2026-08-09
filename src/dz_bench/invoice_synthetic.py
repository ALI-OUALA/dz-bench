"""Original Algerian invoice fixtures with document-level structured ground truth."""

from __future__ import annotations

import random
from typing import Literal, cast

from .models import (
    SCHEMA_VERSION,
    BoundingBox,
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
    Provenance,
    StructuredField,
    ValidationResult,
)
from .synthetic import (
    HEIGHT,
    WIDTH,
    SyntheticCorpus,
    _bac_page_content,
    _bac_record,
    _checksum,
)

_QUALITIES = ("clean", "compressed", "photographed", "skewed")
_BLOCKS: tuple[dict[str, object], ...] = (
    {
        "block_type": "header",
        "text": (
            "شركة الأطلس للتوريدات / SARL Atlas Fournitures\n"
            "NIF: 001626089123456  NIS: 001626089123457  RC: 16/00-1234567B12"
        ),
        "x": 60,
        "y": 55,
        "width": 650,
        "height": 170,
    },
    {
        "block_type": "paragraph",
        "text": "Client / الزبون: EURL El Bahdja Distribution",
        "x": 730,
        "y": 70,
        "width": 410,
        "height": 120,
    },
    {
        "block_type": "title",
        "text": "FACTURE / فاتورة N° FA-2026-{code}",
        "x": 310,
        "y": 260,
        "width": 580,
        "height": 90,
    },
    {
        "block_type": "paragraph",
        "text": "Date / التاريخ: 09/08/2026",
        "x": 760,
        "y": 370,
        "width": 380,
        "height": 70,
    },
    {
        "block_type": "table",
        "x": 60,
        "y": 490,
        "width": 1080,
        "height": 430,
        "table": {
            "rows": 3,
            "columns": 4,
            "cells": (
                {"row": 0, "column": 0, "text": "Désignation / البيان"},
                {"row": 0, "column": 1, "text": "Qté"},
                {"row": 0, "column": 2, "text": "P.U. HT"},
                {"row": 0, "column": 3, "text": "Montant HT"},
                {"row": 1, "column": 0, "text": "Ramette papier A4"},
                {"row": 1, "column": 1, "text": "2"},
                {"row": 1, "column": 2, "text": "250,00"},
                {"row": 1, "column": 3, "text": "500,00"},
                {"row": 2, "column": 0, "text": "Cartouche imprimante"},
                {"row": 2, "column": 1, "text": "1"},
                {"row": 2, "column": 2, "text": "500,00"},
                {"row": 2, "column": 3, "text": "500,00"},
            ),
        },
    },
    {
        "block_type": "paragraph",
        "text": "Total HT: 1 000,00 DZD",
        "x": 700,
        "y": 980,
        "width": 440,
        "height": 70,
    },
    {
        "block_type": "paragraph",
        "text": "TVA 19%: 190,00 DZD",
        "x": 700,
        "y": 1060,
        "width": 440,
        "height": 70,
    },
    {
        "block_type": "paragraph",
        "text": "Net à payer / المبلغ الإجمالي TTC: 1 190,00 DZD",
        "x": 560,
        "y": 1140,
        "width": 580,
        "height": 100,
    },
    {
        "block_type": "footer",
        "text": "Document synthétique original — غير صالح للاستعمال التجاري",
        "x": 220,
        "y": 1450,
        "width": 760,
        "height": 60,
    },
)


def _invoice_extraction(document_id: str, page, code: int) -> DocumentExtraction:
    provenance = Provenance(
        kind="synthetic_generator",
        source="dz-bench.synthetic.invoice-dz",
        version=SCHEMA_VERSION,
        stage="structured-ground-truth",
    )
    confidence = Confidence(score=1.0, calibrated=True, method="authored_ground_truth")
    blocks = page.blocks
    table = blocks[4].table
    assert table is not None

    def field(
        name: str,
        value: str,
        value_type: Literal["string", "date", "identifier", "decimal", "currency"],
        bbox,
        index: int,
    ) -> StructuredField:
        return StructuredField(
            field_id=f"{document_id}-f{index:02d}",
            field_name=name,
            value=value,
            normalized_value=value,
            value_type=value_type,
            confidence=confidence,
            page_id=page.page_id,
            bbox=bbox,
            provenance=provenance,
        )

    specs: list[
        tuple[
            str,
            str,
            Literal["string", "date", "identifier", "decimal", "currency"],
            BoundingBox,
        ]
    ] = [
        ("supplier_company", "SARL Atlas Fournitures", "string", blocks[0].bbox),
        ("buyer_company", "EURL El Bahdja Distribution", "string", blocks[1].bbox),
        ("invoice_number", f"FA-2026-{code}", "identifier", blocks[2].bbox),
        ("invoice_date", "2026-08-09", "date", blocks[3].bbox),
        ("nif", "001626089123456", "identifier", blocks[0].bbox),
        ("nis", "001626089123457", "identifier", blocks[0].bbox),
        ("rc", "16/00-1234567B12", "identifier", blocks[0].bbox),
        ("line_items[0].description", "Ramette papier A4", "string", table.cells[4].bbox),
        ("line_items[0].quantity", "2", "decimal", table.cells[5].bbox),
        ("line_items[0].unit_price_ht", "250.00", "decimal", table.cells[6].bbox),
        ("line_items[0].line_total_ht", "500.00", "decimal", table.cells[7].bbox),
        ("line_items[1].description", "Cartouche imprimante", "string", table.cells[8].bbox),
        ("line_items[1].quantity", "1", "decimal", table.cells[9].bbox),
        ("line_items[1].unit_price_ht", "500.00", "decimal", table.cells[10].bbox),
        ("line_items[1].line_total_ht", "500.00", "decimal", table.cells[11].bbox),
        ("total_ht", "1000.00", "decimal", blocks[5].bbox),
        ("tva_rate", "19.00", "decimal", blocks[6].bbox),
        ("tva_amount", "190.00", "decimal", blocks[6].bbox),
        ("total_ttc", "1190.00", "decimal", blocks[7].bbox),
        ("currency", "DZD", "currency", blocks[7].bbox),
    ]
    return DocumentExtraction(
        document_id=document_id,
        schema_name="invoice-dz",
        schema_version="1.0.0",
        fields=[field(*spec, index) for index, spec in enumerate(specs, start=1)],
        validations=[
            ValidationResult(
                rule="line-items-sum-ht",
                status="pass",
                message="Line totals equal total HT.",
                confidence=1.0,
            ),
            ValidationResult(
                rule="invoice-arithmetic",
                status="pass",
                message="HT plus TVA equals TTC.",
                confidence=1.0,
            ),
        ],
    )


def generate_invoice_corpus(seed: int = 17) -> SyntheticCorpus:
    """Generate four original invoices with identical content and controlled scan quality."""

    rng = random.Random(seed)
    documents: list[ManifestDocument] = []
    truth_documents: list[GroundTruthDocument] = []
    records: list[dict[str, object]] = []
    source = ManifestSource(
        kind="synthetic",
        title="Original DZ-Bench Algerian invoice analogue",
        license_status="not_applicable",
        redistribution="synthetic",
        notes="Fictional companies and identifiers; no real invoice is copied.",
    )
    for index, quality in enumerate(_QUALITIES, start=1):
        document_id = f"invoice-dz-synthetic-{index:03d}"
        page_id = f"{document_id}-p01"
        code = rng.randint(1000, 9999)
        scenario: dict[str, object] = {
            "scan_quality": quality,
            "tags": (
                "invoice-dz",
                "arabic-french",
                "financial-values",
                "table",
                f"scan-quality-{quality}",
            ),
            "blocks": _BLOCKS,
        }
        record = _bac_record(document_id, page_id, "invoice-dz", scenario, seed, code)
        page_checksum = _checksum(record)
        page = _bac_page_content(
            document_id, page_id, "invoice-dz", scenario, seed, code, page_checksum
        )
        provenance = Provenance(
            kind="synthetic_generator",
            source="dz-bench.synthetic.invoice-dz",
            version=SCHEMA_VERSION,
            stage="generation",
            details={"seed": seed, "scan_quality": quality, "category": "invoice-dz"},
        )
        page.provenance = provenance
        for block in page.blocks:
            block.provenance = provenance
            for line in block.lines:
                line.provenance = provenance
                for span in line.spans:
                    span.provenance = provenance
        manifest_page = ManifestPage(
            page_id=page_id,
            page_index=0,
            checksum=page_checksum,
            width=WIDTH,
            height=HEIGHT,
            source_kind="synthetic_record",
            tags=cast(list[str], record["tags"]),
        )
        documents.append(
            ManifestDocument(
                document_id=document_id,
                split="dev",
                category="invoice-dz",
                checksum=_checksum({"document_id": document_id, "pages": [page_checksum.value]}),
                source=source,
                pages=[manifest_page],
            )
        )
        truth_documents.append(
            GroundTruthDocument(
                document_id=document_id,
                pages=[page],
                extractions=[_invoice_extraction(document_id, page, code)],
            )
        )
        records.append(record)
    revision = DatasetRevision(dataset_id="invoice-dz-synthetic", revision="0.1.0")
    coordinates = CoordinateSystem()
    return SyntheticCorpus(
        manifest=Manifest(
            dataset_revision=revision,
            coordinate_system=coordinates,
            documents=documents,
            notes=[
                "Original Algerian Arabic-French invoice analogues.",
                "Fictional identifiers and companies; synthetic scores are not "
                "real-document claims.",
            ],
        ),
        ground_truth=GroundTruth(
            dataset_revision=revision,
            coordinate_system=coordinates,
            annotation_provenance=Provenance(
                kind="synthetic_generator",
                source="dz-bench.synthetic.invoice-dz",
                version=SCHEMA_VERSION,
                stage="generation",
                details={"seed": seed, "profile": "invoice-dz"},
            ),
            documents=truth_documents,
        ),
        records=records,
    )


__all__ = ["generate_invoice_corpus"]
