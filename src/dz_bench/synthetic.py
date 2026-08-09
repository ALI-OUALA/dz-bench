"""Original, image-free synthetic corpus for deterministic Phase A smoke tests."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .io import write_json
from .models import (
    SCHEMA_VERSION,
    Block,
    BlockType,
    BoundingBox,
    Checksum,
    Confidence,
    CoordinateSystem,
    DatasetRevision,
    GroundTruth,
    GroundTruthDocument,
    LanguageTag,
    Manifest,
    ManifestDocument,
    ManifestPage,
    ManifestSource,
    PageContent,
    Provenance,
    ScriptTag,
    TableCell,
    TableStructure,
    TextDirection,
    TextLine,
    TextSpan,
)

WIDTH = 1200
HEIGHT = 1600

TEXT_BANK = {
    "arabic": [
        "أجب عن الأسئلة التالية بدقة.",
        "يمتلك التلميذ ثلاث وحدات من الكتاب.",
        "احسب قيمة المبلغ الإجمالي مع احترام الترتيب.",
        "التمرين الأول: دراسة دالة بسيطة.",
    ],
    "french": [
        "Répondez aux questions suivantes avec précision.",
        "Le montant total est de 1250,50 DA.",
        "Exercice 2 : calculer la vitesse moyenne.",
        "La date de livraison est fixée au 12/06/2026.",
    ],
    "mixed": [
        "المبلغ الإجمالي Total: 1 250,50 DA.",
        "التمرين Exercice 3 — الوثيقة Document A.",
        "رقم الملف DZD-2026-0042 / الصفحة 2.",
        "La réponse النهائية doit garder l'ordre des chiffres.",
    ],
    "bac-style": [
        "اختبار تجريبي: التمرين 1 — الدالة f(x)=x²−2x+1.",
        "Consigne originale : justifier la réponse et indiquer l'unité.",
        "السؤال الثاني: قارن بين النتيجتين في جدول صغير.",
        "La valeur mesurée est 9,81 m/s² dans cet exemple original.",
    ],
}


# Original BAC-like scenarios. They are records only: no protected exam wording,
# diagrams, images, or PDFs are copied into the repository.
BAC_SCENARIOS: tuple[dict[str, object], ...] = (
    {
        "category": "arabic",
        "scan_quality": "clean",
        "tags": ("arabic", "reading-order-rtl", "scan-quality-clean"),
        "blocks": (
            {
                "block_type": "title",
                "text": "البكالوريا التجريبية — اللغة العربية ({code})",
                "x": 100,
                "y": 100,
                "width": 1000,
                "height": 90,
            },
            {
                "block_type": "instruction",
                "text": "أجب عن الأسئلة التالية مع احترام ترتيب الفقرات.",
                "x": 100,
                "y": 230,
                "width": 1000,
                "height": 90,
            },
            {
                "block_type": "exercise",
                "text": "التمرين الأول: استخرج الفكرة الأساسية من النص القصير.",
                "x": 100,
                "y": 380,
                "width": 1000,
                "height": 120,
            },
            {
                "block_type": "paragraph",
                "text": "تساعد القراءة الدقيقة على بناء إجابة واضحة ومترابطة.",
                "x": 100,
                "y": 560,
                "width": 1000,
                "height": 100,
            },
            {
                "block_type": "page_number",
                "text": "الصفحة 1",
                "x": 980,
                "y": 1450,
                "width": 120,
                "height": 50,
            },
        ),
    },
    {
        "category": "arabic",
        "scan_quality": "compressed",
        "tags": ("arabic", "reading-order-rtl", "scan-quality-compressed"),
        "blocks": (
            {
                "block_type": "header",
                "text": "اختبار أصلي للتقييم الآلي — ({code})",
                "x": 100,
                "y": 80,
                "width": 1000,
                "height": 70,
            },
            {
                "block_type": "subquestion",
                "text": "السؤال 2: علل النتيجة بجملة واحدة.",
                "x": 100,
                "y": 220,
                "width": 1000,
                "height": 100,
            },
            {
                "block_type": "list",
                "text": "أ — حدد المصطلح.\nب — قارن بين المثالين.",
                "x": 100,
                "y": 380,
                "width": 1000,
                "height": 150,
            },
            {
                "block_type": "answer_space",
                "text": "مساحة الإجابة",
                "x": 100,
                "y": 620,
                "width": 1000,
                "height": 220,
            },
        ),
    },
    {
        "category": "french",
        "scan_quality": "clean",
        "tags": ("french", "reading-order-ltr", "scan-quality-clean"),
        "blocks": (
            {
                "block_type": "title",
                "text": "Baccalauréat expérimental — français ({code})",
                "x": 100,
                "y": 100,
                "width": 1000,
                "height": 90,
            },
            {
                "block_type": "instruction",
                "text": "Répondez aux questions et conservez les unités.",
                "x": 100,
                "y": 230,
                "width": 1000,
                "height": 90,
            },
            {
                "block_type": "exercise",
                "text": "Exercice 1 : analyser le paragraphe suivant.",
                "x": 100,
                "y": 380,
                "width": 1000,
                "height": 110,
            },
            {
                "block_type": "paragraph",
                "text": "La méthode proposée doit rester claire, vérifiable et concise.",
                "x": 100,
                "y": 550,
                "width": 1000,
                "height": 100,
            },
        ),
    },
    {
        "category": "french",
        "scan_quality": "skewed",
        "tags": ("french", "reading-order-ltr", "scan-quality-skewed", "tables"),
        "blocks": (
            {
                "block_type": "instruction",
                "text": "Complétez le tableau puis justifiez votre choix.",
                "x": 100,
                "y": 120,
                "width": 1000,
                "height": 90,
            },
            {
                "block_type": "table",
                "text": "Grandeur | Unité | Valeur\nDistance | km | 12,5",
                "x": 100,
                "y": 280,
                "width": 1000,
                "height": 260,
                "table": {
                    "rows": 2,
                    "columns": 3,
                    "cells": (
                        {"row": 0, "column": 0, "text": "Grandeur"},
                        {"row": 0, "column": 1, "text": "Unité"},
                        {"row": 0, "column": 2, "text": "Valeur"},
                        {"row": 1, "column": 0, "text": "Distance"},
                        {"row": 1, "column": 1, "text": "km"},
                        {"row": 1, "column": 2, "text": "12,5"},
                    ),
                },
            },
            {
                "block_type": "caption",
                "text": "Tableau 1 — mesure originale.",
                "x": 100,
                "y": 580,
                "width": 1000,
                "height": 70,
            },
        ),
    },
    {
        "category": "mixed",
        "scan_quality": "compressed",
        "tags": ("mixed", "reading-order-bidi", "scan-quality-compressed", "tables"),
        "blocks": (
            {
                "block_type": "title",
                "text": "الوثيقة Document A — résultat ({code})",
                "x": 100,
                "y": 100,
                "width": 1000,
                "height": 90,
            },
            {
                "block_type": "paragraph",
                "text": "المبلغ الإجمالي Total: 1 250,50 DA.",
                "x": 100,
                "y": 240,
                "width": 1000,
                "height": 100,
            },
            {
                "block_type": "table",
                "text": "البند | Article | الكمية\nدفتر | Cahier | 3",
                "x": 100,
                "y": 400,
                "width": 1000,
                "height": 260,
                "table": {
                    "rows": 2,
                    "columns": 3,
                    "cells": (
                        {"row": 0, "column": 0, "text": "البند"},
                        {"row": 0, "column": 1, "text": "Article"},
                        {"row": 0, "column": 2, "text": "الكمية"},
                        {"row": 1, "column": 0, "text": "دفتر"},
                        {"row": 1, "column": 1, "text": "Cahier"},
                        {"row": 1, "column": 2, "text": "3"},
                    ),
                },
            },
        ),
    },
    {
        "category": "mathematics",
        "scan_quality": "clean",
        "tags": ("mathematics", "equations", "tables", "reading-order-ltr", "scan-quality-clean"),
        "blocks": (
            {
                "block_type": "exercise",
                "text": "Exercice 2 — fonction polynomiale ({code})",
                "x": 100,
                "y": 100,
                "width": 1000,
                "height": 100,
            },
            {
                "block_type": "equation",
                "text": "f(x) = x² − 2x + 1",
                "equation_text": "f(x) = x² − 2x + 1",
                "x": 100,
                "y": 260,
                "width": 1000,
                "height": 130,
            },
            {
                "block_type": "subquestion",
                "text": "Déterminer la valeur de f(3) et vérifier le résultat.",
                "x": 100,
                "y": 460,
                "width": 1000,
                "height": 90,
            },
            {
                "block_type": "table",
                "text": "x | 0 | 1\nf(x) | 1 | 0",
                "x": 100,
                "y": 620,
                "width": 1000,
                "height": 260,
                "table": {
                    "rows": 2,
                    "columns": 3,
                    "cells": (
                        {"row": 0, "column": 0, "text": "x"},
                        {"row": 0, "column": 1, "text": "0"},
                        {"row": 0, "column": 2, "text": "1"},
                        {"row": 1, "column": 0, "text": "f(x)"},
                        {"row": 1, "column": 1, "text": "1"},
                        {"row": 1, "column": 2, "text": "0"},
                    ),
                },
            },
        ),
    },
    {
        "category": "mathematics",
        "scan_quality": "skewed",
        "tags": (
            "mathematics",
            "equations",
            "diagrams",
            "reading-order-ltr",
            "scan-quality-skewed",
        ),
        "blocks": (
            {
                "block_type": "instruction",
                "text": "Construire une figure puis interpréter la relation.",
                "x": 100,
                "y": 100,
                "width": 1000,
                "height": 90,
            },
            {
                "block_type": "equation",
                "text": "A = b × h / 2",
                "equation_text": "A = b × h / 2",
                "x": 100,
                "y": 240,
                "width": 1000,
                "height": 120,
            },
            {
                "block_type": "diagram",
                "text": None,
                "x": 180,
                "y": 430,
                "width": 700,
                "height": 500,
            },
            {
                "block_type": "caption",
                "text": "Figure 1 — triangle original.",
                "x": 100,
                "y": 980,
                "width": 1000,
                "height": 70,
            },
        ),
    },
    {
        "category": "physics",
        "scan_quality": "photographed",
        "tags": (
            "physics",
            "equations",
            "tables",
            "diagrams",
            "reading-order-ltr",
            "scan-quality-photographed",
        ),
        "blocks": (
            {
                "block_type": "title",
                "text": "Physique — mouvement rectiligne ({code})",
                "x": 100,
                "y": 100,
                "width": 1000,
                "height": 90,
            },
            {
                "block_type": "equation",
                "text": "v = Δx / Δt",
                "equation_text": "v = Δx / Δt",
                "x": 100,
                "y": 240,
                "width": 1000,
                "height": 120,
            },
            {
                "block_type": "table",
                "text": "t (s) | 0 | 2\nx (m) | 0 | 9,81",
                "x": 100,
                "y": 420,
                "width": 1000,
                "height": 260,
                "table": {
                    "rows": 2,
                    "columns": 3,
                    "cells": (
                        {"row": 0, "column": 0, "text": "t (s)"},
                        {"row": 0, "column": 1, "text": "0"},
                        {"row": 0, "column": 2, "text": "2"},
                        {"row": 1, "column": 0, "text": "x (m)"},
                        {"row": 1, "column": 1, "text": "0"},
                        {"row": 1, "column": 2, "text": "9,81"},
                    ),
                },
            },
            {
                "block_type": "diagram",
                "text": None,
                "x": 150,
                "y": 800,
                "width": 800,
                "height": 400,
            },
            {
                "block_type": "caption",
                "text": "Schéma 1 — trajectoire et repère.",
                "x": 100,
                "y": 1240,
                "width": 1000,
                "height": 70,
            },
        ),
    },
)


@dataclass(frozen=True, slots=True)
class SyntheticCorpus:
    manifest: Manifest
    ground_truth: GroundTruth
    records: list[dict[str, object]]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _checksum(value: object) -> Checksum:
    return Checksum(value=hashlib.sha256(_canonical_bytes(value)).hexdigest())


def _text_metadata(category: str) -> tuple[LanguageTag, ScriptTag, TextDirection]:
    if category == "arabic":
        return "ar", "arabic", "rtl"
    if category == "french":
        return "fr", "latin", "ltr"
    return "mixed", "mixed", "mixed"


def _page_content(
    document_id: str,
    page_id: str,
    page_index: int,
    category: str,
    lines: list[str],
    seed: int,
    checksum: Checksum,
) -> PageContent:
    language, script, direction = _text_metadata(category)
    provenance = Provenance(
        kind="synthetic_generator",
        source="dz-bench.synthetic",
        version=SCHEMA_VERSION,
        stage="generation",
        details={"seed": seed, "category": category},
    )
    confidence = Confidence(score=1.0, calibrated=True, method="authored_ground_truth")
    blocks: list[Block] = []
    reading_order: list[str] = []
    for index, text in enumerate(lines):
        block_id = f"{page_id}-b{index + 1}"
        line_id = f"{page_id}-l{index + 1}"
        bbox = BoundingBox(x=100, y=150 + index * 180, width=1000, height=100)
        span = TextSpan(
            span_id=f"{line_id}-s1",
            raw_text=text,
            normalized_text=text,
            search_text=text,
            language=language,
            script=script,
            direction=direction,
            bbox=bbox,
            confidence=confidence,
            provenance=provenance,
        )
        line = TextLine(
            line_id=line_id,
            raw_text=text,
            normalized_text=text,
            search_text=text,
            language=language,
            script=script,
            direction=direction,
            bbox=bbox,
            reading_order_index=index,
            confidence=confidence,
            provenance=provenance,
            spans=[span],
        )
        blocks.append(
            Block(
                block_id=block_id,
                block_type="exercise" if index == 0 else "paragraph",
                bbox=bbox,
                reading_order_index=index,
                confidence=confidence,
                provenance=provenance,
                lines=[line],
            )
        )
        reading_order.append(line_id)
    return PageContent(
        page_id=page_id,
        page_index=page_index,
        checksum=checksum,
        width=WIDTH,
        height=HEIGHT,
        blocks=blocks,
        reading_order=reading_order,
        provenance=provenance,
    )


def generate_corpus(
    seed: int = 17,
    document_count: int = 4,
    pages_per_document: int = 1,
) -> SyntheticCorpus:
    if document_count < 1 or pages_per_document < 1:
        raise ValueError("document_count and pages_per_document must be positive")
    rng = random.Random(seed)
    categories = list(TEXT_BANK)
    documents: list[ManifestDocument] = []
    ground_truth_documents: list[GroundTruthDocument] = []
    records: list[dict[str, object]] = []
    for document_index in range(document_count):
        category = categories[(document_index + rng.randrange(len(categories))) % len(categories)]
        document_id = f"synthetic-{document_index + 1:03d}"
        manifest_pages: list[ManifestPage] = []
        truth_pages: list[PageContent] = []
        page_checksums: list[str] = []
        for page_index in range(pages_per_document):
            page_id = f"{document_id}-p{page_index + 1:02d}"
            source_lines = list(TEXT_BANK[category])
            rng.shuffle(source_lines)
            lines = [
                source_lines[0],
                source_lines[1],
                f"Référence synthétique {rng.randint(100, 999)} — page {page_index + 1}.",
            ]
            record = {
                "document_id": document_id,
                "page_id": page_id,
                "page_index": page_index,
                "category": category,
                "width": WIDTH,
                "height": HEIGHT,
                "lines": lines,
            }
            page_checksum = _checksum(record)
            page = _page_content(
                document_id, page_id, page_index, category, lines, seed, page_checksum
            )
            manifest_pages.append(
                ManifestPage(
                    page_id=page_id,
                    page_index=page_index,
                    checksum=page_checksum,
                    width=WIDTH,
                    height=HEIGHT,
                    source_kind="synthetic_record",
                    tags=[category, "original-content", "no-image-dependency"],
                )
            )
            truth_pages.append(page)
            page_checksums.append(page_checksum.value)
            records.append(record)
        document_checksum = _checksum({"document_id": document_id, "pages": page_checksums})
        source = ManifestSource(
            kind="synthetic",
            title="Original DZ-Bench synthetic document",
            license_status="not_applicable",
            redistribution="synthetic",
            notes="Authored benchmark text; not copied from a BAC paper.",
        )
        documents.append(
            ManifestDocument(
                document_id=document_id,
                split="dev",
                category=category,
                checksum=document_checksum,
                source=source,
                pages=manifest_pages,
            )
        )
        ground_truth_documents.append(
            GroundTruthDocument(document_id=document_id, pages=truth_pages)
        )
    revision = DatasetRevision(dataset_id="synthetic-phase-a", revision="0.1.0")
    coordinate_system = CoordinateSystem()
    return SyntheticCorpus(
        manifest=Manifest(
            dataset_revision=revision,
            coordinate_system=coordinate_system,
            documents=documents,
            notes=[
                "Original authored Arabic, French, mixed-script, and BAC-style-ish text.",
                "No protected exam pages or copied BAC ground truth are included.",
            ],
        ),
        ground_truth=GroundTruth(
            dataset_revision=revision,
            coordinate_system=coordinate_system,
            annotation_provenance=Provenance(
                kind="synthetic_generator",
                source="dz-bench.synthetic",
                version=SCHEMA_VERSION,
                stage="generation",
                details={"seed": seed},
            ),
            documents=ground_truth_documents,
        ),
        records=records,
    )


def write_corpus(
    output_dir: str | Path,
    seed: int = 17,
    document_count: int = 4,
    pages_per_document: int = 1,
    include_records: bool = False,
) -> dict[str, Path]:
    corpus = generate_corpus(seed, document_count, pages_per_document)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "manifest": write_json(corpus.manifest, directory / "manifest.json"),
        "ground_truth": write_json(corpus.ground_truth, directory / "ground-truth.json"),
    }
    if include_records:
        outputs["records"] = write_json(corpus.records, directory / "records.json")
    return outputs


def _bac_text(value: object, code: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("BAC block text must be a string or null")
    return value.format(code=code)


def _bac_number(spec: dict[str, object], name: str, default: float) -> float:
    value = spec.get(name, default)
    if not isinstance(value, (int, float)):
        raise ValueError(f"BAC block {name} must be numeric")
    return float(value)


def _bac_bbox(spec: dict[str, object]) -> BoundingBox:
    return BoundingBox(
        x=_bac_number(spec, "x", 100),
        y=_bac_number(spec, "y", 100),
        width=_bac_number(spec, "width", 1000),
        height=_bac_number(spec, "height", 90),
    )


def _bac_table(block_id: str, bbox: BoundingBox, value: object, code: int) -> TableStructure | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("BAC table metadata must be an object")
    rows = value.get("rows")
    columns = value.get("columns")
    cell_specs = value.get("cells")
    if (
        not isinstance(rows, int)
        or not isinstance(columns, int)
        or not isinstance(cell_specs, tuple)
    ):
        raise ValueError("BAC table metadata needs integer rows, columns, and tuple cells")
    cells: list[TableCell] = []
    cell_width = bbox.width / columns
    cell_height = bbox.height / rows
    for cell_index, cell_spec in enumerate(cell_specs, start=1):
        if not isinstance(cell_spec, dict):
            raise ValueError("BAC table cells must be objects")
        row = cell_spec.get("row")
        column = cell_spec.get("column")
        if not isinstance(row, int) or not isinstance(column, int):
            raise ValueError("BAC table cell row and column must be integers")
        row_span = cell_spec.get("row_span", 1)
        column_span = cell_spec.get("column_span", 1)
        if not isinstance(row_span, int) or not isinstance(column_span, int):
            raise ValueError("BAC table cell spans must be integers")
        text = _bac_text(cell_spec.get("text", ""), code) or ""
        cells.append(
            TableCell(
                cell_id=f"{block_id}-c{cell_index}",
                row_index=row,
                column_index=column,
                row_span=row_span,
                column_span=column_span,
                bbox=BoundingBox(
                    x=bbox.x + column * cell_width,
                    y=bbox.y + row * cell_height,
                    width=cell_width * column_span,
                    height=cell_height * row_span,
                ),
                raw_text=text,
                normalized_text=text,
            )
        )
    return TableStructure(rows=rows, columns=columns, cells=cells)


def _bac_page_content(
    document_id: str,
    page_id: str,
    category: str,
    scenario: dict[str, object],
    seed: int,
    code: int,
    checksum: Checksum,
) -> PageContent:
    language, script, direction = _text_metadata(category)
    scan_quality = scenario["scan_quality"]
    provenance = Provenance(
        kind="synthetic_generator",
        source="dz-bench.synthetic.bac",
        version=SCHEMA_VERSION,
        stage="generation",
        details={
            "seed": seed,
            "category": category,
            "scan_quality": str(scan_quality),
            "asset_format": "record-only",
        },
    )
    confidence = Confidence(score=1.0, calibrated=True, method="authored_ground_truth")
    blocks: list[Block] = []
    reading_order: list[str] = []
    raw_specs = scenario["blocks"]
    if not isinstance(raw_specs, tuple):
        raise ValueError("BAC scenario blocks must be a tuple")
    for index, raw_spec in enumerate(raw_specs):
        if not isinstance(raw_spec, dict):
            raise ValueError("BAC scenario block must be an object")
        block_id = f"{page_id}-b{index + 1}"
        line_id = f"{block_id}-l1"
        block_type = raw_spec.get("block_type")
        if not isinstance(block_type, str):
            raise ValueError("BAC block type must be a string")
        typed_block_type = cast(BlockType, block_type)
        bbox = _bac_bbox(raw_spec)
        text = _bac_text(raw_spec.get("text"), code)
        equation_text = _bac_text(raw_spec.get("equation_text"), code)
        table = _bac_table(block_id, bbox, raw_spec.get("table"), code)
        lines: list[TextLine] = []
        if text is not None:
            span = TextSpan(
                span_id=f"{line_id}-s1",
                raw_text=text,
                normalized_text=text,
                search_text=text,
                language=language,
                script=script,
                direction=direction,
                bbox=bbox,
                confidence=confidence,
                provenance=provenance,
            )
            lines.append(
                TextLine(
                    line_id=line_id,
                    raw_text=text,
                    normalized_text=text,
                    search_text=text,
                    language=language,
                    script=script,
                    direction=direction,
                    bbox=bbox,
                    reading_order_index=index,
                    confidence=confidence,
                    provenance=provenance,
                    spans=[span],
                )
            )
        blocks.append(
            Block(
                block_id=block_id,
                block_type=typed_block_type,
                bbox=bbox,
                reading_order_index=index,
                confidence=confidence,
                provenance=provenance,
                lines=lines,
                equation_text=equation_text,
                table=table,
            )
        )
        reading_order.append(block_id)
    return PageContent(
        page_id=page_id,
        page_index=0,
        checksum=checksum,
        width=WIDTH,
        height=HEIGHT,
        blocks=blocks,
        reading_order=reading_order,
        provenance=provenance,
    )


def _bac_record(
    document_id: str,
    page_id: str,
    category: str,
    scenario: dict[str, object],
    seed: int,
    code: int,
) -> dict[str, object]:
    raw_specs = scenario["blocks"]
    if not isinstance(raw_specs, tuple):
        raise ValueError("BAC scenario blocks must be a tuple")
    block_records: list[dict[str, object]] = []
    for index, raw_spec in enumerate(raw_specs):
        if not isinstance(raw_spec, dict):
            raise ValueError("BAC scenario block must be an object")
        block_id = f"{page_id}-b{index + 1}"
        bbox = _bac_bbox(raw_spec)
        table = _bac_table(block_id, bbox, raw_spec.get("table"), code)
        block_records.append(
            {
                "block_id": block_id,
                "block_type": raw_spec.get("block_type"),
                "bbox": bbox.model_dump(mode="json"),
                "text": _bac_text(raw_spec.get("text"), code),
                "equation_text": _bac_text(raw_spec.get("equation_text"), code),
                "table": table.model_dump(mode="json") if table else None,
            }
        )
    return {
        "document_id": document_id,
        "page_id": page_id,
        "page_index": 0,
        "category": category,
        "tags": [
            "bac-style",
            "original-content",
            "source-record-only",
            *cast(tuple[str, ...], scenario["tags"]),
        ],
        "scan_quality": scenario["scan_quality"],
        "asset_format": "record-only",
        "seed": seed,
        "blocks": block_records,
    }


def generate_bac_corpus(seed: int = 17, repeats: int = 1) -> SyntheticCorpus:
    """Generate an original BAC-like records-only corpus with layout annotations."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    rng = random.Random(seed)
    documents: list[ManifestDocument] = []
    ground_truth_documents: list[GroundTruthDocument] = []
    records: list[dict[str, object]] = []
    source = ManifestSource(
        kind="synthetic",
        title="Original DZ-Bench BAC-like synthetic record",
        license_status="not_applicable",
        redistribution="synthetic",
        notes="Original benchmark-authored content; no BAC paper, image, or PDF is copied.",
    )
    for document_index in range(repeats * len(BAC_SCENARIOS)):
        scenario = BAC_SCENARIOS[document_index % len(BAC_SCENARIOS)]
        category = scenario.get("category")
        if not isinstance(category, str):
            raise ValueError("BAC scenario category must be a string")
        document_id = f"bac-synthetic-{document_index + 1:03d}"
        page_id = f"{document_id}-p01"
        code = rng.randint(100, 999)
        record = _bac_record(document_id, page_id, category, scenario, seed, code)
        page_checksum = _checksum(record)
        page = _bac_page_content(
            document_id, page_id, category, scenario, seed, code, page_checksum
        )
        tags = list(dict.fromkeys(cast(list[str], record["tags"])))
        manifest_page = ManifestPage(
            page_id=page_id,
            page_index=0,
            checksum=page_checksum,
            width=WIDTH,
            height=HEIGHT,
            source_kind="synthetic_record",
            tags=tags,
        )
        document_checksum = _checksum({"document_id": document_id, "pages": [page_checksum.value]})
        documents.append(
            ManifestDocument(
                document_id=document_id,
                split="dev",
                category=category,
                checksum=document_checksum,
                source=source,
                pages=[manifest_page],
            )
        )
        ground_truth_documents.append(GroundTruthDocument(document_id=document_id, pages=[page]))
        records.append(record)
    revision = DatasetRevision(dataset_id="bac-synthetic", revision="0.2.0")
    coordinate_system = CoordinateSystem()
    return SyntheticCorpus(
        manifest=Manifest(
            dataset_revision=revision,
            coordinate_system=coordinate_system,
            documents=documents,
            notes=[
                "Original BAC-like Arabic, French, mixed, mathematics, and physics records.",
                "Tags cover equations, tables, diagrams, reading order, and scan-quality variants.",
                "Records only: deterministic generation does not ship image or PDF fixtures.",
            ],
        ),
        ground_truth=GroundTruth(
            dataset_revision=revision,
            coordinate_system=coordinate_system,
            annotation_provenance=Provenance(
                kind="synthetic_generator",
                source="dz-bench.synthetic.bac",
                version=SCHEMA_VERSION,
                stage="generation",
                details={"seed": seed, "profile": "bac-synthetic", "repeats": repeats},
            ),
            documents=ground_truth_documents,
        ),
        records=records,
    )


def write_bac_corpus(output_dir: str | Path, seed: int = 17, repeats: int = 1) -> dict[str, Path]:
    corpus = generate_bac_corpus(seed, repeats)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "manifest": write_json(corpus.manifest, directory / "manifest.json"),
        "ground_truth": write_json(corpus.ground_truth, directory / "ground-truth.json"),
        "records": write_json(corpus.records, directory / "records.json"),
    }
