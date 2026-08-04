"""Original, image-free synthetic corpus for deterministic Phase A smoke tests."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from .io import write_json
from .models import (
    SCHEMA_VERSION,
    Block,
    BoundingBox,
    Checksum,
    Confidence,
    CoordinateSystem,
    DatasetRevision,
    GroundTruth,
    GroundTruthDocument,
    Manifest,
    ManifestDocument,
    ManifestPage,
    ManifestSource,
    PageContent,
    Provenance,
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


def _text_metadata(category: str) -> tuple[str, str, str]:
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
