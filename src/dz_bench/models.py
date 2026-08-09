"""Versioned Pydantic models for the public DZ-Bench contract."""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"

Identifier = Annotated[
    str,
    Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
Revision = Annotated[
    str,
    Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
]

LanguageTag = Literal["ar", "fr", "en", "latin", "mixed", "unknown"]
ScriptTag = Literal["arabic", "latin", "mixed", "common", "unknown"]
TextDirection = Literal["rtl", "ltr", "mixed", "unknown"]
BlockType = Literal[
    "title",
    "instruction",
    "exercise",
    "subquestion",
    "paragraph",
    "list",
    "table",
    "equation",
    "figure",
    "diagram",
    "caption",
    "header",
    "footer",
    "page_number",
    "answer_space",
    "stamp",
    "signature",
    "unknown",
]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Checksum(ContractModel):
    algorithm: Literal["sha256"] = "sha256"
    value: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    size_bytes: int | None = Field(default=None, ge=0)


class DatasetRevision(ContractModel):
    dataset_id: Identifier
    revision: Revision
    manifest_checksum: Checksum | None = None


class CoordinateSystem(ContractModel):
    """All coordinates are page pixels, origin top-left, x right and y down."""

    unit: Literal["pixel"] = "pixel"
    origin: Literal["top_left"] = "top_left"
    x_axis: Literal["right"] = "right"
    y_axis: Literal["down"] = "down"


class BoundingBox(ContractModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class TableCell(ContractModel):
    cell_id: Identifier
    row_index: int = Field(ge=0)
    column_index: int = Field(ge=0)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    bbox: BoundingBox
    raw_text: str = ""
    normalized_text: str = ""


class TableStructure(ContractModel):
    rows: int = Field(ge=1)
    columns: int = Field(ge=1)
    cells: list[TableCell] = Field(min_length=1)

    @model_validator(mode="after")
    def cells_fit_grid(self) -> TableStructure:
        cell_ids = [cell.cell_id for cell in self.cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("table cell IDs must be unique")
        for cell in self.cells:
            if cell.row_index + cell.row_span > self.rows:
                raise ValueError(f"table cell {cell.cell_id} exceeds row count")
            if cell.column_index + cell.column_span > self.columns:
                raise ValueError(f"table cell {cell.cell_id} exceeds column count")
        return self


class Confidence(ContractModel):
    score: float = Field(ge=0, le=1)
    calibrated: bool = False
    method: str | None = Field(default=None, min_length=1)


class ProcessingWarning(ContractModel):
    code: Identifier
    message: str = Field(min_length=1)
    severity: Literal["info", "warning", "error"] = "warning"
    stage: str | None = Field(default=None, min_length=1)


class ProcessingError(ContractModel):
    code: Identifier
    message: str = Field(min_length=1)
    stage: str | None = Field(default=None, min_length=1)
    retryable: bool = False


class Provenance(ContractModel):
    kind: Literal[
        "human_annotation",
        "synthetic_generator",
        "system_prediction",
        "native_pdf",
        "ocr",
        "vlm",
        "manual_correction",
        "derived",
    ]
    source: str = Field(min_length=1)
    version: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)
    stage: str | None = Field(default=None, min_length=1)
    details: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RecognitionAlternative(ContractModel):
    raw_text: str
    normalized_text: str
    confidence: Confidence
    provenance: Provenance
    reason: str | None = Field(default=None, min_length=1)


class TextSpan(ContractModel):
    span_id: Identifier
    raw_text: str
    normalized_text: str
    search_text: str | None = None
    language: LanguageTag
    script: ScriptTag
    direction: TextDirection
    bbox: BoundingBox
    confidence: Confidence
    provenance: Provenance
    alternatives: list[RecognitionAlternative] = Field(default_factory=list)
    warnings: list[ProcessingWarning] = Field(default_factory=list)


class TextLine(ContractModel):
    line_id: Identifier
    raw_text: str
    normalized_text: str
    search_text: str | None = None
    language: LanguageTag
    script: ScriptTag
    direction: TextDirection
    bbox: BoundingBox
    reading_order_index: int = Field(ge=0)
    confidence: Confidence
    provenance: Provenance
    spans: list[TextSpan] = Field(min_length=1)
    alternatives: list[RecognitionAlternative] = Field(default_factory=list)
    warnings: list[ProcessingWarning] = Field(default_factory=list)


class Block(ContractModel):
    block_id: Identifier
    block_type: BlockType
    bbox: BoundingBox
    reading_order_index: int = Field(ge=0)
    confidence: Confidence
    provenance: Provenance
    lines: list[TextLine] = Field(default_factory=list)
    equation_text: str | None = None
    table: TableStructure | None = None
    alternatives: list[RecognitionAlternative] = Field(default_factory=list)
    warnings: list[ProcessingWarning] = Field(default_factory=list)


class PageContent(ContractModel):
    page_id: Identifier
    page_index: int = Field(ge=0)
    checksum: Checksum
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    blocks: list[Block] = Field(default_factory=list)
    reading_order: list[Identifier] = Field(default_factory=list)
    provenance: Provenance
    warnings: list[ProcessingWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def reading_order_references_known_units(self) -> PageContent:
        units = {block.block_id for block in self.blocks}
        units.update(line.line_id for block in self.blocks for line in block.lines)
        if len(self.reading_order) != len(set(self.reading_order)):
            raise ValueError("reading_order contains duplicate unit IDs")
        unknown = set(self.reading_order) - units
        if unknown:
            raise ValueError(f"reading_order references unknown units: {sorted(unknown)}")
        return self


class ManifestSource(ContractModel):
    kind: Literal["synthetic", "redistributable", "reference-only", "private-evaluation"]
    title: str = Field(min_length=1)
    canonical_source_url: str | None = None
    source_organization: str | None = None
    retrieval_date: str | None = None
    license_status: Literal["verified", "unverified", "unknown", "not_applicable"]
    redistribution: Literal["redistributable", "reference-only", "private-evaluation", "synthetic"]
    notes: str | None = None


class ManifestPage(ContractModel):
    page_id: Identifier
    page_index: int = Field(ge=0)
    checksum: Checksum
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    source_kind: Literal["native_pdf", "scanned_pdf", "image", "synthetic_record", "unknown"]
    tags: list[Identifier] = Field(default_factory=list)


class ManifestDocument(ContractModel):
    document_id: Identifier
    split: Literal["dev", "validation", "test-public", "test-private"]
    category: Identifier
    checksum: Checksum
    source: ManifestSource
    pages: list[ManifestPage] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_page_ids(self) -> ManifestDocument:
        page_ids = [page.page_id for page in self.pages]
        if len(page_ids) != len(set(page_ids)):
            raise ValueError(f"duplicate page ID in document {self.document_id}")
        return self


class ReferenceSource(ContractModel):
    reference_id: Identifier
    title: str = Field(min_length=1)
    canonical_source_url: str | None = None
    source_organization: str | None = None
    retrieval_date: str | None = None
    license_status: Literal["verified", "unverified", "unknown"]
    redistribution: Literal["reference-only", "redistributable", "private-evaluation"]
    notes: str = Field(min_length=1)


class Manifest(ContractModel):
    schema_version: str = Field(default=SCHEMA_VERSION, pattern=r"^\d+\.\d+\.\d+$")
    dataset_revision: DatasetRevision
    coordinate_system: CoordinateSystem
    documents: list[ManifestDocument] = Field(default_factory=list)
    reference_sources: list[ReferenceSource] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_document_and_page_ids(self) -> Manifest:
        document_ids = [document.document_id for document in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("duplicate document ID")
        page_ids = [page.page_id for document in self.documents for page in document.pages]
        if len(page_ids) != len(set(page_ids)):
            raise ValueError("page IDs must be globally unique")
        reference_ids = [source.reference_id for source in self.reference_sources]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("duplicate reference source ID")
        return self


class AssetRecord(ContractModel):
    document_id: Identifier
    page_id: Identifier
    media_type: Literal["image/png", "image/jpeg", "image/tiff"]
    relative_path: str = Field(min_length=1, max_length=500)
    checksum: Checksum
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @field_validator("relative_path")
    @classmethod
    def relative_posix_path_only(cls, value: str) -> str:
        path = PurePosixPath(value)
        if "\\" in value or path.is_absolute() or ".." in path.parts:
            raise ValueError("asset path must be a safe relative POSIX path")
        return value


class AssetIndex(ContractModel):
    schema_version: str = Field(default=SCHEMA_VERSION, pattern=r"^\d+\.\d+\.\d+$")
    dataset_revision: DatasetRevision
    assets: list[AssetRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_page_assets(self) -> AssetIndex:
        keys = [(asset.document_id, asset.page_id) for asset in self.assets]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate page asset")
        return self


class GroundTruthDocument(ContractModel):
    document_id: Identifier
    pages: list[PageContent] = Field(min_length=1)


class GroundTruth(ContractModel):
    schema_version: str = Field(default=SCHEMA_VERSION, pattern=r"^\d+\.\d+\.\d+$")
    dataset_revision: DatasetRevision
    coordinate_system: CoordinateSystem
    annotation_provenance: Provenance
    documents: list[GroundTruthDocument] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_document_ids(self) -> GroundTruth:
        document_ids = [document.document_id for document in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("duplicate ground-truth document ID")
        return self


class SystemMetadata(ContractModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    adapter_name: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    model_name: str | None = None
    model_version: str | None = None
    execution_provider: str = Field(min_length=1)
    command: str | None = None
    git_commit: str | None = None
    runtime: dict[str, str | int | float | bool] = Field(default_factory=dict)
    hardware: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RunMetadata(ContractModel):
    run_id: Identifier
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    command: str | None = None
    dependency_lock_hash: str | None = None


PredictionStatus = Literal["success", "crashed", "timeout", "missing"]


class PredictionSample(ContractModel):
    document_id: Identifier
    page_id: Identifier
    status: PredictionStatus
    page: PageContent | None = None
    error: ProcessingError | None = None
    runtime_ms: float | None = Field(default=None, ge=0)
    peak_memory_mb: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def status_has_matching_payload(self) -> PredictionSample:
        if self.status == "success" and self.page is None:
            raise ValueError("successful prediction requires page content")
        if self.status != "success" and self.error is None:
            raise ValueError(f"{self.status} prediction requires an error record")
        if self.status == "success" and self.error is not None:
            raise ValueError("successful prediction cannot contain an error record")
        return self


class Predictions(ContractModel):
    schema_version: str = Field(default=SCHEMA_VERSION, pattern=r"^\d+\.\d+\.\d+$")
    dataset_revision: DatasetRevision
    coordinate_system: CoordinateSystem
    system: SystemMetadata
    run: RunMetadata
    samples: list[PredictionSample] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_sample_ids(self) -> Predictions:
        keys = [(sample.document_id, sample.page_id) for sample in self.samples]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate prediction sample")
        return self


MetricUnit = Literal[
    "page",
    "digit_page",
    "block",
    "equation",
    "table",
    "milliseconds",
    "megabytes",
]


class MetricSummary(ContractModel):
    micro: float = Field(ge=0)
    macro: float = Field(ge=0)
    sample_count: int = Field(ge=0)
    unit: MetricUnit
    higher_is_better: bool


class ReportSummary(ContractModel):
    total_pages: int = Field(ge=0)
    scored_pages: int = Field(ge=0)
    missing_pages: int = Field(ge=0)
    crashed_pages: int = Field(ge=0)
    timeout_pages: int = Field(ge=0)
    invalid_pages: int = Field(ge=0)
    ground_truth_missing_pages: int = Field(ge=0)


class FailureRecord(ContractModel):
    document_id: Identifier
    page_id: Identifier
    status: Literal["missing", "crashed", "timeout", "invalid", "ground_truth_missing"]
    message: str = Field(min_length=1)


class CategoryBreakdown(ContractModel):
    category: Identifier
    summary: ReportSummary
    metrics: dict[str, MetricSummary] = Field(default_factory=dict)


class Report(ContractModel):
    schema_version: str = Field(default=SCHEMA_VERSION, pattern=r"^\d+\.\d+\.\d+$")
    dataset_revision: DatasetRevision
    coordinate_system: CoordinateSystem
    system: SystemMetadata
    run: RunMetadata
    generated_at: datetime
    summary: ReportSummary
    metrics: dict[str, MetricSummary] = Field(default_factory=dict)
    category_breakdown: list[CategoryBreakdown] = Field(default_factory=list)
    failures: list[FailureRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
