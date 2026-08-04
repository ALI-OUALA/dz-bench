"""Contract validation and failure-aware Phase A scoring."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

from .io import load_ground_truth, load_manifest, load_predictions, write_json
from .metrics import (
    conservative_normalize,
    digit_exact_accuracy,
    edit_distance,
    equation_text_page_score,
    greedy_block_matches,
    layout_scores,
    reading_order_sequence_score,
    table_structure_page_score,
)
from .models import (
    CategoryBreakdown,
    FailureRecord,
    GroundTruth,
    Manifest,
    MetricSummary,
    PageContent,
    Predictions,
    Report,
    ReportSummary,
)

QUALITY_METRIC_NAMES = (
    "cer",
    "wer",
    "normalized_edit_similarity",
    "digit_exact_accuracy",
    "reading_order_sequence_score",
    "block_type_accuracy",
    "layout_bbox_iou",
    "layout_match_f1",
    "equation_text_similarity",
    "table_structure_similarity",
)
PERFORMANCE_METRIC_NAMES = ("runtime_ms", "peak_memory_mb")
METRIC_NAMES = QUALITY_METRIC_NAMES
_METRIC_SPECS = {
    "cer": ("page", False),
    "wer": ("page", False),
    "normalized_edit_similarity": ("page", True),
    "digit_exact_accuracy": ("digit_page", True),
    "reading_order_sequence_score": ("page", True),
    "block_type_accuracy": ("block", True),
    "layout_bbox_iou": ("block", True),
    "layout_match_f1": ("block", True),
    "equation_text_similarity": ("equation", True),
    "table_structure_similarity": ("table", True),
    "runtime_ms": ("milliseconds", False),
    "peak_memory_mb": ("megabytes", False),
}


@dataclass(slots=True)
class _Counts:
    total: int = 0
    scored: int = 0
    missing: int = 0
    crashed: int = 0
    timeout: int = 0
    invalid: int = 0
    ground_truth_missing: int = 0

    def report(self) -> ReportSummary:
        return ReportSummary(
            total_pages=self.total,
            scored_pages=self.scored,
            missing_pages=self.missing,
            crashed_pages=self.crashed,
            timeout_pages=self.timeout,
            invalid_pages=self.invalid,
            ground_truth_missing_pages=self.ground_truth_missing,
        )


@dataclass(slots=True)
class _MetricValue:
    value: float
    numerator: float
    denominator: float
    sample_count: int
    unit: str
    higher_is_better: bool


@dataclass(slots=True)
class _Observation:
    category: str
    metrics: dict[str, _MetricValue]


def _flatten_lines(page: PageContent) -> list:
    return [
        line
        for block in sorted(page.blocks, key=lambda item: item.reading_order_index)
        for line in sorted(block.lines, key=lambda item: item.reading_order_index)
    ]


def _page_text(page: PageContent) -> str:
    return "\n".join(line.normalized_text for line in _flatten_lines(page))


def _comparable_block_order(
    reference: PageContent, hypothesis: PageContent
) -> tuple[list[str], list[str]]:
    """Map system-local prediction IDs onto reference blocks through geometry."""

    reference_blocks = sorted(reference.blocks, key=lambda item: item.reading_order_index)
    hypothesis_blocks = sorted(hypothesis.blocks, key=lambda item: item.reading_order_index)
    reference_order = [block.block_id for block in reference_blocks]
    mapping = {
        match.hypothesis.block_id: match.reference.block_id
        for match in greedy_block_matches(reference_blocks, hypothesis_blocks)
        if match.hypothesis is not None
    }
    hypothesis_order = [
        mapping.get(block.block_id, f"unmatched:{index}")
        for index, block in enumerate(hypothesis_blocks)
    ]
    return reference_order, hypothesis_order


def _observation(category: str, reference: PageContent, hypothesis: PageContent) -> _Observation:
    reference_text = conservative_normalize(_page_text(reference))
    hypothesis_text = conservative_normalize(_page_text(hypothesis))
    reference_chars = list(reference_text)
    hypothesis_chars = list(hypothesis_text)
    reference_words = reference_text.split()
    hypothesis_words = hypothesis_text.split()
    normalized_distance = edit_distance(reference_chars, hypothesis_chars)
    word_distance = edit_distance(reference_words, hypothesis_words)
    order, predicted_order = _comparable_block_order(reference, hypothesis)
    order_distance = edit_distance(order, predicted_order)
    has_reference_digit = any(character.isdecimal() for character in reference_text)
    similarity_units = max(len(reference_chars), len(hypothesis_chars), 1)
    order_units = max(len(order), len(predicted_order), 1)
    metrics = {
        "cer": _MetricValue(
            normalized_distance / max(len(reference_chars), 1),
            normalized_distance,
            max(len(reference_chars), 1),
            1,
            "page",
            False,
        ),
        "wer": _MetricValue(
            word_distance / max(len(reference_words), 1),
            word_distance,
            max(len(reference_words), 1),
            1,
            "page",
            False,
        ),
        "normalized_edit_similarity": _MetricValue(
            1.0 - normalized_distance / similarity_units,
            similarity_units - normalized_distance,
            similarity_units,
            1,
            "page",
            True,
        ),
        "reading_order_sequence_score": _MetricValue(
            reading_order_sequence_score(order, predicted_order),
            order_units - order_distance,
            order_units,
            1,
            "page",
            True,
        ),
    }
    if has_reference_digit:
        digit_accuracy = digit_exact_accuracy(reference_text, hypothesis_text)
        metrics["digit_exact_accuracy"] = _MetricValue(
            digit_accuracy,
            digit_accuracy,
            1,
            1,
            "digit_page",
            True,
        )
    layout = layout_scores(reference.blocks, hypothesis.blocks)
    if layout.block_count:
        metrics.update(
            {
                "block_type_accuracy": _MetricValue(
                    layout.correct_type / layout.block_count,
                    layout.correct_type,
                    layout.block_count,
                    layout.block_count,
                    "block",
                    True,
                ),
                "layout_bbox_iou": _MetricValue(
                    layout.bbox_iou_sum / layout.block_count,
                    layout.bbox_iou_sum,
                    layout.block_count,
                    layout.block_count,
                    "block",
                    True,
                ),
                "layout_match_f1": _MetricValue(
                    2
                    * layout.matched_count
                    / max(len(reference.blocks) + len(hypothesis.blocks), 1),
                    2 * layout.matched_count,
                    max(len(reference.blocks) + len(hypothesis.blocks), 1),
                    layout.block_count,
                    "block",
                    True,
                ),
            }
        )
    equation_score, equation_count = equation_text_page_score(reference.blocks, hypothesis.blocks)
    if equation_score is not None:
        metrics["equation_text_similarity"] = _MetricValue(
            equation_score,
            equation_score * equation_count,
            equation_count,
            equation_count,
            "equation",
            True,
        )
    table_score, table_count = table_structure_page_score(reference.blocks, hypothesis.blocks)
    if table_score is not None:
        metrics["table_structure_similarity"] = _MetricValue(
            table_score,
            table_score * table_count,
            table_count,
            table_count,
            "table",
            True,
        )
    return _Observation(category=category, metrics=metrics)


def _metric_summary(name: str, values: list[_MetricValue]) -> MetricSummary:
    unit, higher_is_better = _METRIC_SPECS[name]
    if not values:
        return MetricSummary(
            micro=0.0,
            macro=0.0,
            sample_count=0,
            unit=unit,
            higher_is_better=higher_is_better,
        )
    micro = sum(value.numerator for value in values) / max(
        sum(value.denominator for value in values), 1
    )
    return MetricSummary(
        micro=micro,
        macro=fmean(value.value for value in values),
        sample_count=sum(value.sample_count for value in values),
        unit=unit,
        higher_is_better=higher_is_better,
    )


def _metrics(
    rows: list[_Observation], extras: dict[str, list[_MetricValue]] | None = None
) -> dict[str, MetricSummary]:
    values_by_name: dict[str, list[_MetricValue]] = defaultdict(list)
    for row in rows:
        for name, value in row.metrics.items():
            values_by_name[name].append(value)
    for name, values in (extras or {}).items():
        values_by_name[name].extend(values)
    names = list(QUALITY_METRIC_NAMES)
    names.extend(name for name in PERFORMANCE_METRIC_NAMES if values_by_name.get(name))
    return {name: _metric_summary(name, values_by_name.get(name, [])) for name in names}


def _performance_value(name: str, value: float) -> _MetricValue:
    unit, higher_is_better = _METRIC_SPECS[name]
    return _MetricValue(value, value, 1, 1, unit, higher_is_better)


def _page_map(manifest: Manifest) -> dict[tuple[str, str], tuple[str, object]]:
    return {
        (document.document_id, page.page_id): (document.category, page)
        for document in manifest.documents
        for page in document.pages
    }


def validate_bundle(
    manifest: Manifest, ground_truth: GroundTruth, predictions: Predictions
) -> None:
    """Validate cross-file revision, coordinate, and identifier compatibility."""

    if ground_truth.dataset_revision != manifest.dataset_revision:
        raise ValueError("ground truth dataset revision does not match manifest")
    if predictions.dataset_revision != manifest.dataset_revision:
        raise ValueError("prediction dataset revision does not match manifest")
    if ground_truth.coordinate_system != manifest.coordinate_system:
        raise ValueError("ground truth coordinate system does not match manifest")
    if predictions.coordinate_system != manifest.coordinate_system:
        raise ValueError("prediction coordinate system does not match manifest")
    expected = set(_page_map(manifest))
    truth_keys = {
        (document.document_id, page.page_id)
        for document in ground_truth.documents
        for page in document.pages
    }
    prediction_keys = {(sample.document_id, sample.page_id) for sample in predictions.samples}
    if not truth_keys <= expected:
        raise ValueError("ground truth contains a document or page absent from the manifest")
    if not prediction_keys <= expected:
        raise ValueError("predictions contain a document or page absent from the manifest")


def score(manifest: Manifest, ground_truth: GroundTruth, predictions: Predictions) -> Report:
    """Score every manifest page, retaining missing and failed prediction samples."""

    validate_bundle(manifest, ground_truth, predictions)
    target_pages = _page_map(manifest)
    truth_pages = {
        (document.document_id, page.page_id): page
        for document in ground_truth.documents
        for page in document.pages
    }
    prediction_pages = {
        (sample.document_id, sample.page_id): sample for sample in predictions.samples
    }
    counts = _Counts()
    category_counts: dict[str, _Counts] = defaultdict(_Counts)
    observations: list[_Observation] = []
    category_observations: dict[str, list[_Observation]] = defaultdict(list)
    performance_values: dict[str, list[_MetricValue]] = defaultdict(list)
    category_performance: dict[str, dict[str, list[_MetricValue]]] = defaultdict(
        lambda: defaultdict(list)
    )
    failures: list[FailureRecord] = []

    for key, (category, manifest_page) in target_pages.items():
        counts.total += 1
        category_counts[category].total += 1
        document_id, page_id = key
        reference = truth_pages.get(key)
        if reference is None:
            counts.ground_truth_missing += 1
            category_counts[category].ground_truth_missing += 1
            failures.append(
                FailureRecord(
                    document_id=document_id,
                    page_id=page_id,
                    status="ground_truth_missing",
                    message="manifest page has no ground-truth page",
                )
            )
            continue
        sample = prediction_pages.get(key)
        if sample is None or sample.status == "missing":
            counts.missing += 1
            category_counts[category].missing += 1
            failures.append(
                FailureRecord(
                    document_id=document_id,
                    page_id=page_id,
                    status="missing",
                    message="prediction sample was not supplied",
                )
            )
            continue
        for metric_name, measured_value in (
            ("runtime_ms", sample.runtime_ms),
            ("peak_memory_mb", sample.peak_memory_mb),
        ):
            if measured_value is not None:
                value = _performance_value(metric_name, measured_value)
                performance_values[metric_name].append(value)
                category_performance[category][metric_name].append(value)
        if sample.status in {"crashed", "timeout"}:
            setattr(counts, sample.status, getattr(counts, sample.status) + 1)
            setattr(
                category_counts[category],
                sample.status,
                getattr(category_counts[category], sample.status) + 1,
            )
            failures.append(
                FailureRecord(
                    document_id=document_id,
                    page_id=page_id,
                    status=sample.status,
                    message=sample.error.message if sample.error else sample.status,
                )
            )
            continue
        hypothesis = sample.page
        if (
            hypothesis is None
            or hypothesis.page_id != page_id
            or hypothesis.page_index != manifest_page.page_index
            or hypothesis.checksum != manifest_page.checksum
        ):
            counts.invalid += 1
            category_counts[category].invalid += 1
            failures.append(
                FailureRecord(
                    document_id=document_id,
                    page_id=page_id,
                    status="invalid",
                    message="prediction page metadata does not match the manifest",
                )
            )
            continue
        observation = _observation(category, reference, hypothesis)
        observations.append(observation)
        category_observations[category].append(observation)
        counts.scored += 1
        category_counts[category].scored += 1

    category_breakdown = [
        CategoryBreakdown(
            category=category,
            summary=category_counts[category].report(),
            metrics=_metrics(
                category_observations.get(category, []),
                category_performance.get(category, {}),
            ),
        )
        for category in sorted(category_counts)
    ]
    return Report(
        dataset_revision=manifest.dataset_revision,
        coordinate_system=manifest.coordinate_system,
        system=predictions.system,
        run=predictions.run,
        generated_at=datetime.now(UTC),
        summary=counts.report(),
        metrics=_metrics(observations, performance_values),
        category_breakdown=category_breakdown,
        failures=failures,
        notes=[
            "Missing, crashed, timeout, invalid, and ground-truth-missing pages remain "
            "in the report.",
            "Digit exact accuracy is calculated only for pages containing reference digits.",
            "Runtime and peak memory summarize measured prediction samples only.",
        ],
    )


def score_files(
    manifest_path: str | Path,
    ground_truth_path: str | Path,
    predictions_path: str | Path,
    report_path: str | Path,
) -> Report:
    report = score(
        load_manifest(manifest_path),
        load_ground_truth(ground_truth_path),
        load_predictions(predictions_path),
    )
    write_json(report, report_path)
    return report


def render_markdown(report: Report) -> str:
    """Render a compact human-readable projection of the structured report."""

    lines = [
        "# DZ-Bench report",
        "",
        f"- Dataset: `{report.dataset_revision.dataset_id}@{report.dataset_revision.revision}`",
        f"- System: `{report.system.name}@{report.system.version}`",
        f"- Total pages: **{report.summary.total_pages}**",
        f"- Scored pages: **{report.summary.scored_pages}**",
        f"- Missing/crashed/timeout: **{report.summary.missing_pages}/"
        f"{report.summary.crashed_pages}/{report.summary.timeout_pages}**",
        "",
        "## Metrics",
        "",
        "| Metric | Micro | Macro | Samples |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, metric in report.metrics.items():
        lines.append(
            f"| `{name}` | {metric.micro:.4f} | {metric.macro:.4f} | {metric.sample_count} |"
        )
    lines.extend(["", "## Category breakdown", ""])
    for category in report.category_breakdown:
        lines.extend(
            [
                f"### `{category.category}`",
                "",
                f"Pages: {category.summary.scored_pages}/{category.summary.total_pages} scored; "
                f"missing {category.summary.missing_pages}, "
                f"crashed {category.summary.crashed_pages}, "
                f"timeout {category.summary.timeout_pages}.",
                "",
                "| Metric | Micro | Macro | Samples |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for name, metric in category.metrics.items():
            lines.append(
                f"| `{name}` | {metric.micro:.4f} | {metric.macro:.4f} | {metric.sample_count} |"
            )
        lines.append("")
    lines.extend(["## Failures", ""])
    if not report.failures:
        lines.append("None.")
    else:
        lines.extend(
            f"- `{failure.status}` `{failure.document_id}/{failure.page_id}`: {failure.message}"
            for failure in report.failures
        )
    return "\n".join(lines) + "\n"


def write_report_markdown(report: Report, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    return target
