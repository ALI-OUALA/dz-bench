"""Contract validation and failure-aware Phase A scoring."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

from .io import load_ground_truth, load_manifest, load_predictions, write_json
from .metrics import (
    character_error_rate,
    conservative_normalize,
    digit_exact_accuracy,
    edit_distance,
    reading_order_sequence_score,
    word_error_rate,
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

METRIC_NAMES = (
    "cer",
    "wer",
    "normalized_edit_similarity",
    "digit_exact_accuracy",
    "reading_order_sequence_score",
)


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
class _Observation:
    category: str
    cer: float
    wer: float
    similarity: float
    digit: float | None
    reading_order: float
    cer_errors: int
    cer_units: int
    wer_errors: int
    wer_units: int
    similarity_errors: int
    similarity_units: int
    order_errors: int
    order_units: int


def _flatten_lines(page: PageContent) -> list:
    return [
        line
        for block in sorted(page.blocks, key=lambda item: item.reading_order_index)
        for line in sorted(block.lines, key=lambda item: item.reading_order_index)
    ]


def _page_text(page: PageContent) -> str:
    return "\n".join(line.normalized_text for line in _flatten_lines(page))


def _line_order(page: PageContent) -> list[str]:
    if page.reading_order:
        return list(page.reading_order)
    return [line.line_id for line in _flatten_lines(page)]


def _observation(category: str, reference: PageContent, hypothesis: PageContent) -> _Observation:
    reference_text = conservative_normalize(_page_text(reference))
    hypothesis_text = conservative_normalize(_page_text(hypothesis))
    reference_chars = list(reference_text)
    hypothesis_chars = list(hypothesis_text)
    reference_words = reference_text.split()
    hypothesis_words = hypothesis_text.split()
    normalized_distance = edit_distance(reference_chars, hypothesis_chars)
    order = _line_order(reference)
    predicted_order = _line_order(hypothesis)
    order_distance = edit_distance(order, predicted_order)
    has_reference_digit = any(character.isdecimal() for character in reference_text)
    return _Observation(
        category=category,
        cer=character_error_rate(reference_text, hypothesis_text),
        wer=word_error_rate(reference_text, hypothesis_text),
        similarity=1.0 - normalized_distance / max(len(reference_chars), len(hypothesis_chars), 1),
        digit=digit_exact_accuracy(reference_text, hypothesis_text)
        if has_reference_digit
        else None,
        reading_order=reading_order_sequence_score(order, predicted_order),
        cer_errors=edit_distance(reference_chars, hypothesis_chars),
        cer_units=max(len(reference_chars), 1),
        wer_errors=edit_distance(reference_words, hypothesis_words),
        wer_units=max(len(reference_words), 1),
        similarity_errors=normalized_distance,
        similarity_units=max(len(reference_chars), len(hypothesis_chars), 1),
        order_errors=order_distance,
        order_units=max(len(order), len(predicted_order), 1),
    )


def _metric_summary(rows: list[_Observation], name: str) -> MetricSummary:
    applicable = [row for row in rows if name != "digit_exact_accuracy" or row.digit is not None]
    if not applicable:
        return MetricSummary(
            micro=0.0,
            macro=0.0,
            sample_count=0,
            unit="digit_page" if name == "digit_exact_accuracy" else "page",
            higher_is_better=name not in {"cer", "wer"},
        )
    if name == "cer":
        micro = sum(row.cer_errors for row in applicable) / sum(row.cer_units for row in applicable)
        values = [row.cer for row in applicable]
        higher_is_better = False
    elif name == "wer":
        micro = sum(row.wer_errors for row in applicable) / sum(row.wer_units for row in applicable)
        values = [row.wer for row in applicable]
        higher_is_better = False
    elif name == "normalized_edit_similarity":
        micro = 1.0 - sum(row.similarity_errors for row in applicable) / sum(
            row.similarity_units for row in applicable
        )
        values = [row.similarity for row in applicable]
        higher_is_better = True
    elif name == "digit_exact_accuracy":
        micro = fmean(row.digit for row in applicable if row.digit is not None)
        values = [row.digit for row in applicable if row.digit is not None]
        higher_is_better = True
    else:
        micro = 1.0 - sum(row.order_errors for row in applicable) / sum(
            row.order_units for row in applicable
        )
        values = [row.reading_order for row in applicable]
        higher_is_better = True
    return MetricSummary(
        micro=micro,
        macro=fmean(values),
        sample_count=len(applicable),
        unit="digit_page" if name == "digit_exact_accuracy" else "page",
        higher_is_better=higher_is_better,
    )


def _metrics(rows: list[_Observation]) -> dict[str, MetricSummary]:
    return {name: _metric_summary(rows, name) for name in METRIC_NAMES}


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
            metrics=_metrics(category_observations.get(category, [])),
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
        metrics=_metrics(observations),
        category_breakdown=category_breakdown,
        failures=failures,
        notes=[
            "Missing, crashed, timeout, invalid, and ground-truth-missing pages remain "
            "in the report.",
            "Digit exact accuracy is calculated only for pages containing reference digits.",
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
