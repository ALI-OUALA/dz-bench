# DZ-Bench

DZ-Bench is an independent, model-neutral benchmark for Arabic–French document intelligence. It exchanges artifacts with document engines through public JSON contracts; core scoring never imports DzDoc internals or any OCR framework.

Phase A provides:

- draft 2020-12 schemas for manifests, ground truth, predictions, and reports;
- Pydantic v2 validation with checksums, page-pixel coordinates, blocks, lines, spans, provenance, confidence, warnings, and alternatives;
- conservative NFC normalization, CER, WER, normalized edit similarity, digit exact accuracy, and reading-order sequence score;
- failure-aware scoring that counts missing, crashed, timed-out, invalid, and ground-truth-missing pages;
- an original deterministic Arabic/French/mixed/BAC-style-ish synthetic corpus with no image or PDF dependency;
- a public JSON adapter boundary for DzDoc predictions;
- a reference-only BAC manifest template with no protected documents or fabricated BAC ground truth.

## Install and verify

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

If `uv` is unavailable, a normal editable install is sufficient:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

## Contract commands

Generate deterministic smoke data:

```powershell
dz-bench synthetic --output-dir .tmp/synthetic --seed 17 --documents 4 --pages 1 --records
```

Run complete synthetic generation -> public prediction -> scorer -> JSON/Markdown
report path:

```powershell
dz-bench smoke --output-dir .tmp/smoke --seed 17
```

Validate an artifact:

```powershell
dz-bench validate .tmp/synthetic/manifest.json --kind manifest
dz-bench validate .tmp/synthetic/ground-truth.json --kind ground-truth
```

Score a public prediction artifact:

```powershell
dz-bench score `
  --manifest .tmp/synthetic/manifest.json `
  --ground-truth .tmp/synthetic/ground-truth.json `
  --predictions .tmp/predictions.json `
  --output .tmp/report.json
```

The scorer writes both `.tmp/report.json` and `.tmp/report.md`. Prediction samples must declare `success`, `crashed`, `timeout`, or `missing`; an omitted sample is also counted as missing.

## Public schemas

The four public files live under `schemas/` and share definitions through `common.schema.json`. Coordinates are page pixels with origin at the top-left, x increasing rightward, and y increasing downward. Each page and document carries a SHA-256 checksum. Raw text is preserved separately from normalized and search forms; no blind Arabic string reversal is performed.

`datasets/bac/manifests/bac-reference-only-v0.1.json` and [`DATASET_POLICY.md`](DATASET_POLICY.md) define the current legal boundary: no protected BAC PDFs and no fabricated BAC ground truth.
