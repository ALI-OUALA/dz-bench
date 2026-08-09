# DZ-Bench

DZ-Bench is an independent, model-neutral benchmark for Arabic–French document intelligence. It exchanges artifacts with document engines through public JSON contracts; core scoring never imports DzDoc internals or any OCR framework.

Phase A plus the synthetic BAC slice provides:

- packaged draft 2020-12 schemas for manifests, assets, ground truth, predictions, and reports;
- Pydantic v2 validation with checksums, page-pixel coordinates, blocks, lines, spans, provenance, confidence, warnings, and alternatives;
- conservative NFC normalization, CER, WER, normalized edit similarity, digit exact accuracy, reading-order sequence score, block/layout matching, equation text similarity, and table structure similarity;
- failure-aware scoring that counts missing, crashed, timed-out, invalid, and ground-truth-missing pages;
- an original deterministic Arabic/French/mixed BAC-like corpus with mathematics, physics, equations, tables, diagrams, reading-order, and clean/compressed/skewed/photographed tags;
- per-sample runtime and peak-memory fields with report summaries when a system measures them;
- confidence calibration, hallucination, diagram, structured-field, financial-value,
  structured-coordinate, and structured-hallucination metrics;
- an original four-quality Algerian invoice pack with fictional NIF/NIS/RC values,
  line-item tables, HT/TVA/TTC arithmetic, source coordinates, and validations;
- a public JSON adapter boundary for DzDoc predictions;
- a provenance-only BAC manifest with no protected documents or fabricated BAC ground truth.

## Install and verify

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

Install the optional deterministic PNG renderer:

```powershell
uv sync --extra dev --extra raster
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

Generate the records-only BAC-like benchmark slice (no protected exam files are shipped):

```powershell
dz-bench bac-synthetic --output-dir .tmp/bac-synthetic --seed 17 --repeats 1
```

The generated `records.json` is the reproducible source description. Its manifest and ground truth carry page tags for `arabic`, `french`, `mixed`, `mathematics`, `physics`, `equations`, `tables`, `diagrams`, reading order, and scan-quality variants. The base profile intentionally emits records only.

Render the same authored pages to a real PNG bundle with an external font:

```powershell
dz-bench bac-images `
  --font C:\Windows\Fonts\arial.ttf `
  --output-dir .tmp/bac-images `
  --seed 17 --repeats 1
```

`bac-images` writes `images/*.png`, a ground-truth-free `assets.json` index, generator provenance in `records.json`, `source_kind: image`, and SHA-256 checksums over actual PNG bytes. OCR engines receive `manifest.json` plus `assets.json`; they never need authored record text. Clean, compressed, skewed, and photographed variants keep canonical page geometry and ground-truth boxes. No font is bundled.

Render the original invoice pack:

```powershell
dz-bench invoice-images `
  --font C:\Windows\Fonts\arial.ttf `
  --output-dir .tmp/invoice-images `
  --seed 23
```

This writes four quality variants plus public manifests, ground truth, and a
ground-truth-free asset index. Company names and Algerian-format identifiers are
fictional. See `docs/reports/invoice-dz-2026-08-09.md` for the first comparison.

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

The five public artifact schemas live under `schemas/`, ship inside the wheel, and share definitions through `common.schema.json`. Coordinates are page pixels with origin at the top-left, x increasing rightward, and y increasing downward. Each page and document carries a SHA-256 checksum. Raw text is preserved separately from normalized and search forms; no blind Arabic string reversal is performed.

`datasets/bac/manifests/bac-reference-only-v0.1.json` and [`DATASET_POLICY.md`](DATASET_POLICY.md) define the current legal boundary: no protected BAC PDFs and no fabricated real-BAC ground truth. The synthetic BAC-like records are original benchmark-authored content and are reported separately from real BAC data.
