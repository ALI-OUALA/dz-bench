# Data sources

## Current inventory

As of 2026-08-04, no real BAC document has been acquired or verified for redistribution in this repository. There are zero published BAC pages and zero published BAC ground-truth records.

The only real-paper BAC artifact is [`datasets/bac/manifests/bac-reference-only-v0.1.json`](datasets/bac/manifests/bac-reference-only-v0.1.json). It contains two provenance-only private-evaluation records for locally inspected files plus a blank template. Null URLs and unknown licence status are intentional. No source PDF, page image, transcription, or real-paper ground truth is distributed.

## Required record for future sources

Every future source must be recorded before evaluation with its document identity, year/session/stream/subject, source organization, canonical URL, retrieval date, SHA-256 checksum, size, observed copyright/licence notice, redistribution decision, and duplicate status. Rights-unclear sources remain reference-only or private.

## Synthetic source

The local generators in `src/dz_bench/synthetic.py` produce original authored content. The `bac-synthetic` profile adds Arabic, French, mixed-script, mathematics, physics, equation, table, diagram, reading-order, and scan-quality variants as deterministic records. The optional `bac-images` command renders those records to PNG using an external user-supplied font; it ships no font and has no external dataset or exam-text dependency.
