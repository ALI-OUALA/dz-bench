# Data sources

## Current inventory

As of 2026-08-04, no real BAC document has been acquired or verified for redistribution in this repository. There are zero published BAC pages and zero published BAC ground-truth records.

The only BAC artifact is [`datasets/bac/manifests/bac-reference-only-v0.1.json`](datasets/bac/manifests/bac-reference-only-v0.1.json), a reference-only template. Its null URL and unknown licence status are intentional; they are not a claim about a specific exam source.

## Required record for future sources

Every future source must be recorded before evaluation with its document identity, year/session/stream/subject, source organization, canonical URL, retrieval date, SHA-256 checksum, size, observed copyright/licence notice, redistribution decision, and duplicate status. Rights-unclear sources remain reference-only or private.

## Synthetic source

Phase A uses the local generator in `src/dz_bench/synthetic.py`. Its Arabic, French, mixed-script, numeric, and exam-like snippets are original authored content. It has no image, PDF, external dataset, or exam-text dependency.

