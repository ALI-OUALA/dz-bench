# AGENTS.md — DZ-Bench

This file governs every coding and research agent working in the `dz-bench` repository.

## 0. Project identity and context

**Repository:** `dz-bench`  
**Working name:** DZ-Bench  
**Owner and original author:** MERATI Ali Ouala Eddine  
**Relationship to DzDoc:** independent evaluation project  
**Primary domain:** Arabic–French OCR, document parsing, and structured extraction for Algerian and comparable multilingual documents

DZ-Bench is intentionally separate from `dzdoc-engine`.

Its purpose is to evaluate document systems fairly and reproducibly, including DzDoc Engine, alternative OCR engines, PDF text extractors, layout parsers, table systems, equation recognizers, and visual-language models.

DZ-Bench must never become a test suite designed only to make DzDoc look good.

## 1. Mission

Build the strongest practical benchmark for Algerian Arabic–French document intelligence.

The benchmark must measure distinct capabilities rather than hide everything behind one aggregate number:

- native PDF text extraction;
- visual OCR;
- Arabic recognition;
- French and Latin recognition;
- mixed-script text;
- bidirectional reading order;
- digits, dates, money, and identifiers;
- mathematical and scientific notation;
- tables, forms, and diagrams;
- document structure;
- confidence quality;
- structured field extraction;
- latency, memory, and estimated serving cost.

A benchmark result is valid only when it is inspectable, reproducible, versioned, and honest about failures.

## 2. Why Algerian Baccalaureate exams belong in DZ-Bench

Algerian BAC exam papers are a major real-world evaluation family because they combine:

- formal Arabic RTL text;
- French and other Latin-script languages;
- Latin and Arabic-Indic digits;
- mathematics;
- physics and chemistry formulas;
- scientific units, subscripts, and superscripts;
- tables and graphs;
- diagrams and captions;
- numbered exercises and nested subquestions;
- official headers, footers, and page numbering;
- native PDFs, scans, compressed copies, and photographed pages.

BAC exams are valuable because they test text recognition and document reconstruction at the same time. They are not the only benchmark category, and systems must not be optimized only for one exam template.

## 3. BAC benchmark coverage

The target collection should cover multiple years, streams, subjects, languages, and document conditions.

### 3.1 Streams

Where legally and practically available:

- Mathematics;
- Experimental Sciences;
- Technical Mathematics;
- Management and Economics;
- Foreign Languages;
- Literature and Philosophy.

### 3.2 Subjects

Prioritize a balanced subset:

- Arabic language and literature;
- French;
- English;
- Mathematics;
- Physics;
- Natural and Life Sciences;
- Philosophy;
- History and Geography;
- Economics and Management;
- selected technical subjects where layout and notation are useful.

### 3.3 Evaluation value by subject

- **Arabic:** RTL order, punctuation, quotations, diacritics, numbered questions.
- **French/English:** accents, Latin OCR, mixed identifiers, equations embedded in prose.
- **Mathematics:** symbols, fractions, matrices, functions, geometry, logical hierarchy.
- **Physics/Chemistry:** units, formulas, subscripts, superscripts, tables, diagrams.
- **Natural Sciences:** figures, labels, charts, scientific terminology.
- **History/Geography:** long paragraphs, dates, maps, tables, names.
- **Philosophy/Literature:** dense prose and reading order.
- **Economics/Management:** financial tables, percentages, values, structured questions.

### 3.4 Document-condition tags

Every source paper and page should be tagged where applicable:

- native PDF;
- scanned PDF;
- image-only PDF;
- mixed PDF;
- clean;
- compressed;
- skewed;
- rotated;
- low resolution;
- photographed;
- watermarked;
- photocopied;
- annotated;
- equations;
- tables;
- diagrams;
- Arabic-only;
- French-only;
- mixed-script.

## 4. Legal and source policy

Benchmark code and benchmark documents are separate legal objects.

Never assume that a publicly downloadable exam PDF may be redistributed.

For every source document record:

- title;
- year;
- session;
- stream;
- subject;
- source organization;
- canonical source URL;
- retrieval date;
- source checksum;
- file size;
- observed copyright or licence notice;
- redistribution status;
- allowed repository treatment.

Allowed treatments:

### 4.1 Redistributable

The repository may contain the document only when an explicit licence or written authorization permits redistribution.

### 4.2 Reference-only

Store:

- metadata;
- canonical source URL;
- expected checksum;
- expected file size;
- downloader instructions or script.

Do not commit the PDF.

### 4.3 Private evaluation

Store only a local manifest template and ignore the document in git. Do not publish protected pages, answer sheets, or derived content beyond legally allowed ground truth.

### 4.4 Synthetic analogue

Recreate the layout and difficulty using original benchmark-authored content. Do not copy protected questions, diagrams, or wording.

When rights are unclear, default to **reference-only**.

The repository must include:

- `DATASET_POLICY.md`;
- `DATA_SOURCES.md`;
- dataset cards;
- machine-readable provenance;
- a takedown/contact process;
- `.gitignore` rules preventing accidental PDF/image commits.

Prefer official or clearly authorized sources. Do not scrape unofficial websites indiscriminately.

## 5. Independence from DzDoc

DZ-Bench must remain model-neutral and engine-neutral.

It may provide adapters for:

- DzDoc Engine;
- PaddleOCR;
- Tesseract;
- EasyOCR;
- docTR;
- Surya;
- PDF text extractors;
- table and layout parsers;
- VLMs;
- hosted APIs where terms permit.

Rules:

- Core scoring must not import private DzDoc internals.
- A DzDoc adapter belongs under `adapters/dzdoc`.
- Predictions use public versioned schemas.
- Systems may be invoked by CLI, HTTP API, Docker image, or adapter.
- Dataset revisions are independent from engine/model releases.
- Hidden test data cannot be used for public tuning without disclosure.
- Benchmark rules must apply equally to every evaluated system.

## 6. Integration contract with `dzdoc-engine`

The repositories communicate through public versioned artifacts:

- benchmark manifest JSON Schema;
- ground-truth JSON Schema;
- prediction JSON Schema;
- report JSON Schema;
- dataset revision ID;
- model/system metadata;
- content checksums.

Recommended flow:

```text
dz-bench export --dataset bac-small@0.1 --output bundle/

dzdoc benchmark-run \
  --bundle bundle/ \
  --output predictions.json

dz-bench score \
  --dataset bac-small@0.1 \
  --predictions predictions.json \
  --report report/
```

The exact CLI may evolve, but the boundary must remain public and versioned.

## 7. Proposed repository structure

```text
dz-bench/
├── AGENTS.md
├── README.md
├── LICENSE.md
├── DATASET_POLICY.md
├── DATA_SOURCES.md
├── THIRD_PARTY_NOTICES.md
├── pyproject.toml
├── uv.lock
├── schemas/
│   ├── manifest.schema.json
│   ├── ground-truth.schema.json
│   ├── prediction.schema.json
│   └── report.schema.json
├── src/
│   └── dz_bench/
│       ├── cli/
│       ├── manifests/
│       ├── acquisition/
│       ├── validation/
│       ├── ground_truth/
│       ├── metrics/
│       ├── reporting/
│       ├── leaderboards/
│       ├── synthetic/
│       └── adapters/
├── datasets/
│   ├── bac/
│   │   ├── manifests/
│   │   ├── cards/
│   │   └── downloaders/
│   ├── synthetic/
│   └── external/
├── annotations/
│   ├── guidelines/
│   ├── tools/
│   └── adjudication/
├── benchmarks/
│   ├── text/
│   ├── layout/
│   ├── tables/
│   ├── equations/
│   └── structured/
├── reports/
├── tests/
└── scripts/
```

Do not create empty architecture theatre. Do not commit protected documents merely because a dataset directory exists.

## 8. Benchmark tasks

### 8.1 Text recognition

Measure:

- exact match;
- Character Error Rate;
- Word Error Rate;
- normalized edit distance;
- Arabic CER/WER;
- French CER/WER;
- mixed-line CER/WER;
- digits-only accuracy;
- punctuation accuracy;
- diacritic-sensitive Arabic score;
- diacritic-insensitive Arabic score;
- Unicode normalization effects.

### 8.2 Reading order

Measure:

- line order;
- block order;
- column order;
- RTL/LTR mixed order;
- question/subquestion hierarchy;
- table versus surrounding-text order;
- figure/caption placement.

### 8.3 Layout

Measure detection and classification for:

- title;
- instruction;
- exercise;
- subquestion;
- paragraph;
- list;
- table;
- equation;
- figure;
- diagram;
- caption;
- header;
- footer;
- page number;
- answer space;
- unknown.

### 8.4 Tables

Measure:

- table detection;
- row/column boundaries;
- cell adjacency;
- merged cells;
- cell text;
- reading order;
- structure accuracy.

### 8.5 Mathematics and science notation

Measure separately:

- equation-region detection;
- symbol recognition;
- superscripts;
- subscripts;
- fractions;
- matrices;
- units;
- chemical notation;
- expression-to-LaTeX or MathML when ground truth exists.

Do not merge equation errors into prose CER without separate reporting.

### 8.6 Document reconstruction

Measure:

- heading hierarchy;
- exercise numbering;
- block grouping;
- figure/caption association;
- page continuity;
- structured or Markdown similarity.

### 8.7 Confidence and review utility

When systems emit confidence, measure:

- calibration error;
- risk-coverage curve;
- error-detection AUROC;
- percentage of errors captured in the lowest-confidence regions;
- human-review workload at a target accuracy.

### 8.8 Performance

Measure:

- cold start;
- warm start;
- latency per page;
- throughput;
- peak RAM;
- peak VRAM;
- model download size;
- pages avoiding visual OCR;
- dual-recognizer escalation rate;
- VLM escalation rate;
- estimated cost per 1,000 pages.

## 9. Ground-truth policy

Ground truth must be more trustworthy than the systems it evaluates.

### 9.1 Annotation layers

Maintain distinct layers for:

- page image and dimensions;
- blocks and polygons;
- reading order;
- lines;
- text spans;
- language and script;
- equations;
- tables and cells;
- diagrams;
- document hierarchy.

### 9.2 Annotation process

For high-value BAC subsets:

1. Initial transcription or extraction.
2. Independent second review.
3. Disagreement adjudication.
4. Unicode normalization validation.
5. Numeric and equation verification.
6. Final checksum and revision.

Record:

- annotator identifier or pseudonym;
- tool version;
- timestamps;
- disagreement count;
- adjudication outcome;
- revision history.

### 9.3 Arabic rules

- Store source/raw text and logical normalized text separately where needed.
- Use Unicode logical order.
- Preserve digit order.
- Never blindly reverse strings.
- Record Arabic-Indic versus Latin digits.
- Define diacritic scoring.
- Define tatweel, ligature, and presentation-form treatment.
- Preserve punctuation and numbering.

### 9.4 Equation rules

- Preserve image evidence.
- Use a documented canonical representation.
- Store OCR text and LaTeX/MathML separately when available.
- Mark ambiguous symbols rather than inventing certainty.

## 10. Dataset splits and leakage prevention

Create stable splits:

- `dev`;
- `validation`;
- `test-public`;
- optional `test-private`.

Split by source document, not page.

Prevent leakage across:

- the same exam hosted on different websites;
- alternate scans of the same exam;
- corrected and uncorrected copies;
- page images rendered from the same PDF;
- near-duplicate synthetic generations.

Track cryptographic and perceptual hashes.

## 11. BAC release strategy

Do not annotate everything first.

### BAC-Small v0.1

- 50–100 pages;
- balanced Arabic, French, mathematics, and physics;
- several years and streams;
- multiple document qualities;
- double-reviewed ground truth;
- text, reading order, digits, basic layout, and equation-region metrics.

### BAC-Core

- 500–1,000 pages;
- broader subjects and streams;
- text, layout, tables, equations, figures, and hierarchy;
- native and scanned documents.

### BAC-Challenge

- difficult scans;
- phone photos;
- compression;
- stamps and annotations;
- mixed scripts;
- complex equations and diagrams;
- controlled or hidden test subset where lawful.

Every release requires a dataset card, revision ID, and legal/source manifest.

## 12. Synthetic corpus

Create original exam-like documents that reproduce difficulty without copying protected exam content.

Generators should produce:

- Arabic instructions;
- French instructions;
- original mathematics questions;
- original physics/chemistry notation;
- tables;
- diagrams;
- official-looking but original headers and numbering;
- realistic stamps and scan damage;
- native and image-only PDFs;
- intentionally corrupted text layers.

All generated content must be original or permissively licensed.

Always report synthetic and real BAC scores separately.

## 13. Metrics implementation rules

- Every metric requires tests with hand-computed examples.
- Define normalization before scoring.
- Report micro and macro averages.
- Report sample counts.
- Report category scores.
- Report failed pages.
- Do not silently exclude crashes or timeouts.
- Record preprocessing/postprocessing.
- Do not tune on private test data.
- Version metric-definition changes.

## 14. Reproducibility

Every run records:

- benchmark version;
- dataset revision;
- manifest checksum;
- system name/version;
- git commit;
- model revisions;
- weight checksums;
- container digest;
- runtime;
- hardware;
- operating system;
- dependency lock hash;
- command;
- concurrency;
- warmup;
- timeout;
- failures;
- date.

A result without sufficient metadata is not leaderboard-eligible.

## 15. Leaderboard integrity

- Publish category scores, not only aggregate scores.
- Mark self-reported versus independently reproduced results.
- Mark hosted APIs.
- Mark VLM use.
- Mark BAC fine-tuning or training overlap.
- Require known training-overlap disclosure.
- Reject unverifiable or licence-violating submissions.
- Preserve historical benchmark revisions.

A higher aggregate score must not be presented as proof that a system is best for every use case.

## 16. Privacy and safety

Do not include:

- candidate answer sheets;
- names;
- registration numbers;
- signatures;
- student records;
- leaked or pre-release exams;
- files acquired through unauthorized access.

Redact accidental identifiers. Treat downloaded PDFs and images as hostile. Isolate rendering, validate file signatures and limits, and avoid executing embedded content.

## 17. Engineering standards

Use:

- Python 3.12+;
- `uv`;
- Pydantic v2;
- versioned JSON Schemas;
- pytest;
- Hypothesis;
- Ruff;
- strict Pyright or mypy;
- deterministic CLIs;
- Docker where reproducibility requires it.

Core scoring code must remain independent from heavy OCR frameworks. Framework-specific dependencies belong in adapters.

Unit tests must not download models or datasets.

## 18. Agent workflow

For every task:

1. Read this file.
2. Inspect repository state and git history.
3. Identify legal and provenance implications.
4. Write or update the implementation plan.
5. Use TDD for scoring, validation, and behavior.
6. Implement the smallest coherent change.
7. Verify format, lint, types, and tests.
8. Run the smallest meaningful benchmark smoke test.
9. Update dataset cards, sources, schemas, and notices.
10. Report exact evidence and limitations.

Do not:

- fabricate sources;
- invent ground truth;
- copy exam PDFs without verified permission;
- publish private test data;
- change metrics without versioning;
- remove failed samples from reports;
- optimize scoring for DzDoc;
- claim state of the art without fair evidence.

## 19. Delivery phases

### Phase A — Benchmark contract and synthetic smoke set

Deliver:

- repository foundation;
- schemas;
- CLI;
- manifest validation;
- synthetic Arabic/French/mixed pages;
- CER/WER, digit, and reading-order metrics;
- fake-system adapter;
- reproducible smoke report.

### Phase B — BAC acquisition manifests and BAC-Small

Deliver:

- official/authorized source research;
- reference-only manifests by default;
- safe downloader;
- duplicate detection;
- BAC taxonomy;
- annotation guidelines;
- 50–100 page BAC-Small candidate set;
- legal status for every source;
- double-reviewed ground truth where publication is allowed.

### Phase C — Layout, tables, equations, and comparison systems

Deliver:

- layout metrics;
- table metrics;
- equation metrics;
- adapters for DzDoc and major alternatives;
- BAC-Core;
- performance and confidence metrics;
- reproducible comparative report.

### Phase D — Leaderboard and challenge set

Deliver:

- submission format;
- validation;
- containerized evaluation;
- public/private split support;
- leaderboard generation;
- BAC-Challenge;
- transparent result cards.

## 20. Definition of done

A benchmark task is complete only when:

- schemas and metrics are tested;
- source and licence status are recorded;
- dataset revision is reproducible;
- failed samples are accounted for;
- reports include full metadata;
- limitations are explicit;
- no protected or private data was accidentally committed;
- fresh verification commands pass.

“Collected PDFs” is not a benchmark.
