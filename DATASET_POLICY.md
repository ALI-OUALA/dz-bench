# Dataset policy

DZ-Bench code and benchmark documents are separate legal objects. Public benchmark media is not assumed to be redistributable merely because a URL is public.

The default treatment for an Algerian Baccalaureate or other third-party document is **reference-only**. A checked-in reference record may contain provenance metadata, a canonical source URL when independently verified, retrieval date, expected checksum, licence status, and downloader instructions. It must not contain the protected PDF, page images, answer sheets, student identifiers, signatures, or copied exam text unless written redistribution permission is recorded.

The repository currently ships no protected BAC PDF and no BAC ground truth. The checked-in BAC manifest is a template with no asserted source URL or checksum. Rights-unclear material remains local/private and is excluded by `.gitignore`.

Synthetic Phase A records are authored for this project and are deliberately not transcriptions or paraphrases of exam papers. Synthetic scores must always be reported separately from real BAC scores.

For a future source record, reviewers must verify:

- source organization, title, year/session/stream/subject, and canonical URL;
- retrieval date, file size, SHA-256 checksum, and observed notice;
- redistribution status and any restrictions on derivatives or ground truth;
- split assignment by source document, with duplicate and near-duplicate checks.

If a rights holder requests removal, stop distribution of the affected artifact, preserve only the minimum provenance needed to identify it, and contact the repository owner through the project’s authorized channel. Do not replace a removed source with an invented transcription.

