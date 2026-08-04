# DzDoc PP-OCRv5 routing checkpoint — 2026-08-04

This is a measured development checkpoint, not a production accuracy claim. It uses eight original DZ-Bench BAC-style PNG pages (`bac-synthetic-images@0.3.0`) on Windows 11, Python 3.13.7, 16 logical CPUs. All runs used the same detector and images; only recognition routing changed.

| System | CER ↓ | WER ↓ | Digit exact ↑ | Order ↑ | Runtime/page ↓ | Peak RSS ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DzDoc routed Arabic/Latin | **0.1100** | **0.2427** | 0.6250 | 0.4314 | 3506 ms | 355.70 MiB |
| Arabic recognizer only | 0.1353 | 0.2670 | 0.5000 | 0.4314 | 3271 ms | 348.93 MiB |
| Latin recognizer only | 0.3399 | 0.4175 | **0.7500** | 0.4314 | **3009 ms** | **348.25 MiB** |

The routed cascade reduced character error by 18.7% relative to Arabic-only and 67.6% relative to Latin-only on this mixed suite. Its higher latency is the measured cost of uncertain-region fusion. The small digit sample favors Latin-only, so digit routing remains unresolved.

Additional routed results: normalized edit similarity 0.8900, layout match F1 0.5366, block-type accuracy 0.2157, equation similarity 0.3333, and table-structure similarity 0.0000. Category CER was Arabic 0.1037, French 0.0372, mathematics 0.0609, mixed 0.4211, and physics 0.1228.

Model revisions:

- `PP-OCRv5_mobile_det@0d63e78e2b680928f6b1747d76a08db6e645efb7`
- `arabic_PP-OCRv5_mobile_rec@33d91636a65dca87f5562cc48860332ae367ee1b`
- `latin_PP-OCRv5_mobile_rec@ab2cd5cc5fa6309be2e5acdfe66eca2c2c127d57`

Limitations: eight synthetic pages are enough to reject the single-recognizer baselines, not to estimate real BAC accuracy. Table reconstruction is absent, equation recognition is weak, mixed-script routing is the main text failure, and layout/order scores remain low. Rights-unclear real BAC files are provenance-only until lawful ground truth can be produced.
