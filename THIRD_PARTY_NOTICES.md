# Third-party notices

The Phase A runtime keeps dependencies small:

| Package | Use | Licence |
| --- | --- | --- |
| Pydantic 2.x | Typed contract models and validation | MIT |
| Typer 0.x | CLI argument handling | MIT |
| jsonschema 4.x | Draft 2020-12 public-contract validation | MIT |
| referencing 0.x | In-memory JSON Schema reference registry | MIT |
| pytest 8.x (development) | Tests | MIT |
| Ruff 0.x (development) | Lint and formatting | MIT |
| Pyright 1.x (development) | Static type checking | MIT |
| Pillow 10.x/11.x (optional `raster`) | Deterministic PNG rendering | MIT-CMU |
| arabic-reshaper 3.x (optional `raster`) | Arabic presentation shaping for fixtures | MIT |
| python-bidi 0.6.x (optional `raster`) | Unicode bidirectional display ordering for fixtures | LGPL |

Python itself is distributed under the Python Software Foundation License. Optional raster packages are installed separately and are not copied into this repository. No OCR model, model weight, protected document, copied image, font, or third-party code asset is bundled.

Dependency versions and licences must be rechecked before adding adapters, datasets, models, or generated assets. Model and dataset licences are independent of these package notices.
