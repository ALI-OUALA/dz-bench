from pathlib import Path

from fastapi.testclient import TestClient

from dz_bench.adapters.fake import FakeSystemAdapter
from dz_bench.api import app
from dz_bench.io import load_predictions, write_json
from dz_bench.runner import RunnerLimits, run_system
from dz_bench.synthetic import generate_corpus


def _bundle(tmp_path: Path):
    corpus = generate_corpus(seed=3, document_count=1, pages_per_document=1)
    write_json(corpus.manifest, tmp_path / "manifest.json")
    write_json(corpus.ground_truth, tmp_path / "ground-truth.json")
    return corpus


def test_runner_converts_crash_to_failure_samples(tmp_path: Path) -> None:
    corpus = _bundle(tmp_path)
    output = tmp_path / "predictions.json"
    result = run_system(
        ["python", "-c", "raise SystemExit(7)"],
        bundle_dir=tmp_path,
        output_path=output,
        system_name="crasher",
        system_version="1",
        limits=RunnerLimits(timeout_seconds=5),
    )
    predictions = load_predictions(output)
    assert result.status == "crashed"
    assert len(predictions.samples) == sum(len(doc.pages) for doc in corpus.manifest.documents)
    assert {sample.status for sample in predictions.samples} == {"crashed"}


def test_runner_rejects_invalid_success_output(tmp_path: Path) -> None:
    _bundle(tmp_path)
    output = tmp_path / "predictions.json"
    command = [
        "python",
        "-c",
        "from pathlib import Path; Path(r'{predictions}').write_text('[]')",
    ]
    result = run_system(
        command,
        bundle_dir=tmp_path,
        output_path=output,
        system_name="invalid",
        system_version="1",
    )
    assert result.status == "invalid"
    assert load_predictions(output).samples[0].error.code == "runner_invalid"


def test_public_api_scores_contract_artifacts(tmp_path: Path) -> None:
    corpus = _bundle(tmp_path)
    prediction_path = write_json(
        FakeSystemAdapter().predict(corpus.ground_truth), tmp_path / "predictions.json"
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/score",
            files={
                "manifest": ("manifest.json", (tmp_path / "manifest.json").read_bytes()),
                "ground_truth": (
                    "ground-truth.json",
                    (tmp_path / "ground-truth.json").read_bytes(),
                ),
                "predictions": ("predictions.json", prediction_path.read_bytes()),
            },
        )
    assert response.status_code == 200
    assert response.json()["summary"]["scored_pages"] == 1
