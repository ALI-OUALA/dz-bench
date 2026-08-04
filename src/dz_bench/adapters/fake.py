"""Deterministic fixture system for contract and scorer smoke tests."""

from __future__ import annotations

from ..models import (
    GroundTruth,
    Predictions,
    PredictionSample,
    ProcessingError,
    RunMetadata,
    SystemMetadata,
)


class FakeSystemAdapter:
    """Return ground truth as predictions, with optional deterministic failures."""

    name = "fake-ground-truth"
    version = "1.0.0"

    def predict(self, ground_truth: GroundTruth, *, fail_every: int = 0) -> Predictions:
        if fail_every < 0:
            raise ValueError("fail_every must be non-negative")
        samples: list[PredictionSample] = []
        counter = 0
        for document in ground_truth.documents:
            for page in document.pages:
                counter += 1
                if fail_every and counter % fail_every == 0:
                    samples.append(
                        PredictionSample(
                            document_id=document.document_id,
                            page_id=page.page_id,
                            status="crashed",
                            error=ProcessingError(
                                code="fake_failure", message="deterministic fake adapter failure"
                            ),
                        )
                    )
                else:
                    samples.append(
                        PredictionSample(
                            document_id=document.document_id,
                            page_id=page.page_id,
                            status="success",
                            page=page,
                        )
                    )
        return Predictions(
            dataset_revision=ground_truth.dataset_revision,
            coordinate_system=ground_truth.coordinate_system,
            system=SystemMetadata(
                name=self.name,
                version=self.version,
                adapter_name=self.name,
                adapter_version=self.version,
                execution_provider="cpu",
            ),
            run=RunMetadata(run_id="fake-run"),
            samples=samples,
        )


__all__ = ["FakeSystemAdapter"]
