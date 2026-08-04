from dz_bench.adapters.fake import FakeSystemAdapter
from dz_bench.scoring import score
from dz_bench.synthetic import generate_corpus


def test_fake_adapter_produces_exact_smoke_predictions():
    corpus = generate_corpus(seed=5, document_count=2)
    predictions = FakeSystemAdapter().predict(corpus.ground_truth)
    report = score(corpus.manifest, corpus.ground_truth, predictions)
    assert report.summary.scored_pages == report.summary.total_pages
    assert report.metrics["cer"].micro == 0


def test_fake_adapter_can_leave_failure_evidence():
    corpus = generate_corpus(seed=5, document_count=2)
    predictions = FakeSystemAdapter().predict(corpus.ground_truth, fail_every=2)
    report = score(corpus.manifest, corpus.ground_truth, predictions)
    assert report.summary.crashed_pages == 1
    assert report.failures[0].status == "crashed"
