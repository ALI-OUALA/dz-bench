from dz_bench.metrics import greedy_block_matches
from dz_bench.models import Block, BoundingBox


def test_greedy_block_matches_determinism_with_tiebreaker():
    # If two hypothesis blocks overlap identically with a reference block,
    # the tie should be broken deterministically by index.

    reference = [
        Block(
            block_id="ref1",
            block_type="paragraph",
            reading_order_index=0,
            confidence={"score": 1.0},
            provenance={"kind": "system_prediction", "source": "test"},
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            lines=[],
        )
    ]

    hypothesis = [
        Block(
            block_id="hyp1",
            block_type="paragraph",
            reading_order_index=0,
            confidence={"score": 1.0},
            provenance={"kind": "system_prediction", "source": "test"},
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            lines=[],
        ),
        Block(
            block_id="hyp2",
            block_type="paragraph",
            reading_order_index=1,
            confidence={"score": 1.0},
            provenance={"kind": "system_prediction", "source": "test"},
            bbox=BoundingBox(x=0, y=0, width=10, height=10),
            lines=[],
        ),
    ]

    matches = greedy_block_matches(reference, hypothesis)
    assert len(matches) == 1
    assert matches[0].hypothesis is not None
    assert matches[0].hypothesis.block_id == "hyp1"
