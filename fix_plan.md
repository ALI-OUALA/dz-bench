I will fix the deterministic behavior issues explicitly called out in the memory.
There are two main issues:
1. `greedy_block_matches` in `src/dz_bench/metrics.py` (line 232):
```python
        best_index = max(
            remaining,
            key=lambda index: bounding_box_iou(reference_block.bbox, hypothesis[index].bbox),
        )
```
This takes the max of a `set` (unordered collection) based on a float key (`bounding_box_iou`). In case of a tie (which happens very frequently, e.g. when IOU is 0 for multiple candidates), the winner depends on the internal iteration order of the set, which is not strictly guaranteed to be deterministic after mutations. I will change the key to include a secondary tie-breaker, like so:
```python
        best_index = max(
            remaining,
            key=lambda index: (bounding_box_iou(reference_block.bbox, hypothesis[index].bbox), -index),
        )
```

2. `structured_field_scores` in `src/dz_bench/metrics.py` (line 177):
```python
    for name in set(reference_by_name) | set(hypothesis_by_name):
```
This iterates over a set of strings, which is non-deterministic depending on Python's hash seed (it changes across runs). Because floating-point arithmetic is performed inside the loop (`coordinate_scores.append(bounding_box_iou(...))`), the list `coordinate_scores` will contain the same floats but in a different order. Summing floats in different orders can yield slightly different results (non-commutative due to precision), violating deterministic constraints.
I will fix this by sorting the set:
```python
    for name in sorted(set(reference_by_name) | set(hypothesis_by_name)):
```

I will add a regression test for both of these to ensure deterministic behavior across runs with different string hashing seeds and set iteration behaviors.
