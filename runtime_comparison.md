# Learner 2 Runtime Comparison

Benchmarks were run without saving learner outputs, so the existing `Results/` baseline files were not overwritten.

## Runtime Results

### Q215380

| Stage | Old | Previous Optimized | Latest | Speedup vs Old | Speedup vs Previous | Time Saved vs Old |
|---|---:|---:|---:|---:|---:|---:|
| Oracle | `111.6214s` | `2.6039s` | `2.2685s` | `49.20x` | `1.15x` | `109.35s` |
| Propagation | `160.2587s` | `1.0789s` | `0.6066s` | `264.19x` | `1.78x` | `159.65s` |
| Total learner2 | `273.3701s` | `3.8107s` | `2.9998s` | `91.13x` | `1.27x` | `270.37s` |

Semantic counts stayed unchanged for `Q215380`.

### Q6256

| Stage | Old | Previous Optimized | Latest | Speedup vs Old | Speedup vs Previous | Time Saved vs Old |
|---|---:|---:|---:|---:|---:|---:|
| Oracle | `9055.8165s` | `6.7736s` | `5.1781s` | `1748.87x` | `1.31x` | `9050.64s` |
| Propagation | `426.4171s` | `1.8511s` | `1.4261s` | `299.01x` | `1.30x` | `424.99s` |
| Total learner2 | `9485.4482s` | `8.8631s` | `6.8450s` | `1385.75x` | `1.29x` | `9478.60s` |

Semantic counts stayed unchanged for `Q6256`.

### Q82955

| Stage | Old | Previous Optimized | Latest | Speedup vs Old | Speedup vs Previous | Time Saved vs Old |
|---|---:|---:|---:|---:|---:|---:|
| Oracle | `9783.7439s` | not run | `34.4111s` | `284.32x` | n/a | `9749.33s` |
| Propagation | `3528.2207s` | not run | `14.2261s` | `248.01x` | n/a | `3513.99s` |
| Total learner2 | `13322.7398s` | not run | `49.6066s` | `268.57x` | n/a | `13273.13s` |

Semantic counts stayed unchanged for `Q82955`.

## Why The Code Is Faster

The approach and semantic outputs are preserved. The speedups come from reducing Python overhead in the same computations.

### Oracle Phase

The oracle was the largest bottleneck. Previously, every property pair scanned entities and created an `AllenRelation` object for every pair of triples, then built a dictionary of all Allen axiom checks.

Changes:

- Added a reusable property/entity oracle index so each `Query()` call only iterates entities that actually contain both queried properties.
- Removed per-triple-pair `AllenRelation` object allocation.
- Replaced `check_all_axioms()` dictionary creation with direct interval comparisons that increment the same Allen relation counters.
- Added a chunked NumPy vectorized path for normal datetime intervals, so large triple cross-products are evaluated in array operations instead of Python nested loops.
- Store indexed interval endpoints as integer day values instead of datetime scalars, reducing NumPy comparison overhead.
- Kept a scalar fallback for unusual intervals such as missing bounds.

### Propagation Phase

Propagation repeatedly reads and updates QCN domains. Previously, it repeatedly rebuilt score/status dictionaries and repeatedly scanned the queue for membership.

Changes:

- Added a `queued` set next to the `deque`, making queue membership checks constant-time instead of linear-time.
- Replaced repeated hot-path calls to `relation_scores()`, `relation_statuses()`, and `relation_entries()` with direct payload access inside `propagateAndFilter()` where domains are already internally controlled.
- Replaced repeated full inverse-domain rebuilds with incremental converse relation updates through `ALLEN_CONVERSE`.
- Cached each domain's positive scores, composable scores, and observed-status flag, refreshing only changed pairs and their converses.
- Updated propagation caches incrementally for each changed relation instead of rescanning both changed domains.
- Added a memoized Allen composition fast path for repeated scored-domain compositions.

### Copy And Stats Overhead

The learner also spent time copying large QCN structures and scanning them multiple times for report statistics.

Changes:

- Removed safe unnecessary deep copies after oracle and propagation.
- Removed unnecessary deep copies of loaded entities/properties in `main()`.
- Removed the remaining oracle query-result deep copy and replaced validated converse rebuilding with a fast hot-path converse builder.
- Combined after-oracle QCN stats and report stats into one traversal.

## Verification

The optimized code was checked with:

```bash
PYTHONPATH=src .venv/bin/pytest
PYTHONPATH=src .venv/bin/python -m tclkg.ground_truth verify --include-slow
.venv/bin/ruff check src/tclkg/qcn_generator2.py
```
