# Learner 2 Ground Truth

This directory stores semantic ground truth for `tclkg.qcn_generator2` before
performance optimization work.

## Scope

The manifests validate the Learner 2 result semantics:

- `initial`
- `after_oracle`
- `after_propagation`
- status counts
- support counts
- oracle-to-propagation transition counts
- canonical hashes of QCN sections

They intentionally do not validate runtime fields, timestamps, logs, or plot
outputs.

## Datasets

Mandatory full-result baselines:

- `Q6256`
- `Q215380`

Optional slow baseline:

- `Q82955`

The `Q82955` manifest is marked `optional_slow: true`. Its input checksum is
currently `null` because `/home/paris/dev/py/Temporal-Constraint-Learning-KGs/data`
does not contain a `Q82955` quintuplet source file. The manifest still protects
the existing `Results/Q82955/qcn2_Q82955.json` semantics.

## Input Files

The loader requires one of these files per KG:

- `data/<KG>/data.quintuplet`
- `data/<KG>/train_cst_knowledge.quintuplet`

The mandatory `Q6256` and `Q215380` files were copied from:

```text
/home/paris/dev/py/Temporal-Constraint-Learning-KGs/data
```

Large quintuplet files are ignored by Git through `.gitignore`.

## Verification

Run fast semantic tests:

```bash
pytest tests/test_ground_truth_unit.py tests/test_ground_truth_mini_pipeline.py
```

Run mandatory full-result verification:

```bash
python -m tclkg.ground_truth verify
```

Run optional slow verification too:

```bash
python -m tclkg.ground_truth verify --include-slow
```

If the package is installed, the equivalent console script is:

```bash
tclkg-ground-truth verify --include-slow
```

## Regeneration

Regenerate mandatory manifests after intentionally changing semantics:

```bash
python -m tclkg.ground_truth generate
```

Regenerate all manifests, including optional slow baselines:

```bash
python -m tclkg.ground_truth generate --include-slow
```

Only regenerate manifests when the intended behavior changes. For performance
optimizations, manifests should remain unchanged.

## Optimization Workflow

For each optimization change:

1. Run the fast unit and mini-pipeline tests.
2. Run mandatory full verification for `Q6256` and `Q215380`.
3. Run `--include-slow` for major refactors or final validation.
4. Compare performance separately from semantic correctness.
5. Accept the optimization only if semantic verification still passes.
