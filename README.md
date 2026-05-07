# ISWC Submission

This repository contains the Learner 2 pipeline for temporal constraint learning on knowledge graphs.

## Run Learner 2

### 1. Prerequisites

- Linux/macOS shell (or equivalent terminal)
- Python 3.12+
- A virtual environment

### 2. Setup

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Verify the CLI is available

```bash
python -m tclkg.run_experiments --help
python -m tclkg.qcn_generator2 --help
```

If these commands print usage text, your environment is ready.

## Execution Modes

### A) Run configured Learner 2 experiment batch

```bash
python -m tclkg.run_experiments
```

Optional: run only one KG from the configured list:

```bash
python -m tclkg.run_experiments --kg Q6256
```

Optional: limit parallel jobs:

```bash
python -m tclkg.run_experiments --max-parallel 2
```

The batch configuration is defined in:

- `src/tclkg/run_experiments.py` (`EXPERIMENTS_GEN2`)

### B) Run one Learner 2 experiment directly

```bash
python -m tclkg.qcn_generator2 Q6256 3600
```

Arguments:

- `kg`: one of `Q6256`, `Q215380`, `Q82955`
- `timeout`: timeout in seconds

## Input Data

By default, data is read from:

- `data/<KG>/data.quintuplet`

If missing, it falls back to:

- `data/<KG>/train_cst_knowledge.quintuplet`

You can override the data root with:

```bash
export TCLKG_DATA_DIR=/absolute/path/to/data
```

## Output Files

### Direct Learner 2 run (`tclkg.qcn_generator2`)

- JSON QCN output: `Results/<KG>/qcn2_<KG>.json`
- Statistics report: `Results/<KG>/qcn2_<KG>_stats.txt`

### Batch run (`tclkg.run_experiments`)

- Per-run logs: `logs_experiments/`

## Troubleshooting

If `python -m tclkg...` loads code from another repository checkout, reinstall in editable mode from this project root:

```bash
pip uninstall -y temporal-constraint-learning-kgs
pip install -e .
```

Then validate import resolution:

```bash
python -c "import tclkg; print(tclkg.__file__)"
```

The printed path should point to this repository.