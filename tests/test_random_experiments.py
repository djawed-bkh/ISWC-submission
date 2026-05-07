from __future__ import annotations

import json

from tclkg import random_experiments as rexp


def test_check_consistency_loads_after_oracle_and_runs_classic_pc(tmp_path) -> None:
    qcn_path = tmp_path / "sample_qcn.json"
    qcn_payload = {
        "after_oracle": {
            "p1___p2": {"before": 1.0},
            "p2___p1": {"after": 1.0},
            "p2___p3": {"before": 1.0},
            "p3___p2": {"after": 1.0},
            "p1___p3": {"before": 0.2, "meets": 0.9},
            "p3___p1": {"after": 0.2, "met_by": 0.9},
        }
    }
    qcn_path.write_text(json.dumps(qcn_payload), encoding="utf-8")

    assert rexp.checkConsistency(str(qcn_path), threshold=0.5) is True


def test_check_consistency_returns_false_when_classic_pc_collapses(tmp_path) -> None:
    qcn_path = tmp_path / "inconsistent_qcn.json"
    qcn_payload = {
        "after_oracle": {
            "p1___p2": {"before": 1.0},
            "p2___p1": {"after": 1.0},
            "p2___p3": {"before": 1.0},
            "p3___p2": {"after": 1.0},
            "p1___p3": {"after": 1.0},
            "p3___p1": {"before": 1.0},
        }
    }
    qcn_path.write_text(json.dumps(qcn_payload), encoding="utf-8")

    assert rexp.checkConsistency(str(qcn_path), threshold=0.5) is False