"""Automated test for CLI entrypoint."""

import subprocess
import sys
from pathlib import Path


def test_cli_execution(tmp_path):
    out_dir = tmp_path / "test_cli_out"
    cmd = [
        sys.executable,
        "run.py",
        "--scenario",
        "scenarios/R.json",
        "--seed",
        "42",
        "--out",
        str(out_dir),
        "--max-evals",
        "30",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI failed with error:\n{res.stderr}"

    # Verify all expected files are present
    expected_files = [
        "qubo.json",
        "circuit.qasm",
        "counts.json",
        "schedule.json",
        "metrics.json",
        "baselines.json",
        "training.json",
        "surrogate_validation.json",
        "run.log",
    ]
    for fname in expected_files:
        assert (out_dir / fname).is_file(), f"Expected file {fname} not created"
