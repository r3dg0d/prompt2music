"""Acceptance tests with real weights on real hardware (opt-in).

Run on the GPU workstation after installing the backend env:

    prompt2music backends install ace-step --yes
    pytest -m "gpu and network" tests/test_gpu_acceptance.py

PROMPT2MUSIC_ACCEPT_MODEL picks the model (default ace-step-1.5). These tests
download several GB, so they never run in ordinary CI.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.network]
MODEL = os.environ.get("PROMPT2MUSIC_ACCEPT_MODEL", "ace-step-1.5")
EXE = shutil.which("prompt2music") or "prompt2music"


def cli(*args: str, timeout: int = 7200) -> subprocess.CompletedProcess[str]:
    return subprocess.run([EXE, *args], capture_output=True, text=True, timeout=timeout,
                          check=False)


def test_doctor_and_catalogue():
    assert cli("--help").returncode == 0
    report = json.loads(cli("--json", "doctor").stdout)
    assert report["gpus"], "no NVIDIA GPU visible"
    assert MODEL in {m["id"] for m in json.loads(cli("--json", "models", "available").stdout)}


def test_download_and_verify():
    proc = cli("models", "download", MODEL, "--yes", "--accept-license")
    assert proc.returncode == 0, proc.stderr
    proc = cli("models", "verify", MODEL)
    assert proc.returncode == 0, proc.stderr


def test_generate_real_audio(tmp_path):
    out = tmp_path / "test.wav"
    proc = cli("--json", "generate", "Beautiful ambient piano with soft strings",
               "--model", MODEL, "--duration", "20", "--seed", "42", "-o", str(out),
               "--no-daemon")
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    item = result["outputs"][0]
    assert Path(item["path"]).is_file()
    assert 8000 <= item["sample_rate"] <= 192000
    assert item["channels"] in (1, 2)
    assert abs(item["duration"] - 20) / 20 < 0.15
    assert item["validation"]["ok"], item["validation"]  # decodes, not silent
    assert Path(item["sidecar"]).is_file()
    print("RTF", result["rtf"], "memory", result["memory"])


def test_benchmark_writes_measured_json(tmp_path):
    out = tmp_path / "bench.json"
    proc = cli("--json", "benchmark", "--model", MODEL, "--duration", "30", "--runs", "1",
               "-o", str(out))
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    run = data["results"][0]
    assert run["rtf_mean"] > 0 and run["peak_vram_mb"]
