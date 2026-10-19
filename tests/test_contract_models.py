"""Contract tests against real upstream inference code with random weights.

These run the actual transformers MusicGen and Demucs code paths through the
full worker → engine → pipeline stack, using tiny randomly initialised
checkpoints built locally (no download). The audio is noise; the tests prove
API compatibility, shapes, sample rates, durations, continuation and
cancellation — not musical quality. Real-weight runs are in
``test_gpu_acceptance.py`` (``-m gpu``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prompt2music import service
from prompt2music.engine import Engine
from prompt2music.manager import ModelManager
from tests.conftest import needs_ffmpeg

pytestmark = [pytest.mark.slow, needs_ffmpeg]


@pytest.fixture(scope="module")
def tiny_musicgen(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tests.tiny_musicgen import build_tiny_musicgen

    return build_tiny_musicgen(tmp_path_factory.mktemp("ckpt") / "musicgen")


def test_musicgen_pipeline_text_long_form_and_continue(cfg, tiny_musicgen):
    ModelManager(cfg).import_local("musicgen-small", tiny_musicgen, license_accepted=True)
    engine = Engine(cfg)
    try:
        params = service.normalise_params(cfg, {
            "prompt": "ambient piano music", "model": "musicgen-small", "duration": 35,
            "seed": 3, "bpm": 70})
        result = service.run_generate(cfg, engine, params)
        out = result["outputs"][0]
        assert out["sample_rate"] == 32000 and out["channels"] == 1
        assert out["duration"] == pytest.approx(35.0, abs=0.01)  # windowed long-form
        assert out["validation"]["ok"], out["validation"]
        assert result["prompt_used"] == "ambient piano music, 70 BPM"

        cont = service.run_generate(cfg, engine, service.normalise_params(cfg, {
            "task": "continue", "prompt": "dark synth", "model": "musicgen-small",
            "duration": 5, "input_audio": out["path"], "continue_mode": "outro"}))
        assert cont["outputs"][0]["duration"] == pytest.approx(40.0, abs=0.01)
        assert "winding down" in cont["prompt_used"]
        assert cont["load"]["reused"] is True  # same warm worker
        side = json.loads(Path(cont["outputs"][0]["sidecar"]).read_text())
        assert side["model_revision"].endswith("@local")
    finally:
        engine.unload()


def test_musicgen_batch_and_auto_selection(cfg, tiny_musicgen):
    ModelManager(cfg).import_local("musicgen-small", tiny_musicgen, license_accepted=True)
    engine = Engine(cfg)
    try:
        result = service.run_generate(cfg, engine, service.normalise_params(cfg, {
            "prompt": "drums", "model": "auto", "duration": 2, "batch_size": 2,
            "device": "cpu"}))
    finally:
        engine.unload()
    assert result["model"] == "musicgen-small" and result["selection"]["automatic"]
    assert len(result["outputs"]) == 2


def test_demucs_separation_contract(cfg, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("demucs")
    from prompt2music.backends.base import CancelToken
    from prompt2music.backends.demucs_stems import DemucsBackend
    from prompt2music.registry import get_model
    from tests.test_audio_prompts import make_tone

    spec = get_model("htdemucs")
    backend = DemucsBackend(spec.id, None, {"demucs_name": "demucs_unittest"},
                            spec.capabilities)
    backend.load("cpu", "fp32")
    from prompt2music import audio

    wav = audio.to_wav(make_tone(tmp_path / "in.wav", seconds=2), tmp_path / "in44.wav",
                       sample_rate=44100, channels=2)
    out = backend.separate(wav, tmp_path / "stems", ["vocals", "drums"],
                           lambda f, m: None, CancelToken())
    assert set(out["stems"]) == {"vocals", "drums"}
    assert set(out["available"]) == {"drums", "bass", "other", "vocals"}
    for path in out["stems"].values():
        assert audio.info(Path(path)).duration == pytest.approx(2.0, abs=0.05)
