"""FFmpeg toolkit, prompt enhancement, lyrics and local-LLM guards."""

from __future__ import annotations

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from prompt2music import audio as A
from prompt2music import lyrics, prompts
from prompt2music.errors import ConfigError, InvalidRequestError
from prompt2music.llm import get_llm
from tests.conftest import needs_ffmpeg


def make_tone(path: Path, seconds: float = 3.0, freq: int = 440, rate: int = 44100) -> Path:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency={freq}:duration={seconds}:sample_rate={rate}",
                    "-ac", "2", str(path)], check=True)
    return path


def make_silence(path: Path, seconds: float = 2.0) -> Path:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "anullsrc=r=44100:cl=stereo", "-t", str(seconds), str(path)], check=True)
    return path


@needs_ffmpeg
class TestAudioToolkit:
    def test_info_levels_and_validation(self, tmp_path):
        src = make_tone(tmp_path / "a.wav")
        meta = A.info(src)
        assert (meta.sample_rate, meta.channels) == (44100, 2)
        assert meta.duration == pytest.approx(3.0, abs=0.05)
        assert A.levels(src)["max_volume_db"] == pytest.approx(-21.1, abs=1.0)  # lavfi sine
        assert A.validate_audio(src, expected_duration=3.0)["ok"]
        bad = A.validate_audio(src, expected_duration=10.0)
        assert not bad["ok"] and "duration" in bad["problems"][0]

    def test_silence_detected(self, tmp_path):
        report = A.validate_audio(make_silence(tmp_path / "s.wav"))
        assert not report["ok"] and "silent" in report["problems"][0]

    @pytest.mark.parametrize("fmt,codec", [("flac", "flac"), ("mp3", "mp3"),
                                           ("ogg", "vorbis"), ("opus", "opus")])
    def test_convert_formats_with_metadata(self, tmp_path, fmt, codec):
        src = make_tone(tmp_path / "a.wav", seconds=1)
        dest = A.convert(src, tmp_path / f"a.{fmt}", sample_rate=48000 if fmt == "opus" else None,
                         metadata={"title": "Demo"})
        meta = A.info(dest)
        assert meta.codec == codec
        assert meta.tags.get("title") == "Demo"

    def test_resample_and_downmix(self, tmp_path):
        src = make_tone(tmp_path / "a.wav", seconds=1)
        out = A.info(A.convert(src, tmp_path / "b.wav", sample_rate=22050, channels=1))
        assert (out.sample_rate, out.channels) == (22050, 1)

    def test_normalize_peak_and_loudness(self, tmp_path):
        src = make_tone(tmp_path / "a.wav")
        peak = A.normalize(src, tmp_path / "p.wav", mode="peak", target=-3.0)
        assert A.levels(tmp_path / "p.wav")["max_volume_db"] == pytest.approx(-3.0, abs=0.3)
        assert peak["gain_db"] != 0
        loud = A.normalize(src, tmp_path / "l.wav", mode="loudness", target=-16.0)
        assert "output_lufs" in loud
        with pytest.raises(InvalidRequestError):
            A.normalize(make_silence(tmp_path / "s.wav"), tmp_path / "x.wav", mode="peak")

    def test_trim_fade_loop_mix_duck_fit(self, tmp_path):
        src = make_tone(tmp_path / "a.wav")
        assert A.info(A.trim(src, tmp_path / "t.wav", start=0.5, end=2.0)).duration == \
            pytest.approx(1.5, abs=0.05)
        assert A.info(A.fade(src, tmp_path / "f.wav", fade_in=1, fade_out=1)).duration == \
            pytest.approx(3.0, abs=0.05)
        assert A.info(A.loop(src, tmp_path / "l.wav", count=3)).duration == \
            pytest.approx(9.0, abs=0.1)
        assert A.info(A.loop(src, tmp_path / "lx.wav", count=3, crossfade=0.5)).duration == \
            pytest.approx(8.0, abs=0.1)
        other = make_tone(tmp_path / "b.wav", seconds=5, freq=220)
        assert A.info(A.mix([src, other], tmp_path / "m.wav")).duration == \
            pytest.approx(5.0, abs=0.1)
        assert A.info(A.duck(src, other, tmp_path / "d.wav")).duration == \
            pytest.approx(3.0, abs=0.1)
        assert A.info(A.fit_duration(src, tmp_path / "fit.wav", seconds=7)).duration == \
            pytest.approx(7.0, abs=0.05)
        assert A.info(A.crossfade(src, other, tmp_path / "c.wav", seconds=1)).duration == \
            pytest.approx(7.0, abs=0.1)

    def test_images(self, tmp_path):
        src = make_tone(tmp_path / "a.wav", seconds=1)
        for fn in (A.waveform, A.spectrogram):
            out = fn(src, tmp_path / f"{fn.__name__}.png")
            assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_never_overwrites_and_no_partial_files(self, tmp_path):
        src = make_tone(tmp_path / "a.wav", seconds=1)
        dest = tmp_path / "b.flac"
        A.convert(src, dest)
        with pytest.raises(InvalidRequestError, match="exists"):
            A.convert(src, dest)
        broken = tmp_path / "broken.wav"
        broken.write_bytes(b"not audio")
        with pytest.raises(Exception):
            A.convert(broken, tmp_path / "out.mp3")
        assert not list(tmp_path.glob(".p2m-*"))
        assert not (tmp_path / "out.mp3").exists()

    def test_bad_format_rejected(self, tmp_path):
        with pytest.raises(InvalidRequestError, match="unsupported"):
            A.format_for(tmp_path / "a.xyz")


# -- prompts -------------------------------------------------------------------------
@pytest.mark.parametrize("prompt", [
    "sad piano music",
    "Aggressive industrial techno, 145 BPM, distorted kick drums, dark atmosphere",
    "Epic orchestral boss battle music with aggressive strings and cinematic percussion",
    "Dreamy ambient electronic music with shimmering synthesizers",
])
def test_enhancement_preserves_intent(prompt):
    out = prompts.enhance_rules(prompt)
    assert out.prompt.startswith(prompt.rstrip(" .,"))
    assert prompts.preserves_intent(prompt, out.prompt)
    neg_words = set(out.negative_prompt.replace(",", " ").split())
    for inst in prompts.detect(prompt)["instruments"]:
        assert inst not in neg_words


def test_enhancement_example_from_spec():
    out = prompts.enhance_rules("sad piano music")
    assert "melancholic" in out.prompt and "piano" in out.prompt
    assert "vocals" in out.negative_prompt


def test_negative_prompt_respects_requests():
    neg = prompts.enhance_rules("ambient with drums").negative_prompt
    assert "drums" not in neg
    assert "vocals" not in prompts.enhance_rules("pop song", instrumental=False).negative_prompt


def test_fold_controls_and_presets():
    assert prompts.fold_controls_into_prompt("techno", bpm=145, key="Fm") == \
        "techno, 145 BPM, in Fm"
    assert prompts.fold_controls_into_prompt("techno 145 BPM", bpm=145, key=None) == \
        "techno 145 BPM"
    assert prompts.apply_preset("boss-battle")["bpm"] == 150
    with pytest.raises(InvalidRequestError):
        prompts.apply_preset("nope")


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, system, user, max_tokens=600):
        return self.reply


def test_llm_output_rejected_if_it_changes_genre():
    out = prompts.enhance("sad piano music", llm=FakeLLM("Upbeat reggae with steel drums"))
    assert out.method.startswith("rules")
    good = prompts.enhance("sad piano music",
                           llm=FakeLLM("A sad, slow piano piece with soft reverb"))
    assert good.method == "llm"


# -- lyrics -------------------------------------------------------------------------
def test_lyrics_template_and_validation():
    tpl = lyrics.template("pop", topic="city lights")
    assert "[Chorus]" in tpl
    assert not lyrics.validate(tpl).ok  # placeholders present
    good = "[Verse 1]\nWalking home\n[Chorus]\nLights stay on\n"
    report = lyrics.validate(good)
    assert report.ok and report.sections == ["Verse 1", "Chorus"]
    assert lyrics.validate("[Wobble]\nla\n").warnings
    assert not lyrics.validate("x" * 5000, backend="ace-step").ok


def test_lyrics_generation_requires_tags():
    with pytest.raises(InvalidRequestError):
        lyrics.generate_lyrics(FakeLLM("just prose"), "a city")
    assert "[Chorus]" in lyrics.generate_lyrics(FakeLLM("[Verse 1]\na\n[Chorus]\nb"), "x")


# -- local LLM providers -------------------------------------------------------------
def test_llm_disabled_by_default(cfg):
    assert get_llm(cfg) is None
    with pytest.raises(ConfigError, match="no local language model"):
        get_llm(cfg, required=True)


def test_remote_llm_endpoint_refused(cfg):
    cfg.set("llm.provider", "llamacpp")
    cfg.set("llm.endpoint", "https://api.example.com")
    with pytest.raises(ConfigError, match="not local"):
        get_llm(cfg)


def test_ollama_provider_against_local_server(cfg):
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server API
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.update(body)
            payload = json.dumps({"response": "[Verse 1]\nhello\n[Chorus]\nworld"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cfg.set("llm.provider", "ollama")
        cfg.set("llm.model", "llama3.2")
        cfg.set("llm.endpoint", f"http://127.0.0.1:{server.server_port}")
        text = lyrics.generate_lyrics(get_llm(cfg), "the sea")
        assert "[Chorus]" in text
        assert seen["model"] == "llama3.2" and seen["stream"] is False
    finally:
        server.shutdown()
