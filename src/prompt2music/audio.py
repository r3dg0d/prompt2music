"""Audio post-processing with FFmpeg.

These are deterministic DSP operations applied after generation; they are
recorded separately from generation metadata. Peak/loudness normalisation is
level adjustment, not mastering.

Every FFmpeg call is an argv list (never a shell string). Outputs are written
to a temporary file next to the destination and renamed into place, so a
failed run never leaves a truncated file behind.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from prompt2music.errors import InvalidRequestError, MusicError, ToolMissingError

FORMATS: dict[str, dict[str, Any]] = {
    "wav": {"codec": "pcm_s16le", "ext": ".wav"},
    "flac": {"codec": "flac", "ext": ".flac"},
    "mp3": {"codec": "libmp3lame", "ext": ".mp3", "bitrate": "320k"},
    "ogg": {"codec": "libvorbis", "ext": ".ogg", "quality": "6"},
    "opus": {"codec": "libopus", "ext": ".opus", "bitrate": "192k"},
}
SILENCE_DB = -60.0


def ffmpeg_bin() -> str:
    exe = os.environ.get("PROMPT2MUSIC_FFMPEG") or shutil.which("ffmpeg")
    if not exe:
        raise ToolMissingError("ffmpeg is not installed",
                               hint="install ffmpeg (NixOS: pkgs.ffmpeg) and retry")
    return exe


def ffprobe_bin() -> str:
    exe = os.environ.get("PROMPT2MUSIC_FFPROBE") or shutil.which("ffprobe")
    if not exe:
        raise ToolMissingError("ffprobe is not installed", hint="it ships with ffmpeg")
    return exe


def _run(argv: list[str], *, timeout: float = 3600) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise MusicError(f"{Path(argv[0]).name} timed out after {timeout:.0f}s") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        raise MusicError(f"{Path(argv[0]).name} failed: " + " | ".join(tail))
    return proc


def format_for(path: Path, override: str | None = None) -> str:
    fmt = (override or path.suffix.lstrip(".")).lower()
    if fmt == "oga":
        fmt = "ogg"
    if fmt not in FORMATS:
        raise InvalidRequestError(f"unsupported audio format {fmt!r}; "
                                  f"use one of {', '.join(FORMATS)}")
    return fmt


def _require_input(path: Path) -> Path:
    if not path.is_file():
        raise InvalidRequestError(f"input file not found: {path}")
    return path


def _encode_args(fmt: str, bitrate: str | None = None) -> list[str]:
    spec = FORMATS[fmt]
    args = ["-c:a", spec["codec"]]
    if fmt in ("mp3", "opus") and (bitrate or spec.get("bitrate")):
        args += ["-b:a", bitrate or spec["bitrate"]]
    if fmt == "ogg":
        args += ["-q:a", spec["quality"]] if not bitrate else ["-b:a", bitrate]
    return args


class _AtomicOutput:
    """Context manager yielding a temp path that is renamed onto ``dest`` on success."""

    def __init__(self, dest: Path, overwrite: bool) -> None:
        self.dest = dest
        self.overwrite = overwrite

    def __enter__(self) -> Path:
        if self.dest.exists() and not self.overwrite:
            raise InvalidRequestError(f"{self.dest} exists (use --force to overwrite)")
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".p2m-", suffix=self.dest.suffix,
                                   dir=self.dest.parent)
        os.close(fd)
        self.tmp = Path(tmp)
        return self.tmp

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            os.replace(self.tmp, self.dest)
        else:
            self.tmp.unlink(missing_ok=True)


def _ffmpeg(inputs: Sequence[Path], dest: Path, filters: list[str], *, fmt: str | None = None,
            overwrite: bool = False, pre_input: Sequence[str] = (),
            extra: Sequence[str] = (), filter_complex: str | None = None,
            metadata: dict[str, str] | None = None, bitrate: str | None = None) -> Path:
    fmt = format_for(dest, fmt)
    argv = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
    for p in inputs:
        argv += [*pre_input, "-i", str(_require_input(p))]
    if filter_complex:
        argv += ["-filter_complex", filter_complex]
    elif filters:
        argv += ["-af", ",".join(filters)]
    argv += list(extra)
    for key, value in (metadata or {}).items():
        argv += ["-metadata", f"{key}={value}"]
    argv += _encode_args(fmt, bitrate) + ["-f", _muxer(fmt)]
    with _AtomicOutput(dest, overwrite) as tmp:
        _run(argv + [str(tmp)])
    return dest


def _muxer(fmt: str) -> str:
    return {"wav": "wav", "flac": "flac", "mp3": "mp3", "ogg": "ogg", "opus": "opus"}[fmt]


# -- inspection -------------------------------------------------------------
@dataclass
class AudioInfo:
    path: str
    format: str
    codec: str
    duration: float
    sample_rate: int
    channels: int
    bit_rate: int | None
    size_bytes: int
    tags: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def info(path: Path) -> AudioInfo:
    argv = [ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format",
            "-show_streams", "-select_streams", "a:0", str(_require_input(path))]
    data = json.loads(_run(argv, timeout=60).stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise InvalidRequestError(f"{path} contains no audio stream")
    stream, fmt = streams[0], data.get("format", {})
    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    # Ogg/Opus keep tags on the stream; other containers on the format.
    tags = {str(k).lower(): str(v) for k, v in (stream.get("tags") or {}).items()
            if not str(k).startswith("_")}
    tags.update({str(k).lower(): str(v) for k, v in (fmt.get("tags") or {}).items()})
    bit_rate = fmt.get("bit_rate")
    return AudioInfo(
        path=str(path),
        format=fmt.get("format_name", ""),
        codec=stream.get("codec_name", ""),
        duration=duration,
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        bit_rate=int(bit_rate) if bit_rate else None,
        size_bytes=path.stat().st_size,
        tags=tags,
    )


def levels(path: Path) -> dict[str, float | None]:
    """Peak and mean level in dBFS (``volumedetect``)."""
    argv = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-i", str(_require_input(path)),
            "-af", "volumedetect", "-f", "null", "-"]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=600, check=False)
    if proc.returncode != 0:
        raise MusicError(f"ffmpeg could not decode {path}")
    out: dict[str, float | None] = {"max_volume_db": None, "mean_volume_db": None}
    for key in ("max_volume", "mean_volume"):
        m = re.search(rf"{key}:\s*(-?inf|-?[\d.]+) dB", proc.stderr)
        if m:
            out[f"{key}_db"] = -math.inf if "inf" in m.group(1) else float(m.group(1))
    return out


def loudness(path: Path) -> dict[str, float]:
    """EBU R128 integrated loudness / true peak / LRA (loudnorm analysis pass)."""
    argv = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-i", str(_require_input(path)),
            "-af", "loudnorm=print_format=json", "-f", "null", "-"]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=600, check=False)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", proc.stderr, re.S)
    if proc.returncode != 0 or not m:
        raise MusicError(f"loudness analysis failed for {path}")
    data = json.loads(m.group(0))
    return {k: float(v) for k, v in data.items() if k.startswith("input_") or k == "target_offset"}


def is_silent(path: Path, threshold_db: float = SILENCE_DB) -> bool:
    peak = levels(path)["max_volume_db"]
    return peak is None or peak <= threshold_db


def validate_audio(path: Path, *, expected_duration: float | None = None,
                   tolerance: float = 0.15) -> dict[str, Any]:
    """Acceptance checks used after every generation: decodes, sane rate/channels,
    duration within tolerance, not silent."""
    meta = info(path)
    problems = []
    if meta.sample_rate < 8000 or meta.sample_rate > 384000:
        problems.append(f"implausible sample rate {meta.sample_rate}")
    if meta.channels not in (1, 2):
        problems.append(f"unexpected channel count {meta.channels}")
    if expected_duration and expected_duration > 0:
        drift = abs(meta.duration - expected_duration) / expected_duration
        if drift > tolerance:
            problems.append(f"duration {meta.duration:.2f}s differs from requested "
                            f"{expected_duration:.2f}s by {drift:.0%}")
    lv = levels(path)
    silent = lv["max_volume_db"] is None or lv["max_volume_db"] <= SILENCE_DB
    if silent:
        problems.append("output is silent")
    return {"ok": not problems, "problems": problems, "info": meta.as_dict(), "levels": lv}


# -- transforms ---------------------------------------------------------------
def to_wav(src: Path, dest: Path, *, sample_rate: int | None = None,
           channels: int | None = None) -> Path:
    """Decode anything FFmpeg reads into 16-bit PCM WAV (model input format)."""
    extra: list[str] = []
    if sample_rate:
        extra += ["-ar", str(sample_rate)]
    if channels:
        extra += ["-ac", str(channels)]
    return _ffmpeg([src], dest, [], fmt="wav", overwrite=True, extra=["-vn", *extra])


def convert(src: Path, dest: Path, *, sample_rate: int | None = None,
            channels: int | None = None, bitrate: str | None = None,
            metadata: dict[str, str] | None = None, overwrite: bool = False) -> Path:
    if src.resolve() == dest.resolve():
        raise InvalidRequestError("convert needs a different output path")
    extra = ["-vn"]
    if sample_rate:
        extra += ["-ar", str(sample_rate)]
    if channels:
        extra += ["-ac", str(channels)]
    return _ffmpeg([src], dest, [], overwrite=overwrite, extra=extra, metadata=metadata,
                   bitrate=bitrate)


def derived_path(src: Path, label: str, fmt: str | None = None) -> Path:
    ext = FORMATS[fmt]["ext"] if fmt else src.suffix
    return src.with_name(f"{src.stem}-{label}{ext}")


def normalize(src: Path, dest: Path, *, mode: str = "loudness", target: float | None = None,
              true_peak: float = -1.0, overwrite: bool = False) -> dict[str, Any]:
    """``peak``: scale so the sample peak hits ``target`` dBFS (default -1).
    ``loudness``: two-pass EBU R128 loudnorm to ``target`` LUFS (default -14)."""
    if mode == "peak":
        tgt = -1.0 if target is None else target
        peak = levels(src)["max_volume_db"]
        if peak is None or peak <= SILENCE_DB:
            raise InvalidRequestError(f"cannot peak-normalise near-silence (peak {peak} dBFS)")
        gain = tgt - peak
        _ffmpeg([src], dest, [f"volume={gain:.3f}dB"], overwrite=overwrite)
        return {"mode": "peak", "target_dbfs": tgt, "gain_db": round(gain, 3),
                "input_peak_dbfs": peak}
    if mode != "loudness":
        raise InvalidRequestError("normalize mode must be 'peak' or 'loudness'")
    tgt = -14.0 if target is None else target
    measured = loudness(src)
    filt = (f"loudnorm=I={tgt}:TP={true_peak}:LRA=11:"
            f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
            f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
            f"offset={measured.get('target_offset', 0.0)}:linear=true")
    rate = info(src).sample_rate  # loudnorm resamples internally to 192 kHz
    _ffmpeg([src], dest, [filt], overwrite=overwrite, extra=["-ar", str(rate)])
    achieved = loudness(dest)
    result = {"mode": "loudness", "target_lufs": tgt, "true_peak_db": true_peak,
              "input": measured, "output_lufs": achieved["input_i"],
              "output_true_peak_db": achieved["input_tp"]}
    if abs(achieved["input_i"] - tgt) > 1.0:
        result["note"] = ("target not reached: the true-peak ceiling limits linear gain "
                          "for this material")
    return result


def trim(src: Path, dest: Path, *, start: float = 0.0, end: float | None = None,
         overwrite: bool = False) -> Path:
    if start < 0 or (end is not None and end <= start):
        raise InvalidRequestError("trim needs 0 <= start < end")
    filt = f"atrim=start={start}" + (f":end={end}" if end is not None else "")
    return _ffmpeg([src], dest, [filt, "asetpts=PTS-STARTPTS"], overwrite=overwrite)


def fade(src: Path, dest: Path, *, fade_in: float = 0.0, fade_out: float = 0.0,
         overwrite: bool = False) -> Path:
    if fade_in < 0 or fade_out < 0:
        raise InvalidRequestError("fade lengths must be >= 0")
    filters = []
    if fade_in:
        filters.append(f"afade=t=in:st=0:d={fade_in}")
    if fade_out:
        dur = info(src).duration
        filters.append(f"afade=t=out:st={max(0.0, dur - fade_out):.3f}:d={fade_out}")
    if not filters:
        raise InvalidRequestError("give --in and/or --out")
    return _ffmpeg([src], dest, filters, overwrite=overwrite)


def loop(src: Path, dest: Path, *, count: int, crossfade: float = 0.0,
         overwrite: bool = False) -> Path:
    if not 2 <= count <= 1000:
        raise InvalidRequestError("loop count must be between 2 and 1000")
    if crossfade <= 0:
        return _ffmpeg([src], dest, [], overwrite=overwrite,
                       pre_input=["-stream_loop", str(count - 1)])
    inputs = [src] * count
    chain = "[0:a]"
    parts = []
    for i in range(1, count):
        label = f"[x{i}]"
        parts.append(f"{chain}[{i}:a]acrossfade=d={crossfade}:c1=tri:c2=tri{label}")
        chain = label
    graph = ";".join(parts)
    return _ffmpeg(inputs, dest, [], overwrite=overwrite, filter_complex=graph,
                   extra=["-map", chain])


def crossfade(first: Path, second: Path, dest: Path, *, seconds: float = 2.0,
              overwrite: bool = False) -> Path:
    graph = f"[0:a][1:a]acrossfade=d={seconds}:c1=tri:c2=tri[out]"
    return _ffmpeg([first, second], dest, [], overwrite=overwrite, filter_complex=graph,
                   extra=["-map", "[out]"])


def mix(inputs: Sequence[Path], dest: Path, *, weights: Sequence[float] | None = None,
        duration: str = "longest", overwrite: bool = False) -> Path:
    if len(inputs) < 2:
        raise InvalidRequestError("mix needs at least two inputs")
    if weights is not None and len(weights) != len(inputs):
        raise InvalidRequestError("give one weight per input")
    if duration not in ("longest", "shortest", "first"):
        raise InvalidRequestError("duration must be longest, shortest or first")
    w = " ".join(str(x) for x in (weights or [1.0] * len(inputs)))
    labels = "".join(f"[{i}:a]" for i in range(len(inputs)))
    graph = (f"{labels}amix=inputs={len(inputs)}:duration={duration}:normalize=0:"
             f"weights={w},alimiter=limit=0.97[out]")
    return _ffmpeg(list(inputs), dest, [], overwrite=overwrite, filter_complex=graph,
                   extra=["-map", "[out]"])


def duck(voice: Path, music: Path, dest: Path, *, music_gain_db: float = -6.0,
         threshold: float = 0.05, ratio: float = 8.0, overwrite: bool = False,
         fit: str = "voice") -> Path:
    """Mix narration over music, compressing the music whenever the voice is active."""
    dur = "first" if fit == "voice" else "longest"
    graph = (f"[1:a]volume={music_gain_db}dB[m];"
             f"[0:a]asplit=2[v][sc];"
             f"[m][sc]sidechaincompress=threshold={threshold}:ratio={ratio}:"
             f"attack=20:release=400[ducked];"
             f"[v][ducked]amix=inputs=2:duration={dur}:normalize=0,alimiter=limit=0.97[out]")
    return _ffmpeg([voice, music], dest, [], overwrite=overwrite, filter_complex=graph,
                   extra=["-map", "[out]"])


def fit_duration(src: Path, dest: Path, *, seconds: float, fade_out: float = 2.0,
                 overwrite: bool = False) -> Path:
    """Trim (with a fade) or loop-pad so ``src`` lasts exactly ``seconds``."""
    current = info(src).duration
    if current <= 0:
        raise InvalidRequestError(f"{src} has no duration")
    pre = []
    if current < seconds:
        pre = ["-stream_loop", str(math.ceil(seconds / current) - 1)]
    fo = min(fade_out, seconds / 4)
    filters = [f"atrim=end={seconds}", "asetpts=PTS-STARTPTS",
               f"afade=t=out:st={max(0.0, seconds - fo):.3f}:d={fo}"]
    return _ffmpeg([src], dest, filters, overwrite=overwrite, pre_input=pre)


def waveform(src: Path, dest: Path, *, width: int = 1600, height: int = 400,
             overwrite: bool = False) -> Path:
    return _image(src, dest, f"showwavespic=s={width}x{height}:split_channels=1", overwrite)


def spectrogram(src: Path, dest: Path, *, width: int = 1600, height: int = 800,
                overwrite: bool = False) -> Path:
    return _image(src, dest, f"showspectrumpic=s={width}x{height}:legend=1", overwrite)


def _image(src: Path, dest: Path, graph: str, overwrite: bool) -> Path:
    if dest.suffix.lower() != ".png":
        raise InvalidRequestError("image output must be .png")
    argv = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-i", str(_require_input(src)), "-filter_complex", graph, "-frames:v", "1",
            "-f", "image2", "-c:v", "png"]
    with _AtomicOutput(dest, overwrite) as tmp:
        _run(argv + [str(tmp)])
    return dest


def embed_metadata(src: Path, tags: dict[str, str]) -> Path:
    """Rewrite ``src`` in place with container tags (stream copy, no re-encode)."""
    fmt = format_for(src)
    argv = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-i", str(_require_input(src)), "-map", "0", "-c", "copy"]
    for key, value in tags.items():
        argv += ["-metadata", f"{key}={value}"]
    argv += ["-f", _muxer(fmt)]
    with _AtomicOutput(src, True) as tmp:
        _run(argv + [str(tmp)])
    return src
