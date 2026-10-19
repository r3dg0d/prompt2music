"""Cross-utility workflows: music for video, voice-over ducking."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from prompt2music import audio, paths
from prompt2music.errors import InvalidRequestError, MusicError


def video_duration(video: Path) -> float:
    meta = _probe_video(video)
    return float(meta.get("format", {}).get("duration") or 0.0)


def _probe_video(video: Path) -> dict[str, Any]:
    import json
    import subprocess

    if not video.is_file():
        raise InvalidRequestError(f"video not found: {video}")
    proc = subprocess.run([audio.ffprobe_bin(), "-v", "error", "-print_format", "json",
                           "-show_format", "-show_streams", str(video)],
                          capture_output=True, text=True, timeout=60, check=False)
    if proc.returncode != 0:
        raise MusicError(f"ffprobe could not read {video}")
    return json.loads(proc.stdout or "{}")


def has_audio(video: Path) -> bool:
    return any(s.get("codec_type") == "audio" for s in _probe_video(video).get("streams", []))


def mux_soundtrack(video: Path, music: Path, dest: Path, *, keep_original: str = "duck",
                   music_gain_db: float = 0.0, overwrite: bool = False) -> Path:
    """Attach ``music`` to ``video`` (video stream copied, never re-encoded).

    keep_original: ``replace`` drops the original audio, ``mix`` sums it with
    the music, ``duck`` lowers the music while the original audio is active.
    """
    if dest.exists() and not overwrite:
        raise InvalidRequestError(f"{dest} exists (use --force)")
    seconds = video_duration(video)
    work = paths.ensure(paths.cache_dir() / "work" / f"mux-{dest.stem}")
    try:
        fitted = audio.fit_duration(music, work / "music.wav", seconds=seconds, overwrite=True)
        track = fitted
        if keep_original != "replace" and has_audio(video):
            original = audio.to_wav(video, work / "original.wav")
            if keep_original == "duck":
                track = audio.duck(original, fitted, work / "mixed.wav",
                                   music_gain_db=music_gain_db, overwrite=True)
            else:
                track = audio.mix([original, fitted], work / "mixed.wav", duration="first",
                                  overwrite=True)
        argv = [audio.ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                "-i", str(video), "-i", str(track), "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest"]
        tmp = dest.with_name(f".p2m-{dest.name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        audio._run(argv + [str(tmp)])
        tmp.replace(dest)
        return dest
    finally:
        shutil.rmtree(work, ignore_errors=True)
