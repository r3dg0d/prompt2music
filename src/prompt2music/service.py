"""The generation pipeline shared by the CLI, the job daemon and the MCP server.

``params`` is a JSON-serialisable dict (it is stored on the job). One call to
:func:`run_generate` does: model selection → prompt preparation → input
conversion → worker generation → format conversion → optional
post-processing → acceptance checks → library record. Generated audio and
post-processing are recorded separately in the track sidecar.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from prompt2music import audio, paths
from prompt2music._shared.output import slugify
from prompt2music.backends.base import GenerationRequest, Task, check_request_against
from prompt2music.config import MusicConfig
from prompt2music.engine import Engine
from prompt2music.errors import InvalidRequestError
from prompt2music.library import Library
from prompt2music.manager import ModelManager
from prompt2music.prompts import enhance, fold_controls_into_prompt
from prompt2music.registry import ModelSpec, get_model
from prompt2music.selection import Selection, select_model

ProgressFn = Callable[[float | None, str], None]

CONTINUE_MODES = {
    "natural": "",
    "intensify": "building intensity, fuller arrangement, more energy",
    "transition": "transitioning smoothly into {target}",
    "add-instruments": "adding {target}",
    "mood": "shifting to a {target} mood",
    "outro": "winding down into a gentle outro with a natural ending",
}
PARAM_KEYS = (
    "task", "prompt", "model", "duration", "seed", "negative_prompt", "bpm", "key",
    "time_signature", "instrumental", "lyrics", "vocal_language", "input_audio",
    "region_start", "region_end", "strength", "reference_audio", "melody_audio",
    "batch_size", "steps", "guidance", "output", "format", "enhance", "raw_prompt",
    "device", "precision", "vram_limit_gb", "normalize", "fade_in", "fade_out",
    "continue_mode", "continue_target", "only_new", "overwrite", "quality", "extra",
)


def normalise_params(cfg: MusicConfig, raw: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults from config and drop unknown keys."""
    unknown = set(raw) - set(PARAM_KEYS)
    if unknown:
        raise InvalidRequestError(f"unknown parameters: {', '.join(sorted(unknown))}")
    gen = cfg["generation"]
    params: dict[str, Any] = {k: raw.get(k) for k in PARAM_KEYS}
    params["task"] = Task.parse(raw.get("task") or Task.TEXT_TO_MUSIC).value
    params["prompt"] = (raw.get("prompt") or "").strip()
    params["model"] = raw.get("model") or cfg["general"]["default_model"]
    params["duration"] = float(raw.get("duration") or gen["duration"])
    seed = raw.get("seed")
    params["seed"] = gen["seed"] if seed is None else int(seed)
    if params["seed"] is not None and params["seed"] < 0:
        params["seed"] = None
    if raw.get("instrumental") is None:
        params["instrumental"] = (gen["instrumental"] and not raw.get("lyrics")
                                  and params["task"] != Task.SONG.value)
    params["batch_size"] = int(raw.get("batch_size") or gen["batch_size"])
    params["format"] = (raw.get("format") or "").lower() or None
    if params["enhance"] is None:
        params["enhance"] = bool(gen["enhance_prompt"])
    params["extra"] = dict(raw.get("extra") or {})
    for key in ("input_audio", "reference_audio", "melody_audio", "output"):
        if params.get(key):
            params[key] = str(paths.expand(params[key]).resolve())
    return params


@dataclass
class Prepared:
    spec: ModelSpec
    request: GenerationRequest
    selection: Selection
    prompt_used: str
    negative_prompt: str | None
    enhancement: dict[str, Any] | None


def prepare(cfg: MusicConfig, params: dict[str, Any], *,
            manager: ModelManager | None = None) -> Prepared:
    """Select the model and build the backend request (no audio I/O)."""
    task = Task.parse(params["task"])
    prompt = params["prompt"]
    mode = params.get("continue_mode")
    if task is Task.CONTINUE and mode:
        if mode not in CONTINUE_MODES:
            raise InvalidRequestError(f"unknown continue mode {mode!r}; "
                                      f"choose from {', '.join(CONTINUE_MODES)}")
        phrase = CONTINUE_MODES[mode]
        if "{target}" in phrase:
            if not params.get("continue_target"):
                raise InvalidRequestError(f"continue mode {mode!r} needs --to")
            phrase = phrase.format(target=params["continue_target"])
        prompt = ", ".join(p for p in (prompt, phrase) if p)
    probe = GenerationRequest(
        prompt=prompt or "music", model=params["model"], duration=params["duration"],
        task=task, seed=params["seed"], negative_prompt=params.get("negative_prompt"),
        instrumental=bool(params["instrumental"]), lyrics=params.get("lyrics"),
        input_audio=params.get("input_audio"), region_start=params.get("region_start"),
        region_end=params.get("region_end"), reference_audio=params.get("reference_audio"),
        melody_audio=params.get("melody_audio"), batch_size=params["batch_size"],
    )
    selection = select_model(cfg, probe, explicit=params["model"], manager=manager,
                             device=params.get("device"),
                             vram_limit_gb=params.get("vram_limit_gb"),
                             quality=params.get("quality"))
    spec = selection.spec
    caps = spec.capabilities
    enhancement = None
    negative = params.get("negative_prompt")
    if params.get("enhance") and not params.get("raw_prompt") and prompt:
        from prompt2music.llm import get_llm

        result = enhance(prompt, instrumental=bool(params["instrumental"]), bpm=params.get("bpm"),
                         key=params.get("key"), llm=get_llm(cfg))
        prompt = result.prompt
        enhancement = result.as_dict()
        if negative is None and caps.negative_prompt:
            negative = result.negative_prompt
    if not params.get("raw_prompt"):
        # Fold tempo/key into the text only for models without dedicated inputs.
        prompt = fold_controls_into_prompt(
            prompt,
            bpm=None if caps.bpm else params.get("bpm"),
            key=None if caps.key else params.get("key"),
            time_signature=None if caps.time_signature else params.get("time_signature"),
        )
    request = GenerationRequest(
        prompt=prompt, model=spec.id, duration=params["duration"], task=task,
        seed=params["seed"], negative_prompt=negative if caps.negative_prompt else None,
        bpm=params.get("bpm") if caps.bpm else None, key=params.get("key") if caps.key else None,
        time_signature=params.get("time_signature") if caps.time_signature else None,
        instrumental=bool(params["instrumental"]), lyrics=params.get("lyrics"),
        vocal_language=params.get("vocal_language"), input_audio=params.get("input_audio"),
        region_start=params.get("region_start"), region_end=params.get("region_end"),
        strength=params.get("strength"), reference_audio=params.get("reference_audio"),
        melody_audio=params.get("melody_audio"), batch_size=params["batch_size"],
        steps=params.get("steps"), guidance=params.get("guidance"),
        extra={**params["extra"], **({"only_new": True} if params.get("only_new") else {})},
    )
    if negative and not caps.negative_prompt and params.get("negative_prompt"):
        raise InvalidRequestError(f"{spec.id} does not support negative prompts")
    check_request_against(spec.id, caps, request)
    return Prepared(spec, request, selection, prompt, request.negative_prompt, enhancement)


def _final_paths(cfg: MusicConfig, params: dict[str, Any], prepared: Prepared,
                 count: int, produced_suffix: str) -> list[Path]:
    fmt = params.get("format")
    out = params.get("output")
    if out:
        target = Path(out)
        if target.suffix:
            fmt = fmt or target.suffix.lstrip(".")
            base_dir, stem = target.parent, target.stem
        else:
            base_dir, stem = target, None
    else:
        base_dir, stem = cfg.output_dir, None
    fmt = fmt or cfg["general"]["default_format"] or produced_suffix.lstrip(".")
    audio.format_for(Path(f"x.{fmt}"))  # validates the format name
    if stem is None:
        label = prepared.prompt_used or prepared.request.task.value
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        seed = prepared.request.seed
        stem = "_".join([slugify(label, max_len=40), ts] + ([f"s{seed}"] if seed is not None
                                                             else []))
    base_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(count):
        name = stem if count == 1 else f"{stem}-{i + 1}"
        candidate = base_dir / f"{name}.{fmt}"
        n = 2
        while candidate.exists() and not params.get("overwrite"):
            candidate = base_dir / f"{name}-v{n}.{fmt}"
            n += 1
        results.append(candidate)
    return results


def _prepare_inputs(spec: ModelSpec, request: GenerationRequest, workdir: Path,
                    max_input_bytes: int | None) -> dict[str, Any]:
    """Convert user audio to the model's PCM WAV format inside the job workdir."""
    caps = spec.capabilities
    converted: dict[str, Any] = {}
    for attr in ("input_audio", "reference_audio", "melody_audio"):
        src = getattr(request, attr)
        if src is None:
            continue
        src = Path(src)
        if not src.is_file():
            raise InvalidRequestError(f"{attr.replace('_', ' ')} not found: {src}")
        if max_input_bytes and src.stat().st_size > max_input_bytes:
            raise InvalidRequestError(f"{src} exceeds the input size limit")
        meta = audio.info(src)
        dest = workdir / f"{attr}.wav"
        audio.to_wav(src, dest, sample_rate=caps.sample_rate or meta.sample_rate,
                     channels=caps.channels or meta.channels)
        setattr(request, attr, dest)
        converted[attr] = {"source": str(src), "duration": round(meta.duration, 3),
                           "sample_rate": meta.sample_rate, "channels": meta.channels}
    if request.task is Task.INPAINT and "input_audio" in converted:
        if (request.region_end or 0) > converted["input_audio"]["duration"] + 0.01:
            raise InvalidRequestError("inpaint region ends after the input audio")
    return converted


def _post_process(path: Path, params: dict[str, Any]) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    if params.get("fade_in") or params.get("fade_out"):
        tmp = path.with_name(f".fade-{path.name}")
        audio.fade(path, tmp, fade_in=float(params.get("fade_in") or 0),
                   fade_out=float(params.get("fade_out") or 0), overwrite=True)
        os.replace(tmp, path)
        ops.append({"op": "fade", "in": params.get("fade_in") or 0,
                    "out": params.get("fade_out") or 0})
    if params.get("normalize"):
        tmp = path.with_name(f".norm-{path.name}")
        info = audio.normalize(path, tmp, mode=str(params["normalize"]), overwrite=True)
        os.replace(tmp, path)
        ops.append({"op": "normalize", **info})
    return ops


def run_generate(cfg: MusicConfig, engine: Engine, params: dict[str, Any], *,
                 job_id: str | None = None, progress: ProgressFn | None = None,
                 on_stage: Callable[[str], None] | None = None,
                 library: Library | None = None,
                 max_input_bytes: int | None = None) -> dict[str, Any]:
    progress = progress or (lambda _f, _m: None)
    stage = on_stage or (lambda _s: None)
    prepared = prepare(cfg, params, manager=engine.manager)
    spec, request = prepared.spec, prepared.request
    workdir = paths.ensure(paths.cache_dir() / "work" / (job_id or f"fg-{os.getpid()}"))
    try:
        converted = _prepare_inputs(spec, request, workdir, max_input_bytes)
        stage("loading")
        load_info = engine.load(spec, device=params.get("device"),
                                precision=params.get("precision"),
                                vram_limit_gb=params.get("vram_limit_gb"), on_status=progress)
        stage("running")
        request.output_path = workdir / "out.wav"
        started = time.perf_counter()
        data = engine.generate(spec, request, workdir, on_progress=progress)
        wall = time.perf_counter() - started
        stage("processing")
        produced = [Path(o["path"]) for o in data["outputs"]]
        finals = _final_paths(cfg, params, prepared, len(produced),
                              produced[0].suffix if produced else ".wav")
        installed = engine.manager.installed(spec.id) if spec.needs_download else None
        revision = None
        if installed:
            revision = ",".join(f"{r}@{s[:12]}" for r, s in installed.manifest.revisions.items())
        lib = library or Library()
        outputs = []
        for item, src, dest in zip(data["outputs"], produced, finals, strict=True):
            if src.suffix.lower() == dest.suffix.lower():
                shutil.move(str(src), dest)
            else:
                audio.convert(src, dest, overwrite=True)
                src.unlink(missing_ok=True)
            ops = _post_process(dest, params)
            expected = None
            if request.task in (Task.TEXT_TO_MUSIC, Task.SONG) and spec.backend != "yue":
                expected = request.duration
            elif request.task is Task.CONTINUE:
                base = converted.get("input_audio", {}).get("duration", 0.0)
                expected = request.duration + (0.0 if params.get("only_new") else base)
            checks = audio.validate_audio(dest, expected_duration=expected)
            meta = checks["info"]
            track = lib.add(
                path=dest, prompt=prepared.prompt_used, model=spec.id, model_revision=revision,
                task=request.task.value,
                params={k: v for k, v in params.items() if v not in (None, {}, "")},
                duration=meta["duration"], sample_rate=meta["sample_rate"],
                channels=meta["channels"], seed=item.get("seed"), job_id=job_id,
                generation={"backend": spec.backend, "device": load_info.get("device"),
                            "precision": load_info.get("device_plan", {}).get("precision"),
                            "generation_seconds": data.get("generation_seconds"),
                            "negative_prompt": prepared.negative_prompt,
                            "enhancement": prepared.enhancement},
                post_processing=ops,
            )
            outputs.append({
                "path": str(dest), "track_id": track["track_id"], "sidecar": track["sidecar"],
                "duration": meta["duration"], "sample_rate": meta["sample_rate"],
                "channels": meta["channels"], "seed": item.get("seed"),
                "validation": {"ok": checks["ok"], "problems": checks["problems"],
                               "peak_dbfs": checks["levels"]["max_volume_db"]},
                "post_processing": ops,
            })
        gen_seconds = float(data.get("generation_seconds") or wall)
        audio_seconds = sum(o["duration"] for o in outputs) or request.duration
        mem = data.get("memory") or {}
        return {
            "job_id": job_id,
            "model": spec.id,
            "task": request.task.value,
            "selection": prepared.selection.as_dict(),
            "prompt": params.get("prompt"),
            "prompt_used": prepared.prompt_used,
            "negative_prompt": prepared.negative_prompt,
            "enhancement": prepared.enhancement,
            "outputs": outputs,
            "generation_seconds": round(gen_seconds, 3),
            "rtf": round(gen_seconds / audio_seconds, 4) if audio_seconds else None,
            "load": {k: load_info.get(k) for k in ("load_seconds", "reused", "device",
                                                   "worker_pid")},
            "memory": mem,
            "peak_rss_mb": data.get("peak_rss_mb"),
            "inputs": converted,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_separate(cfg: MusicConfig, engine: Engine, params: dict[str, Any], *,
                 job_id: str | None = None, progress: ProgressFn | None = None,
                 max_input_bytes: int | None = None) -> dict[str, Any]:
    spec = get_model(params.get("model") or "htdemucs")
    if spec.kind != "stems":
        raise InvalidRequestError(f"{spec.id} is not a stem-separation model")
    src = Path(params["input_audio"])
    if not src.is_file():
        raise InvalidRequestError(f"input not found: {src}")
    if max_input_bytes and src.stat().st_size > max_input_bytes:
        raise InvalidRequestError(f"{src} exceeds the input size limit")
    out_dir = Path(params.get("output_dir") or (cfg.output_dir / "stems" / slugify(src.stem)))
    workdir = paths.ensure(paths.cache_dir() / "work" / (job_id or f"sep-{os.getpid()}"))
    try:
        wav = audio.to_wav(src, workdir / "input.wav", sample_rate=44100, channels=2)
        engine.load(spec, device=params.get("device"), on_status=progress)
        data = engine.separate(spec, wav, workdir / "stems", params.get("stems"),
                               on_progress=progress)
        out_dir.mkdir(parents=True, exist_ok=True)
        fmt = params.get("format") or "wav"
        written = {}
        for name, path in data["stems"].items():
            dest = out_dir / f"{name}.{fmt}"
            if dest.exists() and not params.get("overwrite"):
                raise InvalidRequestError(f"{dest} exists (use --force)")
            if fmt == "wav":
                shutil.move(path, dest)
            else:
                audio.convert(Path(path), dest, overwrite=True)
            written[name] = str(dest)
        return {"job_id": job_id, "model": spec.id, "input": str(src), "output_dir":
                str(out_dir), "stems": written, "seconds": data.get("seconds"),
                "memory": data.get("memory")}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
