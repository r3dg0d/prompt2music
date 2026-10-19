"""MCP tool implementations for prompt2music.

Long operations (generation, downloads, stem separation, model loading,
benchmarks) are queued on the shared job daemon and return a ``job_id``
immediately; clients poll ``music_job_status`` and fetch outputs with
``music_job_result``. Nothing here assumes progress notifications.

Tool functions are synchronous, with typed parameters from which the MCP SDK
derives input schemas. :func:`build_tools` wraps each one so it runs in a
worker thread (never blocking the event loop), is audited, and reports
errors as structured JSON.
"""


import functools
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from prompt2music import __version__, daemon, service
from prompt2music import audio as A
from prompt2music.config import MusicConfig
from prompt2music.errors import (
    InvalidRequestError,
    MusicError,
    ResourceLimitError,
    SecurityError,
)
from prompt2music.jobs import JobStore
from prompt2music.mcp.security import Policy, audit

Format = Literal["wav", "flac", "mp3", "ogg", "opus"]
NEXT_STEP = "poll music_job_status with this job_id, then call music_job_result"


@dataclass
class ToolDef:
    name: str
    description: str
    fn: Callable[..., Any]
    read_only: bool = False
    destructive: bool = False
    long_running: bool = False


TOOL_SPECS: dict[str, dict[str, Any]] = {}


class State:
    def __init__(self, cfg: MusicConfig) -> None:
        self.cfg = cfg
        self.policy = Policy(cfg)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _submit(state: State, kind: str, params: dict[str, Any], *, model: str | None,
            prompt: str | None) -> dict[str, Any]:
    store = JobStore()
    limit = int(state.cfg["mcp"]["max_queued_jobs"])
    if store.count_active(source="mcp") >= limit:
        raise ResourceLimitError(f"MCP job quota reached ({limit} active jobs)",
                                 hint="wait for jobs to finish or cancel one")
    job_id = store.create(kind, params, model=model, prompt=prompt, source="mcp",
                          max_queued=int(state.cfg["jobs"]["max_queued"]))
    info = daemon.ensure_running(state.cfg)
    return {"job_id": job_id, "status": "queued", "kind": kind, "model": model,
            "daemon_pid": info.get("pid"), "next": NEXT_STEP}


def _output(state: State, raw: str | None) -> str | None:
    path = state.policy.resolve_output(raw)
    if path is not None and path.exists():
        raise InvalidRequestError(f"{path} already exists; choose another output_path")
    return str(path) if path else None


def _derived(state: State, src: Path, label: str, ext: str | None = None) -> Path:
    name = f"{src.stem}-{label}{ext or src.suffix}"
    try:
        candidate = state.policy.resolve_output(str(src.with_name(name)))
    except SecurityError:
        candidate = state.policy.output_roots[0] / name
    assert candidate is not None
    n = 2
    base = candidate
    while candidate.exists():
        candidate = base.with_name(f"{base.stem}-{n}{base.suffix}")
        n += 1
    return candidate


def _generation(state: State, task: str, raw: dict[str, Any]) -> dict[str, Any]:
    pol = state.policy
    pol.check_model(raw.get("model"), kind="music")
    pol.check_duration(raw.get("duration"))
    for key in ("input_audio", "reference_audio", "melody_audio"):
        if raw.get(key):
            raw[key] = str(pol.resolve_input(raw[key]))
    raw["output"] = _output(state, raw.pop("output_path", None))
    raw = {k: v for k, v in raw.items() if v is not None}
    raw["task"] = task
    params = service.normalise_params(state.cfg, raw)
    params["overwrite"] = False
    prepared = service.prepare(state.cfg, params)  # fail fast: support, install, VRAM
    pol.check_model(prepared.spec.id)
    params["model"] = prepared.spec.id
    job = _submit(state, "generate", params, model=prepared.spec.id, prompt=params["prompt"])
    job["selection"] = prepared.selection.as_dict()
    return job


def _join(*parts: str | None) -> str:
    return ", ".join(p.strip() for p in parts if p and p.strip())


# ---------------------------------------------------------------------------
# tool bodies (sync; typed signatures become MCP input schemas)
# ---------------------------------------------------------------------------
def make_tools(state: State) -> list[ToolDef]:
    tools: list[ToolDef] = []

    def tool(name: str, description: str, *, read_only: bool = False,
             destructive: bool = False, long_running: bool = False
             ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            tools.append(ToolDef(name, description, fn, read_only, destructive, long_running))
            return fn
        return deco

    @tool("music_generate", "Queue music generation from a natural-language prompt. Returns "
          "a job_id immediately. model='auto' picks an installed model that supports the "
          "request.", long_running=True)
    def music_generate(prompt: str, model: str = "auto", duration: float | None = None,
                       genre: str | None = None, mood: str | None = None,
                       instrumental: bool = True, bpm: int | None = None,
                       key: str | None = None, seed: int | None = None,
                       negative_prompt: str | None = None, output_path: str | None = None,
                       format: Format | None = None, enhance: bool = False,
                       batch_size: int = 1) -> dict[str, Any]:
        return _generation(state, "text-to-music", {
            "prompt": _join(prompt, genre, mood), "model": model, "duration": duration,
            "instrumental": instrumental, "bpm": bpm, "key": key, "seed": seed,
            "negative_prompt": negative_prompt, "output_path": output_path, "format": format,
            "enhance": enhance, "batch_size": batch_size})

    @tool("music_generate_instrumental", "Queue instrumental-only music generation. Returns a "
          "job_id.", long_running=True)
    def music_generate_instrumental(prompt: str, model: str = "auto",
                                    duration: float | None = None, bpm: int | None = None,
                                    key: str | None = None, seed: int | None = None,
                                    output_path: str | None = None,
                                    format: Format | None = None) -> dict[str, Any]:
        return _generation(state, "text-to-music", {
            "prompt": prompt, "model": model, "duration": duration, "instrumental": True,
            "bpm": bpm, "key": key, "seed": seed, "output_path": output_path, "format": format})

    @tool("music_generate_song", "Queue a vocal song from lyrics (use [Verse 1], [Chorus] ... "
          "section tags) plus a style description. Needs a vocal model such as ace-step-1.5, "
          "heartmula-3b or yue2-3b.", long_running=True)
    def music_generate_song(lyrics: str, style: str, model: str = "auto",
                            duration: float | None = None, genre: str | None = None,
                            mood: str | None = None, vocal_style: str | None = None,
                            language: str | None = None, seed: int | None = None,
                            bpm: int | None = None, key: str | None = None,
                            output_path: str | None = None,
                            format: Format | None = None) -> dict[str, Any]:
        from prompt2music.lyrics import validate

        report = validate(lyrics)
        if not report.ok:
            raise InvalidRequestError("lyrics problems: " + "; ".join(report.errors))
        job = _generation(state, "song", {
            "prompt": _join(style, genre, mood, vocal_style), "lyrics": lyrics, "model": model,
            "duration": duration, "instrumental": False, "vocal_language": language,
            "seed": seed, "bpm": bpm, "key": key, "output_path": output_path,
            "format": format})
        job["lyrics_warnings"] = report.warnings
        return job

    @tool("music_continue", "Queue a generative continuation of an existing audio file "
          "(native audio conditioning; never plain concatenation). duration = seconds of new "
          "audio.", long_running=True)
    def music_continue(input_path: str, duration: float = 30.0, prompt: str = "",
                       model: str = "auto",
                       mode: Literal["natural", "intensify", "transition", "add-instruments",
                                     "mood", "outro"] = "natural",
                       target: str | None = None, only_new: bool = False,
                       output_path: str | None = None) -> dict[str, Any]:
        return _generation(state, "continue", {
            "prompt": prompt, "model": model, "duration": duration, "input_audio": input_path,
            "continue_mode": mode, "continue_target": target, "only_new": only_new,
            "output_path": output_path})

    @tool("music_edit", "Queue an audio-to-audio edit / style transfer of a recording. "
          "strength: 0 keeps the source, 1 ignores it.", long_running=True)
    def music_edit(input_path: str, prompt: str, model: str = "auto",
                   strength: float | None = None, output_path: str | None = None
                   ) -> dict[str, Any]:
        return _generation(state, "edit", {"prompt": prompt, "model": model,
                                           "input_audio": input_path, "strength": strength,
                                           "output_path": output_path, "duration": 30.0})

    @tool("music_inpaint", "Queue regeneration of the region [start, end] seconds of a "
          "recording, keeping the rest.", long_running=True)
    def music_inpaint(input_path: str, start: float, end: float, prompt: str,
                      model: str = "auto", output_path: str | None = None) -> dict[str, Any]:
        src = state.policy.resolve_input(input_path)
        length = A.info(src).duration
        return _generation(state, "inpaint", {
            "prompt": prompt, "model": model, "input_audio": str(src), "region_start": start,
            "region_end": end, "output_path": output_path, "duration": max(length, 1.0)})

    @tool("music_enhance_prompt", "Expand a short description into a detailed music prompt "
          "(keeps the user's genre, mood and instruments) plus a negative prompt.",
          read_only=True)
    def music_enhance_prompt(prompt: str, instrumental: bool = True, bpm: int | None = None,
                             key: str | None = None) -> dict[str, Any]:
        from prompt2music.llm import get_llm
        from prompt2music.prompts import enhance

        return enhance(prompt, instrumental=instrumental, bpm=bpm, key=key,
                       llm=get_llm(state.cfg)).as_dict()

    @tool("music_models_list", "List catalogued models with capabilities, licenses and "
          "install state.", read_only=True)
    def music_models_list(installed_only: bool = False,
                          task: Literal["text-to-music", "song", "continue", "edit", "inpaint"]
                          | None = None,
                          kind: Literal["music", "stems"] | None = "music") -> dict[str, Any]:
        from prompt2music import envs
        from prompt2music.backends.base import Task
        from prompt2music.manager import ModelManager
        from prompt2music.registry import all_models

        mgr = ModelManager(state.cfg)
        out = []
        for spec in all_models(kind=kind):
            if task and Task(task) not in spec.tasks:
                continue
            ready = mgr.is_ready(spec.id)
            if installed_only and not ready:
                continue
            allowed = not state.policy.allowed_models or spec.id in state.policy.allowed_models
            out.append({
                "id": spec.id, "name": spec.name, "kind": spec.kind,
                "tasks": sorted(t.value for t in spec.tasks),
                "vocals": spec.capabilities.vocals,
                "max_duration": spec.capabilities.max_duration,
                "installed": ready, "backend": spec.backend,
                "backend_ready": envs.probe(state.cfg, spec.backend).ok if ready else None,
                "approx_size_gb": spec.approx_size_gb, "min_vram_gb": spec.min_vram_gb,
                "license": spec.license, "commercial_use": spec.commercial_use,
                "gated": spec.gated, "allowed_by_policy": allowed, "diagnostic": spec.diagnostic,
            })
        return {"models": out}

    @tool("music_models_search", "Search the model catalogue by keyword.", read_only=True)
    def music_models_search(query: str) -> dict[str, Any]:
        from prompt2music.registry import search_models

        return {"models": [{"id": s.id, "name": s.name, "kind": s.kind,
                            "tasks": sorted(t.value for t in s.tasks), "license": s.license}
                           for s in search_models(query)]}

    @tool("music_model_info", "Full details for one model: capabilities, sources, license, "
          "hardware needs, verified status and install state.", read_only=True)
    def music_model_info(model: str) -> dict[str, Any]:
        from prompt2music.manager import ModelManager
        from prompt2music.registry import get_model

        spec = get_model(model)
        data = spec.as_dict()
        inst = ModelManager(state.cfg).installed(spec.id) if spec.needs_download else None
        data["installed"] = inst.as_dict() if inst else None
        return data

    @tool("music_model_download", "Queue a model download from Hugging Face. Disabled unless "
          "mcp.allow_model_downloads is true; size-capped by mcp.max_download_gb. Set "
          "accept_license=true only after the user has agreed to the model license.",
          long_running=True)
    def music_model_download(model: str, accept_license: bool = False) -> dict[str, Any]:
        from prompt2music.downloader import HfHub, plan_download
        from prompt2music.registry import get_model

        cfg = state.cfg
        if not cfg["mcp"]["allow_model_downloads"]:
            raise SecurityError("model downloads are disabled for MCP clients",
                                hint="ask the user to run `prompt2music models download "
                                f"{model}` or set mcp.allow_model_downloads = true")
        spec = state.policy.check_model(model) or get_model(model)
        if spec.requires_license_acceptance and not accept_license:
            raise SecurityError(f"{spec.id} is licensed under {spec.license}; the user must "
                                "accept it (accept_license=true)", hint=spec.license_url)
        plan = plan_download(spec, cfg.models_dir, hub=HfHub(cfg["models"]["hf_endpoint"] or None),
                             token=cfg.hf_token(), offline=cfg.offline)
        max_bytes = int(float(cfg["mcp"]["max_download_gb"]) * 1024**3)
        if plan.total_bytes > max_bytes:
            raise SecurityError(f"{spec.id} is {plan.total_bytes / 1024**3:.1f} GB, above "
                                f"mcp.max_download_gb ({cfg['mcp']['max_download_gb']})")
        job = _submit(state, "download", {"model": spec.id, "accept_license": accept_license,
                                          "max_bytes": max_bytes}, model=spec.id, prompt=None)
        job["plan"] = plan.as_dict()
        return job

    @tool("music_model_load", "Queue loading a model into the job daemon so consecutive "
          "generations reuse it.", long_running=True)
    def music_model_load(model: str, device: str | None = None) -> dict[str, Any]:
        from prompt2music.manager import ModelManager

        spec = state.policy.check_model(model)
        assert spec is not None
        ModelManager(state.cfg).require(spec.id)
        return _submit(state, "load", {"model": spec.id, "device": device}, model=spec.id,
                       prompt=None)

    @tool("music_model_unload", "Unload the daemon's model and free GPU memory.")
    def music_model_unload() -> dict[str, Any]:
        if not daemon.status()["running"]:
            return {"unloaded": False, "reason": "daemon is not running"}
        return _submit(state, "unload", {}, model=None, prompt=None)

    @tool("music_model_remove", "Delete an installed model. Disabled unless "
          "mcp.allow_model_deletion is true; requires confirm=true.", destructive=True)
    def music_model_remove(model: str, confirm: bool = False) -> dict[str, Any]:
        from prompt2music.manager import ModelManager

        if not state.cfg["mcp"]["allow_model_deletion"]:
            raise SecurityError("model deletion is disabled for MCP clients",
                                hint="the user can run `prompt2music models remove "
                                f"{model}` or set mcp.allow_model_deletion = true")
        if not confirm:
            raise InvalidRequestError("set confirm=true after the user approves deletion")
        spec = state.policy.check_model(model)
        assert spec is not None
        freed = ModelManager(state.cfg).remove(spec.id)
        return {"removed": spec.id, "freed_bytes": freed}

    @tool("music_audio_info", "Format, duration, sample rate, channels, tags and levels of "
          "an audio file.", read_only=True)
    def music_audio_info(path: str) -> dict[str, Any]:
        src = state.policy.resolve_input(path)
        data = A.info(src).as_dict()
        data["levels"] = A.levels(src)
        data["uri"] = src.as_uri()
        return data

    @tool("music_audio_convert", "Convert between wav, flac, mp3, ogg and opus (optional "
          "resample / channel change).")
    def music_audio_convert(input_path: str, output_path: str, sample_rate: int | None = None,
                            channels: int | None = None, bitrate: str | None = None
                            ) -> dict[str, Any]:
        src = state.policy.resolve_input(input_path)
        dest = Path(_output(state, output_path) or "")
        A.convert(src, dest, sample_rate=sample_rate, channels=channels, bitrate=bitrate)
        return {"output": str(dest), "uri": dest.as_uri()}

    @tool("music_audio_normalize", "Peak (dBFS) or EBU R128 loudness (LUFS) normalisation. "
          "Level adjustment only, not mastering.")
    def music_audio_normalize(input_path: str, output_path: str | None = None,
                              mode: Literal["peak", "loudness"] = "loudness",
                              target: float | None = None) -> dict[str, Any]:
        src = state.policy.resolve_input(input_path)
        dest = Path(_output(state, output_path) or _derived(state, src, "normalized"))
        info = A.normalize(src, dest, mode=mode, target=target)
        return {"output": str(dest), **info}

    @tool("music_audio_trim", "Cut [start, end] seconds out of a file.")
    def music_audio_trim(input_path: str, start: float = 0.0, end: float | None = None,
                         output_path: str | None = None) -> dict[str, Any]:
        src = state.policy.resolve_input(input_path)
        dest = Path(_output(state, output_path) or _derived(state, src, "trim"))
        A.trim(src, dest, start=start, end=end)
        return {"output": str(dest), "duration": A.info(dest).duration}

    @tool("music_audio_mix", "Mix two or more files (optional per-input weights).")
    def music_audio_mix(input_paths: list[str], output_path: str | None = None,
                        weights: list[float] | None = None,
                        duration: Literal["longest", "shortest", "first"] = "longest"
                        ) -> dict[str, Any]:
        srcs = [state.policy.resolve_input(p) for p in input_paths]
        if len(srcs) < 2:
            raise InvalidRequestError("give at least two input_paths")
        dest = Path(_output(state, output_path) or _derived(state, srcs[0], "mix"))
        A.mix(srcs, dest, weights=weights, duration=duration)
        return {"output": str(dest), "duration": A.info(dest).duration}

    @tool("music_audio_loop", "Repeat a file count times (optional crossfade seconds).")
    def music_audio_loop(input_path: str, count: int, crossfade: float = 0.0,
                         output_path: str | None = None) -> dict[str, Any]:
        src = state.policy.resolve_input(input_path)
        dest = Path(_output(state, output_path) or _derived(state, src, f"x{count}"))
        A.loop(src, dest, count=count, crossfade=crossfade)
        return {"output": str(dest), "duration": A.info(dest).duration}

    @tool("music_audio_waveform", "Render a waveform PNG.")
    def music_audio_waveform(input_path: str, output_path: str | None = None
                             ) -> dict[str, Any]:
        src = state.policy.resolve_input(input_path)
        dest = Path(_output(state, output_path) or _derived(state, src, "waveform", ".png"))
        A.waveform(src, dest)
        return {"output": str(dest), "uri": dest.as_uri()}

    @tool("music_audio_spectrogram", "Render a spectrogram PNG.")
    def music_audio_spectrogram(input_path: str, output_path: str | None = None
                                ) -> dict[str, Any]:
        src = state.policy.resolve_input(input_path)
        dest = Path(_output(state, output_path) or _derived(state, src, "spectrogram", ".png"))
        A.spectrogram(src, dest)
        return {"output": str(dest), "uri": dest.as_uri()}

    @tool("music_separate_stems", "Queue Demucs stem separation (vocals, drums, bass, other; "
          "htdemucs_6s adds guitar and piano).", long_running=True)
    def music_separate_stems(input_path: str, model: str = "htdemucs",
                             stems: list[str] | None = None, output_dir: str | None = None,
                             format: Format = "wav") -> dict[str, Any]:
        from prompt2music.manager import ModelManager

        spec = state.policy.check_model(model, kind="stems")
        assert spec is not None
        ModelManager(state.cfg).require(spec.id)
        src = state.policy.resolve_input(input_path)
        out_dir = state.policy.resolve_output(output_dir, is_dir=True)
        return _submit(state, "separate", {
            "model": spec.id, "input_audio": str(src),
            "output_dir": str(out_dir) if out_dir else None, "stems": stems, "format": format,
            "overwrite": False}, model=spec.id, prompt=None)

    @tool("music_system_info", "Hardware, tools, installed models and readiness.",
          read_only=True)
    def music_system_info() -> dict[str, Any]:
        from prompt2music.doctor import collect

        report = collect(state.cfg, quick=True)
        report["mcp"] = {"version": __version__,
                         "output_dirs": [str(p) for p in state.policy.output_roots],
                         "input_dirs": [str(p) for p in state.policy.input_roots],
                         "downloads_allowed": state.cfg["mcp"]["allow_model_downloads"],
                         "deletion_allowed": state.cfg["mcp"]["allow_model_deletion"]}
        return report

    @tool("music_gpu_status", "Current GPU memory/utilisation and the job daemon state.",
          read_only=True)
    def music_gpu_status() -> dict[str, Any]:
        from prompt2music._shared.gpu import query_gpus

        store = JobStore()
        loads = [j for j in store.list(limit=20) if j.kind in ("load", "generate", "unload")
                 and j.status == "completed"]
        last = loads[0] if loads else None
        return {"gpus": [g.as_dict() for g in query_gpus()], "daemon": daemon.status(),
                "last_model_job": last.as_dict(brief=True) if last else None}

    @tool("music_benchmark", "Queue a benchmark (load time, peak VRAM/RAM, RTF = "
          "generation_seconds / audio_seconds).", long_running=True)
    def music_benchmark(model: str, duration: float = 10.0, runs: int = 1) -> dict[str, Any]:
        spec = state.policy.check_model(model)
        assert spec is not None
        state.policy.check_duration(duration)
        if not 1 <= runs <= 5:
            raise InvalidRequestError("runs must be 1..5")
        return _submit(state, "benchmark", {"models": [spec.id], "duration": duration,
                                            "runs": runs}, model=spec.id, prompt=None)

    @tool("music_job_status", "Status and progress of a job.", read_only=True)
    def music_job_status(job_id: str) -> dict[str, Any]:
        return JobStore().get(job_id).as_dict(brief=True)

    @tool("music_job_cancel", "Cancel a queued or running job (cooperative first, then the "
          "worker is stopped).")
    def music_job_cancel(job_id: str) -> dict[str, Any]:
        store = JobStore()
        job = store.get(job_id)
        if job.source != "mcp":
            raise SecurityError("MCP clients may only cancel jobs they created")
        return store.request_cancel(job.id).as_dict(brief=True)

    @tool("music_jobs_list", "Recent jobs (status: queued, running, completed, failed, "
          "cancelled or active).", read_only=True)
    def music_jobs_list(status: str | None = None, limit: int = 20) -> dict[str, Any]:
        jobs = JobStore().list(status=status, limit=max(1, min(limit, 100)))
        return {"jobs": [j.as_dict(brief=True) for j in jobs]}

    @tool("music_job_result", "Result of a completed job: output file paths (with file:// "
          "URIs and music://track/ resource ids), durations, seeds, model and timing.",
          read_only=True)
    def music_job_result(job_id: str) -> dict[str, Any]:
        job = JobStore().get(job_id)
        if not job.final:
            return {"job_id": job.id, "status": job.status, "progress": job.progress,
                    "ready": False, "next": NEXT_STEP}
        if job.status != "completed":
            return {"job_id": job.id, "status": job.status, "ready": True, "error": job.error}
        result = dict(job.result or {})
        for out in result.get("outputs", []):
            path = Path(out["path"])
            out["exists"] = path.is_file()
            out["uri"] = path.as_uri()
            if out.get("track_id"):
                out["resource"] = f"music://track/{out['track_id']}"
        for name, path in (result.get("stems") or {}).items():
            result.setdefault("stem_uris", {})[name] = Path(path).as_uri()
        return {"job_id": job.id, "status": job.status, "ready": True, "result": result}

    return tools


# ---------------------------------------------------------------------------
# wrapping
# ---------------------------------------------------------------------------
def _error_payload(exc: BaseException) -> str:
    if isinstance(exc, MusicError):
        return json.dumps({"error": exc.as_dict()})
    return json.dumps({"error": {"code": "internal_error",
                                 "message": f"{type(exc).__name__}: {exc}"}})


def wrap(state: "State", tool: ToolDef, *, raise_as: type[Exception] = RuntimeError
         ) -> Callable[..., Any]:
    """Async, audited, thread-offloaded version of a sync tool with the same signature."""
    import anyio

    fn = tool.fn

    @functools.wraps(fn)
    async def runner(**kwargs: Any) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            try:
                result = fn(**kwargs)
            except Exception as exc:
                audit(state.cfg, tool.name, kwargs, "error",
                      exc.as_dict() if isinstance(exc, MusicError) else repr(exc))
                raise raise_as(_error_payload(exc)) from exc
            audit(state.cfg, tool.name, kwargs, "ok",
                  {k: result.get(k) for k in ("job_id", "output") if k in result})
            return result

        return await anyio.to_thread.run_sync(call)

    runner.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
    return runner


def _register_static_specs() -> None:
    """Populate TOOL_SPECS (name → description) without a config, for `mcp tools`."""
    class _Stub:
        cfg = None
        policy = None

    for t in make_tools(_Stub()):  # type: ignore[arg-type]
        TOOL_SPECS[t.name] = {"description": t.description, "read_only": t.read_only,
                              "destructive": t.destructive, "long_running": t.long_running}


_register_static_specs()
