"""Benchmarks: measured, never estimated.

RTF (real-time factor) = generation_seconds / audio_seconds. Below 1.0 means
faster than real time. Load time includes starting the backend worker
process. Peak VRAM comes from torch's allocator inside the worker (CUDA only);
peak RAM is the worker's high-water RSS.
"""

from __future__ import annotations

import platform
import shutil
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prompt2music import __version__, envs, paths
from prompt2music._shared.gpu import query_gpus
from prompt2music.backends.base import GenerationRequest
from prompt2music.config import MusicConfig
from prompt2music.engine import Engine
from prompt2music.errors import MusicError
from prompt2music.registry import ModelSpec, all_models, get_model

DEFAULT_PROMPT = "Warm ambient piano with soft strings and gentle pads"


def _one_model(cfg: MusicConfig, engine: Engine, spec: ModelSpec, *, duration: float,
               runs: int, prompt: str, seed: int, device: str | None,
               precision: str | None, progress: Callable[[float | None, str], None]
               ) -> dict[str, Any]:
    engine.unload()
    t0 = time.perf_counter()
    info = engine.load(spec, device=device, precision=precision)
    load_wall = time.perf_counter() - t0
    workdir = paths.ensure(paths.cache_dir() / "bench" / spec.id)
    task = sorted(spec.tasks, key=lambda t: t.value)[0]
    lyrics = "[Verse 1]\nla la la\n[Chorus]\noh oh oh" if task.value == "song" else None
    dur = min(max(duration, spec.capabilities.min_duration), spec.capabilities.max_duration)
    samples = []
    try:
        for i in range(runs):
            progress(i / runs, f"{spec.id}: run {i + 1}/{runs}")
            req = GenerationRequest(prompt=prompt, model=spec.id, duration=dur, task=task,
                                    seed=seed, instrumental=spec.capabilities.instrumental,
                                    lyrics=lyrics, output_path=workdir / f"run{i + 1}.wav")
            t1 = time.perf_counter()
            data = engine.generate(spec, req, workdir)
            wall = time.perf_counter() - t1
            out = data["outputs"][0]
            gen = float(data["generation_seconds"])
            samples.append({
                "run": i + 1,
                "generation_seconds": round(gen, 3),
                "wall_seconds": round(wall, 3),
                "audio_seconds": out["duration"],
                "rtf": round(gen / out["duration"], 4) if out["duration"] else None,
                "sample_rate": out["sample_rate"],
                "channels": out["channels"],
                "peak_vram_mb": (data.get("memory") or {}).get("cuda_peak_allocated_mb"),
                "peak_ram_mb": data.get("peak_rss_mb"),
            })
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    rtfs = [s["rtf"] for s in samples if s["rtf"] is not None]
    vram = [s["peak_vram_mb"] for s in samples if s["peak_vram_mb"] is not None]
    plan = info.get("device_plan", {})
    return {
        "model": spec.id,
        "backend": spec.backend,
        "load_seconds_worker": info.get("load_seconds"),
        "load_seconds_wall": round(load_wall, 3),
        "duration_requested": dur,
        "runs": samples,
        "rtf_mean": round(statistics.mean(rtfs), 4) if rtfs else None,
        "rtf_min": min(rtfs) if rtfs else None,
        "peak_vram_mb": max(vram) if vram else None,
        "peak_ram_mb": max((s["peak_ram_mb"] or 0) for s in samples) or None,
        "sample_rate": samples[-1]["sample_rate"] if samples else None,
        "channels": samples[-1]["channels"] if samples else None,
        "config": {"device": plan.get("device"), "precision": plan.get("precision"),
                   "vram_budget_gb": plan.get("vram_budget_gb"),
                   "cpu_offload": cfg["hardware"]["cpu_offload"],
                   "torch_compile": cfg["hardware"]["torch_compile"],
                   "torch": envs.probe(cfg, spec.backend).torch, "seed": seed,
                   "prompt": prompt},
    }


def run_benchmark(cfg: MusicConfig, engine: Engine, params: dict[str, Any], *,
                  progress: Callable[[float | None, str], None] | None = None
                  ) -> dict[str, Any]:
    progress = progress or (lambda _f, _m: None)
    if params.get("all_installed"):
        specs = [s for s in all_models(include_diagnostic=False, kind="music")
                 if engine.manager.is_ready(s.id) and envs.probe(cfg, s.backend).ok]
        if not specs:
            raise MusicError("no installed music models with ready backends to benchmark")
    else:
        specs = [get_model(m) for m in (params.get("models") or [params.get("model")])
                 if m]
        if not specs:
            raise MusicError("name a model with --model or use --all-installed")
    results, errors = [], {}
    for spec in specs:
        try:
            results.append(_one_model(
                cfg, engine, spec, duration=float(params.get("duration") or 30),
                runs=int(params.get("runs") or 3), prompt=params.get("prompt") or DEFAULT_PROMPT,
                seed=int(params.get("seed") or 1234), device=params.get("device"),
                precision=params.get("precision"), progress=progress))
        except MusicError as exc:
            errors[spec.id] = exc.as_dict()
    report = {
        "schema": "prompt2music.benchmark/1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "app_version": __version__,
        "rtf_definition": "generation_seconds / audio_seconds (<1 is faster than real time)",
        "host": {"platform": platform.platform(), "python": platform.python_version(),
                 "gpus": [g.as_dict() for g in query_gpus()]},
        "results": results,
        "errors": errors,
    }
    out = params.get("output")
    if out:
        import json

        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["output"] = out
    return report
