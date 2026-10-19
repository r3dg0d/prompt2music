"""``prompt2music setup``: guided first run (also scriptable with --non-interactive)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from prompt2music import envs, paths
from prompt2music.doctor import collect
from prompt2music.errors import MusicError
from prompt2music.gpu import plan_device
from prompt2music.manager import ModelManager
from prompt2music.registry import ModelSpec, all_models


def _ask(ctx: Any, question: str, default: str, interactive: bool) -> str:
    if not interactive:
        return default
    reply = input(f"{question} [{default}] ").strip()
    return reply or default


def recommend(ctx: Any) -> list[ModelSpec]:
    """Models whose hardware needs fit this machine, best first."""
    fits = []
    for spec in all_models(include_diagnostic=False, kind="music"):
        try:
            plan_device(ctx.cfg, spec)
        except MusicError:
            continue
        fits.append(spec)
    return sorted(fits, key=lambda s: (-(s.quality + s.speed), s.approx_size_gb))


def run_setup(ctx: Any, args: argparse.Namespace) -> int:
    interactive = not args.non_interactive and ctx.out.is_terminal
    out = ctx.out
    cfg = ctx.cfg
    out.print("[bold cyan]prompt2music setup[/bold cyan] — local music generation\n")

    out.print("[bold]1. Hardware[/bold]")
    report = collect(cfg, quick=True)
    if report["gpus"]:
        for g in report["gpus"]:
            out.print(f"   {g['name']}  {g['memory_total_mb'] / 1024:.0f} GB VRAM")
    else:
        out.print("   no NVIDIA GPU detected — CPU-capable models only")
    out.print(f"   recommended: {report['recommended']['device']} "
              f"{report['recommended']['precision']}")

    out.print("\n[bold]2. Dependencies[/bold]")
    for tool in ("ffmpeg", "ffprobe", "uv", "git"):
        mark = "[green]✓[/green]" if report["tools"].get(tool) else "[red]missing[/red]"
        out.print(f"   {tool:8} {mark}")
    if not report["tools"].get("ffmpeg"):
        out.print("   [yellow]install ffmpeg before generating (NixOS: pkgs.ffmpeg)[/yellow]")

    out.print("\n[bold]3. Storage[/bold]")
    models_dir = _ask(ctx, "   model directory", str(cfg.models_dir), interactive)
    output_dir = _ask(ctx, "   output directory", cfg["general"]["output_directory"],
                      interactive)
    if models_dir != str(cfg.models_dir):
        cfg.set("models.directory", models_dir)
    cfg.set("general.output_directory", output_dir)
    probe = Path(models_dir).expanduser()
    while not probe.exists():
        probe = probe.parent
    out.print(f"   free space: {shutil.disk_usage(probe).free / 1024**3:.0f} GB")

    out.print("\n[bold]4. Recommended models for this machine[/bold]")
    recs = recommend(ctx)
    mgr = ModelManager(cfg)
    for spec in recs[:6]:
        state = "[green]installed[/green]" if mgr.installed(spec.id) else \
            f"~{spec.approx_size_gb:g} GB"
        out.print(f"   {spec.id:28} {state:12} {spec.license.split(' (')[0]} "
                  f"(commercial: {spec.commercial_use})")
    if not recs:
        out.print("   none fit the detected hardware/VRAM budget")

    out.print("\n[bold]5-6. Install a model[/bold]")
    choice = args.model or (recs[0].id if recs else "")
    if interactive:
        choice = _ask(ctx, "   model to install (blank to skip)", choice, True)
    elif not args.model:
        choice = ""
        out.print("   skipped (pass --model ID --yes to install non-interactively)")
    if choice:
        from prompt2music.cli import _download

        spec_backend = None
        try:
            _download(ctx, choice, accept_license=args.accept_license, yes=args.yes)
            from prompt2music.registry import get_model

            spec_backend = get_model(choice).backend
        except MusicError as exc:
            ctx.warn(f"download skipped: {exc}")
        if spec_backend and not envs.probe(cfg, spec_backend).ok:
            cmds = envs.install_commands(cfg, spec_backend)
            out.print(f"   backend env for {spec_backend}:")
            for c in cmds:
                out.print("     $ " + " ".join(x for x in c if x != "#best-effort"))
            if ctx.confirm("   build it now?", assume_yes=args.yes):
                envs.run_install(cfg, spec_backend, cmds, log=ctx.info)
        cfg.set("general.default_model", choice)

    out.print("\n[bold]7. Generation defaults[/bold]")
    duration = _ask(ctx, "   default duration (s)", str(cfg["generation"]["duration"]),
                    interactive)
    try:
        cfg.set("generation.duration", float(duration))
    except (ValueError, MusicError):
        ctx.warn("invalid duration; keeping the previous value")
    saved = cfg.save()
    out.print(f"   saved {saved}")

    out.print("\n[bold]8. MCP clients[/bold]")
    out.print("   Claude Code:  claude mcp add prompt2music -- prompt2music-mcp")
    out.print("   Others:       prompt2music mcp config --client opencode|cursor|vscode")

    out.print("\n[bold]9. Smoke test[/bold]")
    if args.skip_smoke:
        out.print("   skipped")
    else:
        _smoke(ctx, choice or None)

    out.print("\n[bold]10. Next steps[/bold]")
    out.print('   prompt2music generate "Dreamy ambient piano" --duration 20')
    out.print("   prompt2music doctor")
    out.print(f"   config: {paths.config_path()}")
    return 0


def _smoke(ctx: Any, model: str | None) -> None:
    from prompt2music import service
    from prompt2music.engine import Engine

    cfg = ctx.cfg
    engine = Engine(cfg)
    try:
        params = service.normalise_params(cfg, {
            "prompt": "setup smoke test", "model": "test-tone", "duration": 1.0,
            "output": str(paths.cache_dir() / "smoke" / "pipeline.wav"), "overwrite": True})
        result = service.run_generate(cfg, engine, params)
        ctx.out.print(f"   pipeline (diagnostic tone): [green]ok[/green] "
                      f"{result['outputs'][0]['path']}")
        if model and ModelManager(cfg).is_ready(model):
            params = service.normalise_params(cfg, {
                "prompt": "short calm piano phrase", "model": model, "duration": 10.0,
                "output": str(paths.cache_dir() / "smoke" / f"{model}.wav"),
                "overwrite": True})
            result = service.run_generate(cfg, engine, params)
            out = result["outputs"][0]
            ctx.out.print(f"   {model}: [green]ok[/green] {out['path']} "
                          f"({out['duration']:.1f}s, RTF {result['rtf']})")
    except MusicError as exc:
        ctx.warn(f"smoke test failed: {exc}" + (f" ({exc.hint})" if exc.hint else ""))
    finally:
        engine.unload()
