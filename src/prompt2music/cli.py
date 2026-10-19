"""prompt2music / aimusic — local music generation CLI (also ``ai-media music``)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from prompt2music import __version__, paths
from prompt2music._shared.signals import install_sigint_handler, interruptible_cli
from prompt2music.config import MusicConfig, load_music_config, parse_cli_value
from prompt2music.errors import (
    ERRORS_BY_CODE,
    CancelledError,
    InvalidRequestError,
    LicenseNotAcceptedError,
    MusicError,
)

Handler = Callable[["Ctx", argparse.Namespace], int]


class Ctx:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.json = bool(getattr(args, "json", False))
        self.quiet = bool(getattr(args, "quiet", False))
        self.out = Console(quiet=self.quiet and not self.json, highlight=False)
        self.err = Console(stderr=True, highlight=False)
        self._cfg: MusicConfig | None = None

    @property
    def cfg(self) -> MusicConfig:
        if self._cfg is None:
            path = Path(self.args.config).expanduser() if self.args.config else None
            self._cfg = load_music_config(path, create=True)
        return self._cfg

    def emit(self, data: Any) -> None:
        print(json.dumps(data, indent=2, default=str))

    def info(self, msg: str) -> None:
        if not self.quiet and not self.json:
            self.err.print(f"[cyan]…[/cyan] {escape(msg)}")

    def ok(self, msg: str) -> None:
        if not self.json:
            self.out.print(f"[green]✔[/green] {escape(msg)}")

    def warn(self, msg: str) -> None:
        self.err.print(f"[yellow]![/yellow] {escape(msg)}")

    def confirm(self, question: str, *, assume_yes: bool) -> bool:
        if assume_yes:
            return True
        if not sys.stdin.isatty():
            return False
        reply = input(f"{question} [y/N] ").strip().lower()
        return reply in ("y", "yes")


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def _gen_options(p: argparse.ArgumentParser, *, prompt_positional: bool = True,
                 duration_help: str = "seconds of audio") -> None:
    if prompt_positional:
        p.add_argument("prompt_pos", nargs="?", metavar="PROMPT", help="music description")
    p.add_argument("-p", "--prompt", help="music description (alternative to PROMPT)")
    p.add_argument("-m", "--model", help="model id or 'auto' (default from config)")
    p.add_argument("-d", "--duration", type=float, help=duration_help)
    p.add_argument("--seed", type=int, help="seed (-1 = random)")
    p.add_argument("--negative", dest="negative_prompt", help="negative prompt (if supported)")
    p.add_argument("--bpm", type=int, help="tempo")
    p.add_argument("--key", help="musical key, e.g. Fm or 'F minor'")
    p.add_argument("--time-signature", help="e.g. 4/4")
    voc = p.add_mutually_exclusive_group()
    voc.add_argument("--instrumental", dest="instrumental", action="store_true", default=None)
    voc.add_argument("--vocals", dest="instrumental", action="store_false")
    p.add_argument("--reference", dest="reference_audio", help="reference audio (if supported)")
    p.add_argument("--batch", dest="batch_size", type=int, help="outputs per request")
    p.add_argument("--steps", type=int, help="sampling steps (diffusion models)")
    p.add_argument("--guidance", type=float, help="guidance / CFG scale")
    p.add_argument("-o", "--output", help="output file (format from extension) or directory")
    p.add_argument("-f", "--format", choices=["wav", "flac", "mp3", "ogg", "opus"])
    p.add_argument("--enhance", action="store_true", default=None, help="expand the prompt")
    p.add_argument("--raw-prompt", action="store_true",
                   help="send the prompt exactly as written (no enhancement or folding)")
    p.add_argument("--device", help="auto, cpu, cuda, cuda:N, mps")
    p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--vram-limit", help="VRAM cap, e.g. 20G")
    p.add_argument("--quality", choices=["fast", "balanced", "best"],
                   help="preference for --model auto")
    p.add_argument("--normalize", choices=["peak", "loudness"], help="post-process level")
    p.add_argument("--fade-in", type=float, help="post-process fade-in seconds")
    p.add_argument("--fade-out", type=float, help="post-process fade-out seconds")
    p.add_argument("--background", action="store_true", help="queue as a daemon job and exit")
    p.add_argument("--no-daemon", action="store_true",
                   help="run in this process even if the daemon is running")
    p.add_argument("--dry-run", action="store_true", help="show the resolved request only")
    p.add_argument("--force", dest="overwrite", action="store_true", help="overwrite output")


def build_parser(prog: str = "prompt2music") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Local music generation with open-weight models. "
        "Your words. Your music. Your hardware.",
        epilog="Docs: docs/music/ in the ai-media-cli repository.",
    )
    parser.add_argument("--version", action="version", version=f"{prog} {__version__}")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true", help="show tracebacks")
    parser.add_argument("--config", help="config file (default ~/.config/ai-media/"
                        "prompt2music.toml)")
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    p = sub.add_parser("doctor", help="system diagnostics")
    p.add_argument("--quick", action="store_true", help="skip backend env probes")

    p = sub.add_parser("setup", help="interactive first-run setup")
    p.add_argument("--non-interactive", action="store_true")
    p.add_argument("--model", help="model to install")
    p.add_argument("--accept-license", action="store_true")
    p.add_argument("--yes", action="store_true", help="approve downloads/installs")
    p.add_argument("--skip-smoke", action="store_true")

    p = sub.add_parser("generate", help="generate music from a prompt")
    _gen_options(p)
    p.add_argument("--melody", dest="melody_audio", help="melody conditioning audio")
    p.add_argument("--lyrics", help="lyrics file (vocal models)")
    p.add_argument("--preset", help="start from a named preset (see `presets`)")
    p.add_argument("--prompts-file", help="batch: one prompt per line")

    p = sub.add_parser("song", help="generate a song with lyrics")
    _gen_options(p, prompt_positional=False)
    p.add_argument("--lyrics", required=True, help="lyrics file ('-' for stdin)")
    p.add_argument("--genre")
    p.add_argument("--mood")
    p.add_argument("--vocal-style", help="e.g. 'warm female vocal'")
    p.add_argument("--instrumentation")
    p.add_argument("--language", dest="vocal_language", help="ISO 639-1 code, e.g. en")

    p = sub.add_parser("continue", help="extend existing audio (native continuation)")
    _gen_options(p, prompt_positional=False, duration_help="seconds of NEW audio to add")
    p.add_argument("-i", "--input", dest="input_audio", required=True)
    p.add_argument("--mode", dest="continue_mode", default="natural",
                   choices=["natural", "intensify", "transition", "add-instruments", "mood",
                            "outro"])
    p.add_argument("--to", dest="continue_target", help="genre/instruments/mood for --mode")
    p.add_argument("--only-new", action="store_true", help="write only the new section")

    p = sub.add_parser("edit", help="audio-to-audio edit / style transfer")
    _gen_options(p, prompt_positional=False)
    p.add_argument("-i", "--input", dest="input_audio", required=True)
    p.add_argument("--strength", type=float,
                   help="0 keeps the source, 1 ignores it (default model-specific)")

    p = sub.add_parser("inpaint", help="regenerate a time region")
    _gen_options(p, prompt_positional=False)
    p.add_argument("-i", "--input", dest="input_audio", required=True)
    p.add_argument("--start", dest="region_start", type=float, required=True)
    p.add_argument("--end", dest="region_end", type=float, required=True)

    p = sub.add_parser("enhance", help="expand a short prompt (no generation)")
    p.add_argument("prompt")
    p.add_argument("--vocals", action="store_true")
    p.add_argument("--bpm", type=int)
    p.add_argument("--key")
    p.add_argument("--no-llm", action="store_true", help="rule-based only")

    sub.add_parser("presets", help="list prompt presets")

    lp = sub.add_parser("lyrics", help="lyrics helpers")
    lsub = lp.add_subparsers(dest="lyrics_cmd", metavar="ACTION")
    p = lsub.add_parser("generate", help="draft lyrics with a local LLM")
    p.add_argument("--prompt", required=True)
    p.add_argument("--structure", default="standard")
    p.add_argument("--genre")
    p.add_argument("--mood")
    p.add_argument("--language", default="English")
    p.add_argument("-o", "--output")
    p = lsub.add_parser("template", help="print a section template")
    p.add_argument("--structure", default="standard")
    p.add_argument("--topic")
    p = lsub.add_parser("validate", help="check lyric formatting")
    p.add_argument("file")
    p.add_argument("--model")

    mp = sub.add_parser("models", help="discover, download and manage models")
    msub = mp.add_subparsers(dest="models_cmd", metavar="ACTION")
    msub.add_parser("list", help="installed models")
    p = msub.add_parser("available", help="catalogue")
    p.add_argument("--task", choices=["text-to-music", "song", "continue", "edit", "inpaint"])
    p.add_argument("--kind", choices=["music", "stems"])
    p = msub.add_parser("search", help="search the catalogue")
    p.add_argument("query")
    for name, helptext in (("info", "model details"), ("verify", "check file hashes"),
                           ("update", "re-download if upstream changed")):
        p = msub.add_parser(name, help=helptext)
        p.add_argument("model")
        if name == "verify":
            p.add_argument("--quick", action="store_true", help="sizes only")
        if name == "update":
            p.add_argument("--yes", action="store_true")
            p.add_argument("--accept-license", action="store_true")
    p = msub.add_parser("download", help="download from Hugging Face")
    p.add_argument("model")
    p.add_argument("--revision", help="commit/tag, or org/repo@rev[,org/repo@rev]")
    p.add_argument("--accept-license", action="store_true")
    p.add_argument("--yes", action="store_true", help="skip the size confirmation")
    p.add_argument("--trust-repo-code", action="store_true",
                   help="also fetch .py files shipped in the repo (never imported here)")
    p.add_argument("--plan", action="store_true", help="show the plan and exit")
    p = msub.add_parser("import", help="register weights already on disk")
    p.add_argument("model")
    p.add_argument("path")
    p.add_argument("--link", action="store_true", help="symlink instead of copying")
    p.add_argument("--accept-license", action="store_true")
    p = msub.add_parser("load", help="warm a model in the job daemon")
    p.add_argument("model")
    p.add_argument("--device")
    p.add_argument("--precision")
    p.add_argument("--vram-limit")
    msub.add_parser("unload", help="unload the daemon's model")
    p = msub.add_parser("remove", help="delete a model")
    p.add_argument("model")
    p.add_argument("--yes", action="store_true")
    p = msub.add_parser("cache", help="storage usage")
    p.add_argument("--clean", action="store_true", help="delete partial downloads")
    p = msub.add_parser("benchmark", help="benchmark one model")
    p.add_argument("model")
    p.add_argument("--duration", type=float, default=30)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("-o", "--output")

    bp = sub.add_parser("backends", help="backend Python environments")
    bsub = bp.add_subparsers(dest="backends_cmd", metavar="ACTION")
    bsub.add_parser("list", help="status of every backend env")
    p = bsub.add_parser("install", help="create/update a backend env")
    p.add_argument("backend")
    p.add_argument("--python", help="base interpreter or version for the venv")
    p.add_argument("--ref", help="upstream git ref (default: main)")
    p.add_argument("--torch-index", help="PyTorch wheel index URL (pip backends)")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--print", dest="print_only", action="store_true",
                   help="print the commands only")

    sp = sub.add_parser("stems", help="stem separation (Demucs)")
    ssub = sp.add_subparsers(dest="stems_cmd", metavar="ACTION")
    ssub.add_parser("models", help="available stem models")
    p = ssub.add_parser("download", help="download a stem model")
    p.add_argument("model")
    p.add_argument("--yes", action="store_true")
    p = ssub.add_parser("separate", help="split a track into stems")
    p.add_argument("input")
    p.add_argument("-m", "--model", default="htdemucs")
    p.add_argument("--output-dir")
    p.add_argument("--stems", help="comma list, e.g. vocals,drums")
    p.add_argument("-f", "--format", choices=["wav", "flac", "mp3", "ogg", "opus"])
    p.add_argument("--device")
    p.add_argument("--background", action="store_true")
    p.add_argument("--no-daemon", action="store_true")
    p.add_argument("--force", dest="overwrite", action="store_true")

    ap = sub.add_parser("audio", help="audio processing toolkit (FFmpeg)")
    asub = ap.add_subparsers(dest="audio_cmd", metavar="ACTION")
    p = asub.add_parser("info", help="format, duration, levels")
    p.add_argument("file")
    p.add_argument("--loudness", action="store_true")
    p = asub.add_parser("validate", help="decode / silence / duration checks")
    p.add_argument("file")
    p.add_argument("--expect-duration", type=float)
    p = asub.add_parser("convert")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--sample-rate", type=int)
    p.add_argument("--channels", type=int)
    p.add_argument("--bitrate")
    p.add_argument("--tag", action="append", default=[], help="KEY=VALUE metadata")
    p.add_argument("--force", action="store_true")
    p = asub.add_parser("normalize")
    p.add_argument("input")
    p.add_argument("-o", "--output")
    p.add_argument("--mode", choices=["peak", "loudness"], default="loudness")
    p.add_argument("--target", type=float, help="dBFS (peak) or LUFS (loudness)")
    p.add_argument("--force", action="store_true")
    p = asub.add_parser("trim")
    p.add_argument("input")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float)
    p.add_argument("-o", "--output")
    p.add_argument("--force", action="store_true")
    p = asub.add_parser("fade")
    p.add_argument("input")
    p.add_argument("--in", dest="fade_in", type=float, default=0.0)
    p.add_argument("--out", dest="fade_out", type=float, default=0.0)
    p.add_argument("-o", "--output")
    p.add_argument("--force", action="store_true")
    p = asub.add_parser("loop")
    p.add_argument("input")
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--crossfade", type=float, default=0.0)
    p.add_argument("-o", "--output")
    p.add_argument("--force", action="store_true")
    p = asub.add_parser("mix")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--output")
    p.add_argument("--weights", help="comma list, one per input")
    p.add_argument("--duration", choices=["longest", "shortest", "first"], default="longest")
    p.add_argument("--voice", help="narration to duck the music under")
    p.add_argument("--ducking", action="store_true", help="with --voice: sidechain duck")
    p.add_argument("--force", action="store_true")
    p = asub.add_parser("crossfade")
    p.add_argument("first")
    p.add_argument("second")
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--force", action="store_true")
    p = asub.add_parser("fit", help="trim/loop to an exact duration")
    p.add_argument("input")
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("-o", "--output")
    p.add_argument("--force", action="store_true")
    for name in ("waveform", "spectrogram"):
        p = asub.add_parser(name)
        p.add_argument("input")
        p.add_argument("-o", "--output")
        p.add_argument("--force", action="store_true")
    p = asub.add_parser("tag", help="embed metadata in place")
    p.add_argument("input")
    p.add_argument("tags", nargs="+", help="KEY=VALUE")

    p = sub.add_parser("soundtrack", help="score a video with generated music")
    _gen_options(p)
    p.add_argument("--video", required=True)
    p.add_argument("--original-audio", choices=["duck", "mix", "replace"], default="duck")
    p.add_argument("--music-gain", type=float, default=0.0, help="dB applied to the music")

    jp = sub.add_parser("jobs", help="background jobs")
    jsub = jp.add_subparsers(dest="jobs_cmd", metavar="ACTION")
    p = jsub.add_parser("list")
    p.add_argument("--status")
    p.add_argument("--limit", type=int, default=20)
    for name in ("status", "cancel", "result", "wait"):
        p = jsub.add_parser(name)
        p.add_argument("job_id")
    p = jsub.add_parser("logs")
    p.add_argument("job_id")
    p.add_argument("--follow", action="store_true")
    p = jsub.add_parser("prune")
    p.add_argument("--keep", type=int, default=500)

    dp = sub.add_parser("daemon", help="job daemon control")
    dsub = dp.add_subparsers(dest="daemon_cmd", metavar="ACTION")
    dsub.add_parser("start")
    dsub.add_parser("stop")
    dsub.add_parser("status")
    p = dsub.add_parser("run", help="run in the foreground (systemd)")
    p.add_argument("--idle-exit", type=float, default=None)
    dsub.add_parser("install-service", help="write a systemd --user unit")

    p = sub.add_parser("benchmark", help="measure load time, VRAM and RTF")
    p.add_argument("--model", action="append", dest="models")
    p.add_argument("--all-installed", action="store_true")
    p.add_argument("--duration", type=float, default=30)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--prompt")
    p.add_argument("--device")
    p.add_argument("--precision")
    p.add_argument("-o", "--output")

    lbp = sub.add_parser("library", help="generated track library")
    lbsub = lbp.add_subparsers(dest="library_cmd", metavar="ACTION")
    p = lbsub.add_parser("list")
    p.add_argument("--limit", type=int, default=30)
    p = lbsub.add_parser("search")
    p.add_argument("query")
    p = lbsub.add_parser("info")
    p.add_argument("track_id")
    p = lbsub.add_parser("export")
    p.add_argument("track_id")
    p.add_argument("-o", "--output", default=".")
    p.add_argument("-f", "--format", choices=["wav", "flac", "mp3", "ogg", "opus"])
    p.add_argument("--force", action="store_true")
    p = lbsub.add_parser("reproduce", help="regenerate with the same model/seed/params")
    p.add_argument("track_id")
    p.add_argument("-o", "--output")
    p = lbsub.add_parser("remove")
    p.add_argument("track_id")
    p.add_argument("--delete-files", action="store_true")

    cp = sub.add_parser("config", help="show and edit configuration")
    csub = cp.add_subparsers(dest="config_cmd", metavar="ACTION")
    csub.add_parser("show")
    csub.add_parser("path")
    p = csub.add_parser("get")
    p.add_argument("key")
    p = csub.add_parser("set")
    p.add_argument("key")
    p.add_argument("value")
    p = csub.add_parser("reset")
    p.add_argument("--yes", action="store_true")

    xp = sub.add_parser("mcp", help="MCP server")
    xsub = xp.add_subparsers(dest="mcp_cmd", metavar="ACTION")
    p = xsub.add_parser("serve", help="run the MCP server (default)")
    p.add_argument("--transport", choices=["stdio", "http"])
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p = xsub.add_parser("config", help="print client configuration")
    p.add_argument("--client", default="generic",
                   choices=["generic", "claude", "opencode", "cursor", "vscode"])
    xsub.add_parser("tools", help="list exposed tools")

    p = sub.add_parser("completion", help="print a shell completion script")
    p.add_argument("shell", choices=["bash", "zsh", "fish"])
    _share_global_flags(parser)
    return parser


def _share_global_flags(parser: argparse.ArgumentParser) -> None:
    """Accept --json/-q/-v/--debug after subcommands too (SUPPRESS keeps the
    top-level value when the flag is absent at the deeper level)."""
    for action in parser._actions:  # argparse has no public API for this
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for child in action.choices.values():
            child.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                               help=argparse.SUPPRESS)
            child.add_argument("-q", "--quiet", action="store_true",
                               default=argparse.SUPPRESS, help=argparse.SUPPRESS)
            child.add_argument("-v", "--verbose", action="store_true",
                               default=argparse.SUPPRESS, help=argparse.SUPPRESS)
            child.add_argument("--debug", action="store_true", default=argparse.SUPPRESS,
                               help=argparse.SUPPRESS)
            _share_global_flags(child)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _read_text_arg(value: str) -> str:
    if value == "-":
        return sys.stdin.read()
    path = Path(value).expanduser()
    if not path.is_file():
        raise InvalidRequestError(f"file not found: {path}")
    return path.read_text(encoding="utf-8")


def _gen_params(ctx: Ctx, args: argparse.Namespace, task: str) -> dict[str, Any]:
    from prompt2music.gpu import parse_size_gb

    prompt = getattr(args, "prompt", None) or getattr(args, "prompt_pos", None) or ""
    raw: dict[str, Any] = {"task": task, "prompt": prompt}
    for key in ("model", "duration", "seed", "negative_prompt", "bpm", "key", "time_signature",
                "instrumental", "reference_audio", "batch_size", "steps", "guidance", "output",
                "format", "enhance", "raw_prompt", "device", "precision", "quality",
                "normalize", "fade_in", "fade_out", "overwrite", "input_audio",
                "region_start", "region_end", "strength", "melody_audio", "continue_mode",
                "continue_target", "only_new", "vocal_language"):
        value = getattr(args, key, None)
        if value is not None:  # False is meaningful (e.g. --vocals → instrumental=False)
            raw[key] = value
    if getattr(args, "vram_limit", None):
        raw["vram_limit_gb"] = parse_size_gb(args.vram_limit)
    if getattr(args, "lyrics_text", None):
        raw["lyrics"] = args.lyrics_text
    elif getattr(args, "lyrics", None):
        raw["lyrics"] = _read_text_arg(args.lyrics)
    return raw


def _print_table(ctx: Ctx, title: str, columns: tuple[str, ...],
                 rows: Sequence[Sequence[str]], optional: tuple[str, ...] = ()) -> None:
    """One line per row: fixed columns keep their content width, the last
    column absorbs the rest (ellipsis). ``optional`` columns are dropped, in
    order, until the table fits the terminal."""
    widths = [max([len(c)] + [len(r[i]) for r in rows]) for i, c in enumerate(columns)]
    keep = list(range(len(columns)))

    def fixed() -> int:  # every column costs 3 cells of padding+border, plus the edge
        return sum(widths[i] + 3 for i in keep[:-1]) + 3 + 1

    for name in optional:
        if fixed() + 10 <= ctx.out.width or name not in columns:
            continue
        keep.remove(columns.index(name))
    last_width = max(10, ctx.out.width - fixed())
    table = Table(title=title)
    for i in keep:
        if i == keep[-1]:
            table.add_column(columns[i], no_wrap=True, overflow="ellipsis",
                             max_width=last_width)
        else:
            table.add_column(columns[i], no_wrap=True, min_width=widths[i])
    kept = [tuple(str(r[i]) for i in keep) for r in rows]
    for row in kept:
        table.add_row(*row)
    ctx.out.print(table)


def _render_result(ctx: Ctx, result: dict[str, Any]) -> None:
    if ctx.json:
        ctx.emit(result)
        return
    rows = [(out["track_id"], f"{out['duration']:.2f}s", str(out["sample_rate"]),
             str(out["channels"]), str(out["seed"]), out["path"]) for out in result["outputs"]]
    _print_table(ctx, f"{result['model']} · {result['task']}",
                 ("track", "duration", "rate", "ch", "seed", "file"), rows,
                 optional=("rate", "ch", "seed"))
    ctx.out.print(f"generation {result['generation_seconds']:.2f}s · RTF {result['rtf']} "
                  f"(generation/audio seconds; <1 is faster than real time)")
    if result.get("prompt_used") and result.get("prompt_used") != result.get("prompt"):
        ctx.out.print(f"[dim]prompt sent: {result['prompt_used']}[/dim]")
    for out in result["outputs"]:
        for problem in out["validation"]["problems"]:
            ctx.warn(f"{Path(out['path']).name}: {problem}")


def _error_from_job(job: Any) -> MusicError:
    err = job.error or {}
    cls = ERRORS_BY_CODE.get(str(err.get("code")), MusicError)
    return cls(str(err.get("message", "job failed")), hint=err.get("hint"))


def _run_job(ctx: Ctx, kind: str, params: dict[str, Any], *, model: str | None,
             prompt: str | None, background: bool, no_daemon: bool) -> dict[str, Any] | None:
    """Queue (background), hand to a running daemon (warm model), or run here."""
    from prompt2music import daemon
    from prompt2music.jobs import JobStore

    store = JobStore()
    cfg = ctx.cfg
    max_q = int(cfg["jobs"]["max_queued"])
    if background:
        job_id = store.create(kind, params, model=model, prompt=prompt, max_queued=max_q)
        info = daemon.ensure_running(cfg)
        if ctx.json:
            ctx.emit({"job_id": job_id, "status": "queued", "daemon_pid": info.get("pid")})
        else:
            ctx.ok(f"queued job {job_id} (daemon pid {info.get('pid')})")
            ctx.out.print(f"  prompt2music jobs status {job_id}\n"
                          f"  prompt2music jobs wait {job_id}")
        return None
    progress_ui = _Progress(ctx)
    if daemon.status()["running"] and not no_daemon:
        job_id = store.create(kind, params, model=model, prompt=prompt, max_queued=max_q)
        ctx.info(f"handing job {job_id} to the running daemon")
        try:
            with progress_ui:
                job = daemon.follow(store, job_id, on_update=progress_ui.job_update)
        except KeyboardInterrupt:
            store.request_cancel(job_id)
            ctx.warn(f"cancellation requested for {job_id}")
            raise
    else:
        from prompt2music.daemon import execute_job
        from prompt2music.engine import Engine

        job_id = store.create(kind, params, model=model, prompt=prompt, status="loading")
        store.update(job_id, daemon_pid=os.getpid(), started_at=_now())
        engine = Engine(cfg)
        try:
            with progress_ui:
                job = execute_job(cfg, engine, store, store.get(job_id),
                                  on_progress=progress_ui.update)
        except KeyboardInterrupt:
            engine.cancel(grace=0)
            store.update(job_id, status="cancelled", message="interrupted",
                         error={"code": "cancelled", "message": "interrupted (Ctrl+C)"})
            raise
        finally:
            engine.unload()
    if job.status == "completed":
        return dict(job.result or {})
    if job.status == "cancelled":
        raise CancelledError(f"job {job.id} was cancelled")
    raise _error_from_job(job)


def _now() -> str:
    from prompt2music.db import now

    return now()


class _Progress:
    """Rich progress bar on stderr; silent in --json/--quiet mode."""

    def __init__(self, ctx: Ctx) -> None:
        self.enabled = not (ctx.json or ctx.quiet) and ctx.err.is_terminal
        self.ctx = ctx
        self._progress: Any = None
        self._task: Any = None

    def __enter__(self) -> _Progress:
        if self.enabled:
            from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

            self._progress = Progress(TextColumn("[cyan]{task.description}"), BarColumn(),
                                      TextColumn("{task.percentage:>3.0f}%"),
                                      TimeElapsedColumn(), console=self.ctx.err, transient=True)
            self._progress.start()
            self._task = self._progress.add_task("starting", total=1.0)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._progress is not None:
            self._progress.stop()

    def update(self, fraction: float | None, message: str) -> None:
        if self._progress is None:
            if not self.ctx.quiet and not self.ctx.json and fraction is None:
                self.ctx.info(message)
            return
        kwargs: dict[str, Any] = {"description": message[:60]}
        if fraction is not None:
            kwargs["completed"] = max(0.0, min(1.0, fraction))
        self._progress.update(self._task, **kwargs)

    def job_update(self, job: Any) -> None:
        self.update(job.progress, f"{job.status}: {job.message or ''}")


# ---------------------------------------------------------------------------
# generation commands
# ---------------------------------------------------------------------------
def cmd_generate(ctx: Ctx, args: argparse.Namespace, task: str = "text-to-music") -> int:
    from prompt2music import service

    raw = _gen_params(ctx, args, task)
    if getattr(args, "preset", None):
        from prompt2music.prompts import apply_preset

        preset = apply_preset(args.preset, raw.get("prompt") or None)
        raw["prompt"] = preset["prompt"]
        raw.setdefault("bpm", preset.get("bpm"))
    prompts = [raw.get("prompt", "")]
    if getattr(args, "prompts_file", None):
        prompts = [ln.strip() for ln in _read_text_arg(args.prompts_file).splitlines()
                   if ln.strip() and not ln.lstrip().startswith("#")]
        if not prompts:
            raise InvalidRequestError("prompts file is empty")
    if task == "text-to-music" and not any(prompts):
        raise InvalidRequestError("give a prompt: prompt2music generate \"...\"")
    results = []
    for prompt in prompts:
        raw["prompt"] = prompt
        params = service.normalise_params(ctx.cfg, raw)
        prepared = service.prepare(ctx.cfg, params)  # fail fast before queueing
        if args.dry_run:
            plan = {"model": prepared.spec.id, "selection": prepared.selection.as_dict(),
                    "request": prepared.request.to_dict(),
                    "enhancement": prepared.enhancement}
            if ctx.json:
                ctx.emit(plan)
            else:
                ctx.out.print_json(json.dumps(plan, default=str))
            continue
        result = _run_job(ctx, "generate", params, model=params["model"], prompt=prompt,
                          background=args.background, no_daemon=args.no_daemon)
        if result is not None:
            results.append(result)
            if len(prompts) == 1 or not ctx.json:
                _render_result(ctx, result)
    if ctx.json and len(prompts) > 1 and results:
        ctx.emit(results)
    return 0


def cmd_song(ctx: Ctx, args: argparse.Namespace) -> int:
    from prompt2music.lyrics import validate

    lyrics = _read_text_arg(args.lyrics)
    args.lyrics_text = lyrics
    report = validate(lyrics)
    if not report.ok:
        raise InvalidRequestError("lyrics problems: " + "; ".join(report.errors))
    for w in report.warnings:
        ctx.warn(w)
    parts = [args.prompt, args.genre, args.mood, args.vocal_style, args.instrumentation]
    args.prompt = ", ".join(p for p in parts if p) or "song"
    if args.instrumental is None:
        args.instrumental = False
    return cmd_generate(ctx, args, task="song")


def cmd_continue(ctx: Ctx, args: argparse.Namespace) -> int:
    return cmd_generate(ctx, args, task="continue")


def cmd_edit(ctx: Ctx, args: argparse.Namespace) -> int:
    if not args.prompt:
        raise InvalidRequestError("edit needs --prompt describing the change")
    return cmd_generate(ctx, args, task="edit")


def cmd_inpaint(ctx: Ctx, args: argparse.Namespace) -> int:
    return cmd_generate(ctx, args, task="inpaint")


def cmd_soundtrack(ctx: Ctx, args: argparse.Namespace) -> int:
    from prompt2music import service, workflows

    video = Path(args.video).expanduser().resolve()
    seconds = workflows.video_duration(video)
    if seconds <= 0:
        raise InvalidRequestError(f"could not read the duration of {video}")
    raw = _gen_params(ctx, args, "text-to-music")
    final = Path(raw.pop("output")) if raw.get("output") else video.with_name(
        f"{video.stem}-scored{video.suffix}")
    raw["duration"] = raw.get("duration") or seconds
    raw["output"] = str(paths.cache_dir() / "soundtracks" / f"{video.stem}.wav")
    raw["overwrite"] = True
    params = service.normalise_params(ctx.cfg, raw)
    result = _run_job(ctx, "generate", params, model=params["model"], prompt=params["prompt"],
                      background=False, no_daemon=args.no_daemon)
    assert result is not None
    music = Path(result["outputs"][0]["path"])
    workflows.mux_soundtrack(video, music, final, keep_original=args.original_audio,
                             music_gain_db=args.music_gain, overwrite=args.overwrite)
    payload = {"video": str(final), "music": str(music), "video_seconds": seconds,
               "model": result["model"], "original_audio": args.original_audio}
    if ctx.json:
        ctx.emit(payload)
    else:
        ctx.ok(f"scored video written to {final}")
        ctx.out.print(f"  music: {music} ({result['model']}), fitted to {seconds:.2f}s")
    return 0


def cmd_enhance(ctx: Ctx, args: argparse.Namespace) -> int:
    from prompt2music.llm import get_llm
    from prompt2music.prompts import enhance

    llm = None if args.no_llm else get_llm(ctx.cfg)
    result = enhance(args.prompt, instrumental=not args.vocals, bpm=args.bpm, key=args.key,
                     llm=llm)
    if ctx.json:
        ctx.emit(result.as_dict())
    else:
        ctx.out.print(f"[bold]{result.prompt}[/bold]")
        ctx.out.print(f"[dim]negative: {result.negative_prompt}[/dim]")
        ctx.out.print(f"[dim]method: {result.method}[/dim]")
    return 0


def cmd_presets(ctx: Ctx, args: argparse.Namespace) -> int:
    from prompt2music.prompts import PRESETS

    if ctx.json:
        ctx.emit(PRESETS)
        return 0
    table = Table(title="Presets (prompt2music generate --preset NAME)")
    table.add_column("name")
    table.add_column("bpm")
    table.add_column("prompt")
    for name, p in PRESETS.items():
        table.add_row(name, str(p.get("bpm", "")), p["prompt"])
    ctx.out.print(table)
    return 0


def cmd_lyrics(ctx: Ctx, args: argparse.Namespace) -> int:
    from prompt2music import lyrics

    if args.lyrics_cmd == "template":
        text = lyrics.template(args.structure, topic=args.topic)
        print(text, end="")
        return 0
    if args.lyrics_cmd == "validate":
        backend = None
        if args.model:
            from prompt2music.registry import get_model

            backend = get_model(args.model).backend
        report = lyrics.validate(_read_text_arg(args.file), backend=backend)
        if ctx.json:
            ctx.emit(report.as_dict())
        else:
            ctx.out.print(f"sections: {', '.join(report.sections) or '-'} · lines {report.lines}"
                          f" · {report.characters} chars")
            for w in report.warnings:
                ctx.warn(w)
            for e in report.errors:
                ctx.err.print(f"[red]✖[/red] {e}")
            if report.ok:
                ctx.ok("lyrics look valid")
        return 0 if report.ok else 2
    if args.lyrics_cmd == "generate":
        from prompt2music.llm import get_llm

        llm = get_llm(ctx.cfg, required=True)
        text = lyrics.generate_lyrics(llm, args.prompt, structure=args.structure,
                                      genre=args.genre, mood=args.mood, language=args.language)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            ctx.ok(f"lyrics written to {args.output}")
        else:
            print(text, end="")
        return 0
    raise InvalidRequestError("usage: prompt2music lyrics {generate,template,validate}")


# ---------------------------------------------------------------------------
# models / backends / stems
# ---------------------------------------------------------------------------
TASK_ABBR = {"text-to-music": "t2m", "song": "song", "continue": "cont", "edit": "edit",
             "inpaint": "inp"}


def _short_license(text: str) -> str:
    return text.split(" (")[0].replace("Stability AI Community License", "Stability Community")


def _print_models(ctx: Ctx, title: str, specs: list[Any]) -> None:
    from prompt2music.manager import ModelManager

    mgr = ModelManager(ctx.cfg)
    rows = []
    for s in specs:
        tasks = ",".join(TASK_ABBR[t.value] for t in sorted(s.tasks, key=lambda t: t.value))
        installed = "yes" if (not s.needs_download or mgr.installed(s.id)) else ""
        rows.append((s.id, tasks or s.kind, "yes" if s.capabilities.vocals else "",
                     f"{s.approx_size_gb:g}G" if s.needs_download else "-",
                     f"{s.min_vram_gb:g}G" if s.min_vram_gb else "-",
                     s.commercial_use + ("*" if s.gated else ""), installed,
                     _short_license(s.license)))
    _print_table(ctx, title, ("id", "tasks", "vocals", "size", "vram", "commercial", "inst",
                              "license"), rows, optional=("vram", "vocals", "commercial"))
    if not ctx.json:
        ctx.out.print("[dim]tasks: t2m=text-to-music cont=continue inp=inpaint · "
                      "* gated on Hugging Face[/dim]")


def _download(ctx: Ctx, model_id: str, *, revision: str | None = None,
              accept_license: bool = False, yes: bool = False, trust_code: bool = False,
              plan_only: bool = False) -> int:
    from prompt2music.downloader import HfHub, execute_plan, plan_download
    from prompt2music.registry import get_model

    cfg = ctx.cfg
    spec = get_model(model_id)
    if not spec.needs_download:
        ctx.ok(f"{spec.id} needs no download")
        return 0
    hub = HfHub(cfg["models"]["hf_endpoint"] or None)
    ctx.info(f"querying Hugging Face for {spec.id}")
    plan = plan_download(spec, cfg.models_dir, hub=hub, token=cfg.hf_token(),
                         revision=revision, trust_repo_code=trust_code, offline=cfg.offline)
    data = plan.as_dict()
    if ctx.json and plan_only:
        ctx.emit(data)
        return 0
    if not ctx.json:
        ctx.out.print(f"[bold]{spec.name}[/bold] → {plan.destination}")
        ctx.out.print(f"  download size : {data['total_gb']} GB "
                      f"({data['already_staged_bytes'] / 1024**3:.2f} GB already staged)")
        ctx.out.print(f"  free disk     : {plan.free_bytes / 1024**3:.1f} GB")
        ctx.out.print(f"  license       : {spec.license}  <{spec.license_url}>")
        ctx.out.print(f"  commercial use: {spec.commercial_use}"
                      + ("  · gated on Hugging Face" if plan.gated else ""))
        for s in data["sources"]:
            ctx.out.print(f"  source        : {s['repo_id']}@{s['revision'][:12]} "
                          f"({s['files']} files)")
        if plan.skipped_code_files:
            ctx.warn(f"skipping {len(plan.skipped_code_files)} Python file(s) from the repo "
                     "(use --trust-repo-code to fetch them)")
        if plan.pickle_files:
            ctx.warn(f"{len(plan.pickle_files)} pickle-format checkpoint file(s); they are only "
                     "loaded by the backend's own code")
        for note in spec.notes:
            ctx.out.print(f"  [dim]note: {note}[/dim]")
    if plan_only:
        return 0
    accepted = accept_license
    if spec.requires_license_acceptance and not accepted:
        accepted = ctx.confirm(f"Accept the {spec.license} terms for {spec.id}?",
                               assume_yes=False)
        if not accepted:
            raise LicenseNotAcceptedError(f"license not accepted for {spec.id}",
                                          hint="re-run with --accept-license after reviewing "
                                          + spec.license_url)
    if not ctx.confirm(f"Download {data['total_gb']} GB to {plan.destination}?",
                       assume_yes=yes):
        raise InvalidRequestError("download not confirmed (pass --yes for scripts)")
    manifest = execute_plan(plan, hub=hub, token=cfg.hf_token(), license_accepted=accepted,
                            trust_repo_code=trust_code,
                            max_workers=int(cfg["models"]["max_download_workers"]),
                            quiet=ctx.quiet or ctx.json, progress=ctx.info)
    from prompt2music import envs

    ready = envs.probe(cfg, spec.backend).ok
    payload = {"model": spec.id, "path": str(plan.destination),
               "bytes": manifest.total_bytes, "revisions": manifest.revisions,
               "backend": spec.backend, "backend_ready": ready}
    if ctx.json:
        ctx.emit(payload)
    else:
        ctx.ok(f"installed {spec.id} ({manifest.total_bytes / 1024**3:.2f} GB, verified)")
        if not ready:
            ctx.out.print(f"  next: prompt2music backends install {spec.backend}")
    return 0


def cmd_models(ctx: Ctx, args: argparse.Namespace) -> int:
    from prompt2music.manager import ModelManager
    from prompt2music.registry import all_models, get_model, search_models

    cfg = ctx.cfg
    mgr = ModelManager(cfg)
    action = args.models_cmd or "list"
    if action == "list":
        items = [m.as_dict() for m in mgr.list_installed()]
        if ctx.json:
            ctx.emit(items)
        elif not items:
            ctx.out.print("No models installed. See: prompt2music models available")
        else:
            rows = [(m["id"], f"{m['size_gb']} GB", m["installed_at"][:16].replace("T", " "),
                     ", ".join(f"{r.split('/')[-1]}@{s[:10]}" for r, s in m["revisions"].items()))
                    for m in items]
            _print_table(ctx, f"Installed models ({mgr.root})",
                         ("id", "size", "installed", "revision"), rows)
        return 0
    if action == "available":
        from prompt2music.backends.base import Task

        specs = all_models(include_diagnostic=True, kind=args.kind)
        if args.task:
            specs = [s for s in specs if Task(args.task) in s.tasks]
        if ctx.json:
            ctx.emit([s.as_dict() for s in specs])
        else:
            _print_models(ctx, "Supported models", specs)
            ctx.out.print("[dim]Non-commercial and gated weights are marked; read each "
                          "license before use. `models info ID` for details.[/dim]")
        return 0
    if action == "search":
        specs = search_models(args.query)
        if ctx.json:
            ctx.emit([s.as_dict() for s in specs])
        else:
            _print_models(ctx, f"Search: {args.query}", specs)
        return 0
    if action == "info":
        spec = get_model(args.model)
        data = spec.as_dict()
        inst = mgr.installed(spec.id) if spec.needs_download else None
        data["installed"] = inst.as_dict() if inst else None
        if ctx.json:
            ctx.emit(data)
        else:
            ctx.out.print_json(json.dumps(data, default=str))
        return 0
    if action == "download":
        return _download(ctx, args.model, revision=args.revision,
                         accept_license=args.accept_license, yes=args.yes,
                         trust_code=args.trust_repo_code, plan_only=args.plan)
    if action == "update":
        from prompt2music.downloader import HfHub, plan_download

        spec = get_model(args.model)
        inst = mgr.installed(spec.id)
        if inst is None:
            raise MusicError(f"{spec.id} is not installed")
        plan = plan_download(spec, cfg.models_dir, hub=HfHub(cfg["models"]["hf_endpoint"] or None),
                             token=cfg.hf_token(), offline=cfg.offline)
        latest = {ps.source.repo_id: ps.revision_resolved for ps in plan.sources}
        if latest == inst.manifest.revisions:
            ctx.ok(f"{spec.id} is up to date")
            return 0
        ctx.info(f"update available: {inst.manifest.revisions} → {latest}")
        accepted = args.accept_license or bool(inst.manifest.license_accepted_at)
        return _download(ctx, spec.id, accept_license=accepted, yes=args.yes)
    if action == "verify":
        report = mgr.verify(args.model, full=not args.quick)
        if ctx.json:
            ctx.emit(report.as_dict())
        elif report.ok:
            ctx.ok(f"{report.model_id}: {report.checked} files OK "
                   f"({'sha256' if report.full else 'size'} check)")
        else:
            for problem in report.problems:
                ctx.err.print(f"[red]✖[/red] {problem}")
        return 0 if report.ok else 1
    if action == "import":
        manifest = mgr.import_local(args.model, Path(args.path), link=args.link,
                                    license_accepted=args.accept_license)
        ctx.ok(f"imported {manifest.model_id} ({len(manifest.files)} files, "
               f"{manifest.total_bytes / 1024**3:.2f} GB)")
        return 0
    if action == "remove":
        spec = get_model(args.model)
        if not ctx.confirm(f"Delete {spec.id} from {mgr.root}?", assume_yes=args.yes):
            raise InvalidRequestError("removal not confirmed (pass --yes)")
        freed = mgr.remove(spec.id)
        ctx.ok(f"removed {spec.id} ({freed / 1024**3:.2f} GB freed)")
        return 0
    if action == "cache":
        usage = mgr.cache_report()
        if args.clean:
            usage["cleaned_bytes"] = mgr.clean_partial()
        if ctx.json:
            ctx.emit(usage)
        else:
            ctx.out.print(f"models dir : {usage['models_dir']}")
            ctx.out.print(f"installed  : {usage['installed_bytes'] / 1024**3:.2f} GB "
                          f"in {len(usage['installed'])} model(s)")
            for name, size in usage["partial_downloads"].items():
                ctx.out.print(f"partial    : {name} {size / 1024**3:.2f} GB")
            ctx.out.print(f"free disk  : {usage['free_bytes'] / 1024**3:.1f} GB")
            if args.clean:
                ctx.ok(f"cleaned {usage['cleaned_bytes'] / 1024**3:.2f} GB")
        return 0
    if action == "load":
        from prompt2music.gpu import parse_size_gb

        spec = get_model(args.model)
        mgr.require(spec.id)
        params = {"model": spec.id, "device": args.device, "precision": args.precision,
                  "vram_limit_gb": parse_size_gb(args.vram_limit) if args.vram_limit else None}
        from prompt2music import daemon

        daemon.ensure_running(cfg)  # the daemon owns warm models
        result = _run_job(ctx, "load", params, model=spec.id, prompt=None, background=False,
                          no_daemon=False)
        if ctx.json:
            ctx.emit(result)
        else:
            ctx.ok(f"{spec.id} loaded in the job daemon on "
                   f"{(result or {}).get('device', '?')} — consecutive jobs reuse it")
        return 0
    if action == "unload":
        if not _daemon_running():
            ctx.ok("daemon is not running; nothing is loaded")
            return 0
        _run_job(ctx, "unload", {}, model=None, prompt=None, background=False, no_daemon=False)
        ctx.ok("daemon model unloaded")
        return 0
    if action == "benchmark":
        args.models = [args.model]
        args.all_installed = False
        args.prompt = None
        args.device = None
        args.precision = None
        return cmd_benchmark(ctx, args)
    raise InvalidRequestError(f"unknown models action {action}")


def _daemon_running() -> bool:
    from prompt2music import daemon

    return bool(daemon.status()["running"])


def cmd_backends(ctx: Ctx, args: argparse.Namespace) -> int:
    from prompt2music import envs

    cfg = ctx.cfg
    action = args.backends_cmd or "list"
    if action == "list":
        rows = {name: envs.probe(cfg, name).as_dict() for name in envs.BACKEND_ENVS}
        if ctx.json:
            ctx.emit(rows)
            return 0
        table = Table(title="Backend environments")
        for col in ("backend", "ready", "python", "missing", "upstream"):
            table.add_column(col)
        for name, row in rows.items():
            spec = envs.BACKEND_ENVS[name]
            table.add_row(name, "[green]yes[/green]" if row["ok"] else "[red]no[/red]",
                          f"{row['python']} ({row['source']})", ", ".join(row["missing"]),
                          spec.git_url or ", ".join(spec.packages) or "-")
        ctx.out.print(table)
        return 0
    if action == "install":
        spec = envs.env_spec(args.backend)
        cmds = envs.install_commands(cfg, args.backend, python=args.python, ref=args.ref,
                                     torch_index=args.torch_index)
        if args.print_only or not ctx.json:
            ctx.out.print(f"[bold]{args.backend}[/bold] → {envs.env_dir(cfg, args.backend)}")
            for c in cmds:
                ctx.out.print("  $ " + " ".join(x for x in c if x != "#best-effort"))
            for note in spec.notes:
                ctx.out.print(f"  [dim]note: {note}[/dim]")
        if args.print_only:
            return 0
        if not ctx.confirm("Run these commands?", assume_yes=args.yes):
            raise InvalidRequestError("not confirmed (pass --yes)")
        result = envs.run_install(cfg, args.backend, cmds, log=ctx.info)
        rev = envs.installed_revision(cfg, args.backend)
        if ctx.json:
            ctx.emit({**result.as_dict(), "upstream_revision": rev})
        elif result.ok:
            ctx.ok(f"{args.backend} backend ready ({result.python})"
                   + (f" at upstream {rev[:12]}" if rev else ""))
        else:
            ctx.err.print(f"[red]✖[/red] still missing: {', '.join(result.missing)}")
        return 0 if result.ok else 2
    raise InvalidRequestError(f"unknown backends action {action}")


def cmd_stems(ctx: Ctx, args: argparse.Namespace) -> int:
    from prompt2music.registry import all_models

    action = args.stems_cmd or "models"
    if action == "models":
        specs = all_models(kind="stems")
        if ctx.json:
            ctx.emit([s.as_dict() for s in specs])
        else:
            table = Table(title="Stem separation models (Demucs)")
            for col in ("id", "stems", "size", "license", "installed"):
                table.add_column(col)
            from prompt2music.manager import ModelManager

            mgr = ModelManager(ctx.cfg)
            for s in specs:
                table.add_row(s.id, s.description.replace("Source separation into ", ""),
                              f"~{s.approx_size_gb:g} GB", s.license,
                              "[green]yes[/green]" if mgr.installed(s.id) else "")
            ctx.out.print(table)
        return 0
    if action == "download":
        return _download(ctx, args.model, yes=args.yes, accept_license=True)
    if action == "separate":
        params = {"model": args.model, "input_audio": str(Path(args.input).expanduser()
                                                            .resolve()),
                  "output_dir": args.output_dir, "format": args.format,
                  "stems": args.stems.split(",") if args.stems else None,
                  "device": args.device, "overwrite": args.overwrite}
        result = _run_job(ctx, "separate", params, model=args.model, prompt=None,
                          background=args.background, no_daemon=args.no_daemon)
        if result is None:
            return 0
        if ctx.json:
            ctx.emit(result)
        else:
            ctx.ok(f"separated with {result['model']} into {result['output_dir']}")
            for name, path in result["stems"].items():
                ctx.out.print(f"  {name:8} {path}")
        return 0
    raise InvalidRequestError(f"unknown stems action {action}")


# ---------------------------------------------------------------------------
# audio toolkit
# ---------------------------------------------------------------------------
def _tags(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise InvalidRequestError(f"metadata must be KEY=VALUE, got {item!r}")
        out[key] = value
    return out


def cmd_audio(ctx: Ctx, args: argparse.Namespace) -> int:
    from prompt2music import audio as A

    action = args.audio_cmd
    if action is None:
        raise InvalidRequestError("usage: prompt2music audio ACTION ...")
    src = Path(getattr(args, "input", None) or getattr(args, "file", None) or ".").expanduser()

    def out_path(label: str, default_ext: str | None = None) -> Path:
        if getattr(args, "output", None):
            return Path(args.output).expanduser()
        if default_ext:
            return src.with_name(f"{src.stem}-{label}{default_ext}")
        return A.derived_path(src, label)

    force = bool(getattr(args, "force", False))
    result: dict[str, Any]
    if action == "info":
        result = A.info(src).as_dict()
        result["levels"] = A.levels(src)
        if args.loudness:
            result["loudness"] = A.loudness(src)
    elif action == "validate":
        result = A.validate_audio(src, expected_duration=args.expect_duration)
    elif action == "convert":
        dest = A.convert(src, Path(args.output).expanduser(), sample_rate=args.sample_rate,
                         channels=args.channels, bitrate=args.bitrate,
                         metadata=_tags(args.tag), overwrite=force)
        result = {"output": str(dest)}
    elif action == "normalize":
        dest = out_path("normalized")
        result = {"output": str(dest), **A.normalize(src, dest, mode=args.mode,
                                                      target=args.target, overwrite=force)}
    elif action == "trim":
        result = {"output": str(A.trim(src, out_path("trim"), start=args.start, end=args.end,
                                       overwrite=force))}
    elif action == "fade":
        result = {"output": str(A.fade(src, out_path("fade"), fade_in=args.fade_in,
                                       fade_out=args.fade_out, overwrite=force))}
    elif action == "loop":
        result = {"output": str(A.loop(src, out_path(f"x{args.count}"), count=args.count,
                                       crossfade=args.crossfade, overwrite=force))}
    elif action == "mix":
        inputs = [Path(p).expanduser() for p in args.inputs]
        dest = Path(args.output).expanduser() if args.output else inputs[0].with_name(
            f"{inputs[0].stem}-mix{inputs[0].suffix}")
        if args.voice:
            if len(inputs) != 1:
                raise InvalidRequestError("with --voice give exactly one music input")
            if args.ducking:
                A.duck(Path(args.voice).expanduser(), inputs[0], dest, overwrite=force)
            else:
                A.mix([Path(args.voice).expanduser(), inputs[0]], dest, duration="first",
                      overwrite=force)
        else:
            weights = [float(w) for w in args.weights.split(",")] if args.weights else None
            A.mix(inputs, dest, weights=weights, duration=args.duration, overwrite=force)
        result = {"output": str(dest)}
    elif action == "crossfade":
        dest = A.crossfade(Path(args.first).expanduser(), Path(args.second).expanduser(),
                           Path(args.output).expanduser(), seconds=args.seconds,
                           overwrite=force)
        result = {"output": str(dest)}
    elif action == "fit":
        result = {"output": str(A.fit_duration(src, out_path(f"{args.seconds:g}s"),
                                               seconds=args.seconds, overwrite=force))}
    elif action in ("waveform", "spectrogram"):
        fn = A.waveform if action == "waveform" else A.spectrogram
        result = {"output": str(fn(src, out_path(action, ".png"), overwrite=force))}
    elif action == "tag":
        A.embed_metadata(src, _tags(args.tags))
        result = {"output": str(src), "tags": A.info(src).tags}
    else:
        raise InvalidRequestError(f"unknown audio action {action}")
    if ctx.json:
        ctx.emit(result)
    elif "output" in result and len(result) <= 2:
        ctx.ok(f"wrote {result['output']}")
    else:
        ctx.out.print_json(json.dumps(result, default=str))
    return 0


# ---------------------------------------------------------------------------
# jobs / daemon / benchmark / library / config / mcp
# ---------------------------------------------------------------------------
def cmd_jobs(ctx: Ctx, args: argparse.Namespace) -> int:
    from prompt2music import daemon
    from prompt2music.jobs import JobStore

    store = JobStore()
    action = args.jobs_cmd or "list"
    if action == "list":
        jobs = store.list(status=getattr(args, "status", None), limit=getattr(args, "limit", 20))
        if ctx.json:
            ctx.emit([j.as_dict(brief=True) for j in jobs])
            return 0
        rows = [(j.id, j.kind, j.status,
                 "" if j.progress is None else f"{j.progress * 100:.0f}%", j.model or "",
                 j.created_at[5:16].replace("T", " "), j.prompt or "") for j in jobs]
        _print_table(ctx, "Jobs", ("id", "kind", "status", "done", "model", "created",
                                   "prompt"), rows, optional=("kind", "created", "done"))
        return 0
    if action == "status":
        job = store.get(args.job_id)
        data = job.as_dict()
        if ctx.json:
            ctx.emit(data)
        else:
            data.pop("params", None)
            ctx.out.print_json(json.dumps(data, default=str))
        return 0
    if action == "cancel":
        job = store.request_cancel(args.job_id)
        if ctx.json:
            ctx.emit(job.as_dict(brief=True))
        else:
            ctx.ok(f"{job.id}: {job.status}"
                   + (" (cancellation requested)" if not job.final else ""))
        return 0
    if action in ("result", "wait"):
        job = store.get(args.job_id)
        if action == "wait":
            progress_ui = _Progress(ctx)
            with progress_ui:
                job = daemon.follow(store, job.id, on_update=progress_ui.job_update)
        if job.status == "completed":
            if job.kind == "generate" and job.result:
                _render_result(ctx, job.result)
            elif ctx.json:
                ctx.emit(job.result)
            else:
                ctx.out.print_json(json.dumps(job.result, default=str))
            return 0
        if job.final:
            raise _error_from_job(job) if job.status == "failed" else CancelledError(
                f"job {job.id} was cancelled")
        ctx.out.print(f"{job.id} is {job.status} ({(job.progress or 0) * 100:.0f}%)")
        return 3
    if action == "logs":
        path = store.log_path(args.job_id)
        if not path.exists():
            ctx.out.print("(no log yet)")
            return 0
        if not args.follow:
            print(path.read_text(encoding="utf-8", errors="replace"), end="")
            return 0
        import time

        with path.open(encoding="utf-8", errors="replace") as fh:
            while True:
                line = fh.readline()
                if line:
                    print(line, end="")
                    continue
                if store.get(args.job_id).final:
                    return 0
                time.sleep(0.5)
    if action == "prune":
        ctx.ok(f"pruned {store.prune(keep=args.keep)} finished job(s)")
        return 0
    raise InvalidRequestError(f"unknown jobs action {action}")


def cmd_daemon(ctx: Ctx, args: argparse.Namespace) -> int:
    from prompt2music import daemon

    action = args.daemon_cmd or "status"
    if action == "run":
        return daemon.Daemon(ctx.cfg, idle_exit=args.idle_exit).run()
    if action == "start":
        result = daemon.start(ctx.cfg)
    elif action == "stop":
        result = daemon.stop()
    elif action == "status":
        result = daemon.status()
    elif action == "install-service":
        import shutil

        exe = shutil.which("prompt2music") or f"{sys.executable} -m prompt2music.cli"
        unit = Path.home() / ".config" / "systemd" / "user" / "prompt2music.service"
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text(daemon.systemd_unit(exe), encoding="utf-8")
        result = {"unit": str(unit),
                  "next": "systemctl --user daemon-reload && systemctl --user enable --now "
                          "prompt2music.service"}
    else:
        raise InvalidRequestError(f"unknown daemon action {action}")
    if ctx.json:
        ctx.emit(result)
    else:
        ctx.out.print_json(json.dumps(result, default=str))
    return 0


def cmd_benchmark(ctx: Ctx, args: argparse.Namespace) -> int:
    params = {"models": args.models, "all_installed": args.all_installed,
              "duration": args.duration, "runs": args.runs, "prompt": args.prompt,
              "device": args.device, "precision": args.precision,
              "output": str(Path(args.output).expanduser().resolve()) if args.output else None}
    report = _run_job(ctx, "benchmark", params, model=",".join(args.models or ["all"]),
                      prompt=None, background=False, no_daemon=True)
    assert report is not None
    if ctx.json:
        ctx.emit(report)
        return 0
    table = Table(title="Benchmark (measured)")
    for col in ("model", "device", "load s", "RTF mean", "peak VRAM MB", "peak RAM MB",
                "rate", "ch"):
        table.add_column(col)
    for r in report["results"]:
        table.add_row(r["model"], str(r["config"]["device"]), str(r["load_seconds_wall"]),
                      str(r["rtf_mean"]), str(r["peak_vram_mb"] or "-"), str(r["peak_ram_mb"]),
                      str(r["sample_rate"]), str(r["channels"]))
    ctx.out.print(table)
    ctx.out.print("RTF = generation_seconds / audio_seconds (<1 is faster than real time)")
    for model, err in report["errors"].items():
        ctx.warn(f"{model}: {err['message']}")
    if report.get("output"):
        ctx.ok(f"saved {report['output']}")
    return 0


def cmd_library(ctx: Ctx, args: argparse.Namespace) -> int:
    from prompt2music.library import Library

    lib = Library()
    action = args.library_cmd or "list"
    if action in ("list", "search"):
        rows = lib.list(limit=args.limit) if action == "list" else lib.search(args.query)
        if ctx.json:
            ctx.emit(rows)
            return 0
        cells = [(r["id"], r["created_at"][5:16].replace("T", " "), r["model"] or "",
                  f"{r['duration']:.1f}s" if r["duration"] else "", str(r["seed"]),
                  ("" if r["exists"] else "(missing) ") + Path(r["path"]).name) for r in rows]
        _print_table(ctx, "Library", ("id", "created", "model", "dur", "seed", "file"), cells,
                     optional=("seed", "created"))
        return 0
    if action == "info":
        data = lib.get(args.track_id)
        if ctx.json:
            ctx.emit(data)
        else:
            ctx.out.print_json(json.dumps(data, default=str))
        return 0
    if action == "export":
        data = lib.export(args.track_id, Path(args.output).expanduser(), fmt=args.format,
                          overwrite=args.force)
        if ctx.json:
            ctx.emit(data)
        else:
            ctx.ok(f"exported to {data['exported']}")
        return 0
    if action == "reproduce":
        from prompt2music import service

        params = lib.reproduce_params(args.track_id)
        params.pop("output", None)
        if args.output:
            params["output"] = args.output
        params = service.normalise_params(ctx.cfg, params)
        result = _run_job(ctx, "generate", params, model=params["model"],
                          prompt=params["prompt"], background=False, no_daemon=False)
        if result:
            _render_result(ctx, result)
            if not ctx.json:
                ctx.out.print("[dim]Same model revision, seed and parameters; output may still "
                              "differ across GPUs, drivers or library versions.[/dim]")
        return 0
    if action == "remove":
        data = lib.remove(args.track_id, delete_files=args.delete_files)
        if ctx.json:
            ctx.emit(data)
        else:
            ctx.ok(f"removed {data['removed']}")
        return 0
    raise InvalidRequestError(f"unknown library action {action}")


def cmd_config(ctx: Ctx, args: argparse.Namespace) -> int:
    from prompt2music.config import reset_config

    action = args.config_cmd or "show"
    cfg = ctx.cfg
    if action == "path":
        print(cfg.path)
    elif action == "show":
        if ctx.json:
            ctx.emit(cfg.as_dict())
        elif cfg.path.exists():
            from rich.syntax import Syntax

            ctx.out.print(Syntax(cfg.path.read_text(encoding="utf-8"), "toml"))
        else:
            ctx.out.print_json(json.dumps(cfg.as_dict()))
    elif action == "get":
        value = cfg.get(args.key)
        print(json.dumps(value) if ctx.json or not isinstance(value, str) else value)
    elif action == "set":
        section, _, name = args.key.partition(".")
        value = parse_cli_value(section, name, args.value)
        cfg.set(args.key, value)
        cfg.save()
        ctx.ok(f"{args.key} = {json.dumps(value)}  ({cfg.path})")
    elif action == "reset":
        if not ctx.confirm(f"Reset {cfg.path} to defaults (a .bak copy is kept)?",
                           assume_yes=args.yes):
            raise InvalidRequestError("not confirmed (pass --yes)")
        ctx.ok(f"reset {reset_config(cfg.path)}")
    else:
        raise InvalidRequestError(f"unknown config action {action}")
    return 0


def cmd_mcp(ctx: Ctx, args: argparse.Namespace) -> int:
    action = args.mcp_cmd or "serve"
    if action == "config":
        from prompt2music.mcp.server import client_config

        print(client_config(args.client))
        return 0
    if action == "tools":
        from prompt2music.mcp.tools import TOOL_SPECS

        if ctx.json:
            ctx.emit({name: spec["description"] for name, spec in TOOL_SPECS.items()})
        else:
            for name, spec in TOOL_SPECS.items():
                ctx.out.print(f"[bold]{name}[/bold]  {spec['description']}")
        return 0
    from prompt2music.mcp.server import serve

    return serve(ctx.cfg, transport=getattr(args, "transport", None),
                 host=getattr(args, "host", None), port=getattr(args, "port", None))


def cmd_doctor(ctx: Ctx, args: argparse.Namespace) -> int:
    from prompt2music.doctor import collect, render

    report = collect(ctx.cfg, quick=args.quick)
    if ctx.json:
        ctx.emit(report)
    else:
        render(report, ctx.out)
    return 0


def cmd_setup(ctx: Ctx, args: argparse.Namespace) -> int:
    from prompt2music.setup_wizard import run_setup

    return run_setup(ctx, args)


def cmd_completion(ctx: Ctx, args: argparse.Namespace) -> int:
    from prompt2music.completion import render_completion

    print(render_completion(build_parser(), args.shell))
    return 0


COMMANDS: dict[str, Handler] = {
    "doctor": cmd_doctor, "setup": cmd_setup, "generate": cmd_generate, "song": cmd_song,
    "continue": cmd_continue, "edit": cmd_edit, "inpaint": cmd_inpaint,
    "soundtrack": cmd_soundtrack, "enhance": cmd_enhance, "presets": cmd_presets,
    "lyrics": cmd_lyrics, "models": cmd_models, "backends": cmd_backends, "stems": cmd_stems,
    "audio": cmd_audio, "jobs": cmd_jobs, "daemon": cmd_daemon, "benchmark": cmd_benchmark,
    "library": cmd_library, "config": cmd_config, "mcp": cmd_mcp, "completion": cmd_completion,
}


def run(argv: list[str] | None = None, *, prog: str = "prompt2music") -> int:
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0
    from prompt2music._shared.logging import setup_logging

    setup_logging(quiet=args.quiet, verbose=args.verbose, debug=args.debug)
    ctx = Ctx(args)
    try:
        return COMMANDS[args.cmd](ctx, args)
    except MusicError as exc:
        if args.debug:
            raise
        if ctx.json:
            print(json.dumps({"error": exc.as_dict()}), file=sys.stderr)
        else:
            ctx.err.print(f"[red]✖[/red] {escape(str(exc))}")
            if exc.hint:
                ctx.err.print(f"  [dim]hint: {escape(exc.hint)}[/dim]")
        return exc.exit_code


@interruptible_cli
def main(argv: list[str] | None = None, prog: str = "prompt2music") -> None:
    install_sigint_handler()
    sys.exit(run(argv, prog=prog))


def main_aimusic(argv: list[str] | None = None) -> None:
    main(argv, prog="aimusic")


if __name__ == "__main__":
    main()
