"""``prompt2music doctor``: hardware, tools, backend envs, models, readiness."""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Any

from prompt2music import __version__, daemon, envs
from prompt2music._shared.hardware import probe_hardware
from prompt2music._shared.nixos import is_nixos
from prompt2music.config import MusicConfig
from prompt2music.envs import BACKEND_ENVS
from prompt2music.gpu import compute_capability, driver_info
from prompt2music.manager import ModelManager
from prompt2music.registry import all_models


def _version(argv: list[str]) -> str | None:
    exe = shutil.which(argv[0])
    if not exe:
        return None
    try:
        out = subprocess.run([exe, *argv[1:]], capture_output=True, text=True, timeout=10,
                             check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "installed"
    return out.strip().splitlines()[0] if out.strip() else "installed"


def _ram_available_mb() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def collect(cfg: MusicConfig, *, quick: bool = False) -> dict[str, Any]:
    hw = probe_hardware()
    gpus = [g.as_dict() for g in hw.gpus]
    gpu_extra: dict[str, Any] = driver_info() if hw.gpus else {}
    precision = "fp32"
    device = "cpu"
    if hw.gpus:
        cap = compute_capability(0)
        gpu_extra["compute_capability"] = cap
        precision = "bf16" if cap is None or cap >= 8.0 else "fp16"
        device = "cuda:0"
    elif sys.platform == "darwin":
        device, precision = "mps", "fp16"
    manager = ModelManager(cfg)
    models_dir = manager.root
    probe_root = models_dir if models_dir.exists() else models_dir.parent
    while not probe_root.exists():
        probe_root = probe_root.parent
    disk_free = shutil.disk_usage(probe_root).free
    tools = {name: _version(argv) for name, argv in {
        "ffmpeg": ["ffmpeg", "-version"], "ffprobe": ["ffprobe", "-version"],
        "uv": ["uv", "--version"], "git": ["git", "--version"],
        "nvidia-smi": ["nvidia-smi", "--version"],
    }.items()}
    backends: dict[str, Any] = {}
    torch_cuda = None
    if not quick:
        for name in BACKEND_ENVS:
            res = envs.probe(cfg, name)
            backends[name] = res.as_dict()
            if res.ok and res.torch.get("cuda"):
                torch_cuda = True
            elif res.ok and res.torch and torch_cuda is None:
                torch_cuda = False
    installed = []
    for inst in manager.list_installed():
        installed.append({"id": inst.spec.id, "name": inst.spec.name,
                          "size_gb": round(inst.manifest.total_bytes / 1024**3, 2),
                          "backend": inst.spec.backend,
                          "backend_ready": backends.get(inst.spec.backend, {}).get("ok")})
    issues = []
    if not tools["ffmpeg"] or not tools["ffprobe"]:
        issues.append("ffmpeg/ffprobe missing (required for audio I/O)")
    usable = [m for m in installed if m["backend_ready"] or quick]
    if not any(m for m in usable if m["id"] not in ("test-tone",)):
        issues.append("no music model is installed with a ready backend")
    status = "Ready" if not issues else "Not ready"
    return {
        "version": __version__,
        "system": {"os": hw.system, "machine": hw.machine, "python": hw.python,
                   "nixos": is_nixos(), "cpus": hw.cpu_count, "ram_total_mb": hw.ram_total_mb,
                   "ram_available_mb": _ram_available_mb()},
        "gpus": gpus,
        "gpu_driver": gpu_extra,
        "torch_cuda_in_backend_envs": torch_cuda,
        "recommended": {"device": device, "precision": precision},
        "tools": tools,
        "paths": {"config": str(cfg.path), "models": str(models_dir), "envs": str(cfg.envs_dir),
                  "output": str(cfg.output_dir), "disk_free_gb": round(disk_free / 1024**3, 1)},
        "backends": backends,
        "installed_models": installed,
        "catalog_size": len(all_models(include_diagnostic=False)),
        "daemon": daemon.status(),
        "status": status,
        "issues": issues,
    }


def render(report: dict[str, Any], console: Any) -> None:
    from rich.table import Table

    console.print("[bold cyan]prompt2music System Diagnostics[/bold cyan] "
                  f"[dim]v{report['version']}[/dim]\n")
    if report["gpus"]:
        for g in report["gpus"]:
            console.print(f"GPU: {g['name']}")
            console.print(f"VRAM: {g['memory_total_mb'] / 1024:.0f} GB "
                          f"({g['memory_free_mb'] / 1024:.1f} GB free)")
        drv = report["gpu_driver"]
        if drv:
            console.print(f"Driver: {drv.get('driver_version', '?')}  "
                          f"(CUDA {drv.get('driver_cuda_version', '?')}, "
                          f"compute {drv.get('compute_capability', '?')})")
        console.print("CUDA: Available (nvidia-smi)")
    else:
        console.print("GPU: none detected (CPU-capable models only)")
    tc = report["torch_cuda_in_backend_envs"]
    console.print("PyTorch: " + ("CUDA enabled" if tc else "CPU only" if tc is False
                                 else "not checked / no backend env"))
    console.print(f"FFmpeg: {'Installed' if report['tools']['ffmpeg'] else 'MISSING'}")
    sysinfo = report["system"]
    console.print(f"RAM: {sysinfo['ram_total_mb'] or '?'} MB total, "
                  f"{sysinfo['ram_available_mb'] or '?'} MB available"
                  + ("  (NixOS)" if sysinfo["nixos"] else ""))
    console.print(f"Disk free (models): {report['paths']['disk_free_gb']} GB\n")
    rec = report["recommended"]
    console.print(f"Recommended Precision: {rec['precision'].upper()}")
    console.print(f"Recommended Device: {rec['device']}\n")
    if report["backends"]:
        table = Table(title="Backend environments", show_lines=False)
        for col in ("backend", "ready", "interpreter", "torch", "missing"):
            table.add_column(col)
        for name, b in report["backends"].items():
            torch = b["torch"]
            tdesc = (f"{torch.get('version')} cuda={torch.get('cuda')}" if torch.get("version")
                     else "-")
            table.add_row(name, "[green]yes[/green]" if b["ok"] else "[red]no[/red]",
                          b["source"], tdesc, ", ".join(b["missing"]))
        console.print(table)
    console.print("Installed Models:")
    if report["installed_models"]:
        for m in report["installed_models"]:
            ready = m["backend_ready"]
            mark = "" if ready is None else (" [green]✓[/green]" if ready
                                             else " [yellow](backend env missing)[/yellow]")
            console.print(f"  {m['name']}  [dim]{m['size_gb']} GB[/dim]{mark}")
    else:
        console.print("  (none) — try: prompt2music models available")
    d = report["daemon"]
    state = f"running (pid {d.get('pid')})" if d["running"] else "stopped"
    console.print(f"\nJob daemon: {state}")
    colour = "green" if report["status"] == "Ready" else "yellow"
    console.print(f"\nStatus: [{colour}]{report['status']}[/{colour}]")
    for issue in report["issues"]:
        console.print(f"  - {issue}")
