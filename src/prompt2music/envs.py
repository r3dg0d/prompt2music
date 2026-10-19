"""Isolated Python environments for model backends.

Upstream projects pin incompatible stacks (ACE-Step 1.5 wants torch 2.10 with
transformers<4.58; this repo's Qwen env needs transformers>=5.17), so each
backend runs in its own venv under ``~/.local/share/ai-media/envs/music-<name>``
and talks to the CLI through :mod:`prompt2music.worker`.

The prompt2music package is exposed to those venvs through a one-entry
PYTHONPATH shim (a symlink to this package) instead of installing it there,
so the worker code always matches the CLI that launched it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompt2music import paths
from prompt2music._shared.nixos import nixos_ld_library_path
from prompt2music.config import MusicConfig
from prompt2music.errors import BackendNotReadyError, ToolMissingError


@dataclass(frozen=True)
class BackendEnvSpec:
    name: str
    modules: tuple[str, ...]  # must import for the backend to work
    python_version: str
    git_url: str | None = None
    git_ref: str = "main"
    # "pip": install `packages` with uv pip; "uv-sync": sync the cloned project
    method: str = "pip"
    packages: tuple[str, ...] = ()
    extra_steps: tuple[tuple[str, ...], ...] = ()
    best_effort_packages: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    torch_index: bool = True


BACKEND_ENVS: dict[str, BackendEnvSpec] = {
    "tone": BackendEnvSpec(name="tone", modules=(), python_version="3.12", torch_index=False),
    "musicgen": BackendEnvSpec(
        name="musicgen",
        modules=("torch", "transformers", "numpy"),
        python_version="3.12",
        packages=("torch", "transformers>=4.45,<6", "numpy", "accelerate", "sentencepiece"),
    ),
    "ace-step": BackendEnvSpec(
        name="ace-step",
        modules=("torch", "acestep"),
        python_version="3.12",
        git_url="https://github.com/ace-step/ACE-Step-1.5.git",
        method="uv-sync",
        notes=("Uses upstream's uv lockfile: torch 2.10 + CUDA 12.8 on Linux x86_64.",),
        torch_index=False,
    ),
    "stable-audio": BackendEnvSpec(
        name="stable-audio",
        modules=("torch", "stable_audio_3"),
        python_version="3.12",
        git_url="https://github.com/Stability-AI/stable-audio-3.git",
        method="uv-sync",
        best_effort_packages=("flash-attn",),
        notes=("Medium needs flash-attn 2; its build is attempted but may fail on NixOS.",
               "Upstream defaults to the CUDA 12.6 torch build."),
        torch_index=False,
    ),
    "heartmula": BackendEnvSpec(
        name="heartmula",
        modules=("torch", "heartlib"),
        python_version="3.10",
        git_url="https://github.com/HeartMuLa/heartlib.git",
        method="pip-editable",
        notes=("Upstream recommends Python 3.10.",),
    ),
    "yue": BackendEnvSpec(
        name="yue",
        modules=("torch", "yue2"),
        python_version="3.12",
        git_url="https://github.com/multimodal-art-projection/YuE.git",
        method="pip-project",
        notes=("Upstream: Linux, Python 3.12, BF16 GPU with 24 GB VRAM.",),
    ),
    "demucs": BackendEnvSpec(
        name="demucs",
        modules=("torch", "demucs", "numpy"),
        python_version="3.12",
        packages=("torch", "demucs>=4.1", "numpy"),
    ),
}


def env_spec(backend: str) -> BackendEnvSpec:
    try:
        return BACKEND_ENVS[backend]
    except KeyError as exc:
        raise BackendNotReadyError(f"unknown backend {backend!r}") from exc


def env_dir(cfg: MusicConfig, backend: str) -> Path:
    return cfg.envs_dir / f"music-{backend}"


def src_dir(cfg: MusicConfig, backend: str) -> Path:
    return cfg.envs_dir / "src" / f"music-{backend}"


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def python_for(cfg: MusicConfig, backend: str) -> tuple[Path, str]:
    """Interpreter for ``backend`` and where the choice came from."""
    override = cfg["backends"]["python"].get(backend)
    if override:
        return paths.expand(override), "config backends.python"
    env_key = "PROMPT2MUSIC_PYTHON_" + backend.upper().replace("-", "_")
    if os.environ.get(env_key):
        return paths.expand(os.environ[env_key]), env_key
    venv_py = _venv_python(env_dir(cfg, backend))
    if venv_py.exists():
        return venv_py, "managed venv"
    return Path(sys.executable), "current interpreter"


def pythonpath_shim() -> Path:
    """Directory containing only a ``prompt2music`` link to this package."""
    import prompt2music

    pkg = Path(prompt2music.__file__).resolve().parent
    shim = paths.cache_dir() / "pythonpath"
    link = shim / "prompt2music"
    shim.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() == pkg:
        return shim
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        shutil.rmtree(link)
    try:
        link.symlink_to(pkg, target_is_directory=True)
    except OSError:  # e.g. Windows without symlink rights
        shutil.copytree(pkg, link, ignore=shutil.ignore_patterns("__pycache__"))
    return shim


def worker_env(cfg: MusicConfig, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    shim = str(pythonpath_shim())
    env["PYTHONPATH"] = shim  # never leak the caller's site-packages into a backend env
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if sys.platform.startswith("linux"):
        env["LD_LIBRARY_PATH"] = nixos_ld_library_path(env.get("LD_LIBRARY_PATH", ""))
        if Path("/run/opengl-driver/lib").exists():
            env.setdefault("TRITON_LIBCUDA_PATH", "/run/opengl-driver/lib")
    # Weights are local; nothing in a worker may reach the network.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if extra:
        env.update(extra)
    return env


@dataclass
class ProbeResult:
    backend: str
    python: str
    source: str
    ok: bool
    missing: list[str] = field(default_factory=list)
    torch: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "python": self.python,
            "source": self.source,
            "ok": self.ok,
            "missing": self.missing,
            "torch": self.torch,
            "error": self.error,
        }


_PROBE_CODE = r"""
import importlib.util, json, sys
mods = sys.argv[1:]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
info = {}
if importlib.util.find_spec("torch") is not None and "--no-torch" not in mods:
    try:
        import torch
        info = {"version": torch.__version__, "cuda": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "mps": bool(getattr(torch.backends, "mps", None)
                            and torch.backends.mps.is_available())}
        if info["cuda"]:
            info["device"] = torch.cuda.get_device_name(0)
            info["bf16"] = torch.cuda.is_bf16_supported()
    except Exception as exc:
        info = {"error": repr(exc)}
print(json.dumps({"missing": missing, "torch": info}))
"""

_probe_cache: dict[tuple[str, str], ProbeResult] = {}


def probe(cfg: MusicConfig, backend: str, *, refresh: bool = False,
          timeout: float = 120.0) -> ProbeResult:
    spec = env_spec(backend)
    python, source = python_for(cfg, backend)
    key = (backend, str(python))
    if not refresh and key in _probe_cache:
        return _probe_cache[key]
    if not python.exists():
        result = ProbeResult(backend, str(python), source, False, list(spec.modules),
                             error="interpreter not found")
    else:
        import json

        try:
            proc = subprocess.run(
                [str(python), "-c", _PROBE_CODE, *spec.modules],
                capture_output=True, text=True, timeout=timeout, env=worker_env(cfg),
                check=False,
            )
            data = json.loads(proc.stdout.strip().splitlines()[-1])
            missing = list(data["missing"])
            result = ProbeResult(backend, str(python), source, not missing, missing,
                                 data.get("torch", {}))
        except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as exc:
            result = ProbeResult(backend, str(python), source, False, list(spec.modules),
                                 error=str(exc))
    _probe_cache[key] = result
    return result


def require_backend(cfg: MusicConfig, backend: str) -> Path:
    result = probe(cfg, backend)
    if not result.ok:
        detail = result.error or f"missing modules: {', '.join(result.missing)}"
        raise BackendNotReadyError(
            f"backend {backend!r} is not installed ({detail}; python {result.python})",
            hint=f"run `prompt2music backends install {backend}`",
        )
    return Path(result.python)


def install_commands(cfg: MusicConfig, backend: str, *, python: str | None = None,
                     ref: str | None = None, torch_index: str | None = None) -> list[list[str]]:
    """The exact commands ``backends install`` runs. Shown before running."""
    spec = env_spec(backend)
    uv = shutil.which("uv")
    if uv is None:
        raise ToolMissingError("uv is required to build backend environments",
                               hint="install uv: https://docs.astral.sh/uv/")
    venv = env_dir(cfg, backend)
    vpy = str(_venv_python(venv))
    base_python = python or os.environ.get("PROMPT2MUSIC_ENV_PYTHON") or spec.python_version
    cmds: list[list[str]] = []
    src = src_dir(cfg, backend)
    if spec.git_url:
        git = shutil.which("git")
        if git is None:
            raise ToolMissingError("git is required to fetch upstream backend sources")
        if not src.exists():
            cmds.append([git, "clone", "--filter=blob:none", spec.git_url, str(src)])
        cmds.append([git, "-C", str(src), "fetch", "--tags", "origin"])
        cmds.append([git, "-C", str(src), "checkout", ref or spec.git_ref])
    if spec.method == "uv-sync":
        # UV_PROJECT_ENVIRONMENT is applied by run_install for this step.
        cmds.append([uv, "sync", "--project", str(src), "--python", base_python, "--inexact"])
    else:
        cmds.append([uv, "venv", str(venv), "--python", base_python, "--allow-existing"])
        if spec.torch_index and torch_index:
            cmds.append([uv, "pip", "install", "--python", vpy, "torch",
                         "--index-url", torch_index])
        if spec.packages:
            cmds.append([uv, "pip", "install", "--python", vpy, *spec.packages])
        if spec.method == "pip-editable":
            cmds.append([uv, "pip", "install", "--python", vpy, "-e", str(src)])
        elif spec.method == "pip-project":
            cmds.append([uv, "pip", "install", "--python", vpy, str(src)])
    for pkg in spec.best_effort_packages:
        cmds.append([uv, "pip", "install", "--python", vpy, pkg, "--no-build-isolation",
                     "#best-effort"])
    return cmds


def run_install(cfg: MusicConfig, backend: str, commands: list[list[str]],
                log: Any = None) -> ProbeResult:
    venv = env_dir(cfg, backend)
    venv.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(venv)
    if sys.platform.startswith("linux"):
        env["LD_LIBRARY_PATH"] = nixos_ld_library_path(env.get("LD_LIBRARY_PATH", ""))
    for cmd in commands:
        best_effort = cmd[-1] == "#best-effort"
        argv = cmd[:-1] if best_effort else cmd
        if log:
            log(" ".join(argv))
        proc = subprocess.run(argv, env=env, check=False)
        if proc.returncode != 0:
            if best_effort:
                if log:
                    log(f"warning: optional step failed (exit {proc.returncode})")
                continue
            raise BackendNotReadyError(
                f"install step failed (exit {proc.returncode}): {' '.join(argv)}")
    return probe(cfg, backend, refresh=True)


def installed_revision(cfg: MusicConfig, backend: str) -> str | None:
    src = src_dir(cfg, backend)
    git = shutil.which("git")
    if not src.is_dir() or git is None:
        return None
    proc = subprocess.run([git, "-C", str(src), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=False)
    return proc.stdout.strip() or None
