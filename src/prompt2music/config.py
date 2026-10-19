"""prompt2music configuration (``~/.config/ai-media/prompt2music.toml``).

Schema-driven: every key has a type, default and validator, so ``config set``
rejects bad values before anything is written. Files are edited through
tomlkit, which keeps user comments and ordering intact.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt2music import paths
from prompt2music.errors import ConfigError


@dataclass(frozen=True)
class Key:
    type: type | tuple[type, ...]
    default: Any
    help: str
    choices: tuple[Any, ...] | None = None
    check: Callable[[Any], bool] | None = None
    check_msg: str = ""


def _nonneg(v: Any) -> bool:
    return v >= 0


def _pos(v: Any) -> bool:
    return v > 0


def _port(v: Any) -> bool:
    return 1 <= v <= 65535


def _str_list(v: Any) -> bool:
    return all(isinstance(x, str) for x in v)


SCHEMA: dict[str, dict[str, Key]] = {
    "general": {
        "default_model": Key(str, "auto", "Model id, or 'auto' to choose per request"),
        "output_directory": Key(str, "~/Music/AI/prompt2music",
                                "Where generated audio is written"),
        "offline": Key(bool, False, "Never contact the network (downloads fail fast)"),
        "default_format": Key(str, "wav", "Output format",
                              choices=("wav", "flac", "mp3", "ogg", "opus")),
        "quality": Key(str, "balanced", "Preference used by --model auto",
                       choices=("fast", "balanced", "best")),
        "allow_noncommercial_models": Key(
            bool, True, "Let --model auto pick non-commercial-licensed checkpoints"),
    },
    "hardware": {
        "device": Key(str, "auto", "auto, cpu, cuda, cuda:N or mps",
                      check=lambda v: bool(re.fullmatch(r"auto|cpu|mps|cuda(:\d+)?", v)),
                      check_msg="expected auto, cpu, mps, cuda or cuda:N"),
        "precision": Key(str, "auto", "Inference precision",
                         choices=("auto", "bf16", "fp16", "fp32")),
        "vram_limit_gb": Key((int, float), 0, "Cap for model VRAM; 0 = total minus headroom",
                             check=_nonneg, check_msg="must be >= 0"),
        "vram_headroom_gb": Key((int, float), 2.0,
                                "VRAM kept free for the desktop and other apps",
                                check=_nonneg, check_msg="must be >= 0"),
        "cpu_offload": Key(bool, False, "Offload model parts to system RAM when supported"),
        "allow_tf32": Key(bool, True, "Enable TF32 matmuls on Ampere+ GPUs"),
        "torch_compile": Key(bool, False,
                             "Try torch.compile where a backend supports it (benchmark first)"),
    },
    "models": {
        "directory": Key(str, "", "Model storage; empty = XDG data dir"),
        "auto_download": Key(bool, False, "Download missing models without asking"),
        "verify_on_load": Key(bool, True, "Check file sizes against the manifest before load"),
        "max_download_workers": Key(int, 8, "Parallel file downloads", check=_pos,
                                    check_msg="must be > 0"),
        "hf_endpoint": Key(str, "", "Optional Hugging Face mirror endpoint (HF_ENDPOINT)"),
        "token_env": Key(str, "HF_TOKEN", "Environment variable holding a Hugging Face token"),
    },
    "generation": {
        "duration": Key((int, float), 30, "Default duration in seconds", check=_pos,
                        check_msg="must be > 0"),
        "instrumental": Key(bool, True, "Default to instrumental output"),
        "seed": Key(int, -1, "Default seed; -1 = random"),
        "batch_size": Key(int, 1, "Default batch size", check=lambda v: 1 <= v <= 16,
                          check_msg="must be 1..16"),
        "enhance_prompt": Key(bool, False, "Apply prompt enhancement by default (opt-in)"),
    },
    "backends": {
        "envs_directory": Key(str, "", "Backend virtualenvs; empty = XDG data dir"),
        "python": Key(dict, {}, "Per-backend python override, e.g. {musicgen = '/path/python'}"),
    },
    "jobs": {
        "idle_unload_seconds": Key(int, 600, "Unload a warm model after this idle time",
                                   check=_nonneg, check_msg="must be >= 0"),
        "daemon_idle_exit_seconds": Key(int, 1800,
                                        "Auto-started daemon exits after this idle time "
                                        "(0 = never)", check=_nonneg,
                                        check_msg="must be >= 0"),
        "max_queued": Key(int, 50, "Maximum queued jobs", check=_pos, check_msg="must be > 0"),
        "timeout_seconds": Key(int, 7200, "Kill a generation that runs longer", check=_pos,
                               check_msg="must be > 0"),
    },
    "llm": {
        "provider": Key(str, "none", "Local LLM for enhancement/lyrics",
                        choices=("none", "ollama", "llamacpp", "transformers")),
        "endpoint": Key(str, "", "Override server URL (ollama/llama.cpp); must be local"),
        "model": Key(str, "", "Model name (ollama) or HF id / path (transformers)"),
        "allow_remote_endpoint": Key(bool, False, "Permit non-localhost LLM endpoints"),
    },
    "mcp": {
        "transport": Key(str, "stdio", "Default transport", choices=("stdio", "http")),
        "host": Key(str, "127.0.0.1", "HTTP bind address"),
        "port": Key(int, 8765, "HTTP port", check=_port, check_msg="must be 1..65535"),
        "auth_token_env": Key(str, "PROMPT2MUSIC_MCP_TOKEN",
                              "Env var holding the bearer token required for non-local HTTP"),
        "max_concurrent_jobs": Key(int, 1, "Jobs running at once", check=_pos,
                                   check_msg="must be > 0"),
        "max_queued_jobs": Key(int, 8, "Queued+running MCP jobs allowed", check=_pos,
                               check_msg="must be > 0"),
        "allow_model_downloads": Key(bool, False, "Permit music_model_download"),
        "max_download_gb": Key((int, float), 20, "Largest download MCP may start",
                               check=_pos, check_msg="must be > 0"),
        "allow_model_deletion": Key(bool, False, "Permit music_model_remove"),
        "allowed_models": Key(list, [], "Model allowlist for MCP; empty = all catalogued",
                              check=_str_list, check_msg="must be a list of strings"),
        "allowed_input_dirs": Key(list, [], "Extra directories MCP may read audio from",
                                  check=_str_list, check_msg="must be a list of strings"),
        "allowed_output_dirs": Key(list, [], "Extra directories MCP may write into",
                                   check=_str_list, check_msg="must be a list of strings"),
        "max_input_file_mb": Key(int, 500, "Largest input audio file", check=_pos,
                                 check_msg="must be > 0"),
        "max_duration_seconds": Key((int, float), 600, "Longest generation MCP may request",
                                    check=_pos, check_msg="must be > 0"),
    },
    "logging": {
        "level": Key(str, "info", "Log level", choices=("debug", "info", "warning", "error")),
        "audit_log": Key(bool, True, "Write MCP tool calls to an audit log (JSON lines)"),
    },
}


def defaults() -> dict[str, dict[str, Any]]:
    return {s: {k: copy.deepcopy(key.default) for k, key in keys.items()}
            for s, keys in SCHEMA.items()}


def _type_ok(key: Key, value: Any) -> bool:
    types = key.type if isinstance(key.type, tuple) else (key.type,)
    if isinstance(value, bool) and bool not in types:
        return False  # bool is an int subclass; never accept it for numbers
    return isinstance(value, types)


def validate_value(section: str, name: str, value: Any) -> Any:
    keys = SCHEMA.get(section)
    if keys is None:
        raise ConfigError(f"unknown config section [{section}]")
    key = keys.get(name)
    if key is None:
        raise ConfigError(f"unknown config key {section}.{name}",
                          hint="run `prompt2music config show` to list keys")
    if not _type_ok(key, value):
        expected = key.type.__name__ if isinstance(key.type, type) else "number"
        raise ConfigError(f"{section}.{name} must be {expected}, got {value!r}")
    if key.choices is not None and value not in key.choices:
        raise ConfigError(f"{section}.{name} must be one of {', '.join(map(str, key.choices))}")
    if key.check is not None and not key.check(value):
        raise ConfigError(f"{section}.{name} {key.check_msg}")
    return value


def parse_cli_value(section: str, name: str, raw: str) -> Any:
    """Convert a ``config set`` string to the schema type."""
    key = SCHEMA.get(section, {}).get(name)
    if key is None:
        validate_value(section, name, None)  # raises the unknown-key error
    assert key is not None
    types = key.type if isinstance(key.type, tuple) else (key.type,)
    text = raw.strip()
    if bool in types:
        lowered = text.lower()
        if lowered in {"true", "yes", "on", "1"}:
            return True
        if lowered in {"false", "no", "off", "0"}:
            return False
        raise ConfigError(f"{section}.{name} expects true/false")
    if list in types or dict in types:
        import tomllib

        try:
            return tomllib.loads(f"v = {text}")["v"]
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{section}.{name} expects a TOML value: {exc}") from exc
    if int in types or float in types:
        try:
            num = float(text)
        except ValueError as exc:
            raise ConfigError(f"{section}.{name} expects a number") from exc
        if float not in types:
            if not num.is_integer() or "." in text:
                raise ConfigError(f"{section}.{name} expects an integer")
            return int(num)
        return int(num) if num.is_integer() and "." not in text else num
    return text


class MusicConfig:
    """Validated view over the TOML document."""

    def __init__(self, data: dict[str, dict[str, Any]], path: Path,
                 document: Any | None = None) -> None:
        self.data = data
        self.path = path
        self._document = document

    # -- access ---------------------------------------------------------
    def get(self, dotted: str) -> Any:
        section, _, name = dotted.partition(".")
        if section not in SCHEMA:
            raise ConfigError(f"unknown config section [{section}]")
        if not name:
            return dict(self.data[section])
        if name not in SCHEMA[section]:
            raise ConfigError(f"unknown config key {dotted}")
        return self.data[section][name]

    def __getitem__(self, section: str) -> dict[str, Any]:
        return self.data[section]

    # -- derived paths -------------------------------------------------
    @property
    def models_dir(self) -> Path:
        raw = self.data["models"]["directory"]
        return paths.expand(raw) if raw else paths.default_models_dir()

    @property
    def envs_dir(self) -> Path:
        raw = self.data["backends"]["envs_directory"]
        return paths.expand(raw) if raw else paths.default_envs_dir()

    @property
    def output_dir(self) -> Path:
        return paths.expand(self.data["general"]["output_directory"])

    @property
    def offline(self) -> bool:
        env = os.environ.get("PROMPT2MUSIC_OFFLINE") or os.environ.get("HF_HUB_OFFLINE")
        if env and env.lower() in {"1", "true", "yes"}:
            return True
        return bool(self.data["general"]["offline"])

    def hf_token(self) -> str | None:
        name = self.data["models"]["token_env"]
        return os.environ.get(name) or None

    # -- mutation ------------------------------------------------------
    def set(self, dotted: str, value: Any) -> None:
        section, _, name = dotted.partition(".")
        if not name:
            raise ConfigError("use section.key, e.g. general.default_model")
        validate_value(section, name, value)
        self.data[section][name] = value
        doc = self._ensure_document()
        if section not in doc:
            import tomlkit

            doc[section] = tomlkit.table()
        doc[section][name] = value

    def save(self) -> Path:
        import tomlkit

        doc = self._ensure_document()
        if self.managed:
            raise ConfigError(
                f"{self.path} is managed declaratively (read-only / Nix store)",
                hint="change programs.prompt2music.settings in your Home Manager config, "
                "or point PROMPT2MUSIC_CONFIG at a writable file",
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".toml.tmp")
        tmp.write_text(tomlkit.dumps(doc), encoding="utf-8")
        os.replace(tmp, self.path)
        return self.path

    @property
    def managed(self) -> bool:
        """True when the file is a Home Manager / Nix store symlink or read-only."""
        if not self.path.exists():
            return False
        if self.path.is_symlink() and str(self.path.resolve()).startswith("/nix/store/"):
            return True
        return not os.access(self.path, os.W_OK)

    def _ensure_document(self) -> Any:
        if self._document is None:
            self._document = render_default_document(self.data)
        return self._document

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self.data)


def render_default_document(data: dict[str, dict[str, Any]] | None = None) -> Any:
    import tomlkit

    data = data or defaults()
    doc = tomlkit.document()
    doc.add(tomlkit.comment("prompt2music configuration — part of ai-media-cli"))
    doc.add(tomlkit.comment("Edit freely; `prompt2music config set` keeps comments."))
    for section, keys in SCHEMA.items():
        table = tomlkit.table()
        for name, key in keys.items():
            item = tomlkit.item(data[section][name])
            item.comment(key.help)
            table.add(name, item)
        doc.add(section, table)
    return doc


def load_music_config(path: Path | None = None, *, create: bool = False) -> MusicConfig:
    """Load and validate; unknown keys are rejected so typos surface early."""
    cfg_path = path or paths.config_path()
    data = defaults()
    if not cfg_path.exists():
        cfg = MusicConfig(data, cfg_path)
        if create:
            cfg.save()
        return cfg
    import tomlkit

    try:
        doc = tomlkit.parse(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:  # tomlkit raises several parse error types
        raise ConfigError(f"cannot parse {cfg_path}: {exc}") from exc
    raw = doc.unwrap()
    for section, values in raw.items():
        if section not in SCHEMA:
            raise ConfigError(f"{cfg_path}: unknown section [{section}]")
        if not isinstance(values, dict):
            raise ConfigError(f"{cfg_path}: [{section}] must be a table")
        for name, value in values.items():
            try:
                data[section][name] = validate_value(section, name, value)
            except ConfigError as exc:
                raise ConfigError(f"{cfg_path}: {exc}") from exc
    return MusicConfig(data, cfg_path, doc)


def reset_config(path: Path | None = None) -> Path:
    cfg_path = path or paths.config_path()
    if cfg_path.exists():
        backup = cfg_path.with_suffix(".toml.bak")
        os.replace(cfg_path, backup)
    return MusicConfig(defaults(), cfg_path).save()
