# prompt2music

**Your words. Your music. Your hardware.**

prompt2music turns natural-language
prompts and lyrics into music using open-weight models on your own machine. It
is available to people (`prompt2music`, `aimusic`) and to
coding agents (`prompt2music-mcp`, `aimusic-mcp`). No cloud inference APIs are
used.

* [Installation](installation.md): CLI, backend environments, models, paths
* [Models](models.md): supported checkpoints, licences, capabilities, verification status
* [Commands](commands.md): full CLI reference
* [MCP server](mcp.md): Claude Code / OpenCode / Cursor / VS Code setup, tools, security
* [NixOS](nixos.md): flake, Home Manager and NixOS modules
* [Performance](optimization.md): GPU memory management and benchmarking
* [Troubleshooting](troubleshooting.md)

## Architecture

```
prompt2music / aimusic                          prompt2music-mcp / aimusic-mcp
        │  (argparse + Rich CLI)                          │ (MCP SDK v1/v2, stdio | HTTP)
        └──────────────┬──────────────────────────────────┘
                       ▼
   service.py  — select model → prepare prompt → convert inputs → generate
                 → convert format → post-process → validate → library
        │                       │                          │
   selection.py            jobs.py + daemon.py         library.py
   (auto model choice)     (SQLite queue, detached     (SQLite + JSON sidecars)
                            daemon, cancel, warm model)
                       ▼
   engine.py  — one loaded model per process, GPU lock, VRAM budget (gpu.py)
                       │ JSON lines over stdin/stdout
                       ▼
   worker.py  — runs inside the backend's own venv (envs.py)
        backends/{ace_step, stable_audio, musicgen, heartmula, yue, demucs_stems, tone}.py
                       ▲
   downloader.py / manager.py / manifest.py — Hugging Face plans, verification,
                                              atomic installs under ~/.local/share
```

Small helpers shared with [ai-media-cli](https://github.com/r3dg0d/ai-media-cli)
(XDG paths, `nvidia-smi` parsing, signal handling, sidecar metadata, the NixOS
driver-library path) live in `prompt2music._shared`. The XDG layout is the
same, so both installs share models, config, jobs and the library.
