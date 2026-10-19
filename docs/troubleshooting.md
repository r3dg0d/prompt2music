# Troubleshooting prompt2music

Start with `prompt2music doctor` (add `--json` for bug reports). Every error
prints a `hint:` line; `--debug` shows tracebacks.

| Symptom | Cause / fix |
| ------- | ----------- |
| `backend 'X' is not installed` | Build its env: `prompt2music backends install X --yes`. `backends list` shows which interpreter each backend uses. |
| `X is not installed` (model) | `prompt2music models download X` (or `models import X PATH` for weights you already have). |
| `no installed model can do …` | `--model auto` found nothing suitable; the hint lists models that would work and their sizes. |
| `X does not support continue` | Only models with native audio conditioning can continue (MusicGen, Stable Audio 3). ACE-Step is refused on purpose. |
| `needs ~N GB VRAM; budget is …` | Raise `hardware.vram_limit_gb` / `--vram-limit`, lower `vram_headroom_gb`, enable `cpu_offload`, or pick a smaller model. |
| `only N GB VRAM is free` | Another process holds the GPU (`nvidia-smi`). `prompt2music models unload` frees the daemon's model. |
| `waiting for GPU lock` | Another prompt2music process has a model loaded. Use the daemon (`--background`) instead of parallel foreground runs. |
| `… is gated on Hugging Face` | Accept the licence on the model page, then `export HF_TOKEN=…` (or `hf auth login`). |
| `403` / network errors on download | Your network policy blocks huggingface.co. Set `models.hf_endpoint` to a permitted mirror, or download elsewhere and use `models import`. |
| Download interrupted | Re-run the same `models download`; files in `.staging/` are reused. `models cache --clean` discards partial downloads. |
| `failed verification` | A file did not match the Hub's sha256; it was deleted, so re-run the download. `models verify ID` re-checks an installed model. |
| `out_of_memory` | Shorter duration, `--batch 1`, smaller model, or close other GPU apps. |
| Job stuck in `loading`/`running` after a crash | `prompt2music daemon status`. Orphaned jobs are marked failed when the daemon starts again (`daemon start`). |
| Cancel seems slow | Cancellation is cooperative first; after the grace period (15 s) the worker is terminated. Native kernels cannot be interrupted mid-call. |
| Output silent / wrong length warning | Shown from post-generation checks; the file and sidecar are kept so you can inspect them. Report with `audio info FILE --json`. |
| `config … is managed declaratively` | Home Manager owns the file; change `programs.prompt2music.settings`. |
| `ffmpeg is not installed` | Install FFmpeg (NixOS: `pkgs.ffmpeg`); set `PROMPT2MUSIC_FFMPEG` for a custom binary. |
| MCP client shows no tools | Run `prompt2music-mcp --help` in the same environment the client uses; use the absolute path in the client config. |
| MCP call returns `forbidden` | Path outside the allowed directories, model not allowlisted, downloads/deletion disabled, or a limit exceeded; the message names the rule and config key. |
| NixOS: `libstdc++.so.6` / `libcuda` errors in a backend | Make sure `AI_MEDIA_EXTRA_LD_PATH` is exported (the modules do it) and `/run/opengl-driver/lib` exists. See [nixos.md](nixos.md). |

## Logs

* Worker stderr (library output, tracebacks): `~/.cache/ai-media/music/logs/worker-<backend>.log`
* Per-job log: `prompt2music jobs logs JOB_ID [--follow]`
* Daemon: `~/.cache/ai-media/music/logs/daemon.log`
* MCP audit trail: `~/.cache/ai-media/music/logs/mcp-audit.jsonl`
