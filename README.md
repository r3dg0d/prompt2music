# prompt2music

**Your words. Your music. Your hardware.**

prompt2music turns natural-language prompts and lyrics into music on your own
machine, using open-weight models. It is a polished CLI (`prompt2music`,
alias `aimusic`) and an MCP server (`prompt2music-mcp`, alias `aimusic-mcp`)
that lets coding agents such as Claude Code, OpenCode, Cursor and VS Code
generate, edit and manage music. No cloud inference APIs are used.

> prompt2music is also shipped inside [ai-media-cli](https://github.com/r3dg0d/ai-media-cli)
> as `ai-media music`. Both use the same XDG locations, so models, config, jobs
> and the library are shared between the two installs.

## Quick start

```bash
git clone https://github.com/r3dg0d/prompt2music.git
cd prompt2music
uv sync --extra mcp                      # CLI + MCP server; no torch here
uv run prompt2music doctor

prompt2music models available            # tasks, sizes, licences
prompt2music backends install ace-step --yes   # isolated venv for the model family
prompt2music models download ace-step-1.5      # shows size + licence, verifies every file
prompt2music generate "Beautiful ambient piano with soft strings" \
  --model ace-step-1.5 --duration 20 --output test.wav
```

As a tool: `uv tool install "prompt2music[mcp] @ git+https://github.com/r3dg0d/prompt2music"`.
FFmpeg must be on `PATH`. NixOS: `nix run github:r3dg0d/prompt2music -- doctor`,
or use the Home Manager module (see [docs/nixos.md](docs/nixos.md)).

## Models

| Model | Tasks | Vocals | Weights licence |
| ----- | ----- | ------ | --------------- |
| ACE-Step 1.5 (+ XL turbo) | text-to-music, song, edit, inpaint | yes | MIT |
| Stable Audio 3 Small-Music / Medium | text-to-music, edit, inpaint, continue | no | Stability AI Community (gated, conditional commercial) |
| MusicGen small / medium / large / melody | text-to-music, continue, melody | no | CC-BY-NC-4.0 (non-commercial) |
| HeartMuLa oss-3B | song | yes | Apache-2.0 |
| YuE2 3B | song (24 GB GPU) | yes | CC-BY-NC-4.0 (non-commercial) |
| Demucs (htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra) | stem separation | — | MIT |

The application code is MIT. Model weights have their own licences, which are
shown before every download and recorded with your acceptance. Unsupported
operations are refused rather than approximated (for example, ACE-Step has no
upstream continuation mode, so `continue` won't use it). See
[docs/models.md](docs/models.md) for capabilities, hardware needs and
**verification status**: MusicGen and Demucs run against the real upstream
code in CI with tiny random-weight checkpoints. ACE-Step, Stable Audio 3,
HeartMuLa and YuE2 follow upstream's documented API but have not yet been run
on real weights.

## Features

* **Generation**: `generate`, `song` (structured lyrics), `continue` (native
  audio conditioning, with modes such as intensify / transition / outro),
  `edit`, `inpaint`, presets, batch, seeds, `--model auto` selection based on
  task, installed models, VRAM, licence policy and quality preference.
* **Models**: Hugging Face downloader with exact-size plans, licence
  acceptance, revision pinning, resumable staging, sha256 / git-blob
  verification and atomic installs; `verify`, `update`, `import`, `remove`,
  `cache`.
* **Backends** run in isolated per-family venvs (upstream stacks conflict)
  through a worker process, with cooperative cancellation and forced stop as
  a fallback.
* **Jobs**: SQLite queue plus a detached daemon that keeps the last model warm,
  survives the CLI exiting, and can run as a systemd user service.
* **Audio toolkit** (FFmpeg): info, validate, convert (wav/flac/mp3/ogg/opus),
  peak/loudness normalise, trim, fade, loop, mix, duck, crossfade, fit,
  waveform, spectrogram, tags. `soundtrack --video` scores a video.
* **Stems** with Demucs; **prompt enhancement** (rule-based, optional local
  LLM); **lyrics** templates, validation and LLM drafting.
* **GPU**: device/precision planning, VRAM budget (`--vram-limit 20G`, desktop
  headroom), cross-process GPU lock, OOM handling, measured `benchmark` (RTF =
  generation_seconds / audio_seconds).
* **Library**: every track gets a JSON sidecar and SQLite entry (prompt, model
  revision, seed, parameters, post-processing) for search, export and
  reproduction.

## MCP

```bash
claude mcp add prompt2music -- prompt2music-mcp
prompt2music mcp config --client opencode|cursor|vscode
```

There are 30 tools (generate / song / continue / edit / inpaint, models, audio,
stems, system, jobs). Long operations return a `job_id` to poll, and results
are file paths and `file://` URIs. Inputs are untrusted: paths are confined to
configured directories (with symlink protection), downloads and deletion are
off by default, quotas and size limits apply, and calls are audited. stdio is
the default; HTTP binds to localhost and needs a bearer token anywhere else.
See [docs/mcp.md](docs/mcp.md).

## Documentation

[Installation](docs/installation.md) · [Models](docs/models.md) ·
[Commands](docs/commands.md) · [MCP](docs/mcp.md) · [NixOS](docs/nixos.md) ·
[Performance](docs/optimization.md) · [Troubleshooting](docs/troubleshooting.md) ·
[Architecture](docs/README.md)

## Development

```bash
uv sync --extra dev --extra mcp
uv run ruff check src tests && uv run mypy && uv run pytest
uv run pytest -m "gpu and network" tests/test_gpu_acceptance.py   # real weights, GPU
```

## Roadmap

1. Run the GPU acceptance tests and benchmarks on an RTX 4090; publish measured numbers.
2. ACE-Step base-model tasks (lego / extract / complete), YuE2 cover and score editing.
3. Benchmark-driven optimisations (torch.compile, quantisation presets), PyPI release.

## License

MIT for this code; see [LICENSE](LICENSE). Model weights are licensed
separately (see above and [docs/models.md](docs/models.md)).
