# Installing prompt2music

Installing prompt2music gives you `prompt2music`, `aimusic`,
`prompt2music-mcp` and `aimusic-mcp` (it is also bundled in
[ai-media-cli](https://github.com/r3dg0d/ai-media-cli) as `ai-media music`).
Model weights and heavy ML dependencies are **not** installed
with it.

There are three layers:

1. **The CLI / MCP server** (this package): light; needs Python ≥ 3.11 and
   FFmpeg.
2. **Backend environments**: one venv per model family, built on demand with
   `prompt2music backends install NAME`. Upstream projects pin incompatible
   stacks (ACE-Step 1.5 wants torch 2.10 with transformers < 4.58; the Qwen image
   env here wants transformers ≥ 5.17), so they are never mixed.
3. **Model weights**: downloaded explicitly with `prompt2music models download
   ID` after showing size and licence.

## 1. Install the CLI

```bash
git clone https://github.com/r3dg0d/prompt2music.git
cd prompt2music
uv sync --extra mcp            # or: python -m venv .venv && pip install -e ".[mcp]"
uv run prompt2music doctor
```

As a tool:

```bash
uv tool install "prompt2music[mcp] @ git+https://github.com/r3dg0d/prompt2music"
prompt2music --help
```

The package is not on PyPI. `uv tool install prompt2music` will work once it is
published.

FFmpeg/ffprobe must be on `PATH` (`apt install ffmpeg`, `brew install ffmpeg`,
`pkgs.ffmpeg` on NixOS). For NixOS see [nixos.md](nixos.md).

## 2. Build a backend environment

```bash
prompt2music backends list                      # status of every backend
prompt2music backends install musicgen --print  # show the exact commands
prompt2music backends install musicgen --yes    # build ~/.local/share/ai-media/envs/music-musicgen
```

| backend | how it is built | notes |
| ------- | --------------- | ----- |
| `musicgen` | `uv pip install torch transformers…` | add `--torch-index https://download.pytorch.org/whl/cu128` to pick a CUDA build |
| `ace-step` | clones ACE-Step-1.5, `uv sync` with upstream's lockfile | torch 2.10 + CUDA 12.8 on Linux x86_64 |
| `stable-audio` | clones stable-audio-3, `uv sync`, then tries `flash-attn` | Medium needs flash-attn 2 |
| `heartmula` | clones heartlib, editable install, Python 3.10 | |
| `yue` | clones YuE, installs it, Python 3.12 | 24 GB GPU |
| `demucs` | `uv pip install torch demucs>=4.1` | stem separation |

Pin an upstream revision with `--ref TAG_OR_COMMIT`. The resolved commit is
printed after install. To reuse an interpreter you already manage, set
`PROMPT2MUSIC_PYTHON_<BACKEND>` (e.g. `PROMPT2MUSIC_PYTHON_ACE_STEP=/path/to/python`)
or `backends.python` in the config.

## 3. Download a model

```bash
prompt2music models available
prompt2music models download ace-step-1.5 --plan     # size, licence, revision; no download
prompt2music models download ace-step-1.5            # asks to accept licence + size
prompt2music models verify ace-step-1.5              # full sha256 check
```

Downloads are resumable: interrupted files stay in `<models>/.staging/ID` and
are reused by the next attempt. The model only appears as installed after
every file has been verified against the Hub's sha256 / git blob id and the
directory has been renamed into place. For gated repositories (Stable Audio 3),
accept the licence on the Hugging Face page and `export HF_TOKEN=...` first.

Already have weights? Register them without downloading:

```bash
prompt2music models import musicgen-small /path/to/musicgen-small --accept-license
```

## 4. First track

```bash
prompt2music generate "Beautiful ambient piano with soft strings" \
  --model ace-step-1.5 --duration 20 --output test.wav
```

Or run the guided setup, which checks hardware, recommends models and runs a
smoke test:

```bash
prompt2music setup
prompt2music setup --non-interactive --model musicgen-small --accept-license --yes
```

## Paths

| What | Default | Override |
| ---- | ------- | -------- |
| Config | `~/.config/ai-media/prompt2music.toml` | `--config`, `PROMPT2MUSIC_CONFIG` |
| Models | `~/.local/share/ai-media/models/music/` | `models.directory`, `PROMPT2MUSIC_MODELS_DIR` |
| Backend venvs | `~/.local/share/ai-media/envs/music-*` | `backends.envs_directory`, `PROMPT2MUSIC_ENVS_DIR` |
| Jobs / library DB | `~/.local/share/ai-media/music/music.db` | `PROMPT2MUSIC_DATA_DIR` |
| Logs, scratch | `~/.cache/ai-media/music/` | `PROMPT2MUSIC_CACHE_DIR` |
| Output | `~/Music/AI/prompt2music/` | `general.output_directory`, `-o` |

These follow the existing ai-media layout (`~/.local/share/ai-media/envs/qwen`
etc.) rather than a separate `prompt2music` tree.

## Offline use

After models and backends are installed, set `general.offline = true` (or
`PROMPT2MUSIC_OFFLINE=1`). Workers always run with `HF_HUB_OFFLINE=1`, so
generation never touches the network.
