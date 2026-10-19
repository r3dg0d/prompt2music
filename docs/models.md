# prompt2music models

Facts below come from the upstream repositories and model cards as of
2026-10-18. `prompt2music models info ID` prints the same data from the
registry (`src/prompt2music/registry.py`), and the downloader always shows the
exact byte count from Hugging Face before anything is fetched.

## How each integration has been verified

| Level | Meaning |
| ----- | ------- |
| **contract-tested** | The real upstream inference code (same library, same calls) runs in CI against a tiny *randomly initialised* checkpoint built locally. This proves API compatibility, shapes, sample rates, durations, continuation and cancellation. It does not prove musical quality. |
| **api-documented** | Written against upstream's documented Python API and current source signatures. It has **not** been executed in the development container (no GPU, and Hugging Face/GitHub downloads were blocked there). Run `pytest -m gpu` on the workstation to verify. |
| **diagnostic** | Not a music model. |

No real-weight generation or benchmark has been run yet. See
[Verification status](#verification-status) for the exact commands that close
that gap on the RTX 4090.

## Music models

| id | Upstream | Tasks | Vocals | Download | Min VRAM | Weights license | Commercial | Verified |
| -- | -------- | ----- | ------ | -------- | -------- | --------------- | ---------- | -------- |
| `ace-step-1.5` | [ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5) turbo DiT + 1.7B LM | text-to-music, song, edit (cover), inpaint (repaint) | yes | ~10 GB | 6 GB (upstream: <4 GB with INT8 + offload, LM off) | MIT | yes | api-documented |
| `ace-step-1.5-xl-turbo` | ACE-Step 1.5 XL turbo | same | yes | ~15 GB | 12 GB with offload, 20 GB without | MIT | yes | api-documented |
| `stable-audio-3-small-music` | [stable-audio-3](https://github.com/Stability-AI/stable-audio-3) Small-Music (433M) | text-to-music, edit, inpaint, continue | no | ~2 GB | CPU ok | Stability AI Community License | conditional, gated | api-documented |
| `stable-audio-3-medium` | Stable Audio 3 Medium (1.4B) | text-to-music, edit, inpaint, continue | no | ~6 GB | 6.5 GB, CUDA + flash-attn 2 | Stability AI Community License | conditional, gated | api-documented |
| `musicgen-small` | [MusicGen](https://github.com/facebookresearch/audiocraft) via transformers | text-to-music, continue | no | ~2.4 GB | CPU ok | CC-BY-NC-4.0 | **no** | contract-tested |
| `musicgen-medium` | MusicGen 1.5B | text-to-music, continue | no | ~8 GB | 6 GB | CC-BY-NC-4.0 | **no** | contract-tested |
| `musicgen-large` | MusicGen 3.3B | text-to-music, continue | no | ~13 GB | 10 GB | CC-BY-NC-4.0 | **no** | contract-tested |
| `musicgen-melody` | MusicGen Melody 1.5B | text-to-music (+ melody conditioning) | no | ~6.5 GB | 6 GB | CC-BY-NC-4.0 | **no** | api-documented |
| `heartmula-3b` | [HeartMuLa heartlib](https://github.com/HeartMuLa/heartlib) oss-3B (happy-new-year) + HeartCodec | song, text-to-music | yes | ~16 GB | 8 GB (lazy load) | Apache-2.0 | yes | api-documented |
| `yue2-3b` | [YuE2](https://github.com/multimodal-art-projection/YuE) 3B + YuE2-Vae | song | yes | ~10 GB | **24 GB**, BF16 | CC-BY-NC-4.0 (+ creator permission) | **no** | api-documented |
| `test-tone` | built in | text-to-music | no | — | — | MIT | yes | diagnostic |

Sizes are approximate; VRAM numbers are upstream's guidance, not measurements
made here.

### Licences, in plain terms

* **Application code** (this repository): MIT.
* **Open-weight checkpoints** have their own licences, which prompt2music shows
  before every download and records (with your acceptance time) in the model's
  `manifest.json`.
* **Non-commercial**: MusicGen (CC-BY-NC-4.0) and YuE2 (CC-BY-NC-4.0 with extra
  creator terms). Set `general.allow_noncommercial_models = false` to stop
  `--model auto` from picking them.
* **Conditional / gated**: Stable Audio 3 uses the Stability AI Community
  License (free below a revenue threshold; larger organisations need an
  enterprise licence). The Hugging Face repos may require accepting the licence
  on the model page and an `HF_TOKEN`.
* **Permissive**: ACE-Step 1.5 (MIT) and HeartMuLa (Apache-2.0, relicensed
  2026-01-20).

Always read the upstream licence; the summaries above are not legal advice.

### Deliberately not offered

* Stable Audio 3 **Large**: API-only, no downloadable weights.
* HeartMuLa **7B**: described upstream but not released.
* Voice cloning: none of the integrated models offers arbitrary voice cloning.
  Reference-audio conditioning is exposed only for ACE-Step, where upstream
  supports it.

## Per-model capability notes

**ACE-Step 1.5** is driven through `acestep.inference.generate_music` with
`AceStepHandler.initialize_service` and `LLMHandler.initialize`. Task mapping:
text-to-music and song → `text2music`; edit → `cover` (`--strength` maps to
`audio_cover_strength = 1 - strength`); inpaint → `repaint`. Upstream documents
no extend/continuation task, so `prompt2music continue` refuses ACE-Step rather
than pretending. Captions are limited to 512 characters and lyrics to 4096.
Instrumental requests send the documented `[Instrumental]` lyric. BPM, key and
time signature are native inputs.

**Stable Audio 3** uses `StableAudioModel.from_pretrained(name, device,
model_half)` and `generate(...)`. Edit uses `init_audio` and
`init_noise_level` (= `--strength`). Inpaint uses `inpaint_audio` with mask
seconds. Continue follows upstream's recipe: inpaint from the end of the source
with a longer total duration. Upstream resolves weights with `hf_hub_download`,
so the worker points `HF_HUB_CACHE` at the managed model directory and forces
offline mode.

**MusicGen** runs through `transformers.MusicgenForConditionalGeneration`.
Native windows are 30 s. Longer requests and `continue` use windowed
continuation: each new window is generated conditioned on the last 10 s of
audio (genuine audio-prompted generation, the same approach audiocraft uses),
loudness-matched and joined with a 50 ms crossfade. Generation stops
cooperatively through a `StoppingCriteria` when a job is cancelled. BPM and key
have no dedicated inputs; they are written into the text prompt, and the prompt
actually sent is recorded.

**HeartMuLa** follows upstream `examples/run_music_generation.py`
(`HeartMuLaGenPipeline`, lazy loading, BF16 MuLa + FP32 codec). Upstream has no
seed parameter; seeding goes through the global torch RNG and is labelled that
way in the track metadata.

**YuE2** uses `YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B")` and
`song.save_artifacts()`, and outputs 48 kHz FLAC. Song length follows the
lyrics and the generated plan; upstream documents no duration control, so
`--duration` is advisory.

## Stem separation models (Demucs 4.1)

| id | Stems | Download | Licence | Verified |
| -- | ----- | -------- | ------- | -------- |
| `htdemucs` | vocals, drums, bass, other | ~0.1 GB | MIT | contract-tested |
| `htdemucs_ft` | same, higher quality, ~4× slower | ~0.35 GB | MIT | contract-tested (code path) |
| `htdemucs_6s` | + guitar, piano | ~0.1 GB | MIT | contract-tested (code path) |
| `mdx_extra` | vocals, drums, bass, other | ~0.6 GB | MIT | contract-tested (code path) |

Demucs ≥ 4.1 loads safetensors from `adefossez/*` Hugging Face repos, so stem
models use the same verified downloader as the music models (no pickles).

## Verification status

Done in the development container (CPU, no Hugging Face access):

* MusicGen: real transformers code with a random-weight checkpoint. Covers text
  generation, exact durations (2 s, 35 s windowed, 38 s continued), batch, seeds,
  continuation, cooperative cancel mid-generation, CLI → daemon → library, and the
  MCP agent workflow.
* Demucs: real `demucs.api.Separator` with the `demucs_unittest` random model.
* All other code paths (downloader, jobs, daemon, MCP, audio, config) are covered
  by the CPU test suite.

Still to run on the RTX 4090 workstation:

```bash
prompt2music backends install ace-step --yes
pytest -m "gpu and network" tests/test_gpu_acceptance.py   # downloads ~10 GB
PROMPT2MUSIC_ACCEPT_MODEL=stable-audio-3-medium pytest -m "gpu and network" tests/test_gpu_acceptance.py
prompt2music benchmark --all-installed --duration 30 --runs 3 -o benchmarks.json
```
