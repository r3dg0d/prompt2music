# prompt2music command reference

`prompt2music` and `aimusic` are the same CLI (`ai-media music` in ai-media-cli). Global flags
(`--json`, `-q`, `-v`, `--debug`, `--config`) work before or after the
subcommand. Exit codes: `0` ok, `1` runtime failure, `2` invalid
request/config/missing model, `130` cancelled.

## Generate

```bash
prompt2music generate "Dark cyberpunk synthwave with heavy bass" --duration 120 \
  --instrumental --model stable-audio-3-medium --output cyberpunk.wav
prompt2music generate "Aggressive industrial techno" --bpm 145 --key Fm
prompt2music generate "Beautiful piano instrumental" --model auto --quality best
prompt2music generate --preset boss-battle --duration 60 --format flac
prompt2music generate --prompts-file prompts.txt --model musicgen-small   # batch, one per line
prompt2music generate "Ambient electronic music" --vram-limit 20G --background
prompt2music generate "lofi" --dry-run --json      # show the resolved request only
```

| Option | Meaning |
| ------ | ------- |
| `-m/--model` | model id or `auto` (never silently replaced when explicit) |
| `-d/--duration` | seconds |
| `--seed` | seed (`-1` random); recorded per output |
| `--negative` | negative prompt (Stable Audio only; refused elsewhere) |
| `--bpm --key --time-signature` | native inputs for ACE-Step; written into the prompt for text-only models |
| `--instrumental / --vocals` | vocals need a vocal model |
| `--reference FILE` | reference-audio conditioning (ACE-Step) |
| `--melody FILE` | melody conditioning (`musicgen-melody`) |
| `--batch N` | N outputs from one request |
| `--steps --guidance` | sampler controls where the backend has them |
| `-o FILE.ext` / `-o DIR` / `--format` | output path and format (wav, flac, mp3, ogg, opus) |
| `--enhance` / `--raw-prompt` | opt-in prompt expansion / send text exactly as written |
| `--device --precision --vram-limit` | hardware overrides |
| `--normalize peak\|loudness --fade-in --fade-out` | post-processing (recorded separately) |
| `--background` | queue for the daemon and return a job id |
| `--no-daemon` | run in this process even if the daemon is up |
| `--force` | overwrite an existing output (otherwise `-v2` is appended) |

Every output gets a JSON sidecar (`song.wav.json`) and a library entry with
prompt, sent prompt, model + revision, seed, parameters, timing and
post-processing.

## Songs, continuation, editing

```bash
prompt2music song --lyrics lyrics.txt --genre "alternative rock" --model ace-step-1.5 -o song.wav
prompt2music continue --input intro.wav --duration 90 --mode intensify -o extended.wav
prompt2music continue -i intro.wav -d 30 --mode transition --to "drum and bass"
prompt2music edit --input original.wav --prompt "Replace the percussion with industrial drums"
prompt2music inpaint --input original.wav --start 30 --end 45 --prompt "Epic orchestral transition"
```

`continue --duration` is the length of **new** audio. The output contains the
original plus the continuation unless `--only-new` is given. Modes are natural,
intensify, transition `--to GENRE`, add-instruments `--to INSTRUMENTS`, mood
`--to MOOD` and outro. Continuation always uses the model's own audio
conditioning. Models without it (ACE-Step) are refused instead of faking it
with concatenation.

## Prompts and lyrics

```bash
prompt2music enhance "sad piano music"
prompt2music presets
prompt2music lyrics template --structure pop --topic "a city after humanity disappears"
prompt2music lyrics validate lyrics.txt --model ace-step-1.5
prompt2music lyrics generate --prompt "A futuristic city after humanity disappears"  # needs [llm]
```

Lyrics generation needs a local LLM (`llm.provider` = `ollama`, `llamacpp` or
`transformers`). Endpoints must be on localhost unless
`llm.allow_remote_endpoint` is set. Music generation never depends on it.

## Models and backends

```bash
prompt2music models list | available [--task song] [--kind stems] | search vocals
prompt2music models info ace-step-1.5
prompt2music models download ID [--revision REV] [--accept-license] [--yes] [--plan]
prompt2music models verify ID [--quick]
prompt2music models update ID
prompt2music models import ID PATH [--link]
prompt2music models load ID        # warm it in the job daemon
prompt2music models unload
prompt2music models remove ID [--yes]
prompt2music models cache [--clean]
prompt2music models benchmark ID --duration 30 --runs 3
prompt2music backends list | install NAME [--ref REF] [--torch-index URL] [--print] [--yes]
```

## Stems

```bash
prompt2music stems models
prompt2music stems download htdemucs
prompt2music stems separate song.wav --model htdemucs --output-dir stems/ [--stems vocals,drums]
```

## Audio toolkit (FFmpeg)

```bash
prompt2music audio info song.wav [--loudness]
prompt2music audio validate song.wav --expect-duration 30
prompt2music audio convert song.wav song.mp3 [--sample-rate 48000] [--channels 1] [--tag title=X]
prompt2music audio normalize song.wav [--mode peak --target -1 | --mode loudness --target -14]
prompt2music audio trim song.wav --start 10 --end 40
prompt2music audio fade song.wav --in 2 --out 4
prompt2music audio loop song.wav --count 4 [--crossfade 1]
prompt2music audio mix drums.wav bass.wav [--weights 1,0.8]
prompt2music audio mix soundtrack.wav --voice narration.wav --ducking -o mixed.wav
prompt2music audio crossfade a.wav b.wav --seconds 3 -o joined.wav
prompt2music audio fit song.wav --seconds 62.5
prompt2music audio waveform song.wav
prompt2music audio spectrogram song.wav
prompt2music audio tag song.flac artist=me album=demo
```

Normalisation is level adjustment (peak or EBU R128 loudness), not mastering.
When the true-peak ceiling stops loudness normalisation from reaching its
target, the output says so and reports the achieved LUFS.

## Video soundtrack

```bash
prompt2music soundtrack --video clip.mp4 -p "Epic cinematic orchestral soundtrack" \
  [--original-audio duck|mix|replace] [-o clip-scored.mp4]
```

The music is generated at the video's length (or `--duration`), fitted with a
fade-out, and muxed with the video stream copied (not re-encoded). With `duck`
the original audio stays and the music dips under it.

## Jobs and the daemon

```bash
prompt2music generate "Epic orchestral soundtrack" --background
prompt2music jobs [list] [--status active]
prompt2music jobs status|wait|result|logs [--follow]|cancel JOB_ID
prompt2music daemon start|stop|status|run|install-service
```

States: queued → loading → running → processing → completed | failed | cancelled.
Jobs live in SQLite and are run by a detached daemon that outlives the CLI. The
daemon keeps the last model loaded, so consecutive jobs skip reloading; it
unloads after `jobs.idle_unload_seconds` and an auto-started daemon exits after
`jobs.daemon_idle_exit_seconds`. Cancelling sends a cooperative stop first. If
the backend does not stop within the grace period (native code that never
yields), the worker process is terminated, which also unloads the model.
Cancelling while a model is still loading stops that load immediately.

## Benchmark

```bash
prompt2music benchmark --model ace-step-1.5 --duration 30 --runs 3
prompt2music benchmark --all-installed --output benchmarks.json
```

Measures load time (worker and wall clock), generation time, RTF
(`generation_seconds / audio_seconds`; below 1 is faster than real time), peak
VRAM (torch allocator, CUDA only), peak worker RAM, sample rate, channels and
the inference configuration. Nothing is estimated.

## Library

```bash
prompt2music library list | search cyberpunk | info TRACK_ID
prompt2music library export TRACK_ID -o ~/exports -f flac
prompt2music library reproduce TRACK_ID     # same model, seed and parameters
prompt2music library remove TRACK_ID [--delete-files]
```

Reproduction is best effort: the same model revision, seed and parameters do
not guarantee bit-identical audio across GPUs, drivers or library versions.

## Configuration

```bash
prompt2music config show | path
prompt2music config get general.default_model
prompt2music config set general.default_model ace-step-1.5
prompt2music config set mcp.allowed_input_dirs '["~/Music", "~/Projects/game/audio"]'
prompt2music config reset
```

Values are validated against the schema before saving, and comments in the
file are preserved.

## Other

```bash
prompt2music doctor [--quick] [--json]
prompt2music setup [--non-interactive]
prompt2music completion bash|zsh|fish > ~/.local/share/bash-completion/completions/prompt2music
prompt2music mcp [serve] | config --client claude|opencode|cursor|vscode | tools
```
