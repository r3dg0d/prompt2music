# Performance and GPU memory

No performance numbers are published here yet: none have been measured on real
weights (see [models.md](models.md#verification-status)). Use
`prompt2music benchmark` on your hardware. It records only measured values.

## What is implemented

| Area | Behaviour |
| ---- | --------- |
| Model reuse | The job daemon keeps the last model loaded in its worker; consecutive jobs skip loading (`load.reused = true` in results). Idle models unload after `jobs.idle_unload_seconds`. |
| One model per GPU | A cross-process lock (`~/.local/share/ai-media/music/gpu.lock`) stops two prompt2music processes loading models onto the GPU at once. Loading a different model shuts the old worker down, returning all its memory. |
| VRAM budget | Before loading: `min(vram_limit_gb, total) ` or `total − vram_headroom_gb` (default 2 GB for the desktop). Models whose upstream minimum exceeds the budget, or the currently free VRAM, are refused with a hint. Inside the worker the same budget is enforced with `torch.cuda.set_per_process_memory_fraction`. |
| `--vram-limit 20G` | per-request override of the budget |
| Precision | `auto` picks BF16 on compute capability ≥ 8.0 (RTX 30/40), FP16 below, FP32 on CPU; `--precision` overrides |
| TF32 | `hardware.allow_tf32` (default on) enables TF32 matmuls where the backend respects torch flags |
| Fragmentation | workers start with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| CPU offload | `hardware.cpu_offload` is forwarded to backends that support it (ACE-Step `offload_to_cpu`) |
| Quantization | ACE-Step `quantization` option is forwarded when set in the model options |
| Attention | Backends use their upstream defaults (PyTorch SDPA in transformers; flash-attn 2 for Stable Audio Medium as required upstream) |
| `torch.compile` | `hardware.torch_compile` (default **off**) compiles the MusicGen decoder / enables ACE-Step `compile_model`. Benchmark before enabling. |
| OOM handling | CUDA OOM becomes a typed `out_of_memory` error with a hint; the worker survives when the backend recovers, otherwise it is restarted on the next job |

CUDA Graphs and TensorRT were not implemented. They need per-model shape
specialisation and a quality comparison against the baseline path, and should
not be enabled blindly. The baseline path (upstream defaults) is always
available, since every optimisation above is opt-in or matches upstream
defaults.

## Benchmark methodology

```bash
prompt2music benchmark --model ace-step-1.5 --duration 30 --runs 3 -o bench.json
prompt2music benchmark --all-installed --output benchmarks.json
```

* Each model is loaded fresh (worker start + load are timed separately from
  generation).
* Every run uses the same prompt and seed; outputs are discarded.
* `rtf = generation_seconds / audio_seconds` per run, plus the mean and min.
* `peak_vram_mb` is `torch.cuda.max_memory_allocated` in the worker (reset
  before each run); `peak_ram_mb` is the worker's VmHWM.
* `config` records device, precision, budget, offload/compile flags, torch
  version and CUDA availability.

To compare an optimisation, run the benchmark twice with only that setting
changed, and listen to both outputs (same seed) before keeping it.

## RTX 4090 (24 GB) suggestions

* Leave `vram_headroom_gb = 2` (or `vram_limit_gb = 20`) if you run a desktop
  compositor or games.
* `ace-step-1.5` fits comfortably; `ace-step-1.5-xl-turbo` is upstream's
  20–24 GB tier.
* `yue2-3b` needs the full 24 GB upstream, so close other GPU apps or raise the
  limit for that job only: `--vram-limit 24G`.
* Keep the daemon running (`programs.prompt2music.daemon.enable` or
  `prompt2music daemon start`) so iterative prompting reuses the loaded model.
