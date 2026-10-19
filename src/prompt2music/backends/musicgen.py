"""MusicGen via Hugging Face transformers (facebook/musicgen-*).

Verified against the transformers ``MusicgenForConditionalGeneration`` API:
text conditioning through the processor, ``generate(max_new_tokens=...)``
returning ``(batch, channels, samples)`` audio, and audio-prompted generation
by passing ``audio=`` to the processor, in which case the returned audio
starts with the (re-encoded) prompt.

Long requests and ``continue`` use windowed continuation: each 30 s window is
conditioned on the last ``context_seconds`` of audio generated so far. This is
the same strategy audiocraft uses for extended generation; it is generative,
not concatenation, and boundaries are joined with a short crossfade.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from prompt2music.backends.audio_io import output_name, read_wav, write_wav
from prompt2music.backends.base import (
    CancelToken,
    GeneratedAudio,
    GenerationRequest,
    GenerationResult,
    MusicBackend,
    ProgressCallback,
    Task,
)
from prompt2music.errors import CancelledError, GenerationError, OutOfMemoryError

WINDOW_SECONDS = 30.0
CONTEXT_SECONDS = 10.0
CROSSFADE_SECONDS = 0.05


def _dtype(torch: Any, precision: str, device: str) -> Any:
    if not device.startswith("cuda") and device != "mps":
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, torch.float32)


class MusicGenBackend(MusicBackend):
    required_modules = ("torch", "transformers", "numpy")

    def load(self, device: str, precision: str) -> None:
        import time

        import torch
        import transformers

        started = time.perf_counter()
        if self.model_dir is None:
            raise GenerationError("musicgen needs a local model directory")
        cls_name = self.options.get("hf_class", "MusicgenForConditionalGeneration")
        model_cls = getattr(transformers, cls_name)
        self.melody = "Melody" in cls_name
        if device.startswith("cuda") and self.options.get("allow_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.torch = torch
        self.processor = transformers.AutoProcessor.from_pretrained(str(self.model_dir))
        dtype = _dtype(torch, precision, device)
        major = int(transformers.__version__.split(".")[0])
        dtype_kw = {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}
        self.model = model_cls.from_pretrained(str(self.model_dir), **dtype_kw)
        self.model.to(device)
        self.model.eval()
        self.device = device
        enc = self.model.config.audio_encoder
        self.sample_rate = int(enc.sampling_rate)
        self.frame_rate = float(getattr(enc, "frame_rate", 0) or self.sample_rate / 640)
        self.channels = int(getattr(self.model.config, "audio_channels", 1) or 1)
        if self.options.get("torch_compile") and device.startswith("cuda"):
            self.model.decoder = torch.compile(self.model.decoder)
        self.load_seconds = time.perf_counter() - started

    def unload(self) -> None:
        for name in ("model", "processor"):
            if hasattr(self, name):
                delattr(self, name)
        super().unload()

    # -- helpers --------------------------------------------------------
    def _stopping(self, cancel: CancelToken, progress: ProgressCallback, start_len: int,
                  new_tokens: int, done: float, span: float) -> Any:
        from transformers import StoppingCriteria, StoppingCriteriaList

        outer = self

        class _Watch(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
                generated = max(0, input_ids.shape[-1] - start_len)
                frac = min(1.0, generated / max(1, new_tokens))
                if generated % 25 == 0:
                    progress(done + span * frac, "generating")
                stop = cancel.cancelled
                return outer.torch.full((input_ids.shape[0],), stop, dtype=outer.torch.bool,
                                        device=input_ids.device)

        return StoppingCriteriaList([_Watch()])

    def _gen_kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"do_sample": True,
                                  "guidance_scale": request.guidance or 3.0}
        for key in ("top_k", "top_p", "temperature"):
            if key in request.extra:
                kwargs[key] = request.extra[key]
        return kwargs

    def _run(self, texts: list[str], prompt_audio: Any | None, seconds: float,
             request: GenerationRequest, cancel: CancelToken, progress: ProgressCallback,
             done: float, span: float) -> Any:
        """Generate ``seconds`` of new audio per text. Returns float32 (batch, ch, samples)."""
        import numpy as np

        torch = self.torch
        cancel.raise_if_cancelled()
        # The codebook delay pattern costs (num_codebooks - 1) frames; ask for
        # them too and trim to the exact length afterwards.
        want = int(round(seconds * self.sample_rate))
        delay = int(getattr(self.model.decoder.config, "num_codebooks", 1)) - 1
        new_tokens = max(1, math.ceil(seconds * self.frame_rate) + delay)
        proc_kwargs: dict[str, Any] = {"text": texts, "padding": True, "return_tensors": "pt"}
        prompt_samples = 0
        if prompt_audio is not None:
            mono = prompt_audio if self.channels > 1 else prompt_audio.mean(axis=0)
            batch_audio = [mono] * len(texts)
            proc_kwargs.update(audio=batch_audio, sampling_rate=self.sample_rate)
            prompt_samples = mono.shape[-1]
        inputs = self.processor(**proc_kwargs).to(self.device)
        start_len = 0
        if "decoder_input_ids" in inputs:
            start_len = inputs["decoder_input_ids"].shape[-1]
        stopping = self._stopping(cancel, progress, start_len, new_tokens, done, span)
        try:
            with torch.inference_mode():
                audio = self.model.generate(**inputs, max_new_tokens=new_tokens,
                                            stopping_criteria=stopping,
                                            **self._gen_kwargs(request))
        except Exception as exc:
            if cancel.cancelled:  # early stop can leave generate() with a partial batch
                raise CancelledError("generation cancelled") from exc
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise OutOfMemoryError(f"CUDA out of memory: {exc}",
                                       hint="shorten the request, lower batch size or use a "
                                       "smaller model") from exc
            raise
        if cancel.cancelled:
            raise CancelledError("generation cancelled")
        out = audio.float().cpu().numpy().astype(np.float32)
        if out.ndim == 2:
            out = out[:, None, :]
        if prompt_samples and not self.melody:
            # The decoded output begins with the re-encoded prompt; keep only new audio.
            out = out[..., min(prompt_samples, out.shape[-1]):]
        return out[..., :want]

    @staticmethod
    def _crossfade_join(head: Any, tail: Any, sr: int) -> Any:
        import numpy as np

        n = min(int(CROSSFADE_SECONDS * sr), head.shape[-1], tail.shape[-1])
        if n <= 0:
            return np.concatenate([head, tail], axis=-1)
        fade = np.linspace(0.0, 1.0, n, dtype=np.float32)
        mixed = head[..., -n:] * (1 - fade) + tail[..., :n] * fade
        return np.concatenate([head[..., :-n], mixed, tail[..., n:]], axis=-1)

    @staticmethod
    def _match_loudness(reference: Any, segment: Any) -> Any:
        """Scale ``segment`` so its RMS matches ``reference`` (bounded to ±6 dB)."""
        import numpy as np

        ref = float(np.sqrt(np.mean(reference**2)) + 1e-9)
        cur = float(np.sqrt(np.mean(segment**2)) + 1e-9)
        return segment * float(np.clip(ref / cur, 0.5, 2.0))

    def _extend(self, texts: list[str], audio: Any, seconds: float,
                request: GenerationRequest, cancel: CancelToken,
                progress: ProgressCallback, done: float, span: float) -> Any:
        """Windowed continuation of ``audio`` (batch, ch, samples) by ``seconds``."""
        import numpy as np

        sr = self.sample_rate
        target = audio.shape[-1] + int(round(seconds * sr))
        overlap = CROSSFADE_SECONDS
        produced = 0.0
        step_new = WINDOW_SECONDS - CONTEXT_SECONDS
        while produced < seconds - 1e-6:
            cancel.raise_if_cancelled()
            chunk = min(step_new, seconds - produced)
            ctx = audio[..., -int(CONTEXT_SECONDS * sr):]
            pieces = []
            for b in range(len(texts)):
                piece = self._run([texts[b]], ctx[b], chunk + overlap, request, cancel, progress,
                                  done + span * produced / seconds, span * chunk / seconds)
                piece = self._match_loudness(ctx[b], piece[0])
                pieces.append(self._crossfade_join(audio[b], piece, sr))
            audio = np.stack(pieces)
            produced += chunk
        return audio[..., :target]

    # -- entry point -------------------------------------------------------
    def _generate(self, request: GenerationRequest, output_dir: Path,
                  progress: ProgressCallback, cancel: CancelToken) -> GenerationResult:
        torch = self.torch
        seed = request.seed
        if seed is None:
            seed = int(torch.seed() % (2**31))
        torch.manual_seed(seed)
        texts = [request.prompt] * request.batch_size
        progress(0.0, "preparing")
        if request.task is Task.CONTINUE:
            assert request.input_audio is not None
            source, rate = read_wav(request.input_audio)
            if rate != self.sample_rate:
                raise GenerationError(f"input must be resampled to {self.sample_rate} Hz")
            src = source if self.channels > 1 else source.mean(axis=0, keepdims=True)
            import numpy as np

            audio = np.stack([src] * request.batch_size)
            audio = self._extend(texts, audio, request.duration, request, cancel, progress,
                                 0.0, 1.0)
            if request.extra.get("only_new"):
                audio = audio[..., src.shape[-1]:]
        else:
            melody = None
            if request.melody_audio is not None:
                melody, rate = read_wav(request.melody_audio)
                if rate != self.sample_rate:
                    raise GenerationError(f"melody must be resampled to {self.sample_rate} Hz")
            first = min(request.duration, WINDOW_SECONDS)
            span = first / request.duration
            audio = self._run(texts, melody, first, request, cancel, progress, 0.0, span)
            if request.duration > first:
                audio = self._extend(texts, audio, request.duration - first, request, cancel,
                                     progress, span, 1.0 - span)
        progress(1.0, "writing audio")
        base = request.output_path or (output_dir / "musicgen.wav")
        outputs = []
        for b in range(audio.shape[0]):
            path = output_name(base, b, audio.shape[0])
            channels, frames = write_wav(path, audio[b], self.sample_rate)
            outputs.append(GeneratedAudio(path, self.sample_rate, channels,
                                          frames / self.sample_rate, seed))
        return GenerationResult(outputs, self.model_id, 0.0,
                                {"seed": seed, "frame_rate": self.frame_rate,
                                 "guidance_scale": request.guidance or 3.0})
