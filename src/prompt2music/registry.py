"""Catalog of supported music checkpoints.

Every entry records where its weights come from, the license that governs
them, and the operations the upstream inference code supports. Facts were
taken from upstream READMEs and source on 2026-10-18; ``verification``
says how far each integration has been exercised. Nothing here downloads.

Sizes are upstream-reported or estimated from parameter counts; the
downloader always shows the exact byte count from the Hub before it starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from prompt2music.backends.base import Capabilities, Task
from prompt2music.errors import UnknownModelError

Layout = Literal["local-dir", "hf-cache"]
Commercial = Literal["yes", "no", "conditional"]
Verification = Literal["contract-tested", "api-documented", "diagnostic"]
Kind = Literal["music", "stems"]


@dataclass(frozen=True)
class Source:
    """One Hugging Face repository that contributes files to an installed model."""

    repo_id: str
    revision: str = "main"
    allow_patterns: tuple[str, ...] = ()
    subdir: str = ""
    layout: Layout = "local-dir"

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "allow_patterns": list(self.allow_patterns),
            "subdir": self.subdir,
            "layout": self.layout,
        }


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    family: str
    backend: str
    description: str
    capabilities: Capabilities
    sources: tuple[Source, ...]
    approx_size_gb: float
    min_vram_gb: float
    recommended_vram_gb: float
    license: str
    license_url: str
    commercial_use: Commercial
    gated: bool
    requires_license_acceptance: bool
    upstream: str
    cpu_capable: bool = False
    quality: int = 3  # 1..5, relative and subjective; used only for --model auto
    speed: int = 3  # 1..5, higher is faster
    verification: Verification = "api-documented"
    notes: tuple[str, ...] = ()
    unsupported: tuple[str, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    diagnostic: bool = False
    kind: Kind = "music"

    @property
    def tasks(self) -> frozenset[Task]:
        return self.capabilities.tasks

    @property
    def needs_download(self) -> bool:
        return bool(self.sources)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "backend": self.backend,
            "description": self.description,
            "capabilities": self.capabilities.as_dict(),
            "sources": [s.as_dict() for s in self.sources],
            "approx_size_gb": self.approx_size_gb,
            "min_vram_gb": self.min_vram_gb,
            "recommended_vram_gb": self.recommended_vram_gb,
            "cpu_capable": self.cpu_capable,
            "license": self.license,
            "license_url": self.license_url,
            "commercial_use": self.commercial_use,
            "gated": self.gated,
            "requires_license_acceptance": self.requires_license_acceptance,
            "upstream": self.upstream,
            "verification": self.verification,
            "notes": list(self.notes),
            "unsupported": list(self.unsupported),
            "aliases": list(self.aliases),
            "diagnostic": self.diagnostic,
            "kind": self.kind,
        }


_T = Task

_ACE_LICENSE = "MIT"
_ACE_URL = "https://github.com/ace-step/ACE-Step-1.5/blob/main/LICENSE"
_ACE_MAIN = Source(
    repo_id="ACE-Step/Ace-Step1.5",
    allow_patterns=(
        "acestep-v15-turbo/*",
        "vae/*",
        "Qwen3-Embedding-0.6B/*",
        "acestep-5Hz-lm-1.7B/*",
    ),
    subdir="checkpoints",
)
_ACE_CAPS: dict[str, Any] = dict(
    vocals=True,
    lyrics=True,
    instrumental=True,
    bpm=True,
    key=True,
    time_signature=True,
    reference_audio=True,
    batch=True,
    min_duration=10.0,
    max_duration=600.0,
    sample_rate=48000,
    channels=2,
    lyrics_format="section-tags",
)
_ACE_UNSUPPORTED = (
    "continuation: upstream documents no 'extend' task, so `continue` is refused",
    "voice cloning: not offered by upstream",
    "lego/extract/complete tasks need the base checkpoint and are not wired yet",
)

_SA_URL = "https://stability.ai/license"
_SA_LICENSE = "Stability AI Community License"
_SA_NOTES = (
    "Free for research and for organisations under the Community License revenue "
    "threshold; larger commercial use needs an Enterprise license from Stability AI.",
    "Hugging Face may require accepting the license on the model page before download.",
)

_MG_LICENSE = "CC-BY-NC-4.0 (weights); MIT (code)"
_MG_URL = "https://huggingface.co/facebook/musicgen-small"
_MG_CAPS: dict[str, Any] = dict(
    vocals=False,
    lyrics=False,
    instrumental=True,
    negative_prompt=False,
    batch=True,
    cooperative_cancel=True,
    min_duration=1.0,
    max_duration=300.0,
    sample_rate=32000,
)
_MG_NOTES = (
    "Native window is 30 s; longer requests use windowed continuation, where each "
    "new window is conditioned on the last 10 s of generated audio (generative, "
    "not concatenation).",
    "BPM and key are not separate inputs; they are written into the text prompt.",
)
_MG_SOURCE_PATTERNS = ("*.json", "*.safetensors", "*.model", "*.txt")

MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="ace-step-1.5",
        name="ACE-Step 1.5 (turbo + 1.7B LM)",
        family="ACE-Step",
        backend="ace-step",
        description="Fast full-song model: instrumentals and vocal songs with lyrics, "
        "cover/style transfer and repaint. Default upstream bundle.",
        capabilities=Capabilities(
            tasks=frozenset({_T.TEXT_TO_MUSIC, _T.SONG, _T.EDIT, _T.INPAINT}), **_ACE_CAPS
        ),
        sources=(_ACE_MAIN,),
        approx_size_gb=10.0,
        min_vram_gb=6.0,
        recommended_vram_gb=12.0,
        cpu_capable=True,
        license=_ACE_LICENSE,
        license_url=_ACE_URL,
        commercial_use="yes",
        gated=False,
        requires_license_acceptance=False,
        upstream="https://github.com/ace-step/ACE-Step-1.5",
        quality=4,
        speed=5,
        notes=(
            "Upstream VRAM tiers: <=6 GB runs turbo with INT8 + CPU offload and the LM "
            "disabled; 8-16 GB fits turbo with the 0.6B/1.7B LM.",
            "Upstream pins torch 2.10 and transformers<4.58, so it needs its own env.",
        ),
        unsupported=_ACE_UNSUPPORTED,
        options={"config_path": "acestep-v15-turbo", "lm_model": "acestep-5Hz-lm-1.7B",
                 "lm_backend": "pt", "inference_steps": 8},
        aliases=("ace-step", "acestep", "ace-step-1.5-turbo"),
    ),
    ModelSpec(
        id="ace-step-1.5-xl-turbo",
        name="ACE-Step 1.5 XL turbo",
        family="ACE-Step",
        backend="ace-step",
        description="Larger ACE-Step DiT for higher quality on 20-24 GB GPUs.",
        capabilities=Capabilities(
            tasks=frozenset({_T.TEXT_TO_MUSIC, _T.SONG, _T.EDIT, _T.INPAINT}), **_ACE_CAPS
        ),
        sources=(
            Source(
                repo_id="ACE-Step/Ace-Step1.5",
                allow_patterns=("vae/*", "Qwen3-Embedding-0.6B/*", "acestep-5Hz-lm-1.7B/*"),
                subdir="checkpoints",
            ),
            Source(repo_id="ACE-Step/acestep-v15-xl-turbo",
                   subdir="checkpoints/acestep-v15-xl-turbo"),
        ),
        approx_size_gb=15.0,
        min_vram_gb=12.0,
        recommended_vram_gb=20.0,
        license=_ACE_LICENSE,
        license_url=_ACE_URL,
        commercial_use="yes",
        gated=False,
        requires_license_acceptance=False,
        upstream="https://github.com/ace-step/ACE-Step-1.5",
        quality=5,
        speed=4,
        notes=("Upstream: XL needs >=12 GB with offload+quantization, >=20 GB without.",),
        unsupported=_ACE_UNSUPPORTED,
        options={"config_path": "acestep-v15-xl-turbo", "lm_model": "acestep-5Hz-lm-1.7B",
                 "lm_backend": "pt", "inference_steps": 8},
    ),
    ModelSpec(
        id="stable-audio-3-small-music",
        name="Stable Audio 3 Small (music)",
        family="Stable Audio 3",
        backend="stable-audio",
        description="433M-parameter instrumental model that runs on CPU; supports "
        "audio-to-audio, inpainting and continuation.",
        capabilities=Capabilities(
            tasks=frozenset({_T.TEXT_TO_MUSIC, _T.EDIT, _T.INPAINT, _T.CONTINUE}),
            instrumental=True,
            negative_prompt=True,
            batch=True,
            min_duration=1.0,
            max_duration=120.0,
            sample_rate=44100,
            channels=2,
        ),
        sources=(Source(repo_id="stabilityai/stable-audio-3-small-music", layout="hf-cache"),),
        approx_size_gb=2.0,
        min_vram_gb=2.5,
        recommended_vram_gb=4.0,
        cpu_capable=True,
        license=_SA_LICENSE,
        license_url=_SA_URL,
        commercial_use="conditional",
        gated=True,
        requires_license_acceptance=True,
        upstream="https://github.com/Stability-AI/stable-audio-3",
        quality=3,
        speed=5,
        notes=_SA_NOTES + ("Upstream peak VRAM on H200: 1.9 GB for 30 s.",),
        unsupported=("vocals/lyrics: instrumental and sound model only",),
        options={"upstream_name": "small-music"},
        aliases=("stable-audio-3-small", "sa3-small"),
    ),
    ModelSpec(
        id="stable-audio-3-medium",
        name="Stable Audio 3 Medium",
        family="Stable Audio 3",
        backend="stable-audio",
        description="1.4B-parameter text-to-audio model, up to 380 s, stereo 44.1 kHz. "
        "Requires CUDA with Flash Attention 2.",
        capabilities=Capabilities(
            tasks=frozenset({_T.TEXT_TO_MUSIC, _T.EDIT, _T.INPAINT, _T.CONTINUE}),
            instrumental=True,
            negative_prompt=True,
            batch=True,
            min_duration=1.0,
            max_duration=380.0,
            sample_rate=44100,
            channels=2,
        ),
        sources=(Source(repo_id="stabilityai/stable-audio-3-medium", layout="hf-cache"),),
        approx_size_gb=6.0,
        min_vram_gb=6.5,
        recommended_vram_gb=8.0,
        license=_SA_LICENSE,
        license_url=_SA_URL,
        commercial_use="conditional",
        gated=True,
        requires_license_acceptance=True,
        upstream="https://github.com/Stability-AI/stable-audio-3",
        quality=4,
        speed=4,
        notes=_SA_NOTES
        + ("Upstream peak VRAM on H200: 5.5 GB for 30 s, 6.5 GB for 380 s.",
           "Stable Audio 3 Large is API-only and deliberately not offered."),
        unsupported=("vocals/lyrics", "CPU inference (upstream requires CUDA + flash-attn)"),
        options={"upstream_name": "medium"},
        aliases=("stable-audio-3", "sa3-medium"),
    ),
    ModelSpec(
        id="musicgen-small",
        name="MusicGen Small (300M)",
        family="MusicGen",
        backend="musicgen",
        description="Small text-to-music model via Hugging Face transformers. Runs on CPU.",
        capabilities=Capabilities(tasks=frozenset({_T.TEXT_TO_MUSIC, _T.CONTINUE}), channels=1,
                                  **_MG_CAPS),
        sources=(Source(repo_id="facebook/musicgen-small",
                        allow_patterns=_MG_SOURCE_PATTERNS),),
        approx_size_gb=2.4,
        min_vram_gb=2.0,
        recommended_vram_gb=4.0,
        cpu_capable=True,
        license=_MG_LICENSE,
        license_url=_MG_URL,
        commercial_use="no",
        gated=False,
        requires_license_acceptance=True,
        upstream="https://github.com/facebookresearch/audiocraft",
        quality=2,
        speed=4,
        verification="contract-tested",
        notes=_MG_NOTES,
        unsupported=("vocals/lyrics", "inpainting", "audio-to-audio editing"),
        options={"hf_class": "MusicgenForConditionalGeneration"},
        aliases=("musicgen",),
    ),
    ModelSpec(
        id="musicgen-medium",
        name="MusicGen Medium (1.5B)",
        family="MusicGen",
        backend="musicgen",
        description="Medium text-to-music model via transformers.",
        capabilities=Capabilities(tasks=frozenset({_T.TEXT_TO_MUSIC, _T.CONTINUE}), channels=1,
                                  **_MG_CAPS),
        sources=(Source(repo_id="facebook/musicgen-medium",
                        allow_patterns=_MG_SOURCE_PATTERNS),),
        approx_size_gb=8.0,
        min_vram_gb=6.0,
        recommended_vram_gb=10.0,
        license=_MG_LICENSE,
        license_url="https://huggingface.co/facebook/musicgen-medium",
        commercial_use="no",
        gated=False,
        requires_license_acceptance=True,
        upstream="https://github.com/facebookresearch/audiocraft",
        quality=3,
        speed=3,
        verification="contract-tested",
        notes=_MG_NOTES,
        unsupported=("vocals/lyrics", "inpainting", "audio-to-audio editing"),
        options={"hf_class": "MusicgenForConditionalGeneration"},
    ),
    ModelSpec(
        id="musicgen-large",
        name="MusicGen Large (3.3B)",
        family="MusicGen",
        backend="musicgen",
        description="Largest MusicGen text-to-music checkpoint via transformers.",
        capabilities=Capabilities(tasks=frozenset({_T.TEXT_TO_MUSIC, _T.CONTINUE}), channels=1,
                                  **_MG_CAPS),
        sources=(Source(repo_id="facebook/musicgen-large",
                        allow_patterns=_MG_SOURCE_PATTERNS),),
        approx_size_gb=13.0,
        min_vram_gb=10.0,
        recommended_vram_gb=16.0,
        license=_MG_LICENSE,
        license_url="https://huggingface.co/facebook/musicgen-large",
        commercial_use="no",
        gated=False,
        requires_license_acceptance=True,
        upstream="https://github.com/facebookresearch/audiocraft",
        quality=3,
        speed=2,
        verification="contract-tested",
        notes=_MG_NOTES,
        unsupported=("vocals/lyrics", "inpainting", "audio-to-audio editing"),
        options={"hf_class": "MusicgenForConditionalGeneration"},
    ),
    ModelSpec(
        id="musicgen-melody",
        name="MusicGen Melody (1.5B)",
        family="MusicGen",
        backend="musicgen",
        description="Text-to-music with optional melody conditioning from a reference "
        "recording (chromagram).",
        capabilities=Capabilities(
            tasks=frozenset({_T.TEXT_TO_MUSIC}),
            melody_conditioning=True,
            channels=1,
            **{**_MG_CAPS, "max_duration": 30.0},
        ),
        sources=(Source(repo_id="facebook/musicgen-melody",
                        allow_patterns=_MG_SOURCE_PATTERNS),),
        approx_size_gb=6.5,
        min_vram_gb=6.0,
        recommended_vram_gb=10.0,
        license=_MG_LICENSE,
        license_url="https://huggingface.co/facebook/musicgen-melody",
        commercial_use="no",
        gated=False,
        requires_license_acceptance=True,
        upstream="https://github.com/facebookresearch/audiocraft",
        quality=3,
        speed=3,
        notes=(
            "Uses transformers MusicgenMelodyForConditionalGeneration; melody "
            "conditioning is a chromagram of the --melody file.",
            "Limited to one 30 s window; windowed continuation is not wired for "
            "the melody variant.",
        ),
        unsupported=("vocals/lyrics", "continuation beyond 30 s", "inpainting"),
        options={"hf_class": "MusicgenMelodyForConditionalGeneration"},
    ),
    ModelSpec(
        id="heartmula-3b",
        name="HeartMuLa oss-3B (happy-new-year)",
        family="HeartMuLa",
        backend="heartmula",
        description="Lyrics + tags conditioned song model with vocals (HeartMuLa + "
        "HeartCodec, 48 kHz stereo).",
        capabilities=Capabilities(
            tasks=frozenset({_T.SONG, _T.TEXT_TO_MUSIC}),
            vocals=True,
            lyrics=True,
            instrumental=True,
            seed=False,
            min_duration=10.0,
            max_duration=240.0,
            sample_rate=48000,
            channels=2,
            lyrics_format="section-tags",
        ),
        sources=(
            Source(repo_id="HeartMuLa/HeartMuLaGen"),
            Source(repo_id="HeartMuLa/HeartMuLa-oss-3B-happy-new-year",
                   subdir="HeartMuLa-oss-3B"),
            Source(repo_id="HeartMuLa/HeartCodec-oss-20260123", subdir="HeartCodec-oss"),
        ),
        approx_size_gb=16.0,
        min_vram_gb=8.0,
        recommended_vram_gb=16.0,
        license="Apache-2.0",
        license_url="https://github.com/HeartMuLa/heartlib",
        commercial_use="yes",
        gated=False,
        requires_license_acceptance=False,
        upstream="https://github.com/HeartMuLa/heartlib",
        quality=4,
        speed=2,
        notes=(
            "Weights relicensed to Apache-2.0 upstream on 2026-01-20.",
            "Uses lazy loading by default to fit 16-24 GB GPUs.",
            "The 7B model is not publicly released and is not offered.",
            "Instrumental requests are sent with an [Instrumental] lyric sheet; "
            "upstream does not document a dedicated instrumental mode.",
        ),
        unsupported=("reference-audio conditioning (upstream TODO)", "seeded generation",
                     "continuation", "inpainting"),
        options={"version": "3B", "lazy_load": True},
        aliases=("heartmula",),
    ),
    ModelSpec(
        id="yue2-3b",
        name="YuE2 3B",
        family="YuE",
        backend="yue",
        description="Lyrics-to-song model that plans melody/chords (ABC score) then "
        "renders a full song at 48 kHz stereo.",
        capabilities=Capabilities(
            tasks=frozenset({_T.SONG}),
            vocals=True,
            lyrics=True,
            instrumental=False,
            min_duration=30.0,
            max_duration=360.0,
            sample_rate=48000,
            channels=2,
            lyrics_format="section-tags",
        ),
        sources=(
            Source(repo_id="m-a-p/YuE2-3B", layout="hf-cache"),
            Source(repo_id="m-a-p/YuE2-Vae", layout="hf-cache"),
        ),
        approx_size_gb=10.0,
        min_vram_gb=24.0,
        recommended_vram_gb=24.0,
        license="CC-BY-NC-4.0 (weights, with additional creator permission); Apache-2.0 (code)",
        license_url="https://github.com/multimodal-art-projection/YuE",
        commercial_use="no",
        gated=False,
        requires_license_acceptance=True,
        upstream="https://github.com/multimodal-art-projection/YuE",
        quality=5,
        speed=1,
        notes=(
            "Upstream hardware requirement: NVIDIA GPU with BF16 and 24 GB VRAM. On a "
            "24 GB card close other GPU applications first.",
            "Song length is decided by the lyrics/plan; --duration is advisory.",
            "Cover/score-editing workflows exist upstream but are not wired here yet.",
        ),
        unsupported=("instrumental-only generation", "continuation", "inpainting"),
        options={"repo_id": "m-a-p/YuE2-3B", "cot": "full"},
        aliases=("yue2", "yue"),
    ),
    ModelSpec(
        id="test-tone",
        name="Diagnostic test tone (not a music model)",
        family="diagnostic",
        backend="tone",
        description="Synthesises a deterministic chord to test the worker, job, library "
        "and MCP plumbing without downloading weights. Never chosen by --model auto.",
        capabilities=Capabilities(
            tasks=frozenset({_T.TEXT_TO_MUSIC}),
            batch=True,
            cooperative_cancel=True,
            min_duration=0.1,
            max_duration=600.0,
            sample_rate=44100,
            channels=2,
        ),
        sources=(),
        approx_size_gb=0.0,
        min_vram_gb=0.0,
        recommended_vram_gb=0.0,
        cpu_capable=True,
        license="MIT (part of ai-media-cli)",
        license_url="https://github.com/r3dg0d/ai-media-cli/blob/main/LICENSE",
        commercial_use="yes",
        gated=False,
        requires_license_acceptance=False,
        upstream="ai-media-cli",
        quality=0,
        speed=5,
        verification="diagnostic",
        notes=("Output is a sine chord, not generated music.",),
        diagnostic=True,
    ),
)

def _demucs(model_id: str, name: str, repo: str, stems: str, size: float, quality: int,
            speed: int, notes: tuple[str, ...] = ()) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        name=name,
        family="Demucs",
        backend="demucs",
        description=f"Source separation into {stems}.",
        capabilities=Capabilities(tasks=frozenset(), instrumental=True, seed=False,
                                  max_duration=3600.0, sample_rate=44100, channels=2),
        sources=(Source(repo_id=repo, layout="hf-cache",
                        allow_patterns=("*.yaml", "*.safetensors")),),
        approx_size_gb=size,
        min_vram_gb=2.0,
        recommended_vram_gb=4.0,
        cpu_capable=True,
        license="MIT",
        license_url="https://github.com/adefossez/demucs/blob/main/LICENSE",
        commercial_use="yes",
        gated=False,
        requires_license_acceptance=False,
        upstream="https://github.com/adefossez/demucs",
        quality=quality,
        speed=speed,
        notes=("Weights are safetensors on the Hub (demucs>=4.1 layout).", *notes),
        verification="contract-tested",
        options={"demucs_name": model_id},
        kind="stems",
    )


STEM_MODELS: tuple[ModelSpec, ...] = (
    _demucs("htdemucs", "Hybrid Transformer Demucs", "adefossez/HTDemucs",
            "vocals, drums, bass, other", 0.1, 4, 4),
    _demucs("htdemucs_ft", "Hybrid Transformer Demucs (fine-tuned)", "adefossez/HTDemucs-ft",
            "vocals, drums, bass, other", 0.35, 5, 2, ("About 4x slower than htdemucs.",)),
    _demucs("htdemucs_6s", "Hybrid Transformer Demucs 6-stem", "adefossez/HTDemucs-6s",
            "vocals, drums, bass, guitar, piano, other", 0.1, 3, 4,
            ("Upstream notes the piano stem is of limited quality.",)),
    _demucs("mdx_extra", "Demucs MDX extra", "adefossez/Demucs-mdx_extra",
            "vocals, drums, bass, other", 0.6, 4, 3),
)

_BY_ID: dict[str, ModelSpec] = {}
for _spec in (*MODELS, *STEM_MODELS):
    for _key in (_spec.id, *_spec.aliases):
        if _key in _BY_ID:
            raise RuntimeError(f"duplicate model id/alias {_key}")
        _BY_ID[_key] = _spec


def get_model(model_id: str) -> ModelSpec:
    spec = _BY_ID.get(model_id.strip().lower())
    if spec is None:
        known = ", ".join(s.id for s in MODELS)
        raise UnknownModelError(f"unknown model {model_id!r}", hint=f"known models: {known}")
    return spec


def all_models(*, include_diagnostic: bool = True, kind: Kind | None = None) -> list[ModelSpec]:
    """Music models by default plus stem models when ``kind`` is None."""
    pool = (*MODELS, *STEM_MODELS)
    return [m for m in pool
            if (include_diagnostic or not m.diagnostic) and (kind is None or m.kind == kind)]


def models_for_task(task: Task) -> list[ModelSpec]:
    return [m for m in MODELS if task in m.tasks and not m.diagnostic]


def search_models(query: str) -> list[ModelSpec]:
    """Case-insensitive match over ids, names, families, descriptions and tasks."""
    terms = [t for t in query.lower().split() if t]
    hits = []
    for spec in (*MODELS, *STEM_MODELS):
        hay = " ".join(
            [spec.id, spec.name, spec.family, spec.description, spec.license,
             *spec.aliases, *(t.value for t in spec.tasks),
             "vocals" if spec.capabilities.vocals else "instrumental"]
        ).lower()
        if all(term in hay for term in terms):
            hits.append(spec)
    return hits
