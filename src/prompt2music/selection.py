"""``--model auto``: choose an installed model that can genuinely do the job.

Considered, in order: task and feature support (vocals, lyrics, melody,
negative prompt, duration), license policy, installation, hardware fit (VRAM
budget / CPU capability) and whether the backend environment is ready.
Remaining candidates are ranked by the configured quality/speed preference.
An explicitly named model is never swapped for another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prompt2music import envs
from prompt2music.backends.base import GenerationRequest, check_request_against
from prompt2music.config import MusicConfig
from prompt2music.errors import ModelNotInstalledError, MusicError
from prompt2music.gpu import plan_device
from prompt2music.manager import ModelManager
from prompt2music.registry import ModelSpec, get_model, models_for_task

WEIGHTS = {"fast": (1, 2), "balanced": (1, 1), "best": (2, 1)}  # (quality, speed)


@dataclass
class Selection:
    spec: ModelSpec
    automatic: bool
    reasons: list[str] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"model": self.spec.id, "automatic": self.automatic, "reasons": self.reasons,
                "rejected": self.rejected}


def select_model(cfg: MusicConfig, request: GenerationRequest, *, explicit: str | None = None,
                 manager: ModelManager | None = None, device: str | None = None,
                 vram_limit_gb: float | None = None, quality: str | None = None,
                 check_env: bool = True) -> Selection:
    choice = explicit or cfg["general"]["default_model"]
    if choice and choice != "auto":
        return Selection(get_model(choice), automatic=False, reasons=["explicitly requested"])
    manager = manager or ModelManager(cfg)
    pref = quality or cfg["general"]["quality"]
    wq, ws = WEIGHTS.get(pref, WEIGHTS["balanced"])
    rejected: dict[str, str] = {}
    capable_uninstalled: list[ModelSpec] = []
    ranked: list[tuple[float, ModelSpec]] = []
    for spec in models_for_task(request.task):
        try:
            check_request_against(spec.id, spec.capabilities, request)
        except MusicError as exc:
            rejected[spec.id] = str(exc)
            continue
        if spec.commercial_use == "no" and not cfg["general"]["allow_noncommercial_models"]:
            rejected[spec.id] = f"license policy excludes non-commercial weights ({spec.license})"
            continue
        try:
            plan_device(cfg, spec, device=device, vram_limit_gb=vram_limit_gb)
        except MusicError as exc:
            rejected[spec.id] = str(exc)
            continue
        if not manager.is_ready(spec.id):
            capable_uninstalled.append(spec)
            rejected[spec.id] = "not installed"
            continue
        if check_env and not envs.probe(cfg, spec.backend).ok:
            rejected[spec.id] = f"backend env {spec.backend!r} not installed"
            capable_uninstalled.append(spec)
            continue
        ranked.append((wq * spec.quality + ws * spec.speed, spec))
    if not ranked:
        if capable_uninstalled:
            best = sorted(capable_uninstalled, key=lambda s: -(wq * s.quality + ws * s.speed))
            lines = [f"{s.id} (~{s.approx_size_gb:g} GB, {s.license})" for s in best[:4]]
            raise ModelNotInstalledError(
                f"no installed model can do {request.task.value} with these options",
                hint="install one of: " + "; ".join(lines)
                + " — e.g. `prompt2music models download " + best[0].id
                + "` (and `prompt2music backends install " + best[0].backend + "`)",
            )
        detail = "; ".join(f"{k}: {v}" for k, v in list(rejected.items())[:4])
        raise ModelNotInstalledError(
            f"no catalogued model supports this request ({request.task.value})",
            hint=detail or None)
    ranked.sort(key=lambda item: (-item[0], item[1].id))
    spec = ranked[0][1]
    reasons = [f"best {pref} score among installed models that support "
               f"{request.task.value}", f"quality {spec.quality}/5, speed {spec.speed}/5"]
    return Selection(spec, automatic=True, reasons=reasons, rejected=rejected)
