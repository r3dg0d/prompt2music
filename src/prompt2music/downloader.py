"""Hugging Face model downloads: plan, consent, resumable fetch, verify, atomic install.

Flow:

1. :func:`plan_download` asks the Hub for exact file sizes, the commit each
   source resolves to, gating, and whether the repo ships Python code or
   pickle checkpoints. Nothing is downloaded yet.
2. The caller shows the plan (size, license, destination) and gets consent.
3. :func:`execute_plan` downloads into ``<models>/.staging/<id>`` pinned to the
   resolved commits. huggingface_hub resumes partial files there after an
   interruption. Every file is checked against the Hub's LFS sha256 or git
   blob id, a manifest is written, and the directory is renamed into place.

Python files in a repo are skipped unless the user explicitly trusts that
repository's code; nothing downloaded is ever imported by this process.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from prompt2music import __version__
from prompt2music.errors import (
    DownloadError,
    LicenseNotAcceptedError,
    ResourceLimitError,
    VerificationError,
)
from prompt2music.manifest import FileEntry, Manifest, git_blob_sha1, sha256_file
from prompt2music.registry import ModelSpec, Source

CODE_SUFFIXES = (".py",)
PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle")
DISK_MARGIN_BYTES = 1 * 1024**3


class Hub(Protocol):
    """The subset of huggingface_hub we use; tests substitute a fake."""

    def model_info(self, repo_id: str, revision: str | None, token: str | None) -> Any: ...

    def snapshot_download(self, **kwargs: Any) -> str: ...


class HfHub:
    def __init__(self, endpoint: str | None = None) -> None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # pragma: no cover - core dependency
            raise DownloadError("huggingface_hub is not installed") from exc
        self.endpoint = endpoint or None
        self._api = HfApi(endpoint=self.endpoint)

    def model_info(self, repo_id: str, revision: str | None, token: str | None) -> Any:
        return self._api.model_info(repo_id, revision=revision, files_metadata=True, token=token)

    def snapshot_download(self, **kwargs: Any) -> str:
        from huggingface_hub import snapshot_download

        if self.endpoint:
            kwargs.setdefault("endpoint", self.endpoint)
        return snapshot_download(**kwargs)


@dataclass
class PlannedFile:
    path: str  # repo-relative
    size: int
    sha256: str | None
    git_sha1: str | None


@dataclass
class PlannedSource:
    source: Source
    revision_resolved: str
    files: list[PlannedFile]
    gated: bool
    license: str | None

    @property
    def bytes(self) -> int:
        return sum(f.size for f in self.files)

    def local_root(self, model_dir: Path) -> Path:
        base = model_dir / self.source.subdir if self.source.subdir else model_dir
        if self.source.layout == "hf-cache":
            folder = "models--" + self.source.repo_id.replace("/", "--")
            return base / "hf-cache" / folder / "snapshots" / self.revision_resolved
        return base

    def download_kwargs(self, model_dir: Path) -> dict[str, Any]:
        base = model_dir / self.source.subdir if self.source.subdir else model_dir
        kwargs: dict[str, Any] = {
            "repo_id": self.source.repo_id,
            "revision": self.revision_resolved,
            "allow_patterns": [f.path for f in self.files],
        }
        if self.source.layout == "hf-cache":
            kwargs["cache_dir"] = str(base / "hf-cache")
        else:
            kwargs["local_dir"] = str(base)
        return kwargs


@dataclass
class DownloadPlan:
    spec: ModelSpec
    sources: list[PlannedSource]
    destination: Path
    staging: Path
    free_bytes: int
    staged_bytes: int
    code_files: list[str] = field(default_factory=list)
    pickle_files: list[str] = field(default_factory=list)
    skipped_code_files: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(s.bytes for s in self.sources)

    @property
    def remaining_bytes(self) -> int:
        return max(0, self.total_bytes - self.staged_bytes)

    @property
    def gated(self) -> bool:
        return self.spec.gated or any(s.gated for s in self.sources)

    def disk_ok(self) -> bool:
        return self.free_bytes >= self.remaining_bytes + DISK_MARGIN_BYTES

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.spec.id,
            "destination": str(self.destination),
            "total_bytes": self.total_bytes,
            "total_gb": round(self.total_bytes / 1024**3, 2),
            "already_staged_bytes": self.staged_bytes,
            "free_bytes": self.free_bytes,
            "disk_ok": self.disk_ok(),
            "license": self.spec.license,
            "license_url": self.spec.license_url,
            "commercial_use": self.spec.commercial_use,
            "gated": self.gated,
            "requires_license_acceptance": self.spec.requires_license_acceptance,
            "sources": [
                {
                    "repo_id": s.source.repo_id,
                    "revision": s.revision_resolved,
                    "files": len(s.files),
                    "bytes": s.bytes,
                    "hub_license": s.license,
                }
                for s in self.sources
            ],
            "code_files": self.code_files,
            "skipped_code_files": self.skipped_code_files,
            "pickle_files": self.pickle_files,
        }


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    return not patterns or any(fnmatch.fnmatch(path, p) for p in patterns)


def _free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _translate_hub_error(exc: Exception, repo_id: str) -> Exception:
    name = type(exc).__name__
    if name == "GatedRepoError":
        return LicenseNotAcceptedError(
            f"{repo_id} is gated on Hugging Face",
            hint=f"accept the license at https://huggingface.co/{repo_id}, then export "
            "HF_TOKEN (or run `hf auth login`)",
        )
    if name in {"RepositoryNotFoundError", "RevisionNotFoundError"}:
        return DownloadError(f"{repo_id}: {exc}", hint="check the revision or your token")
    return DownloadError(f"{repo_id}: {exc}",
                         hint="check network access to huggingface.co (or set "
                         "models.hf_endpoint to a permitted mirror)")


def parse_revision_pins(spec: ModelSpec, revision: str | None) -> dict[str, str]:
    """``REV`` pins a single-repo model; ``org/repo@REV[,org/repo@REV]`` pins any."""
    if not revision:
        return {}
    repos = {s.repo_id for s in spec.sources}
    if "@" not in revision:
        if len(repos) != 1:
            raise DownloadError(
                f"{spec.id} combines {len(repos)} repositories; pin with "
                "org/repo@revision (comma-separated)")
        return {next(iter(repos)): revision}
    pins: dict[str, str] = {}
    for part in revision.split(","):
        repo, _, rev = part.strip().partition("@")
        if repo not in repos or not rev:
            raise DownloadError(f"invalid pin {part!r}; repositories: {', '.join(sorted(repos))}")
        pins[repo] = rev
    return pins


def plan_download(spec: ModelSpec, models_dir: Path, *, hub: Hub, token: str | None = None,
                  revision: str | None = None, trust_repo_code: bool = False,
                  offline: bool = False) -> DownloadPlan:
    if not spec.sources:
        raise DownloadError(f"{spec.id} has nothing to download")
    if offline:
        raise DownloadError("offline mode is enabled; downloads are disabled",
                            hint="unset general.offline / PROMPT2MUSIC_OFFLINE")
    planned: list[PlannedSource] = []
    code_files: list[str] = []
    skipped: list[str] = []
    pickles: list[str] = []
    pins = parse_revision_pins(spec, revision)
    for source in spec.sources:
        rev = pins.get(source.repo_id, source.revision)
        try:
            info = hub.model_info(source.repo_id, rev, token)
        except Exception as exc:
            raise _translate_hub_error(exc, source.repo_id) from exc
        files: list[PlannedFile] = []
        for sib in info.siblings or []:
            name = sib.rfilename
            if not _matches(name, source.allow_patterns):
                continue
            if name.endswith(CODE_SUFFIXES):
                code_files.append(f"{source.repo_id}/{name}")
                if not trust_repo_code:
                    skipped.append(f"{source.repo_id}/{name}")
                    continue
            if name.endswith(PICKLE_SUFFIXES):
                pickles.append(f"{source.repo_id}/{name}")
            lfs = getattr(sib, "lfs", None)
            size = sib.size if sib.size is not None else (lfs.size if lfs else 0)
            files.append(PlannedFile(
                path=name,
                size=int(size or 0),
                sha256=getattr(lfs, "sha256", None) if lfs else None,
                git_sha1=None if lfs else getattr(sib, "blob_id", None),
            ))
        if not files:
            raise DownloadError(f"{source.repo_id}@{rev}: no files matched "
                                f"{list(source.allow_patterns) or '*'}")
        card = getattr(info, "card_data", None)
        hub_license = None
        if card is not None:
            hub_license = card.get("license") if hasattr(card, "get") else None
        planned.append(PlannedSource(
            source=source,
            revision_resolved=info.sha or rev or "main",
            files=files,
            gated=bool(getattr(info, "gated", False)),
            license=hub_license,
        ))
    destination = models_dir / spec.id
    staging = models_dir / ".staging" / spec.id
    staged = 0
    if staging.exists():
        from prompt2music.manifest import dir_size

        staged = dir_size(staging)
    return DownloadPlan(
        spec=spec,
        sources=planned,
        destination=destination,
        staging=staging,
        free_bytes=_free_bytes(models_dir),
        staged_bytes=staged,
        code_files=code_files,
        pickle_files=pickles,
        skipped_code_files=skipped,
    )


def _verify_staged(plan: DownloadPlan, progress: Callable[[str], None]) -> dict[str, FileEntry]:
    entries: dict[str, FileEntry] = {}
    problems: list[str] = []
    for ps in plan.sources:
        root = ps.local_root(plan.staging)
        for pf in ps.files:
            path = root / pf.path
            rel = str(path.relative_to(plan.staging))
            if not path.is_file():
                problems.append(f"missing after download: {rel}")
                continue
            size = path.stat().st_size
            if pf.size and size != pf.size:
                problems.append(f"size mismatch: {rel} ({size} != {pf.size})")
                continue
            progress(f"verifying {rel}")
            digest = sha256_file(path)
            if pf.sha256 and digest != pf.sha256:
                problems.append(f"sha256 mismatch: {rel}")
                continue
            git_sha = None
            if not pf.sha256 and pf.git_sha1:
                git_sha = git_blob_sha1(path)
                if git_sha != pf.git_sha1:
                    problems.append(f"git blob hash mismatch: {rel}")
                    continue
            entries[rel] = FileEntry(size=size, sha256=digest, git_sha1=git_sha)
    if problems:
        raise VerificationError(
            "downloaded files failed verification: " + "; ".join(problems[:5]),
            hint="re-run the download; corrupt partial files are discarded",
        )
    return entries


def _discard_corrupt(plan: DownloadPlan, err: VerificationError) -> None:
    """Remove files named in a failed verification so a retry refetches them."""
    for ps in plan.sources:
        root = ps.local_root(plan.staging)
        for pf in ps.files:
            path = root / pf.path
            rel = str(path.relative_to(plan.staging))
            if rel in str(err) and path.exists():
                target = path.resolve() if path.is_symlink() else path
                target.unlink(missing_ok=True)
                path.unlink(missing_ok=True)


def install_staging(staging: Path, destination: Path) -> None:
    """Swap the verified staging directory into place atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    old: Path | None = None
    if destination.exists():
        old = destination.parent / ".trash" / f"{destination.name}-{int(time.time())}"
        old.parent.mkdir(parents=True, exist_ok=True)
        os.replace(destination, old)
    try:
        os.replace(staging, destination)
    except OSError:
        if old is not None:
            os.replace(old, destination)
        raise
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)


def execute_plan(plan: DownloadPlan, *, hub: Hub, token: str | None = None,
                 license_accepted: bool = False, trust_repo_code: bool = False,
                 max_workers: int = 8, quiet: bool = False,
                 progress: Callable[[str], None] | None = None) -> Manifest:
    progress = progress or (lambda _msg: None)
    spec = plan.spec
    if spec.requires_license_acceptance and not license_accepted:
        raise LicenseNotAcceptedError(
            f"{spec.id} is distributed under {spec.license}; acceptance is required",
            hint=f"review {spec.license_url} and pass --accept-license",
        )
    if not plan.disk_ok():
        need = (plan.remaining_bytes + DISK_MARGIN_BYTES) / 1024**3
        raise ResourceLimitError(
            f"not enough free disk space: need ~{need:.1f} GB, have "
            f"{plan.free_bytes / 1024**3:.1f} GB under {plan.destination.parent}",
            hint="free space or set models.directory to a larger disk",
        )
    plan.staging.mkdir(parents=True, exist_ok=True)
    if quiet:
        try:
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
        except ImportError:  # pragma: no cover
            pass
    for ps in plan.sources:
        progress(f"downloading {ps.source.repo_id}@{ps.revision_resolved[:12]} "
                 f"({ps.bytes / 1024**3:.2f} GB)")
        try:
            hub.snapshot_download(token=token, max_workers=max_workers,
                                  **ps.download_kwargs(plan.staging))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            raise _translate_hub_error(exc, ps.source.repo_id) from exc
    try:
        entries = _verify_staged(plan, progress)
    except VerificationError as err:
        _discard_corrupt(plan, err)
        raise
    now = datetime.now(UTC).isoformat()
    manifest = Manifest(
        model_id=spec.id,
        sources=[{**ps.source.as_dict(), "revision_resolved": ps.revision_resolved}
                 for ps in plan.sources],
        files=entries,
        license=spec.license,
        license_accepted_at=now if license_accepted else None,
        trusted_repo_code=trust_repo_code,
        app_version=__version__,
    )
    manifest.write(plan.staging)
    install_staging(plan.staging, plan.destination)
    progress(f"installed {spec.id} → {plan.destination}")
    return manifest
