"""Installed-model bookkeeping: status, removal, cache usage, readiness checks."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt2music.config import MusicConfig
from prompt2music.errors import ModelNotInstalledError, SecurityError, VerificationError
from prompt2music.manifest import Manifest, VerifyReport, dir_size, verify_model_dir
from prompt2music.registry import ModelSpec, all_models, get_model


@dataclass
class InstalledModel:
    spec: ModelSpec
    path: Path
    manifest: Manifest

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.spec.id,
            "path": str(self.path),
            "size_bytes": self.manifest.total_bytes,
            "size_gb": round(self.manifest.total_bytes / 1024**3, 2),
            "installed_at": self.manifest.installed_at,
            "revisions": self.manifest.revisions,
            "license": self.manifest.license,
            "license_accepted_at": self.manifest.license_accepted_at,
        }


class ModelManager:
    def __init__(self, cfg: MusicConfig) -> None:
        self.cfg = cfg
        self.root = cfg.models_dir

    def path_for(self, model_id: str) -> Path:
        return self.root / get_model(model_id).id

    def installed(self, model_id: str) -> InstalledModel | None:
        spec = get_model(model_id)
        if not spec.needs_download:
            return None
        path = self.root / spec.id
        manifest = Manifest.read(path)
        if manifest is None:
            return None
        return InstalledModel(spec, path, manifest)

    def is_ready(self, model_id: str) -> bool:
        spec = get_model(model_id)
        return not spec.needs_download or self.installed(spec.id) is not None

    def list_installed(self) -> list[InstalledModel]:
        out = []
        for spec in all_models():
            inst = self.installed(spec.id) if spec.needs_download else None
            if inst is not None:
                out.append(inst)
        return out

    def require(self, model_id: str) -> Path | None:
        """Model directory for a ready model (None for weightless diagnostics)."""
        spec = get_model(model_id)
        if not spec.needs_download:
            return None
        inst = self.installed(spec.id)
        if inst is None:
            raise ModelNotInstalledError(
                f"{spec.id} is not installed",
                hint=f"run `prompt2music models download {spec.id}` "
                f"(~{spec.approx_size_gb:g} GB, {spec.license})",
            )
        if self.cfg["models"]["verify_on_load"]:
            report = verify_model_dir(inst.path, inst.manifest, full=False)
            if not report.ok:
                raise VerificationError(
                    f"{spec.id} failed its size check: {'; '.join(report.problems[:3])}",
                    hint=f"run `prompt2music models verify {spec.id}` or re-download it",
                )
        return inst.path

    def verify(self, model_id: str, *, full: bool = True) -> VerifyReport:
        inst = self.installed(model_id)
        if inst is None:
            raise ModelNotInstalledError(f"{get_model(model_id).id} is not installed")
        return verify_model_dir(inst.path, inst.manifest, full=full)

    def _safe_child(self, path: Path) -> Path:
        root = self.root.resolve()
        resolved = path.resolve()
        if path.is_symlink() or root not in resolved.parents:
            raise SecurityError(f"refusing to delete {path}: not inside {root}")
        return resolved

    def remove(self, model_id: str) -> int:
        spec = get_model(model_id)
        freed = 0
        for path in (self.root / spec.id, self.root / ".staging" / spec.id):
            if path.exists():
                target = self._safe_child(path)
                freed += dir_size(target)
                shutil.rmtree(target)
        if freed == 0:
            raise ModelNotInstalledError(f"{spec.id} is not installed")
        return freed

    def cache_report(self) -> dict[str, Any]:
        staging = self.root / ".staging"
        partial = {}
        if staging.is_dir():
            for child in sorted(staging.iterdir()):
                partial[child.name] = dir_size(child)
        installed = {m.spec.id: m.manifest.total_bytes for m in self.list_installed()}
        free = shutil.disk_usage(self.root if self.root.exists() else Path.home()).free
        return {
            "models_dir": str(self.root),
            "installed_bytes": sum(installed.values()),
            "installed": installed,
            "partial_downloads": partial,
            "free_bytes": free,
        }

    def clean_partial(self) -> int:
        staging = self.root / ".staging"
        trash = self.root / ".trash"
        freed = 0
        for path in (staging, trash):
            if path.is_dir():
                freed += dir_size(path)
                shutil.rmtree(self._safe_child(path))
        return freed

    def import_local(self, model_id: str, source: Path, *, link: bool = False,
                     license_accepted: bool = False) -> Manifest:
        """Register weights already on disk (offline installs, existing checkouts).

        Files are copied (or symlinked with ``link``) into a staging directory,
        hashed into a manifest, and swapped into place like a download.
        """
        from datetime import UTC, datetime

        from prompt2music import __version__
        from prompt2music.downloader import install_staging
        from prompt2music.errors import LicenseNotAcceptedError
        from prompt2music.manifest import FileEntry, iter_model_files, sha256_file

        spec = get_model(model_id)
        if not spec.needs_download:
            raise VerificationError(f"{spec.id} has no weights to import")
        if spec.requires_license_acceptance and not license_accepted:
            raise LicenseNotAcceptedError(f"{spec.id} requires accepting {spec.license}",
                                          hint="pass --accept-license")
        source = source.expanduser().resolve()
        if not source.is_dir():
            raise ModelNotInstalledError(f"{source} is not a directory")
        staging = self.root / ".staging" / spec.id
        if staging.exists():
            shutil.rmtree(staging)
        files: dict[str, FileEntry] = {}
        for src in iter_model_files(source):
            rel = src.relative_to(source)
            dest = staging / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if link:
                dest.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dest)
            files[str(rel)] = FileEntry(size=src.stat().st_size, sha256=sha256_file(src))
        if not files:
            raise ModelNotInstalledError(f"no files found under {source}")
        manifest = Manifest(
            model_id=spec.id,
            sources=[{"repo_id": "local:" + str(source), "revision_resolved": "local",
                      "layout": spec.sources[0].layout}],
            files=files,
            license=spec.license,
            license_accepted_at=datetime.now(UTC).isoformat() if license_accepted else None,
            app_version=__version__,
        )
        manifest.write(staging)
        install_staging(staging, self.root / spec.id)
        return manifest
