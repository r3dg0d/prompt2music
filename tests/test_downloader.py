"""Downloader: plans, consent, interruption/resume, corruption, atomic install."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from prompt2music import downloader
from prompt2music.downloader import execute_plan, parse_revision_pins, plan_download
from prompt2music.errors import (
    DownloadError,
    LicenseNotAcceptedError,
    ResourceLimitError,
    VerificationError,
)
from prompt2music.manager import ModelManager
from prompt2music.manifest import Manifest
from prompt2music.registry import Source, get_model


@dataclass
class Lfs:
    size: int
    sha256: str


@dataclass
class Sibling:
    rfilename: str
    size: int
    blob_id: str | None = None
    lfs: Lfs | None = None


@dataclass
class Info:
    sha: str
    siblings: list[Sibling]
    gated: bool = False
    card_data: dict | None = None


def git_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


class FakeHub:
    """In-memory Hugging Face: repos → {path: bytes}."""

    def __init__(self, repos: dict[str, dict[str, bytes]], *, sha: str = "a" * 40) -> None:
        self.repos = repos
        self.sha = sha
        self.fail_after: int | None = None
        self.corrupt: set[str] = set()
        self.downloads = 0
        self.gated: set[str] = set()

    def model_info(self, repo_id, revision, token):
        if repo_id in self.gated and not token:
            raise type("GatedRepoError", (Exception,), {})("gated")
        files = self.repos[repo_id]
        sibs = []
        for name, data in files.items():
            if name.endswith(".safetensors"):
                sibs.append(Sibling(name, len(data), lfs=Lfs(len(data),
                                                              hashlib.sha256(data).hexdigest())))
            else:
                sibs.append(Sibling(name, len(data), blob_id=git_sha1(data)))
        return Info(sha=revision if revision and len(revision) == 40 else self.sha,
                    siblings=sibs, card_data={"license": "mit"})

    def snapshot_download(self, *, repo_id, revision, allow_patterns, token, max_workers,
                          local_dir=None, cache_dir=None):
        if local_dir:
            root = Path(local_dir)
        else:
            folder = "models--" + repo_id.replace("/", "--")
            root = Path(cache_dir) / folder / "snapshots" / revision
        for name in allow_patterns:
            if self.fail_after is not None and self.downloads >= self.fail_after:
                raise ConnectionError("network dropped")
            dest = root / name
            if dest.exists():
                continue  # resume: already fetched
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = self.repos[repo_id][name]
            if name in self.corrupt:
                data = b"X" + data[1:]
            dest.write_bytes(data)
            self.downloads += 1
        return str(root)


@pytest.fixture()
def spec():
    base = get_model("musicgen-small")
    return replace(base, sources=(Source(repo_id="facebook/musicgen-small",
                                         allow_patterns=("*.json", "*.safetensors")),))


@pytest.fixture()
def hub():
    return FakeHub({"facebook/musicgen-small": {
        "config.json": b'{"a": 1}',
        "model.safetensors": os.urandom(4096),
        "tokenizer.json": b"{}",
        "README.md": b"ignored by allow_patterns",
        "modeling_custom.json.py": b"print('pwned')",
    }})


def test_plan_reports_exact_sizes_license_and_revision(cfg, spec, hub):
    plan = plan_download(spec, cfg.models_dir, hub=hub)
    data = plan.as_dict()
    assert data["total_bytes"] == 4096 + 8 + 2
    assert data["sources"][0]["revision"] == "a" * 40
    assert data["license"].startswith("CC-BY-NC")
    assert data["requires_license_acceptance"] is True
    assert not (cfg.models_dir / spec.id).exists()  # planning downloads nothing


def test_code_files_skipped_unless_trusted(cfg, spec, hub):
    loose = replace(spec, sources=(Source(repo_id="facebook/musicgen-small"),))
    plan = plan_download(loose, cfg.models_dir, hub=hub)
    assert plan.skipped_code_files == ["facebook/musicgen-small/modeling_custom.json.py"]
    assert all(not f.path.endswith(".py") for f in plan.sources[0].files)
    trusted = plan_download(loose, cfg.models_dir, hub=hub, trust_repo_code=True)
    assert any(f.path.endswith(".py") for f in trusted.sources[0].files)


def test_license_acceptance_required(cfg, spec, hub):
    plan = plan_download(spec, cfg.models_dir, hub=hub)
    with pytest.raises(LicenseNotAcceptedError):
        execute_plan(plan, hub=hub)
    assert hub.downloads == 0


def test_disk_space_checked_before_download(cfg, spec, hub, monkeypatch):
    monkeypatch.setattr(downloader, "_free_bytes", lambda _p: 1024)
    plan = plan_download(spec, cfg.models_dir, hub=hub)
    with pytest.raises(ResourceLimitError, match="disk space"):
        execute_plan(plan, hub=hub, license_accepted=True)
    assert hub.downloads == 0


def test_successful_install_is_verified_and_atomic(cfg, spec, hub):
    plan = plan_download(spec, cfg.models_dir, hub=hub)
    manifest = execute_plan(plan, hub=hub, license_accepted=True)
    dest = cfg.models_dir / spec.id
    assert (dest / "manifest.json").is_file()
    assert not plan.staging.exists()
    assert manifest.license_accepted_at
    assert set(manifest.files) == {"config.json", "model.safetensors", "tokenizer.json"}
    assert Manifest.read(dest).revisions == {"facebook/musicgen-small": "a" * 40}


def test_interrupted_download_is_not_installed_and_resumes(cfg, spec, hub, monkeypatch):
    mgr = ModelManager(cfg)
    monkeypatch.setattr("prompt2music.manager.get_model", lambda _m: spec)
    hub.fail_after = 1
    plan = plan_download(spec, cfg.models_dir, hub=hub)
    with pytest.raises(DownloadError):
        execute_plan(plan, hub=hub, license_accepted=True)
    assert mgr.installed(spec.id) is None
    assert plan.staging.exists()  # partial files kept for resume
    hub.fail_after = None
    before = hub.downloads
    plan = plan_download(spec, cfg.models_dir, hub=hub)
    assert plan.staged_bytes > 0
    execute_plan(plan, hub=hub, license_accepted=True)
    assert hub.downloads - before == 2  # only the missing files were fetched
    assert mgr.installed(spec.id) is not None


def test_corrupted_file_fails_verification_and_is_discarded(cfg, spec, hub):
    hub.corrupt = {"model.safetensors"}
    plan = plan_download(spec, cfg.models_dir, hub=hub)
    with pytest.raises(VerificationError, match="sha256 mismatch"):
        execute_plan(plan, hub=hub, license_accepted=True)
    assert not (cfg.models_dir / spec.id).exists()
    assert not (plan.staging / "model.safetensors").exists()
    hub.corrupt = set()
    execute_plan(plan_download(spec, cfg.models_dir, hub=hub), hub=hub, license_accepted=True)


def test_non_lfs_files_checked_by_git_blob_hash(cfg, spec, hub):
    hub.corrupt = {"config.json"}
    with pytest.raises(VerificationError, match="git blob"):
        execute_plan(plan_download(spec, cfg.models_dir, hub=hub), hub=hub,
                     license_accepted=True)


def test_verify_detects_post_install_corruption_and_remove(cfg, spec, hub, monkeypatch):
    monkeypatch.setattr("prompt2music.manager.get_model", lambda _m: spec)
    execute_plan(plan_download(spec, cfg.models_dir, hub=hub), hub=hub, license_accepted=True)
    mgr = ModelManager(cfg)
    assert mgr.verify(spec.id).ok
    target = cfg.models_dir / spec.id / "model.safetensors"
    target.write_bytes(b"\0" * target.stat().st_size)
    report = mgr.verify(spec.id)
    assert not report.ok and "sha256 mismatch" in report.problems[0]
    assert mgr.verify(spec.id, full=False).ok  # size-only check cannot see it
    freed = mgr.remove(spec.id)
    assert freed > 0 and mgr.installed(spec.id) is None


def test_update_swaps_atomically(cfg, spec, hub):
    execute_plan(plan_download(spec, cfg.models_dir, hub=hub), hub=hub, license_accepted=True)
    hub.sha = "b" * 40
    hub.repos["facebook/musicgen-small"]["config.json"] = b'{"a": 2}'
    execute_plan(plan_download(spec, cfg.models_dir, hub=hub), hub=hub, license_accepted=True)
    manifest = Manifest.read(cfg.models_dir / spec.id)
    assert manifest.revisions["facebook/musicgen-small"] == "b" * 40
    assert (cfg.models_dir / spec.id / "config.json").read_bytes() == b'{"a": 2}'
    assert not any((cfg.models_dir / ".trash").glob("*"))


def test_hf_cache_layout(cfg, hub):
    spec = replace(get_model("stable-audio-3-small-music"),
                   sources=(Source(repo_id="facebook/musicgen-small", layout="hf-cache",
                                   allow_patterns=("*.json",)),))
    manifest = execute_plan(plan_download(spec, cfg.models_dir, hub=hub), hub=hub,
                            license_accepted=True)
    assert all(k.startswith("hf-cache/models--facebook--musicgen-small/snapshots/")
               for k in manifest.files)


def test_gated_and_offline_errors(cfg, spec, hub):
    hub.gated = {"facebook/musicgen-small"}
    with pytest.raises(LicenseNotAcceptedError, match="gated"):
        plan_download(spec, cfg.models_dir, hub=hub)
    with pytest.raises(DownloadError, match="offline"):
        plan_download(spec, cfg.models_dir, hub=hub, offline=True)


def test_revision_pins():
    single = get_model("musicgen-small")
    assert parse_revision_pins(single, "abc") == {"facebook/musicgen-small": "abc"}
    multi = get_model("heartmula-3b")
    with pytest.raises(DownloadError):
        parse_revision_pins(multi, "abc")
    pins = parse_revision_pins(multi, "HeartMuLa/HeartMuLaGen@v1")
    assert pins == {"HeartMuLa/HeartMuLaGen": "v1"}
    with pytest.raises(DownloadError):
        parse_revision_pins(multi, "evil/repo@v1")


def test_import_local_registers_manifest(cfg, tmp_path):
    src = tmp_path / "ckpt"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "model.safetensors").write_bytes(b"w" * 100)
    mgr = ModelManager(cfg)
    with pytest.raises(LicenseNotAcceptedError):
        mgr.import_local("musicgen-small", src)
    manifest = mgr.import_local("musicgen-small", src, license_accepted=True)
    assert manifest.total_bytes == 102
    assert mgr.verify("musicgen-small").ok
    assert mgr.require("musicgen-small") == cfg.models_dir / "musicgen-small"
