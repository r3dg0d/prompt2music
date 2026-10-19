"""Typed errors shared by the CLI, MCP server, job daemon, and backend workers."""

from __future__ import annotations


class MusicError(Exception):
    """Base class. ``code`` is stable and safe to return to MCP clients."""

    code = "error"
    exit_code = 1

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint

    def as_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"code": self.code, "message": str(self)}
        if self.hint:
            data["hint"] = self.hint
        return data


class ConfigError(MusicError):
    code = "config_error"
    exit_code = 2


class UnknownModelError(MusicError):
    code = "unknown_model"
    exit_code = 2


class ModelNotInstalledError(MusicError):
    code = "model_not_installed"
    exit_code = 2


class BackendNotReadyError(MusicError):
    """Backend Python environment is missing or lacks its dependencies."""

    code = "backend_not_ready"
    exit_code = 2


class UnsupportedTaskError(MusicError):
    """The selected model does not genuinely support the requested operation."""

    code = "unsupported_task"
    exit_code = 2


class InvalidRequestError(MusicError):
    code = "invalid_request"
    exit_code = 2


class ResourceLimitError(MusicError):
    """VRAM/RAM/disk/quota limits would be exceeded."""

    code = "resource_limit"
    exit_code = 2


class OutOfMemoryError(ResourceLimitError):
    code = "out_of_memory"


class DownloadError(MusicError):
    code = "download_error"


class LicenseNotAcceptedError(MusicError):
    code = "license_not_accepted"
    exit_code = 2


class VerificationError(MusicError):
    code = "verification_failed"


class GenerationError(MusicError):
    code = "generation_failed"


class CancelledError(MusicError):
    code = "cancelled"
    exit_code = 130


class SecurityError(MusicError):
    """Rejected by an MCP/path/policy guard."""

    code = "forbidden"
    exit_code = 2


class ToolMissingError(MusicError):
    """An external program (ffmpeg, ffprobe) is not installed."""

    code = "tool_missing"
    exit_code = 2


class JobNotFoundError(MusicError):
    code = "job_not_found"


ERRORS_BY_CODE: dict[str, type[MusicError]] = {
    cls.code: cls
    for cls in (
        MusicError,
        ConfigError,
        UnknownModelError,
        ModelNotInstalledError,
        BackendNotReadyError,
        UnsupportedTaskError,
        InvalidRequestError,
        ResourceLimitError,
        OutOfMemoryError,
        DownloadError,
        LicenseNotAcceptedError,
        VerificationError,
        GenerationError,
        CancelledError,
        SecurityError,
        ToolMissingError,
        JobNotFoundError,
    )
}


def error_from_dict(data: dict[str, object]) -> MusicError:
    """Rebuild a typed error sent across the worker protocol."""
    cls = ERRORS_BY_CODE.get(str(data.get("code", "error")), MusicError)
    hint = data.get("hint")
    return cls(str(data.get("message", "unknown error")), hint=str(hint) if hint else None)
