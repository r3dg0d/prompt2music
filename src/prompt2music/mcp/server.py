"""prompt2music-mcp / aimusic-mcp: MCP server over stdio (default) or Streamable HTTP.

stdout carries only protocol messages in stdio mode; logs go to stderr and
tool calls to ``~/.cache/ai-media/music/logs/mcp-audit.jsonl``.

HTTP binds to 127.0.0.1 unless configured otherwise. Binding to any
non-loopback address requires a bearer token in the environment variable
named by ``mcp.auth_token_env``; the token is then required on every request.

Works with the official MCP Python SDK v2 (``MCPServer``) and v1
(``FastMCP``).
"""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import logging
import os
import sys
from typing import Any

from prompt2music import __version__
from prompt2music.config import MusicConfig, load_music_config
from prompt2music.errors import ConfigError, MusicError
from prompt2music.mcp.tools import State, make_tools, wrap

INSTRUCTIONS = """\
prompt2music generates music locally with open-weight models.
Workflow: music_models_list (installed=true) → music_generate / music_generate_song /
music_continue ... (returns job_id) → poll music_job_status until completed →
music_job_result for output paths. Audio is returned as local file paths and
file:// URIs, never inline. Output paths must stay inside the configured output
directories. Respect each model's license (see music_model_info).
"""


def _server_class() -> Any:
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP  # type: ignore[attr-defined]  # SDK v1
        except ImportError as exc:
            raise ConfigError("the MCP SDK is not installed",
                              hint="pip install 'ai-media-cli[mcp]'") from exc
        return FastMCP


def _tool_error_class() -> type[Exception]:
    for mod in ("mcp.server.mcpserver.exceptions", "mcp.server.fastmcp.exceptions"):
        try:
            module = __import__(mod, fromlist=["ToolError"])
            return module.ToolError
        except ImportError:
            continue
    return RuntimeError


def build_server(cfg: MusicConfig) -> Any:
    from mcp.types import ToolAnnotations

    cls = _server_class()
    try:
        server = cls(name="prompt2music", instructions=INSTRUCTIONS, version=__version__)
    except TypeError:  # SDK v1 FastMCP has no version argument
        server = cls(name="prompt2music", instructions=INSTRUCTIONS)
    state = State(cfg)
    tool_error = _tool_error_class()
    for tool in make_tools(state):
        # Wire-format aliases validate on both SDK v1 (camelCase fields) and v2.
        annotations = ToolAnnotations.model_validate({
            "readOnlyHint": tool.read_only,
            "destructiveHint": tool.destructive,
            "openWorldHint": tool.name == "music_model_download",
        })
        server.add_tool(wrap(state, tool, raise_as=tool_error), name=tool.name,
                        description=tool.description, annotations=annotations,
                        structured_output=True)

    @server.resource("music://track/{track_id}", name="track", mime_type="application/json",
                     description="Library metadata (sidecar) for a generated track")
    def track_resource(track_id: str) -> str:
        from prompt2music.library import Library

        data = Library().get(track_id)
        return json.dumps({k: v for k, v in data.items() if k != "params"}, default=str)

    @server.resource("music://job/{job_id}", name="job", mime_type="application/json",
                     description="Status and result of a prompt2music job")
    def job_resource(job_id: str) -> str:
        from prompt2music.jobs import JobStore

        return json.dumps(JobStore().get(job_id).as_dict(brief=False), default=str)

    return server


def _is_loopback(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class BearerAuth:
    """Minimal ASGI middleware requiring ``Authorization: Bearer <token>``."""

    def __init__(self, app: Any, token: str) -> None:
        self.app = app
        self.token = token.encode()

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            given = headers.get(b"authorization", b"")
            expected = b"Bearer " + self.token
            if not hmac.compare_digest(given, expected):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json"),
                                        (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body",
                            "body": b'{"error":"unauthorized"}'})
                return
        await self.app(scope, receive, send)


def _http_app(server: Any, host: str, port: int) -> Any:
    kwargs: dict[str, Any] = {}
    try:
        from mcp.server.transport_security import TransportSecuritySettings

        allowed = [f"{h}:{port}" for h in ("127.0.0.1", "localhost", host)] + [host]
        kwargs["transport_security"] = TransportSecuritySettings(
            enable_dns_rebinding_protection=_is_loopback(host), allowed_hosts=allowed,
            allowed_origins=[f"http://{h}:{port}" for h in ("127.0.0.1", "localhost", host)])
    except ImportError:
        pass
    try:
        return server.streamable_http_app(host=host, **kwargs)
    except TypeError:
        try:
            return server.streamable_http_app(**kwargs)
        except TypeError:
            return server.streamable_http_app()


def serve(cfg: MusicConfig, *, transport: str | None = None, host: str | None = None,
          port: int | None = None) -> int:
    logging.basicConfig(stream=sys.stderr, level=cfg["logging"]["level"].upper(),
                        format="[prompt2music-mcp] %(levelname)s %(message)s")
    transport = transport or cfg["mcp"]["transport"]
    server = build_server(cfg)
    if transport == "stdio":
        server.run("stdio")
        return 0
    host = host or cfg["mcp"]["host"]
    port = port or int(cfg["mcp"]["port"])
    token = os.environ.get(cfg["mcp"]["auth_token_env"], "")
    if not _is_loopback(host) and not token:
        raise ConfigError(
            f"refusing to listen on non-loopback address {host} without authentication",
            hint=f"export {cfg['mcp']['auth_token_env']}=<long random token>")
    app = _http_app(server, host, port)
    if token:
        app = BearerAuth(app, token)
    import uvicorn

    print(f"prompt2music-mcp listening on http://{host}:{port}/mcp"
          + (" (bearer token required)" if token else " (loopback only)"), file=sys.stderr)
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


def client_config(client: str = "generic") -> str:
    entry = {"command": "prompt2music-mcp", "args": []}
    if client == "claude":
        return ("claude mcp add prompt2music -- prompt2music-mcp\n"
                "# project-scoped alternative: claude mcp add --scope project prompt2music "
                "-- prompt2music-mcp")
    if client == "opencode":
        return json.dumps({"$schema": "https://opencode.ai/config.json",
                           "mcp": {"prompt2music": {"type": "local",
                                                    "command": ["prompt2music-mcp"],
                                                    "enabled": True}}}, indent=2)
    if client == "vscode":
        return json.dumps({"servers": {"prompt2music": {"type": "stdio", **entry}}}, indent=2)
    return json.dumps({"mcpServers": {"prompt2music": entry}}, indent=2)


def main(argv: list[str] | None = None, prog: str = "prompt2music-mcp") -> None:
    parser = argparse.ArgumentParser(prog=prog, description="prompt2music MCP server "
                                     "(stdio by default; HTTP binds to localhost)")
    parser.add_argument("--version", action="version", version=f"{prog} {__version__}")
    parser.add_argument("--transport", choices=["stdio", "http"])
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--config", help="config file path")
    parser.add_argument("--print-config", choices=["generic", "claude", "opencode", "cursor",
                                                   "vscode"],
                        help="print client configuration and exit")
    args = parser.parse_args(argv)
    if args.print_config:
        print(client_config(args.print_config))
        sys.exit(0)
    try:
        from pathlib import Path

        cfg = load_music_config(Path(args.config).expanduser() if args.config else None,
                                create=True)
        sys.exit(serve(cfg, transport=args.transport, host=args.host, port=args.port))
    except MusicError as exc:
        print(f"prompt2music-mcp: {exc}" + (f"\nhint: {exc.hint}" if exc.hint else ""),
              file=sys.stderr)
        sys.exit(exc.exit_code)
    except KeyboardInterrupt:
        sys.exit(130)


def main_aimusic(argv: list[str] | None = None) -> None:
    main(argv, prog="aimusic-mcp")


if __name__ == "__main__":
    main()
