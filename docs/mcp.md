# prompt2music MCP server

`prompt2music-mcp` (alias `aimusic-mcp`, also `prompt2music mcp`) exposes music
generation, model management, audio processing and jobs to MCP clients such as
Claude Code, OpenCode, Cursor and VS Code. It shares the model registry,
backends, job daemon, library and configuration with the CLI. It works with
the official MCP Python SDK v1 (`FastMCP`) and v2 (`MCPServer`), and needs no
graphical desktop.

## Transports

* **stdio** (default): protocol on stdout, logs on stderr.
* **Streamable HTTP** (optional): `prompt2music-mcp --transport http`. It binds
  `127.0.0.1:8765` by default, with DNS-rebinding protection on loopback.
  Binding any other interface is refused unless the environment variable named
  by `mcp.auth_token_env` (default `PROMPT2MUSIC_MCP_TOKEN`) holds a token, which
  every request must then send as `Authorization: Bearer <token>`.

## Client configuration

Claude Code (checked against the `claude mcp add` syntax: name, then `--`, then
the command):

```bash
claude mcp add prompt2music -- prompt2music-mcp
claude mcp add --scope project prompt2music -- prompt2music-mcp   # shared .mcp.json
```

Generic `mcpServers` JSON (Claude Desktop, Cursor `~/.cursor/mcp.json`, and
others):

```json
{
  "mcpServers": {
    "prompt2music": { "command": "prompt2music-mcp", "args": [] }
  }
}
```

OpenCode (`opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": { "prompt2music": { "type": "local", "command": ["prompt2music-mcp"], "enabled": true } }
}
```

VS Code (`.vscode/mcp.json`):

```json
{ "servers": { "prompt2music": { "type": "stdio", "command": "prompt2music-mcp", "args": [] } } }
```

`prompt2music mcp config --client claude|opencode|cursor|vscode|generic` prints
these. If `prompt2music-mcp` is not on the client's `PATH`, use its absolute
path (the Home Manager module writes `~/.config/ai-media/prompt2music-mcp.json`
with the store path).

## Agent workflow

1. `music_models_list` with `installed_only=true` to see what can run.
2. `music_generate` (or `_instrumental`, `_song`, `music_continue`,
   `music_edit`, `music_inpaint`). This returns `{"job_id", "status": "queued"}`
   immediately.
3. Poll `music_job_status` until `completed` / `failed` / `cancelled`. Progress
   notifications are not required by clients.
4. `music_job_result` returns output paths, `file://` URIs, `music://track/ID`
   resource ids, durations, seeds, model and timing. Audio is never embedded in
   JSON.
5. `music_audio_info` to inspect a file.

Resources: `music://track/{track_id}` (library metadata) and
`music://job/{job_id}`.

## Tools

| Tool | Kind |
| ---- | ---- |
| `music_generate`, `music_generate_instrumental`, `music_generate_song`, `music_continue`, `music_edit`, `music_inpaint` | queued job |
| `music_enhance_prompt` | read-only |
| `music_models_list`, `music_models_search`, `music_model_info` | read-only |
| `music_model_download` | queued job; disabled by default |
| `music_model_load`, `music_model_unload` | queued job |
| `music_model_remove` | destructive; disabled by default; needs `confirm=true` |
| `music_audio_info` | read-only |
| `music_audio_convert`, `music_audio_normalize`, `music_audio_trim`, `music_audio_mix`, `music_audio_loop`, `music_audio_waveform`, `music_audio_spectrogram` | synchronous, bounded by FFmpeg |
| `music_separate_stems` | queued job |
| `music_system_info`, `music_gpu_status` | read-only |
| `music_benchmark` | queued job |
| `music_job_status`, `music_jobs_list`, `music_job_result` | read-only |
| `music_job_cancel` | cancels jobs created over MCP |

Each tool has a typed input schema (generated from its signature), MCP
annotations (`readOnlyHint`, `destructiveHint`, `openWorldHint`) and structured
output. Errors come back as `isError` results whose text is JSON:
`{"error": {"code": "forbidden", "message": ..., "hint": ...}}`.

## Security model

Tool inputs are treated as untrusted:

* **Paths**: outputs must resolve, after following symlinks, inside
  `general.output_directory` or `mcp.allowed_output_dirs`. Inputs must resolve
  inside those or `mcp.allowed_input_dirs`. `..`, absolute paths elsewhere,
  URLs, NUL bytes, symlinked output files and symlinked directories pointing
  outside are rejected. Existing files are never overwritten.
* **Limits**: `mcp.max_input_file_mb`, `mcp.max_duration_seconds`,
  `mcp.max_queued_jobs` (MCP-created active jobs), `jobs.max_queued`, and
  `mcp.max_download_gb`.
* **Models**: catalogued ids only, optionally restricted by `mcp.allowed_models`.
  Downloads need `mcp.allow_model_downloads = true` and an explicit
  `accept_license=true` for licences that require acceptance. Python files in
  model repos are never fetched over MCP.
* **Deletion**: `mcp.allow_model_deletion = true` plus `confirm=true`.
* **Execution**: subprocesses are argv lists; no shell strings are built from
  inputs, and nothing from a model repository is imported by the server.
* **Concurrency**: each call uses its own SQLite connection (WAL); job claiming
  is atomic; generation runs one job at a time in the daemon behind a
  cross-process GPU lock.
* **Audit**: every call is appended to `~/.cache/ai-media/music/logs/mcp-audit.jsonl`
  (long strings truncated).
