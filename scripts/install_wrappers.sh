#!/usr/bin/env bash
# Install prompt2music / aimusic / prompt2music-mcp / aimusic-mcp into ~/.local/bin
# from an XDG venv (no flake required).
#
# The CLI venv is light (no torch). Model backends get their own venvs later:
#   prompt2music backends install ace-step --yes
set -euo pipefail

XDG_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
VENV_DIR="${AI_MEDIA_MUSIC_VENV:-$XDG_DATA_HOME/ai-media/envs/prompt2music}"
BIN_DIR="${HOME}/.local/bin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXTRA_LD="${AI_MEDIA_EXTRA_LD_PATH:-}"

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      echo "Usage: $0    (venv: $VENV_DIR, wrappers: $BIN_DIR)"
      echo "NixOS: export AI_MEDIA_EXTRA_LD_PATH=<gcc-lib>/lib:<zlib>/lib before running"
      exit 0 ;;
  esac
done

command -v ffmpeg >/dev/null 2>&1 || echo "WARN: ffmpeg not on PATH (required for audio I/O)" >&2

mkdir -p "$(dirname "$VENV_DIR")" "$BIN_DIR"
if command -v uv >/dev/null 2>&1; then
  [[ -d "$VENV_DIR" ]] || uv venv "$VENV_DIR" --python "${AI_MEDIA_PYTHON:-3.12}"
  uv pip install --python "$VENV_DIR/bin/python" -e "$REPO_ROOT[mcp]"
else
  [[ -d "$VENV_DIR" ]] || python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install -e "$REPO_ROOT[mcp]"
fi

for NAME in prompt2music aimusic prompt2music-mcp aimusic-mcp; do
  WRAPPER="$BIN_DIR/$NAME"
  cat > "$WRAPPER" <<WRAP
#!/usr/bin/env bash
# prompt2music ${NAME} → ${VENV_DIR}
export AI_MEDIA_EXTRA_LD_PATH="\${AI_MEDIA_EXTRA_LD_PATH:-${EXTRA_LD}}"
exec "${VENV_DIR}/bin/${NAME}" "\$@"
WRAP
  chmod +x "$WRAPPER"
  echo "Installed $WRAPPER"
done
echo "Next: prompt2music doctor && prompt2music setup"
