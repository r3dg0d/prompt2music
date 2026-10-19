# prompt2music as a Nix package.
#
# Only the light CLI/MCP layer is packaged. Model backends (torch, upstream
# repos) live in per-backend venvs created by `prompt2music backends install`,
# and weights live under $XDG_DATA_HOME — never in the Nix store.
{ lib
, python3Packages
, ffmpeg-headless
, bash
, withMcp ? true
}:

python3Packages.buildPythonApplication {
  pname = "prompt2music";
  version = "0.1.0";
  pyproject = true;

  src = lib.cleanSourceWith {
    src = ../.;
    filter = path: type:
      let base = baseNameOf path; in
      !(builtins.elem base [ ".venv" "venv" ".git" "result" "__pycache__" ".pytest_cache"
                             ".ruff_cache" ".mypy_cache" "demos" ]);
  };

  build-system = [ python3Packages.hatchling ];

  dependencies = with python3Packages; [
    rich
    tomlkit
    huggingface-hub
  ] ++ lib.optionals withMcp [ python3Packages.mcp ];

  # ffmpeg/ffprobe are runtime requirements of the audio toolkit.
  makeWrapperArgs = [ "--prefix" "PATH" ":" (lib.makeBinPath [ ffmpeg-headless ]) ];

  nativeCheckInputs = [ python3Packages.pytestCheckHook ffmpeg-headless bash ];
  preCheck = ''
    export HOME=$(mktemp -d)
    export PATH=$out/bin:$PATH
  '';
  # GPU / network acceptance tests are deselected by the pyproject markers.

  pythonImportsCheck = [ "prompt2music.cli" "prompt2music.mcp.server" ];

  meta = {
    description = "Local music generation with open-weight models (CLI + MCP server)";
    homepage = "https://github.com/r3dg0d/prompt2music";
    license = lib.licenses.mit;
    mainProgram = "prompt2music";
    platforms = lib.platforms.unix;
  };
}
