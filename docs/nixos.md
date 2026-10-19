# prompt2music on NixOS

The flake packages the light CLI/MCP layer. Torch-based model backends live in
per-backend venvs under `$XDG_DATA_HOME`, and weights live under
`~/.local/share/ai-media/models/music`. Nothing writes to the Nix store, and
nothing requires disabling NixOS security features.

## What has been verified

On nixpkgs `nixos-unstable` (26.11pre, Python 3.14, MCP SDK 1.29):

* `nix-build` of `nix/package.nix` succeeds and its check phase runs the test
  suite (223 passed; GPU/network tests deselected; torch contract tests skipped
  because nixpkgs torch is not in the closure).
* The built `prompt2music`, `aimusic`, `prompt2music-mcp` and `aimusic-mcp`
  run. In ai-media-cli's identical build, a background job runs through the detached daemon, and
  a real MCP client session lists all 30 tools.
* `nix/home-manager.nix` and `nix/module.nix` evaluate through the nixpkgs
  module system (stubbed host options). They produce the expected TOML, systemd
  `ExecStart`, MCP JSON and session variables.

Not yet verified: a full `home-manager switch` on the workstation, and building
CUDA backend venvs on NixOS (needs the GPU machine).

## Flake

```nix
{
  inputs.prompt2music.url = "github:r3dg0d/prompt2music";

  outputs = { nixpkgs, home-manager, prompt2music, ... }: {
    homeConfigurations.vincent = home-manager.lib.homeManagerConfiguration {
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
      modules = [
        prompt2music.homeManagerModules.default
        {
          programs.prompt2music = {
            enable = true;
            daemon.enable = true;             # systemd --user job daemon
            settings = {                      # optional; makes the config read-only
              general.default_model = "ace-step-1.5";
              hardware.vram_limit_gb = 20;
              mcp.allowed_input_dirs = [ "~/Music" ];
            };
          };
        }
      ];
    };
  };
}
```

Try it without installing:

```bash
nix run github:r3dg0d/prompt2music#prompt2music -- doctor
nix run github:r3dg0d/prompt2music#prompt2music-mcp -- --help
nix develop github:r3dg0d/prompt2music      # dev shell with ffmpeg + uv
```

### Home Manager options

| Option | Default | Purpose |
| ------ | ------- | ------- |
| `programs.prompt2music.enable` | false | installs the package, ffmpeg, uv, git |
| `.package` | flake package | |
| `.settings` | `{}` | TOML written to `~/.config/ai-media/prompt2music.toml`; when set, the file is read-only and `prompt2music config set` explains where to change it |
| `.backendPython` | `pkgs.python312` | base interpreter for `backends install` (exported as `PROMPT2MUSIC_ENV_PYTHON`) |
| `.extraLibraries` | `[]` | extra libs for wheels in backend venvs |
| `.daemon.enable` | false | `systemd.user.services.prompt2music` running `prompt2music daemon run --idle-exit 0` |
| `.mcp.writeClientConfig` | true | writes `~/.config/ai-media/prompt2music-mcp.json` with the absolute server path |

The NixOS module (`nixosModules.default`) installs the package system-wide and
can enable the same user service for every user.

## CUDA and backend venvs

PyPI CUDA wheels (torch etc.) expect FHS libraries. The modules export
`AI_MEDIA_EXTRA_LD_PATH` (libstdc++ and zlib from nixpkgs). Workers prepend
`/run/opengl-driver/lib` (the NVIDIA driver) plus that path to
`LD_LIBRARY_PATH`, exactly like the `install_*_wrapper.sh` scripts in
ai-media-cli. Then:

```bash
prompt2music backends install ace-step --yes   # uv sync of upstream's lockfile
prompt2music doctor                            # PyTorch: CUDA enabled?
```

If `uv` downloads its own CPython and that interpreter cannot run on your
system, keep the default `backendPython` (a Nix-built Python passed as
`--python`), or enable `programs.nix-ld` yourself. prompt2music never turns it
on for you.

## Without flakes

`scripts/install_wrappers.sh` creates an XDG venv
(`~/.local/share/ai-media/envs/music-cli`) with this repo installed and
writes `~/.local/bin/{prompt2music,aimusic,prompt2music-mcp,aimusic-mcp}`.
