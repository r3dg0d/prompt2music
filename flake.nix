{
  description = "prompt2music — local music generation with open-weight models (CLI + MCP server)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    let
      overlay = final: _prev: {
        prompt2music = final.callPackage ./nix/package.nix { };
      };
    in
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; overlays = [ overlay ]; };
        app = name: { type = "app"; program = "${pkgs.prompt2music}/bin/${name}"; };
      in {
        packages = {
          default = pkgs.prompt2music;
          prompt2music = pkgs.prompt2music;
        };
        apps = {
          default = app "prompt2music";
          prompt2music = app "prompt2music";
          aimusic = app "aimusic";
          prompt2music-mcp = app "prompt2music-mcp";
        };
        # Builds the package and runs its pytest suite (GPU/network tests deselected).
        checks.prompt2music = pkgs.prompt2music;
        devShells.default = pkgs.mkShell {
          buildInputs = [ pkgs.python312 pkgs.uv pkgs.ruff pkgs.ffmpeg pkgs.git ];
          shellHook = ''
            export AI_MEDIA_EXTRA_LD_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}"
            echo "prompt2music dev shell — run: uv sync --extra dev --extra mcp && uv run pytest"
          '';
        };
      }) // {
        overlays.default = overlay;
        homeManagerModules.default = import ./nix/home-manager.nix self;
        homeManagerModules.prompt2music = import ./nix/home-manager.nix self;
        nixosModules.default = import ./nix/module.nix self;
        nixosModules.prompt2music = import ./nix/module.nix self;
      };
}
