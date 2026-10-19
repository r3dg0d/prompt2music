# Home Manager module for prompt2music.
#
#   imports = [ prompt2music.homeManagerModules.default ];
#   programs.prompt2music = {
#     enable = true;
#     daemon.enable = true;                    # systemd --user job daemon
#     settings.hardware.vram_limit_gb = 20;    # optional: makes the config read-only
#   };
#
# Model weights and backend venvs stay under $XDG_DATA_HOME (never the store).
self:
{ config, lib, pkgs, ... }:

let
  cfg = config.programs.prompt2music;
  tomlFormat = pkgs.formats.toml { };
  system = pkgs.stdenv.hostPlatform.system;
  # Libraries that pip/uv wheels (torch, numpy, …) expect on a FHS system.
  wheelLibs = lib.makeLibraryPath ([ pkgs.stdenv.cc.cc.lib pkgs.zlib ] ++ cfg.extraLibraries);
  runtimePath = lib.makeBinPath [ pkgs.ffmpeg pkgs.uv pkgs.git pkgs.coreutils ];
  envVars = {
    AI_MEDIA_EXTRA_LD_PATH = wheelLibs;
    PROMPT2MUSIC_ENV_PYTHON = "${cfg.backendPython}/bin/python3";
  };
in
{
  options.programs.prompt2music = {
    enable = lib.mkEnableOption "prompt2music local music generation";

    package = lib.mkOption {
      type = lib.types.package;
      default = self.packages.${system}.default;
      defaultText = lib.literalExpression "prompt2music.packages.\${system}.default";
      description = "The package providing prompt2music, aimusic and the MCP servers.";
    };

    settings = lib.mkOption {
      type = tomlFormat.type;
      default = { };
      example = lib.literalExpression ''
        {
          general.default_model = "ace-step-1.5";
          hardware.vram_limit_gb = 20;
          mcp.allowed_input_dirs = [ "~/Music" ];
        }
      '';
      description = ''
        Contents of ~/.config/ai-media/prompt2music.toml. When non-empty the file is
        managed by Home Manager (read-only) and `prompt2music config set` is refused.
        Leave empty to manage the file with the CLI instead.
      '';
    };

    backendPython = lib.mkOption {
      type = lib.types.package;
      default = pkgs.python312;
      defaultText = lib.literalExpression "pkgs.python312";
      description = "Base interpreter for backend venvs created by `prompt2music backends install`.";
    };

    extraLibraries = lib.mkOption {
      type = lib.types.listOf lib.types.package;
      default = [ ];
      description = "Extra libraries exposed to backend venvs via AI_MEDIA_EXTRA_LD_PATH.";
    };

    daemon.enable = lib.mkEnableOption "the prompt2music job daemon as a systemd user service";

    mcp.writeClientConfig = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Write ~/.config/ai-media/prompt2music-mcp.json, an `mcpServers` snippet with the
        absolute store path of prompt2music-mcp, for MCP clients that read JSON configs.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    home.packages = [ cfg.package pkgs.ffmpeg pkgs.uv pkgs.git ];
    home.sessionVariables = envVars;

    xdg.configFile."ai-media/prompt2music.toml" = lib.mkIf (cfg.settings != { }) {
      source = tomlFormat.generate "prompt2music.toml" cfg.settings;
    };

    xdg.configFile."ai-media/prompt2music-mcp.json" = lib.mkIf cfg.mcp.writeClientConfig {
      text = builtins.toJSON {
        mcpServers.prompt2music = {
          command = "${cfg.package}/bin/prompt2music-mcp";
          args = [ ];
        };
      };
    };

    systemd.user.services.prompt2music = lib.mkIf cfg.daemon.enable {
      Unit = {
        Description = "prompt2music job daemon";
        After = [ "default.target" ];
      };
      Service = {
        ExecStart = "${cfg.package}/bin/prompt2music daemon run --idle-exit 0";
        Restart = "on-failure";
        RestartSec = 5;
        Environment = [
          "PATH=${runtimePath}"
          "AI_MEDIA_EXTRA_LD_PATH=${wheelLibs}"
          "PROMPT2MUSIC_ENV_PYTHON=${cfg.backendPython}/bin/python3"
          "PYTHONUNBUFFERED=1"
        ];
      };
      Install.WantedBy = [ "default.target" ];
    };
  };
}
