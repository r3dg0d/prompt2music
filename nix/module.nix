# NixOS module: installs prompt2music system-wide and can run the
# job daemon as a systemd *user* service for every user. Per-user settings belong
# in the Home Manager module (nix/home-manager.nix).
#
# GPU access uses the regular driver stack (/run/opengl-driver); nothing here
# disables sandboxing or other NixOS security features.
self:
{ config, lib, pkgs, ... }:

let
  cfg = config.programs.prompt2music;
  wheelLibs = lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];
in
{
  options.programs.prompt2music = {
    enable = lib.mkEnableOption "prompt2music local music generation";
    package = lib.mkOption {
      type = lib.types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
      defaultText = lib.literalExpression "prompt2music.packages.\${system}.default";
      description = "The prompt2music package.";
    };
    daemon.enable = lib.mkEnableOption "the prompt2music job daemon as a systemd user service";
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = [ cfg.package pkgs.ffmpeg pkgs.uv pkgs.git ];
    environment.sessionVariables.AI_MEDIA_EXTRA_LD_PATH = wheelLibs;

    systemd.user.services.prompt2music = lib.mkIf cfg.daemon.enable {
      description = "prompt2music job daemon";
      wantedBy = [ "default.target" ];
      path = [ pkgs.ffmpeg pkgs.uv pkgs.git ];
      environment = {
        AI_MEDIA_EXTRA_LD_PATH = wheelLibs;
        PYTHONUNBUFFERED = "1";
      };
      serviceConfig = {
        ExecStart = "${cfg.package}/bin/prompt2music daemon run --idle-exit 0";
        Restart = "on-failure";
        RestartSec = 5;
      };
    };
  };
}
