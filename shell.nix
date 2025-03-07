{ pkgs ? import <nixpkgs> {} }:

(pkgs.buildFHSUserEnv {
  name = "pixi-env";
  targetPkgs = _: [
    pkgs.pixi
  ];
  multiPkgs = pkgs: [
    pkgs.zlib
    pkgs.glibc
  ];
}).env
