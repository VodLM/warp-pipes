{
  description = "Warp Pipes";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";

  inputs.poetry2nix = {
    url = "github:nix-community/poetry2nix";
    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix, ... }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
      inherit (poetry2nix.legacyPackages.${system}) mkPoetryEnv mkPoetryApplication;
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      # nix build .#name
      packages = {
        warppipes_environment = mkPoetryEnv { projectDir = self; preferWheels = false; };
        warppipes_application = mkPoetryApplication { projectDir = self; };
      };
      packages.default = self.packages.${system}.warppipes_application; 

      # nix develop .#name
      devShells = {
        docs = pkgs.mkShellNoCC {
          packages = [ pkgs.mdbook ];
        };
        publish = pkgs.mkShellNoCC {
          packages = [];
        };

        develop = pkgs.mkShell {
          buildInputs = [ ];
          packages = [
            poetry2nix.packages.${system}.poetry
            pkgs.bashInteractive
            pkgs.pre-commit
            pkgs.du-dust
            pkgs.starship
            pkgs.magic-wormhole
            pkgs.mdbook
          ];

          shellHook = ''
          eval "$(starship init bash)"
          '';
        };

      };
      devShells.default = self.devShells.${system}.develop;
    });
}