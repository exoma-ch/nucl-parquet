{
  description = "nucl-parquet — reproducible dev + CI environment (self-hostable, NixOS-native)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        lib = pkgs.lib;

        # Native libraries that generic-linux Python wheels (duckdb, h5py, numpy)
        # and prebuilt tools dynamically link against. Putting them on
        # LD_LIBRARY_PATH is what makes `uv run` / duckdb / h5py work on NixOS
        # without the manual `nix-build -A zlib`/`gcc.cc.lib` dance we used to do.
        nativeLibs = with pkgs; [
          stdenv.cc.cc.lib # libstdc++.so.6
          zlib # libz.so.1
        ];

        toolchain = with pkgs; [
          uv # Python + deps (manages its own interpreters)
          rustc
          cargo
          clippy
          rustfmt
          nodejs_22
          go
          ruff
          git
          cacert # TLS roots for uv/npm/go/cargo fetches
          minisign # data-release signing + verification (#289)
          zstd # unpack data tarballs when verifying a release
        ];

        # Only the code needed for the pure `nix flake check` lint — deliberately
        # excludes data/ (thousands of parquets) so the check doesn't copy ~GB into
        # the Nix store.
        lintSrc = lib.fileset.toSource {
          root = ./.;
          fileset = lib.fileset.unions [
            ./nucl_parquet
            ./scripts
            ./tests
            ./pyproject.toml
          ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = toolchain ++ nativeLibs;
          env = {
            LD_LIBRARY_PATH = lib.makeLibraryPath nativeLibs;
            UV_PYTHON_DOWNLOADS = "automatic";
          };
          shellHook = ''
            echo "nucl-parquet devShell: uv $(uv --version | cut -d' ' -f2) · rust $(rustc --version | cut -d' ' -f2) · node $(node --version) · go $(go version | cut -d' ' -f3)"
            echo "run the full CI locally with:  ./scripts/ci.sh"
          '';
        };

        # `nix flake check` runs the fast, network-free lint. The full suite
        # (pytest/cargo/npm/go) needs network to fetch deps, so it runs via the
        # devShell — `nix develop -c ./scripts/ci.sh` — which the CI shim invokes.
        checks.lint = pkgs.runCommand "nucl-parquet-lint" { nativeBuildInputs = [ pkgs.ruff ]; } ''
          # The source lives in the read-only Nix store, so redirect ruff's cache
          # (both `check` and `format` write it) to the writable build tmpdir.
          export HOME="$TMPDIR"
          export RUFF_CACHE_DIR="$TMPDIR/ruff-cache"
          cd ${lintSrc}
          ruff check nucl_parquet scripts tests
          ruff format --check nucl_parquet scripts tests
          touch $out
        '';

        formatter = pkgs.nixfmt;
      }
    );
}
