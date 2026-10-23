#!/usr/bin/env bash
# Runs cargo-mutants with the excludes for this host's arch, passing any
# arguments through. The one place the per-arch excludes live: CI
# (.github/workflows/mutants.yml) and `mise run mutants` both call this.
#
# cargo-mutants doesn't evaluate `#[cfg]`, so it also mutates code the host
# doesn't compile, and every such mutant "survives". Each arch excludes the
# other's dispatch: aarch64 never compiles the AVX2 path or the scalar
# fallback (`*_scalar`); x86_64 never compiles the NEON path, the `neon`
# module or ops/aarch64.rs. Excludes that hold on every host live in
# .cargo/mutants.toml.
set -euo pipefail

case "$(uname -m)" in
  arm64 | aarch64) exclude='avx2|_scalar' ;;
  x86_64) exclude='_neon|neon::|ops/aarch64\.rs' ;;
  *)
    echo "mutants.sh: no excludes defined for $(uname -m)" >&2
    exit 1
    ;;
esac

exec cargo mutants --exclude-re "$exclude" "$@"
