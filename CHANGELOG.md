# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this crate adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- The aarch64 NEON decode kernel takes `&[u8; 20]` / `&mut [u8; 16]` instead
  of open slices, so its loads and stores are in bounds by type and it is no
  longer an `unsafe fn` (matching the AVX2 kernel). Throughput is unchanged
  from 256 bytes up; inputs under 256 bytes decode ~1–2 ns slower.

- CI: Miri runs the test suite on the x86_64 AVX2 and scalar paths, and a
  cargo-mutants gate on aarch64 and x86_64 fails PRs that leave mutants alive.
- Tests: the x86_64 scalar fallback, `DecodeError`'s `Display` output, and
  `decode`'s loop bounds with mismatched buffers are now tested directly.

## [0.1.0] - 2026-05-02

Initial release.

- `encode(&[u8]) -> String` and `decode(&str) -> Result<Vec<u8>, DecodeError>`,
  byte-for-byte compatible with the `base85` crate v2.0.0 on the success path.
- aarch64 NEON-accelerated encode (4 blocks/iter) and decode (4 blocks/iter,
  ~10× the reference scalar throughput on Apple M-series).
- x86_64 AVX2-accelerated encode and decode (8 blocks/iter), with runtime
  feature detection and scalar fallback for hosts lacking AVX2.
- Portable scalar implementation for other architectures.
- Strict overflow detection: 5-character blocks whose value exceeds `u32::MAX`
  return `DecodeError::Overflow` rather than silently wrapping (release) or
  panicking (debug) as the reference does.
- Diagnostic `DecodeError` enum with byte positions; marked `#[non_exhaustive]`
  for forward-compatibility.
- MSRV: Rust 1.85.

[Unreleased]: https://github.com/cipherstash/base85-simd/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cipherstash/base85-simd/releases/tag/v0.1.0
