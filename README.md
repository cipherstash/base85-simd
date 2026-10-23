# base85-simd

Fast Base85 (RFC 1924 / Z85-style) encoder and decoder for Rust.

SIMD-accelerated on aarch64 (NEON, 4 blocks per iteration) and x86_64
(AVX2, 8 blocks per iteration), with a portable scalar fallback for
everything else and for x86_64 hosts lacking AVX2 (rare on server
hardware after ~2013). Output is byte-for-byte compatible with the
[`base85`](https://crates.io/crates/base85) crate.

## Usage

```rust
let data = b"hello, world!";
let encoded = base85_simd::encode(data);
let decoded = base85_simd::decode(&encoded).unwrap();
assert_eq!(decoded, data);
```

## Alphabet

```text
0123456789
ABCDEFGHIJKLMNOPQRSTUVWXYZ
abcdefghijklmnopqrstuvwxyz
!#$%&()*+-;<=>?@^_`{|}~
```

Inputs whose length is not a multiple of 4 are encoded as
`floor(len / 4) * 5 + (len % 4) + 1` characters. The trailing partial block
is padded with zero bytes for encoding and with `~` (the maximum digit, 84)
for decoding.

## Status

- Public API: `encode(&[u8]) -> String`, `decode(&str) -> Result<Vec<u8>, DecodeError>`.
- aarch64: NEON-accelerated (4 blocks at a time, always available on aarch64).
- x86_64 with AVX2: AVX2-accelerated (8 blocks at a time). Runtime feature
  detection at the public API entry — hosts without AVX2 fall back to scalar.
- Other architectures: portable scalar implementation.
- The decode path validates char range and detects `u32` overflow lane-wise;
  any invalid input falls back to the scalar path so the resulting
  `DecodeError` carries a precise byte position.
- Tested against the `base85` reference crate via quickcheck round-trip and
  parity checks.
- The x86_64 AVX2 and scalar paths run under [Miri](https://github.com/rust-lang/miri)
  in CI to catch undefined behaviour in the `unsafe` SIMD code. Miri can't
  interpret the NEON intrinsics, so the aarch64 path runs under
  AddressSanitizer in CI instead.
- Mutation-tested with [cargo-mutants](https://mutants.rs) on both aarch64
  and x86_64; CI fails a PR that adds logic its tests don't pin down.

## Drop-in compatibility with the `base85` crate

This crate is a drop-in replacement for [`base85`](https://crates.io/crates/base85)
v2.0.0 on the success path. Verified by `tests/base85_parity.rs`:

- **Encode**: byte-identical for *every* `&[u8]` we've tested — every length
  0..=512, every single-byte and two-byte input exhaustively, a 200 k
  three-byte stratified sample, a 1 MiB stress test, and two 10 000-iter
  quickcheck properties.
- **Decode of valid input**: byte-identical for every base85 string the
  reference accepts (10 000-iter quickcheck plus length sweeps).
- **Both reject the same invalid inputs** (different error types, but
  both surface a rejection).

### Intentional safety divergence

For 5-character blocks whose value exceeds `u32::MAX` (e.g. `"~~~~~"` or
`"|NsC1"`), the `base85` crate **silently wraps the value in release
builds** and **panics in debug builds**. We instead always return
`DecodeError::Overflow` with the offending block's byte position. This
is a strict safety improvement — silent wrapping returns
attacker-controlled bytes; panicking is a DoS vector. Replicating either
behaviour behind a compat feature flag is **not** offered: any caller
depending on the wrap output is doing something dangerous.

### Error-type migration

The reference crate exposes `base85::Error::{InvalidCharacter(u8), UnexpectedEof}`.
We expose `DecodeError::{InvalidChar { byte, position }, InvalidLength { len }, Overflow { position }}` —
strictly more diagnostic information. Code that pattern-matches on
`base85::Error` will need to update its match arms.

## Benchmarks

`cargo bench --bench encode`, criterion, release profile, single-threaded.
Times are the criterion-reported median; throughput computed from it.

### aarch64 (Apple M-series)

#### Encode

| size  | `base85` | `base85-simd` (NEON) | speedup | throughput |
|-------|---------:|---------------------:|--------:|-----------:|
| 16 B  | 17.4 ns  | 16.3 ns              | 1.07×   | ~940 MiB/s |
| 64 B  | 33.0 ns  | 22.6 ns              | 1.46×   | 2.71 GiB/s |
| 256 B | 115.7 ns | 73.2 ns              | 1.58×   | 3.26 GiB/s |
| 1 KiB | 378.3 ns | 225.1 ns             | 1.68×   | 4.24 GiB/s |
| 16 KiB| 5.56 µs  | 3.60 µs              | 1.55×   | 4.24 GiB/s |
| 256 KiB| 89.3 µs | 55.4 µs              | 1.61×   | 4.41 GiB/s |
| 1 MiB | 356 µs   | 222 µs               | 1.61×   | 4.40 GiB/s |

#### Decode

| size  | `base85` | `base85-simd` (NEON) | speedup | throughput |
|-------|---------:|---------------------:|--------:|-----------:|
| 16 B  | 32.4 ns  | 14.8 ns              | 2.18×   | ~1.0 GiB/s |
| 64 B  | 123.5 ns | 24.6 ns              | 5.02×   | 2.42 GiB/s |
| 256 B | 579 ns   | 57.6 ns              | 10.05×  | 4.14 GiB/s |
| 1 KiB | 2.27 µs  | 226 ns               | 10.06×  | 4.22 GiB/s |
| 16 KiB| 36.8 µs  | 3.49 µs              | 10.55×  | 4.38 GiB/s |
| 256 KiB| 591 µs  | 54.3 µs              | 10.89×  | 4.50 GiB/s |
| 1 MiB | 2.28 ms  | 217.6 µs             | 10.49×  | 4.49 GiB/s |

### x86_64 (AMD EPYC 7763, Zen 3)

Numbers from a GitHub Actions hosted Ubuntu runner — shared/virtualised
hardware so noise is higher than aarch64 (~5–15% variance), but the
relative speedups are stable.

#### Encode

| size  | `base85` | `base85-simd` (AVX2) | speedup | throughput |
|-------|---------:|---------------------:|--------:|-----------:|
| 16 B  | 41.0 ns  | 57.8 ns              | 0.71×   | (scalar fallback; chunk doesn't fit) |
| 64 B  | 100.5 ns | 61.7 ns              | 1.63×   | 989 MiB/s  |
| 256 B | 341.9 ns | 165.9 ns             | 2.06×   | 1.44 GiB/s |
| 1 KiB | 1.32 µs  | 507.0 ns             | 2.61×   | 1.88 GiB/s |
| 16 KiB| 20.4 µs  | 7.48 µs              | 2.72×   | 2.04 GiB/s |
| 256 KiB| 323.9 µs| 118.9 µs             | 2.72×   | 2.05 GiB/s |
| 1 MiB | 1.31 ms  | 482.9 µs             | 2.71×   | 2.07 GiB/s |

#### Decode

| size  | `base85` | `base85-simd` (AVX2) | speedup | throughput |
|-------|---------:|---------------------:|--------:|-----------:|
| 16 B  | 70.2 ns  | 61.2 ns              | 1.15×   | (scalar fallback) |
| 64 B  | 244.8 ns | 67.6 ns              | 3.62×   | 903 MiB/s  |
| 256 B | 1.046 µs | 164.4 ns             | 6.36×   | 1.45 GiB/s |
| 1 KiB | 4.19 µs  | 466.5 ns             | 8.98×   | 2.04 GiB/s |
| 16 KiB| 65.7 µs  | 6.75 µs              | 9.73×   | 2.26 GiB/s |
| 256 KiB| 1.058 ms| 107.2 µs             | 9.86×   | 2.28 GiB/s |
| 1 MiB | 4.14 ms  | 431.6 µs             | 9.59×   | 2.32 GiB/s |

### Steady-state summary

At sizes large enough to amortise the SIMD loop setup (≥ 256 B):

| arch / ISA  | encode throughput | encode speedup | decode throughput | decode speedup |
|-------------|------------------:|---------------:|------------------:|---------------:|
| aarch64 NEON| 4.40 GiB/s        | 1.61×          | 4.49 GiB/s        | 10.49×         |
| x86_64 AVX2 | 2.07 GiB/s        | 2.71×          | 2.32 GiB/s        | 9.59×          |

The decode speedup ratio is roughly the same on both architectures
(~10×), driven by SIMD-accelerated ASCII → digit table lookup
replacing the reference's per-character branchy match. NEON sustains
roughly 2× the absolute throughput of AVX2 because its `vqtbl4q_u8`
does a 64-entry lookup in a single instruction, where x86 PSHUFB is
limited to 16 entries (so the lookup expands to ~6 PSHUFB+OR per
chunk on x86). AVX-512 VBMI's `vpermb` would close that gap but
isn't available on the AMD silicon used by GitHub's runner fleet.

Reproduce with:

```sh
cargo bench --bench encode
```

## License

Licensed under either of

- Apache License, Version 2.0
  ([LICENSE-APACHE](https://github.com/cipherstash/base85-simd/blob/main/LICENSE-APACHE)
  or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license
  ([LICENSE-MIT](https://github.com/cipherstash/base85-simd/blob/main/LICENSE-MIT)
  or <https://opensource.org/licenses/MIT>)

at your option.
