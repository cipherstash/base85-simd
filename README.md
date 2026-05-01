# base85-simd

Fast Base85 (RFC 1924 / Z85-style) encoder and decoder for Rust.

On `aarch64` the encode path uses NEON intrinsics to process 16 input bytes
per iteration; the decode path and the fallback for other architectures use
a portable scalar implementation. Output is byte-for-byte compatible with
the [`base85`](https://crates.io/crates/base85) crate.

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
- Encode and decode are both NEON-accelerated on `aarch64`, scalar elsewhere.
- The decode path validates char range and detects `u32` overflow lane-wise;
  any invalid input falls back to the scalar path so the resulting
  `DecodeError` carries a precise byte position.
- Tested against the `base85` reference crate via quickcheck round-trip and
  parity checks.

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

Apple M-series (aarch64), `cargo bench --bench encode`, criterion, release
profile, single-threaded. Times are the criterion-reported median; throughput
is computed from the median.

### Encode

| size  | `base85` time | `base85-simd` time | speedup | `base85-simd` throughput |
|-------|--------------:|-------------------:|--------:|-------------------------:|
| 16 B  | 17.4 ns       | 16.3 ns            | 1.07×   | ~940 MiB/s               |
| 64 B  | 33.0 ns       | 22.6 ns            | 1.46×   | 2.71 GiB/s               |
| 256 B | 115.7 ns      | 73.2 ns            | 1.58×   | 3.26 GiB/s               |
| 1 KiB | 378.3 ns      | 225.1 ns           | 1.68×   | 4.24 GiB/s               |
| 16 KiB| 5.56 µs       | 3.60 µs            | 1.55×   | 4.24 GiB/s               |
| 256 KiB| 89.3 µs      | 55.4 µs            | 1.61×   | 4.41 GiB/s               |
| 1 MiB | 356 µs        | 222 µs             | 1.61×   | 4.40 GiB/s               |

### Decode

| size  | `base85` time | `base85-simd` time | speedup | `base85-simd` throughput |
|-------|--------------:|-------------------:|--------:|-------------------------:|
| 16 B  | 32.4 ns       | 14.8 ns            | 2.18×   | ~1.0 GiB/s               |
| 64 B  | 123.5 ns      | 24.6 ns            | 5.02×   | 2.42 GiB/s               |
| 256 B | 579 ns        | 57.6 ns            | 10.05×  | 4.14 GiB/s               |
| 1 KiB | 2.27 µs       | 226 ns             | 10.06×  | 4.22 GiB/s               |
| 16 KiB| 36.8 µs       | 3.49 µs            | 10.55×  | 4.38 GiB/s               |
| 256 KiB| 591 µs       | 54.3 µs            | 10.89×  | 4.50 GiB/s               |
| 1 MiB | 2.28 ms       | 217.6 µs           | 10.49×  | 4.49 GiB/s               |

### Steady-state summary

At sizes large enough to amortise loop setup (≥ 256 B), `base85-simd`
sustains **~4.4 GiB/s** for both encode and decode on Apple M-series,
roughly **1.6× faster** than the reference for encode and **~10×
faster** for decode. The decode advantage comes from `TBL`-based ASCII
→ digit lookup that replaces the reference's per-character branchy
match; the encode advantage comes from 4-lane parallel `divmod 85`
plus `TBX`-based digit → ASCII / output permutation.

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
