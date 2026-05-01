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
- Encode: NEON-accelerated on `aarch64`, scalar elsewhere.
- Decode: scalar (a NEON decode is planned but not yet shipped).
- Tested against the `base85` reference crate via quickcheck round-trip and
  parity checks.

## Benchmarks

Run with:

```sh
cargo bench --bench encode
```

Sample results (Apple M-series, 1 KB input, encode):

| Implementation | Time   |
|----------------|--------|
| `base85`       | 361 ns |
| `base85-simd`  | 236 ns |

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <https://opensource.org/licenses/MIT>)

at your option.
