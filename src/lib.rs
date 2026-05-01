//! Fast Base85 (RFC 1924 / Z85-style) encoding and decoding.
//!
//! On `aarch64` the encoder uses NEON intrinsics to process 16 input bytes
//! per iteration. On other architectures and on the decode path, a portable
//! scalar implementation is used. The output is byte-for-byte compatible
//! with the [`base85`](https://crates.io/crates/base85) crate.
//!
//! # Alphabet
//!
//! ```text
//! 0123456789
//! ABCDEFGHIJKLMNOPQRSTUVWXYZ
//! abcdefghijklmnopqrstuvwxyz
//! !#$%&()*+-;<=>?@^_`{|}~
//! ```
//!
//! # Tail handling
//!
//! Inputs whose length is not a multiple of 4 are encoded as
//! `floor(len / 4) * 5 + (len % 4) + 1` characters. The trailing partial
//! block is padded with zero bytes for encoding, and with the maximum digit
//! (`~` = 84) for decoding — the choice ensures the discarded low bytes
//! cannot perturb the kept high bytes.
//!
//! # Example
//!
//! ```
//! let data = b"hello, world!";
//! let encoded = base85_simd::encode(data);
//! let decoded = base85_simd::decode(&encoded).unwrap();
//! assert_eq!(decoded, data);
//! ```

#![deny(unsafe_op_in_unsafe_fn)]

mod block;
mod ops;

use block::{decode_block, decode_tail, encode_block, encode_tail};
use std::fmt;

/// Errors returned by [`decode`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// The input contained a byte that is not part of the base85 alphabet.
    InvalidChar {
        /// The offending byte.
        byte: u8,
        /// 0-based byte position within the input.
        position: usize,
    },
    /// The input length is not a valid base85-encoded length.
    ///
    /// Valid lengths satisfy `len % 5 ∈ {0, 2, 3, 4}` (a length of 1 modulo 5
    /// is impossible because partial blocks are at least 2 characters).
    InvalidLength {
        /// The actual length in bytes.
        len: usize,
    },
    /// A block decoded to a value greater than `u32::MAX`. This can only
    /// happen for adversarial input; valid encodings always fit.
    Overflow {
        /// 0-based byte position of the start of the offending block.
        position: usize,
    },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidChar { byte, position } => write!(
                f,
                "invalid base85 character {byte:#04x} at position {position}"
            ),
            Self::InvalidLength { len } => {
                write!(f, "invalid base85 input length: {len}")
            }
            Self::Overflow { position } => {
                write!(f, "base85 block at position {position} overflows u32")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// Encodes `input` as a base85 string.
///
/// The output length is `(len / 4) * 5` for whole blocks plus `(len % 4) + 1`
/// for any partial trailing block.
pub fn encode(input: &[u8]) -> String {
    let full_blocks = input.len() / 4;
    let tail_len = input.len() % 4;
    let out_len = full_blocks * 5 + if tail_len == 0 { 0 } else { tail_len + 1 };

    let mut out = vec![0u8; out_len];
    encode_into(input, &mut out);

    // SAFETY: `encode_into` writes only ASCII bytes from the base85 alphabet.
    debug_assert!(out.is_ascii());
    unsafe { String::from_utf8_unchecked(out) }
}

/// Decodes a base85 string into the original bytes.
pub fn decode(input: &str) -> Result<Vec<u8>, DecodeError> {
    let bytes = input.as_bytes();
    let (full_blocks_chars, tail_chars) = match bytes.len() % 5 {
        0 => (bytes.len(), 0),
        r @ 2..=4 => (bytes.len() - r, r),
        _ => return Err(DecodeError::InvalidLength { len: bytes.len() }),
    };

    let full_blocks = full_blocks_chars / 5;
    let tail_bytes = if tail_chars == 0 { 0 } else { tail_chars - 1 };
    let mut out = vec![0u8; full_blocks * 4 + tail_bytes];

    decode_into(bytes, &mut out)?;
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────
// Internal: pre-sized buffer fills
// ─────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
fn encode_into(input: &[u8], out: &mut [u8]) {
    use block::NeonEncoder;

    let encoder = NeonEncoder::new();
    let mut in_off = 0;
    let mut out_off = 0;

    // Process 16-byte chunks with NEON.
    while in_off + 16 <= input.len() {
        // SAFETY: bounds checked in the loop condition; out has capacity for
        // at least 20 more bytes since out_len = ceil(in_len / 4) * 5.
        unsafe {
            encoder.encode_block_x4(&input[in_off..], &mut out[out_off..]);
        }
        in_off += 16;
        out_off += 20;
    }

    // Process remaining 4-byte blocks scalar.
    while in_off + 4 <= input.len() {
        let block: &[u8; 4] = (&input[in_off..in_off + 4]).try_into().unwrap();
        let chunk: &mut [u8; 5] = (&mut out[out_off..out_off + 5]).try_into().unwrap();
        encode_block(block, chunk);
        in_off += 4;
        out_off += 5;
    }

    // Tail.
    let tail = &input[in_off..];
    if !tail.is_empty() {
        encode_tail(tail, &mut out[out_off..out_off + tail.len() + 1]);
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn encode_into(input: &[u8], out: &mut [u8]) {
    let mut in_off = 0;
    let mut out_off = 0;

    while in_off + 4 <= input.len() {
        let block: &[u8; 4] = (&input[in_off..in_off + 4]).try_into().unwrap();
        let chunk: &mut [u8; 5] = (&mut out[out_off..out_off + 5]).try_into().unwrap();
        encode_block(block, chunk);
        in_off += 4;
        out_off += 5;
    }

    let tail = &input[in_off..];
    if !tail.is_empty() {
        encode_tail(tail, &mut out[out_off..out_off + tail.len() + 1]);
    }
}

#[cfg(target_arch = "aarch64")]
fn decode_into(input: &[u8], out: &mut [u8]) -> Result<(), DecodeError> {
    use block::try_decode_block_x4;

    let mut in_off = 0;
    let mut out_off = 0;

    // Process 20-char (4-block) chunks with NEON. On any per-chunk failure
    // (invalid char or u32 overflow) we fall back to scalar for the whole
    // chunk so the precise DecodeError surfaces with the correct position.
    while in_off + 20 <= input.len() {
        // SAFETY: in_off + 20 <= input.len() and out has been sized so that
        // out_off + 16 <= out.len() whenever in_off + 20 <= input.len().
        let neon_ok =
            unsafe { try_decode_block_x4(&input[in_off..], &mut out[out_off..]) }.is_ok();
        if neon_ok {
            in_off += 20;
            out_off += 16;
        } else {
            for _ in 0..4 {
                let block: &[u8; 5] = (&input[in_off..in_off + 5]).try_into().unwrap();
                let chunk: &mut [u8; 4] = (&mut out[out_off..out_off + 4]).try_into().unwrap();
                decode_block(block, chunk, in_off)?;
                in_off += 5;
                out_off += 4;
            }
        }
    }

    while in_off + 5 <= input.len() {
        let block: &[u8; 5] = (&input[in_off..in_off + 5]).try_into().unwrap();
        let chunk: &mut [u8; 4] = (&mut out[out_off..out_off + 4]).try_into().unwrap();
        decode_block(block, chunk, in_off)?;
        in_off += 5;
        out_off += 4;
    }

    let tail = &input[in_off..];
    if !tail.is_empty() {
        decode_tail(tail, &mut out[out_off..out_off + tail.len() - 1], in_off)?;
    }

    Ok(())
}

#[cfg(not(target_arch = "aarch64"))]
fn decode_into(input: &[u8], out: &mut [u8]) -> Result<(), DecodeError> {
    let mut in_off = 0;
    let mut out_off = 0;

    while in_off + 5 <= input.len() {
        let block: &[u8; 5] = (&input[in_off..in_off + 5]).try_into().unwrap();
        let chunk: &mut [u8; 4] = (&mut out[out_off..out_off + 4]).try_into().unwrap();
        decode_block(block, chunk, in_off)?;
        in_off += 5;
        out_off += 4;
    }

    let tail = &input[in_off..];
    if !tail.is_empty() {
        decode_tail(tail, &mut out[out_off..out_off + tail.len() - 1], in_off)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::quickcheck;

    #[test]
    fn encode_matches_reference_full_block() {
        let input = vec![17, 33, 21, 13];
        assert_eq!(encode(&input), base85::encode(&input));
    }

    #[test]
    fn encode_empty() {
        assert_eq!(encode(&[]), "");
    }

    #[test]
    fn decode_empty() {
        assert_eq!(decode("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn round_trip_short_inputs() {
        for n in 0..=64 {
            let input: Vec<u8> = (0..n).map(|i| i as u8).collect();
            let encoded = encode(&input);
            assert_eq!(encoded, base85::encode(&input), "encode mismatch len={n}");
            let decoded = decode(&encoded).unwrap();
            assert_eq!(decoded, input, "round-trip mismatch len={n}");
        }
    }

    #[test]
    fn round_trip_long_inputs() {
        // Exercises both the NEON decode chunk path (20+ chars) and the
        // scalar tail. 1024 bytes → 1280 chars = 64 NEON chunks.
        for n in [20usize, 80, 256, 1024, 4096] {
            let input: Vec<u8> = (0..n).map(|i| (i.wrapping_mul(31)) as u8).collect();
            let encoded = encode(&input);
            assert_eq!(encoded, base85::encode(&input), "encode mismatch len={n}");
            let decoded = decode(&encoded).unwrap();
            assert_eq!(decoded, input, "round-trip mismatch len={n}");
        }
    }

    #[test]
    fn decode_matches_reference_for_long_input() {
        // Reference encodes, we decode — verifies our NEON decode against
        // an independent implementation, not just our own encode.
        let input: Vec<u8> = (0..1024u32).map(|i| (i * 17) as u8).collect();
        let encoded_by_ref = base85::encode(&input);
        let decoded_by_us = decode(&encoded_by_ref).unwrap();
        assert_eq!(decoded_by_us, input);
    }

    #[test]
    fn decode_invalid_char_in_neon_chunk() {
        // Inject an invalid char inside what would be the second NEON
        // chunk so the fallback path has to kick in mid-stream.
        let input: Vec<u8> = (0..40).map(|i| i as u8).collect();
        let mut encoded = encode(&input).into_bytes();
        encoded[25] = b' '; // position 25 is inside the 2nd 20-char chunk
        let s = std::str::from_utf8(&encoded).unwrap();
        let err = decode(s).unwrap_err();
        assert!(
            matches!(err, DecodeError::InvalidChar { byte: b' ', position: 25 }),
            "got {err:?}"
        );
    }

    #[test]
    fn decode_overflow_is_caught() {
        // "|NsC1" decodes to u32::MAX + 1 (since "|NsC0" encodes u32::MAX).
        // Pad with a valid block to land in the NEON path.
        let mut s = String::from("00000"); // valid leading block (0u32)
        s.push_str("|NsC1"); // overflowing block
        s.push_str("00000"); // valid trailing block
        s.push_str("00000"); // 4 blocks total → exactly one NEON chunk
        let err = decode(&s).unwrap_err();
        assert!(matches!(err, DecodeError::Overflow { .. }), "got {err:?}");
    }

    #[test]
    fn decode_invalid_char() {
        let err = decode("00 00").unwrap_err();
        assert!(matches!(
            err,
            DecodeError::InvalidChar {
                byte: b' ',
                position: 2
            }
        ));
    }

    #[test]
    fn decode_invalid_length() {
        // length % 5 == 1 is impossible for a valid encoding
        let err = decode("0").unwrap_err();
        assert!(matches!(err, DecodeError::InvalidLength { len: 1 }));
    }

    quickcheck! {
        fn round_trip_prop(input: Vec<u8>) -> bool {
            let encoded = encode(&input);
            decode(&encoded).map(|d| d == input).unwrap_or(false)
        }

        fn encode_matches_reference(input: Vec<u8>) -> bool {
            encode(&input) == base85::encode(&input)
        }
    }
}
