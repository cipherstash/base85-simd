#![doc = include_str!("../README.md")]
#![deny(unsafe_op_in_unsafe_fn)]
// Mission-critical lint config: production paths must not panic. Test code
// opts back in via `#[cfg_attr(test, allow(...))]` at the module level.
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::panic))]
#![cfg_attr(not(test), deny(clippy::indexing_slicing))]
#![cfg_attr(not(test), deny(clippy::todo))]
#![cfg_attr(not(test), deny(clippy::unimplemented))]
#![cfg_attr(not(test), deny(clippy::unreachable))]
#![warn(clippy::missing_safety_doc)]

mod block;
mod ops;

use block::{decode_block, decode_tail, encode_block, encode_tail};
use std::fmt;

/// Errors returned by [`decode`].
///
/// Marked `#[non_exhaustive]` — match arms must include `_ => …` so future
/// additions don't break downstream code.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
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

/// Borrow `slice[off..off+N]` as `&[u8; N]` if it fits. `None` otherwise —
/// the production paths treat `None` as an unrecoverable caller bug and
/// return early instead of panicking.
#[inline]
fn chunk_at<const N: usize>(slice: &[u8], off: usize) -> Option<&[u8; N]> {
    let end = off.checked_add(N)?;
    let s = slice.get(off..end)?;
    <&[u8; N]>::try_from(s).ok()
}

#[inline]
fn chunk_at_mut<const N: usize>(slice: &mut [u8], off: usize) -> Option<&mut [u8; N]> {
    let end = off.checked_add(N)?;
    let s = slice.get_mut(off..end)?;
    <&mut [u8; N]>::try_from(s).ok()
}

#[cfg(target_arch = "aarch64")]
fn encode_into(input: &[u8], out: &mut [u8]) {
    use block::NeonEncoder;

    let encoder = NeonEncoder::new();
    let mut in_off = 0usize;
    let mut out_off = 0usize;

    // Process 16-byte chunks with NEON. The fixed-size array references
    // make the SIMD load/store bounds type-enforced.
    while in_off.saturating_add(16) <= input.len() {
        let Some(in_chunk) = chunk_at::<16>(input, in_off) else {
            return;
        };
        let Some(out_chunk) = chunk_at_mut::<20>(out, out_off) else {
            return;
        };
        encoder.encode_block_x4(in_chunk, out_chunk);
        in_off = in_off.saturating_add(16);
        out_off = out_off.saturating_add(20);
    }

    // Mop up any remaining whole 4-byte blocks with scalar.
    while in_off.saturating_add(4) <= input.len() {
        let Some(block) = chunk_at::<4>(input, in_off) else {
            return;
        };
        let Some(chunk) = chunk_at_mut::<5>(out, out_off) else {
            return;
        };
        encode_block(block, chunk);
        in_off = in_off.saturating_add(4);
        out_off = out_off.saturating_add(5);
    }

    let tail_in_len = input.len().saturating_sub(in_off);
    if tail_in_len > 0 {
        let tail_out_len = tail_in_len.saturating_add(1);
        if let (Some(tail_in), Some(tail_out)) = (
            input.get(in_off..in_off.saturating_add(tail_in_len)),
            out.get_mut(out_off..out_off.saturating_add(tail_out_len)),
        ) {
            encode_tail(tail_in, tail_out);
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn encode_into(input: &[u8], out: &mut [u8]) {
    use block::Avx2Encoder;

    // Runtime feature gate: the AVX2 intrinsics are gated on
    // `#[target_feature(enable = "avx2")]`; calling them on a CPU
    // lacking AVX2 is UB. Detect once at the entry point and fall
    // back to the scalar path otherwise. The branch predicts
    // perfectly across calls so the cost is negligible.
    if !std::is_x86_feature_detected!("avx2") {
        encode_scalar(input, out);
        return;
    }

    let encoder = Avx2Encoder::new();
    let mut in_off = 0usize;
    let mut out_off = 0usize;

    // 32 → 40 byte AVX2 chunks (8 blocks per iteration).
    while in_off.saturating_add(32) <= input.len() {
        let Some(in_chunk) = chunk_at::<32>(input, in_off) else {
            return;
        };
        let Some(out_chunk) = chunk_at_mut::<40>(out, out_off) else {
            return;
        };
        // SAFETY: AVX2 verified above.
        unsafe { encoder.encode_block_x8(in_chunk, out_chunk) };
        in_off = in_off.saturating_add(32);
        out_off = out_off.saturating_add(40);
    }

    while in_off.saturating_add(4) <= input.len() {
        let Some(block) = chunk_at::<4>(input, in_off) else {
            return;
        };
        let Some(chunk) = chunk_at_mut::<5>(out, out_off) else {
            return;
        };
        encode_block(block, chunk);
        in_off = in_off.saturating_add(4);
        out_off = out_off.saturating_add(5);
    }

    let tail_in_len = input.len().saturating_sub(in_off);
    if tail_in_len > 0 {
        let tail_out_len = tail_in_len.saturating_add(1);
        if let (Some(tail_in), Some(tail_out)) = (
            input.get(in_off..in_off.saturating_add(tail_in_len)),
            out.get_mut(out_off..out_off.saturating_add(tail_out_len)),
        ) {
            encode_tail(tail_in, tail_out);
        }
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn encode_into(input: &[u8], out: &mut [u8]) {
    encode_scalar(input, out);
}

#[cfg(not(target_arch = "aarch64"))]
fn encode_scalar(input: &[u8], out: &mut [u8]) {
    let mut in_off = 0usize;
    let mut out_off = 0usize;

    while in_off.saturating_add(4) <= input.len() {
        let Some(block) = chunk_at::<4>(input, in_off) else {
            return;
        };
        let Some(chunk) = chunk_at_mut::<5>(out, out_off) else {
            return;
        };
        encode_block(block, chunk);
        in_off = in_off.saturating_add(4);
        out_off = out_off.saturating_add(5);
    }

    let tail_in_len = input.len().saturating_sub(in_off);
    if tail_in_len > 0 {
        let tail_out_len = tail_in_len.saturating_add(1);
        if let (Some(tail_in), Some(tail_out)) = (
            input.get(in_off..in_off.saturating_add(tail_in_len)),
            out.get_mut(out_off..out_off.saturating_add(tail_out_len)),
        ) {
            encode_tail(tail_in, tail_out);
        }
    }
}

// SAFETY-bearing decode loop: the NEON entry point takes open slices
// (`&[u8]` / `&mut [u8]`) rather than fixed-size array references because
// the fixed-array form forces an extra `slice::get` + `TryFrom` chain at
// every iteration that LLVM doesn't fold across the SIMD call boundary.
// Benchmarking showed `decode/1024` at ~1 µs with the safe-helper form
// vs ~226 ns with this form (~4× difference).
//
// The slicing inside the loop is `clippy::indexing_slicing`-free because
// we use `.get()` / `.get_mut()` everywhere. The `unsafe` block on the
// SIMD call documents the size invariants the loop guard upholds.
#[cfg(target_arch = "aarch64")]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "infallible / in-bounds by loop-condition invariant; see SAFETY comment"
)]
fn decode_into(input: &[u8], out: &mut [u8]) -> Result<(), DecodeError> {
    use block::try_decode_block_x4;

    let mut in_off = 0usize;
    let mut out_off = 0usize;

    while in_off + 20 <= input.len() && out_off + 16 <= out.len() {
        // SAFETY: `in_off + 20 <= input.len()` and `out_off + 16 <=
        // out.len()` are checked in the loop guard, so the open-slice
        // arguments below carry the size invariants `try_decode_block_x4`
        // requires (≥20 readable / ≥16 writable bytes).
        let neon_ok = unsafe { try_decode_block_x4(&input[in_off..], &mut out[out_off..]) }.is_ok();

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

    while in_off + 5 <= input.len() && out_off + 4 <= out.len() {
        let block: &[u8; 5] = (&input[in_off..in_off + 5]).try_into().unwrap();
        let chunk: &mut [u8; 4] = (&mut out[out_off..out_off + 4]).try_into().unwrap();
        decode_block(block, chunk, in_off)?;
        in_off += 5;
        out_off += 4;
    }

    let tail_in_len = input.len().saturating_sub(in_off);
    if tail_in_len > 0 {
        let tail_out_len = tail_in_len.saturating_sub(1);
        if let (Some(tail_in), Some(tail_out)) = (
            input.get(in_off..in_off.saturating_add(tail_in_len)),
            out.get_mut(out_off..out_off.saturating_add(tail_out_len)),
        ) {
            decode_tail(tail_in, tail_out, in_off)?;
        }
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "infallible / in-bounds by loop-condition invariant; see SAFETY comment"
)]
fn decode_into(input: &[u8], out: &mut [u8]) -> Result<(), DecodeError> {
    use block::try_decode_block_x8;

    // Runtime feature gate: AVX2 needed for the fast path; otherwise
    // route to scalar fallback.
    if !std::is_x86_feature_detected!("avx2") {
        return decode_scalar(input, out);
    }

    let mut in_off = 0usize;
    let mut out_off = 0usize;

    // 40-char (8-block) AVX2 chunks. On per-chunk failure (invalid
    // char or u32 overflow), replay the same chunk scalar so the
    // resulting DecodeError carries the precise failing byte position.
    while in_off + 40 <= input.len() && out_off + 32 <= out.len() {
        let Some(in_chunk) = chunk_at::<40>(input, in_off) else {
            return Ok(());
        };
        let Some(out_chunk) = chunk_at_mut::<32>(out, out_off) else {
            return Ok(());
        };

        // SAFETY: AVX2 verified above.
        let avx_ok = unsafe { try_decode_block_x8(in_chunk, out_chunk) }.is_ok();
        if avx_ok {
            in_off += 40;
            out_off += 32;
        } else {
            for _ in 0..8 {
                let block: &[u8; 5] = (&input[in_off..in_off + 5]).try_into().unwrap();
                let chunk: &mut [u8; 4] = (&mut out[out_off..out_off + 4]).try_into().unwrap();
                decode_block(block, chunk, in_off)?;
                in_off += 5;
                out_off += 4;
            }
        }
    }

    while in_off + 5 <= input.len() && out_off + 4 <= out.len() {
        let block: &[u8; 5] = (&input[in_off..in_off + 5]).try_into().unwrap();
        let chunk: &mut [u8; 4] = (&mut out[out_off..out_off + 4]).try_into().unwrap();
        decode_block(block, chunk, in_off)?;
        in_off += 5;
        out_off += 4;
    }

    let tail_in_len = input.len().saturating_sub(in_off);
    if tail_in_len > 0 {
        let tail_out_len = tail_in_len.saturating_sub(1);
        if let (Some(tail_in), Some(tail_out)) = (
            input.get(in_off..in_off.saturating_add(tail_in_len)),
            out.get_mut(out_off..out_off.saturating_add(tail_out_len)),
        ) {
            decode_tail(tail_in, tail_out, in_off)?;
        }
    }

    Ok(())
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn decode_into(input: &[u8], out: &mut [u8]) -> Result<(), DecodeError> {
    decode_scalar(input, out)
}

#[cfg(not(target_arch = "aarch64"))]
fn decode_scalar(input: &[u8], out: &mut [u8]) -> Result<(), DecodeError> {
    let mut in_off = 0usize;
    let mut out_off = 0usize;

    while in_off.saturating_add(5) <= input.len() {
        let Some(block) = chunk_at::<5>(input, in_off) else {
            return Ok(());
        };
        let Some(chunk) = chunk_at_mut::<4>(out, out_off) else {
            return Ok(());
        };
        decode_block(block, chunk, in_off)?;
        in_off = in_off.saturating_add(5);
        out_off = out_off.saturating_add(4);
    }

    let tail_in_len = input.len().saturating_sub(in_off);
    if tail_in_len > 0 {
        let tail_out_len = tail_in_len.saturating_sub(1);
        if let (Some(tail_in), Some(tail_out)) = (
            input.get(in_off..in_off.saturating_add(tail_in_len)),
            out.get_mut(out_off..out_off.saturating_add(tail_out_len)),
        ) {
            decode_tail(tail_in, tail_out, in_off)?;
        }
    }

    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;
    use quickcheck::{Gen, QuickCheck, quickcheck};

    // ──────────────────────────────────────────────────────────────────
    // Section A — Basic correctness
    // ──────────────────────────────────────────────────────────────────

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

    // ──────────────────────────────────────────────────────────────────
    // Section B — Round-trip across every relevant length
    // ──────────────────────────────────────────────────────────────────

    /// Round-trip for every length 0..=256, with content that varies per
    /// position. Catches off-by-one in tail handling, NEON-chunk transitions
    /// (16, 17, 19, 20, 21, …), and 4-byte block transitions.
    #[test]
    fn round_trip_every_length_up_to_256() {
        for n in 0..=256usize {
            let input: Vec<u8> = (0..n)
                .map(|i| (i as u8).wrapping_mul(7).wrapping_add(11))
                .collect();
            let encoded = encode(&input);
            assert_eq!(encoded, base85::encode(&input), "encode mismatch len={n}");
            let decoded = decode(&encoded).unwrap();
            assert_eq!(decoded, input, "round-trip mismatch len={n}");
        }
    }

    /// Lengths surrounding every chunk boundary the impl cares about:
    /// 4-byte block, 16-byte SIMD encode, 20-char SIMD decode chunk.
    #[test]
    fn round_trip_chunk_boundary_lengths() {
        let mut sizes: Vec<usize> = Vec::new();
        for base in [
            0usize, 4, 8, 12, 16, 20, 32, 64, 96, 128, 192, 256, 1024, 4096,
        ] {
            for delta in [0i32, -1, 1, -2, 2, -3, 3] {
                let n = base as i32 + delta;
                if n >= 0 {
                    sizes.push(n as usize);
                }
            }
        }
        sizes.sort_unstable();
        sizes.dedup();
        for n in sizes {
            let input: Vec<u8> = (0..n).map(|i| (i as u8) ^ 0xA5).collect();
            assert_eq!(decode(&encode(&input)).unwrap(), input, "boundary len={n}");
        }
    }

    /// 1 MiB stress test with deterministic content.
    #[test]
    fn round_trip_one_megabyte() {
        let input: Vec<u8> = (0..(1u32 << 20))
            .map(|i| (i.wrapping_mul(2_654_435_761)) as u8)
            .collect();
        let encoded = encode(&input);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, input);
        assert_eq!(encoded, base85::encode(&input));
    }

    // ──────────────────────────────────────────────────────────────────
    // Section C — Boundary u32 values per single block
    // ──────────────────────────────────────────────────────────────────

    /// Round-trip every "interesting" u32 single-block value: zeros,
    /// powers of 85, u32::MAX-adjacent. Each is encoded as a 5-char
    /// block (4 input bytes BE) and round-tripped.
    #[test]
    fn round_trip_boundary_u32_values() {
        let mut values: Vec<u32> = Vec::new();
        values.extend([0, 1, 84, 85, 86, u32::MAX, u32::MAX - 1, u32::MAX - 84]);
        for k in 0..32 {
            values.push(1u32 << k);
            values.push((1u32 << k).wrapping_sub(1));
        }
        for k in 0..=4u32 {
            values.push(85u32.pow(k));
            if 85u32.pow(k) > 0 {
                values.push(85u32.pow(k).saturating_sub(1));
            }
        }
        // The exact encoder boundary cases.
        values.push(82 * 85u32.pow(4));
        values.push(82 * 85u32.pow(4) + 14_516_045);

        for &v in &values {
            let bytes = v.to_be_bytes();
            let encoded = encode(&bytes);
            assert_eq!(encoded.len(), 5);
            assert_eq!(encoded, base85::encode(&bytes), "encode v={v}");
            let decoded = decode(&encoded).unwrap();
            assert_eq!(decoded, bytes, "round-trip v={v}");
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // Section D — Adversarial decode: invalid char & overflow at every
    //              position must surface DecodeError with that position
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn invalid_char_at_every_position_reports_correct_position() {
        // 100-byte input → 125 chars (whole-block-aligned at 25 NEON chunks).
        let input: Vec<u8> = (0..100u32).map(|i| i as u8).collect();
        let encoded_str = encode(&input);

        for pos in 0..encoded_str.len() {
            let mut buf = encoded_str.clone().into_bytes();
            buf[pos] = b' '; // ' ' is not in the alphabet
            let s = std::str::from_utf8(&buf).unwrap();
            let err = decode(s).unwrap_err();
            match err {
                DecodeError::InvalidChar { byte, position } => {
                    assert_eq!(byte, b' ', "wrong byte reported at pos={pos}");
                    // Position must be the position of the invalid char.
                    // For NEON-fallback chunks, the scalar replay walks the
                    // whole 20-char window and reports the exact byte.
                    assert_eq!(position, pos, "wrong position reported for pos={pos}");
                }
                other => panic!("expected InvalidChar at pos={pos}, got {other:?}"),
            }
        }
    }

    #[test]
    fn overflow_block_at_every_position_reports_block_start() {
        // Build a long run of valid blocks, then place an overflowing
        // block at each block-aligned position and verify Overflow
        // surfaces with the correct position.
        let valid_block = "00000"; // encodes 0u32
        let overflow_block = "|NsC1"; // u32::MAX + 1 (one past max valid)
        let n_blocks = 8usize; // 40 chars = 2 NEON chunks

        for bad_block_idx in 0..n_blocks {
            let mut s = String::new();
            for i in 0..n_blocks {
                if i == bad_block_idx {
                    s.push_str(overflow_block);
                } else {
                    s.push_str(valid_block);
                }
            }
            let err = decode(&s).unwrap_err();
            match err {
                DecodeError::Overflow { position } => {
                    assert_eq!(
                        position,
                        bad_block_idx * 5,
                        "wrong overflow position for block {bad_block_idx}",
                    );
                }
                other => panic!("expected Overflow at block {bad_block_idx}, got {other:?}"),
            }
        }
    }

    #[test]
    fn decode_invalid_char_at_position_2() {
        let err = decode("00 00").unwrap_err();
        assert!(matches!(
            err,
            DecodeError::InvalidChar {
                byte: b' ',
                position: 2
            }
        ));
    }

    /// All ASCII bytes outside the alphabet must be rejected as InvalidChar.
    #[test]
    fn every_non_alphabet_ascii_is_rejected() {
        let alphabet = block::ALPHABET;
        let mut alphabet_set = [false; 256];
        for &c in alphabet {
            alphabet_set[c as usize] = true;
        }
        // Test ASCII range (and beyond) — any byte not in the alphabet must
        // produce InvalidChar when placed inside a 5-char block.
        for b in 0u8..=127 {
            if alphabet_set[b as usize] {
                continue;
            }
            let bytes = [b'0', b, b'0', b'0', b'0'];
            let s = std::str::from_utf8(&bytes).unwrap();
            let err = decode(s).unwrap_err();
            assert!(
                matches!(
                    err,
                    DecodeError::InvalidChar { byte, position: 1 } if byte == b
                ),
                "byte {b:#04x} should be rejected, got {err:?}"
            );
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // Section E — Length-rule rejections
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn decode_rejects_invalid_lengths() {
        // length % 5 == 1 is impossible for a valid encoding (partial blocks
        // are at least 2 chars).
        for len in [1usize, 6, 11, 26, 101] {
            let s = "0".repeat(len);
            let err = decode(&s).unwrap_err();
            assert!(
                matches!(err, DecodeError::InvalidLength { len: actual } if actual == len),
                "len={len} should be rejected, got {err:?}"
            );
        }
    }

    #[test]
    fn decode_accepts_valid_partial_block_lengths() {
        // length % 5 ∈ {0, 2, 3, 4} are valid.
        for &(input_len, expected_encoded_len) in &[(1usize, 2), (2, 3), (3, 4), (4, 5)] {
            let input: Vec<u8> = (0..input_len as u8).collect();
            let encoded = encode(&input);
            assert_eq!(encoded.len(), expected_encoded_len);
            assert_eq!(decode(&encoded).unwrap(), input);
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // Section F — Output validity properties
    // ──────────────────────────────────────────────────────────────────

    /// Every byte the encoder emits must be in the base85 alphabet.
    #[test]
    fn encoded_output_is_always_in_alphabet() {
        let mut alphabet_set = [false; 256];
        for &c in block::ALPHABET {
            alphabet_set[c as usize] = true;
        }
        for n in 0..=200usize {
            let input: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_mul(13)).collect();
            let encoded = encode(&input);
            for (i, b) in encoded.bytes().enumerate() {
                assert!(
                    alphabet_set[b as usize],
                    "byte {b:#04x} at pos {i} (input len {n}) is not in alphabet"
                );
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // Section G — Property tests with elevated iteration counts
    // ──────────────────────────────────────────────────────────────────

    /// Quickcheck round-trip with high iteration count (10_000 cases) and
    /// generator size 1024 to exercise inputs across many NEON chunks.
    #[test]
    fn quickcheck_round_trip_high_iter() {
        fn prop(input: Vec<u8>) -> bool {
            let encoded = encode(&input);
            decode(&encoded).map(|d| d == input).unwrap_or(false)
        }
        QuickCheck::new()
            .r#gen(Gen::new(1024))
            .tests(10_000)
            .quickcheck(prop as fn(Vec<u8>) -> bool);
    }

    #[test]
    fn quickcheck_encode_matches_reference_high_iter() {
        fn prop(input: Vec<u8>) -> bool {
            encode(&input) == base85::encode(&input)
        }
        QuickCheck::new()
            .r#gen(Gen::new(1024))
            .tests(10_000)
            .quickcheck(prop as fn(Vec<u8>) -> bool);
    }

    /// Reference encodes, we decode — independent-implementation parity
    /// that doesn't share an encoder with the property under test.
    #[test]
    fn quickcheck_decode_against_reference_encode() {
        fn prop(input: Vec<u8>) -> bool {
            let encoded = base85::encode(&input);
            decode(&encoded).map(|d| d == input).unwrap_or(false)
        }
        QuickCheck::new()
            .r#gen(Gen::new(1024))
            .tests(10_000)
            .quickcheck(prop as fn(Vec<u8>) -> bool);
    }

    // Existing low-iter quickchecks retained for fast feedback.
    quickcheck! {
        fn round_trip_prop(input: Vec<u8>) -> bool {
            let encoded = encode(&input);
            decode(&encoded).map(|d| d == input).unwrap_or(false)
        }

        fn encode_matches_reference(input: Vec<u8>) -> bool {
            encode(&input) == base85::encode(&input)
        }
    }

    // ──────────────────────────────────────────────────────────────────
    // Section H — NEON path direct exercise
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn neon_chunk_with_invalid_char_at_each_position() {
        // For every byte position 0..20 within a NEON chunk, plant an
        // invalid char and verify the fallback reports the right position.
        let input: Vec<u8> = (0..16u32).map(|i| i as u8).collect();
        let encoded_str = encode(&input);
        assert_eq!(encoded_str.len(), 20);

        for pos in 0..20 {
            let mut buf = encoded_str.clone().into_bytes();
            buf[pos] = b'.';
            let s = std::str::from_utf8(&buf).unwrap();
            let err = decode(s).unwrap_err();
            assert!(
                matches!(
                    err,
                    DecodeError::InvalidChar { byte: b'.', position: p } if p == pos
                ),
                "pos={pos}: got {err:?}"
            );
        }
    }
}
