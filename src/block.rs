//! Block-level encoders and decoders.
//!
//! - [`encode_block`] / [`decode_block`] handle one 4-byte ↔ 5-char block.
//! - [`encode_tail`] / [`decode_tail`] handle the trailing partial block.
//! - On aarch64, [`NeonEncoder`] processes four 4-byte blocks at a time.
//!
//! All callers in `lib.rs` are responsible for passing correctly-sized
//! slices; these functions assert their preconditions in debug builds.

use crate::DecodeError;

pub(crate) const ALPHABET: &[u8; 85] =
    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~";

#[inline]
pub(crate) fn byte_to_char85(x85: u8) -> u8 {
    debug_assert!(x85 < 85);
    ALPHABET[x85 as usize]
}

#[inline]
pub(crate) fn char85_to_byte(c: u8, position: usize) -> Result<u8, DecodeError> {
    let v = match c {
        b'0'..=b'9' => c - b'0',
        b'A'..=b'Z' => c - b'A' + 10,
        b'a'..=b'z' => c - b'a' + 36,
        b'!' => 62,
        b'#' => 63,
        b'$' => 64,
        b'%' => 65,
        b'&' => 66,
        b'(' => 67,
        b')' => 68,
        b'*' => 69,
        b'+' => 70,
        b'-' => 71,
        b';' => 72,
        b'<' => 73,
        b'=' => 74,
        b'>' => 75,
        b'?' => 76,
        b'@' => 77,
        b'^' => 78,
        b'_' => 79,
        b'`' => 80,
        b'{' => 81,
        b'|' => 82,
        b'}' => 83,
        b'~' => 84,
        _ => return Err(DecodeError::InvalidChar { byte: c, position }),
    };
    Ok(v)
}

#[inline]
pub(crate) fn encode_block(input: &[u8; 4], out: &mut [u8; 5]) {
    let mut n = u32::from_be_bytes(*input);
    out[4] = byte_to_char85((n % 85) as u8);
    n /= 85;
    out[3] = byte_to_char85((n % 85) as u8);
    n /= 85;
    out[2] = byte_to_char85((n % 85) as u8);
    n /= 85;
    out[1] = byte_to_char85((n % 85) as u8);
    n /= 85;
    // n is now < 85 (since u32::MAX < 85^5 = 4_437_053_125)
    out[0] = byte_to_char85(n as u8);
}

#[inline]
pub(crate) fn decode_block(
    input: &[u8; 5],
    out: &mut [u8; 4],
    base_position: usize,
) -> Result<(), DecodeError> {
    let mut n: u32 = 0;
    for (i, &c) in input.iter().enumerate() {
        let d = char85_to_byte(c, base_position + i)? as u32;
        // checked because the maximum representable value in 5 base-85 digits
        // (84 * (85^4 + 85^3 + ...) = 4_437_053_040) exceeds u32::MAX.
        n = n
            .checked_mul(85)
            .and_then(|v| v.checked_add(d))
            .ok_or(DecodeError::Overflow {
                position: base_position,
            })?;
    }
    *out = n.to_be_bytes();
    Ok(())
}

/// Encodes a 1..=3 byte tail. Output length is `input.len() + 1`.
///
/// The tail is padded with zero bytes to form a full block, encoded, and
/// only the first `input.len() + 1` characters are emitted.
#[inline]
pub(crate) fn encode_tail(input: &[u8], out: &mut [u8]) {
    debug_assert!((1..=3).contains(&input.len()));
    debug_assert_eq!(out.len(), input.len() + 1);

    let mut padded = [0u8; 4];
    padded[..input.len()].copy_from_slice(input);
    let mut full = [0u8; 5];
    encode_block(&padded, &mut full);
    out.copy_from_slice(&full[..input.len() + 1]);
}

/// Decodes a 2..=4 character tail. Output length is `input.len() - 1`.
///
/// The tail is padded with `~` (digit value 84, the maximum) to form a full
/// 5-char block, decoded, and only the first `input.len() - 1` bytes are kept.
/// The padding choice is arithmetically chosen so that the kept bytes are
/// independent of the padding (see crate-level docs).
#[inline]
pub(crate) fn decode_tail(
    input: &[u8],
    out: &mut [u8],
    base_position: usize,
) -> Result<(), DecodeError> {
    debug_assert!((2..=4).contains(&input.len()));
    debug_assert_eq!(out.len(), input.len() - 1);

    let mut padded = [b'~'; 5];
    padded[..input.len()].copy_from_slice(input);
    let mut full = [0u8; 4];
    decode_block(&padded, &mut full, base_position)?;
    out.copy_from_slice(&full[..input.len() - 1]);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// NEON encoder (aarch64)
// ─────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub(crate) use neon::NeonEncoder;

#[cfg(target_arch = "aarch64")]
mod neon {
    use crate::ops::quotients_and_remainders;
    use std::arch::aarch64::{
        uint8x16_t, uint32x4_t, uint32x4x2_t, vbslq_u8, vcltq_u8, vdupq_n_u8, vextq_u8, vld1q_u32,
        vld1q_u8, vld1q_u8_x2, vld1q_u8_x4, vqtbl2q_u8, vqtbl4q_u8, vqtbx1q_u8,
        vreinterpretq_u32_u8, vreinterpretq_u8_u32, vrev32q_u8, vst1q_lane_u32, vst1q_u8, vsubq_u8,
    };

    /// Encodes 16 input bytes into 20 output characters using NEON.
    ///
    /// Pre-computes table-lookup index vectors once at construction so
    /// the per-block work is just four `divmod 85` SIMD steps and two
    /// table merges.
    pub(crate) struct NeonEncoder {
        indexes_1: [uint8x16_t; 5],
        indexes_2: [uint8x16_t; 4],
    }

    impl NeonEncoder {
        pub(crate) fn new() -> Self {
            // Index pattern: each output 5-char block writes 4 chars into
            // out_1 (the main 16-byte vector) and 1 char into out_2 (the
            // overflow 4-byte vector). The offsets-of-4 within out_1 are
            // computed by `vextq_u8`-rotating a base index vector.
            let index_1_base: [u8; 16] = [
                0, 0xFF, 0xFF, 0xFF, 0xFF, 4, 0xFF, 0xFF, 0xFF, 0xFF, 8, 0xFF, 0xFF, 0xFF, 0xFF, 12,
            ];
            let index_1_base: uint8x16_t = unsafe { vld1q_u8(index_1_base.as_ptr()) };
            let blanks: uint8x16_t = unsafe { vdupq_n_u8(0xFF) };

            let index_4_1 = unsafe { vextq_u8(blanks, index_1_base, 12) };
            let index_3_1 = unsafe { vextq_u8(blanks, index_1_base, 13) };
            let index_2_1 = unsafe { vextq_u8(blanks, index_1_base, 14) };
            let index_1_1 = unsafe { vextq_u8(blanks, index_1_base, 15) };

            let index_4_2 = unsafe { vextq_u8(index_1_base, blanks, 12) };
            let index_3_2 = unsafe { vextq_u8(index_4_2, index_4_2, 1) };
            let index_2_2 = unsafe { vextq_u8(index_3_2, index_3_2, 1) };
            let index_1_2 = unsafe { vextq_u8(index_2_2, index_2_2, 1) };

            Self {
                indexes_1: [index_4_1, index_3_1, index_2_1, index_1_1, index_1_base],
                indexes_2: [index_4_2, index_3_2, index_2_2, index_1_2],
            }
        }

        /// Encode exactly 16 input bytes into exactly 20 output bytes.
        ///
        /// # Safety
        ///
        /// Caller must ensure `in_bytes.len() >= 16` and `out_bytes.len() >= 20`.
        #[inline]
        pub(crate) unsafe fn encode_block_x4(&self, in_bytes: &[u8], out_bytes: &mut [u8]) {
            debug_assert!(in_bytes.len() >= 16);
            debug_assert!(out_bytes.len() >= 20);

            let mut input = unsafe { load_u32x4_be(in_bytes) };
            let mut out_1: uint8x16_t = unsafe { vdupq_n_u8(0) };
            let mut out_2: uint8x16_t = unsafe { vdupq_n_u8(0) };

            for i in 0..4 {
                let uint32x4x2_t(quotients, rem) = quotients_and_remainders(input);
                let rem = unsafe { vreinterpretq_u8_u32(rem) };
                out_1 = unsafe { vqtbx1q_u8(out_1, rem, self.indexes_1[i]) };
                out_2 = unsafe { vqtbx1q_u8(out_2, rem, self.indexes_2[i]) };
                input = quotients;
            }

            // Final digit (leading char in each block) — `input` is already
            // < 85 lane-wise so we just splice it in.
            let final_digit = unsafe { vreinterpretq_u8_u32(input) };
            out_1 = unsafe { vqtbx1q_u8(out_1, final_digit, self.indexes_1[4]) };

            out_1 = unsafe { byte_to_char85_neon(out_1) };
            out_2 = unsafe { byte_to_char85_neon(out_2) };

            unsafe {
                vst1q_u8(out_bytes.as_mut_ptr(), out_1);
                let out_2_u32 = vreinterpretq_u32_u8(out_2);
                vst1q_lane_u32::<0>(out_bytes.as_mut_ptr().add(16) as *mut u32, out_2_u32);
            }
        }
    }

    /// Maps 16 lane-wise digit values (0..85) to ASCII characters.
    #[inline]
    unsafe fn byte_to_char85_neon(x85: uint8x16_t) -> uint8x16_t {
        // 64 chars (fits in vqtbl4) + 21 chars (padded to 32, fits in vqtbl2).
        // The padding bytes ('#') in B85_X2 are unreachable when x85 < 85.
        static B85_X4: [u8; 64] =
            *b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#";
        static B85_X2: [u8; 32] = *b"$%&()*+-;<=>?@^_`{|}~###########";

        let t1 = unsafe { vld1q_u8_x4(B85_X4.as_ptr()) };
        let t2 = unsafe { vld1q_u8_x2(B85_X2.as_ptr()) };

        let mask_lt64 = unsafe { vcltq_u8(x85, vdupq_n_u8(64)) };
        let res_lo = unsafe { vqtbl4q_u8(t1, x85) };
        let index_hi = unsafe { vsubq_u8(x85, vdupq_n_u8(64)) };
        let res_hi = unsafe { vqtbl2q_u8(t2, index_hi) };
        unsafe { vbslq_u8(mask_lt64, res_lo, res_hi) }
    }

    /// Loads 16 bytes and reinterprets as 4 big-endian u32s. Compiles to a
    /// single `ldr q + rev32.16b` pair.
    #[inline]
    unsafe fn load_u32x4_be(bytes: &[u8]) -> uint32x4_t {
        unsafe {
            let raw = vld1q_u32(bytes.as_ptr() as *const u32);
            let raw_bytes = vreinterpretq_u8_u32(raw);
            let swapped = vrev32q_u8(raw_bytes);
            vreinterpretq_u32_u8(swapped)
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_block_round_trip() {
        for &input in &[
            [0u8, 0, 0, 0],
            [0, 1, 2, 3],
            [255, 255, 255, 255],
            [17, 33, 21, 13],
        ] {
            let mut encoded = [0u8; 5];
            encode_block(&input, &mut encoded);
            let mut decoded = [0u8; 4];
            decode_block(&encoded, &mut decoded, 0).expect("valid encoding");
            assert_eq!(decoded, input);
        }
    }

    #[test]
    fn encode_tail_lengths() {
        for n in 1..=3 {
            let input: Vec<u8> = (0u8..n as u8).collect();
            let mut out = vec![0u8; n + 1];
            encode_tail(&input, &mut out);
            // base85 reference comparison
            let reference = base85::encode(&input);
            assert_eq!(out.as_slice(), reference.as_bytes(), "len={n}");
        }
    }

    #[test]
    fn decode_tail_round_trip() {
        for n in 1..=3usize {
            let input: Vec<u8> = (0u8..=255).step_by(if n == 0 { 1 } else { 256 / n + 1 }).take(n).collect();
            let encoded = base85::encode(&input);
            let mut out = vec![0u8; n];
            decode_tail(encoded.as_bytes(), &mut out, 0).expect("valid encoding");
            assert_eq!(out, input);
        }
    }

    #[test]
    fn decode_block_invalid_char() {
        let bad: [u8; 5] = *b"00 00";
        let mut out = [0u8; 4];
        let err = decode_block(&bad, &mut out, 100).unwrap_err();
        match err {
            DecodeError::InvalidChar { byte, position } => {
                assert_eq!(byte, b' ');
                assert_eq!(position, 102);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
