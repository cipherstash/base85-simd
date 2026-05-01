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
pub(crate) use neon::{NeonEncoder, try_decode_block_x4};

#[cfg(target_arch = "aarch64")]
mod neon {
    use crate::ops::quotients_and_remainders;
    use std::arch::aarch64::{
        uint8x16_t, uint8x16x2_t, uint32x4_t, uint32x4x2_t, vandq_u8, vbslq_u8, vceqq_u8,
        vcgtq_u8, vcltq_u32, vcltq_u8, vdupq_n_u8, vextq_u8, vld1q_u32, vld1q_u8, vld1q_u8_x2,
        vld1q_u8_x4, vmaxvq_u32, vmaxvq_u8, vmlaq_n_u32, vmulq_n_u32, vorrq_u8, vqtbl2q_u8,
        vqtbl4q_u8, vqtbx1q_u8, vreinterpretq_u32_u8, vreinterpretq_u8_u32, vrev32q_u8,
        vst1q_lane_u32, vst1q_u8, vsubq_u8,
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

    /// Attempts to decode 20 input chars into 16 output bytes using NEON.
    ///
    /// Returns `Err(())` (so the caller falls back to the scalar path,
    /// which will surface a precise [`crate::DecodeError`]) when:
    /// - any input byte is outside the base85 alphabet, or
    /// - any of the four 5-char blocks decodes to a value > `u32::MAX`.
    ///
    /// On success, exactly 16 big-endian bytes are written to `out`.
    ///
    /// # Safety
    ///
    /// Caller must ensure `input.len() >= 20` and `out.len() >= 16`.
    #[inline]
    pub(crate) unsafe fn try_decode_block_x4(input: &[u8], out: &mut [u8]) -> Result<(), ()> {
        debug_assert!(input.len() >= 20);
        debug_assert!(out.len() >= 16);

        // Load chars 0..16 directly; load chars 16..19 into the low 4 lanes
        // of a zero-padded vector so we can use a unified TBL pipeline.
        let chars_main = unsafe { vld1q_u8(input.as_ptr()) };
        let mut tail_buf = [0u8; 16];
        tail_buf[..4].copy_from_slice(&input[16..20]);
        let chars_tail = unsafe { vld1q_u8(tail_buf.as_ptr()) };

        // Mask: 0xFF for the 4 real-input lanes of `chars_tail`, 0 for the
        // 12 padding lanes — used to suppress validation noise from padding.
        static TAIL_VALID_MASK: [u8; 16] = [
            0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let tail_valid_mask = unsafe { vld1q_u8(TAIL_VALID_MASK.as_ptr()) };

        // Char range validation: any byte not in [33, 126] is invalid.
        let too_low_main = unsafe { vcltq_u8(chars_main, vdupq_n_u8(33)) };
        let too_high_main = unsafe { vcgtq_u8(chars_main, vdupq_n_u8(126)) };
        let invalid_range_main = unsafe { vorrq_u8(too_low_main, too_high_main) };

        let too_low_tail = unsafe { vcltq_u8(chars_tail, vdupq_n_u8(33)) };
        let too_high_tail = unsafe { vcgtq_u8(chars_tail, vdupq_n_u8(126)) };
        let invalid_range_tail =
            unsafe { vandq_u8(vorrq_u8(too_low_tail, too_high_tail), tail_valid_mask) };

        // ASCII → digit lookup. In-range-but-not-in-alphabet chars come back
        // as 0xFF; we OR those into the invalid-mask. Out-of-range chars
        // index past the table and yield 0 (which we'd otherwise mistake
        // for digit '0'), hence the explicit range check above.
        let digits_main = unsafe { char85_to_byte_neon(chars_main) };
        let digits_tail = unsafe { char85_to_byte_neon(chars_tail) };

        let invalid_dig_main = unsafe { vceqq_u8(digits_main, vdupq_n_u8(0xFF)) };
        let invalid_dig_tail =
            unsafe { vandq_u8(vceqq_u8(digits_tail, vdupq_n_u8(0xFF)), tail_valid_mask) };

        let any_invalid = unsafe {
            vorrq_u8(
                vorrq_u8(invalid_range_main, invalid_range_tail),
                vorrq_u8(invalid_dig_main, invalid_dig_tail),
            )
        };
        if unsafe { vmaxvq_u8(any_invalid) } != 0 {
            return Err(());
        }

        // Permute the 20 digits into five u32x4 vectors v_d0..v_d4 where
        // lane i of v_dk is the k-th digit of block i.
        //
        // Block layout (chars 0..19): a=0..4, b=5..9, c=10..14, d=15..19.
        // For each digit position k, we take bytes (k, k+5, k+10, k+15)
        // from the concatenation of digits_main (indices 0..15) and
        // digits_tail (indices 16..31) using vqtbl2q_u8. The 0xFF lanes
        // in the index produce 0 in the result, leaving each digit as
        // the low byte of a u32 lane.
        let combined = uint8x16x2_t(digits_main, digits_tail);

        const fn idx(off: u8) -> [u8; 16] {
            [
                off,
                0xFF,
                0xFF,
                0xFF,
                off + 5,
                0xFF,
                0xFF,
                0xFF,
                off + 10,
                0xFF,
                0xFF,
                0xFF,
                off + 15,
                0xFF,
                0xFF,
                0xFF,
            ]
        }
        static IDX_D0: [u8; 16] = idx(0);
        static IDX_D1: [u8; 16] = idx(1);
        static IDX_D2: [u8; 16] = idx(2);
        static IDX_D3: [u8; 16] = idx(3);
        static IDX_D4: [u8; 16] = idx(4);

        let v_d0 = unsafe {
            vreinterpretq_u32_u8(vqtbl2q_u8(combined, vld1q_u8(IDX_D0.as_ptr())))
        };
        let v_d1 = unsafe {
            vreinterpretq_u32_u8(vqtbl2q_u8(combined, vld1q_u8(IDX_D1.as_ptr())))
        };
        let v_d2 = unsafe {
            vreinterpretq_u32_u8(vqtbl2q_u8(combined, vld1q_u8(IDX_D2.as_ptr())))
        };
        let v_d3 = unsafe {
            vreinterpretq_u32_u8(vqtbl2q_u8(combined, vld1q_u8(IDX_D3.as_ptr())))
        };
        let v_d4 = unsafe {
            vreinterpretq_u32_u8(vqtbl2q_u8(combined, vld1q_u8(IDX_D4.as_ptr())))
        };

        // Horner: n = ((((d0 * 85 + d1) * 85 + d2) * 85 + d3) * 85 + d4).
        // Each `vmlaq_n_u32(acc, v, k)` is `acc + v * k` lane-wise. The
        // intermediate after step 4 is bounded by 84*(85^3+85^2+85+1) =
        // 52_251_024, which fits in u32. The last `* 85 + d4` may overflow
        // u32 for adversarial input — detected below.
        let n = unsafe { vmlaq_n_u32(v_d1, v_d0, 85) };
        let n = unsafe { vmlaq_n_u32(v_d2, n, 85) };
        let n = unsafe { vmlaq_n_u32(v_d3, n, 85) };
        let n = unsafe { vmlaq_n_u32(v_d4, n, 85) };

        // Overflow detection: the un-wrapped value n_real = d0 * 85^4 + rest
        // satisfies n_real >= d0 * 85^4. If n (computed with wrapping mul)
        // is less than d0 * 85^4, exactly one wrap occurred — the only
        // case for our bounds since intermediate < 2^26 and rest < 85*2^26.
        // d0 * 85^4 itself never overflows because d0 <= 84 and 84*85^4 =
        // 4_384_852_500 < 2^32.
        let d0_85_4 = unsafe { vmulq_n_u32(v_d0, 52_200_625) };
        let overflow_mask = unsafe { vcltq_u32(n, d0_85_4) };
        // vmaxvq_u32 returns the max u32 across the 4 lanes; non-zero means
        // at least one lane had its compare set (0xFFFF_FFFF).
        if unsafe { vmaxvq_u32(overflow_mask) } != 0 {
            return Err(());
        }

        // Store as big-endian: byte-swap each u32 lane, then store 16 bytes.
        let bytes_le = unsafe { vreinterpretq_u8_u32(n) };
        let bytes_be = unsafe { vrev32q_u8(bytes_le) };
        unsafe { vst1q_u8(out.as_mut_ptr(), bytes_be) };

        Ok(())
    }

    /// Maps 16 ASCII chars to base85 digit values lane-wise. Chars in the
    /// alphabet map to 0..84; in-range chars not in the alphabet map to
    /// 0xFF; out-of-range chars (< 33 or > 126) yield unspecified output
    /// and must be filtered separately by the caller.
    #[inline]
    unsafe fn char85_to_byte_neon(encoded: uint8x16_t) -> uint8x16_t {
        // Two tables covering ASCII [33, 96] and [97, 128]. Gaps in the
        // alphabet are filled with 0xFF.
        static CHAR_TO_85: [u8; 64] = [
            62, 0xFF, 63, 64, 65, 66, 0xFF, 67, 68, 69, 70, 0xFF, 71, 0xFF, 0xFF, 0, 1, 2, 3, 4,
            5, 6, 7, 8, 9, 0xFF, 72, 73, 74, 75, 76, 77, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 0xFF, 0xFF, 0xFF,
            78, 79, 80,
        ];
        static CHAR_TO_85_X2: [u8; 32] = [
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
            57, 58, 59, 60, 61, 81, 82, 83, 84, 0xFF, 0xFF,
        ];

        let t1 = unsafe { vld1q_u8_x4(CHAR_TO_85.as_ptr()) };
        let t2 = unsafe { vld1q_u8_x2(CHAR_TO_85_X2.as_ptr()) };

        let mask_lt = unsafe { vcltq_u8(encoded, vdupq_n_u8(97)) };
        let index_lo = unsafe { vsubq_u8(encoded, vdupq_n_u8(33)) };
        let res_lo = unsafe { vqtbl4q_u8(t1, index_lo) };
        let index_hi = unsafe { vsubq_u8(encoded, vdupq_n_u8(97)) };
        let res_hi = unsafe { vqtbl2q_u8(t2, index_hi) };
        unsafe { vbslq_u8(mask_lt, res_lo, res_hi) }
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
