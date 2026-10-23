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
    // `x85` is a digit in [0, 84] by construction in every call site
    // (each `n % 85` or final `n < 85` quotient). The `unwrap_or(b'0')`
    // is defensive — never observed at runtime.
    ALPHABET.get(x85 as usize).copied().unwrap_or(b'0')
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
/// only the first `input.len() + 1` characters are emitted. Mismatched
/// lengths are silently ignored (no panic) to keep this function on the
/// no-panic production path.
#[inline]
pub(crate) fn encode_tail(input: &[u8], out: &mut [u8]) {
    debug_assert!((1..=3).contains(&input.len()));
    debug_assert_eq!(out.len(), input.len() + 1);

    let mut padded = [0u8; 4];
    let Some(padded_head) = padded.get_mut(..input.len()) else {
        return;
    };
    padded_head.copy_from_slice(input);
    let mut full = [0u8; 5];
    encode_block(&padded, &mut full);
    let Some(emitted) = full.get(..input.len().saturating_add(1)) else {
        return;
    };
    if out.len() == emitted.len() {
        out.copy_from_slice(emitted);
    }
}

/// Decodes a 2..=4 character tail. Output length is `input.len() - 1`.
///
/// The tail is padded with `~` (digit value 84, the maximum) to form a full
/// 5-char block, decoded, and only the first `input.len() - 1` bytes are
/// kept. The padding choice ensures the kept bytes are independent of the
/// padding (see crate-level docs).
#[inline]
pub(crate) fn decode_tail(
    input: &[u8],
    out: &mut [u8],
    base_position: usize,
) -> Result<(), DecodeError> {
    debug_assert!((2..=4).contains(&input.len()));
    debug_assert_eq!(out.len(), input.len() - 1);

    let mut padded = [b'~'; 5];
    let Some(padded_head) = padded.get_mut(..input.len()) else {
        return Ok(());
    };
    padded_head.copy_from_slice(input);
    let mut full = [0u8; 4];
    decode_block(&padded, &mut full, base_position)?;
    let kept = input.len().saturating_sub(1);
    let Some(emitted) = full.get(..kept) else {
        return Ok(());
    };
    if out.len() == emitted.len() {
        out.copy_from_slice(emitted);
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────
// SIMD shared tables (used by both NEON and AVX2 paths)
// ─────────────────────────────────────────────────────────────────────────

// Per-digit splice indexes for the 4-block encode shuffle. Output position
// of digit k of block i is `5*i + k`; positions ≥ 16 spill into out_2 (with
// offset −16). Each v_dk is a u32x4 with the digit in byte 0/4/8/12 of its
// lane, hence the source offsets `4*i`. Both NEON (vqtbx1q_u8) and AVX2
// (PSHUFB, broadcast to both 128-bit halves) consume the identical patterns.

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[rustfmt::skip]
pub(super) static IDX_OUT_1: [[u8; 16]; 5] = [
    // k=0: out_1[0,5,10,15] ← src bytes 0,4,8,12
    [0,    0xFF, 0xFF, 0xFF, 0xFF, 4,    0xFF, 0xFF, 0xFF, 0xFF, 8,    0xFF, 0xFF, 0xFF, 0xFF, 12],
    // k=1: out_1[1,6,11] ← 0,4,8 (block 3 → out_2)
    [0xFF, 0,    0xFF, 0xFF, 0xFF, 0xFF, 4,    0xFF, 0xFF, 0xFF, 0xFF, 8,    0xFF, 0xFF, 0xFF, 0xFF],
    // k=2: out_1[2,7,12] ← 0,4,8
    [0xFF, 0xFF, 0,    0xFF, 0xFF, 0xFF, 0xFF, 4,    0xFF, 0xFF, 0xFF, 0xFF, 8,    0xFF, 0xFF, 0xFF],
    // k=3: out_1[3,8,13] ← 0,4,8
    [0xFF, 0xFF, 0xFF, 0,    0xFF, 0xFF, 0xFF, 0xFF, 4,    0xFF, 0xFF, 0xFF, 0xFF, 8,    0xFF, 0xFF],
    // k=4: out_1[4,9,14] ← 0,4,8
    [0xFF, 0xFF, 0xFF, 0xFF, 0,    0xFF, 0xFF, 0xFF, 0xFF, 4,    0xFF, 0xFF, 0xFF, 0xFF, 8,    0xFF],
];

// Indexes for digits 1..4 of block 3 (src byte 12) spilling into out_2.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[rustfmt::skip]
pub(super) static IDX_OUT_2: [[u8; 16]; 4] = [
    [12,   0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
    [0xFF, 12,   0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
    [0xFF, 0xFF, 12,   0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
    [0xFF, 0xFF, 0xFF, 12,   0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
];

// Mask used by the SIMD decoders: 0xFF for the 4 real-input lanes of the
// padded tail register, 0 for the 12 padding lanes. Lets us suppress
// validation noise from the padding bytes.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(super) static TAIL_VALID_MASK: [u8; 16] =
    [0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

// ─────────────────────────────────────────────────────────────────────────
// NEON encoder (aarch64)
// ─────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
pub(crate) use neon::{NeonEncoder, try_decode_block_x4};

#[cfg(target_arch = "aarch64")]
mod neon {
    // Indexing/slicing inside this module is over fixed-size arrays
    // (`[u8; 16]`, `[u8; 20]`) with compile-time-constant bounds, so
    // panic-on-OOB cannot occur at runtime.
    #![allow(clippy::indexing_slicing)]
    //
    // SAFETY (register-only intrinsics): the NEON intrinsics are `unsafe`
    // only because they are `#[target_feature(enable = "neon")]`, and NEON
    // is part of the aarch64 baseline that every aarch64 target enables.
    // An `unsafe { … }` around an intrinsic that touches no memory (lane
    // arithmetic, compares, table lookups on registers) relies on that alone
    // and carries no comment of its own. Every load or store through a
    // pointer has its own `// SAFETY:` comment.

    use crate::ops::{div_85, div_85_cube, div_85_sq};
    use std::arch::aarch64::{
        uint8x16_t, uint8x16x2_t, uint32x4_t, vandq_u8, vbslq_u8, vceqq_u8, vcgtq_u8, vcltq_u8,
        vcltq_u32, vdupq_n_u8, vld1q_u8, vld1q_u8_x2, vld1q_u8_x4, vmaxvq_u8, vmaxvq_u32,
        vmlaq_n_u32, vmlsq_n_u32, vmulq_n_u32, vorrq_u8, vqtbl2q_u8, vqtbl4q_u8, vqtbx1q_u8,
        vreinterpretq_u8_u32, vreinterpretq_u32_u8, vrev32q_u8, vst1q_u8, vsubq_u8,
    };

    use super::{IDX_OUT_1, IDX_OUT_2, TAIL_VALID_MASK};

    /// Encodes 16 input bytes into 20 output characters using NEON.
    ///
    /// Computes all four quotients (`n/85, n/85², n/85³, n/85⁴`) in
    /// parallel from `n` rather than chaining four `÷ 85` steps in
    /// series. Three are direct (independent), and `q4 = q2 / 85²`
    /// composes the `/85²` magic twice because the direct `/85⁴` magic
    /// doesn't satisfy libdivide's precision bound for u32 inputs.
    pub(crate) struct NeonEncoder;

    impl NeonEncoder {
        #[inline]
        pub(crate) fn new() -> Self {
            Self
        }

        /// Encode exactly 16 input bytes into exactly 20 output bytes.
        ///
        /// The fixed-size array references make all NEON load and store
        /// bounds type-enforced — every intrinsic call below operates on
        /// pointers known to be valid for at least 16 (or 20) bytes.
        #[inline]
        pub(crate) fn encode_block_x4(&self, in_bytes: &[u8; 16], out_bytes: &mut [u8; 20]) {
            let n = load_u32x4_be(in_bytes);

            // Three independent divides from `n`, plus one composed.
            // The OoO scheduler can issue q1/q2/q3 in parallel, then q4
            // chains off q2 (one extra ~5-op stage).
            let q1 = div_85(n);
            let q2 = div_85_sq(n);
            let q3 = div_85_cube(n);
            let q4 = div_85_sq(q2);

            // Five remainders, all independent given the q's. Each is
            // one fused `mls.4s` (multiply-subtract) instruction:
            //   d0 = q4
            //   d1 = q3 - 85·q4
            //   d2 = q2 - 85·q3
            //   d3 = q1 - 85·q2
            //   d4 = n  - 85·q1
            let d0 = q4;
            let d1 = unsafe { vmlsq_n_u32(q3, q4, 85) };
            let d2 = unsafe { vmlsq_n_u32(q2, q3, 85) };
            let d3 = unsafe { vmlsq_n_u32(q1, q2, 85) };
            let d4 = unsafe { vmlsq_n_u32(n, q1, 85) };

            // Reinterpret as u8 vectors so TBX can read individual bytes.
            let d0_u8 = unsafe { vreinterpretq_u8_u32(d0) };
            let d1_u8 = unsafe { vreinterpretq_u8_u32(d1) };
            let d2_u8 = unsafe { vreinterpretq_u8_u32(d2) };
            let d3_u8 = unsafe { vreinterpretq_u8_u32(d3) };
            let d4_u8 = unsafe { vreinterpretq_u8_u32(d4) };

            // Load splice indexes from .rodata. Each `vld1q_u8` is one
            // load + the indexes get held in vector regs across the
            // five-deep TBX chain.
            // SAFETY: each `IDX_OUT_*` row is a `[u8; 16]`, one vector.
            let idx0 = unsafe { vld1q_u8(IDX_OUT_1[0].as_ptr()) };
            let idx1 = unsafe { vld1q_u8(IDX_OUT_1[1].as_ptr()) };
            let idx2 = unsafe { vld1q_u8(IDX_OUT_1[2].as_ptr()) };
            let idx3 = unsafe { vld1q_u8(IDX_OUT_1[3].as_ptr()) };
            let idx4 = unsafe { vld1q_u8(IDX_OUT_1[4].as_ptr()) };

            let mut out_1 = unsafe { vdupq_n_u8(0) };
            out_1 = unsafe { vqtbx1q_u8(out_1, d0_u8, idx0) };
            out_1 = unsafe { vqtbx1q_u8(out_1, d1_u8, idx1) };
            out_1 = unsafe { vqtbx1q_u8(out_1, d2_u8, idx2) };
            out_1 = unsafe { vqtbx1q_u8(out_1, d3_u8, idx3) };
            out_1 = unsafe { vqtbx1q_u8(out_1, d4_u8, idx4) };

            // SAFETY: as above, `[u8; 16]` rows.
            let idx2_0 = unsafe { vld1q_u8(IDX_OUT_2[0].as_ptr()) };
            let idx2_1 = unsafe { vld1q_u8(IDX_OUT_2[1].as_ptr()) };
            let idx2_2 = unsafe { vld1q_u8(IDX_OUT_2[2].as_ptr()) };
            let idx2_3 = unsafe { vld1q_u8(IDX_OUT_2[3].as_ptr()) };

            let mut out_2 = unsafe { vdupq_n_u8(0) };
            out_2 = unsafe { vqtbx1q_u8(out_2, d1_u8, idx2_0) };
            out_2 = unsafe { vqtbx1q_u8(out_2, d2_u8, idx2_1) };
            out_2 = unsafe { vqtbx1q_u8(out_2, d3_u8, idx2_2) };
            out_2 = unsafe { vqtbx1q_u8(out_2, d4_u8, idx2_3) };

            out_1 = byte_to_char85_neon(out_1);
            out_2 = byte_to_char85_neon(out_2);

            // Store: 16 bytes from `out_1` to out_bytes[0..16] (aligned-
            // friendly via vst1q_u8), then 4 bytes from `out_2` lane-0
            // copied byte-wise to avoid any *u32 alignment hazard.
            // SAFETY: `out_bytes` is `&mut [u8; 20]`, so bytes 0..16 take the
            // first store and bytes 16..20 the 4-byte copy; `tail_buf` is a
            // local `[u8; 16]`, one vector, and doesn't overlap `out_bytes`.
            unsafe {
                vst1q_u8(out_bytes.as_mut_ptr(), out_1);
                let mut tail_buf = [0u8; 16];
                vst1q_u8(tail_buf.as_mut_ptr(), out_2);
                std::ptr::copy_nonoverlapping(tail_buf.as_ptr(), out_bytes.as_mut_ptr().add(16), 4);
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
    #[inline]
    pub(crate) fn try_decode_block_x4(input: &[u8; 20], out: &mut [u8; 16]) -> Result<(), ()> {
        // Load chars 0..16 directly; load chars 16..19 into the low 4 lanes
        // of a zero-padded vector so we can use a unified TBL pipeline.
        // SAFETY: `input` is `&[u8; 20]`, so its first 16 bytes are readable.
        let chars_main = unsafe { vld1q_u8(input.as_ptr()) };
        let mut tail_buf = [0u8; 16];
        tail_buf[..4].copy_from_slice(&input[16..20]);
        // SAFETY: `tail_buf` is a local `[u8; 16]`, one vector.
        let chars_tail = unsafe { vld1q_u8(tail_buf.as_ptr()) };

        // SAFETY: `TAIL_VALID_MASK` is a `[u8; 16]`, one vector.
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
        let digits_main = char85_to_byte_neon(chars_main);
        let digits_tail = char85_to_byte_neon(chars_tail);

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

        // SAFETY: each `IDX_D*` is a `[u8; 16]`, one vector.
        let v_d0 = unsafe { vreinterpretq_u32_u8(vqtbl2q_u8(combined, vld1q_u8(IDX_D0.as_ptr()))) };
        let v_d1 = unsafe { vreinterpretq_u32_u8(vqtbl2q_u8(combined, vld1q_u8(IDX_D1.as_ptr()))) };
        let v_d2 = unsafe { vreinterpretq_u32_u8(vqtbl2q_u8(combined, vld1q_u8(IDX_D2.as_ptr()))) };
        let v_d3 = unsafe { vreinterpretq_u32_u8(vqtbl2q_u8(combined, vld1q_u8(IDX_D3.as_ptr()))) };
        let v_d4 = unsafe { vreinterpretq_u32_u8(vqtbl2q_u8(combined, vld1q_u8(IDX_D4.as_ptr()))) };

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
        // SAFETY: `out` is `&mut [u8; 16]`, exactly one vector's worth.
        unsafe { vst1q_u8(out.as_mut_ptr(), bytes_be) };

        Ok(())
    }

    /// Maps 16 ASCII chars to base85 digit values lane-wise. Chars in the
    /// alphabet map to 0..84; in-range chars not in the alphabet map to
    /// 0xFF; out-of-range chars (< 33 or > 126) yield unspecified output
    /// and must be filtered separately by the caller.
    #[inline]
    fn char85_to_byte_neon(encoded: uint8x16_t) -> uint8x16_t {
        // Two tables covering ASCII [33, 96] and [97, 128]. Gaps in the
        // alphabet are filled with 0xFF.
        static CHAR_TO_85: [u8; 64] = [
            62, 0xFF, 63, 64, 65, 66, 0xFF, 67, 68, 69, 70, 0xFF, 71, 0xFF, 0xFF, 0, 1, 2, 3, 4, 5,
            6, 7, 8, 9, 0xFF, 72, 73, 74, 75, 76, 77, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 0xFF, 0xFF, 0xFF, 78, 79,
            80,
        ];
        static CHAR_TO_85_X2: [u8; 32] = [
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 81, 82, 83, 84, 0xFF, 0xFF,
        ];

        // SAFETY: the tables are `[u8; 64]` and `[u8; 32]`, exactly the four
        // and two vectors each load reads.
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
    fn byte_to_char85_neon(x85: uint8x16_t) -> uint8x16_t {
        // 64 chars (fits in vqtbl4) + 21 chars (padded to 32, fits in vqtbl2).
        // The padding bytes ('#') in B85_X2 are unreachable when x85 < 85.
        static B85_X4: [u8; 64] =
            *b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#";
        static B85_X2: [u8; 32] = *b"$%&()*+-;<=>?@^_`{|}~###########";

        // SAFETY: the tables are `[u8; 64]` and `[u8; 32]`, exactly the four
        // and two vectors each load reads.
        let t1 = unsafe { vld1q_u8_x4(B85_X4.as_ptr()) };
        let t2 = unsafe { vld1q_u8_x2(B85_X2.as_ptr()) };

        let mask_lt64 = unsafe { vcltq_u8(x85, vdupq_n_u8(64)) };
        let res_lo = unsafe { vqtbl4q_u8(t1, x85) };
        let index_hi = unsafe { vsubq_u8(x85, vdupq_n_u8(64)) };
        let res_hi = unsafe { vqtbl2q_u8(t2, index_hi) };
        unsafe { vbslq_u8(mask_lt64, res_lo, res_hi) }
    }

    /// Loads 16 bytes and reinterprets as 4 big-endian u32s. Uses the
    /// byte-aligned `vld1q_u8` rather than `vld1q_u32` (which would
    /// require 4-byte alignment of the source pointer that a `&[u8]`
    /// can't guarantee). Compiles to `ldr q + rev32.16b`.
    #[inline]
    fn load_u32x4_be(bytes: &[u8; 16]) -> uint32x4_t {
        // SAFETY: `bytes` is `&[u8; 16]`, one vector.
        unsafe {
            let raw = vld1q_u8(bytes.as_ptr());
            let swapped = vrev32q_u8(raw);
            vreinterpretq_u32_u8(swapped)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// AVX2 encoder / decoder (x86_64)
// ─────────────────────────────────────────────────────────────────────────
//
// Processes 8 blocks per call (32 input bytes, 40 output chars) by
// running the same SSE4.1 algorithm across both 128-bit lanes of an
// `__m256i`. AVX2's lane-restricted byte/word ops (PSHUFB, PSHUFD,
// BLEND) work fine here because the algorithm is fully per-lane —
// each lane handles 4 independent blocks.
//
// On hosts without AVX2 the crate's runtime feature gate routes to
// the scalar fallback (see `crate::encode_scalar` / `crate::decode_scalar`).

#[cfg(target_arch = "x86_64")]
pub(crate) use avx2::{Avx2Encoder, try_decode_block_x8};

#[cfg(target_arch = "x86_64")]
mod avx2 {
    //! AVX2 encode/decode — 8 blocks per call.
    //!
    //! Both 128-bit lanes of each `__m256i` independently run the
    //! algorithm we developed for SSE4.1 (4 blocks per lane, 8 total).
    //! AVX2 byte/word ops are lane-restricted (PSHUFB / BLEND /
    //! SHUFFLE_EPI32 / etc. operate per 128-bit half), but every
    //! step of the algorithm is per-lane anyway, so that's fine.
    //!
    //! Every entry point is `unsafe fn` with
    //! `#[target_feature(enable = "avx2")]`. The caller
    //! (`crate::encode_into` / `crate::decode_into`) does a one-shot
    //! `is_x86_feature_detected!("avx2")` check; on hosts lacking
    //! AVX2 the crate routes to the scalar fallback.
    //!
    //! The module-level `#![allow(unsafe_op_in_unsafe_fn)]` keeps the
    //! body of each `unsafe fn` an implicit unsafe scope on MSRV 1.85
    //! (without it, each intrinsic call needs its own `unsafe { … }`).

    #![allow(unsafe_op_in_unsafe_fn)]
    #![allow(clippy::indexing_slicing)]

    use crate::ops::{div_85, div_85_cube, div_85_sq};
    use std::arch::x86_64::{
        __m128i, __m256i, _mm_loadu_si128, _mm_storeu_si128, _mm256_add_epi32, _mm256_and_si256,
        _mm256_broadcastsi128_si256, _mm256_cmpeq_epi8, _mm256_cmpgt_epi8, _mm256_cmpgt_epi32,
        _mm256_extract_epi32, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_movemask_epi8,
        _mm256_mullo_epi32, _mm256_or_si256, _mm256_set_m128i, _mm256_set1_epi8, _mm256_set1_epi32,
        _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_storeu_si256,
        _mm256_sub_epi32, _mm256_xor_si256,
    };

    use super::{IDX_OUT_1, IDX_OUT_2, TAIL_VALID_MASK};

    static REV32_IDX: [u8; 16] = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12];

    #[rustfmt::skip]
    static ALPHABET_PADDED: &[u8; 96] =
        b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~\0\0\0\0\0\0\0\0\0\0\0";

    #[rustfmt::skip]
    static CHAR_TO_85_PADDED: [u8; 96] = [
        62, 0xFF, 63, 64, 65, 66, 0xFF, 67, 68, 69, 70, 0xFF, 71, 0xFF, 0xFF, 0,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 0xFF, 72, 73, 74, 75, 76, 77,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 0xFF, 0xFF, 0xFF, 78, 79, 80,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 81, 82, 83, 84, 0xFF, 0xFF,
    ];

    #[rustfmt::skip]
    static IDX_DEC: [([u8; 16], [u8; 16]); 5] = [
        (
            [0, 0xFF, 0xFF, 0xFF, 5, 0xFF, 0xFF, 0xFF, 10, 0xFF, 0xFF, 0xFF, 15, 0xFF, 0xFF, 0xFF],
            [0xFF; 16],
        ),
        (
            [1, 0xFF, 0xFF, 0xFF, 6, 0xFF, 0xFF, 0xFF, 11, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
            [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0xFF, 0xFF, 0xFF],
        ),
        (
            [2, 0xFF, 0xFF, 0xFF, 7, 0xFF, 0xFF, 0xFF, 12, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
            [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 1, 0xFF, 0xFF, 0xFF],
        ),
        (
            [3, 0xFF, 0xFF, 0xFF, 8, 0xFF, 0xFF, 0xFF, 13, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
            [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 2, 0xFF, 0xFF, 0xFF],
        ),
        (
            [4, 0xFF, 0xFF, 0xFF, 9, 0xFF, 0xFF, 0xFF, 14, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
            [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 3, 0xFF, 0xFF, 0xFF],
        ),
    ];

    /// Loads a 16-byte index from .rodata and broadcasts it to both
    /// 128-bit lanes of an `__m256i`.
    ///
    /// # Safety
    /// `ptr` must be valid for a 16-byte read. Caller has AVX2.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn broadcast16(ptr: *const u8) -> __m256i {
        _mm256_broadcastsi128_si256(_mm_loadu_si128(ptr.cast::<__m128i>()))
    }

    /// AVX2 8-block encoder. Mirrors the structure of NEON's
    /// `NeonEncoder` but processes 8 input blocks (32 bytes → 40 chars)
    /// per call.
    pub(crate) struct Avx2Encoder;

    impl Avx2Encoder {
        #[inline]
        pub(crate) fn new() -> Self {
            Self
        }

        /// Encode exactly 32 input bytes into exactly 40 output bytes.
        ///
        /// # Safety
        /// Requires AVX2 to be available on the host CPU.
        #[inline]
        #[target_feature(enable = "avx2")]
        pub(crate) unsafe fn encode_block_x8(&self, in_bytes: &[u8; 32], out_bytes: &mut [u8; 40]) {
            // Load 32 bytes (8 × 4-byte blocks) and reverse each 4-byte
            // group to get 8 BE u32s.
            let raw = _mm256_loadu_si256(in_bytes.as_ptr().cast::<__m256i>());
            let n = _mm256_shuffle_epi8(raw, broadcast16(REV32_IDX.as_ptr()));

            // Parallel-magic divides — 8 lanes wide.
            let q1 = div_85(n);
            let q2 = div_85_sq(n);
            let q3 = div_85_cube(n);
            let q4 = div_85_sq(q2);

            // Five digits per block via lane-wise multiply-subtract.
            let m85 = _mm256_set1_epi32(85);
            let d0 = q4;
            let d1 = _mm256_sub_epi32(q3, _mm256_mullo_epi32(q4, m85));
            let d2 = _mm256_sub_epi32(q2, _mm256_mullo_epi32(q3, m85));
            let d3 = _mm256_sub_epi32(q1, _mm256_mullo_epi32(q2, m85));
            let d4 = _mm256_sub_epi32(n, _mm256_mullo_epi32(q1, m85));

            // Splice 5 digits into out_1 (32 bytes; 16 per lane — each
            // lane holds the first 16 chars of its 4-block group).
            // PSHUFB is per-lane; the same index broadcast to both lanes
            // does the right thing.
            let idx0 = broadcast16(IDX_OUT_1[0].as_ptr());
            let idx1 = broadcast16(IDX_OUT_1[1].as_ptr());
            let idx2 = broadcast16(IDX_OUT_1[2].as_ptr());
            let idx3 = broadcast16(IDX_OUT_1[3].as_ptr());
            let idx4 = broadcast16(IDX_OUT_1[4].as_ptr());
            let p0 = _mm256_shuffle_epi8(d0, idx0);
            let p1 = _mm256_shuffle_epi8(d1, idx1);
            let p2 = _mm256_shuffle_epi8(d2, idx2);
            let p3 = _mm256_shuffle_epi8(d3, idx3);
            let p4 = _mm256_shuffle_epi8(d4, idx4);
            let out_1 = _mm256_or_si256(
                _mm256_or_si256(p0, p1),
                _mm256_or_si256(p2, _mm256_or_si256(p3, p4)),
            );

            // Same for out_2 (8 bytes total; 4 per lane — the trailing
            // 4 chars of each 4-block group).
            let i20 = broadcast16(IDX_OUT_2[0].as_ptr());
            let i21 = broadcast16(IDX_OUT_2[1].as_ptr());
            let i22 = broadcast16(IDX_OUT_2[2].as_ptr());
            let i23 = broadcast16(IDX_OUT_2[3].as_ptr());
            let q21 = _mm256_shuffle_epi8(d1, i20);
            let q22 = _mm256_shuffle_epi8(d2, i21);
            let q23 = _mm256_shuffle_epi8(d3, i22);
            let q24 = _mm256_shuffle_epi8(d4, i23);
            let out_2 = _mm256_or_si256(_mm256_or_si256(q21, q22), _mm256_or_si256(q23, q24));

            // Digit value (0..84) → ASCII char.
            let out_1 = byte_to_char85_avx2(out_1);
            let out_2 = byte_to_char85_avx2(out_2);

            // Store. Output layout is non-contiguous between lanes:
            //   bytes  0..16  — lane 0's 16-char main
            //   bytes 16..20  — lane 0's 4-char tail
            //   bytes 20..36  — lane 1's 16-char main
            //   bytes 36..40  — lane 1's 4-char tail
            let lo_main = _mm256_extracti128_si256::<0>(out_1);
            let hi_main = _mm256_extracti128_si256::<1>(out_1);
            let lo_tail = _mm256_extract_epi32::<0>(out_2) as u32;
            let hi_tail = _mm256_extract_epi32::<4>(out_2) as u32;

            _mm_storeu_si128(out_bytes.as_mut_ptr().cast::<__m128i>(), lo_main);
            std::ptr::write_unaligned(out_bytes.as_mut_ptr().add(16).cast::<u32>(), lo_tail);
            _mm_storeu_si128(out_bytes.as_mut_ptr().add(20).cast::<__m128i>(), hi_main);
            std::ptr::write_unaligned(out_bytes.as_mut_ptr().add(36).cast::<u32>(), hi_tail);
        }
    }

    /// Maps 32 lane-wise digit values (0..84) to ASCII characters.
    ///
    /// Same chunk-index-selection trick as the SSE port (PSHUFB only
    /// zeros on bit-7-set indexes), just doubled.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn byte_to_char85_avx2(x85: __m256i) -> __m256i {
        let high_nib = _mm256_and_si256(_mm256_srli_epi16::<4>(x85), _mm256_set1_epi8(0x0F));
        let low_nib = _mm256_and_si256(x85, _mm256_set1_epi8(0x0F));

        let t0 = broadcast16(ALPHABET_PADDED.as_ptr());
        let t1 = broadcast16(ALPHABET_PADDED.as_ptr().add(16));
        let t2 = broadcast16(ALPHABET_PADDED.as_ptr().add(32));
        let t3 = broadcast16(ALPHABET_PADDED.as_ptr().add(48));
        let t4 = broadcast16(ALPHABET_PADDED.as_ptr().add(64));
        let t5 = broadcast16(ALPHABET_PADDED.as_ptr().add(80));

        let r0 = _mm256_shuffle_epi8(t0, low_nib);
        let r1 = _mm256_shuffle_epi8(t1, low_nib);
        let r2 = _mm256_shuffle_epi8(t2, low_nib);
        let r3 = _mm256_shuffle_epi8(t3, low_nib);
        let r4 = _mm256_shuffle_epi8(t4, low_nib);
        let r5 = _mm256_shuffle_epi8(t5, low_nib);

        let m0 = _mm256_cmpeq_epi8(high_nib, _mm256_setzero_si256());
        let m1 = _mm256_cmpeq_epi8(high_nib, _mm256_set1_epi8(1));
        let m2 = _mm256_cmpeq_epi8(high_nib, _mm256_set1_epi8(2));
        let m3 = _mm256_cmpeq_epi8(high_nib, _mm256_set1_epi8(3));
        let m4 = _mm256_cmpeq_epi8(high_nib, _mm256_set1_epi8(4));
        let m5 = _mm256_cmpeq_epi8(high_nib, _mm256_set1_epi8(5));

        _mm256_or_si256(
            _mm256_or_si256(
                _mm256_or_si256(_mm256_and_si256(r0, m0), _mm256_and_si256(r1, m1)),
                _mm256_or_si256(_mm256_and_si256(r2, m2), _mm256_and_si256(r3, m3)),
            ),
            _mm256_or_si256(_mm256_and_si256(r4, m4), _mm256_and_si256(r5, m5)),
        )
    }

    /// Maps 32 ASCII chars to base85 digit values (0..84) lane-wise.
    /// Chars not in the alphabet (or out of [33, 126]) come back as 0xFF.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn char85_to_byte_avx2(chars: __m256i) -> __m256i {
        let normalised = _mm256_sub_epi32(chars, _mm256_set1_epi8(33));
        let high_nib = _mm256_and_si256(_mm256_srli_epi16::<4>(normalised), _mm256_set1_epi8(0x0F));
        let low_nib = _mm256_and_si256(normalised, _mm256_set1_epi8(0x0F));

        let t0 = broadcast16(CHAR_TO_85_PADDED.as_ptr());
        let t1 = broadcast16(CHAR_TO_85_PADDED.as_ptr().add(16));
        let t2 = broadcast16(CHAR_TO_85_PADDED.as_ptr().add(32));
        let t3 = broadcast16(CHAR_TO_85_PADDED.as_ptr().add(48));
        let t4 = broadcast16(CHAR_TO_85_PADDED.as_ptr().add(64));
        let t5 = broadcast16(CHAR_TO_85_PADDED.as_ptr().add(80));

        let r0 = _mm256_shuffle_epi8(t0, low_nib);
        let r1 = _mm256_shuffle_epi8(t1, low_nib);
        let r2 = _mm256_shuffle_epi8(t2, low_nib);
        let r3 = _mm256_shuffle_epi8(t3, low_nib);
        let r4 = _mm256_shuffle_epi8(t4, low_nib);
        let r5 = _mm256_shuffle_epi8(t5, low_nib);

        let m0 = _mm256_cmpeq_epi8(high_nib, _mm256_setzero_si256());
        let m1 = _mm256_cmpeq_epi8(high_nib, _mm256_set1_epi8(1));
        let m2 = _mm256_cmpeq_epi8(high_nib, _mm256_set1_epi8(2));
        let m3 = _mm256_cmpeq_epi8(high_nib, _mm256_set1_epi8(3));
        let m4 = _mm256_cmpeq_epi8(high_nib, _mm256_set1_epi8(4));
        let m5 = _mm256_cmpeq_epi8(high_nib, _mm256_set1_epi8(5));

        _mm256_or_si256(
            _mm256_or_si256(
                _mm256_or_si256(_mm256_and_si256(r0, m0), _mm256_and_si256(r1, m1)),
                _mm256_or_si256(_mm256_and_si256(r2, m2), _mm256_and_si256(r3, m3)),
            ),
            _mm256_or_si256(_mm256_and_si256(r4, m4), _mm256_and_si256(r5, m5)),
        )
    }

    /// AVX2 8-block decoder. 40 chars → 32 bytes.
    ///
    /// Returns `Err(())` (so the caller falls back to the scalar path,
    /// which will surface a precise [`crate::DecodeError`]) when:
    /// - any input byte is outside the base85 alphabet, or
    /// - any of the eight 5-char blocks decodes to a value > `u32::MAX`.
    ///
    /// # Safety
    /// Requires AVX2 on the host CPU.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn try_decode_block_x8(
        input: &[u8; 40],
        out: &mut [u8; 32],
    ) -> Result<(), ()> {
        // Layout the 40 chars into the 256-bit register so each
        // 128-bit lane handles 4 blocks. Lane 0 ← chars 0..15
        // (first 16 chars of blocks 0..3); lane 1 ← chars 20..35
        // (first 16 chars of blocks 4..7). The 4-char tails (chars
        // 16..19 and 36..39) go into a separate "chars_tail" with
        // 12 zero-padding bytes per lane.
        let chars_main_lo = _mm_loadu_si128(input.as_ptr().cast::<__m128i>());
        let chars_main_hi = _mm_loadu_si128(input.as_ptr().add(20).cast::<__m128i>());
        let chars_main = _mm256_set_m128i(chars_main_hi, chars_main_lo);

        let mut tail_buf_lo = [0u8; 16];
        let mut tail_buf_hi = [0u8; 16];
        tail_buf_lo[..4].copy_from_slice(&input[16..20]);
        tail_buf_hi[..4].copy_from_slice(&input[36..40]);
        let chars_tail = _mm256_set_m128i(
            _mm_loadu_si128(tail_buf_hi.as_ptr().cast::<__m128i>()),
            _mm_loadu_si128(tail_buf_lo.as_ptr().cast::<__m128i>()),
        );
        let tail_valid_mask = broadcast16(TAIL_VALID_MASK.as_ptr());

        // Range validation: chars must be in [33, 126].
        let too_low_main = _mm256_cmpgt_epi8(_mm256_set1_epi8(33), chars_main);
        let too_high_main = _mm256_cmpgt_epi8(chars_main, _mm256_set1_epi8(126));
        let invalid_range_main = _mm256_or_si256(too_low_main, too_high_main);

        let too_low_tail = _mm256_cmpgt_epi8(_mm256_set1_epi8(33), chars_tail);
        let too_high_tail = _mm256_cmpgt_epi8(chars_tail, _mm256_set1_epi8(126));
        let invalid_range_tail = _mm256_and_si256(
            _mm256_or_si256(too_low_tail, too_high_tail),
            tail_valid_mask,
        );

        // ASCII → digit lookup (0xFF for in-range invalid chars).
        let digits_main = char85_to_byte_avx2(chars_main);
        let digits_tail = char85_to_byte_avx2(chars_tail);

        let invalid_dig_main = _mm256_cmpeq_epi8(digits_main, _mm256_set1_epi8(0xFF_u8 as i8));
        let invalid_dig_tail = _mm256_and_si256(
            _mm256_cmpeq_epi8(digits_tail, _mm256_set1_epi8(0xFF_u8 as i8)),
            tail_valid_mask,
        );

        let any_invalid = _mm256_or_si256(
            _mm256_or_si256(invalid_range_main, invalid_range_tail),
            _mm256_or_si256(invalid_dig_main, invalid_dig_tail),
        );
        // _mm256_movemask_epi8 returns a 32-bit mask (one bit per
        // byte of the 256-bit register).
        if _mm256_movemask_epi8(any_invalid) != 0 {
            return Err(());
        }

        // Permute digits into per-position vectors. Same indexes
        // as the SSE path, broadcast to both lanes.
        let v_dk = |idx_main: &[u8; 16], idx_tail: &[u8; 16]| -> __m256i {
            let im = broadcast16(idx_main.as_ptr());
            let it = broadcast16(idx_tail.as_ptr());
            _mm256_or_si256(
                _mm256_shuffle_epi8(digits_main, im),
                _mm256_shuffle_epi8(digits_tail, it),
            )
        };
        let v_d0 = v_dk(&IDX_DEC[0].0, &IDX_DEC[0].1);
        let v_d1 = v_dk(&IDX_DEC[1].0, &IDX_DEC[1].1);
        let v_d2 = v_dk(&IDX_DEC[2].0, &IDX_DEC[2].1);
        let v_d3 = v_dk(&IDX_DEC[3].0, &IDX_DEC[3].1);
        let v_d4 = v_dk(&IDX_DEC[4].0, &IDX_DEC[4].1);

        // Horner: n = ((((d0·85 + d1)·85 + d2)·85 + d3)·85 + d4)
        let m85 = _mm256_set1_epi32(85);
        let n = _mm256_add_epi32(v_d1, _mm256_mullo_epi32(v_d0, m85));
        let n = _mm256_add_epi32(v_d2, _mm256_mullo_epi32(n, m85));
        let n = _mm256_add_epi32(v_d3, _mm256_mullo_epi32(n, m85));
        let n = _mm256_add_epi32(v_d4, _mm256_mullo_epi32(n, m85));

        // Overflow detection (bias-and-signed-cmpgt).
        let d0_85_4 = _mm256_mullo_epi32(v_d0, _mm256_set1_epi32(52_200_625));
        let bias = _mm256_set1_epi32(0x80000000_u32 as i32);
        let n_b = _mm256_xor_si256(n, bias);
        let d0_b = _mm256_xor_si256(d0_85_4, bias);
        let overflow_mask = _mm256_cmpgt_epi32(d0_b, n_b);
        if _mm256_movemask_epi8(overflow_mask) != 0 {
            return Err(());
        }

        // Byte-swap each u32 lane to BE and store. Output is 32
        // contiguous bytes (16 from each lane), so a single
        // _mm256_storeu_si256 works.
        let bytes_be = _mm256_shuffle_epi8(n, broadcast16(REV32_IDX.as_ptr()));
        _mm256_storeu_si256(out.as_mut_ptr().cast::<__m256i>(), bytes_be);

        Ok(())
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
            // n ∈ {1, 2, 3}; step_by stride spreads `take(n)` samples across
            // the byte range. `n == 0` would divide-by-zero, but it's
            // excluded by the loop bounds.
            let input: Vec<u8> = (0u8..=255).step_by(256 / n + 1).take(n).collect();
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
