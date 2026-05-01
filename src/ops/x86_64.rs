//! x86_64 SSSE3 / SSE4.1 helpers — the structural mirror of `aarch64.rs`.
//!
//! The magic constants are the same (libdivide form, computed from the
//! divisor); only the SIMD intrinsics differ. Mapping cheat-sheet:
//!
//! | NEON                   | x86 (SSE2 / SSSE3 / SSE4.1) | what it does                              |
//! |------------------------|------------------------------|-------------------------------------------|
//! | `vmull_u32`            | `_mm_mul_epu32`              | u32×u32 → u64 widening (2 lanes per call) |
//! | `vshrq_n_u32`          | `_mm_srli_epi32`             | logical shift right per u32 lane          |
//! | `vmulq_n_u32`          | `_mm_mullo_epi32` (SSE4.1)   | u32×u32 → u32 (low 32 bits)               |
//! | `vsubq_u32`            | `_mm_sub_epi32`              | u32 lane-wise subtract                    |
//! | `vqtbl1q_u8`           | `_mm_shuffle_epi8` (SSSE3)   | byte-shuffle / lookup                     |
//! | `vbslq_u8`             | `_mm_blendv_epi8` (SSE4.1)   | byte-wise blend with mask                 |
//! | `vrev32q_u8`           | `_mm_shuffle_epi8` + idx     | reverse bytes in each u32 lane            |
//!
//! All SIMD entry points carry `#[target_feature(enable = "sse4.1,ssse3")]`
//! so the file compiles on any `x86_64-*-*` target regardless of the
//! default target-feature set (`x86_64-unknown-linux-gnu` defaults to
//! just SSE2). Callers must perform a runtime feature detection via
//! `std::is_x86_feature_detected!` before invoking.

use std::arch::x86_64::{
    __m128i, _mm_blend_epi16, _mm_mul_epu32, _mm_set1_epi32, _mm_shuffle_epi32, _mm_srli_epi32,
    _mm_srli_epi64,
};

/// Lane-wise `input / 85`. Magic constants verified for the full u32 range.
///
/// # Safety
/// Requires SSE4.1 + SSSE3 to be available on the host CPU.
#[inline]
#[target_feature(enable = "sse4.1,ssse3")]
pub(crate) unsafe fn div_85(input: __m128i) -> __m128i {
    // m = ceil(2^38 / 85) = 3_233_857_729; m * 85 - 2^38 = 21 < 2^6.
    // SAFETY: caller upholds SSE4.1+SSSE3 (function-level target_feature).
    unsafe { div_magic::<3_233_857_729, 6>(input) }
}

/// Lane-wise `input / 7225` (i.e. `input / 85²`). Valid for all u32 inputs.
///
/// # Safety
/// Requires SSE4.1 + SSSE3 to be available on the host CPU.
#[inline]
#[target_feature(enable = "sse4.1,ssse3")]
pub(crate) unsafe fn div_85_sq(input: __m128i) -> __m128i {
    // m = ceil(2^44 / 7225) = 2_434_904_643; m * 7225 - 2^44 = 1259 < 2^12.
    // SAFETY: caller upholds SSE4.1+SSSE3.
    unsafe { div_magic::<2_434_904_643, 12>(input) }
}

/// Lane-wise `input / 614125` (i.e. `input / 85³`). Valid for all u32 inputs.
///
/// # Safety
/// Requires SSE4.1 + SSSE3 to be available on the host CPU.
#[inline]
#[target_feature(enable = "sse4.1,ssse3")]
pub(crate) unsafe fn div_85_cube(input: __m128i) -> __m128i {
    // m = ceil(2^51 / 614125) = 3_666_679_933; bound check passes for u32.
    // SAFETY: caller upholds SSE4.1+SSSE3.
    unsafe { div_magic::<3_666_679_933, 19>(input) }
}

/// Generic libdivide-style div: `q = (n * MAGIC) >> (32 + SHIFT)`.
///
/// `_mm_mul_epu32` multiplies the **low u32 of each u64 lane**, so it
/// only touches input lanes 0 and 2 per call. We do two multiplies —
/// one as-is (for even lanes 0, 2), one after `_mm_srli_epi64::<32>`
/// (which moves odd lanes 1, 3 into the low u32 of each u64 lane).
///
/// After the two multiplies we have:
///   even = [(in[0]·m)lo, (in[0]·m)hi, (in[2]·m)lo, (in[2]·m)hi]
///   odd  = [(in[1]·m)lo, (in[1]·m)hi, (in[3]·m)lo, (in[3]·m)hi]
///
/// We want u32x4 = [(in[k]·m)>>32 for k=0..3] = the four `hi` lanes.
/// One `_mm_shuffle_epi32::<0xF5>` per side broadcasts the `hi` u32
/// across each u64 half (so even_hi = [hi0, hi0, hi2, hi2] etc.), and
/// `_mm_blend_epi16::<0xCC>` interleaves them into [hi0, hi1, hi2, hi3].
/// Finally `_mm_srli_epi32::<SHIFT>` applies the libdivide shift.
///
/// # Safety
/// Requires SSE4.1 + SSSE3 (the function-level `target_feature`
/// satisfies this for any caller that has done the runtime check).
#[inline]
#[target_feature(enable = "sse4.1,ssse3")]
unsafe fn div_magic<const MAGIC: u32, const SHIFT: i32>(input: __m128i) -> __m128i {
    let magic = _mm_set1_epi32(MAGIC as i32);

    // Even-lane multiply (touches input lanes 0 and 2).
    let even = _mm_mul_epu32(input, magic);
    // Shift right by 32 in u64 view to bring lanes 1 and 3 into
    // the low u32 of each u64 lane, then multiply.
    let input_odd = _mm_srli_epi64::<32>(input);
    let odd = _mm_mul_epu32(input_odd, magic);

    // 0b11_11_01_01 picks lane 1 / lane 1 / lane 3 / lane 3 — i.e.
    // broadcasts the high u32 of each u64 product across the pair.
    let even_hi = _mm_shuffle_epi32::<0b11_11_01_01>(even);
    let odd_hi = _mm_shuffle_epi32::<0b11_11_01_01>(odd);

    // Blend at u16 granularity: 0xCC = 0b1100_1100 picks the upper
    // u16 pair (i.e. odd_hi) for lanes 1 and 3 of the u32 view, and
    // the lower (even_hi) for lanes 0 and 2.
    let combined = _mm_blend_epi16::<0xCC>(even_hi, odd_hi);

    _mm_srli_epi32::<SHIFT>(combined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::{Arbitrary, quickcheck};
    use std::{
        arch::x86_64::{_mm_loadu_si128, _mm_storeu_si128},
        array,
    };

    #[derive(Debug, Clone, Copy)]
    struct InputBlock([u32; 4]);

    impl Arbitrary for InputBlock {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            InputBlock(array::from_fn(|_| u32::arbitrary(g)))
        }
    }

    fn lanes(v: __m128i) -> [u32; 4] {
        let mut out = [0u32; 4];
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast::<__m128i>(), v) };
        out
    }

    fn load(input: &[u32; 4]) -> __m128i {
        unsafe { _mm_loadu_si128(input.as_ptr().cast::<__m128i>()) }
    }

    quickcheck! {
        fn div_85_matches_scalar(block: InputBlock) -> bool {
            let InputBlock(input) = block;
            let q_out = lanes(unsafe { div_85(load(&input)) });
            (0..4).all(|i| q_out[i] == input[i] / 85)
        }

        fn div_85_sq_matches_scalar(block: InputBlock) -> bool {
            let InputBlock(input) = block;
            let q_out = lanes(unsafe { div_85_sq(load(&input)) });
            (0..4).all(|i| q_out[i] == input[i] / (85 * 85))
        }

        fn div_85_cube_matches_scalar(block: InputBlock) -> bool {
            let InputBlock(input) = block;
            let q_out = lanes(unsafe { div_85_cube(load(&input)) });
            (0..4).all(|i| q_out[i] == input[i] / (85 * 85 * 85))
        }

        fn div_85_to_the_4_via_composition(block: InputBlock) -> bool {
            let InputBlock(input) = block;
            let q2 = unsafe { div_85_sq(load(&input)) };
            let q4 = unsafe { div_85_sq(q2) };
            let q4_out = lanes(q4);
            (0..4).all(|i| q4_out[i] == input[i] / (85u32.pow(4)))
        }
    }
}
