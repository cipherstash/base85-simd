//! x86_64 SSE2 / SSSE3 helpers — the structural mirror of `aarch64.rs`.
//!
//! The magic constants are the same (libdivide form, computed from the
//! divisor); only the SIMD intrinsics differ. Mapping cheat-sheet:
//!
//! | NEON                   | x86 (SSE2 / SSSE3)        | what it does                              |
//! |------------------------|---------------------------|-------------------------------------------|
//! | `vmull_u32`            | `_mm_mul_epu32`           | u32×u32 → u64 widening multiply (2 lanes) |
//! | `vshrq_n_u32`          | `_mm_srli_epi32`          | logical shift right per u32 lane          |
//! | `vmulq_n_u32`          | `_mm_mullo_epi32` (SSE4.1)| u32×u32 → u32 (low 32 bits)               |
//! | `vsubq_u32`            | `_mm_sub_epi32`           | u32 lane-wise subtract                    |
//! | `vqtbl1q_u8`           | `_mm_shuffle_epi8` (SSSE3)| byte-shuffle / lookup                     |
//! | `vbslq_u8`             | `_mm_blendv_epi8` (SSE4.1)| byte-wise blend with mask                 |
//! | `vrev32q_u8`           | `_mm_shuffle_epi8` + idx  | reverse bytes in each u32 lane            |
//!
//! Required CPU features for the decode hot path: **SSSE3** (for
//! `pshufb`). For the encode hot path: **SSE4.1** (for `pmulld`).
//! Both are universal on x86_64 silicon since ~2008.
//!
//! ## Implementation status
//!
//! Currently a stub: the helpers below compile and produce correct
//! results, but they go through the scalar path (manual 4-lane
//! unroll). Replace the bodies with intrinsics one at a time and
//! verify against the quickcheck tests at the bottom of the file.

use std::arch::x86_64::__m128i;

/// Lane-wise `input / 85`. Magic constants verified for the full u32 range.
#[inline]
pub(crate) fn div_85(input: __m128i) -> __m128i {
    // m = ceil(2^38 / 85) = 3_233_857_729; m * 85 - 2^38 = 21 < 2^6.
    div_magic::<3_233_857_729, 6>(input)
}

/// Lane-wise `input / 7225` (i.e. `input / 85²`). Valid for all u32 inputs.
#[inline]
pub(crate) fn div_85_sq(input: __m128i) -> __m128i {
    // m = ceil(2^44 / 7225) = 2_434_904_643; m * 7225 - 2^44 = 1259 < 2^12.
    div_magic::<2_434_904_643, 12>(input)
}

/// Lane-wise `input / 614125` (i.e. `input / 85³`). Valid for all u32 inputs.
#[inline]
pub(crate) fn div_85_cube(input: __m128i) -> __m128i {
    // m = ceil(2^51 / 614125) = 3_666_679_933; bound check passes for u32.
    div_magic::<3_666_679_933, 19>(input)
}

/// Generic libdivide-style div: `q = (n * MAGIC) >> (32 + SHIFT)`.
///
/// SSE2 path. The widening multiply on x86 is `_mm_mul_epu32`, which
/// multiplies the *even* u32 lanes of two `__m128i` registers and
/// produces a `__m128i` containing two u64 results in the even
/// positions. We multiply twice — once for even lanes, once after
/// shuffling odd lanes into even positions — then shift to extract
/// the high u32 of each u64 product, and finally apply the SHIFT.
//
// TODO(x86 SIMD): replace this scalar fallback body with the SSE2
// multiply + shift sequence sketched above. The interface is fixed;
// only the body needs to change. See the existing aarch64.rs for the
// matching NEON sequence.
#[inline]
fn div_magic<const MAGIC: u32, const SHIFT: i32>(input: __m128i) -> __m128i {
    use std::arch::x86_64::{_mm_loadu_si128, _mm_storeu_si128};

    // Stub: dump to stack, do scalar division, reload. Correct but slow.
    let mut lanes = [0u32; 4];
    unsafe {
        _mm_storeu_si128(lanes.as_mut_ptr().cast::<__m128i>(), input);
    }
    for lane in &mut lanes {
        let xl = MAGIC as u64;
        let yl = *lane as u64;
        let q = ((xl * yl) >> 32) as u32;
        *lane = q >> SHIFT;
    }
    unsafe { _mm_loadu_si128(lanes.as_ptr().cast::<__m128i>()) }
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
            let q_out = lanes(div_85(load(&input)));
            (0..4).all(|i| q_out[i] == input[i] / 85)
        }

        fn div_85_sq_matches_scalar(block: InputBlock) -> bool {
            let InputBlock(input) = block;
            let q_out = lanes(div_85_sq(load(&input)));
            (0..4).all(|i| q_out[i] == input[i] / (85 * 85))
        }

        fn div_85_cube_matches_scalar(block: InputBlock) -> bool {
            let InputBlock(input) = block;
            let q_out = lanes(div_85_cube(load(&input)));
            (0..4).all(|i| q_out[i] == input[i] / (85 * 85 * 85))
        }

        fn div_85_to_the_4_via_composition(block: InputBlock) -> bool {
            let InputBlock(input) = block;
            let q2 = div_85_sq(load(&input));
            let q4 = div_85_sq(q2);
            let q4_out = lanes(q4);
            (0..4).all(|i| q4_out[i] == input[i] / (85u32.pow(4)))
        }
    }
}
