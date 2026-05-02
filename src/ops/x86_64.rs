//! x86_64 AVX2 helpers — structural mirror of `aarch64.rs`, doubled.
//!
//! Every helper operates on a `__m256i` carrying **eight** u32 lanes
//! (vs the four-lane `uint32x4_t` of NEON). The lower 128-bit half
//! handles blocks 0..3 and the upper half handles blocks 4..7;
//! AVX2's lane-restricted byte/word ops (PSHUFB, BLEND, SHUFFLE)
//! work fine because the algorithm is fully per-lane — each 128-bit
//! half runs an independent copy of the SSE4.1 algorithm we replaced.
//!
//! The magic constants are the same libdivide values used on aarch64;
//! only the SIMD intrinsics differ. AVX2 instructions used:
//!
//! | NEON                   | AVX2                          | what it does                        |
//! |------------------------|-------------------------------|-------------------------------------|
//! | `vmull_u32`            | `_mm256_mul_epu32`            | u32×u32 → u64 widening (4 lanes)    |
//! | `vshrq_n_u32`          | `_mm256_srli_epi32`           | logical shift right per u32 lane    |
//! | `vmulq_n_u32`          | `_mm256_mullo_epi32`          | u32×u32 → u32 (low 32 bits, 8 lanes)|
//! | `vsubq_u32`            | `_mm256_sub_epi32`            | u32 lane-wise subtract              |
//! | `vqtbl1q_u8`           | `_mm256_shuffle_epi8`         | byte-shuffle, **per 128-bit lane**  |
//! | `vrev32q_u8`           | `_mm256_shuffle_epi8` + idx   | reverse bytes in each u32 lane      |
//!
//! All SIMD entry points carry `#[target_feature(enable = "avx2")]`
//! so the file compiles on any `x86_64-*-*` target regardless of the
//! default target-feature set. Callers must perform a runtime
//! `is_x86_feature_detected!("avx2")` check before invoking; on hosts
//! lacking AVX2 the crate routes to the scalar fallback.
//!
//! The module-level `#![allow(unsafe_op_in_unsafe_fn)]` keeps the body
//! of each `unsafe fn` an implicit unsafe scope on MSRV 1.85 (without
//! it, each intrinsic call needs its own `unsafe { … }`).

#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::{
    __m256i, _mm256_blend_epi16, _mm256_mul_epu32, _mm256_set1_epi32, _mm256_shuffle_epi32,
    _mm256_srli_epi32, _mm256_srli_epi64,
};

/// Lane-wise `input / 85` (8 u32 lanes). Magic verified for full u32 range.
///
/// # Safety
/// Requires AVX2 to be available on the host CPU.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn div_85(input: __m256i) -> __m256i {
    // m = ceil(2^38 / 85) = 3_233_857_729; m * 85 - 2^38 = 21 < 2^6.
    div_magic::<3_233_857_729, 6>(input)
}

/// Lane-wise `input / 7225` (i.e. `input / 85²`). Valid for all u32 inputs.
///
/// # Safety
/// Requires AVX2 to be available on the host CPU.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn div_85_sq(input: __m256i) -> __m256i {
    // m = ceil(2^44 / 7225) = 2_434_904_643; m * 7225 - 2^44 = 1259 < 2^12.
    div_magic::<2_434_904_643, 12>(input)
}

/// Lane-wise `input / 614125` (i.e. `input / 85³`). Valid for all u32 inputs.
///
/// # Safety
/// Requires AVX2 to be available on the host CPU.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn div_85_cube(input: __m256i) -> __m256i {
    // m = ceil(2^51 / 614125) = 3_666_679_933; bound check passes for u32.
    div_magic::<3_666_679_933, 19>(input)
}

/// Generic libdivide-style div over 8 u32 lanes:
/// `q = (n * MAGIC) >> (32 + SHIFT)`.
///
/// Same algorithm as the SSE4.1 version it replaces, just doubled.
/// `_mm256_mul_epu32` multiplies the low u32 of each u64 lane (so it
/// touches input lanes 0, 2, 4, 6 per call); we do two multiplies —
/// one as-is, one after `_mm256_srli_epi64::<32>` — and combine the
/// high u32 of each u64 product via `_mm256_shuffle_epi32::<0xF5>` +
/// `_mm256_blend_epi16::<0xCC>`. Both of those are lane-restricted ops
/// (per 128-bit half), but our blend pattern lives entirely within each
/// half so that's fine.
///
/// # Safety
/// Requires AVX2 (the function-level `target_feature` satisfies this
/// for any caller that has done the runtime check).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn div_magic<const MAGIC: u32, const SHIFT: i32>(input: __m256i) -> __m256i {
    let magic = _mm256_set1_epi32(MAGIC as i32);

    let even = _mm256_mul_epu32(input, magic);
    let input_odd = _mm256_srli_epi64::<32>(input);
    let odd = _mm256_mul_epu32(input_odd, magic);

    // Per-128-bit-lane shuffle: broadcasts hi u32 of each u64 product
    // across the pair, in each lane independently.
    let even_hi = _mm256_shuffle_epi32::<0b11_11_01_01>(even);
    let odd_hi = _mm256_shuffle_epi32::<0b11_11_01_01>(odd);

    // Per-lane blend at u16 granularity. The 8-bit MASK is applied to
    // each 128-bit lane, so 0xCC interleaves [hi0, hi1, hi2, hi3]
    // within each lane independently.
    let combined = _mm256_blend_epi16::<0xCC>(even_hi, odd_hi);

    _mm256_srli_epi32::<SHIFT>(combined)
}
