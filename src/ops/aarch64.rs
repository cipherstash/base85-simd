use std::arch::aarch64::{
    uint32x4_t, vget_low_u32, vmull_high_n_u32, vmull_n_u32, vreinterpretq_u32_u64, vshrq_n_u32,
    vuzp2q_u32,
};

/// Lane-wise `input / 85`. Magic constants verified for the full u32 range.
#[inline]
pub(crate) fn div_85(input: uint32x4_t) -> uint32x4_t {
    // m = ceil(2^38 / 85) = 3_233_857_729; m * 85 - 2^38 = 21 < 2^6.
    div_magic::<3_233_857_729, 6>(input)
}

/// Lane-wise `input / 7225` (i.e. `input / 85²`). Valid for all u32 inputs.
#[inline]
pub(crate) fn div_85_sq(input: uint32x4_t) -> uint32x4_t {
    // m = ceil(2^44 / 7225) = 2_434_904_643; m * 7225 - 2^44 = 1259 < 2^12.
    div_magic::<2_434_904_643, 12>(input)
}

/// Lane-wise `input / 614125` (i.e. `input / 85³`). Valid for all u32 inputs.
#[inline]
pub(crate) fn div_85_cube(input: uint32x4_t) -> uint32x4_t {
    // m = ceil(2^51 / 614125) = 3_666_679_933; bound check passes for u32.
    div_magic::<3_666_679_933, 19>(input)
}

/// Generic libdivide-style div: `q = (n * MAGIC) >> (32 + SHIFT)`.
///
/// The caller is responsible for choosing `MAGIC` and `SHIFT` such that
/// the result is `floor(n / d)` for all u32 `n`. The standard libdivide
/// rule is `MAGIC = ceil(2^(32+SHIFT) / d)` together with the precision
/// bound `MAGIC * d - 2^(32+SHIFT) < 2^SHIFT`.
#[inline]
fn div_magic<const MAGIC: u32, const SHIFT: i32>(input: uint32x4_t) -> uint32x4_t {
    unsafe {
        let lo = vmull_n_u32(vget_low_u32(input), MAGIC);
        let hi = vmull_high_n_u32(input, MAGIC);
        let lo_u32 = vreinterpretq_u32_u64(lo);
        let hi_u32 = vreinterpretq_u32_u64(hi);
        // uzp2 takes the high u32 of each u64 (i.e. (n * m) >> 32)
        // — compiles to a single `uzp2.4s`.
        let high_u32 = vuzp2q_u32(lo_u32, hi_u32);
        vshrq_n_u32(high_u32, SHIFT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::{Arbitrary, quickcheck};
    use std::{
        arch::aarch64::{vld1q_u32, vst1q_u32},
        array,
    };

    #[derive(Debug, Clone, Copy)]
    struct InputBlock([u32; 4]);

    impl Arbitrary for InputBlock {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            InputBlock(array::from_fn(|_| u32::arbitrary(g)))
        }
    }

    fn lanes(v: uint32x4_t) -> [u32; 4] {
        let mut out = [0u32; 4];
        unsafe { vst1q_u32(out.as_mut_ptr(), v) };
        out
    }

    fn load(input: &[u32; 4]) -> uint32x4_t {
        unsafe { vld1q_u32(input.as_ptr()) }
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

        // q4 = q2 / 85^2 — the composed path that yields n / 85^4. Direct
        // /85^4 magic would not satisfy the libdivide precision bound.
        fn div_85_to_the_4_via_composition(block: InputBlock) -> bool {
            let InputBlock(input) = block;
            let q2 = div_85_sq(load(&input));
            let q4 = div_85_sq(q2);
            let q4_out = lanes(q4);
            (0..4).all(|i| q4_out[i] == input[i] / (85u32.pow(4)))
        }
    }
}
