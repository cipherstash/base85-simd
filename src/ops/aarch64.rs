use std::arch::aarch64::{
    uint32x4_t, uint32x4x2_t, vget_low_u32, vmull_high_n_u32, vmull_n_u32, vmulq_n_u32,
    vreinterpretq_u32_u64, vshrq_n_u32, vsubq_u32, vuzp2q_u32,
};

/// Computes `(input / 85, input % 85)` lane-wise on a u32x4 vector.
///
/// Uses the libdivide-style multiply-then-shift trick:
/// `q = (input * MAGIC) >> 32 >> SHIFT`, then `r = input - q * 85`.
#[inline]
pub(crate) fn quotients_and_remainders(input: uint32x4_t) -> uint32x4x2_t {
    // Magic numbers for n / 85 valid for all u32 inputs.
    const MAGIC: u32 = 3_233_857_729;
    const SHIFT: i32 = 6;
    const DIVISOR: u32 = 85;

    unsafe {
        let lo = vmull_n_u32(vget_low_u32(input), MAGIC);
        let hi = vmull_high_n_u32(input, MAGIC);

        // Reinterpret as u32x4 so we can pull out the high u32 of each
        // u64 product with a single uzp2 — this compiles to a no-op-equivalent
        // pair of moves rather than a vshlq_n_u64.
        let lo_u32 = vreinterpretq_u32_u64(lo);
        let hi_u32 = vreinterpretq_u32_u64(hi);

        let quotients = vshrq_n_u32(vuzp2q_u32(lo_u32, hi_u32), SHIFT);
        let remainders = vsubq_u32(input, vmulq_n_u32(quotients, DIVISOR));

        uint32x4x2_t(quotients, remainders)
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

    quickcheck! {
        fn divmod_85_matches_scalar(block: InputBlock) -> bool {
            let InputBlock(input) = block;
            let v: uint32x4_t = unsafe { vld1q_u32(input.as_ptr()) };
            let uint32x4x2_t(q, r) = quotients_and_remainders(v);

            let mut q_out = [0u32; 4];
            let mut r_out = [0u32; 4];
            unsafe {
                vst1q_u32(q_out.as_mut_ptr(), q);
                vst1q_u32(r_out.as_mut_ptr(), r);
            }

            (0..4).all(|i| q_out[i] == input[i] / 85 && r_out[i] == input[i] % 85)
        }
    }
}
