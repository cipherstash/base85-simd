use std::arch::aarch64::{uint32x2_t, uint32x2x2_t, uint32x4_t, uint64x2_t, vcombine_u32, vdup_n_u32, vdupq_n_u32, vmull_u32, vmulq_u32, vshrn_n_u64, vshrq_n_u32, vsubq_u32};
use crate::ops::{FastRadix85, Radix85Digit, ValidDigit};


pub(crate) struct Aarch64Radix85<const S: u8> {
    magic: uint32x2_t,
    multiplier: uint32x4_t,
}

impl<const S: u8> Aarch64Radix85<S> where Radix85Digit<S>: ValidDigit {
    /// Creates a new instance of `Aarch64Radix85` with the appropriate magic number and multiplier preloaded.
    #[inline(always)]
    pub fn new() -> Self {
        let magic = unsafe { vdup_n_u32(Radix85Digit::<S>::MAGIC) };
        let multiplier = unsafe { vdupq_n_u32(Radix85Digit::<S>::MULTIPLIER) };
        Self { magic, multiplier }
    }
}


impl FastRadix85<uint32x2x2_t, 1> for Aarch64Radix85<1> {
    type Output = uint32x4_t;

    fn fast_radix_shift(&self, input: uint32x2x2_t) -> Self::Output {
        divide_simd_uint32x2_t::<{ Radix85Digit::<1>::SHIFT }>(self.magic, input)
    }

    // FIXME: Make the mod a different trait so we can avoid this duplication
    fn mod_from_quotient(&self, input: uint32x2x2_t, quotient: Self::Output) -> Self::Output {
        unsafe {
            // Multiply the quotient by the divisor (eg. 85^4)
            let product: uint32x4_t = vmulq_u32(quotient, self.multiplier);

            // Subtract the product from the original input to get the remainder
            let uint32x2x2_t(low, hi) = input;
            vsubq_u32(vcombine_u32(low, hi), product)
        }
    }
} 

impl FastRadix85<uint32x2x2_t, 2> for Aarch64Radix85<2> {
    type Output = uint32x4_t;

    fn fast_radix_shift(&self, input: uint32x2x2_t) -> Self::Output {
        divide_simd_uint32x2_t::<{ Radix85Digit::<2>::SHIFT }>(self.magic, input)
    }

    fn mod_from_quotient(&self, input: uint32x2x2_t, quotient: Self::Output) -> Self::Output {
        unsafe {
            // Multiply the quotient by the divisor (eg. 85^4)
            let product: uint32x4_t = vmulq_u32(quotient, self.multiplier);

            // Subtract the product from the original input to get the remainder
            let uint32x2x2_t(low, hi) = input;
            vsubq_u32(vcombine_u32(low, hi), product)
        }
    }
}

impl FastRadix85<uint32x2x2_t, 3> for Aarch64Radix85<3> {
    type Output = uint32x4_t;

    fn fast_radix_shift(&self, input: uint32x2x2_t) -> Self::Output {
        divide_simd_uint32x2_t::<{ Radix85Digit::<3>::SHIFT }>(self.magic, input)
    }

    fn mod_from_quotient(&self, input: uint32x2x2_t, quotient: Self::Output) -> Self::Output {
        unsafe {
            // Multiply the quotient by the divisor (eg. 85^4)
            let product: uint32x4_t = vmulq_u32(quotient, self.multiplier);

            // Subtract the product from the original input to get the remainder
            let uint32x2x2_t(low, hi) = input;
            vsubq_u32(vcombine_u32(low, hi), product)
        }
    }
}

#[inline]
fn divide_simd_uint32x2_t<const S: i32>(magic: uint32x2_t, input: uint32x2x2_t) -> uint32x4_t {
    let uint32x2x2_t(low, hi) = input;

    unsafe {
        // Uses a widening to avoid overflow
        let low_mul: uint64x2_t = vmull_u32(low, magic);
        let hi_mul: uint64x2_t = vmull_u32(hi, magic);

        // Then shift to complete the division
        let low_shifted: uint32x2_t = vshrn_n_u64(low_mul, 32);
        let hi_shifted: uint32x2_t = vshrn_n_u64(hi_mul, 32);

        let combined: uint32x4_t = vcombine_u32(low_shifted, hi_shifted);
        vshrq_n_u32(combined, S)
    }
}

#[cfg(test)]
mod tests {
    use std::{arch::aarch64::{vld1_u32_x2, vst1q_u32}, array, ops::Range};
    use quickcheck::{quickcheck, Arbitrary};
    use paste::paste;
    use super::*;

    #[derive(Debug, Clone, Copy)]
    struct InputBlock([u32; 4]);

    impl Arbitrary for InputBlock {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let blocks = array::from_fn(|_| u32::arbitrary(g));
            InputBlock(blocks)
        }
    }

    macro_rules! test_digit {
        ($digit:literal) => {
            paste! {
                quickcheck! {
                    fn [<test_divide_ $digit>](block: InputBlock) -> bool {
                        let InputBlock(input) = block;
                        let simd_input: uint32x2x2_t = unsafe { vld1_u32_x2(input.as_ptr() as *const u32) };

                        let check: [u32; 4] = array::from_fn(|i| {
                            input[i] as u32 / Radix85Digit::<$digit>::MULTIPLIER
                        });

                        let result = Aarch64Radix85::<$digit>::new().fast_radix_shift(simd_input);
                        let mut expected: [u32; 4] = [0; 4];
                        unsafe { vst1q_u32(expected.as_mut_ptr() as *mut u32, result) };

                        expected == check
                    }

                    fn [<test_mod_ $digit>](block: InputBlock) -> bool {
                        let InputBlock(input) = block;
                        let simd_input: uint32x2x2_t = unsafe { vld1_u32_x2(input.as_ptr() as *const u32) };

                        let check: [u32; 4] = array::from_fn(|i| {
                            input[i] as u32 % Radix85Digit::<$digit>::MULTIPLIER
                        });

                        let result = Aarch64Radix85::<$digit>::new().fast_radix_mod(simd_input);
                        let mut expected: [u32; 4] = [0; 4];
                        unsafe { vst1q_u32(expected.as_mut_ptr() as *mut u32, result) };

                        expected == check
                    }
                }
            }
        };
    }

    test_digit!(3);
    test_digit!(2);
    test_digit!(1);
}
