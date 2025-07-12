use std::arch::aarch64::{uint32x2_t, uint32x2x2_t, uint32x4_t, uint64x2_t, vcombine_u32, vmull_u32, vmulq_u32, vshrn_n_u64, vshrq_n_u32, vsubq_u32};
use crate::digit::{Digit, ScalarDigit};

pub(crate) trait SimdDivision: Digit {
    /// Performs SIMD division using the provided digit.
    fn divide_simd(&self, input: uint32x2x2_t) -> uint32x4_t;

    fn mod_simd(&self, input: uint32x2x2_t) -> uint32x4_t {
        // Calculate the quotient using SIMD division
        let quotient = self.divide_simd(input);

        unsafe {
            // Multiply the quotient by the divisor (eg. 85^4)
            let divisor: uint32x4_t = self.multiplier();
            let product: uint32x4_t = vmulq_u32(quotient, divisor);

            // Subtract the product from the original input to get the remainder
            let uint32x2x2_t(low, hi) = input;
            vsubq_u32(vcombine_u32(low, hi), product)
        }
    }
}

impl SimdDivision for ScalarDigit<3> {
    fn divide_simd(&self, input: uint32x2x2_t) -> uint32x4_t {
        divide_simd::<{ Self::SHIFT }>(self.magic(), input)
    }
}

impl SimdDivision for ScalarDigit<2> {
    fn divide_simd(&self, input: uint32x2x2_t) -> uint32x4_t {
        divide_simd::<{ Self::SHIFT }>(self.magic(), input)
    }
}

impl SimdDivision for ScalarDigit<1> {
    fn divide_simd(&self, input: uint32x2x2_t) -> uint32x4_t {
        divide_simd::<{ Self::SHIFT }>(self.magic(), input)
    }
}

#[inline]
fn divide_simd<const S: i32>(magic: uint32x2_t, input: uint32x2x2_t) -> uint32x4_t {
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
    use itertools::Itertools;
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
                            input[i] as u32 / ScalarDigit::<$digit>::MULTIPLIER
                        });

                        let result = ScalarDigit::<$digit>.divide_simd(simd_input);
                        let mut expected: [u32; 4] = [0; 4];
                        unsafe { vst1q_u32(expected.as_mut_ptr() as *mut u32, result) };

                        expected == check
                    }

                    fn [<test_mod_ $digit>](block: InputBlock) -> bool {
                        let InputBlock(input) = block;
                        let simd_input: uint32x2x2_t = unsafe { vld1_u32_x2(input.as_ptr() as *const u32) };

                        let check: [u32; 4] = array::from_fn(|i| {
                            input[i] as u32 % ScalarDigit::<$digit>::MULTIPLIER
                        });

                        let result = ScalarDigit::<$digit>.mod_simd(simd_input);
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

    fn test_all_u32<const S: usize>(range: impl IntoIterator<Item = u32>)
    where
        ScalarDigit<S>: SimdDivision,
    {
        range
            .into_iter()
            .chunks(4)
            .into_iter()
            .for_each(|chunk| {
                let input: Vec<u32> = chunk.collect();
                let simd_input: uint32x2x2_t = unsafe { vld1_u32_x2(input.as_ptr() as *const u32) };

                let check: [u32; 4] = array::from_fn(|i| {
                    input[i] as u32 / ScalarDigit::<S>::MULTIPLIER
                });

                let result = ScalarDigit::<S>.divide_simd(simd_input);
                let mut expected: [u32; 4] = [0; 4];
                unsafe { vst1q_u32(expected.as_mut_ptr() as *mut u32, result) };

                assert_eq!(expected, check, "Division failed for input: {input:?}, expected: {check:?}, got: {expected:?}");
            });
    }

    #[test]
    fn test_all_u32_1_a() {
        test_all_u32::<1>(0..=(u32::MAX >> 3));
    }

    #[test]
    fn test_all_u32_1_b() {
        test_all_u32::<1>((u32::MAX >> 3)..(u32::MAX >> 2));
    }
}
