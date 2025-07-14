use std::arch::aarch64::{uint32x2_t, uint32x2x2_t, uint32x4_t, uint64x2_t, vcombine_u32, vdup_n_u32, vdupq_n_u32, vget_low_u32, vmull_high_n_u32, vmull_n_u32, vmull_u32, vmulq_n_u32, vmulq_u32, vshrn_n_u64, vshrq_n_u32, vsubq_u32};
use crate::ops::{FastRadix85, FastRemainder85, Radix85Digit, ValidDivisor, ValidMultiplier};


pub(crate) struct Aarch64Radix85<const S: u8>;


impl FastRadix85<uint32x4_t, 1> for Aarch64Radix85<1> {
    type Quotient = uint32x4_t;

    fn fast_radix_shift(&self, input: uint32x4_t) -> Self::Quotient {
        divide_simd_uint32x2_t::<{ Radix85Digit::<1>::SHIFT }, { Radix85Digit::<1>::MAGIC }>(input)
    }
}

impl FastRadix85<uint32x4_t, 2> for Aarch64Radix85<2> {
    type Quotient = uint32x4_t;

    fn fast_radix_shift(&self, input: uint32x4_t) -> Self::Quotient {
        divide_simd_uint32x2_t::<{ Radix85Digit::<2>::SHIFT }, { Radix85Digit::<2>::MAGIC }>(input)
    }
}

impl FastRadix85<uint32x4_t, 3> for Aarch64Radix85<3> {
    type Quotient = uint32x4_t;

    fn fast_radix_shift(&self, input: uint32x4_t) -> Self::Quotient {
        divide_simd_uint32x2_t::<{ Radix85Digit::<3>::SHIFT }, { Radix85Digit::<3>::MAGIC }>(input)
    }
}


impl FastRemainder85<uint32x4_t, 1> for Aarch64Radix85<1> {
    type Remainder = uint32x4_t;

    fn mod_from_quotient(&self, input: uint32x4_t, quotient: uint32x4_t) -> Self::Remainder {
        remainder_simd_uint32x2_t::< { Radix85Digit::<1>::MULTIPLIER }>(input, quotient)
    }
} 

impl FastRemainder85<uint32x4_t, 2> for Aarch64Radix85<2> {
    type Remainder = uint32x4_t;

    fn mod_from_quotient(&self, input: uint32x4_t, quotient: uint32x4_t) -> Self::Remainder {
        remainder_simd_uint32x2_t::< { Radix85Digit::<2>::MULTIPLIER }>(input, quotient)
    }
}

impl FastRemainder85<uint32x4_t, 3> for Aarch64Radix85<3> {
    type Remainder = uint32x4_t;

    fn mod_from_quotient(&self, input: uint32x4_t, quotient: uint32x4_t) -> Self::Remainder {
        remainder_simd_uint32x2_t::< { Radix85Digit::<3>::MULTIPLIER }>(input, quotient)
    }
}

impl FastRemainder85<uint32x4_t, 4> for Aarch64Radix85<4> {
    type Remainder = uint32x4_t;

    fn mod_from_quotient(&self, input: uint32x4_t, quotient: uint32x4_t) -> Self::Remainder {
        remainder_simd_uint32x2_t::< { Radix85Digit::<4>::MULTIPLIER }>(input, quotient)
    }
}


#[inline]
fn divide_simd_uint32x2_t<const S: i32, const M: u32>(input: uint32x4_t) -> uint32x4_t {
    unsafe {
        let mult_h = vmull_high_n_u32(input, M);
        let mult_l = vmull_n_u32(vget_low_u32(input), M);

        // TODO: Consider using a reinterpret and transpose here instead
        // Then shift to complete the division
        let low_shifted: uint32x2_t = vshrn_n_u64(mult_l, 32);
        let hi_shifted: uint32x2_t = vshrn_n_u64(mult_h, 32);

        let combined: uint32x4_t = vcombine_u32(low_shifted, hi_shifted);
        vshrq_n_u32(combined, S)
    }
}

#[inline]
fn remainder_simd_uint32x2_t<const M: u32>(input: uint32x4_t, quotient: uint32x4_t) -> uint32x4_t {
    unsafe {
        // Multiply the quotient by the divisor (eg. 85^1)
        let product: uint32x4_t = vmulq_n_u32(quotient, M);

        // Subtract the product from the original input to get the remainder
        vsubq_u32(input, product)
    }
}

#[cfg(test)]
mod tests {
    use std::{arch::aarch64::{vld1q_u32, vst1q_u32}, array};
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

    macro_rules! test_divide {
        ($digit:literal) => {
            paste! {
                quickcheck! {
                    fn [<test_divide_ $digit>](block: InputBlock) -> bool {
                        let InputBlock(input) = block;
                        let simd_input: uint32x4_t = unsafe { vld1q_u32(input.as_ptr() as *const u32) };

                        let check: [u32; 4] = array::from_fn(|i| {
                            input[i] as u32 / Radix85Digit::<$digit>::MULTIPLIER
                        });

                        let result = Aarch64Radix85::<$digit>.fast_radix_shift(simd_input);
                        let mut expected: [u32; 4] = [0; 4];
                        unsafe { vst1q_u32(expected.as_mut_ptr() as *mut u32, result) };

                        expected == check
                    }
                }
            }
        };
    }

    macro_rules! test_remainder {
        ($digit:literal) => {
            paste! {
                quickcheck! {
                    fn [<test_remainder_from_quotient_ $digit>](block: InputBlock) -> bool {
                        let InputBlock(input) = block;
                        let simd_input: uint32x4_t = unsafe { vld1q_u32(input.as_ptr() as *const u32) };

                        let check: [u32; 4] = array::from_fn(|i| {
                            input[i] as u32 % Radix85Digit::<$digit>::MULTIPLIER
                        });

                        let quotients = Aarch64Radix85::<$digit>.fast_radix_shift(simd_input);
                        let result = Aarch64Radix85::<$digit>.mod_from_quotient(simd_input, quotients);

                        let mut expected: [u32; 4] = [0; 4];
                        unsafe { vst1q_u32(expected.as_mut_ptr() as *mut u32, result) };

                        expected == check
                    }
                }
            }
        };
    }

    test_divide!(3);
    test_divide!(2);
    test_divide!(1);

    test_remainder!(3);
    test_remainder!(2);
    test_remainder!(1);

    quickcheck! {
        // Special case for digit 4
        // Note that we need to calculate the remainder from the quotient for digit 4 but not the quotient
        fn test_remainder_from_quotient_4(block: InputBlock) -> bool {
            let InputBlock(input) = block;
            let simd_input: uint32x4_t = unsafe { vld1q_u32(input.as_ptr() as *const u32) };

            let check: [u32; 4] = array::from_fn(|i| {
                input[i] as u32 % Radix85Digit::<4>::MULTIPLIER
            });

            // Divide the input by 85^2 twice to get the quotient for digit 4
            // We don't need this calculation outside of tests so it isn't implemented in the trait
            let quotients = Aarch64Radix85::<2>.fast_radix_shift(simd_input);
            let quotients = Aarch64Radix85::<2>.fast_radix_shift(quotients);
            let result = Aarch64Radix85::<4>.mod_from_quotient(simd_input, quotients);

            let mut expected: [u32; 4] = [0; 4];
            unsafe { vst1q_u32(expected.as_mut_ptr() as *mut u32, result) };

            expected == check
        }
    }
}
