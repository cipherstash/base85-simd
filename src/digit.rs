use std::arch::aarch64::{uint32x2_t, uint32x4_t, vdup_n_u32, vdupq_n_u32};

// TODO: Reverse the names (e.g., Digit<4> should be Base85Digit<4>)
pub(crate) trait Digit {
    /// The magic number used for division.
    const MAGIC: u32;
    /// The right shift amount used after multiplication.
    const SHIFT: i32;
    /// Multiplier used for remainder calculations.
    const MULTIPLIER: u32;

    #[inline(always)]
    fn multiplier(&self) -> uint32x4_t {
        unsafe { vdupq_n_u32(Self::MULTIPLIER) }
    }

    #[inline(always)]
    fn magic(&self) -> uint32x2_t {
        unsafe { vdup_n_u32(Self::MAGIC) }
    }

    #[cfg(test)]
    fn fast_divide(n: u32) -> u32 {
        let xl = Self::MAGIC as u64;
        let yl = n as u64;
        let q = ((xl * yl) >> 32) as u32;

        q as u32 >> Self::SHIFT
    }

    #[cfg(test)]
    fn fast_mod(n: u32) -> u32 {
        let q = (Self::fast_divide(n) as u64 * Self::MULTIPLIER as u64) as u32;
        n - q
    }
}

// TODO: make S a u8
pub(crate) struct ScalarDigit<const S: usize>;

// TODO: For some reason, when the denominator is 85^4, we lose precision.
// So the trait shouldn't include the constants I guess
// NOTE: I don't even think this is needed
/*impl Base85Digit for Digit<4> {
    //const MAGIC: u32 = 2760794303;
    const MAGIC: u32 = 2760794302;
    // 2760794302.5
    const SHIFT: i32 = 25;
    //const SHIFT: i32 = 24;
    const MULTIPLIER: u32 = 85 * 85 * 85 * 85;

    #[cfg(test)]
    fn fast_divide(n: u32) -> u32 {
        let a = Digit::<2>::fast_divide(n);
        Digit::<2>::fast_divide(a)
    }
}*/

impl Digit for ScalarDigit<3> {
    const MAGIC: u32 = 3666679933;
    const SHIFT: i32 = 19;
    const MULTIPLIER: u32 = 85 * 85 * 85;
}

impl Digit for ScalarDigit<2> {
    const MAGIC: u32 = 2434904643;
    const SHIFT: i32 = 12;
    //const SHIFT: i32 = 11;
    const MULTIPLIER: u32 = 85 * 85;
}

impl Digit for ScalarDigit<1> {
    const MAGIC: u32 = 3233857729;
    //const MAGIC: u64 = 6467715458;
    const SHIFT: i32 = 6;
    //const SHIFT: i32 = 7;
    const MULTIPLIER: u32 = 85;
}

#[cfg(test)]
mod tests {
    use quickcheck::quickcheck;
    use paste::paste;
    use super::*;

    macro_rules! test_digit {
        ($digit:literal) => {
            paste! {
                quickcheck! {
                    fn [<test_divide_ $digit>](input: u32) -> bool {
                        let fast = ScalarDigit::<$digit>::fast_divide(input);
                        let check = input / ScalarDigit::<$digit>::MULTIPLIER;
                        fast == check
                    }

                    fn [<test_mod_ $digit>](input: u32) -> bool {
                        let fast = ScalarDigit::<$digit>::fast_mod(input);
                        let check = input % ScalarDigit::<$digit>::MULTIPLIER;
                        fast == check
                    }
                }
            }
        };
    }

    mod quickchecks {
        use super::*;
        test_digit!(3);
        test_digit!(2);
        test_digit!(1);
    }

    mod all_values {
        use super::*;

        mod divide {
            use super::*;

            /// This function tests the fast division method for all u32 values and the given Digit.
            /// Digit 1 is n/85, digit 2 is n/(85*85), and digit 3 is n/(85*85*85).
            fn divide_all_u32<const S: usize>() where ScalarDigit<S>: Digit {
                (0..u32::MAX)
                    .for_each(|input| {
                        let expected = input / ScalarDigit::<S>::MULTIPLIER;
                        let result = ScalarDigit::<S>::fast_divide(input);
                        assert_eq!(result, expected, "Dividing input {input}, expected {expected} but got {result}");
                    });
            }

            #[test]
            fn u32_3() {
                divide_all_u32::<3>();
            }

            #[test]
            fn u32_2() {
                divide_all_u32::<2>();
            }

            #[test]
            fn u32_1() {
                divide_all_u32::<1>();
            }
        }

        mod modulo {
            use super::*;

            /// This function tests the fast modulo method for all u32 values and the given Digit.
            /// Digit 1 is n % 85, digit 2 is n % (85*85), and digit 3 is n % (85*85*85).
            fn mod_all_u32<const S: usize>() where ScalarDigit<S>: Digit {
                (0..u32::MAX)
                    .for_each(|input| {
                        let expected = input % ScalarDigit::<S>::MULTIPLIER;
                        let result = ScalarDigit::<S>::fast_mod(input);
                        assert_eq!(result, expected, "Modulo input {input}, expected {expected} but got {result}");
                    });
            }

            #[test]
            fn u32_3() {
                mod_all_u32::<3>();
            }

            #[test]
            fn u32_2() {
                mod_all_u32::<2>();
            }

            #[test]
            fn u32_1() {
                mod_all_u32::<1>();
            }
        }
    }
}