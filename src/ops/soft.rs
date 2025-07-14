use crate::ops::{FastRadix85, FastRemainder85, Radix85Digit, ValidDivisor, ValidMultiplier};

pub(crate) struct ScalarRadix85<const S: u8>;

// TODO: consider making ValidDigit<S> a supertrait
impl<const S: u8> FastRadix85<u32, S> for ScalarRadix85<S> where Radix85Digit<S>: ValidDivisor {
    type Quotient = u32;

    fn fast_radix_shift(&self, input: u32) -> Self::Quotient {
        let xl = Radix85Digit::<S>::MAGIC as u64;
        let yl = input as u64;
        let q = ((xl * yl) >> 32) as u32;

        q as u32 >> Radix85Digit::<S>::SHIFT
    }
}

impl<const S: u8> FastRemainder85<u32, S> for ScalarRadix85<S> where Radix85Digit<S>: ValidMultiplier {
    type Remainder = u32;

    fn mod_from_quotient(&self, input: u32, quotient: u32) -> Self::Remainder {
        // remainder = input - (input / 85^S) * 85^S
        // i.e. divide by 85^S (which will round down to the nearest whole number), multiply by 85^S, then subtract from input
        input - (quotient * Radix85Digit::<S>::MULTIPLIER)
    }
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
                        let fast = ScalarRadix85::<$digit>.fast_radix_shift(input);
                        let check = input / Radix85Digit::<$digit>::MULTIPLIER;
                        fast == check
                    }

                    fn [<test_mod_ $digit>](input: u32) -> bool {
                        let quotient = ScalarRadix85::<$digit>.fast_radix_shift(input);
                        let remainder = ScalarRadix85::<$digit>.mod_from_quotient(input, quotient);
                        let check = input % Radix85Digit::<$digit>::MULTIPLIER;
                        remainder == check
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
        use std::fmt::Debug;
        use super::*;

        mod divide {
            use super::*;

            /// This function tests the fast division method for all u32 values and the given Digit.
            /// Digit 1 is n/85, digit 2 is n/(85*85), and digit 3 is n/(85*85*85).
            fn divide_all_u32<const S: u8>()
            where
                ScalarRadix85<S>: FastRadix85<u32, S>, Radix85Digit<S>: ValidDivisor + ValidMultiplier,
                <ScalarRadix85<S> as FastRadix85<u32, S>>::Quotient: PartialEq<u32> + Debug
            {
                (0..u32::MAX)
                    .for_each(|input| {
                        let expected = input / Radix85Digit::<S>::MULTIPLIER;
                        let result = ScalarRadix85::<S>.fast_radix_shift(input);
                        assert_eq!(result, expected, "Dividing input {input}, expected {expected} but got {result:?}");
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
            use std::process::Output;

            use super::*;

            /// This function tests the fast modulo method for all u32 values and the given Digit.
            /// Digit 1 is n % 85, digit 2 is n % (85*85), and digit 3 is n % (85*85*85).
            fn mod_all_u32<const S: u8>()
            where
                Radix85Digit<S>: ValidDivisor + ValidMultiplier,
                ScalarRadix85<S>: FastRemainder85<u32, S, Remainder = u32>,
                ScalarRadix85<S>: FastRadix85<u32, S, Quotient = u32>,
            {
                (0..u32::MAX)
                    .for_each(|input| {
                        let expected = input % Radix85Digit::<S>::MULTIPLIER;
                        let quotient = ScalarRadix85::<S>.fast_radix_shift(input);
                        let result = ScalarRadix85::<S>.mod_from_quotient(input, quotient);
                        assert_eq!(result, expected, "Modulo input {input}, expected {expected} but got {result:?}");
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