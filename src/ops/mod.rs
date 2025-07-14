mod aarch64;
mod soft;

pub(crate) use aarch64::Aarch64Radix85;
pub(crate) use soft::ScalarRadix85;

/// 
pub(crate) trait FastRadix85<T, const S: u8> {
    type Quotient;

    /// Performs a fast radix shift operation in base 85.
    fn fast_radix_shift(&self, input: T) -> Self::Quotient;
}

pub(crate) trait FastRemainder85<T, const S: u8> {
    type Remainder;

    /// Computes the remainder from a quotient.
    /// For example, if you divide `input` by `85^S`, this function computes `input % 85^S` using the quotient.
    /// This is useful for cases where you already have the quotient and want to compute the modulus
    /// without performing the full division again.
    fn mod_from_quotient(&self, input: T, quotient: T) -> Self::Remainder;
}

/// Struct to store the constants for fast radix 85 digit operations.
/// Each variant stores a magic number, a shift value, and a multiplier.
/// The constants are used for efficient division and modulus operations in base 85.
/// The operations use techniques similar to those used in the `libdivide` library.
/// See https://ridiculousfish.com/blog/posts/labor-of-division-episode-i.html.
struct Radix85Digit<const S: u8>;

trait ValidDivisor {
    const MAGIC: u32;
    const SHIFT: i32;
}

impl ValidDivisor for Radix85Digit<1> {
    const MAGIC: u32 = 3233857729;
    const SHIFT: i32 = 6;
}

impl ValidDivisor for Radix85Digit<2> {
    const MAGIC: u32 = 2434904643;
    const SHIFT: i32 = 12;
}

impl ValidDivisor for Radix85Digit<3> {
    const MAGIC: u32 = 3666679933;
    const SHIFT: i32 = 19;
}

trait ValidMultiplier {
    const MULTIPLIER: u32;
}

impl ValidMultiplier for Radix85Digit<1> {
    const MULTIPLIER: u32 = 85;
}

impl ValidMultiplier for Radix85Digit<2> {
    // 85^2
    const MULTIPLIER: u32 = 7225;
}

impl ValidMultiplier for Radix85Digit<3> {
    // 85^3
    const MULTIPLIER: u32 = 614_125;
}

impl ValidMultiplier for Radix85Digit<4> {
     // 85^4
    const MULTIPLIER: u32 = 52_200_625;
}


