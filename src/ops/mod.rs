mod aarch64;
mod soft;

/// 
pub(crate) trait FastRadix85<T, const S: u8> {
    type Output;

    /// Performs a fast radix shift operation in base 85.
    fn fast_radix_shift(&self, input: T) -> Self::Output;

    /// Performs a fast radix modulus operation in base 85.
    fn fast_radix_mod(&self, input: T) -> Self::Output where T: Clone {
        let quotient = self.fast_radix_shift(input.clone());
        self.mod_from_quotient(input, quotient)
    }

    /// Computes the remainder from a quotient.
    /// For example, if you divide `input` by `85^S`, this function computes `input % 85^S` using the quotient.
    /// This is useful for cases where you already have the quotient and want to compute the modulus
    /// without performing the full division again.
    fn mod_from_quotient(&self, input: T, quotient: Self::Output) -> Self::Output;
}

/// Struct to store the constants for fast radix 85 digit operations.
/// Each variant stores a magic number, a shift value, and a multiplier.
/// The constants are used for efficient division and modulus operations in base 85.
/// The operations use techniques similar to those used in the `libdivide` library.
/// See https://ridiculousfish.com/blog/posts/labor-of-division-episode-i.html.
struct Radix85Digit<const S: u8>;

trait ValidDigit {
    const MAGIC: u32;
    const SHIFT: i32;
    const MULTIPLIER: u32;
}

impl ValidDigit for Radix85Digit<1> {
    const MAGIC: u32 = 3233857729;
    const SHIFT: i32 = 6;
    const MULTIPLIER: u32 = 85;
}

impl ValidDigit for Radix85Digit<2> {
    const MAGIC: u32 = 2434904643;
    const SHIFT: i32 = 12;
    const MULTIPLIER: u32 = 7225;
}

impl ValidDigit for Radix85Digit<3> {
    const MAGIC: u32 = 3666679933;
    const SHIFT: i32 = 19;
    const MULTIPLIER: u32 = 614125;
}


