use std::arch::aarch64::*;
use crate::ops::{Aarch64Radix85, FastRadix85, FastRemainder85, ScalarRadix85};

pub struct Block {
    pub(crate) data: u32,
    // TODO: Use the len to detect if the input needs padding
    input_len: usize,
}

impl Block {
    pub fn new(bytes: &[u8]) -> Self {
        let mut input_len = 0;
        // TODO: Handle the case where bytes is less than 4
        let data = u32::from_be_bytes(<[u8; 4]>::try_from(bytes).unwrap());

        Self { data, input_len }
    }

    pub fn encode_into(&self, out_bytes: &mut [u8]) {
        // Start with the least significant byte
        let mut quotient = self.data;
        out_bytes[4] = byte_to_char85(self.data % 85);
        quotient /= 85;
        out_bytes[3] = byte_to_char85(quotient % 85);
        quotient /= 85;
        out_bytes[2] = byte_to_char85(quotient % 85);
        quotient /= 85;
        out_bytes[1] = byte_to_char85(quotient % 85);
        quotient /= 85;
        out_bytes[0] = byte_to_char85(quotient % 85);
    }

    pub fn encode_into_orig(&self, out_bytes: &mut [u8]) {
        //let mut out_bytes: [u8; 5] = [0; 5];
        // Keep a quotient value that we update at each step
        let mut quotient = ScalarRadix85::<1>.fast_radix_shift(self.data);
        // value % 85
        let out4 = ScalarRadix85::<1>.mod_from_quotient(self.data, quotient);

        // input % 85^2 / 85
        // Update the quotient by dividing by 85 again
        // TODO: It would make life a lot easier if the return type told us what value was used for the division
        // That way, the mod_from_quotient function could have a strong type check
        quotient = ScalarRadix85::<1>.fast_radix_shift(quotient);
        // Get the remainder
        let rem = ScalarRadix85::<2>.mod_from_quotient(self.data, quotient);
        // And divide that by 85 to get the next byte
        let out3 = ScalarRadix85::<1>.fast_radix_shift(rem);

        // input % 85^3 / 85^2
        // Update the quotient by dividing by 85 again
        quotient = ScalarRadix85::<1>.fast_radix_shift(quotient);
        // Get the remainder
        let rem = ScalarRadix85::<3>.mod_from_quotient(self.data, quotient);
        // And divide that by 85^2 to get the next byte
        let out2 = ScalarRadix85::<2>.fast_radix_shift(rem);

        // input % 85^4 / 85^3
        // Update the quotient by dividing by 85 again
        quotient = ScalarRadix85::<1>.fast_radix_shift(quotient);
        // Get the remainder
        let rem = ScalarRadix85::<4>.mod_from_quotient(self.data, quotient);
        // And divide that by 85^3 to get the next byte
        let out1 = ScalarRadix85::<3>.fast_radix_shift(rem);

        // input / 85^4
        // The quotient is already the value we want for the first byte
        let out0 = quotient;

        println!("Encoded block: {:?}", [out0, out1, out2, out3, out4]);

        out_bytes[0] = byte_to_char85(out0);
        out_bytes[1] = byte_to_char85(out1);
        out_bytes[2] = byte_to_char85(out2);
        out_bytes[3] = byte_to_char85(out3);
        out_bytes[4] = byte_to_char85(out4);
    }
}

#[inline]
pub(crate) fn byte_to_char85(x85: u32) -> u8 {
    static B85_TO_CHAR: &[u8] =
        b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~";
    //B85_TO_CHAR[x85 as usize]
    x85 as u8
}

pub struct BlockNeon {
    indexes: [uint8x16_t; 5],
}

impl BlockNeon {
    pub fn new() -> Self {
        let index_1_base: [u8; 16] = [0, 0xFF, 0xFF, 0xFF, 0xFF, 4, 0xFF, 0xFF, 0xFF, 0xFF, 8, 0xFF, 0xFF, 0xFF, 0xFF, 12];
        let index_1_base: uint8x16_t = unsafe { vld1q_u8(index_1_base.as_ptr()) };
        let blanks: uint8x16_t = unsafe { vdupq_n_u8(0xFF) };

        // Indexes for table lookup which is used to map the remainders to the output bytes
        // We have 32 SIMD registers per core so we can precompute the indices
        // TODO: This could actually just be loaded from a static array
        // Consider using vld4q_u32
        let index_4_1 = unsafe { vextq_u8(blanks, index_1_base, 12) };
        let index_3_1 = unsafe { vextq_u8(blanks, index_1_base, 13) };
        let index_2_1 = unsafe { vextq_u8(blanks, index_1_base, 14) };
        let index_1_1 = unsafe { vextq_u8(blanks, index_1_base, 15) };

        Self {
            indexes: [
                index_4_1,
                index_3_1,
                index_2_1,
                index_1_1,
                index_1_base,
            ],
        }
    }

    pub fn encode_into(&self, in_bytes: &[u8], out_bytes: &mut [u8]) {
        // TODO: Check that bytes is at least 16 bytes long
        // Pad with zeros if necessary
        let mut data = unsafe { load_u32x4_be(in_bytes) };

        // Initialize the output vector
        let mut out_1: uint8x16_t = unsafe { vdupq_n_u8(0) };

        for i in 0..=4 {
            let uint32x4x2_t(quotients, rem) =
                quotients_and_remainders(data);

            let rem = unsafe { vreinterpretq_u8_u32(rem) };
            // Use vqtbx1q_u8 to insert the value into the output vector
            out_1 = unsafe { vqtbx1q_u8(out_1, rem, self.indexes[i]) };

            data = quotients;
        }

        // Store the output vector into the output bytes
        unsafe { vst1q_u8(out_bytes.as_mut_ptr(), out_1) };
    }

    pub fn encode_into_orig(&self, in_bytes: &[u8], out_bytes: &mut [u8]) {
        // TODO: Check that bytes is at least 16 bytes long
        // Pad with zeros if necessary
        let data = unsafe { load_u32x4_be(in_bytes) };

        // Initialize the output array
        let mut out_1: uint8x16_t = unsafe { vdupq_n_u8(0) };

        // Round 1 -> R4
        let uint32x4x2_t(quotients, r4) =
            quotients_and_remainders(data);

        let r4 = unsafe { vreinterpretq_u8_u32(r4) };
        // Use vqtbx1q_u8 to insert the value into the output vector
        out_1 = unsafe { vqtbx1q_u8(out_1, r4, self.indexes[0]) };

        // Round 2 -> r3
        let uint32x4x2_t(quotients, r3) =
            quotients_and_remainders(quotients);

        let r3 = unsafe { vreinterpretq_u8_u32(r3) };
        // Use vqtbx1q_u8 to insert the value into the output vector
        out_1 = unsafe { vqtbx1q_u8(out_1, r3, self.indexes[1]) };

        // Round 3 -> r2
        let uint32x4x2_t(quotients, r2) =
            quotients_and_remainders(quotients);

        let r2 = unsafe { vreinterpretq_u8_u32(r2) };
        // Use vqtbx1q_u8 to insert the value into the output vector
        out_1 = unsafe { vqtbx1q_u8(out_1, r2, self.indexes[2]) };

        // Round 3 -> r1
        let uint32x4x2_t(quotients, r1) =
            quotients_and_remainders(quotients);

        let r1 = unsafe { vreinterpretq_u8_u32(r1) };
        // Use vqtbx1q_u8 to insert the value into the output vector
        out_1 = unsafe { vqtbx1q_u8(out_1, r1, self.indexes[3]) };

        // Round 3 -> r0
        // NOTE: We don't need the quotient on the last round - so perhaps there is an optimization here?
        let uint32x4x2_t(_, r0) =
            quotients_and_remainders(quotients);

        // Use vqtbx1q_u8 to insert the value into the output vector
        let r0 = unsafe { vreinterpretq_u8_u32(r0) };
        out_1 = unsafe { vqtbx1q_u8(out_1, r0, self.indexes[4]) };

        // Store the output vector into the output bytes
        unsafe { vst1q_u8(out_bytes.as_mut_ptr(), out_1) };
    }
}

// TODO: Move this to the ops module
// TODO: Tests
// TODO: The generics can be constants inside this function
/// Calculates the quotients and remainders for a given input and a denominator of 85 using SIMD operations.
fn quotients_and_remainders(
    input: uint32x4_t,
) -> uint32x4x2_t {
    // Magic numbers for n/85
    const MAGIC: u32 = 3233857729;
    const SHIFT: i32 = 6;
    const MULTIPLIER: u32 = 85;

    unsafe {
        let lo = vmull_n_u32(vget_low_u32(input), MAGIC);
        let hi = vmull_high_n_u32(input, MAGIC);

        // NOTE: This should be the same as shifting the 64-bit value >> 32
        let lo_u32 = vreinterpretq_u32_u64(lo);
        let hi_u32 = vreinterpretq_u32_u64(hi);

        let quotients = vshrq_n_u32(vuzp2q_u32(lo_u32, hi_u32), SHIFT);
        let remainders = vsubq_u32(input, vmulq_n_u32(quotients, MULTIPLIER));

        uint32x4x2_t(quotients, remainders)
    }
}

// TODO: This step is annpying but required because the original base85 crate converts each word to big-endian integers
// We might be able to optimize by reversing the entire input array before processing
// and recalculating the indices
unsafe fn load_u32x4_be(bytes: &[u8]) -> uint32x4_t {
    // Load the 16 bytes into a uint32x4_t (little endian by default)
    unsafe {
        let raw = vld1q_u32(bytes.as_ptr() as *const u32);

        // Reinterpret as u8x16 so we can byte-swap
        let raw_bytes: uint8x16_t = vreinterpretq_u8_u32(raw);

        // Swap each 4-byte chunk
        let swapped_bytes = vrev32q_u8(raw_bytes);

        // Convert back to uint32x4_t
        vreinterpretq_u32_u8(swapped_bytes)
    }
}