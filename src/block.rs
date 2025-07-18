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

static B85_TO_CHAR: &[u8] =
    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~";

#[inline]
pub(crate) fn byte_to_char85(x85: u32) -> u8 {
    B85_TO_CHAR[x85 as usize]
}

pub struct BlockNeon {
    indexes_1: [uint8x16_t; 5],
    indexes_2: [uint8x16_t; 4],
}

impl BlockNeon {
    pub fn new() -> Self {
        // TODO: Make all of these load from a static array
        // Calculate the indexes for the base 85 encoding
        // Insanely, this is faster than loading the indexes from a static array, presumably because it avoids the overhead of loading from memory
        let index_1_base: [u8; 16] = [0, 0xFF, 0xFF, 0xFF, 0xFF, 4, 0xFF, 0xFF, 0xFF, 0xFF, 8, 0xFF, 0xFF, 0xFF, 0xFF, 12];
        let index_1_base: uint8x16_t = unsafe { vld1q_u8(index_1_base.as_ptr()) };
        let blanks: uint8x16_t = unsafe { vdupq_n_u8(0xFF) };

        // TODO: It might be possible to just have a single index for each output that we run at the end!
        // Steps 0, 1, 2 and 3 write output bytes 4, 3, 2, and 1 into the aligned output out_1
        // Though we'd still need to do a shift each time to get the next byte into the correct position
        // Next we map some of out_1 to out_2 and the rest into the correct positions in out_1 (2 lookups)
        // Finally, we generate the last remainders, shift them into the correct positions (shr u32), reinterpret and AND with out_1
        // This should be _much_ faster than the current implementation

        // The index vectors are used to map the remainders to the correct positions in the output
        // The first index is the base index, the others are offsets from that index
        let index_4_1 = unsafe { vextq_u8(blanks, index_1_base, 12) };
        let index_3_1 = unsafe { vextq_u8(blanks, index_1_base, 13) };
        let index_2_1 = unsafe { vextq_u8(blanks, index_1_base, 14) };
        let index_1_1 = unsafe { vextq_u8(blanks, index_1_base, 15) };

        // TODO: Precomputing these made bugger all difference to performance so we can probably
        // just use vext in each iteration and reduce the register pressure
        let index_4_2 = unsafe { vextq_u8(index_1_base, blanks, 12) };
        let index_3_2 = unsafe { vextq_u8(index_4_2, index_4_2, 1) };
        let index_2_2 = unsafe { vextq_u8(index_3_2, index_3_2, 1) };
        let index_1_2 = unsafe { vextq_u8(index_2_2, index_2_2, 1) };

        Self {
            indexes_1: [
                index_4_1,
                index_3_1,
                index_2_1,
                index_1_1,
                index_1_base,
            ],
            indexes_2: [
                index_4_2,
                index_3_2,
                index_2_2,
                index_1_2,
            ],
        }
    }

    pub fn encode_into(&self, in_bytes: &[u8], out_bytes: &mut [u8]) {
        // TODO: Check that bytes is at least 16 bytes long
        // Pad with zeros if necessary
        let mut input = unsafe { load_u32x4_be(in_bytes) };

        // Initialize the output vectors
        let mut out_1: uint8x16_t = unsafe { vdupq_n_u8(0) };
        let mut out_2: uint8x16_t = unsafe { vdupq_n_u8(0) };

        for i in 0..4 {
            let uint32x4x2_t(quotients, rem) = quotients_and_remainders(input);
            let rem = unsafe { vreinterpretq_u8_u32(rem) };
            // Use vqtbx1q_u8 to insert the value into the output vectors
            out_1 = unsafe { vqtbx1q_u8(out_1, rem, self.indexes_1[i]) };
            out_2 = unsafe { vqtbx1q_u8(out_2, rem, self.indexes_2[i]) };
            input = quotients;
        }

        let uint32x4x2_t(_, rem) = quotients_and_remainders(input);
        let rem = unsafe { vreinterpretq_u8_u32(rem) };
        out_1 = unsafe { vqtbx1q_u8(out_1, rem, self.indexes_1[4]) };

        out_1 = unsafe { byte_to_char85_neon(out_1) };
        out_2 = unsafe { byte_to_char85_neon(out_2) };

        // Store the output vector into the output bytes
        unsafe {
            vst1q_u8(out_bytes.as_mut_ptr(), out_1);
            let out_2_u32 = vreinterpretq_u32_u8(out_2);
            vst1q_lane_u32::<0>(out_bytes[16..].as_mut_ptr() as *mut u32, out_2_u32);
        };
    }
}

fn print_quad(label: &str, input: uint8x16x4_t) {
    println!("\n{}:", label);
    let uint8x16x4_t(a, b, c, d) = input;
    print_vec("a: ", a);
    print_vec("b: ", b);
    print_vec("c: ", c);
    print_vec("d: ", d);
    println!();
}

fn print_double(label: &str, input: uint8x16x2_t) {
    println!("\n{}:", label);
    let uint8x16x2_t(a, b) = input;
    print_vec("a: ", a);
    print_vec("b: ", b);
    println!();
}

fn print_vec(label: &str, input: uint8x16_t) {
    print!("{}", label);
    let arr = unsafe {
        let mut arr: [u8; 16] = [0; 16];
        vst1q_u8(arr.as_mut_ptr(), input);
        arr
    };
    arr.iter().for_each(|&x| print!("{:02} ", x));
    println!();
}

// TODO: We should load the tbls in the constructor instead
unsafe fn byte_to_char85_neon(x85: uint8x16_t) -> uint8x16_t {
    // B85: 85 characters
    static B85_X4: [u8; 64] = *b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#";
    // 21 characters for the rest of the table padded to 32 bytes so we can fit into 2 vectors
    // The # char is arbitrary and will never be accessed because the input is always < 21 (85 - 64)
    static B85_X2: [u8; 32] = *b"$%&()*+-;<=>?@^_`{|}~###########";

    // Because these are loaded from static arrays, the compiler will optimize the loads
    let t1: uint8x16x4_t = unsafe { vld1q_u8_x4(B85_X4.as_ptr()) };
    let t2: uint8x16x2_t = unsafe { vld1q_u8_x2(B85_X2.as_ptr()) };

    // mask: x85 < 64
    let mask_lt64 = unsafe { vcltq_u8(x85, vdupq_n_u8(64)) };

    // For values < 64 → use vqtbl4q_u8
    let res_lo = unsafe { vqtbl4q_u8(t1, x85) };

    // For values ≥ 64 → subtract 64 and index into t2
    let index_hi = unsafe { vsubq_u8(x85, vdupq_n_u8(64)) };
    let res_hi = unsafe { vqtbl2q_u8(t2, index_hi) };

    // Select result
    unsafe { vbslq_u8(mask_lt64, res_lo, res_hi) }
}


// TODO: Move this to the ops module
// TODO: Tests
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

        // NOTE: This is the same as shifting the 64-bit value >> 32 but will compile to a no-op
        // and will usually be faster than using vshlq_n_u64
        let lo_u32 = vreinterpretq_u32_u64(lo);
        let hi_u32 = vreinterpretq_u32_u64(hi);

        let quotients = vshrq_n_u32(vuzp2q_u32(lo_u32, hi_u32), SHIFT);
        let remainders = vsubq_u32(input, vmulq_n_u32(quotients, MULTIPLIER));

        uint32x4x2_t(quotients, remainders)
    }
}

// This step is required because the original base85 crate converts each word to big-endian integers
// It optimises down to a single `rev32.16b` instruction so it adds very little overhead
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