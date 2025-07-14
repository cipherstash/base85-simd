use std::{arch::aarch64::{uint32x2_t, uint32x2x2_t, uint32x4_t, vandq_u32, vcombine_u32, vld1_u32, vld1_u32_x2, vld1q_u32, vorrq_u32, vst1q_u32}, array, io::{Cursor, Read}, mem::MaybeUninit};
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

        //println!("Encoded block: {:?}", out_bytes);
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
    B85_TO_CHAR[x85 as usize]
}

pub struct BlockNeon {
    data: uint32x4_t,
}

impl BlockNeon {
    pub fn new(bytes: &[u8]) -> Self {
        Self { data: unsafe { vld1q_u32(bytes.as_ptr() as *const u32) } }
    }

    pub fn encode_into(&self, out_bytes: &mut [u8]) {
        let mut quotient = Aarch64Radix85::<1>.fast_radix_shift(self.data);

        let out4 = Aarch64Radix85::<1>.mod_from_quotient(self.data, quotient);

        quotient = Aarch64Radix85::<1>.fast_radix_shift(quotient);
        let rem = Aarch64Radix85::<2>.mod_from_quotient(self.data, quotient);
        // TODO: This AND is bogus
        let out3 = unsafe { vorrq_u32(Aarch64Radix85::<1>.fast_radix_shift(rem), out4) };

        quotient = Aarch64Radix85::<1>.fast_radix_shift(quotient);
        let rem = Aarch64Radix85::<3>.mod_from_quotient(self.data, quotient);
        // TODO: This AND is bogus
        let out2 = unsafe { vorrq_u32(Aarch64Radix85::<2>.fast_radix_shift(rem), out3) };

        quotient = Aarch64Radix85::<1>.fast_radix_shift(quotient);
        let rem = Aarch64Radix85::<4>.mod_from_quotient(self.data, quotient);
        // TODO: This AND is bogus
        let out1 = unsafe { vorrq_u32(Aarch64Radix85::<3>.fast_radix_shift(rem), out2) };

        // TODO: This AND is bogus
        let out0 = unsafe { vorrq_u32(quotient, out1) };

        unsafe {
            vst1q_u32(out_bytes.as_mut_ptr() as *mut u32, out0);
        }
    }
}