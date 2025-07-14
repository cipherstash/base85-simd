use std::{arch::aarch64::{uint32x2_t, uint32x4_t, uint64x2_t, uint8x16_t, vaesmcq_u8, vandq_u32, vcombine_u32, vdupq_n_u32, vextq_u32, vget_low_u32, vld1_u32, vld1q_dup_u32, vld1q_s32, vld1q_u32, vld1q_u8, vmull_high_n_u32, vmull_n_u32, vmull_u32, vmulq_u32, vorrq_u8, vreinterpretq_u32_u64, vreinterpretq_u64_u32, vreinterpretq_u8_u32, vshlq_u32, vshrn_n_u32, vshrn_n_u64, vshrq_n_u32, vsubq_u32, vtrn1q_u32, vtrn2q_u32, vzip1q_u32, vzip2q_u32, vzipq_u32}, mem::transmute};

use vitaminc::random::{Generatable, SafeRand, SeedableRng};

const MAGIC: u64 = 13889313184910721217; // Magic multiplier for 85
const MAGIC_U32: u32 = 3233857729; // Magic multiplier for 85
const SHIFT: u32 = 6;         // Right shift amount

const MAGIC_U32_2: u32 = 2434904643; // Magic multiplier for 85*85
const SHIFT_2: u32 = 12; // Right shift amount for 85*85

const MAGIC_U32_3: u32 = 3666679933; // Magic multiplier for 85*85*85
const SHIFT_3: u32 = 19; // Right shift amount for 85*85*85

const MAGIC_U32_4: u32 = 2760794303; // Magic multiplier for 85*85*85
const SHIFT_4: i32 = 25; // Right shift amount for 85*85*85

const SHIFT_4_SIMD: i32 = 26; // Right shift amount for 85*85*85






#[inline(always)]
fn floor_log2(n: u64) -> u8 {
    assert_ne!(n, 0);
    //63u8 - (n.leading_zeros() as u8)
    n.ilog2() as u8
}

fn floor_log2_32(n: u32) -> u8 {
    assert_ne!(n, 0);
    31u8 - (n.leading_zeros() as u8)
}

#[inline(always)]
fn libdivide_mullhi_u64(x: u64, y: u64) -> u64 {
    let xl = x as u128;
    let yl = y as u128;
    ((xl * yl) >> 64) as u64
}

fn libdivide_mullhi_u32(x: u32, y: u32) -> u32 {
    let xl = x as u64;
    let yl = y as u64;
    let multiplied = xl * yl;
    println!("multiplied: {}", multiplied);
    (multiplied >> 32) as u32
    //((xl * yl) >> 32) as u32
}

fn fast_path(divisor: u32) -> (u32, u8) {
    //let floor_log_2_d: u8 = floor_log2(divisor);
    //let p = divisor.ilog2();
    let p = 31 - divisor.leading_zeros();
    /*let u = 1u128 << (floor_log_2_d + 64);
    let proposed_magic_number: u128 = u / divisor as u128;
    let reminder: u64 = (u - proposed_magic_number * (divisor as u128)) as u64;
    assert!(reminder > 0 && reminder < divisor);
    let e: u64 = divisor - reminder;
    println!("e: {}, floor_log_2_d: {}", e, floor_log_2_d);
    // This is a sufficient condition for our 64-bits magic number
    // condition to work as described in
    // See https://ridiculousfish.com/blog/posts/labor-of-division-episode-i.html
    if e >= (1u64 << floor_log_2_d) {
        panic!("invalid e");
    }

    let magic = (proposed_magic_number as u64) + 1u64;
    let shift = floor_log_2_d;*/

    let magic = 2u64.pow(32 + p).div_ceil(divisor as u64);

    (magic as u32, (p - 1) as u8)
}

fn fast_path_32(divisor: u32) -> (u32, u8) {
    let floor_log_2_d: u8 = floor_log2_32(divisor);
    let u = 1u64 << (floor_log_2_d + 32);
    let proposed_magic_number: u64 = u / divisor as u64;
    let reminder: u32 = (u - proposed_magic_number * (divisor as u64)) as u32;
    assert!(reminder > 0 && reminder < divisor);
    let e: u32 = divisor - reminder;
    // This is a sufficient condition for our 64-bits magic number
    // condition to work as described in
    // See https://ridiculousfish.com/blog/posts/labor-of-division-episode-i.html
    //if e >= (1u32 << floor_log_2_d) {
    //    panic!("invalid e");
    //}

    let magic = (proposed_magic_number as u32) + 1u32;
    let shift = floor_log_2_d;

    (magic, shift)
}

fn fast_mod_85(n: u64) -> u64 {
    //let q = ((n as u64 * MAGIC) >> SHIFT) as u32;
    let q = libdivide_mullhi_u64(MAGIC, n) >> SHIFT;
    n - q * 85
}

fn fast_divide_by_85(n: u64) -> u64 {
    //((n as u64 * MAGIC) >> SHIFT) as u32
    libdivide_mullhi_u64(MAGIC, n) >> SHIFT
}

fn fast_divide_by_85_32(n: u32) -> u32 {
    
    let xl = MAGIC_U32 as u64;
    let yl = n as u64;
    ((xl * yl) >> 32) as u32 >> SHIFT

    //libdivide_mullhi_u32(MAGIC_U32, n) >> SHIFT
}

fn fast_divide_by_85_4_32(n: u32) -> u32 {
    //((n as u64 * MAGIC) >> SHIFT) as u32
    libdivide_mullhi_u32(MAGIC_U32_4, n) >> SHIFT_4
}

fn fast_mod_85_u32(n: u32) -> u32 {
    //let q = ((n as u64 * MAGIC) >> SHIFT) as u32;
    let q = libdivide_mullhi_u32(MAGIC_U32, n) >> SHIFT;
    n - q * 85
}

fn fast_mod_85_4_u32(n: u32) -> u32 {
    //let q = ((n as u64 * MAGIC) >> SHIFT) as u32;
    let q = libdivide_mullhi_u32(MAGIC_U32_4, n) >> SHIFT_4;
    n - q * 85*85*85*85
}

fn naive(input: [u8; 16]) {
    let blocks: [u32; 4] = unsafe { transmute(input)};
    for block in blocks {
        println!("F {}", libdivide_mullhi_u32(MAGIC_U32_4, block) >> SHIFT_4);
    }
}

fn divide_simd<const S: i32>(low: uint32x2_t, hi: uint32x2_t, magic: uint32x2_t) -> uint32x4_t {
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

fn remainder_simd<const S: i32>(low: uint32x2_t, hi: uint32x2_t, magic: uint32x2_t) -> uint32x4_t {
    let quotient: uint32x4_t = divide_simd::<S>(low, hi, magic);
    unsafe {
        // Multiply the quotient by the divisor (85^4)
        // TODO: Can we precompute this? - this is specific to 85^4
        let divisor: uint32x4_t = vdupq_n_u32(85 * 85 * 85 * 85);
        let product: uint32x4_t = vmulq_u32(quotient, divisor);
        // TODO: Can we avoid the combine here?
        vsubq_u32(vcombine_u32(low, hi), product)
    }
}

fn main() {
    /*dbg!(fast_path_32(85*85*85*85));
    let x = 85*85*85*85 + 42;
    dbg!(fast_divide_by_85_4_32(x));
    dbg!(fast_mod_85_4_u32(x));*/

    // Steps
    // out[0] = in / (85^4)
    // out[1] = in % (85^4) / (85^3)
    // out[2] = in % (85^3) / (85^2)
    // out[3] = in % (85^2) / 85
    // out[4] = in % 85

    let mut rand = SafeRand::from_entropy();
    let input: [u8; 16] = Generatable::random(&mut rand).unwrap();
    
    /*let mask1 = [0, 0xFF, 0xFF, 0xFF];
    let mask2 = [0xFF, 0, 0, 0];
    let input1: [u32; 4] = [1, 2, 3, 4];
    let input2: [u32; 4] = [5, 6, 7, 8];
    let input3: [u32; 4] = [9, 10, 11, 12];
    let input4: [u32; 4] = [13, 14, 15, 16];
    let input5: [u32; 4] = [17, 18, 19, 20];

    let input1: uint32x4_t = unsafe { vld1q_u32(input1.as_ptr() as *const u32) };
    let mask1: uint32x4_t = unsafe { vld1q_u32(mask1.as_ptr() as *const u32) };
    let mask2: uint32x4_t = unsafe { vld1q_u32(mask2.as_ptr() as *const u32) };

    println!("input1: {:?}", input1);

    let rotated = unsafe { vextq_u32(input1, input1, 3) };
    println!("rotated: {:?}", rotated);
    let out1 = unsafe { vandq_u32(rotated, mask1) };
    println!("out1: {:?}", out1);
    let out2 = unsafe { vandq_u32(rotated, mask2) };
    println!("out2: {:?}", out2);*/
    /*let input2: uint32x4_t = unsafe { vld1q_u32(input2.as_ptr() as *const u32) };
    let input3: uint32x4_t = unsafe { vld1q_u32(input3.as_ptr() as *const u32) };
    let input4: uint32x4_t = unsafe { vld1q_u32(input4.as_ptr() as *const u32) };
    let input5: uint32x4_t = unsafe { vld1q_u32(input5.as_ptr() as *const u32) };

    let shift_values: [i32; 4] = [24, 16, 8, 0];

    let mut out = unsafe {
        let shifts = vld1q_s32(shift_values.as_ptr() as *const i32);
        vreinterpretq_u8_u32(vshlq_u32(input1, shifts))
    };

    //println!("shifted: {:?}", out);

    let shift_values: [i32; 4] = [16, 8, 0, -8];

    println!("input2: {:?}", input2);

    out = unsafe {
        let shifts = vld1q_s32(shift_values.as_ptr() as *const i32);
        vorrq_u8(vreinterpretq_u8_u32(vshlq_u32(input2, shifts)), out)
    };*/
    //println!("shifted: {:?}", out);

    /*let shift_values: [i32; 4] = [16, 24, 32, 32];
    out = unsafe {
        let shifts = vld1q_s32(shift_values.as_ptr() as *const i32);
        vorrq_u8(vreinterpretq_u8_u32(vshlq_u32(input3, shifts)), out)
    };
    println!("shifted: {:?}", out);

    let shift_values: [i32; 4] = [24, 32, 32, 32];
    out = unsafe {
        let shifts = vld1q_s32(shift_values.as_ptr() as *const i32);
        vorrq_u8(vreinterpretq_u8_u32(vshlq_u32(input4, shifts)), out)
    };
    println!("shifted: {:?}", out);

    let shift_values: [i32; 4] = [24, 32, 32, 32];
    out = unsafe {
        let shifts = vld1q_s32(shift_values.as_ptr() as *const i32);
        vorrq_u8(vreinterpretq_u8_u32(vshlq_u32(input4, shifts)), out)
    };*/

    //println!("out: {:?}", out);

    
    /*let mult_h = unsafe { vmull_high_n_u32(input, 10) };
    let mult_l = unsafe { vmull_n_u32(vget_low_u32(input), 10) };


    let mult_h_u32 = unsafe { vreinterpretq_u32_u64(mult_h) };
    let mult_l_u32 = unsafe { vreinterpretq_u32_u64(mult_l) };

    let out3 = unsafe { vtrn1q_u32(mult_h_u32, mult_l_u32) };
    println!("out3: {:?}", out3);*/

    //let input: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let out = base85_simd::encode_slice(&input);
    println!("Encoded: {:?}", out);


}
