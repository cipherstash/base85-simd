use std::{arch::aarch64::{
    uint16x8_t, vreinterpretq_u16_u32,
    vzip1q_u16, vzip2q_u16,
    vtbl1_u8, uint8x8_t, vld1_u8,
    vqtbl1q_u8, vqtbx1q_u8,
    vextq_u8,
    vdupq_n_u8,
    uint32x2_t, uint32x4_t, uint64x2_t, uint8x16_t, vaesmcq_u8, vandq_u32, vcombine_u32, vdupq_n_u32, vextq_u32, vget_low_u32, vld1_u32, vld1q_dup_u32, vld1q_s32, vld1q_u32, vld1q_u8, vmull_high_n_u32, vmull_n_u32, vmull_u32, vmulq_u32, vorrq_u8, vreinterpretq_u32_u64, vreinterpretq_u64_u32, vreinterpretq_u8_u32, vshlq_u32, vshrn_n_u32, vshrn_n_u64, vshrq_n_u32, vsubq_u32, vtrn1q_u32, vtrn2q_u32, vzip1q_u32, vzip2q_u32, vzipq_u32}, mem::transmute};

use vitaminc::random::{Generatable, SafeRand, SeedableRng};





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
    //let input: [u8; 16] = Generatable::random(&mut rand).unwrap();
    let input: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    println!("Input: {:?}", input);
    println!("{}", base85_simd::encode_slice(&input));
    println!("{}", base85_simd::encode_slice_neon(&input));

    /*let index_1_base: [u8; 16] = [0, 0xFF, 0xFF, 0xFF, 0xFF, 4, 0xFF, 0xFF, 0xFF, 0xFF, 8, 0xFF, 0xFF, 0xFF, 0xFF, 12];

    unsafe {
        let index_1_base: uint8x16_t = vld1q_u8(index_1_base.as_ptr());
        let blanks: uint8x16_t = vdupq_n_u8(0xFF);
        let index_4_1 = vextq_u8(blanks, index_1_base, 12);
        let index_3_1 = vextq_u8(blanks, index_1_base, 13);
        let index_2_1 = vextq_u8(blanks, index_1_base, 14);
        let index_1_1 = vextq_u8(blanks, index_1_base, 15);
        let index_0_1 = index_1_base;

        println!("Index 1 Base: {:?}", index_1_base);
        println!("Index 4_1:    {:?}", index_4_1);
        println!("Index 3_1:    {:?}", index_3_1);
        println!("Index 2_1:    {:?}", index_2_1);
        println!("Index 1_1:    {:?}", index_1_1);
        println!("Index 0_1:    {:?}", index_0_1);
    }*/
}
