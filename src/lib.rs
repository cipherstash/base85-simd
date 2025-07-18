//mod digit;
//mod neon;
mod block;
mod ops;

pub use block::Block;
use crate::block::BlockNeon;

#[inline(never)]
pub fn encode_slice(input: &[u8]) -> String {
    let out_len = input.len().div_ceil(4) * 5;
    let mut out = Vec::<u8>::with_capacity(out_len);

    // SAFETY: encode_into will fill the entire buffer
    unsafe {
        out.set_len(out_len);
    }

    let out_chunks = out.chunks_exact_mut(5);

    for (chunk, out) in input.chunks(4).zip(out_chunks) {
        let block = Block::new(chunk);
        block.encode_into(out);
    }
    //println!("Encoded: {:?}\n\n", out);
    unsafe { String::from_utf8_unchecked(out) }
    //"Foo".to_string()
}

#[inline(never)]
pub fn encode_slice_neon(input: &[u8]) -> String {
    let out_len = input.len().div_ceil(4) * 5;
    let mut out = Vec::<u8>::with_capacity(out_len);

    // SAFETY: encode_into will fill the entire buffer
    unsafe {
        out.set_len(out_len);
    }

    let out_chunks = out.chunks_exact_mut(20);
    let encoder = BlockNeon::new();
    for (chunk, out) in input.chunks(16).zip(out_chunks) {
        encoder.encode_into(chunk, out);
    }

    //println!("Encoded: {:?}", out);
    unsafe { String::from_utf8_unchecked(out) }
    //"Foo".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_slice() {
        let input = vec![17, 33, 21, 13];
        let encoded = encode_slice(&input);
        let check = base85::encode(&input);

        assert_eq!(encoded, check);
    }
}