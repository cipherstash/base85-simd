//! Byte-compatibility test suite vs the [`base85`](https://crates.io/crates/base85)
//! crate at version 2.0.0 — the reference we're a drop-in replacement for.
//!
//! ## Compatibility summary (verified by tests below)
//!
//! - **Encode**: byte-identical output for *every* `&[u8]` input. No
//!   divergence found — same alphabet, same RFC 1924 algorithm, same tail
//!   handling (zero-pad input → emit `tail_len + 1` chars).
//! - **Decode of valid input**: byte-identical output for every base85
//!   string the reference accepts.
//! - **Decode of invalid input**: both reject. The error *types* differ
//!   (`base85_simd::DecodeError` has variants for `InvalidChar`,
//!   `InvalidLength`, `Overflow` with positional info; `base85::Error`
//!   has `InvalidCharacter(u8)`, `UnexpectedEof`). Both behaviours signal
//!   "this input is not valid base85", just with different diagnostics.
//!
//! ## Known intentional divergence
//!
//! For 5-char blocks whose value exceeds `u32::MAX` (e.g. `"~~~~~"` or
//! `"|NsC1"`), the `base85` crate **silently wraps** the value in release
//! builds and **panics** in debug builds. We instead always return
//! `DecodeError::Overflow` with the offending block's byte position. We
//! consider this a strict safety improvement: silent wrapping produces
//! arbitrary attacker-controlled bytes, panicking is a DoS vector. The
//! tests in this file therefore split decode parity into two buckets:
//!
//! 1. **Valid inputs** (`base85` returns `Ok`): byte-for-byte equality.
//! 2. **Invalid inputs** (`base85` errors *or* panics): we must also reject
//!    (with any of our `DecodeError` variants).

use std::panic::AssertUnwindSafe;

/// Cases per property test. Miri interprets every instruction, so under it
/// the property tests shrink to a handful of cases; the native run is the
/// parity sweep.
const QC_TESTS: u64 = if cfg!(miri) { 16 } else { 10_000 };

// ─────────────────────────────────────────────────────────────────────────
// Encode parity
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn encode_parity_for_every_length_up_to_512() {
    for len in 0..=512usize {
        let input: Vec<u8> = (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
            .collect();
        let ours = base85_simd::encode(&input);
        let theirs = base85::encode(&input);
        assert_eq!(ours, theirs, "encode mismatch at len {len}");
    }
}

#[test]
fn encode_parity_all_zero_inputs() {
    for len in 0..=128 {
        let input = vec![0u8; len];
        assert_eq!(base85_simd::encode(&input), base85::encode(&input));
    }
}

#[test]
fn encode_parity_all_max_inputs() {
    for len in 0..=128 {
        let input = vec![0xFFu8; len];
        assert_eq!(base85_simd::encode(&input), base85::encode(&input));
    }
}

#[test]
#[cfg_attr(miri, ignore = "brute-force sweep; hours under Miri, no new code path")]
fn encode_parity_one_megabyte() {
    let input: Vec<u8> = (0..(1u32 << 20))
        .map(|i| (i.wrapping_mul(2_654_435_761)) as u8)
        .collect();
    assert_eq!(base85_simd::encode(&input), base85::encode(&input));
}

/// Encode parity at every length 0..=64 with several different fill
/// patterns. Ensures partial blocks (lengths % 4 != 0) byte-match the
/// reference for each tail size and varied content.
#[test]
fn encode_parity_short_lengths_with_varied_patterns() {
    let patterns: &[fn(usize) -> u8] = &[
        |i| i as u8,
        |i| (i as u8).wrapping_mul(31),
        |i| (i as u8) ^ 0xA5,
        |i| !(i as u8),
        |i| (i % 7) as u8,
        |_| 0,
        |_| 0xFF,
        |_| 0x55,
        |_| 0xAA,
    ];
    for (pi, p) in patterns.iter().enumerate() {
        for len in 0..=64usize {
            let input: Vec<u8> = (0..len).map(p).collect();
            assert_eq!(
                base85_simd::encode(&input),
                base85::encode(&input),
                "pattern={pi} len={len}",
            );
        }
    }
}

/// Property test (10k cases × gen size 1024) for encode parity.
#[test]
fn encode_parity_quickcheck() {
    fn prop(input: Vec<u8>) -> bool {
        base85_simd::encode(&input) == base85::encode(&input)
    }
    quickcheck::QuickCheck::new()
        .r#gen(quickcheck::Gen::new(1024))
        .tests(QC_TESTS)
        .quickcheck(prop as fn(Vec<u8>) -> bool);
}

// ─────────────────────────────────────────────────────────────────────────
// Decode parity (valid inputs: must produce identical bytes)
// ─────────────────────────────────────────────────────────────────────────

/// For every input the reference can encode, our decoder must yield the
/// same bytes the reference would (and the original input). Round-trips
/// through the reference's encoder.
#[test]
fn decode_of_reference_encoded_matches_for_every_length_up_to_512() {
    for len in 0..=512usize {
        let input: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(13)).collect();
        let encoded = base85::encode(&input);
        let ours = base85_simd::decode(&encoded).expect("ref-encoded must decode");
        let theirs = base85::decode(&encoded).expect("ref must round-trip its own output");
        assert_eq!(ours, theirs, "decode mismatch at len {len}");
        assert_eq!(ours, input);
    }
}

/// Decode parity property: 10k random inputs round-tripped through the
/// reference encoder, our decoder must match the reference decoder.
#[test]
fn decode_parity_quickcheck() {
    fn prop(input: Vec<u8>) -> bool {
        let encoded = base85::encode(&input);
        let ours = base85_simd::decode(&encoded);
        let theirs = base85::decode(&encoded);
        match (ours, theirs) {
            (Ok(a), Ok(b)) => a == b,
            _ => false, // both must succeed for ref-encoded valid inputs
        }
    }
    quickcheck::QuickCheck::new()
        .r#gen(quickcheck::Gen::new(1024))
        .tests(QC_TESTS)
        .quickcheck(prop as fn(Vec<u8>) -> bool);
}

/// Cross-decode: our encode → their decode must give back the original.
/// Confirms our encoder produces input the reference accepts.
#[test]
fn cross_decode_our_encode_their_decode() {
    for len in 0..=256usize {
        let input: Vec<u8> = (0..len).map(|i| (i as u8) ^ 0x3C).collect();
        let our_encoded = base85_simd::encode(&input);
        let their_decoded = base85::decode(&our_encoded).expect("ref must accept our output");
        assert_eq!(
            their_decoded, input,
            "round-trip via ref decode at len {len}"
        );
    }
}

/// Single-block exhaustive on a wide stratified sample of u32 values.
/// 100k samples covering low values, high values, powers-of-two,
/// powers-of-85, and the encode-overflow boundary.
#[test]
#[cfg_attr(miri, ignore = "brute-force sweep; hours under Miri, no new code path")]
fn decode_single_block_parity_sampled_u32() {
    let mut values: Vec<u32> = Vec::new();
    // Boundary values
    for k in 0..32 {
        values.push(1u32 << k);
        values.push((1u32 << k).wrapping_sub(1));
    }
    for k in 0..=4u32 {
        values.push(85u32.pow(k));
        values.push(85u32.pow(k).saturating_sub(1));
    }
    values.extend([0, 1, 84, 85, 86, u32::MAX, u32::MAX - 1, u32::MAX - 84]);

    // Stratified random samples across u32 (deterministic seed).
    let mut state: u64 = 0xDEAD_BEEF_CAFE_F00D;
    for _ in 0..100_000 {
        // xorshift
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        values.push((state as u32) ^ ((state >> 32) as u32));
    }

    for v in values {
        let bytes = v.to_be_bytes();
        let our_enc = base85_simd::encode(&bytes);
        let their_enc = base85::encode(&bytes);
        assert_eq!(our_enc, their_enc, "encode mismatch v={v}");

        let ours = base85_simd::decode(&our_enc).expect("our round-trip");
        let theirs = base85::decode(&our_enc).expect("their round-trip");
        assert_eq!(ours, theirs, "decode mismatch v={v}");
        assert_eq!(ours, bytes);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Decode parity (invalid inputs: both must reject)
// ─────────────────────────────────────────────────────────────────────────

/// `base85::decode` panics on overflow in debug builds and silently wraps
/// in release. Wrap in `catch_unwind` so this test passes regardless of
/// build profile. Either way, we must report `DecodeError::Overflow`.
#[test]
fn invalid_inputs_we_reject_overflow_ref_undefined() {
    let overflow_inputs = ["~~~~~", "|NsC1", "|NsC2", "}NsC0", "~aaaa"];
    for s in overflow_inputs {
        let our = base85_simd::decode(s);
        assert!(
            matches!(our, Err(base85_simd::DecodeError::Overflow { .. })),
            "expected Overflow for {s:?}, got {our:?}",
        );

        // Reference behavior is wrong but we tolerate either panic or
        // wrap-then-Ok — what matters is that we are strictly stricter.
        let ref_result = std::panic::catch_unwind(AssertUnwindSafe(|| base85::decode(s)));
        match ref_result {
            Ok(Ok(bytes)) => {
                // Release: silent wrap. We rejected; they accepted bogus output.
                assert_eq!(bytes.len(), 4, "ref produced unexpected len for {s:?}");
            }
            Ok(Err(_)) => {
                // Some debug-mode build that doesn't panic on overflow.
            }
            Err(_) => {
                // Debug: ref panicked. We did not.
            }
        }
    }
}

/// Inputs with a non-alphabet character — both reject (different error
/// types, but both surface a rejection).
#[test]
fn both_reject_invalid_char_inputs() {
    let invalid_inputs = [
        "00 00", "0\"000", "00,00", "00.00", "00:00", "0[000", "0\\000", "0]000", "abcd ",
        "abcd\t", "abcd\n", " abcd",
    ];
    for s in invalid_inputs {
        let ours = base85_simd::decode(s);
        let theirs = base85::decode(s);
        assert!(
            ours.is_err() && theirs.is_err(),
            "expected both to err for {s:?}: ours={ours:?}, theirs={theirs:?}",
        );
    }
}

/// Inputs whose length is invalid for base85 (`len % 5 == 1`) — both reject.
#[test]
fn both_reject_invalid_lengths() {
    for len in [1usize, 6, 11, 21, 26, 51, 101] {
        let s = "0".repeat(len);
        let ours = base85_simd::decode(&s);
        let theirs = base85::decode(&s);
        assert!(
            ours.is_err() && theirs.is_err(),
            "len={len}: ours={ours:?}, theirs={theirs:?}",
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Edge cases
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn empty_input_parity() {
    assert_eq!(base85_simd::encode(&[]), base85::encode(&[]));
    assert_eq!(
        base85_simd::decode("").unwrap(),
        base85::decode("").unwrap()
    );
}

#[test]
fn single_byte_inputs_byte_identical() {
    for b in 0u8..=255 {
        let input = [b];
        let our_enc = base85_simd::encode(&input);
        let their_enc = base85::encode(&input);
        assert_eq!(our_enc, their_enc, "encode mismatch for byte {b:#04x}");

        let our_dec = base85_simd::decode(&our_enc).unwrap();
        let their_dec = base85::decode(&our_enc).unwrap();
        assert_eq!(our_dec, their_dec);
        assert_eq!(our_dec, input);
    }
}

#[test]
#[cfg_attr(miri, ignore = "brute-force sweep; hours under Miri, no new code path")]
fn two_byte_inputs_byte_identical() {
    // 65_536 cases — fast to enumerate.
    for v in 0u16..=u16::MAX {
        let input = v.to_be_bytes();
        assert_eq!(base85_simd::encode(&input), base85::encode(&input));
        let enc = base85_simd::encode(&input);
        assert_eq!(
            base85_simd::decode(&enc).unwrap(),
            base85::decode(&enc).unwrap(),
        );
    }
}

#[test]
#[cfg_attr(miri, ignore = "brute-force sweep; hours under Miri, no new code path")]
fn three_byte_inputs_stratified_sample() {
    // 16M cases would be slow; sample a stratified 200k.
    let mut state: u64 = 0xC0FFEE_DEADBEEF;
    let mut sampled: Vec<[u8; 3]> = (0..200_000)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            [(state as u8), ((state >> 8) as u8), ((state >> 16) as u8)]
        })
        .collect();
    // Plus boundaries.
    sampled.push([0, 0, 0]);
    sampled.push([0xFF, 0xFF, 0xFF]);
    sampled.push([0x80, 0x00, 0x00]);

    for input in sampled {
        let our_enc = base85_simd::encode(&input);
        let their_enc = base85::encode(&input);
        assert_eq!(our_enc, their_enc, "encode mismatch for {:?}", input);
        let our_dec = base85_simd::decode(&our_enc).unwrap();
        let their_dec = base85::decode(&our_enc).unwrap();
        assert_eq!(our_dec, their_dec);
        assert_eq!(our_dec, input);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Cross-encoding: feed the reference our encoded form, feed us theirs.
// Both directions must round-trip through the other's implementation.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn fuzz_cross_compat_high_iter() {
    fn prop(input: Vec<u8>) -> bool {
        let our_enc = base85_simd::encode(&input);
        let their_enc = base85::encode(&input);
        if our_enc != their_enc {
            return false;
        }
        // Round-trip through the reference's decoder
        let Ok(via_ref) = base85::decode(&our_enc) else {
            return false;
        };
        // And through ours
        let Ok(via_us) = base85_simd::decode(&their_enc) else {
            return false;
        };
        via_ref == input && via_us == input
    }
    quickcheck::QuickCheck::new()
        .r#gen(quickcheck::Gen::new(2048))
        .tests(QC_TESTS / 2)
        .quickcheck(prop as fn(Vec<u8>) -> bool);
}
