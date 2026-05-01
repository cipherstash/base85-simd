//! Low-level radix-85 helpers.
//!
//! Production scalar code uses plain `n / 85` / `n % 85` because the
//! compiler lowers those to the libdivide-style multiply-then-shift
//! sequence in release builds. The hand-written NEON helpers in
//! [`aarch64`] cover the parallel-divide-by-`85ᵏ` cases the compiler
//! cannot auto-vectorise across u32x4 lanes.

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "aarch64")]
pub(crate) use aarch64::{div_85, div_85_cube, div_85_sq};
