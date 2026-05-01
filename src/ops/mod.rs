//! Low-level radix-85 helpers.
//!
//! Most production paths just use `n / 85` / `n % 85` because the compiler
//! lowers those to the libdivide-style multiply-then-shift sequence in
//! release builds. The hand-written NEON path lives in [`aarch64`] because
//! the compiler doesn't auto-vectorize the divisions across u32x4 lanes.

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "aarch64")]
pub(crate) use aarch64::quotients_and_remainders;
