//! Double-double-precision real FFT engine, gated behind the `extended-precision-fft` feature.
//!
//! `ardftsrc`'s default FFT backend (`realfft`, backed by `rustfft`) computes twiddle factors in
//! plain `f64`, which is what caps [`Config::quality`](crate::Config::quality) for `f64` output
//! (and rejects anything above 8192 for `f32` -- see `Error::QualityTooHighForF32`). This module
//! is an alternative backend for callers who want to push past that ceiling: it runs the same FFT
//! algorithms (mixed-radix, Bluestein's, Rader's -- vendored from `rustfft`, see `vendor` for
//! details and rationale) but with every twiddle factor and internal accumulation computed in
//! double-double precision via [`twofloat::TwoFloat`] (~106-bit mantissa, vs. 53 for `f64`).
//!
//! It is not a general replacement: there's no SIMD, and double-double arithmetic costs roughly
//! one to two orders of magnitude more than `f64` per operation, so this is meant for offline /
//! opt-in use at extreme quality settings, not realtime streaming.
//!
//! Only real-input FFTs are exposed (see `real`), matching what `ardftsrc`'s core pipeline needs
//! from `realfft`.

mod numeric;
mod real;
mod vendor;

pub(crate) use real::{plan_fft_forward, plan_fft_inverse};
