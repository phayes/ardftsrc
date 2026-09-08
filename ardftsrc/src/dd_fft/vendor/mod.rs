//! Third-party code vendored for the [`Dd`](super::numeric::Dd)-precision FFT engine, one
//! subdirectory per upstream crate (named after it, license noted in each subdirectory's module
//! docs). Nothing here is depended on via Cargo -- see below for why.
//!
//! - [`rustfft`] -- the scalar (non-SIMD) FFT algorithms (MIT OR Apache-2.0). This is the one
//!   that's actually modified (see its module docs) rather than a straight copy: every twiddle
//!   factor and derived rotation constant in upstream `rustfft`/`realfft` is computed by building
//!   the angle and taking its `sin`/`cos` in plain `f64`, then converting into whatever
//!   `T: FftNum` the caller asked for. That means a generic `T` never actually gets more than
//!   `f64` precision, no matter how much precision `T` itself can represent -- so implementing
//!   `FftNum` for `Dd` and handing it to the real `rustfft` crate would silently cap accuracy at
//!   `f64` while still paying for the extra arithmetic cost. Getting an actual precision benefit
//!   out of a double-double type requires generating twiddles in that type's own precision, which
//!   is the one thing changed here (`rustfft::common::FftNum::twiddle`).
//! - [`realfft`] -- the real-input/real-output FFT layer built on top of `rustfft` (MIT). Carries
//!   the same one-function fix as `rustfft`, for the same reason.
//! - [`primal_check`] and [`strength_reduce`] (both MIT OR Apache-2.0) -- small helper crates
//!   `rustfft` itself depends on (Miller-Rabin primality testing for Rader's algorithm; fast
//!   integer division for Bluestein's chirp index reduction). Vendored as source rather than
//!   pulled in as Cargo dependencies: both are tiny, single-purpose, and low-maintenance-turnover,
//!   so taking them as source under our own review is a smaller supply-chain footprint than
//!   adding two more external crates to the dependency tree for this one optional feature.

pub(crate) mod primal_check;
pub(crate) mod realfft;
pub(crate) mod rustfft;
pub(crate) mod strength_reduce;
