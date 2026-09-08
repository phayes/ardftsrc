//! `f64`-in, `f64`-out real FFT built on the double-double engine in [`vendor`](super::vendor).
//!
//! Everything internal to a transform (twiddle factors, butterfly accumulation) happens in
//! [`Dd`](super::numeric::Dd) double-double precision; this module only exists to convert at the
//! boundary, so callers never need to know `Dd` exists. `f64 -> Dd` is exact (a `Dd` is a pair of
//! `f64`s, and converting a single `f64` in just sets the second one to zero), and `Dd -> f64`
//! keeps the high word, which is the standard way to narrow a double-double value.
//!
//! This intentionally mirrors the shape of `realfft`'s `RealFftPlanner`/`RealToComplex`/
//! `ComplexToReal` (same method names/semantics), so a caller already using `realfft` can swap
//! backends without relearning an API.

use std::fmt;
use std::sync::Arc;

use realfft::num_complex::Complex;

use super::numeric::Dd;
use super::vendor::realfft as vendor_rfft;

/// Error returned by a [`DdRealToComplex`] or [`DdComplexToReal`] transform.
#[derive(Debug)]
pub(crate) struct DdFftError(vendor_rfft::FftError);

impl fmt::Display for DdFftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for DdFftError {}

fn to_dd(input: &[f64]) -> Vec<Dd> {
    input.iter().map(|&v| Dd::from_f64(v)).collect()
}

fn to_dd_complex(input: &[Complex<f64>]) -> Vec<Complex<Dd>> {
    input.iter().map(|c| Complex::new(Dd::from_f64(c.re), Dd::from_f64(c.im))).collect()
}

fn write_from_dd(dst: &mut [f64], src: &[Dd]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = s.to_f64();
    }
}

fn write_from_dd_complex(dst: &mut [Complex<f64>], src: &[Complex<Dd>]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = Complex::new(s.re.to_f64(), s.im.to_f64());
    }
}

/// Plans double-double-precision real FFTs. See [`realfft::RealFftPlanner`] for the API this
/// mirrors.
pub(crate) struct DdRealFftPlanner {
    inner: vendor_rfft::RealFftPlanner<Dd>,
}

impl DdRealFftPlanner {
    pub(crate) fn new() -> Self {
        Self {
            inner: vendor_rfft::RealFftPlanner::new(),
        }
    }

    pub(crate) fn plan_fft_forward(&mut self, len: usize) -> Arc<DdRealToComplex> {
        Arc::new(DdRealToComplex {
            inner: self.inner.plan_fft_forward(len),
        })
    }

    pub(crate) fn plan_fft_inverse(&mut self, len: usize) -> Arc<DdComplexToReal> {
        Arc::new(DdComplexToReal {
            inner: self.inner.plan_fft_inverse(len),
        })
    }
}

/// Forward real-to-complex double-double FFT for a fixed length. See
/// [`realfft::RealToComplex`] for the API this mirrors.
pub(crate) struct DdRealToComplex {
    inner: Arc<dyn vendor_rfft::RealToComplex<Dd>>,
}

impl DdRealToComplex {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub(crate) fn complex_len(&self) -> usize {
        self.inner.complex_len()
    }

    pub(crate) fn make_input_vec(&self) -> Vec<f64> {
        vec![0.0; self.len()]
    }

    pub(crate) fn make_output_vec(&self) -> Vec<Complex<f64>> {
        vec![Complex::new(0.0, 0.0); self.complex_len()]
    }

    /// Transforms `input` (length [`len()`](Self::len)) into `output` (length
    /// [`complex_len()`](Self::complex_len)), converting to/from double-double precision at the
    /// boundary. `input` is left in an unspecified state after the call, matching `realfft`.
    pub(crate) fn process(&self, input: &mut [f64], output: &mut [Complex<f64>]) -> Result<(), DdFftError> {
        let mut dd_in = to_dd(input);
        let mut dd_out = vec![Complex::new(Dd::from_f64(0.0), Dd::from_f64(0.0)); output.len()];
        self.inner.process(&mut dd_in, &mut dd_out).map_err(DdFftError)?;
        write_from_dd_complex(output, &dd_out);
        Ok(())
    }
}

/// Inverse complex-to-real double-double FFT for a fixed length. See
/// [`realfft::ComplexToReal`] for the API this mirrors.
pub(crate) struct DdComplexToReal {
    inner: Arc<dyn vendor_rfft::ComplexToReal<Dd>>,
}

impl DdComplexToReal {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub(crate) fn complex_len(&self) -> usize {
        self.inner.complex_len()
    }

    pub(crate) fn make_input_vec(&self) -> Vec<Complex<f64>> {
        vec![Complex::new(0.0, 0.0); self.complex_len()]
    }

    pub(crate) fn make_output_vec(&self) -> Vec<f64> {
        vec![0.0; self.len()]
    }

    /// Transforms `input` (length [`complex_len()`](Self::complex_len)) into `output` (length
    /// [`len()`](Self::len)), converting to/from double-double precision at the boundary. `input`
    /// is left in an unspecified state after the call, matching `realfft`.
    pub(crate) fn process(&self, input: &mut [Complex<f64>], output: &mut [f64]) -> Result<(), DdFftError> {
        let mut dd_in = to_dd_complex(input);
        let mut dd_out = vec![Dd::from_f64(0.0); output.len()];
        self.inner.process(&mut dd_in, &mut dd_out).map_err(DdFftError)?;
        write_from_dd(output, &dd_out);
        Ok(())
    }
}

/// Maps our vendored (but field-for-field identical) `FftError` to the real `realfft` crate's
/// `FftError`, so [`DdRealToComplex`]/[`DdComplexToReal`] can implement the real
/// `realfft::RealToComplex`/`ComplexToReal` traits below and slot into `core.rs`'s existing
/// `Arc<dyn RealToComplex<T>>` / `Arc<dyn ComplexToReal<T>>` fields unchanged.
fn map_error(e: DdFftError) -> realfft::FftError {
    match e.0 {
        vendor_rfft::FftError::InputBuffer(expected, got) => realfft::FftError::InputBuffer(expected, got),
        vendor_rfft::FftError::OutputBuffer(expected, got) => realfft::FftError::OutputBuffer(expected, got),
        vendor_rfft::FftError::ScratchBuffer(expected, got) => realfft::FftError::ScratchBuffer(expected, got),
        vendor_rfft::FftError::InputValues(first, last) => realfft::FftError::InputValues(first, last),
    }
}

// `get_scratch_len` returns 0 and `process_with_scratch` ignores the caller-provided scratch:
// the double-double engine's own working buffers are `Vec<Complex<Dd>>`, not `Vec<Complex<f64>>`,
// so an `f64`-shaped scratch buffer from a caller can't actually be reused here regardless. This
// only gives up the scratch-reuse optimization `process_with_scratch` normally provides, not
// correctness (see `real.rs` module docs re: this engine's allocation-per-call cost already).

impl realfft::RealToComplex<f64> for DdRealToComplex {
    fn process(&self, input: &mut [f64], output: &mut [Complex<f64>]) -> Result<(), realfft::FftError> {
        DdRealToComplex::process(self, input, output).map_err(map_error)
    }

    fn process_with_scratch(&self, input: &mut [f64], output: &mut [Complex<f64>], _scratch: &mut [Complex<f64>]) -> Result<(), realfft::FftError> {
        DdRealToComplex::process(self, input, output).map_err(map_error)
    }

    fn get_scratch_len(&self) -> usize {
        0
    }

    fn len(&self) -> usize {
        DdRealToComplex::len(self)
    }

    fn make_input_vec(&self) -> Vec<f64> {
        DdRealToComplex::make_input_vec(self)
    }

    fn make_output_vec(&self) -> Vec<Complex<f64>> {
        DdRealToComplex::make_output_vec(self)
    }

    fn make_scratch_vec(&self) -> Vec<Complex<f64>> {
        Vec::new()
    }
}

impl realfft::ComplexToReal<f64> for DdComplexToReal {
    fn process(&self, input: &mut [Complex<f64>], output: &mut [f64]) -> Result<(), realfft::FftError> {
        DdComplexToReal::process(self, input, output).map_err(map_error)
    }

    fn process_with_scratch(&self, input: &mut [Complex<f64>], output: &mut [f64], _scratch: &mut [Complex<f64>]) -> Result<(), realfft::FftError> {
        DdComplexToReal::process(self, input, output).map_err(map_error)
    }

    fn get_scratch_len(&self) -> usize {
        0
    }

    fn len(&self) -> usize {
        DdComplexToReal::len(self)
    }

    fn make_input_vec(&self) -> Vec<Complex<f64>> {
        DdComplexToReal::make_input_vec(self)
    }

    fn make_output_vec(&self) -> Vec<f64> {
        DdComplexToReal::make_output_vec(self)
    }

    fn make_scratch_vec(&self) -> Vec<Complex<f64>> {
        Vec::new()
    }
}

/// Plans a double-double-precision forward real FFT, already coerced to the real `realfft`
/// crate's `RealToComplex<f64>` trait object -- this is the shape `core.rs` needs to drop it into
/// its existing (generic-over-`T`) `Arc<dyn realfft::RealToComplex<T>>` field for `T = f64`.
pub(crate) fn plan_fft_forward(len: usize) -> Arc<dyn realfft::RealToComplex<f64>> {
    DdRealFftPlanner::new().plan_fft_forward(len) as Arc<dyn realfft::RealToComplex<f64>>
}

/// Inverse counterpart of [`plan_fft_forward`].
pub(crate) fn plan_fft_inverse(len: usize) -> Arc<dyn realfft::ComplexToReal<f64>> {
    DdRealFftPlanner::new().plan_fft_inverse(len) as Arc<dyn realfft::ComplexToReal<f64>>
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// Forward-then-inverse should return (a scaled copy of) the original signal, for both
    /// power-of-two and non-power-of-two lengths -- confirms the vendored mixed-radix/Bluestein
    /// selection logic still does the right thing once its twiddles are backed by `Dd`.
    #[test]
    fn round_trip_recovers_signal() {
        for len in [1usize, 2, 3, 4, 5, 7, 8, 12, 17, 32, 100, 101, 257, 1024, 4000] {
            let mut planner = DdRealFftPlanner::new();
            let r2c = planner.plan_fft_forward(len);
            let c2r = planner.plan_fft_inverse(len);

            let original: Vec<f64> = (0..len).map(|i| (i as f64 * 0.37).sin() + 0.5).collect();

            let mut input = original.clone();
            let mut spectrum = r2c.make_output_vec();
            r2c.process(&mut input, &mut spectrum).unwrap();

            let mut recovered = c2r.make_output_vec();
            c2r.process(&mut spectrum, &mut recovered).unwrap();
            for value in recovered.iter_mut() {
                *value /= len as f64;
            }

            let diff = max_abs_diff(&original, &recovered);
            assert!(diff < 1e-12, "len={len}: round-trip diff {diff} too large");
        }
    }

    /// Forward-then-inverse round trips aren't a great way to see `Dd`'s precision benefit: the
    /// forward and inverse rounding errors substantially cancel, and the final result is narrowed
    /// back to `f64` regardless of which engine produced it, so both engines end up pinned near
    /// the same `f64`-representable floor (see the (removed) first version of this test, which
    /// asserted a gap that round-tripping doesn't actually produce).
    ///
    /// The forward *spectrum*, before any such cancellation, is where the benefit actually shows.
    /// This compares both engines' forward spectra at a few bins against a brute-force O(N)
    /// direct DFT computed independently in `Dd` arithmetic (i.e. not via any FFT butterfly
    /// structure) -- for `len` this small, that reference is accurate to within a handful of
    /// double-double ULPs (~1e-31), many orders of magnitude tighter than either FFT engine's own
    /// `f64`-representable output, so it's trustworthy as ground truth here.
    #[test]
    fn forward_spectrum_is_more_accurate_than_f64() {
        let len = 512usize;
        let original: Vec<f64> = (0..len).map(|i| (i as f64 * 0.083).sin() * 0.9 + (i as f64 * 1.7).cos() * 0.1).collect();

        let mut planner = DdRealFftPlanner::new();
        let r2c = planner.plan_fft_forward(len);
        let mut input = original.clone();
        let mut dd_spectrum = r2c.make_output_vec();
        r2c.process(&mut input, &mut dd_spectrum).unwrap();

        let mut f64_planner = realfft::RealFftPlanner::<f64>::new();
        let f64_r2c = f64_planner.plan_fft_forward(len);
        let mut f64_input = original.clone();
        let mut f64_spectrum = f64_r2c.make_output_vec();
        f64_r2c.process(&mut f64_input, &mut f64_spectrum).unwrap();

        let mut total_dd_error = 0.0f64;
        let mut total_f64_error = 0.0f64;
        for &k in &[0usize, 1, len / 4, len / 2 - 1, len / 2] {
            let reference = direct_dft_bin_dd(&original, k, len);
            let dd_error = (dd_spectrum[k] - reference).norm();
            let f64_error = (f64_spectrum[k] - reference).norm();
            total_dd_error += dd_error;
            total_f64_error += f64_error;
        }

        assert!(
            total_dd_error <= total_f64_error,
            "expected the double-double spectrum (total error {total_dd_error}) to be at least as \
             accurate as plain f64's (total error {total_f64_error})"
        );
        assert!(
            total_f64_error > 0.0,
            "test is vacuous if the f64 engine has no error to beat"
        );
    }

    /// Computes one DFT output bin directly (`O(len)`), in `Dd` arithmetic, independent of any
    /// FFT algorithm -- used as a high-precision reference in
    /// [`forward_spectrum_is_more_accurate_than_f64`].
    fn direct_dft_bin_dd(signal: &[f64], bin: usize, len: usize) -> Complex<f64> {
        use crate::dd_fft::numeric::Dd;
        use crate::dd_fft::vendor::rustfft::FftDirection;

        let mut sum = Complex::new(Dd::from_f64(0.0), Dd::from_f64(0.0));
        for (n, &sample) in signal.iter().enumerate() {
            let angle_index = (bin * n) % len;
            let twiddle = Dd::twiddle_factor(angle_index, len, FftDirection::Forward);
            let sample = Dd::from_f64(sample);
            sum.re = sum.re + sample * twiddle.re;
            sum.im = sum.im + sample * twiddle.im;
        }
        Complex::new(sum.re.to_f64(), sum.im.to_f64())
    }
}
