//! Double-double numeric type used internally by the [`dd_fft`](super) engine.
//!
//! [`twofloat::TwoFloat`] represents a value as the unevaluated sum of two non-overlapping `f64`s
//! (~106 bits of mantissa, vs. 53 for a plain `f64`), and provides correctly-range-reduced
//! `sin`/`cos` plus a double-double `PI`/`TAU` constant. That's exactly what's needed to compute
//! FFT twiddle factors at much higher precision than `f64` -- see the `FftNum` doc comment in
//! `vendor::common` for why that matters here.
//!
//! `TwoFloat` itself can't be used directly as the vendored engine's `T: FftNum` because it
//! doesn't implement the `num_traits` traits (`Zero`, `One`, `Num`, `Signed`, `FromPrimitive`)
//! that `FftNum` and `num_complex::Complex<T>` require, and orphan rules mean we can't add those
//! impls to a foreign type from here. `Dd` is a local newtype that exists purely to carry those
//! impls -- every method below is a thin forward onto the wrapped `TwoFloat`.

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

use num_complex::Complex;
use num_traits::{FromPrimitive, Num, One, Signed, Zero};
use twofloat::TwoFloat;

use super::vendor::rustfft::FftDirection;

#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub(crate) struct Dd(pub(crate) TwoFloat);

impl Dd {
    #[inline]
    pub(crate) fn from_f64(value: f64) -> Self {
        Dd(TwoFloat::from(value))
    }

    #[inline]
    pub(crate) fn to_f64(self) -> f64 {
        f64::from(self.0)
    }

    /// Builds the unit-magnitude twiddle factor `exp(-2*pi*i*numerator/denominator)` (or its
    /// conjugate for the inverse direction), computing the angle and its sine/cosine entirely in
    /// double-double precision rather than routing through an `f64` intermediate.
    pub(crate) fn twiddle_factor(numerator: usize, denominator: usize, direction: FftDirection) -> Complex<Dd> {
        debug_assert!(denominator > 0);
        let turns = Dd::from_f64(numerator as f64) / Dd::from_f64(denominator as f64);
        let angle = -turns.0 * twofloat::consts::TAU;
        let (sin, cos) = angle.sin_cos();
        let result = Complex {
            re: Dd(cos),
            im: Dd(sin),
        };
        match direction {
            FftDirection::Forward => result,
            FftDirection::Inverse => result.conj(),
        }
    }
}

impl fmt::Debug for Dd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

macro_rules! forward_binop {
    ($trait_:ident, $method:ident) => {
        impl $trait_ for Dd {
            type Output = Dd;
            #[inline]
            fn $method(self, rhs: Dd) -> Dd {
                Dd($trait_::$method(self.0, rhs.0))
            }
        }
    };
}

forward_binop!(Add, add);
forward_binop!(Sub, sub);
forward_binop!(Mul, mul);
forward_binop!(Div, div);
forward_binop!(Rem, rem);

macro_rules! forward_assign_op {
    ($trait_:ident, $method:ident) => {
        impl $trait_ for Dd {
            #[inline]
            fn $method(&mut self, rhs: Dd) {
                $trait_::$method(&mut self.0, rhs.0)
            }
        }
    };
}

forward_assign_op!(AddAssign, add_assign);
forward_assign_op!(SubAssign, sub_assign);
forward_assign_op!(MulAssign, mul_assign);
forward_assign_op!(DivAssign, div_assign);
forward_assign_op!(RemAssign, rem_assign);

impl Neg for Dd {
    type Output = Dd;
    #[inline]
    fn neg(self) -> Dd {
        Dd(-self.0)
    }
}

impl Zero for Dd {
    #[inline]
    fn zero() -> Self {
        Dd(TwoFloat::from(0.0))
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == TwoFloat::from(0.0)
    }
}

impl One for Dd {
    #[inline]
    fn one() -> Self {
        Dd(TwoFloat::from(1.0))
    }
}

impl Num for Dd {
    type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;
    #[inline]
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        <f64 as Num>::from_str_radix(str, radix).map(Dd::from_f64)
    }
}

impl Signed for Dd {
    #[inline]
    fn abs(&self) -> Self {
        Dd(self.0.abs())
    }
    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        if *self <= *other { Dd::zero() } else { *self - *other }
    }
    #[inline]
    fn signum(&self) -> Self {
        Dd(self.0.signum())
    }
    #[inline]
    fn is_positive(&self) -> bool {
        self.0.is_sign_positive()
    }
    #[inline]
    fn is_negative(&self) -> bool {
        self.0.is_sign_negative()
    }
}

impl FromPrimitive for Dd {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        Some(Dd::from_f64(n as f64))
    }
    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        Some(Dd::from_f64(n as f64))
    }
    #[inline]
    fn from_f32(n: f32) -> Option<Self> {
        Some(Dd::from_f64(n as f64))
    }
    #[inline]
    fn from_f64(n: f64) -> Option<Self> {
        Some(Dd::from_f64(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_roundtrip_is_exact() {
        for value in [0.0, 1.0, -1.0, 0.1, 123.456, 1e-300, 1e300] {
            assert_eq!(Dd::from_f64(value).to_f64(), value);
        }
    }

    #[test]
    fn twiddle_matches_f64_to_within_f64_precision() {
        for denominator in [4usize, 5, 7, 1024] {
            for numerator in 0..denominator {
                let dd = Dd::twiddle_factor(numerator, denominator, FftDirection::Forward);
                let f64_result: Complex<f64> = {
                    let angle = -2.0 * std::f64::consts::PI * numerator as f64 / denominator as f64;
                    Complex::new(angle.cos(), angle.sin())
                };
                assert!((dd.re.to_f64() - f64_result.re).abs() < 1e-14);
                assert!((dd.im.to_f64() - f64_result.im).abs() < 1e-14);
            }
        }
    }
}
