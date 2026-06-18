//! The [`Scalar`] algebra trait — the linchpin of tmflow's modular design.
//!
//! A vector field ([`crate::system::System`]) is written **once**, generically
//! over `Scalar`. The same field definition then runs under different number
//! systems by monomorphization:
//!
//! * `f64`            → fast, non-rigorous (used by the RK4 integrator and for
//!   Monte-Carlo reference trajectories),
//! * [`Interval`]     → rigorous interval evaluation (a priori enclosures),
//! * [`TaylorModel`]  → verified set propagation (the Taylor-Model integrator),
//! * [`Jet`]          → automatic time-Taylor expansion of *any* field, so the
//!   verified integrator needs no hand-derived recurrence.
//!
//! [`Interval`]: crate::interval::Interval
//! [`TaylorModel`]: crate::taylor_model::TaylorModel
//! [`Jet`]: crate::jet::Jet
//!
//! Keeping the surface small (add/sub/neg/mul + scalar lifting) is deliberate:
//! it is exactly what polynomial vector fields need, it is trivially
//! implementable for every backend, and it keeps the field author's code
//! readable (`x * (one - x*x)` rather than method soup).

use std::ops::{Add, Mul, Neg, Sub};

/// A commutative-ring-like algebra over the reals supporting the operations a
/// (polynomial) ODE vector field needs.
///
/// Implementors: [`f64`], [`crate::interval::Interval`],
/// [`crate::taylor_model::TaylorModel`], and [`crate::jet::Jet`]. Each must be
/// `Clone` and provide the four arithmetic ops plus lifting of real constants.
///
/// The trait is value-semantic (ops consume `self`) which keeps field code
/// clean; backends with heap state (TM, Jet) are `Clone` and clone internally.
pub trait Scalar:
    Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
{
    /// Lift a real constant into the algebra (e.g. a parameter like `mu`).
    fn from_f64(v: f64) -> Self;

    /// The multiplicative identity. Default lifts `1.0`.
    #[inline]
    fn one() -> Self {
        Self::from_f64(1.0)
    }

    /// The additive identity. Default lifts `0.0`.
    #[inline]
    fn zero() -> Self {
        Self::from_f64(0.0)
    }

    /// Multiply by a real scalar. Default routes through [`Scalar::from_f64`];
    /// backends may override for efficiency.
    #[inline]
    fn scale_real(self, alpha: f64) -> Self {
        self * Self::from_f64(alpha)
    }

    /// Divide by a positive integer `d`, **rigorously** for verified backends.
    ///
    /// `1/d` is generally not representable in f64, so rigorous backends
    /// (interval, Taylor model) must enclose the true rational. The default is
    /// the exact-for-f64 `self.scale_real(1.0/d)`; [`Interval`] and
    /// [`TaylorModel`] override it with an outward-rounded reciprocal so the
    /// time-series recurrence (which divides coefficients by `j+1`) stays sound.
    ///
    /// [`Interval`]: crate::interval::Interval
    /// [`TaylorModel`]: crate::taylor_model::TaylorModel
    #[inline]
    fn div_u32(self, d: u32) -> Self {
        self.scale_real(1.0 / d as f64)
    }

    /// Non-negative integer power by binary exponentiation. A default is
    /// provided in terms of `mul`/`one`; backends may override for tighter or
    /// faster powers.
    fn powi(self, mut n: u32) -> Self {
        if n == 0 {
            return Self::one();
        }
        let mut base = self;
        let mut acc: Option<Self> = None;
        while n > 0 {
            if n & 1 == 1 {
                acc = Some(match acc {
                    None => base.clone(),
                    Some(a) => a * base.clone(),
                });
            }
            n >>= 1;
            if n > 0 {
                base = base.clone() * base;
            }
        }
        acc.unwrap()
    }
}

impl Scalar for f64 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn scale_real(self, alpha: f64) -> Self {
        self * alpha
    }
    #[inline]
    fn powi(self, n: u32) -> Self {
        f64::powi(self, n as i32)
    }
}
