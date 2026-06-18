//! Rigorous f64 interval arithmetic with guaranteed outward rounding.
//!
//! This is the verified-numerics foundation of the crate. An [`Interval`]
//! `[lo, hi]` is a guaranteed enclosure of a real number (or a set of reals):
//! every operation returns an interval that is *certain* to contain the true
//! mathematical result, including all floating-point round-off.
//!
//! # Soundness model
//!
//! The original Python demonstrator used `mpmath.iv` (arbitrary-precision
//! interval arithmetic, `iv.dps = 40`) and then projected the bounds to f64
//! with `np.nextafter` outward rounding at the boundary. Here we work in f64
//! throughout and obtain rigour by *outward rounding every elementary
//! operation*: after computing a result in round-to-nearest, we nudge the
//! lower bound down one ULP and the upper bound up one ULP. This is
//! conservative (the standard "epsilon-inflation" / `next_after` approach used
//! when hardware rounding-mode switching is unavailable or undesirable) and
//! requires no `unsafe`, no global FPU state, and no heavyweight bignum
//! dependency. The enclosure property holds for `+ - * /`, powers, and the
//! min/max/hull operations the Taylor-model integrator needs.
//!
//! The small price is that bounds are typically one ULP wider than strictly
//! necessary; for a Taylor-model reachability demonstrator the interval width
//! is dominated by the mathematics (truncation remainder), not by this ULP, so
//! the effect on the reported set sizes is negligible — exactly as in the
//! Python version.

use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// A rigorous interval enclosure `[lo, hi]` over the reals (f64 bounds).
///
/// Invariant: `lo <= hi` and neither bound is NaN. The represented set always
/// encloses the true mathematical value of whatever it models.
#[derive(Clone, Copy, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

/// One step toward +infinity (outward for an upper bound).
#[inline]
fn up(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else {
        x.next_up()
    }
}

/// One step toward -infinity (outward for a lower bound).
#[inline]
fn down(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else {
        x.next_down()
    }
}

impl Interval {
    /// Construct `[lo, hi]`. Panics if `lo > hi` or a bound is NaN — this is a
    /// programming error, not a recoverable condition.
    #[inline]
    pub fn new(lo: f64, hi: f64) -> Self {
        debug_assert!(!lo.is_nan() && !hi.is_nan(), "interval bound is NaN");
        debug_assert!(lo <= hi, "interval lo>hi: [{lo}, {hi}]");
        Interval { lo, hi }
    }

    /// A degenerate (point) interval `[v, v]`.
    #[inline]
    pub fn point(v: f64) -> Self {
        Interval { lo: v, hi: v }
    }

    /// The interval `[0, 0]`.
    #[inline]
    pub fn zero() -> Self {
        Interval { lo: 0.0, hi: 0.0 }
    }

    /// The interval `[1, 1]`.
    #[inline]
    pub fn one() -> Self {
        Interval { lo: 1.0, hi: 1.0 }
    }

    /// The unit symmetric interval `[-1, 1]` (the normalized domain coordinate).
    #[inline]
    pub fn unit() -> Self {
        Interval { lo: -1.0, hi: 1.0 }
    }

    /// True if this interval is a single point.
    #[inline]
    pub fn is_point(&self) -> bool {
        self.lo == self.hi
    }

    /// True if both bounds are exactly zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.lo == 0.0 && self.hi == 0.0
    }

    /// Width `hi - lo`, rounded up (outward) so it never under-reports.
    #[inline]
    pub fn width(&self) -> f64 {
        up(self.hi - self.lo)
    }

    /// Midpoint as a plain f64 (interval-safe; uses the outward f64 bounds).
    #[inline]
    pub fn mid(&self) -> f64 {
        0.5 * (self.lo + self.hi)
    }

    /// Radius (half-width), rounded up.
    #[inline]
    pub fn rad(&self) -> f64 {
        up(0.5 * (self.hi - self.lo))
    }

    /// True if `val` lies inside the (closed) interval.
    #[inline]
    pub fn contains_point(&self, val: f64) -> bool {
        self.lo <= val && val <= self.hi
    }

    /// True if `self` contains the whole interval `other`.
    #[inline]
    pub fn contains_interval(&self, other: &Interval) -> bool {
        self.lo <= other.lo && self.hi >= other.hi
    }

    /// Convex hull (smallest interval containing both).
    #[inline]
    pub fn hull(&self, other: &Interval) -> Interval {
        Interval {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// Symmetric inflation about the midpoint by `factor >= 1`, plus a small
    /// absolute pad so degenerate intervals can still grow. Mirrors
    /// `viv.vec_inflate`. The result is rounded outward.
    #[inline]
    pub fn inflate(&self, factor: f64, abs_eps: f64) -> Interval {
        let c = self.mid();
        let r = up(self.rad() * factor + abs_eps);
        Interval {
            lo: down(c - r),
            hi: up(c + r),
        }
    }

    /// Multiply by a real scalar (rounded outward).
    #[inline]
    pub fn scale(&self, alpha: f64) -> Interval {
        *self * Interval::point(alpha)
    }

    /// Rigorous enclosure of `1 / d` for a positive integer `d`, rounded
    /// outward. Exact when `1/d` is representable (d a power of two); otherwise
    /// widened one ULP each way so the true rational is enclosed.
    #[inline]
    pub fn recip_u32(d: u32) -> Interval {
        let q = 1.0_f64 / d as f64;
        if q * d as f64 == 1.0 {
            Interval::point(q)
        } else {
            Interval::new(q.next_down(), q.next_up())
        }
    }

    /// Integer power `self^n`, `n >= 0`, by interval multiplication. Even powers
    /// are tightened to be non-negative (a standard interval-power refinement).
    pub fn powi(&self, n: u32) -> Interval {
        if n == 0 {
            return Interval::one();
        }
        // Tighter rule for even powers of an interval straddling zero.
        if n.is_multiple_of(2) && self.lo < 0.0 && self.hi > 0.0 {
            let a = self.lo.abs();
            let b = self.hi.abs();
            let m = a.max(b);
            // [0, m]^n
            let hi = pow_pos_up(m, n);
            return Interval { lo: 0.0, hi };
        }
        // General case: repeated multiplication (already outward-rounded).
        let mut acc = *self;
        for _ in 1..n {
            acc = acc * *self;
        }
        acc
    }
}

/// `x^n` for `x >= 0`, rounded up — helper for the even-power refinement.
#[inline]
fn pow_pos_up(x: f64, n: u32) -> f64 {
    let mut acc = 1.0_f64;
    for _ in 0..n {
        acc = up(acc * x);
    }
    acc
}

impl Add for Interval {
    type Output = Interval;
    #[inline]
    fn add(self, o: Interval) -> Interval {
        Interval {
            lo: down(self.lo + o.lo),
            hi: up(self.hi + o.hi),
        }
    }
}

impl Sub for Interval {
    type Output = Interval;
    #[inline]
    fn sub(self, o: Interval) -> Interval {
        Interval {
            lo: down(self.lo - o.hi),
            hi: up(self.hi - o.lo),
        }
    }
}

impl Neg for Interval {
    type Output = Interval;
    #[inline]
    fn neg(self) -> Interval {
        Interval {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
}

impl Mul for Interval {
    type Output = Interval;
    #[inline]
    fn mul(self, o: Interval) -> Interval {
        // Products of the four endpoint combinations; min/max gives the range,
        // then round outward.
        let p1 = self.lo * o.lo;
        let p2 = self.lo * o.hi;
        let p3 = self.hi * o.lo;
        let p4 = self.hi * o.hi;
        let lo = p1.min(p2).min(p3).min(p4);
        let hi = p1.max(p2).max(p3).max(p4);
        Interval {
            lo: down(lo),
            hi: up(hi),
        }
    }
}

impl fmt::Debug for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.6e}, {:.6e}]", self.lo, self.hi)
    }
}

impl crate::scalar::Scalar for Interval {
    #[inline]
    fn from_f64(v: f64) -> Self {
        Interval::point(v)
    }
    #[inline]
    fn one() -> Self {
        Interval::one()
    }
    #[inline]
    fn zero() -> Self {
        Interval::zero()
    }
    #[inline]
    fn scale_real(self, alpha: f64) -> Self {
        self.scale(alpha)
    }
    #[inline]
    fn div_u32(self, d: u32) -> Self {
        // Rigorous: enclose the true rational 1/d (outward-rounded when 1/d is
        // not exactly representable), then multiply.
        self * Interval::recip_u32(d)
    }
    #[inline]
    fn powi(self, n: u32) -> Self {
        Interval::powi(&self, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enclosure_add_sub() {
        let a = Interval::new(1.0, 2.0);
        let b = Interval::new(3.0, 4.0);
        let s = a + b;
        assert!(s.lo <= 4.0 && s.hi >= 6.0);
        let d = a - b;
        assert!(d.lo <= -3.0 && d.hi >= -1.0);
    }

    #[test]
    fn enclosure_mul() {
        let a = Interval::new(-2.0, 3.0);
        let b = Interval::new(-1.0, 4.0);
        let p = a * b;
        // true range of products is [-8, 12]
        assert!(p.lo <= -8.0 && p.hi >= 12.0);
    }

    #[test]
    fn even_power_nonnegative() {
        let a = Interval::new(-3.0, 2.0);
        let p = a.powi(2);
        assert!(p.lo >= 0.0);
        assert!(p.hi >= 9.0); // contains 9 = (-3)^2
    }

    #[test]
    fn outward_rounding_holds_for_tenths() {
        // 0.1 is not representable; repeated addition must stay enclosing.
        let tenth = Interval::point(0.1);
        let mut acc = Interval::zero();
        for _ in 0..10 {
            acc = acc + tenth;
        }
        assert!(acc.lo <= 1.0 && acc.hi >= 1.0, "0.1*10 must enclose 1.0");
    }
}
