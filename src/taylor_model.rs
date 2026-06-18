//! Verified k-th order [`TaylorModel`]s over `n` variables, interval coeffs.
//!
//! A Taylor Model (TM) of a function `g` over the domain `D = [-1, 1]^n`
//! (normalized initial-box coordinates) is
//!
//! ```text
//!     g(s)  ∈  p(s) + I      for all s ∈ D,
//! ```
//!
//! where `p` is a multivariate polynomial of degree `<= k` with *interval*
//! coefficients, and `I` is an interval remainder bounding everything `p`
//! omits (truncation error + floating round-off, swept outward). Every
//! operation is rigorous: the represented set of functions always encloses the
//! true one.
//!
//! `TaylorModel` implements [`Scalar`], so a vector
//! field written generically (see [`crate::system::System`]) evaluates directly
//! in TM arithmetic — and, composed with [`Jet`](crate::jet::Jet), yields the
//! verified time-Taylor expansion the integrator needs, with no per-field code.
//!
//! Multiplication convolves coefficients; terms of degree `> k` are bounded
//! over `D` and swept into the remainder (rigorous truncation), and the
//! cross-terms `p·I` and `I·I` are propagated exactly as
//! `(p1 + I1)(p2 + I2) = p1 p2 + p1 I2 + I1 p2 + I1 I2`.

use std::collections::BTreeMap;
use std::ops::{Add, Mul, Neg, Sub};

use crate::interval::Interval;
use crate::scalar::Scalar;

/// Exponent vector (one entry per variable). `BTreeMap` key → deterministic
/// term ordering.
pub type Exp = Vec<u32>;

/// A scalar-valued verified Taylor Model. A state vector is an array of these,
/// one per component.
///
/// `order == 0 && nvar == 0` marks a *bare constant* produced by
/// [`Scalar::from_f64`]/[`Scalar::one`]/[`Scalar::zero`], which has no fixed
/// dimension yet; binary operations promote it to the dimension/order of the
/// other operand. This is what lets `A::from_f64(mu)` mix cleanly with seeded
/// TM variables inside a generic field.
#[derive(Clone)]
pub struct TaylorModel {
    /// Polynomial coefficients keyed by exponent vector (degree `<= order`).
    pub poly: BTreeMap<Exp, Interval>,
    /// Interval remainder bounding all omitted contributions.
    pub rem: Interval,
    /// Maximum polynomial degree `k`.
    pub order: u32,
    /// Number of variables `n`.
    pub nvar: usize,
}

impl TaylorModel {
    /// Constant TM `c` (zero remainder) of explicit order/nvar.
    pub fn constant(c: Interval, order: u32, nvar: usize) -> TaylorModel {
        let mut poly = BTreeMap::new();
        if !c.is_zero() {
            poly.insert(vec![0u32; nvar], c);
        }
        TaylorModel {
            poly,
            rem: Interval::zero(),
            order,
            nvar,
        }
    }

    /// A *bare constant* (dimension-agnostic) — used by the `Scalar` impl.
    fn bare(c: Interval) -> TaylorModel {
        let mut poly = BTreeMap::new();
        if !c.is_zero() {
            poly.insert(Vec::new(), c);
        }
        TaylorModel {
            poly,
            rem: Interval::zero(),
            order: 0,
            nvar: 0,
        }
    }

    /// True if this is a bare (dimension-agnostic) constant.
    #[inline]
    fn is_bare(&self) -> bool {
        self.nvar == 0
    }

    /// TM for the physical coordinate `x_i = center + half * s_i`,
    /// `s_i ∈ [-1, 1]`. The result is a TM in the normalized variables `s`.
    pub fn variable(i: usize, order: u32, nvar: usize, center: f64, half: f64) -> TaylorModel {
        let mut poly = BTreeMap::new();
        if center != 0.0 {
            poly.insert(vec![0u32; nvar], Interval::point(center));
        }
        let mut e = vec![0u32; nvar];
        e[i] = 1;
        poly.insert(e, Interval::point(half));
        TaylorModel {
            poly,
            rem: Interval::zero(),
            order,
            nvar,
        }
    }

    /// Re-key a bare constant to `(order, nvar)`; identity for non-bare TMs.
    fn promote(&self, order: u32, nvar: usize) -> TaylorModel {
        if !self.is_bare() {
            return self.clone();
        }
        let c = self.poly.get(&Vec::new()).copied().unwrap_or_else(Interval::zero);
        TaylorModel::constant(c, order, nvar)
    }

    /// Pick the dimensioned `(order, nvar)` from a pair (the non-bare one).
    fn dims(a: &TaylorModel, b: &TaylorModel) -> (u32, usize) {
        if !a.is_bare() {
            (a.order, a.nvar)
        } else {
            (b.order, b.nvar)
        }
    }

    /// Drop exactly-zero coefficients.
    fn clean(mut self) -> TaylorModel {
        self.poly.retain(|_, c| !c.is_zero());
        self
    }

    /// Multiply by an interval scalar (coefficients and remainder scale).
    pub fn scale(&self, alpha: Interval) -> TaylorModel {
        let poly = self
            .poly
            .iter()
            .map(|(e, c)| (e.clone(), *c * alpha))
            .collect();
        TaylorModel {
            poly,
            rem: self.rem * alpha,
            order: self.order,
            nvar: self.nvar,
        }
        .clean()
    }

    /// Enclose `Σ c_e s^e` over `s ∈ [-1, 1]^n`. Each `|s_i| <= 1` ⇒
    /// `|s^e| <= 1`, so degree-`>=1` monomials range over `c·[-1,1]`; the
    /// degree-0 term contributes its constant.
    fn poly_bound_over_domain(poly: &BTreeMap<Exp, Interval>) -> Interval {
        let mut total = Interval::zero();
        let unit = Interval::unit();
        for (e, c) in poly {
            let deg: u32 = e.iter().sum();
            if deg == 0 {
                total = total + *c;
            } else {
                total = total + (*c * unit);
            }
        }
        total
    }

    /// Verified interval enclosure of the whole TM over `[-1, 1]^n`: `p(D) + I`.
    pub fn bound(&self) -> Interval {
        Self::poly_bound_over_domain(&self.poly) + self.rem
    }

    /// The constant (degree-0) coefficient.
    pub fn const_part(&self) -> Interval {
        let z = vec![0u32; self.nvar];
        self.poly.get(&z).copied().unwrap_or_else(Interval::zero)
    }

    /// Evaluate poly + remainder at a concrete normalized point `s` (as an
    /// interval). Used by the MC containment check (pointwise TM enclosure).
    pub fn eval_at(&self, s: &[f64]) -> Interval {
        self.eval_terms(|i| Interval::point(s[i])) + self.rem
    }

    /// Evaluate the polynomial part only at a concrete point, midpoint f64 —
    /// for drawing the curved image boundary.
    pub fn eval_poly_mid(&self, s: &[f64]) -> f64 {
        (self.eval_terms(|i| Interval::point(s[i]))).mid()
    }

    /// Rigorous enclosure of poly + remainder over an interval box. Used by the
    /// soundness tests.
    pub fn bound_over(&self, box_iv: &[Interval]) -> Interval {
        self.eval_terms(|i| box_iv[i]) + self.rem
    }

    /// Shared monomial evaluation: `Σ c_e Π var(i)^{e_i}`.
    fn eval_terms<F: Fn(usize) -> Interval>(&self, var: F) -> Interval {
        let mut val = Interval::zero();
        for (e, c) in &self.poly {
            let mut term = *c;
            for (i, &p) in e.iter().enumerate() {
                if p > 0 {
                    term = term * var(i).powi(p);
                }
            }
            val = val + term;
        }
        val
    }
}

// --- operator-overload impls (consume self), used by the `Scalar` blanket ---

impl Add for TaylorModel {
    type Output = TaylorModel;
    fn add(self, other: TaylorModel) -> TaylorModel {
        let (order, nvar) = TaylorModel::dims(&self, &other);
        let a = self.promote(order, nvar);
        let b = other.promote(order, nvar);
        let mut poly = a.poly;
        for (e, c) in b.poly {
            poly.entry(e).and_modify(|x| *x = *x + c).or_insert(c);
        }
        TaylorModel {
            poly,
            rem: a.rem + b.rem,
            order,
            nvar,
        }
        .clean()
    }
}

impl Sub for TaylorModel {
    type Output = TaylorModel;
    fn sub(self, other: TaylorModel) -> TaylorModel {
        let (order, nvar) = TaylorModel::dims(&self, &other);
        let a = self.promote(order, nvar);
        let b = other.promote(order, nvar);
        let mut poly = a.poly;
        for (e, c) in b.poly {
            poly.entry(e).and_modify(|x| *x = *x - c).or_insert(-c);
        }
        TaylorModel {
            poly,
            rem: a.rem - b.rem,
            order,
            nvar,
        }
        .clean()
    }
}

impl Neg for TaylorModel {
    type Output = TaylorModel;
    fn neg(self) -> TaylorModel {
        let poly = self.poly.into_iter().map(|(e, c)| (e, -c)).collect();
        TaylorModel {
            poly,
            rem: -self.rem,
            order: self.order,
            nvar: self.nvar,
        }
    }
}

impl Mul for TaylorModel {
    type Output = TaylorModel;
    /// Degree-`k`-truncated product; excess swept into the remainder.
    fn mul(self, other: TaylorModel) -> TaylorModel {
        let (k, nvar) = TaylorModel::dims(&self, &other);
        let a = self.promote(k, nvar);
        let b = other.promote(k, nvar);
        let unit = Interval::unit();

        let mut new_poly: BTreeMap<Exp, Interval> = BTreeMap::new();
        let mut over_deg = Interval::zero();

        for (e1, c1) in &a.poly {
            for (e2, c2) in &b.poly {
                let e: Exp = e1.iter().zip(e2).map(|(x, y)| x + y).collect();
                let prod = *c1 * *c2;
                let deg: u32 = e.iter().sum();
                if deg <= k {
                    new_poly.entry(e).and_modify(|x| *x = *x + prod).or_insert(prod);
                } else {
                    over_deg = over_deg + (prod * unit);
                }
            }
        }

        // (p1 + I1)(p2 + I2) = p1 p2 + p1 I2 + I1 p2 + I1 I2
        let b1 = Self::poly_bound_over_domain(&a.poly);
        let b2 = Self::poly_bound_over_domain(&b.poly);
        let rem = over_deg + (b1 * b.rem) + (a.rem * b2) + (a.rem * b.rem);

        TaylorModel {
            poly: new_poly,
            rem,
            order: k,
            nvar,
        }
        .clean()
    }
}

impl Scalar for TaylorModel {
    #[inline]
    fn from_f64(v: f64) -> Self {
        TaylorModel::bare(Interval::point(v))
    }
    #[inline]
    fn one() -> Self {
        TaylorModel::bare(Interval::one())
    }
    #[inline]
    fn zero() -> Self {
        TaylorModel::bare(Interval::zero())
    }
    #[inline]
    fn scale_real(self, alpha: f64) -> Self {
        self.scale(Interval::point(alpha))
    }
    #[inline]
    fn div_u32(self, d: u32) -> Self {
        self.scale(Interval::recip_u32(d))
    }
}
