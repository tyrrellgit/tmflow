//! [`Jet`] — truncated power series in time `t`, with [`Scalar`] coefficients.
//!
//! A `Jet<A>` represents `Σ_{j=0}^{order} c[j] · t^j`. Implementing [`Scalar`]
//! for `Jet<A>` (with the Cauchy product for multiplication) turns any
//! [`System`](crate::system::System) `eval` into an *automatic* time-Taylor
//! expander — no hand-derived recurrence per field.
//!
//! # The recurrence (`time_series`)
//!
//! For `x'(t) = f(x(t))` with `x(t) = Σ A[j] t^j`, matching powers of `t` gives
//!
//! ```text
//!     (j+1) A[j+1] = [ f(x) ]_j ,
//! ```
//!
//! i.e. the `(j+1)`-th solution coefficient is the `j`-th Taylor coefficient of
//! `f` evaluated on the series so far, divided by `j+1`. We build the solution
//! coefficients one order at a time: knowing `A[0..=j]`, evaluate `f` in `Jet`
//! arithmetic (which propagates all lower coefficients exactly) and read off its
//! `j`-th coefficient. Because each `eval` only *uses* coefficients up to the
//! current order, the bootstrap is well-defined.
//!
//! This is exact in the coefficient algebra `A`: with `A = TaylorModel` the
//! coefficients are exact TM functions of the initial box; the *time*
//! truncation error of stopping at `order` is bounded separately by the
//! validated remainder in the integrator.

use crate::scalar::Scalar;
use std::ops::{Add, Mul, Neg, Sub};

/// A truncated time-power-series `Σ c[j] t^j`, `j = 0..=order`, with coeffs in
/// the [`Scalar`] algebra `A`.
#[derive(Clone)]
pub struct Jet<A: Scalar> {
    /// Coefficients `c[0..=order]`; `c[j]` multiplies `t^j`.
    pub c: Vec<A>,
}

impl<A: Scalar> Jet<A> {
    /// Maximum time order retained.
    #[inline]
    pub fn order(&self) -> usize {
        self.c.len() - 1
    }

    /// A constant jet `c0 + 0·t + …` of the given order.
    pub fn constant(c0: A, order: usize) -> Self {
        let mut c = vec![A::zero(); order + 1];
        c[0] = c0;
        Jet { c }
    }

    /// A seed jet for a state variable: value `c0` at `t=0`, all higher
    /// coefficients zero (they are filled in by the recurrence).
    pub fn seed(c0: A, order: usize) -> Self {
        Jet::constant(c0, order)
    }

    /// The `j`-th coefficient (0 if out of range).
    #[inline]
    pub fn coeff(&self, j: usize) -> A {
        self.c.get(j).cloned().unwrap_or_else(A::zero)
    }
}

impl<A: Scalar> Add for Jet<A> {
    type Output = Jet<A>;
    fn add(self, o: Jet<A>) -> Jet<A> {
        let n = self.c.len().max(o.c.len());
        let mut c = vec![A::zero(); n];
        for (j, cj) in c.iter_mut().enumerate() {
            *cj = self.coeff(j) + o.coeff(j);
        }
        Jet { c }
    }
}

impl<A: Scalar> Sub for Jet<A> {
    type Output = Jet<A>;
    fn sub(self, o: Jet<A>) -> Jet<A> {
        let n = self.c.len().max(o.c.len());
        let mut c = vec![A::zero(); n];
        for (j, cj) in c.iter_mut().enumerate() {
            *cj = self.coeff(j) - o.coeff(j);
        }
        Jet { c }
    }
}

impl<A: Scalar> Neg for Jet<A> {
    type Output = Jet<A>;
    fn neg(self) -> Jet<A> {
        Jet {
            c: self.c.into_iter().map(|x| -x).collect(),
        }
    }
}

impl<A: Scalar> Mul for Jet<A> {
    type Output = Jet<A>;
    /// Cauchy product. A degree-0 jet acts as a true scalar and broadcasts to
    /// the other operand's order (so lifting a constant via `from_f64`/`one`
    /// does not collapse the series); otherwise the product is truncated to the
    /// common order.
    fn mul(self, o: Jet<A>) -> Jet<A> {
        // Broadcast pure constants instead of truncating to order 0.
        if self.order() == 0 {
            let s = self.coeff(0);
            return Jet {
                c: o.c.into_iter().map(|x| s.clone() * x).collect(),
            };
        }
        if o.order() == 0 {
            let s = o.coeff(0);
            return Jet {
                c: self.c.into_iter().map(|x| x * s.clone()).collect(),
            };
        }
        let order = self.order().min(o.order());
        let mut c = vec![A::zero(); order + 1];
        for i in 0..=order {
            for j in 0..=(order - i) {
                c[i + j] = c[i + j].clone() + self.coeff(i) * o.coeff(j);
            }
        }
        Jet { c }
    }
}

impl<A: Scalar> Scalar for Jet<A> {
    fn from_f64(v: f64) -> Self {
        // Order is unknown here; a length-1 constant jet. In practice `eval`
        // combines such constants with seeded jets, and `Add`/`Mul` broadcast
        // by padding, so a degree-0 constant lifts correctly.
        Jet {
            c: vec![A::from_f64(v)],
        }
    }
    fn one() -> Self {
        Jet { c: vec![A::one()] }
    }
    fn zero() -> Self {
        Jet { c: vec![A::zero()] }
    }
    fn scale_real(self, alpha: f64) -> Self {
        Jet {
            c: self.c.into_iter().map(|x| x.scale_real(alpha)).collect(),
        }
    }
    fn div_u32(self, d: u32) -> Self {
        Jet {
            c: self.c.into_iter().map(|x| x.div_u32(d)).collect(),
        }
    }
}

/// Compute the time-Taylor coefficients `A[0..=order]` of the solution of
/// `x' = f(x)` with `x(0) = x0`, in the coefficient algebra `A`.
///
/// Returns, per state component, the vector of coefficients `[A0, A1, …,
/// A_order]`. Works for *any* [`System`](crate::system::System) because it only
/// uses `eval` in [`Jet`] arithmetic. Outer index = component, inner = time
/// order.
pub fn time_series<S, const N: usize, A>(sys: &S, x0: &[A; N], order: usize) -> [Vec<A>; N]
where
    S: crate::system::System<N>,
    A: Scalar,
{
    // coeffs[i][j] = j-th time coefficient of component i, built incrementally.
    let mut coeffs: [Vec<A>; N] = std::array::from_fn(|_| Vec::with_capacity(order + 1));
    for i in 0..N {
        coeffs[i].push(x0[i].clone());
    }

    for j in 0..order {
        // Build jets carrying coefficients known so far (orders 0..=j), at jet
        // order j so the Cauchy products are exact up to t^j.
        let jets: [Jet<A>; N] = std::array::from_fn(|i| {
            let mut c = vec![A::zero(); j + 1];
            for (m, ci) in coeffs[i].iter().enumerate() {
                if m <= j {
                    c[m] = ci.clone();
                }
            }
            Jet { c }
        });

        let f = sys.eval(&jets);

        // (j+1) A[j+1] = [f(x)]_j  (rigorous division for verified backends).
        let d = (j as u32) + 1;
        for i in 0..N {
            let fj = f[i].coeff(j);
            coeffs[i].push(fj.div_u32(d));
        }
    }

    coeffs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::System;

    /// Trivial linear system x' = x → solution e^t, A[j] = 1/j!.
    struct Exp;
    impl System<1> for Exp {
        fn eval<A: Scalar>(&self, x: &[A; 1]) -> [A; 1] {
            [x[0].clone()]
        }
    }

    #[test]
    fn exp_series_is_reciprocal_factorials() {
        let s = time_series(&Exp, &[1.0_f64], 6);
        let mut fact = 1.0;
        for (j, &coeff) in s[0].iter().enumerate() {
            if j > 0 {
                fact *= j as f64;
            }
            assert!((coeff - 1.0 / fact).abs() < 1e-12, "coeff {j}");
        }
    }
}
