//! The [`System`] trait — an autonomous ODE vector field, written once.
//!
//! A system of dimension `N` is `x' = f(x)`, where `f: R^N -> R^N`. The field is
//! defined **generically over the [`Scalar`] algebra**, so the very same code
//! evaluates under:
//!
//! * `f64` for the RK4 reference integrator and Monte-Carlo trajectories,
//! * [`Interval`](crate::interval::Interval) for rigorous a priori enclosures,
//! * [`TaylorModel`](crate::taylor_model::TaylorModel) for verified set
//!   propagation,
//! * [`Jet`](crate::jet::Jet) for automatic time-Taylor expansion.
//!
//! This is what lets the Taylor-Model integrator work for *any* polynomial field
//! without a hand-derived series recurrence: it simply evaluates `eval` in
//! [`Jet`] arithmetic.
//!
//! # Example
//! ```
//! use tmflow::{System, Scalar};
//!
//! /// The unforced Van der Pol oscillator: x0' = x1, x1' = mu(1-x0^2)x1 - x0.
//! struct VanDerPol { mu: f64 }
//!
//! impl System<2> for VanDerPol {
//!     fn eval<A: Scalar>(&self, x: &[A; 2]) -> [A; 2] {
//!         let mu = A::from_f64(self.mu);
//!         let one = A::one();
//!         let f0 = x[1].clone();
//!         let f1 = mu * (one - x[0].clone() * x[0].clone()) * x[1].clone()
//!             - x[0].clone();
//!         [f0, f1]
//!     }
//!     fn name(&self) -> &str { "Van der Pol" }
//! }
//! ```

use crate::scalar::Scalar;

/// An autonomous ODE vector field `x' = f(x)` on `R^N`.
///
/// Implement [`System::eval`] generically over [`Scalar`]; everything else in
/// the crate (integrators, enclosures, Taylor expansion) is derived from it.
pub trait System<const N: usize> {
    /// Evaluate the field `f(x)` in any [`Scalar`] arithmetic.
    fn eval<A: Scalar>(&self, x: &[A; N]) -> [A; N];

    /// Human-readable name (for reports and plots). Defaults to `"system"`.
    fn name(&self) -> &str {
        "system"
    }

    /// Dimension `N` (convenience).
    fn dim(&self) -> usize {
        N
    }
}
