//! The Van der Pol oscillator — the bundled example [`System`].
//!
//! ```text
//!     x0' = x1
//!     x1' = mu (1 - x0^2) x1 - x0
//! ```
//!
//! A classic stiff-ish nonlinear oscillator with a limit cycle. Written once,
//! generically over [`Scalar`], so it runs unchanged under RK4 (`f64`) and the
//! verified Taylor-Model integrator (interval / TM arithmetic).

use crate::scalar::Scalar;
use crate::system::System;

/// Van der Pol oscillator with parameter `mu`.
#[derive(Clone, Copy)]
pub struct VanDerPol {
    pub mu: f64,
}

impl VanDerPol {
    /// Construct with damping parameter `mu`.
    pub fn new(mu: f64) -> Self {
        VanDerPol { mu }
    }
}

impl Default for VanDerPol {
    fn default() -> Self {
        VanDerPol { mu: 1.0 }
    }
}

impl System<2> for VanDerPol {
    fn eval<A: Scalar>(&self, x: &[A; 2]) -> [A; 2] {
        let mu = A::from_f64(self.mu);
        let one = A::one();
        // f0 = x1
        let f0 = x[1].clone();
        // f1 = mu (1 - x0^2) x1 - x0
        let x0sq = x[0].clone() * x[0].clone();
        let f1 = mu * (one - x0sq) * x[1].clone() - x[0].clone();
        [f0, f1]
    }

    fn name(&self) -> &str {
        "Van der Pol"
    }
}
