//! Classical explicit Runge–Kutta (RK4) integrator — the fast, non-rigorous
//! reference. Propagates a single `f64` point (the initial-box center). Used as
//! the wall-time/quality benchmark against verified Taylor Models.

use crate::integrator::{BBox, Integrator};
use crate::system::System;

/// RK4 on a single point. Holds a reference to the system.
pub struct Rk4<'a, S, const N: usize> {
    sys: &'a S,
}

impl<'a, S, const N: usize> Rk4<'a, S, N>
where
    S: System<N>,
{
    /// Build an RK4 integrator for `sys`.
    pub fn new(sys: &'a S) -> Self {
        Rk4 { sys }
    }
}

impl<'a, S, const N: usize> Integrator<N> for Rk4<'a, S, N>
where
    S: System<N>,
{
    type State = [f64; N];

    fn label(&self) -> String {
        "RK4".to_string()
    }

    fn init(&self, center: [f64; N], _half: [f64; N]) -> Self::State {
        center
    }

    fn step(&self, x: &Self::State, h: f64) -> Self::State {
        let f = |s: &[f64; N]| self.sys.eval(s);
        let k1 = f(x);
        let x2: [f64; N] = std::array::from_fn(|i| x[i] + 0.5 * h * k1[i]);
        let k2 = f(&x2);
        let x3: [f64; N] = std::array::from_fn(|i| x[i] + 0.5 * h * k2[i]);
        let k3 = f(&x3);
        let x4: [f64; N] = std::array::from_fn(|i| x[i] + h * k3[i]);
        let k4 = f(&x4);
        std::array::from_fn(|i| x[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
    }

    fn bbox(&self, x: &Self::State) -> BBox<N> {
        std::array::from_fn(|i| (x[i], x[i]))
    }

    fn is_rigorous(&self) -> bool {
        false
    }
}
