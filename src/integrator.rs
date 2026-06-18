//! The [`Integrator`] trait — a swappable time-stepping scheme.
//!
//! An integrator advances a *state* (whatever it propagates) by one step of
//! size `h`. Different integrators carry different state and offer different
//! guarantees, unified behind one trait:
//!
//! * [`Rk4`](crate::integrators::rk4::Rk4) — classical explicit RK4 on a single
//!   `f64` point. Fast, **not** rigorous; used as the benchmark/reference.
//! * [`TaylorModelIntegrator`](crate::integrators::taylor::TaylorModelIntegrator)
//!   — verified set propagation: the state is a Taylor Model in the initial-box
//!   coordinates, and every step yields a **guaranteed** enclosure.
//!
//! Both are constructed from the same [`System`], so you write a vector field
//! once and choose the integrator at the call site.

use crate::interval::Interval;
use crate::system::System;

/// A bounding box `[x_lo, x_hi] × …` in `f64` (outward bounds for verified
/// integrators). One `(lo, hi)` pair per dimension.
pub type BBox<const N: usize> = [(f64, f64); N];

/// A swappable ODE integrator for a fixed system of dimension `N`.
///
/// Implementors own a reference/handle to the [`System`] and define how a step
/// transforms their [`Integrator::State`]. The crate's
/// [`propagate`](crate::driver::propagate) driver works for any implementor.
pub trait Integrator<const N: usize> {
    /// The propagated state (e.g. an `f64` point, or a vector of Taylor Models).
    type State: Clone;

    /// Short label for reports/plots (e.g. `"RK4"`, `"TaylorModel(k=4)"`).
    fn label(&self) -> String;

    /// Build the initial state from an initial box: center `c` and half-widths
    /// `half` (so component `i` ranges over `[c_i - half_i, c_i + half_i]`).
    ///
    /// Point integrators (RK4) collapse the box to its center; set-valued
    /// integrators (Taylor Model) represent the whole box exactly.
    fn init(&self, center: [f64; N], half: [f64; N]) -> Self::State;

    /// Advance the state by one step of size `h`.
    fn step(&self, state: &Self::State, h: f64) -> Self::State;

    /// Verified (or, for point integrators, degenerate) bounding box of a state.
    fn bbox(&self, state: &Self::State) -> BBox<N>;

    /// Whether this integrator's `bbox` is a *rigorous* enclosure (true) or a
    /// non-guaranteed approximation (false). Used by reports to label results.
    fn is_rigorous(&self) -> bool;
}

/// Convenience: bounding-box "volume" (product of widths).
pub fn bbox_measure<const N: usize>(b: &BBox<N>) -> f64 {
    b.iter().map(|(lo, hi)| hi - lo).product()
}

/// Helper for verified integrators: convert per-component [`Interval`] bounds to
/// an f64 [`BBox`].
pub fn bbox_from_intervals<const N: usize>(iv: &[Interval; N]) -> BBox<N> {
    std::array::from_fn(|i| (iv[i].lo, iv[i].hi))
}

/// Marker that an integrator is tied to a particular system (documentation aid;
/// no methods). Most integrators simply hold `&S`.
pub trait ForSystem<S, const N: usize>
where
    S: System<N>,
{
}
