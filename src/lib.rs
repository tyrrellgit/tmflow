//! # tmflow
//!
//! **Modular ODE flow propagation with swappable integrators.** Write a vector
//! field *once*, then propagate it under different integration schemes — a fast
//! `f64` RK4 reference, or a **verified Taylor-Model** integrator that returns
//! mathematically guaranteed reachable-set enclosures (validated-numerics sense,
//! à la COSY / CAPD).
//!
//! ## Architecture
//!
//! Two small trait abstractions do the heavy lifting:
//!
//! * [`Scalar`] — a ring-like algebra (`+ - * neg`, constant lifting). A vector
//!   field is written generically over it, so the same code evaluates in `f64`,
//!   [`Interval`], [`TaylorModel`], or [`Jet`] arithmetic.
//! * [`System`] — an ODE field `x' = f(x)`, defined by one generic
//!   [`System::eval`].
//! * [`Integrator`] — a swappable stepping scheme. The verified
//!   [`TaylorModelIntegrator`] gets its time-Taylor expansion *automatically*
//!   from [`Jet`]-based differentiation of the field — no per-field recurrence.
//!
//! ```
//! use tmflow::prelude::*;
//!
//! let sys = VanDerPol::new(1.0);
//!
//! // Verified reachable set from an initial box, order k = 4.
//! let tm = TaylorModelIntegrator::new(&sys, 4);
//! let verified = propagate(&tm, [1.4, 0.0], [0.08, 0.08], 0.1, 20);
//! assert!(verified.rigorous);
//!
//! // Same field, fast non-rigorous RK4 reference (point = box center).
//! let rk = Rk4::new(&sys);
//! let approx = propagate(&rk, [1.4, 0.0], [0.08, 0.08], 0.1, 20);
//! assert!(!approx.rigorous);
//! ```
//!
//! [`Scalar`]: crate::scalar::Scalar
//! [`System`]: crate::system::System
//! [`Integrator`]: crate::integrator::Integrator
//! [`Interval`]: crate::interval::Interval
//! [`TaylorModel`]: crate::taylor_model::TaylorModel
//! [`Jet`]: crate::jet::Jet
//! [`TaylorModelIntegrator`]: crate::integrators::TaylorModelIntegrator
//!
//! ## New here? Read the guide
//!
//! The [`docs`] module is a prose-first walkthrough of the concept, the
//! introductory theory behind verified enclosures, the architecture, and a
//! hands-on tutorial. Start there if reachability / validated numerics is new to
//! you, then come back to the per-item API reference below.
//!
//! * [`docs::intro`] — why a *set* and not a *point*
//! * [`docs::theory`] — intervals, Taylor Models, and how a guarantee is possible
//! * [`docs::architecture`] — the four-trait tour
//! * [`docs::tutorial`] — hands-on, including bringing your own field
//! * [`docs::faq`] — design decisions and honest limitations

pub mod docs;

pub mod driver;
pub mod integrator;
pub mod integrators;
pub mod interval;
pub mod jet;
pub mod scalar;
pub mod system;
pub mod systems;
pub mod taylor_model;

pub use driver::{propagate, Trajectory};
pub use integrator::Integrator;
pub use interval::Interval;
pub use scalar::Scalar;
pub use system::System;
pub use taylor_model::TaylorModel;

/// Everything you need for typical use: traits, the driver, both integrators,
/// and the bundled Van der Pol system.
pub mod prelude {
    pub use crate::driver::{propagate, Trajectory};
    pub use crate::integrator::{bbox_measure, BBox, Integrator};
    pub use crate::integrators::{Rk4, TaylorModelIntegrator};
    pub use crate::interval::Interval;
    pub use crate::scalar::Scalar;
    pub use crate::system::System;
    pub use crate::systems::VanDerPol;
    pub use crate::taylor_model::TaylorModel;
}
