//! Integrator implementations. Each implements
//! [`Integrator`](crate::integrator::Integrator) and is interchangeable.

pub mod rk4;
pub mod taylor;

pub use rk4::Rk4;
pub use taylor::{boundary_json, TaylorModelIntegrator};
