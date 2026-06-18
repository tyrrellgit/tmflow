//! Verified Taylor-Model integrator (generic over the [`System`]).
//!
//! The state is a vector of [`TaylorModel`]s in the original initial-box
//! coordinates `s ∈ [-1, 1]^N`, so the curved reachable set is carried as a
//! curved polynomial (the anti-wrapping mechanism). Each step is fully
//! verified:
//!
//! 1. **A priori enclosure (Picard–Lindelöf).** Iterate the Picard operator
//!    `x0 + [0,h]·f(B) ⊆ B` to a self-map fixed point — proves existence,
//!    uniqueness, and an enclosure of `x(t)` on `[0, h]`.
//! 2. **Time-Taylor polynomial.** Coefficients `A[0..k]` via the generic
//!    [`time_series`] (automatic differentiation of the
//!    field in [`Jet`](crate::jet::Jet)+TM arithmetic) — no per-field recurrence.
//! 3. **Validated remainder (Lagrange form).** Bound the `(k+1)`-th time-Taylor
//!    coefficient over the a priori box `B` and multiply by `[0,h]^{k+1}`.
//! 4. **Compose & advance** at `t = h`, fold in the remainder, carry the result
//!    as TMs in the same `s`. Adaptive time bisection (up to `2^max_subdiv`
//!    sub-steps) engages when the enclosure fails to contract.
//!
//! Everything is generic: the only field-specific input is `System::eval`.

use crate::integrator::{bbox_from_intervals, BBox, Integrator};
use crate::interval::Interval;
use crate::jet::time_series;
use crate::system::System;
use crate::taylor_model::TaylorModel;

/// Verified TM integrator. Holds the system and the propagation order `k`.
pub struct TaylorModelIntegrator<'a, S, const N: usize> {
    sys: &'a S,
    /// Polynomial + time order `k`.
    pub order: u32,
    /// Maximum adaptive time-bisection depth (2^depth sub-steps).
    pub max_subdiv: u32,
    /// Picard inflation factor (> 1) and absolute pad for the a priori box.
    pub picard_inflate: f64,
    pub picard_abs_eps: f64,
    pub picard_max_iter: usize,
}

impl<'a, S, const N: usize> TaylorModelIntegrator<'a, S, N>
where
    S: System<N>,
{
    /// Build a verified TM integrator of order `k` with sensible defaults.
    pub fn new(sys: &'a S, order: u32) -> Self {
        TaylorModelIntegrator {
            sys,
            order,
            max_subdiv: 6,
            picard_inflate: 1.2,
            picard_abs_eps: 1e-6,
            picard_max_iter: 200,
        }
    }

    /// A priori enclosure of the flow on `[0, h]` by validated Picard iteration.
    /// Returns the contracted self-map image, or `None` if it fails to contract.
    fn apriori_enclosure(&self, x0: &[Interval; N], h: f64) -> Option<[Interval; N]> {
        let h_iv = Interval::new(0.0, h);
        let f0 = self.sys.eval(x0);
        let mut b: [Interval; N] = std::array::from_fn(|i| {
            (x0[i] + h_iv * f0[i]).inflate(self.picard_inflate, self.picard_abs_eps)
        });

        for _ in 0..self.picard_max_iter {
            let fb = self.sys.eval(&b);
            let bnew: [Interval; N] = std::array::from_fn(|i| x0[i] + h_iv * fb[i]);
            if (0..N).all(|i| b[i].contains_interval(&bnew[i])) {
                return Some(bnew);
            }
            b = std::array::from_fn(|i| {
                b[i].hull(&bnew[i]).inflate(self.picard_inflate, self.picard_abs_eps)
            });
        }
        None
    }

    /// Bound the `(k+1)`-th time-Taylor coefficient over the box `B` — the
    /// Lagrange remainder factor — using the generic time series in interval
    /// arithmetic seeded with the whole enclosure.
    fn coeff_k1_over_box(&self, b: &[Interval; N]) -> [Interval; N] {
        let k = self.order as usize;
        let series = time_series(self.sys, b, k + 1);
        std::array::from_fn(|i| series[i][k + 1])
    }

    /// One raw step over `[0, h]`. `None` if the Picard enclosure fails.
    fn step_raw(&self, state: &[TaylorModel; N], h: f64) -> Option<[TaylorModel; N]> {
        let k = self.order as usize;
        let x0box: [Interval; N] = std::array::from_fn(|i| state[i].bound());

        let benc = self.apriori_enclosure(&x0box, h)?;
        let ck1 = self.coeff_k1_over_box(&benc);

        // h^{k+1}, taken symmetric to be safe for sign.
        let mut hk1 = Interval::one();
        for _ in 0..(k + 1) {
            hk1 = hk1 * Interval::point(h);
        }
        let hk1_sym = Interval::new(-hk1.hi, hk1.hi);

        // Exact TM time-Taylor coefficients in s, to order k.
        let series = time_series(self.sys, state, k);

        // Evaluate the order-k time polynomial at t = h, per component.
        let mut next: [TaylorModel; N] = std::array::from_fn(|i| {
            let mut acc = TaylorModel::constant(Interval::zero(), self.order, N);
            let mut hp = Interval::one();
            for coeff in series[i].iter().take(k + 1) {
                acc = acc + coeff.scale(hp);
                hp = hp * Interval::point(h);
            }
            acc
        });

        // Fold the validated remainder in.
        for i in 0..N {
            next[i].rem = next[i].rem + (ck1[i] * hk1_sym);
        }
        Some(next)
    }
}

impl<'a, S, const N: usize> Integrator<N> for TaylorModelIntegrator<'a, S, N>
where
    S: System<N>,
{
    type State = [TaylorModel; N];

    fn label(&self) -> String {
        format!("TaylorModel(k={})", self.order)
    }

    fn init(&self, center: [f64; N], half: [f64; N]) -> Self::State {
        std::array::from_fn(|i| TaylorModel::variable(i, self.order, N, center[i], half[i]))
    }

    fn step(&self, state: &Self::State, h: f64) -> Self::State {
        // Adaptive time bisection: try the full step, then halve until the
        // Picard enclosure contracts at every sub-step.
        for depth in 0..=self.max_subdiv {
            let m = 1u32 << depth;
            let hsub = h / m as f64;
            let mut cur = state.clone();
            let mut ok = true;
            for _ in 0..m {
                match self.step_raw(&cur, hsub) {
                    Some(n) => cur = n,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                return cur;
            }
        }
        panic!(
            "TaylorModelIntegrator: a priori enclosure failed after {} sub-steps at h={h}; \
             reduce the step or the set radius.",
            1u32 << self.max_subdiv
        );
    }

    fn bbox(&self, state: &Self::State) -> BBox<N> {
        let iv: [Interval; N] = std::array::from_fn(|i| state[i].bound());
        bbox_from_intervals(&iv)
    }

    fn is_rigorous(&self) -> bool {
        true
    }
}

/// Trace the curved verified-set boundary of a 2-D Taylor-Model state by walking
/// the perimeter of the parameter square `s ∈ [-1, 1]²` and evaluating the
/// polynomial image. Returns `(per_segment_points, remainder_widths)` as a JSON
/// fragment ready to splice into [`Trajectory::to_json`].
///
/// `states` is the `states` field of a Taylor-Model [`Trajectory`]; `samples`
/// controls how finely each side of the square is sampled (e.g. 24).
///
/// [`Trajectory::to_json`]: crate::driver::Trajectory::to_json
/// [`Trajectory`]: crate::driver::Trajectory
pub fn boundary_json(states: &[[TaylorModel; 2]], samples: usize) -> String {
    // Perimeter of [-1,1]^2, counter-clockwise.
    let mut params: Vec<[f64; 2]> = Vec::new();
    let n = samples.max(2);
    let lin = |a: f64, b: f64, i: usize| a + (b - a) * (i as f64) / (n as f64);
    for i in 0..n {
        params.push([lin(-1.0, 1.0, i), -1.0]);
    }
    for i in 0..n {
        params.push([1.0, lin(-1.0, 1.0, i)]);
    }
    for i in 0..n {
        params.push([lin(1.0, -1.0, i), 1.0]);
    }
    for i in 0..n {
        params.push([-1.0, lin(1.0, -1.0, i)]);
    }
    params.push([-1.0, -1.0]); // close the loop

    let mut bnd = String::from("\"boundaries\":[");
    let mut rem = String::from("\"rem_w\":[");
    for (si, st) in states.iter().enumerate() {
        if si > 0 {
            bnd.push(',');
            rem.push(',');
        }
        bnd.push('[');
        for (pi, p) in params.iter().enumerate() {
            if pi > 0 {
                bnd.push(',');
            }
            let x = st[0].eval_poly_mid(p);
            let y = st[1].eval_poly_mid(p);
            bnd.push_str(&format!("[{x},{y}]"));
        }
        bnd.push(']');
        rem.push_str(&format!("[{},{}]", st[0].rem.width(), st[1].rem.width()));
    }
    bnd.push(']');
    rem.push(']');
    format!("{bnd},{rem}")
}
