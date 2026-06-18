//! Generic multi-step propagation driver, integrator-agnostic.
//!
//! [`propagate`] loops any [`Integrator`] over a fixed step `h` and records the
//! per-step bounding boxes (and their measures). It is intentionally minimal —
//! the integrator owns all the scheme-specific logic.

use crate::integrator::{bbox_measure, BBox, Integrator};

/// Result of a propagation run.
pub struct Trajectory<const N: usize, St> {
    /// State at each recorded time (index 0 = initial).
    pub states: Vec<St>,
    /// Bounding box at each step.
    pub boxes: Vec<BBox<N>>,
    /// Bounding-box measure (product of widths) at each step.
    pub measures: Vec<f64>,
    pub h: f64,
    pub n_steps: usize,
    /// Integrator label and whether its boxes are rigorous enclosures.
    pub label: String,
    pub rigorous: bool,
}

/// Propagate `integrator` from the initial box for `n_steps` steps of size `h`.
pub fn propagate<const N: usize, I>(
    integrator: &I,
    center: [f64; N],
    half: [f64; N],
    h: f64,
    n_steps: usize,
) -> Trajectory<N, I::State>
where
    I: Integrator<N>,
{
    let s0 = integrator.init(center, half);
    let b0 = integrator.bbox(&s0);
    let mut states = vec![s0];
    let mut boxes = vec![b0];
    let mut measures = vec![bbox_measure(&b0)];

    for _ in 0..n_steps {
        let next = integrator.step(states.last().unwrap(), h);
        let bb = integrator.bbox(&next);
        boxes.push(bb);
        measures.push(bbox_measure(&bb));
        states.push(next);
    }

    Trajectory {
        states,
        boxes,
        measures,
        h,
        n_steps,
        label: integrator.label(),
        rigorous: integrator.is_rigorous(),
    }
}

impl<const N: usize, St> Trajectory<N, St> {
    /// Serialize the per-step bounding boxes and measures to a small JSON string
    /// (zero-dependency, hand-rolled). Intended for the optional plotting helper
    /// in `scripts/plot.py`; the crate itself stays plotting-free by design.
    ///
    /// `extra` lets a caller splice in scheme-specific fields (e.g. the curved
    /// set boundary from a Taylor-Model run). Pass `""` for none.
    pub fn to_json(&self, extra: &str) -> String {
        let mut s = String::from("{");
        s.push_str(&format!("\"label\":{:?},", self.label));
        s.push_str(&format!("\"rigorous\":{},", self.rigorous));
        s.push_str(&format!("\"h\":{},\"n_steps\":{},", self.h, self.n_steps));

        s.push_str("\"boxes\":[");
        for (i, b) in self.boxes.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push('[');
            for (j, (lo, hi)) in b.iter().enumerate() {
                if j > 0 {
                    s.push(',');
                }
                s.push_str(&format!("{lo},{hi}"));
            }
            s.push(']');
        }
        s.push_str("],");

        s.push_str("\"measures\":[");
        for (i, m) in self.measures.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("{m}"));
        }
        s.push(']');

        if !extra.is_empty() {
            s.push(',');
            s.push_str(extra);
        }
        s.push('}');
        s
    }
}
