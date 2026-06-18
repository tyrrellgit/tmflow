//! End-to-end verified-enclosure validation of the Taylor-Model integrator.
//!
//! Rigorous property: every true Van der Pol trajectory started in the initial
//! box must lie inside the verified TM enclosure at EVERY step (target: 100%).
//! We compare against many fine-RK4 reference trajectories, checking both the
//! pointwise TM enclosure (evaluate the TM at the sample's own normalized
//! coordinate) and the axis-aligned bounding box.

use tmflow::integrators::TaylorModelIntegrator;
use tmflow::prelude::*;
use tmflow::system::System;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1))
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * u
    }
}

/// Fine fixed-step RK4 reference (10 sub-steps per macro step) in plain f64.
fn rk4_ref(sys: &VanDerPol, mut x: [f64; 2], h: f64, sub: usize) -> [f64; 2] {
    let dt = h / sub as f64;
    for _ in 0..sub {
        let k1 = sys.eval(&x);
        let x2 = [x[0] + 0.5 * dt * k1[0], x[1] + 0.5 * dt * k1[1]];
        let k2 = sys.eval(&x2);
        let x3 = [x[0] + 0.5 * dt * k2[0], x[1] + 0.5 * dt * k2[1]];
        let k3 = sys.eval(&x3);
        let x4 = [x[0] + dt * k3[0], x[1] + dt * k3[1]];
        let k4 = sys.eval(&x4);
        x = [
            x[0] + dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
            x[1] + dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        ];
    }
    x
}

fn min_containment(order: u32, n_mc: usize) -> (f64, f64, f64) {
    let sys = VanDerPol::new(1.0);
    let center = [1.4, 0.0];
    let half = [0.08, 0.08];
    let h = 0.1;
    let n_steps = 20;

    let tm = TaylorModelIntegrator::new(&sys, order);
    let traj = propagate(&tm, center, half, h, n_steps);

    let mut rng = Lcg::new(0);
    let mut inside_tm = vec![0usize; n_steps + 1];
    let mut inside_bb = vec![0usize; n_steps + 1];

    for _ in 0..n_mc {
        let s = [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)];
        let mut state = [center[0] + half[0] * s[0], center[1] + half[1] * s[1]];
        for st in 0..=n_steps {
            if st > 0 {
                state = rk4_ref(&sys, state, h, 10);
            }
            // bbox containment
            let bb = &traj.boxes[st];
            if bb[0].0 <= state[0]
                && state[0] <= bb[0].1
                && bb[1].0 <= state[1]
                && state[1] <= bb[1].1
            {
                inside_bb[st] += 1;
            }
            // pointwise TM containment
            let tms = &traj.states[st];
            let vx = tms[0].eval_at(&s);
            let vy = tms[1].eval_at(&s);
            if vx.lo <= state[0] && state[0] <= vx.hi && vy.lo <= state[1] && state[1] <= vy.hi {
                inside_tm[st] += 1;
            }
        }
    }

    let min_tm = inside_tm
        .iter()
        .map(|&c| 100.0 * c as f64 / n_mc as f64)
        .fold(f64::INFINITY, f64::min);
    let min_bb = inside_bb
        .iter()
        .map(|&c| 100.0 * c as f64 / n_mc as f64)
        .fold(f64::INFINITY, f64::min);
    let final_area = *traj.measures.last().unwrap();
    (min_tm, min_bb, final_area)
}

#[test]
fn full_containment_all_orders() {
    let mut areas = Vec::new();
    for k in [2u32, 3, 4, 5] {
        let (min_tm, min_bb, area) = min_containment(k, 1500);
        assert_eq!(min_tm, 100.0, "k={k}: TM pointwise containment < 100%");
        assert_eq!(min_bb, 100.0, "k={k}: bbox containment < 100%");
        areas.push((k, area));
    }
    // Tightness must improve with order.
    for w in areas.windows(2) {
        assert!(
            w[1].1 <= w[0].1 * 1.001,
            "area did not shrink: k={} ({:.4e}) -> k={} ({:.4e})",
            w[0].0,
            w[0].1,
            w[1].0,
            w[1].1
        );
    }
}
