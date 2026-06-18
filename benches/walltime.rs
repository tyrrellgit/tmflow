//! Wall-time benchmark: verified Taylor-Model integrator vs RK4 reference, on
//! the standard Van der Pol scenario, using `criterion`.
//!
//! Run with:  cargo bench
//!
//! Reports the per-run wall time for:
//!   * RK4 (single point, non-rigorous) — the cheap baseline,
//!   * TaylorModel(k) for k = 2..5 (verified reachable-set enclosures).
//!
//! The two are not the same task (a point trajectory vs a guaranteed set
//! enclosure); the comparison shows the cost of rigour at each order.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

use tmflow::integrators::{Rk4, TaylorModelIntegrator};
use tmflow::prelude::*;

const CENTER: [f64; 2] = [1.4, 0.0];
const HALF: [f64; 2] = [0.08, 0.08];
const H: f64 = 0.1;
const N_STEPS: usize = 20;

fn bench_integrators(c: &mut Criterion) {
    let sys = VanDerPol::new(1.0);

    let mut g = c.benchmark_group("vanderpol_20steps");

    // RK4 reference (point propagation).
    g.bench_function("RK4", |b| {
        let rk = Rk4::new(&sys);
        b.iter(|| {
            let t = propagate(&rk, black_box(CENTER), black_box(HALF), H, N_STEPS);
            black_box(t.measures.len())
        });
    });

    // Verified Taylor-Model integrator across orders.
    for k in [2u32, 3, 4, 5] {
        g.bench_with_input(BenchmarkId::new("TaylorModel", k), &k, |b, &k| {
            let tm = TaylorModelIntegrator::new(&sys, k);
            b.iter(|| {
                let t = propagate(&tm, black_box(CENTER), black_box(HALF), H, N_STEPS);
                black_box(t.measures.len())
            });
        });
    }

    g.finish();
}

criterion_group!(benches, bench_integrators);
criterion_main!(benches);
