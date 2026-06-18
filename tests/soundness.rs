//! Soundness of the Taylor-Model algebra and the generic time-series.
//!
//! Property: a TM modelling `g` over `[-1,1]^n` must ENCLOSE the true value of
//! `g` everywhere on the domain. We evaluate the TM over a tiny interval box
//! around random points (genuine non-degenerate intervals, avoiding ULP
//! coin-flips on exactly-representable polynomials) and check the true value
//! lies inside. An unsound TM (missing truncation, inward rounding) is caught.

use tmflow::interval::Interval;
use tmflow::scalar::Scalar;
use tmflow::taylor_model::TaylorModel;

/// Deterministic LCG (reproducible, no RNG dependency).
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

fn var(i: usize, order: u32, nvar: usize, c: f64, h: f64) -> TaylorModel {
    TaylorModel::variable(i, order, nvar, c, h)
}

fn check<F>(name: &str, tm: &TaylorModel, f_real: F, c: &[f64], h: &[f64], n: usize, seed: u64)
where
    F: Fn(&[f64]) -> f64,
{
    let mut rng = Lcg::new(seed);
    let nvar = c.len();
    let eps = 1e-6;
    let mut fails = 0usize;
    for _ in 0..n {
        let s: Vec<f64> = (0..nvar)
            .map(|_| rng.uniform(-1.0 + eps, 1.0 - eps))
            .collect();
        let xphys: Vec<f64> = (0..nvar).map(|i| c[i] + h[i] * s[i]).collect();
        let truth = f_real(&xphys);
        let nb: Vec<Interval> = s.iter().map(|&v| Interval::new(v - eps, v + eps)).collect();
        let tmb = tm.bound_over(&nb);
        if !(tmb.lo <= truth && truth <= tmb.hi) {
            fails += 1;
        }
    }
    assert_eq!(fails, 0, "{name}: {fails}/{n} soundness failures");
}

#[test]
fn univariate_powers_and_field() {
    let k = 4;
    let x01 = || var(0, k, 1, 0.5, 0.5); // x in [0,1]
    check("x^5 [0,1]", &x01().powi(5), |v| v[0].powi(5), &[0.5], &[0.5], 4000, 1);
    check("x^6 [0,1]", &x01().powi(6), |v| v[0].powi(6), &[0.5], &[0.5], 4000, 2);
    check("x^8 [0,1]", &x01().powi(8), |v| v[0].powi(8), &[0.5], &[0.5], 4000, 3);
    check(
        "(x^3)(x^3) [0,1]",
        &(x01().powi(3) * x01().powi(3)),
        |v| v[0].powi(6),
        &[0.5],
        &[0.5],
        4000,
        4,
    );
    let one = || TaylorModel::constant(Interval::one(), k, 1);
    check(
        "vdp f1 (1-x^2)x-x [0,1]",
        &((one() - x01().powi(2)) * x01() - x01()),
        |v| (1.0 - v[0].powi(2)) * v[0] - v[0],
        &[0.5],
        &[0.5],
        4000,
        5,
    );

    let x22 = || var(0, k, 1, 0.0, 2.0); // x in [-2,2]
    check("x^6 [-2,2]", &x22().powi(6), |v| v[0].powi(6), &[0.0], &[2.0], 4000, 6);
    check("x^7 [-2,2]", &x22().powi(7), |v| v[0].powi(7), &[0.0], &[2.0], 4000, 7);
    let one2 = || TaylorModel::constant(Interval::one(), k, 1);
    check(
        "(1-x^2)x [-2,2]",
        &((one2() - x22().powi(2)) * x22()),
        |v| (1.0 - v[0].powi(2)) * v[0],
        &[0.0],
        &[2.0],
        4000,
        8,
    );
}

#[test]
fn bivariate_field_and_products() {
    let k = 4;
    let (cx, hx, cy, hy) = (1.5, 0.5, 0.0, 1.0);
    let cc = [cx, cy];
    let hh = [hx, hy];
    let xv = || TaylorModel::variable(0, k, 2, cx, hx);
    let yv = || TaylorModel::variable(1, k, 2, cy, hy);
    let one = || TaylorModel::constant(Interval::one(), k, 2);

    check(
        "vdp f1 (1-x^2)y-x",
        &((one() - xv().powi(2)) * yv() - xv()),
        |v| (1.0 - v[0].powi(2)) * v[1] - v[0],
        &cc,
        &hh,
        4000,
        11,
    );
    check(
        "x^3 y^3 (deg6 trunc)",
        &(xv().powi(3) * yv().powi(3)),
        |v| v[0].powi(3) * v[1].powi(3),
        &cc,
        &hh,
        4000,
        12,
    );
    check(
        "(x y)^4",
        &(xv() * yv()).powi(4),
        |v| (v[0] * v[1]).powi(4),
        &cc,
        &hh,
        4000,
        13,
    );
    check(
        "x^2y^2 + xy^3 + y^4",
        &(xv().powi(2) * yv().powi(2) + xv() * yv().powi(3) + yv().powi(4)),
        |v| v[0].powi(2) * v[1].powi(2) + v[0] * v[1].powi(3) + v[1].powi(4),
        &cc,
        &hh,
        4000,
        14,
    );
}
