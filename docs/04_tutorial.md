# 4. Tutorial — hands on

A practical walkthrough, from "propagate a box" to "bring your own ODE." Every
code block here is compiled and run as part of the test suite, so it is known to
work against the current API.

## 4.1 Propagate a verified set

Take the bundled Van der Pol oscillator, start from a small box, and propagate a
*verified* set with a Taylor-Model integrator of order 4:

```
use tmflow::prelude::*;

let sys = VanDerPol::new(1.0);                 // μ = 1
let tm = TaylorModelIntegrator::new(&sys, 4);  // order k = 4

// initial box: center (1.4, 0), half-widths ±0.08, step h = 0.1, 20 steps.
let traj = propagate(&tm, [1.4, 0.0], [0.08, 0.08], 0.1, 20);

assert!(traj.rigorous);                 // this enclosure is guaranteed
assert_eq!(traj.boxes.len(), 21);       // initial + 20 steps
let area = traj.measures.last().unwrap();
println!("final verified box area = {area:.4e}");
```

`traj.boxes[i]` is the rigorous bounding box at step `i` (one `(lo, hi)` pair per
dimension); `traj.measures[i]` is its area.

## 4.2 Compare against the fast point reference

The same field, integrated with RK4, propagates only the box *center* as a single
point — fast, but with no guarantee:

```
use tmflow::prelude::*;

let sys = VanDerPol::new(1.0);
let rk = Rk4::new(&sys);
let traj = propagate(&rk, [1.4, 0.0], [0.08, 0.08], 0.1, 20);

assert!(!traj.rigorous);   // RK4 gives an approximation, not an enclosure
// Its "box" is degenerate (zero width) — it is really a point.
let (lo, hi) = traj.boxes.last().unwrap()[0];
assert!((hi - lo).abs() < 1e-9);
```

The contrast is the whole point of the library: RK4 answers "where does one
trajectory go?" in microseconds; the Taylor-Model integrator answers "where can
*every* trajectory from the box go?" with a proof, in milliseconds.

## 4.3 Tighten the enclosure by raising the order

Higher Taylor-Model order means a tighter (still rigorous) enclosure. The
verified box area shrinks monotonically with `k`:

```
use tmflow::prelude::*;

let sys = VanDerPol::new(1.0);
let mut prev = f64::INFINITY;
for k in 2..=5 {
    let tm = TaylorModelIntegrator::new(&sys, k);
    let traj = propagate(&tm, [1.4, 0.0], [0.08, 0.08], 0.1, 20);
    let area = *traj.measures.last().unwrap();
    assert!(area <= prev, "order {k} should not be looser than {}", k - 1);
    prev = area;
}
```

(For reference, the final areas are roughly 2.75, 0.35, 0.24, 0.23 for
k = 2, 3, 4, 5 — diminishing returns, as expected once truncation error is below
the wrapping effect.)

## 4.4 Bring your own field

This is where the trait design pays off. Implement [`System`](crate::system::System)
*once*, generically over [`Scalar`](crate::scalar::Scalar), and every integrator
works on it immediately. Here is the Brusselator:

```
use tmflow::prelude::*;

struct Brusselator { a: f64, b: f64 }

impl System<2> for Brusselator {
    fn eval<A: Scalar>(&self, x: &[A; 2]) -> [A; 2] {
        let a = A::from_f64(self.a);
        // x0^2 * x1
        let x2y = x[0].clone() * x[0].clone() * x[1].clone();
        [
            a + x2y.clone() - x[0].clone().scale_real(self.b + 1.0),
            x[0].clone().scale_real(self.b) - x2y,
        ]
    }
    fn name(&self) -> &str { "Brusselator" }
}

let sys = Brusselator { a: 1.0, b: 3.0 };

// Verified set propagation — no per-field recurrence, it just works.
let tm = TaylorModelIntegrator::new(&sys, 3);
let verified = propagate(&tm, [1.0, 1.0], [0.02, 0.02], 0.05, 5);
assert!(verified.rigorous);

// …and the same field under RK4, for free.
let rk = Rk4::new(&sys);
let approx = propagate(&rk, [1.0, 1.0], [0.02, 0.02], 0.05, 5);
assert!(!approx.rigorous);
```

A few rules of thumb when writing `eval`:

- Use `A::from_f64(c)` to lift a constant, or `x.scale_real(c)` to multiply by
  one. Do **not** reach for `f64` literals mid-expression — the whole point is
  that the body is generic over `A`.
- `Scalar` is `Clone` but not `Copy` (a `TaylorModel` owns a polynomial map), so
  clone operands you reuse, as above.
- Stick to `+ - *`, `powi`, `scale_real`, and `div_u32` — the operations the
  verified backends can bound rigorously. (Transcendental functions are a planned
  extension; see the [FAQ](../faq/index.html).)

## 4.5 Visualize it

The crate is plotting-free, but the `vanderpol` example serializes a run to JSON
and a small Python helper renders the figure:

```sh
cargo run --release --example vanderpol   # writes out/vanderpol_tm.json
python3 scripts/plot.py                    # writes out/vanderpol_tm.png
```

You get the curved verified set sweeping through the phase plane (coloured by
time) on the left, and the enclosure's tightness — bounding-box area and
remainder widths — on the right. That picture is the fastest way to build
intuition for everything in chapter 2.

## 4.6 Where to go next

- Browse the API: [`Scalar`](crate::scalar::Scalar),
  [`System`](crate::system::System),
  [`Integrator`](crate::integrator::Integrator),
  [`TaylorModel`](crate::taylor_model::TaylorModel),
  [`propagate`](crate::driver::propagate).
- Read [`faq`](../faq/index.html) for the design rationale and current limits.
