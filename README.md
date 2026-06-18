# tmflow

**Modular ODE flow propagation with swappable integrators** — write a vector field
*once*, then propagate it under different integration schemes: a fast `f64`
Runge–Kutta reference, or a **verified Taylor-Model** integrator that returns
mathematically guaranteed reachable-set enclosures (validated-numerics sense, in
the spirit of COSY-Infinity / CAPD).

- **Rigorous.** The Taylor-Model backend uses outward-rounded interval arithmetic
  everywhere, an a-priori Picard enclosure, and a Lagrange remainder bound. Every
  returned box is a guaranteed over-approximation of the true reachable set.
- **Generic.** A field is written once over a small `Scalar` algebra trait and runs
  unchanged in `f64`, `Interval`, `TaylorModel`, or `Jet` (time-Taylor) arithmetic.
- **Modular.** Integrators are swappable behind the `Integrator` trait. Adding a new
  scheme does not touch any field definition.
- **Lightweight.** Zero runtime dependencies. Rigour comes from native `f64` ULP
  rounding (`next_up` / `next_down`), no `unsafe`, no external GMP/MPFR build.

## Quick start

```rust
use tmflow::prelude::*;

let sys = VanDerPol::new(1.0);

// Verified reachable set from an initial box, Taylor-Model order k = 4.
let tm = TaylorModelIntegrator::new(&sys, 4);
let verified = propagate(&tm, [1.4, 0.0], [0.08, 0.08], 0.1, 20);
assert!(verified.rigorous);
println!("final verified box area = {:.4e}", verified.measures.last().unwrap());

// Same field, fast non-rigorous RK4 reference (propagates the box center).
let rk = Rk4::new(&sys);
let approx = propagate(&rk, [1.4, 0.0], [0.08, 0.08], 0.1, 20);
assert!(!approx.rigorous);
```

Run the bundled showcase:

```sh
cargo run --release --example vanderpol     # TM vs RK4, side by side (+ writes plot JSON)
cargo run --release --example validate      # 100% containment validation report
cargo test  --release                       # soundness + containment tests
cargo bench                                 # criterion wall-time benchmark
```

### Demonstration figure

The crate is plotting-free by design (zero deps), but ships an optional Python
helper so you can *see* the verified set propagate and build intuition for what
the library does. The `vanderpol` example serializes the run to
`out/vanderpol_tm.json` (via `Trajectory::to_json` + the Taylor-Model
`boundary_json` extra); `scripts/plot.py` renders it:

```sh
cargo run --release --example vanderpol     # writes out/vanderpol_tm.json
python3 scripts/plot.py                      # writes out/vanderpol_tm.png
```

The figure shows, on the left, the **curved verified reachable set** evolving in
the phase plane with its rigorous bounding boxes (coloured by time); on the
right, the bounding-box area and Taylor-Model remainder widths vs time on a log
scale (the "tightness" of the enclosure). Requires `matplotlib` + `numpy`.

## Architecture

Two small trait abstractions do the heavy lifting. The implementations are
deliberately thin — the design is in the interfaces.

| Trait / type | Role |
|---|---|
| `Scalar` (`src/scalar.rs`) | Ring-like algebra: `+ - * neg`, constant lifting, `scale_real`, `div_u32`, `powi`. A field is written once over `Scalar`; the **same** code monomorphizes into `f64`, `Interval`, `TaylorModel`, or `Jet`. `div_u32` is rigorous (outward-rounded) so verified backends stay sound. |
| `System<const N>` (`src/system.rs`) | An ODE field `x' = f(x)`, given by a single generic `fn eval<A: Scalar>(&self, x: &[A; N]) -> [A; N]`. |
| `Integrator<const N>` (`src/integrator.rs`) | A swappable stepping scheme with an associated `State` and `label / init / step / bbox / is_rigorous`. |
| `Jet<A>` (`src/jet.rs`) | Truncated time-power series with `Scalar` coefficients; itself a `Scalar` (Cauchy product). `time_series(sys, x0, order)` performs **automatic** time-Taylor differentiation of *any* `System` — so the verified integrator needs **no hand-derived recurrence** per field. |

### Backends

- **`Rk4`** (`src/integrators/rk4.rs`) — classic 4th-order Runge–Kutta in `f64`.
  Propagates a single point (the box center). Fast, **not** rigorous. The honest
  "what does a non-verified solver cost" baseline.
- **`TaylorModelIntegrator`** (`src/integrators/taylor.rs`) — verified set
  propagation. Each step:
  1. builds an a-priori enclosure of the flow over `[t, t+h]` via a Picard /
     Banach fixed-point iteration in interval arithmetic;
  2. expands the flow to order `k` in time using `Jet` automatic differentiation,
     carrying the spatial dependence as a multivariate Taylor Model;
  3. bounds the truncation error with a Lagrange remainder over the a-priori
     enclosure and folds it into the model's interval remainder;
  4. adaptively bisects the domain (up to a small fixed depth) when the remainder
     would otherwise blow up.

### Adding your own field

Implement `System<N>` once — generically over `Scalar` — and every integrator
works on it immediately:

```rust
use tmflow::prelude::*;

struct Brusselator { a: f64, b: f64 }

impl System<2> for Brusselator {
    fn eval<A: Scalar>(&self, x: &[A; 2]) -> [A; 2] {
        let a = A::from_f64(self.a);
        let x2y = x[0].clone() * x[0].clone() * x[1].clone();
        [
            a + x2y.clone() - x[0].clone().scale_real(self.b + 1.0),
            x[0].clone().scale_real(self.b) - x2y,
        ]
    }
    fn name(&self) -> &str { "Brusselator" }
}
```

Adding a new **integrator** is symmetric: implement `Integrator<N>` and it slots
into `propagate` without touching any field.

## Design notes

- **No external interval crate.** `inari` (IEEE-1788) pulls in `gmp-mpfr-sys`,
  which compiles GMP/MPFR from source. tmflow instead rounds outward with stable
  `f64::next_up` / `f64::next_down` — zero deps, no `unsafe`, and faithful to the
  original `f64`-boundary Python prototype.
- **Soundness is tested, not asserted.** `tests/soundness.rs` checks that interval
  / Taylor-Model evaluation encloses ground-truth `f64` evaluation across randomized
  inputs; `tests/containment.rs` enforces 100% Monte-Carlo containment and that the
  enclosure shrinks with `k`.