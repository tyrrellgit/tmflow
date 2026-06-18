# 3. Architecture tour — the four traits

tmflow's design philosophy: **put the cleverness in a few small interfaces, keep
the implementations thin.** Four abstractions carry the whole library. Understand
these and you understand tmflow.

```text
        write your field once …            … run it under any number system
        ────────────────────────           ──────────────────────────────────
        System<N>::eval<A: Scalar>   ─────▶  A = f64        (RK4 point)
                                             A = Interval   (a-priori enclosure)
                                             A = TaylorModel (verified set)
                                             A = Jet<A>     (time derivatives)

        Integrator<N>  ── step ──▶  propagate()  ──▶  Trajectory (boxes + measures)
```

## 3.1 `Scalar` — the linchpin

[`Scalar`](crate::scalar::Scalar) is a small ring-like algebra:
`Add`, `Sub`, `Mul`, `Neg`, plus constant lifting (`from_f64`, `one`, `zero`),
`scale_real`, a *rigorous* `div_u32` (outward-rounded reciprocal, so verified
backends stay sound), and `powi`.

Its purpose is **write-once genericity**. A vector field is written generically
over `A: Scalar`. Monomorphization then specializes that *same* source into four
arithmetics:

| `A` =          | gives you                                  |
|----------------|--------------------------------------------|
| `f64`          | fast, non-rigorous evaluation (RK4, Monte-Carlo) |
| [`Interval`](crate::interval::Interval)   | rigorous interval bounds (a-priori enclosures) |
| [`TaylorModel`](crate::taylor_model::TaylorModel) | verified set propagation |
| [`Jet`](crate::jet::Jet)`<A>`             | automatic time-Taylor differentiation |

There is no separate "interval version" and "Taylor-Model version" of your field
to keep in sync — there is one definition.

## 3.2 `System<N>` — your ODE field, written once

[`System`](crate::system::System) is the field `x' = f(x)`, defined by a single
generic method — `fn eval<A: Scalar>(&self, x: &[A; N]) -> [A; N]`. Here it is in
context on a minimal one-dimensional decay system `x' = -x`:

```
use tmflow::{System, Scalar};

struct Decay;

impl System<1> for Decay {
    fn eval<A: Scalar>(&self, x: &[A; 1]) -> [A; 1] {
        [-x[0].clone()]
    }
    fn name(&self) -> &str { "Decay" }
}

// The same `eval` runs in every Scalar arithmetic — here just f64.
let f = Decay.eval(&[2.0_f64]);
assert_eq!(f[0], -2.0);
```

The `const N: usize` is the state dimension, fixed at the type level (so arrays,
not heap vectors — no allocation in the hot path). You implement `eval` once; the
library supplies the four scalar types. The bundled
[`VanDerPol`](crate::systems::VanDerPol) is the worked example.

## 3.3 `Integrator<N>` — the swappable scheme

[`Integrator`](crate::integrator::Integrator) abstracts *how* a step happens. It
has an associated `State` (a point, or a vector of Taylor Models) and a handful of
methods:

- `label()` — for reports and plots;
- `init(center, half)` — build the initial state from a box;
- `step(state, h)` — advance one step;
- `bbox(state)` — the (rigorous, for verified schemes) bounding box;
- `is_rigorous()` — does `bbox` come with a guarantee?

Two implementations ship:

- [`Rk4`](crate::integrators::Rk4) — classic 4th-order Runge–Kutta in `f64`.
  Propagates the box *center* as a single point. Fast, **not** rigorous; the
  honest "cost of rigour" baseline.
- [`TaylorModelIntegrator`](crate::integrators::TaylorModelIntegrator) — the
  verified set integrator from chapter 2 (Picard a-priori enclosure → time-Taylor
  expansion via `Jet` → Lagrange remainder → adaptive bisection).

Because both implement the same trait, the
[`propagate`](crate::driver::propagate) driver runs either one unchanged, and a
*new* integrator slots in without touching any field.

## 3.4 `Jet` — automatic time differentiation

[`Jet`](crate::jet::Jet)`<A>` is a truncated power series in time whose
coefficients are `Scalar` values; it implements `Scalar` itself (multiplication =
Cauchy product). The free function
[`time_series`](crate::jet::time_series)`(sys, x0, order)` evaluates *any*
`System` on jets and returns the time-Taylor coefficients of its solution — the
automatic differentiation that frees the verified integrator from per-field
hand-derived recurrences.

> **Implementation note / footgun avoided.** A degree-0 jet (produced by
> `from_f64` / `one` / `zero`) represents a *constant*, and is broadcast across
> the other operand's order during multiplication. An earlier version naively
> truncated to the *minimum* order, which silently collapsed the whole series to
> order 0 and produced wrong high-order coefficients (and 0 % containment). The
> fix is documented in `jet.rs`; the containment test would catch any regression.

## 3.5 The driver and its output

[`propagate`](crate::driver::propagate)`(integrator, center, half, h, n_steps)`
loops `step`, records the bounding box and its measure (area/volume) at each step,
and returns a [`Trajectory`](crate::driver::Trajectory):

```text
Trajectory { states, boxes, measures, h, n_steps, label, rigorous }
```

[`Trajectory::to_json`](crate::driver::Trajectory::to_json) serializes the run
(zero-dependency) for the optional plotting helper. The crate itself never pulls
in a plotting dependency — visualization is a separate Python script fed by JSON.

## 3.6 The whole thing in one breath

`Scalar` lets a field be written once. `System` is that field. `Jet` gives its
time-derivatives for free. `Integrator` decides how to step — point (RK4) or
verified set (Taylor Model). `propagate` runs it and hands you a `Trajectory`.
Everything else is detail.
