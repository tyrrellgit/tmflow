# 5. FAQ, design decisions, and honest limitations

## Why no external interval-arithmetic crate?

The natural choice would be `inari` (an IEEE-1788 implementation). It pulls in
`gmp-mpfr-sys`, which compiles GMP and MPFR from source — heavy, slow, and it
timed out in our build environment. tmflow instead rounds outward using the
stable `f64::next_up` / `f64::next_down` primitives. The result is **zero runtime
dependencies, no `unsafe`, no C build, no FPU-mode fiddling** — and it is faithful
to the `f64`-boundary approach of the original Python prototype. The trade-off:
our intervals are correctly-rounded at the ULP level rather than to a configurable
precision; for `f64`-domain reachability that is exactly what we want.

## Why is RK4 ten-thousand times faster? Is the Taylor-Model integrator slow?

They answer different questions. RK4 carries **one point** with no error control.
The Taylor-Model integrator carries a **guaranteed enclosure of an entire set**,
with a validated remainder, automatic differentiation, an a-priori Picard
enclosure, and adaptive bisection. The gap is the *cost of rigour*, and it is
honest to show it. (For context, the equivalent Python/`mpmath` set propagation is
~150× slower still than tmflow's verified path at order 4, so the Rust
implementation is already a large speedup over the prototype while producing
identical enclosures.)

## Does "verified" mean bug-free?

No — it means the *method* is mathematically sound: outward rounding, a-priori
enclosure, and Lagrange remainder together guarantee an over-approximation
*assuming the code is correct*. We back that assumption with tests: the suite
checks interval/TM evaluation encloses ground-truth `f64` evaluation across
randomized inputs (`tests/soundness.rs`) and enforces **100 % Monte-Carlo
containment** plus shrink-with-order on the Van der Pol benchmark
(`tests/containment.rs`). A logic bug that broke soundness would fail those tests.

## What fields can I write?

Anything built from the operations [`Scalar`](crate::scalar::Scalar) provides:
`+`, `-`, `*`, `neg`, `powi`, `scale_real`, `div_u32`, and constant lifting. That
covers **polynomial vector fields** — a large and important class (Van der Pol,
Brusselator, Lorenz, Lotka–Volterra, rigid-body / rotational dynamics, many
control and aerospace models).

## What about sin, exp, division by a variable, …?

Not yet. Transcendental functions and general (variable) division require adding
*rigorously bounded* implementations of those operations to the `Scalar` trait for
each backend — in particular a sound Taylor-Model `sin`/`exp`/`recip` with proper
remainder propagation. This is a clean, well-bounded extension, deliberately left
out to keep the first release lightweight. It is the natural next step when you
need it.

## Why is the curved-boundary plot 2-D only?

The boundary tracer in `boundary_json` walks the perimeter of the parameter
square `[-1,1]²` and evaluates the Taylor-Model polynomial image — inherently
2-D. Higher-dimensional systems still get rigorous bounding boxes and tightness
curves from the generic `Trajectory::to_json`; visualizing them just needs a
projection choice (pick two coordinates), which is an easy add when required.

## Why fix the state dimension `N` at the type level?

`const N: usize` means states are stack arrays `[A; N]`, not heap `Vec`s — no
allocation in the stepping loop, and the dimension is checked at compile time.
The cost is that `N` is not run-time dynamic, which is rarely a limitation for the
small-to-medium systems this kind of verified integration targets.

## Why are integrators trait objects-free (generic, not `dyn`)?

Each integrator is monomorphized against its concrete system and scalar types, so
there is no virtual-dispatch overhead in the hot path. You still get full
swappability at the *source* level via the [`Integrator`](crate::integrator::Integrator)
trait; you simply choose the concrete type at the call site.

## How do I add a new integrator?

Implement [`Integrator`](crate::integrator::Integrator)`<N>` — pick a `State`,
write `init` / `step` / `bbox` / `label` / `is_rigorous` — and it works with
[`propagate`](crate::driver::propagate) and every existing system immediately. No
field code changes. A higher-order verified Runge–Kutta or a Hermite-based scheme
would both fit naturally here.

## Roadmap (lightweight, in rough order)

1. Transcendental operations on `Scalar` (sound TM `sin`/`cos`/`exp`/`recip`).
2. Shrink-wrapping / preconditioning to fight the wrapping effect over long
   horizons.
3. Variable step-size control driven by the remainder width.
4. N-dimensional visualization via coordinate projection.

None of these change the architecture — they are new `Scalar` ops or new
`Integrator` impls, which is exactly what the trait design was built to absorb.
