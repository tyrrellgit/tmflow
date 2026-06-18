# 1. Introduction — why a *set* and not a *point*

Suppose you integrate an ODE

```text
x'(t) = f(x(t)),   x(0) = x0.
```

A classical solver — Runge–Kutta, say — takes one initial point `x0` and walks
one approximate trajectory forward. That is exactly what tmflow's `Rk4`
integrator does, and it is fast and useful. But it answers a narrow question:
*"where does this one nearby-correct trajectory go?"* It does **not** tell you:

- how the answer changes if `x0` is only known to lie in a **box** (measurement
  uncertainty, a range of operating conditions, a set of initial states);
- how much **error** the numerical method itself introduced;
- whether the true solution could ever enter some unsafe region.

`tmflow` is built to answer the harder question rigorously: given an initial
**set** `X0` (a box of possible starting states), produce a region `X(t)` that is
**guaranteed** to contain *every* true trajectory starting in `X0`, accounting
for both the spread of initial conditions and all numerical error.

```text
   point solver                 set solver (tmflow, verified)
   ------------                  -----------------------------
   x0  ──>  one curve            X0 (a box) ──> X(t), a region
                                 that PROVABLY contains all
                                 trajectories from X0
```

This is the field of **validated** (or *verified*, *rigorous*) numerics, in the
spirit of tools like COSY-Infinity and CAPD. The key word is *guaranteed*: the
output is not "the answer plus a heuristic error bar," it is a mathematically
sound over-approximation. If tmflow says the set is contained in box `B`, then it
is — no trajectory can escape, modulo the correctness of the implementation
(which the test suite exercises against Monte-Carlo ground truth: 100 %
containment at every step).

## What you get

- A **verified Taylor-Model integrator** that propagates a box forward as a
  curved polynomial set with a rigorous interval error term.
- A fast **RK4** integrator for the non-rigorous point reference — the honest
  baseline for "what does rigour cost?".
- A **trait-based design** so the *same* vector field runs under both, and so new
  integrators or new systems slot in without rewriting anything.

## What this guide covers

The next chapter, [theory](../theory/index.html), explains *how* a guaranteed
enclosure is even possible — the two ideas (interval arithmetic and Taylor
Models) that make it work, kept deliberately introductory. If you would rather
start coding, jump to the [tutorial](../tutorial/index.html); the theory will
make more sense on a second read.
