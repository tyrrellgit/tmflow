# 2. Introductory theory — how a guarantee is possible

This chapter builds intuition for the two ideas that let tmflow return a
*provable* enclosure. It is deliberately light on formalism; the goal is to help
you *grok* the concept, not to replace a textbook. References at the end point to
the rigorous treatments.

## 2.1 The enemy: errors accumulate, and we usually ignore them

Two kinds of error creep into any numerical integration:

1. **Truncation error** — a method like RK4 or a Taylor expansion keeps only
   finitely many terms; the tail is dropped.
2. **Rounding error** — `f64` cannot represent most reals exactly, so every
   arithmetic operation is slightly off.

A standard solver throws both away and reports a single number. A *validated*
solver instead carries an explicit, guaranteed bound on both at every step. The
two tools below are how it does that.

## 2.2 Tool one: interval arithmetic (never lie about rounding)

Represent a quantity not as one `f64`, but as an interval `[lo, hi]` that is
*guaranteed* to contain the true value. Arithmetic is defined to preserve that
guarantee — for example

```text
[a, b] + [c, d] = [a + c,  b + d]
[a, b] · [c, d] = [min(ac, ad, bc, bd),  max(ac, ad, bc, bd)]
```

The crucial subtlety is **rounding direction**. When we compute `a + c` in `f64`,
the true sum may not be representable. If we round the *lower* endpoint **down**
and the *upper* endpoint **up** — "outward rounding" — the resulting interval is
slightly *wider* than the exact one, and therefore still a valid enclosure. It
can never be too small. tmflow's [`Interval`](crate::interval::Interval) does
exactly this using the stable `f64::next_down` / `f64::next_up` primitives, so
every interval result is a sound enclosure of the real-number result. This is the
foundation: as long as every operation rounds outward, *no step can ever
under-report the truth.*

The price is **over-estimation**. Naively, intervals grow pessimistically — the
classic example is `x - x`, which interval arithmetic evaluates to `[-w, w]`
(width `2w`) instead of `0`, because it forgets the two `x`'s are the same.
That pessimism is what the next tool fixes.

## 2.3 Tool two: Taylor Models (track *shape*, not just *bounds*)

A **Taylor Model** (TM) represents a function over a box not as a crude interval,
but as

```text
   p(s)  +  I
```

where `p` is a **polynomial** in the box's normalized coordinates
`s ∈ [-1, 1]^n` (the *known shape* of the dependence) and `I` is a small
**interval remainder** that rigorously bounds *everything the polynomial misses*
— truncation tail plus accumulated rounding. The guarantee is:

```text
   true value  ∈  { p(s) : s ∈ [-1,1]^n }  +  I       (set sum)
```

The win over plain intervals is that correlations are preserved inside `p`. Now
`x - x` collapses to `0` in the polynomial part, because the solver *knows* it is
the same variable. The remainder `I` stays tiny. This is why a TM enclosure of a
reachable set is a thin curved sliver instead of a giant axis-aligned box — see
the demonstration figure produced by `scripts/plot.py`.

In tmflow, [`TaylorModel`](crate::taylor_model::TaylorModel) stores `p` as a map
from exponent-vectors to interval coefficients, plus the remainder `I`. Its
`+`, `-`, `*` operations multiply the polynomials and *rigorously* fold the
leftover high-order terms into `I`. Because it implements the
[`Scalar`](crate::scalar::Scalar) trait, your vector field — written once,
generically — evaluates directly in TM arithmetic with no special-casing.

## 2.4 Putting them together: one verified step

To advance the set from time `t` to `t + h`, the
[`TaylorModelIntegrator`](crate::integrators::TaylorModelIntegrator) does four
things:

1. **A-priori enclosure (Picard / Banach).** Before computing the precise answer,
   it first finds *some* crude box `[B]` guaranteed to contain the whole flow over
   the time interval `[t, t+h]`. It does this with a Picard iteration: the ODE in
   integral form is `x(t+τ) = x(t) + ∫ f(x) dτ`, and the right-hand side is a
   contraction for small `h`, so iterating it in interval arithmetic until the box
   stops growing yields a self-consistent enclosure (Banach fixed-point). This
   `[B]` is where the truncation error will be evaluated.

2. **Time-Taylor expansion.** It expands the solution in powers of time,
   `x(t+τ) = Σ_{j=0}^{k} a_j τ^j + (remainder)`, where each coefficient `a_j` is
   itself a *spatial* Taylor Model carrying the dependence on the initial box.
   The coefficients come from the field automatically (see
   [`Jet`](crate::jet::Jet) below) — no hand-derived recurrence per system.

3. **Lagrange remainder.** The dropped tail of the time series is bounded with the
   classical Lagrange form, evaluated over the a-priori box `[B]` from step 1.
   That bound is added into the interval remainder, so the truncation error is
   *accounted for*, not ignored.

4. **Adaptive bisection.** If the remainder would balloon (a stiff turn, a large
   box), the domain is split and each piece propagated separately, then reunited
   — trading a little speed for a tighter, still-rigorous enclosure.

The result is a new TM at `t + h` whose enclosure provably contains the image of
the old set under the true flow. Iterate, and you have a verified trajectory of
sets.

## 2.5 Where the Taylor coefficients come from: `Jet` automatic differentiation

Step 2 needs the time-derivatives of the solution. Computing those by hand for
each new ODE is tedious and error-prone. tmflow instead uses a
[`Jet`](crate::jet::Jet): a truncated power series in time whose coefficients are
themselves `Scalar` values, with multiplication defined by the **Cauchy
product**. A `Jet` *is* a `Scalar`, so evaluating your field on jets and reading
off the coefficients yields the time-Taylor expansion of the solution
**automatically**, for *any* field you write. This is automatic differentiation
in the truncated-power-series style, and it is what keeps the integrator generic.

## 2.6 The one-paragraph summary

> Carry sets, not points. Represent each set as a polynomial (its shape) plus a
> small interval (everything the polynomial misses). Do all arithmetic with
> outward rounding so bounds can never be too small. Get the time-derivatives for
> free via power-series automatic differentiation, bound the truncation tail with
> a Lagrange remainder over a Picard a-priori box, and split the domain when
> things get tight. The output is a region mathematically guaranteed to contain
> every true trajectory.

## Further reading

- Makino & Berz, *Taylor Models and Other Validated Functional Inclusion
  Methods* (the canonical TM reference; COSY-Infinity).
- Moore, Kearfott & Cloud, *Introduction to Interval Analysis* (intervals and
  outward rounding).
- Nedialkov, *Interval Tools for ODEs and DAEs* (validated ODE integration;
  CAPD-style methods).
- IEEE Std 1788-2015, *Standard for Interval Arithmetic*.
