//! # The tmflow guide
//!
//! A self-contained, prose-first introduction to **what tmflow does, why, and
//! how to use it.** It assumes no prior exposure to validated numerics — just
//! comfort with ODEs and Rust. Read the chapters in order, or jump to the one
//! you need:
//!
//! 1. [`intro`] — the problem: why a *set* and not a *point*.
//! 2. [`theory`] — introductory theory: intervals, Taylor Models, and how a
//!    *guaranteed* enclosure is produced.
//! 3. [`architecture`] — the trait tour: `Scalar`, `System`, `Integrator`, `Jet`.
//! 4. [`tutorial`] — a hands-on walkthrough, including bringing your own field.
//! 5. [`faq`] — design decisions, limitations, and honest caveats.
//!
//! These pages are documentation only — they contain no runnable items. The API
//! reference lives on the individual types and functions in the crate root.

#[doc = include_str!("../docs/01_intro.md")]
pub mod intro {}

#[doc = include_str!("../docs/02_theory.md")]
pub mod theory {}

#[doc = include_str!("../docs/03_architecture.md")]
pub mod architecture {}

#[doc = include_str!("../docs/04_tutorial.md")]
pub mod tutorial {}

#[doc = include_str!("../docs/05_faq.md")]
pub mod faq {}
