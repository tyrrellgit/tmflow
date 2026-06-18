//! Showcase: propagate the Van der Pol oscillator under both integrators and
//! print the verified reachable-set boxes alongside the RK4 reference point.
//!
//! Run with:  cargo run --release --example vanderpol

use tmflow::integrators::{boundary_json, Rk4, TaylorModelIntegrator};
use tmflow::prelude::*;
use std::fs;

fn main() {
    let sys = VanDerPol::new(1.0);
    let center = [1.4, 0.0];
    let half = [0.08, 0.08];
    let (h, n) = (0.1, 20);

    println!("System: {}", sys.name());
    println!("Initial box: center={center:?} half={half:?},  h={h}, {n} steps\n");

    // Verified Taylor-Model reachable set (order k = 4).
    let tm = TaylorModelIntegrator::new(&sys, 4);
    let verified = propagate(&tm, center, half, h, n);

    // Fast non-rigorous RK4 reference (the box center as a point).
    let rk = Rk4::new(&sys);
    let approx = propagate(&rk, center, half, h, n);

    println!(
        "{:>4}  {:>6}   {:<34}   {:<20}",
        "step", "t", format!("{} (rigorous bbox)", verified.label), format!("{} (point)", approx.label)
    );
    for st in (0..=n).step_by(2) {
        let b = &verified.boxes[st];
        let p = &approx.boxes[st]; // degenerate (lo==hi)
        println!(
            "{st:>4}  {:>6.2}   x[{:+.4},{:+.4}] y[{:+.4},{:+.4}]   ({:+.4}, {:+.4})",
            st as f64 * h,
            b[0].0,
            b[0].1,
            b[1].0,
            b[1].1,
            p[0].0,
            p[1].0,
        );
    }

    println!(
        "\nFinal verified box area = {:.4e}  (rigorous: {})",
        verified.measures.last().unwrap(),
        verified.rigorous
    );
    println!(
        "Try other orders: `TaylorModelIntegrator::new(&sys, k)` for k = 2..=5,\n\
         or define your own field by implementing `System<N>`."
    );

    // ---- optional: dump JSON for the plotting helper (scripts/plot.py) ----
    // The crate stays plotting-free; this just serializes the run so the Python
    // helper can render the curved set + bounding boxes. The curved boundary is
    // a Taylor-Model-specific extra spliced into the generic trajectory JSON.
    fs::create_dir_all("out").ok();
    let extra = boundary_json(&verified.states, 24);
    let json = verified.to_json(&extra);
    let path = "out/vanderpol_tm.json";
    if fs::write(path, &json).is_ok() {
        println!("\nWrote {path} -> render with: python3 scripts/plot.py");
    }
}
