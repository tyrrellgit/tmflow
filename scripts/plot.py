#!/usr/bin/env python3
"""Render the verified tmflow run as a demonstration figure.

Reads `out/vanderpol_tm.json` (written by `cargo run --release --example
vanderpol`) and draws:

  * left  — the curved verified reachable set evolving in the phase plane, with
            its rigorous bounding boxes, coloured by time;
  * right — verified bounding-box area and Taylor-Model remainder width vs time
            (log scale), showing how rigorously the enclosure stays controlled.

The tmflow crate is plotting-free by design; this is an optional visualization
helper. Requires matplotlib + numpy.

Usage:
    cargo run --release --example vanderpol      # writes the JSON
    python3 scripts/plot.py                      # writes out/vanderpol_tm.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")


def main():
    path = os.path.join(OUT, "vanderpol_tm.json")
    if not os.path.exists(path):
        sys.exit(
            f"missing {path}\n"
            "run first: cargo run --release --example vanderpol"
        )
    with open(path) as fh:
        d = json.load(fh)

    # boxes: flat [xlo, xhi, ylo, yhi] per step (2-D).
    boxes = [(b[0], b[1], b[2], b[3]) for b in d["boxes"]]
    boundaries = d["boundaries"]
    measures = d["measures"]
    rem_w = d["rem_w"]
    h = d["h"]
    n_steps = d["n_steps"]
    label = d.get("label", "TaylorModel")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- left: curved verified reachable set + bounding boxes ----
    cmap = plt.cm.viridis
    for i, (poly, bb) in enumerate(zip(boundaries, boxes)):
        frac = i / max(1, len(boundaries) - 1)
        col = cmap(frac)
        P = np.array(poly)
        ax.plot(P[:, 0], P[:, 1], color=col, lw=1.3, alpha=0.9)
        xlo, xhi, ylo, yhi = bb
        ax.add_patch(Rectangle((xlo, ylo), xhi - xlo, yhi - ylo,
                               fill=False, edgecolor=col, lw=0.5,
                               linestyle=":", alpha=0.5))
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(0, n_steps * h))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label("time t")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title(f"Verified reachable set ({label})")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")

    # ---- right: bbox area + remainder width vs time (log) ----
    # Step 0 has an exact, zero-width remainder; skip it on the log remainder
    # curves so the axis is not blown out.
    t = [i * h for i in range(len(measures))]
    ax2.semilogy(t, measures, "o-", color="#1f77b4", label="verified bbox area")
    t1 = t[1:]
    wx = [w[0] for w in rem_w[1:]]
    wy = [w[1] for w in rem_w[1:]]
    ax2.semilogy(t1, wx, "s--", color="#d62728", ms=3, label="remainder width x0")
    ax2.semilogy(t1, wy, "^--", color="#2ca02c", ms=3, label="remainder width x1")
    ax2.set_xlabel("time t")
    ax2.set_ylabel("area / remainder width (log)")
    ax2.set_title("Tightness over time")
    ax2.grid(alpha=0.25, which="both")
    ax2.legend(fontsize=9)

    fig.suptitle(
        "tmflow — verified Taylor-Model reachability of the Van der Pol "
        f"oscillator (h={h}, {n_steps} steps)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    dst = os.path.join(OUT, "vanderpol_tm.png")
    fig.savefig(dst, dpi=140)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
