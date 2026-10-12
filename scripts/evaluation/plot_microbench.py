"""Render throughput / latency figures from a latency-microbench run.

Reads ``summary.json`` from a run directory (produced by
``run_latency_microbench.py``) and emits PDFs suitable for inclusion
in the paper's case-study section. Default output is alongside the
summary in the run directory; override with ``--out-dir``.

Two figures by default:

1. ``throughput-{domain}.pdf`` — concurrency on x, two y-series:
   end-to-end tasks/s (left axis, log) and engine auth/s from the
   Rust loadgen (right axis, log). Shows the 5+ orders-of-magnitude
   gap between agent-paced throughput and engine ceiling.

2. ``latency-{domain}.pdf`` — concurrency on x, p50/p95/p99 lines
   for each of the three latency sources (SDK / server / loadgen).
   Lets the reader see the tail composition at a glance.

Usage::

    uv run python scripts/evaluation/plot_microbench.py outputs/latency-microbench/<ts>
    uv run python scripts/evaluation/plot_microbench.py outputs/latency-microbench/<ts> \\
        --out-dir paper/paper-ccs/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Paper-friendly defaults: tight layout, small fonts, vector PDF.
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _cells_for(summary: dict, domain: str) -> list[dict]:
    return [c for c in summary["cells"] if c["domain"] == domain]


def _xy(cells: list[dict], y_field: str) -> tuple[list[int], list[float]]:
    pairs = sorted(
        ((int(c["concurrency"]), c[y_field]) for c in cells if c[y_field] is not None),
        key=lambda p: p[0],
    )
    return [p[0] for p in pairs], [p[1] for p in pairs]


SOURCES = [
    ("Client side", "#0d4a8a", "auth"),
    ("Server side", "#1f7a3a", "server"),
    ("Replay",      "#c44e1a", "loadgen"),
]


def _draw_throughput(ax, sasy: list[dict]):
    """Draw the engine throughput line (Rust loadgen replay) onto
    ``ax``. Linear y-scale — the auth/s curve plateaus at the
    engine ceiling, which is more readable as a flat line than as
    a bend on a log axis. Returns the list of line handles.

    When ``auth_throughput_std`` is populated (i.e. multi-trial
    replay), draw error bars at ±1 stdev so the reader can see
    the noise band; otherwise fall back to a plain line.

    The client-side ``tasks/s`` is intentionally not drawn here:
    it lives on a different scale (sub-1 vs 10⁴) and combining
    them on a twin axis makes both unreadable. That column is
    available in the throughput table in ``results.md`` /
    ``results.tex``, with the per-task duration mean / stdev that
    explains its shape.
    """
    x, y = _xy(sasy, "auth_throughput_calls_per_s")
    by_x = {int(c["concurrency"]): c for c in sasy}
    yerr = [by_x[xi].get("auth_throughput_std") or 0.0 for xi in x]
    has_err = any(e and e > 0 for e in yerr)
    handles: list = []
    if x:
        # NOTE: ``ax.errorbar`` returns an ``ErrorbarContainer`` —
        # only the container carries the legend label, the inner
        # ``Line2D`` does not. Append the container directly so
        # ``get_label()`` returns "Replay (authorizations/s)" instead of
        # matplotlib's default ``_nolegend_`` placeholder.
        if has_err:
            container = ax.errorbar(
                x, y, yerr=yerr,
                color="#c44e1a", linewidth=1.5,
                linestyle="-", capsize=3, elinewidth=1.0,
                label="Replay (authorizations/s)",
            )
            handles.append(container)
        else:
            line, = ax.plot(
                x, y,
                color="#c44e1a", linewidth=1.5,
                linestyle="-", label="Replay (authorizations/s)",
            )
            handles.append(line)
    ax.set_xscale("log", base=2)
    if x:
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel("Agent concurrency")
    ax.set_ylabel("Authorizations / s")
    # Force the y-axis to start at 0 so plateauing at the engine
    # ceiling is visually obvious.
    if y:
        ax.set_ylim(bottom=0)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    return handles


def _draw_latency(ax, sasy: list[dict]) -> None:
    """Draw the multi-source latency plot onto ``ax``."""
    quantiles = [
        ("median", "_p50_us", "-"),
        ("p99",    "_p99_us", "--"),
    ]
    for label, color, prefix in SOURCES:
        for qname, qsuffix, ls in quantiles:
            field = f"{prefix}{qsuffix}"
            x, y = _xy(sasy, field)
            if not y:
                continue
            y_ms = [v / 1000.0 for v in y]
            ax.plot(
                x, y_ms,
                marker="o" if qname == "median" else "s",
                linestyle=ls,
                color=color,
                linewidth=1.4,
                markersize=4,
                label=f"{label} {qname}",
            )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    if sasy:
        xs = sorted({int(c["concurrency"]) for c in sasy})
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("Agent concurrency")
    ax.set_ylabel("Authorization latency (ms)")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)


def plot_throughput(domain: str, cells: list[dict], out_path: Path) -> None:
    sasy = [c for c in cells if c["mode"] == "sasy"]
    fig, ax_left = plt.subplots(figsize=(3.8, 2.5))
    handles = _draw_throughput(ax_left, sasy)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        ncol=1,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.34)
    fig.savefig(out_path)
    plt.close(fig)


def plot_latency(domain: str, cells: list[dict], out_path: Path) -> None:
    sasy = [c for c in cells if c["mode"] == "sasy"]
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    _draw_latency(ax, sasy)
    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        ncol=2,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.42)
    fig.savefig(out_path)
    plt.close(fig)


def plot_combined(domain: str, cells: list[dict], out_path: Path) -> None:
    """Two-panel figure: (a) latency, (b) throughput. Both share an
    x-axis (agent concurrency) and a single legend region below.
    Sized to fit a paper's full text width.
    """
    sasy = [c for c in cells if c["mode"] == "sasy"]
    fig, (ax_lat, ax_thr) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    _draw_latency(ax_lat, sasy)
    thru_handles = _draw_throughput(ax_thr, sasy)

    # Sub-panel labels under each axes — matches the (a)/(b)
    # convention used elsewhere in the paper.
    ax_lat.text(
        0.5, -0.32, "(a) Authorization latency",
        transform=ax_lat.transAxes, ha="center", va="top",
    )
    ax_thr.text(
        0.5, -0.32, "(b) Throughput",
        transform=ax_thr.transAxes, ha="center", va="top",
    )

    # Compose a single combined legend below the figure.
    handles, labels = ax_lat.get_legend_handles_labels()
    handles += thru_handles
    labels  += [h.get_label() for h in thru_handles]
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        frameon=True,
        ncol=4,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.36, wspace=0.45)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dir", type=Path,
        help="Path to a latency-microbench run directory containing summary.json",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Where to write the PDFs (default: alongside summary.json).",
    )
    parser.add_argument(
        "--domains", nargs="+", default=None,
        help="Limit to specific domains (default: all in summary).",
    )
    args = parser.parse_args()

    summary_path = args.run_dir / "summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found", file=sys.stderr)
        return 1
    summary = json.loads(summary_path.read_text())

    out_dir = args.out_dir or args.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = args.domains or summary["domains"]
    for domain in domains:
        cells = _cells_for(summary, domain)
        if not cells:
            print(f"  skip {domain}: no cells")
            continue
        thru = out_dir / f"throughput-{domain}.pdf"
        lat  = out_dir / f"latency-{domain}.pdf"
        combo = out_dir / f"microbench-{domain}.pdf"
        plot_throughput(domain, cells, thru)
        plot_latency(domain, cells, lat)
        plot_combined(domain, cells, combo)
        print(f"  wrote {thru.relative_to(args.run_dir.parent)}")
        print(f"  wrote {lat.relative_to(args.run_dir.parent)}")
        print(f"  wrote {combo.relative_to(args.run_dir.parent)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
