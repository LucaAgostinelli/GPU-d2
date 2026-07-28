#!/usr/bin/env python3
"""
generates the report's performance figures from
outputs/csv/*_summary.csv (plus outputs/strong_scaling/strong-*.out directly
for the comm/compute breakdown). Every figure is written as a PNG into
outputs/figures/.

Usage:
  python scripts/plot_charts.py
  (run from the project root, no arguments)
"""
import glob
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from result_parser import parse_files, chunk_repetitions

OUT_DIR = "outputs/figures"

# Palette: colors assigned by entity, kept consistent across every figure.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
RED = "#e34948"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
INK_DARK_GRAY = "#3a3936"
GRID = "#e1e0d9"
BASELINE_LINE = "#c3c2b7"
SURFACE = "#ffffff"

# Entity -> color, held constant everywhere it appears.
COMM_COLOR = {
    "ghost_nccl": AQUA,
    "checkerboard_nccl": MAGENTA,
}

STRUCTURED = ["Queen_4147.mtx", "nlpkkt240.mtx"]
UNSTRUCTURED = ["webbase-2001.mtx", "com-Orkut.mtx"]
CATEGORY_COLOR = {"structured": BLUE, "power-law / unstructured": ORANGE}

# cyclic is the reference scheme; RCM/Block get their own hues.
PARTITION_COLOR = {"cyclic": INK_SECONDARY, "rcm": BLUE, "block": YELLOW}
PARTITION_LABEL = {"cyclic": "cyclic (baseline)", "rcm": "RCM", "block": "Block"}
PARTITION_COMM_TAG = {"cyclic": "ghost_nccl", "rcm": "rcm_nccl", "block": "block_nccl"}
BONUS_DIR = "bonus_partitioning_strategies"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE_LINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "font.size": 10,
        "figure.titlesize": 14,
        "axes.titlesize": 10,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
    })


def short(m):
    return m.replace(".mtx", "")


def matrix_order(strong):
    """Order matrices by P=1 T_median_ms (a size proxy) so faceted figures walk small -> large."""
    sizes = (strong[(strong.comm == "ghost_mpi") & (strong.P == 1)]
             .drop_duplicates("matrix"))
    if sizes.empty:
        return sorted(strong.matrix.unique())
    return list(sizes.sort_values("T_median_ms").matrix)


def load_csv(name, required=True):
    path = f"outputs/csv/{name}"
    if not os.path.exists(path):
        if required:
            print(f"missing {path} -- run scripts/analyze_{name.replace('_summary.csv','')}.py first",
                  file=sys.stderr)
        return None
    return pd.read_csv(path)


def load_bonus_csv(name):
    path = f"{BONUS_DIR}/outputs/csv/{name}"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def prototype_series(paths, comm_tag, kernel="cusparse"):
    """Parse raw RESULT logs into {matrix: {P: {t_max_ms, gflops, compute_ms, comm_ms}}}.
    t_max_ms/gflops are medians over reps (gflops = 2*nnz_total/T_max, not a
    sum of per-rank effective_gflops). compute_ms/comm_ms are the slowest-rank
    split for the median rep.
    """
    if not paths:
        return {}
    rows = [r for r in parse_files(paths)
            if r.get("comm") == comm_tag and r.get("kernel") == kernel]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["matrix"], r["P"])].append(r)
    out = defaultdict(dict)
    for (matrix, P), grp in groups.items():
        chunks = chunk_repetitions(grp, P)
        reps = []
        for chunk in chunks:
            t_max = max(r["total_avg_ms"] for r in chunk)
            nnz_total = sum(r["nnz_local"] for r in chunk)
            gflops = 2.0 * nnz_total / (t_max / 1e3) / 1e9
            compute = max(r["compute_avg_ms"] for r in chunk)
            comm = max(r["comm_avg_ms"] for r in chunk)
            reps.append((t_max, gflops, compute, comm))
        reps.sort(key=lambda r: r[0])
        mid = reps[len(reps) // 2]
        out[matrix][P] = {
            "t_max_ms": float(np.median([r[0] for r in reps])),
            "gflops": float(np.median([r[1] for r in reps])),
            "compute_ms": mid[2],
            "comm_ms": mid[3],
        }
    return out


def order_matrices_from_series(series_by_method, p_ref=1):
    """Order matrices by the cyclic reference's P=p_ref t_max_ms (size proxy),
    falling back to alphabetical where cyclic has no data at that P."""
    cyclic = series_by_method.get("cyclic", {})
    matrices = set()
    for series in series_by_method.values():
        matrices.update(series.keys())

    def key(m):
        d = cyclic.get(m, {}).get(p_ref)
        return (0, d["t_max_ms"]) if d else (1, m)

    return sorted(matrices, key=key)


def savefig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# 5: NCCL vs MPI point-to-point speedup for 1D ghost exchange, across P.
def fig_nccl_speedup(df_in, fname, title, figsize=(8, 6), legend_fontsize=10):
    df = df_in[(df_in.kernel == "cusparse") &
               (df_in.comm.isin(["ghost_mpi", "ghost_nccl"]))]
    matrices = [m for m in matrix_order(df_in) if m in df.matrix.unique()]
    Ps = sorted(df.P.unique())

    fig, ax = plt.subplots(figsize=figsize)
    by_P = {p: [] for p in Ps}
    for m in matrices:
        sub = df[df.matrix == m]
        xs, ys = [], []
        for p in Ps:
            mpi_t = sub[(sub.P == p) & (sub.comm == "ghost_mpi")].T_median_ms
            nccl_t = sub[(sub.P == p) & (sub.comm == "ghost_nccl")].T_median_ms
            if mpi_t.empty or nccl_t.empty:
                continue
            sp = mpi_t.iloc[0] / nccl_t.iloc[0]
            xs.append(p)
            ys.append(sp)
            by_P[p].append(sp)
        ax.plot(xs, ys, color=INK_MUTED, alpha=0.35, linewidth=1.1,
                 marker="o", markersize=3, zorder=1)

    median_ys = [np.median(by_P[p]) for p in Ps]
    ax.plot(Ps, median_ys, color=AQUA, linewidth=3, marker="o", markersize=8,
             label="median across matrices", zorder=3)
    ax.axhline(1.0, color=INK_PRIMARY, linestyle="--", linewidth=2.2, zorder=2)
    ax.text(Ps[-1], 1.0, "parity  ", color=INK_PRIMARY, fontsize=10.5,
            fontweight="bold", va="bottom", ha="right", zorder=4)
    ax.set_xticks(Ps)
    ax.set_xlabel("P (ranks/GPUs)")
    ax.set_ylabel("NCCL speedup  (T_mpi / T_nccl)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=legend_fontsize)
    fig.tight_layout()
    savefig(fig, fname)


# 8: weak-scaling efficiency, two panels (saturated vs. partial ghost
# coverage), 4 series each (1D ghost, 2D checkerboard, RCM, Block), NCCL/cuSPARSE.
WEAK_COMBOS = [
    ("ghost_nccl", "1D ghost"),
    ("checkerboard_nccl", "2D checkerboard"),
    ("rcm_nccl", "RCM"),
    ("block_nccl", "Block"),
]


def fig_weak_scaling(weak, weak_sparse, weak_part, weak_part_sparse):
    color_for = {
        "ghost_nccl": COMM_COLOR["ghost_nccl"],
        "checkerboard_nccl": COMM_COLOR["checkerboard_nccl"],
        "rcm_nccl": PARTITION_COLOR["rcm"],
        "block_nccl": PARTITION_COLOR["block"],
    }
    panels = [
        (weak, weak_part, "saturated ghost coverage (~100%)"),
        (weak_sparse, weak_part_sparse, "partial ghost coverage (~40-60%)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), sharey=True)
    for ax, (base_df, part_df, subtitle) in zip(axes, panels):
        all_ps = set()
        for comm, label in WEAK_COMBOS:
            src = part_df if comm in ("rcm_nccl", "block_nccl") else base_df
            if src is None:
                continue
            s = src[(src.comm == comm) & (src.kernel == "cusparse")].sort_values("P")
            if s.empty:
                continue
            all_ps.update(s.P.tolist())
            ax.plot(s.P, s.E_weak, marker="o", color=color_for[comm],
                    linewidth=2.6, markersize=8, label=label, zorder=3)
        ps_sorted = sorted(all_ps)
        if ps_sorted:
            ax.axhline(1.0, color=INK_PRIMARY, linestyle="--", linewidth=2.4, zorder=2)
            ax.text(ps_sorted[-1], 1.0, "ideal (E=1)  ", color=INK_PRIMARY,
                    fontsize=13, fontweight="bold", va="bottom", ha="right", zorder=4)
            ax.set_xticks(ps_sorted)
        ax.set_xlabel("P (ranks/GPUs)", fontsize=13.5)
        ax.set_title(subtitle, fontsize=14.5)
        ax.tick_params(axis="both", labelsize=13)
        # log scale: linear crushes the low end where strategies differ.
        ax.set_yscale("log")
        ax.set_ylim(0.07, 1.35)
        yticks = [0.1, 0.2, 0.3, 0.5, 1.0]
        ax.yaxis.set_major_locator(mticker.FixedLocator(yticks))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_locator(mticker.NullLocator())

    axes[0].set_ylabel("E_weak(P) = T(1) / T(P)", fontsize=13.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.1), fontsize=19)
    fig.suptitle("Weak scaling efficiency by partitioning strategy (NCCL, cuSPARSE)\n"
                 "synthetic Erdos-Renyi matrices, ~1.5M nnz/rank", y=1.08, fontsize=17)
    fig.tight_layout()
    savefig(fig, "08_weak_scaling_efficiency")


# 11: structured vs. unstructured, large matrices, P=4, cuSPARSE.
# ghost/naive volume ratio isn't in the CSV, so it's recomputed from raw logs.
def fig_structured_vs_unstructured(large, large_cyclic_paths):
    matrices = STRUCTURED + UNSTRUCTURED

    rows = parse_files(large_cyclic_paths)
    rows = [r for r in rows if r["P"] == 4 and r["matrix"] in matrices
            and r["kernel"] == "cusparse" and r["comm"] in ("bcast", "ghost_nccl")]
    ghost_over_naive = {}
    for m in matrices:
        naive_vals = [r["ncols_global"] for r in rows if r["matrix"] == m and r["comm"] == "bcast"]
        ghost_vals = [r["n_ghost"] for r in rows if r["matrix"] == m and r["comm"] == "ghost_nccl"]
        ghost_over_naive[m] = (float(np.mean(ghost_vals)) / float(np.mean(naive_vals))
                                if naive_vals and ghost_vals else np.nan)

    st = large[(large.kernel == "cusparse") & (large.P == 4) &
               (large.matrix.isin(matrices))]
    speedup = {}
    for m in matrices:
        b = st[(st.matrix == m) & (st.comm == "bcast")].T_median_ms
        n = st[(st.matrix == m) & (st.comm == "ghost_nccl")].T_median_ms
        speedup[m] = (b.iloc[0] / n.iloc[0]) if not b.empty and not n.empty else np.nan

    categories = ["structured" if m in STRUCTURED else "power-law / unstructured"
                  for m in matrices]
    colors = [CATEGORY_COLOR[c] for c in categories]
    labels = [short(m) for m in matrices]
    x = np.arange(len(matrices))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].bar(x, [ghost_over_naive[m] for m in matrices], color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=12.5, color=INK_DARK_GRAY)
    axes[0].set_ylabel("1D ghost / naive\nvolume, P=4", fontsize=13, color=INK_DARK_GRAY)
    axes[0].set_title("Ghost-exchange saturation", fontsize=14.5, color=INK_DARK_GRAY)
    axes[0].tick_params(axis="y", labelsize=12, colors=INK_DARK_GRAY)

    axes[1].bar(x, [speedup[m] for m in matrices], color=colors)
    axes[1].axhline(1.0, color=BASELINE_LINE, linestyle="--", linewidth=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=12.5, color=INK_DARK_GRAY)
    axes[1].set_ylabel("speedup of 1D+NCCL\nvs. naive bcast, P=4", fontsize=13, color=INK_DARK_GRAY)
    axes[1].set_title("Payoff of selective exchange", fontsize=14.5, color=INK_DARK_GRAY)
    axes[1].tick_params(axis="y", labelsize=12, colors=INK_DARK_GRAY)

    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLOR[c])
               for c in ["structured", "power-law / unstructured"]]
    leg = fig.legend(handles, ["structured (FEM)", "power-law / unstructured (web/social graph)"],
                      loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.14), fontsize=17)
    for text in leg.get_texts():
        text.set_color(INK_DARK_GRAY)
    fig.suptitle("Structured vs. unstructured matrices: does 1D locality pay off?",
                 y=1.05, fontsize=16.5, color=INK_DARK_GRAY)
    fig.tight_layout()
    savefig(fig, "11_structured_vs_unstructured")


# 15b: GFLOP/s vs. P, restricted to the 6 large matrices with a clear RCM
# P-scaling trend; cyclic/RCM/Block only.
def fig_partitioning_gflops_large_variant(cyclic_paths, rcm_paths, block_paths,
                                           fname, title):
    label = dict(PARTITION_LABEL, cyclic="1D cyclic")
    series = {
        "cyclic": prototype_series(cyclic_paths, PARTITION_COMM_TAG["cyclic"]),
        "rcm": prototype_series(rcm_paths, PARTITION_COMM_TAG["rcm"]),
        "block": prototype_series(block_paths, PARTITION_COMM_TAG["block"]),
    }
    keep = ["Queen_4147.mtx", "arabic-2005.mtx", "europe_osm.mtx",
            "mycielskian19.mtx", "nlpkkt240.mtx", "uk-2005.mtx"]
    matrices = [m for m in order_matrices_from_series(series) if m in keep]
    methods = ["cyclic", "rcm", "block"]
    ncols = 2
    n = len(matrices)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 3.05 * nrows), squeeze=False)
    for i, matrix in enumerate(matrices):
        ax = axes[i // ncols][i % ncols]
        for method in methods:
            d = series[method].get(matrix, {})
            ps = sorted(d.keys())
            if not ps:
                continue
            ax.plot(ps, [d[p]["gflops"] for p in ps], marker="o", markersize=6.5,
                     linewidth=2.3, color=PARTITION_COLOR[method], label=label[method])
        ax.set_title(short(matrix), fontsize=13, color=INK_DARK_GRAY)
        all_ps = sorted({p for d in series.values() for p in d.get(matrix, {})})
        if all_ps:
            ax.set_xticks(all_ps)
        ax.tick_params(axis="both", labelsize=11.5, colors=INK_DARK_GRAY)
        if i % ncols == 0:
            ax.set_ylabel("effective GFLOP/s (aggregate)", fontsize=12, color=INK_DARK_GRAY)
        if i // ncols == nrows - 1:
            ax.set_xlabel("P (ranks/GPUs)", fontsize=12, color=INK_DARK_GRAY)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(handles, legend_labels, loc="lower center", ncol=3,
                      bbox_to_anchor=(0.5, -0.04), fontsize=19)
    for text in leg.get_texts():
        text.set_color(INK_DARK_GRAY)
    fig.suptitle(title, y=1.02, fontsize=15.5, color=INK_DARK_GRAY)
    fig.tight_layout(h_pad=3.2)
    savefig(fig, fname)


# 17b: horizontal-bar comm-vs-compute breakdown, P=4, cuSPARSE, cyclic/RCM/Block
# only; mawi/webbase-2001 excluded (their bars dwarf the rest on a linear axis).
def fig_partitioning_breakdown_large_variant(cyclic_paths, rcm_paths, block_paths,
                                              fname, title):
    label = dict(PARTITION_LABEL, cyclic="1D cyclic")
    series = {
        "cyclic": prototype_series(cyclic_paths, PARTITION_COMM_TAG["cyclic"]),
        "rcm": prototype_series(rcm_paths, PARTITION_COMM_TAG["rcm"]),
        "block": prototype_series(block_paths, PARTITION_COMM_TAG["block"]),
    }
    exclude = {"mawi_201512020330.mtx", "webbase-2001.mtx"}
    matrices = [m for m in order_matrices_from_series(series) if m not in exclude]
    methods = ["cyclic", "rcm", "block"]
    y = np.arange(len(matrices))
    height = 0.24
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.9 * len(matrices))))

    xmax = 0.0
    for i, method in enumerate(methods):
        offset = (1 - i) * height
        compute_vals, comm_vals, missing = [], [], []
        for m in matrices:
            d = series[method].get(m, {}).get(4)
            if d is None:
                compute_vals.append(0.0)
                comm_vals.append(0.0)
                missing.append(True)
            else:
                compute_vals.append(d["compute_ms"])
                comm_vals.append(d["comm_ms"])
                missing.append(False)
                xmax = max(xmax, d["compute_ms"] + d["comm_ms"])
        ys = y + offset
        ax.barh(ys, compute_vals, height, color=PARTITION_COLOR[method],
                 label=PARTITION_LABEL[method])
        ax.barh(ys, comm_vals, height, left=compute_vals, color=PARTITION_COLOR[method],
                 hatch="//", edgecolor=SURFACE, linewidth=0.4, alpha=0.65)
        for yi, miss in zip(ys, missing):
            if miss:
                ax.text(0.01 * (xmax if xmax > 0 else 1), yi, "OOM", va="center", ha="left",
                        fontsize=9.5, color=RED, fontweight="bold")

    ax.set_xlim(0, xmax * 1.12 if xmax > 0 else 10)
    ax.set_yticks(y)
    ax.set_yticklabels([short(m) for m in matrices], fontsize=12.5, color=INK_DARK_GRAY)
    ax.tick_params(axis="x", labelsize=11.5, colors=INK_DARK_GRAY)
    ax.tick_params(axis="y", colors=INK_DARK_GRAY)
    ax.set_xlabel("time per SpMV, ms", fontsize=13, color=INK_DARK_GRAY)
    ax.set_title(title, fontsize=14.5, color=INK_DARK_GRAY)
    ax.invert_yaxis()
    fig.tight_layout()

    method_handles = [plt.Rectangle((0, 0), 1, 1, color=PARTITION_COLOR[m]) for m in methods]
    texture_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=INK_MUTED),
                        plt.Rectangle((0, 0), 1, 1, facecolor=INK_MUTED, hatch="//",
                                      edgecolor=SURFACE, alpha=0.65)]
    leg1 = ax.legend(method_handles, [label[m] for m in methods],
                      loc="center right", bbox_to_anchor=(0.99, 0.68),
                      fontsize=17, title="method")
    ax.add_artist(leg1)
    leg2 = ax.legend(texture_handles, ["compute", "communication"],
                      loc="center right", bbox_to_anchor=(0.99, 0.30),
                      fontsize=17)
    for leg in (leg1, leg2):
        leg.get_title().set_color(INK_DARK_GRAY)
        leg.get_title().set_fontsize(17)
        for text in leg.get_texts():
            text.set_color(INK_DARK_GRAY)
    savefig(fig, fname)


def main():
    style()
    weak = load_csv("weak_scaling_summary.csv", required=False)
    weak_sparse = load_csv("weak_scaling_sparse_summary.csv", required=False)
    weak_part = load_bonus_csv("weak_partitioned_summary.csv")
    weak_part_sparse = load_bonus_csv("weak_partitioned_summary_sparse.csv")
    large = load_csv("large_matrices_summary.csv", required=False)
    large_cyclic = sorted(glob.glob("outputs/large_matrices/strong_large_night-*.out"))

    if weak is not None:
        fig_weak_scaling(weak, weak_sparse, weak_part, weak_part_sparse)
    else:
        print("weak_scaling_summary.csv missing -- skipping fig 08", file=sys.stderr)

    if large is not None:
        fig_nccl_speedup(
            large, "05_nccl_vs_mpi_speedup_large",
            "1D ghost exchange: NCCL vs. MPI point-to-point, across P\n"
            "(large matrices, up to 1.02B nnz; thin lines: one per matrix)",
            figsize=(8, 4), legend_fontsize=17)
    else:
        print("large_matrices_summary.csv missing -- skipping fig 05", file=sys.stderr)

    if large is not None and large_cyclic:
        fig_structured_vs_unstructured(large, large_cyclic)
    else:
        print("large_matrices_summary.csv or outputs/large_matrices/ logs missing "
              "-- skipping fig 11", file=sys.stderr)

    large_partitioned = sorted(glob.glob(f"{BONUS_DIR}/outputs/large_matrices/large_night-*.out"))
    if large_cyclic and large_partitioned:
        fig_partitioning_gflops_large_variant(
            large_cyclic, large_partitioned, large_partitioned,
            "15b_partitioning_gflops_large_rcm_trend",
            "Partitioning strategies: GFLOP/s vs. P\n"
            "(matrices with a clear RCM P-scaling trend, cuSPARSE)")
        fig_partitioning_breakdown_large_variant(
            large_cyclic, large_partitioned, large_partitioned,
            "17b_partitioning_breakdown_large_log_horizontal",
            "Partitioning strategies: compute vs. communication, P=4, cuSPARSE\n"
            "(large matrices, log scale, mawi/webbase-2001 excluded)")
    else:
        print("no large-matrix partitioning-prototype logs found -- skipping figs 15b/17b", file=sys.stderr)

    print(f"\nAll figures written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
