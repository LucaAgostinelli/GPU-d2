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
GREEN = "#008300"
VIOLET = "#4a3aa7"
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
    "bcast": ORANGE,
    "ghost_mpi": BLUE,
    "ghost_nccl": AQUA,
    "checkerboard_mpi": VIOLET,
    "checkerboard_nccl": MAGENTA,
    "cpu_serial": RED,
}
COMM_LABEL = {
    "bcast": "naive bcast",
    "ghost_mpi": "1D ghost (MPI)",
    "ghost_nccl": "1D ghost (NCCL)",
    "checkerboard_mpi": "2D checkerboard (MPI)",
    "checkerboard_nccl": "2D checkerboard (NCCL)",
}
KERNEL_COLOR = {"cusparse": BLUE, "acc": ORANGE}

STRUCTURED = ["Queen_4147.mtx", "nlpkkt240.mtx"]
UNSTRUCTURED = ["webbase-2001.mtx", "com-Orkut.mtx"]
CATEGORY_COLOR = {"structured": BLUE, "power-law / unstructured": ORANGE}

# cyclic is the reference scheme; RCM/Fennel/Block get their own hues.
PARTITION_COLOR = {"cyclic": INK_SECONDARY, "rcm": BLUE, "fennel": GREEN, "block": YELLOW}
PARTITION_LABEL = {"cyclic": "cyclic (baseline)", "rcm": "RCM", "fennel": "Fennel/LDG", "block": "Block"}
PARTITION_COMM_TAG = {"cyclic": "ghost_nccl", "rcm": "rcm_nccl", "fennel": "fennel_nccl", "block": "block_nccl"}
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


# 1-3: strong scaling small multiples (time / speedup / GFLOP/s vs P).
def fig_strong_small_multiples(strong, value_col, ylabel, fname, title,
                                ideal=None, log=False):
    comms = ["bcast", "ghost_mpi", "ghost_nccl"]
    df = strong[(strong.kernel == "cusparse") & (strong.comm.isin(comms))]
    matrices = matrix_order(strong)
    matrices = [m for m in matrices if m in df.matrix.unique()]
    n = len(matrices)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
                              squeeze=False)
    for i, matrix in enumerate(matrices):
        ax = axes[i // ncols][i % ncols]
        sub = df[df.matrix == matrix]
        for comm in comms:
            s = sub[sub.comm == comm].sort_values("P")
            if s.empty:
                continue
            ax.plot(s.P, s[value_col], marker="o", color=COMM_COLOR[comm],
                    label=COMM_LABEL[comm])
        if ideal is not None:
            ps = sorted(sub.P.unique())
            ax.plot(ps, [ideal(p) for p in ps], linestyle="--", linewidth=1.2,
                     color=BASELINE_LINE, label="ideal", zorder=0)
        ax.set_title(short(matrix), fontsize=10)
        ax.set_xticks(sorted(sub.P.unique()))
        if log:
            ax.set_yscale("log")
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
        if i // ncols == nrows - 1:
            ax.set_xlabel("P (ranks/GPUs)")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, y=1.02, fontsize=14)
    fig.tight_layout()
    savefig(fig, fname)


# 4: kernel choice (cuSPARSE vs ACC) at P=4, comm=ghost_mpi.
def fig_kernel_choice(strong):
    df = strong[(strong.comm == "ghost_mpi") & (strong.P == 4)]
    matrices = [m for m in matrix_order(strong) if m in df.matrix.unique()]
    x = np.arange(len(matrices))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, kernel in enumerate(["cusparse", "acc"]):
        vals = [df[(df.matrix == m) & (df.kernel == kernel)].aggregate_effective_gflops
                .mean() for m in matrices]
        ax.bar(x + (i - 0.5) * width, vals, width, color=KERNEL_COLOR[kernel],
               label=kernel)
    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in matrices], rotation=30, ha="right")
    ax.set_ylabel("effective GFLOP/s (aggregate, P=4)")
    ax.set_title("Kernel choice at fixed communication strategy (1D ghost, MPI), P=4")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "04_kernel_choice_gflops")


# 5: NCCL vs MPI point-to-point speedup for 1D ghost exchange, across P.
# Generic df param: reused for both the small and large-matrix sweeps.
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


# 6: communication volume per rank, P=4 -- ghost/naive and checkerboard/naive.
def fig_comm_volume(comm_vol):
    df = comm_vol[comm_vol.P == 4].dropna(subset=["ghost_over_naive"])
    df = df.sort_values("ghost_over_naive")
    matrices = df.matrix.tolist()
    x = np.arange(len(matrices))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, df.ghost_over_naive, width, color=COMM_COLOR["ghost_mpi"],
           label="1D ghost / naive")
    ax.bar(x + width / 2, df.checkerboard_over_naive, width,
           color=COMM_COLOR["checkerboard_mpi"], label="2D checkerboard / naive")
    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in matrices], rotation=30, ha="right")
    ax.set_ylabel("fraction of naive's per-rank comm volume (bytes)")
    ax.set_title("Communication volume per rank relative to full-broadcast, P=4")
    ax.legend()
    for m in STRUCTURED + UNSTRUCTURED:
        if m in matrices:
            ax.get_xticklabels()[matrices.index(m)].set_color(
                CATEGORY_COLOR["structured" if m in STRUCTURED else "power-law / unstructured"])
    fig.tight_layout()
    savefig(fig, "06_comm_volume_vs_naive")


# 7: load-balance (max/avg NNZ-per-rank), 1D cyclic vs 2D checkerboard, P=4.
def fig_load_balance(load_bal):
    df = load_bal[(load_bal.P == 4) & (~load_bal.matrix.str.startswith("weak_"))]
    matrices = sorted(df.matrix.unique(),
                       key=lambda m: df[(df.matrix == m) & (df.family == "1d_cyclic")]
                       .max_over_avg.max() if not df[(df.matrix == m) & (df.family == "1d_cyclic")].empty else 0)
    x = np.arange(len(matrices))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, family in enumerate(["1d_cyclic", "2d_checkerboard"]):
        color = COMM_COLOR["ghost_mpi"] if family == "1d_cyclic" else COMM_COLOR["checkerboard_mpi"]
        vals = [df[(df.matrix == m) & (df.family == family)].max_over_avg
                .mean() for m in matrices]
        ax.bar(x + (i - 0.5) * width, vals, width, color=color,
               label="1D cyclic" if family == "1d_cyclic" else "2D checkerboard")
    ax.axhline(1.0, color=BASELINE_LINE, linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in matrices], rotation=30, ha="right")
    ax.set_ylabel("NNZ-per-rank imbalance (max / avg), P=4")
    ax.set_title("Load balance: 1D cyclic vs. 2D checkerboard partitioning, P=4")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "07_load_balance")


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


# 9: speedup over serial CPU baseline (only matrices with a baseline run).
def fig_baseline_speedup(baseline):
    cfg_cols = [c for c in baseline.columns if c.startswith("speedup_")]
    matrices = baseline.matrix.tolist()
    x = np.arange(len(matrices))
    n = len(cfg_cols)
    width = 0.8 / n
    colors = [BLUE, VIOLET, AQUA, MAGENTA]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, col in enumerate(cfg_cols):
        ax.bar(x + (i - (n - 1) / 2) * width, baseline[col], width,
               color=colors[i % len(colors)], label=col.replace("speedup_", ""))
    ax.axhline(1.0, color=BASELINE_LINE, linestyle="--", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in matrices], rotation=15, ha="right")
    ax.set_ylabel("speedup over 1-core serial CPU reference (log scale)")
    ax.set_title(f"Speedup over CPU baseline ({len(matrices)}/10 matrices measured so far)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "09_baseline_speedup")


# 10: comm vs compute breakdown, P=4, comm=ghost_nccl kernel=cusparse (parsed from raw RESULT lines).
def fig_comm_compute_breakdown():
    paths = sorted(glob.glob("outputs/strong_scaling/strong-*.out"))
    if not paths:
        print("no outputs/strong_scaling/strong-*.out found, skipping comm/compute breakdown", file=sys.stderr)
        return
    rows = parse_files(paths)
    rows = [r for r in rows if r.get("comm") == "ghost_nccl" and r.get("kernel") == "cusparse"
            and r.get("P") == 4]
    if not rows:
        print("no ghost_nccl/cusparse P=4 rows found, skipping comm/compute breakdown", file=sys.stderr)
        return
    by_matrix = {}
    for m in sorted(set(r["matrix"] for r in rows)):
        grp = [r for r in rows if r["matrix"] == m]
        chunks = chunk_repetitions(grp, 4)
        compute_reps = [max(r["compute_avg_ms"] for r in c) for c in chunks]
        comm_reps = [max(r["comm_avg_ms"] for r in c) for c in chunks]
        by_matrix[m] = (float(np.median(compute_reps)), float(np.median(comm_reps)))

    matrices = sorted(by_matrix, key=lambda m: sum(by_matrix[m]))
    compute = [by_matrix[m][0] for m in matrices]
    comm = [by_matrix[m][1] for m in matrices]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(matrices))
    ax.barh(y, compute, color=BLUE, label="compute (kernel)")
    ax.barh(y, comm, left=compute, color=ORANGE, label="communication (1D ghost, NCCL)")
    ax.set_yticks(y)
    ax.set_yticklabels([short(m) for m in matrices])
    ax.set_xlabel("time per SpMV call, ms (slowest rank, median over reps)")
    ax.set_title("Communication vs. compute breakdown, 1D ghost + NCCL + cuSPARSE, P=4")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "10_comm_compute_breakdown")


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


# 12: 1D ghost vs 2D checkerboard, both NCCL, faceted by P (P=3 skipped:
# no non-degenerate 2D grid at P=3).
def fig_checkerboard_vs_1d(strong):
    df = strong[(strong.kernel == "cusparse") &
                (strong.comm.isin(["ghost_nccl", "checkerboard_nccl"]))]
    Ps = [p for p in [1, 2, 4] if p in df.P.unique()]
    fig, axes = plt.subplots(1, len(Ps), figsize=(5.2 * len(Ps), 6.5))
    if len(Ps) == 1:
        axes = [axes]
    for ax, p in zip(axes, Ps):
        sub = df[df.P == p]
        rows = []
        for m in sub.matrix.unique():
            g = sub[(sub.matrix == m) & (sub.comm == "ghost_nccl")].T_median_ms
            c = sub[(sub.matrix == m) & (sub.comm == "checkerboard_nccl")].T_median_ms
            if g.empty or c.empty:
                continue
            rows.append((m, g.iloc[0] / c.iloc[0]))
        rows.sort(key=lambda r: r[1])
        matrices = [short(r[0]) for r in rows]
        speedups = [r[1] for r in rows]
        y = np.arange(len(matrices))
        colors = [COMM_COLOR["checkerboard_nccl"] if v >= 1 else COMM_COLOR["ghost_nccl"]
                  for v in speedups]
        ax.barh(y, speedups, color=colors)
        ax.axvline(1.0, color=BASELINE_LINE, linestyle="--", linewidth=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels(matrices, fontsize=8.5)
        ax.set_title(f"P={p}")
        ax.set_xlabel("speedup of 2D over 1D\n(T_ghost_nccl / T_checkerboard_nccl)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COMM_COLOR["checkerboard_nccl"]),
               plt.Rectangle((0, 0), 1, 1, color=COMM_COLOR["ghost_nccl"])]
    fig.legend(handles, ["2D checkerboard wins", "1D ghost wins"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Row-cyclic (1D) vs. checkerboard (2D) partitioning, NCCL transport", y=1.02)
    fig.tight_layout()
    savefig(fig, "12_checkerboard_vs_1d")


# 13: large-matrix strong scaling (bcast/ghost_mpi/ghost_nccl, cuSPARSE),
# reuses the fig 1-3 small-multiples layout.
def fig_large_matrix_scaling(large):
    fig_strong_small_multiples(
        large, "speedup", "speedup S(P) = T(1)/T(P)",
        "13_large_matrix_strong_scaling_speedup",
        "Interconnect sensitivity: strong scaling on large matrices "
        "(up to 1.02B nnz, baseline scheme)",
        ideal=lambda p: p)


# 14-15: partitioning-prototype GFLOP/s vs. P (cyclic/RCM/Fennel/Block),
# one panel per matrix, cuSPARSE only.
def fig_partitioning_gflops(cyclic_paths, rcm_paths, fennel_paths, block_paths,
                             fname, title):
    series = {
        "cyclic": prototype_series(cyclic_paths, PARTITION_COMM_TAG["cyclic"]),
        "rcm": prototype_series(rcm_paths, PARTITION_COMM_TAG["rcm"]),
        "fennel": prototype_series(fennel_paths, PARTITION_COMM_TAG["fennel"]),
        "block": prototype_series(block_paths, PARTITION_COMM_TAG["block"]),
    }
    matrices = order_matrices_from_series(series)
    n = len(matrices)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    for i, matrix in enumerate(matrices):
        ax = axes[i // ncols][i % ncols]
        for method in ["cyclic", "rcm", "fennel", "block"]:
            d = series[method].get(matrix, {})
            ps = sorted(d.keys())
            if not ps:
                continue
            ax.plot(ps, [d[p]["gflops"] for p in ps], marker="o",
                    color=PARTITION_COLOR[method], label=PARTITION_LABEL[method])
        ax.set_title(short(matrix), fontsize=10)
        all_ps = sorted({p for d in series.values() for p in d.get(matrix, {})})
        if all_ps:
            ax.set_xticks(all_ps)
        if i % ncols == 0:
            ax.set_ylabel("effective GFLOP/s (aggregate)")
        if i // ncols == nrows - 1:
            ax.set_xlabel("P (ranks/GPUs)")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, y=1.02, fontsize=14)
    fig.tight_layout()
    savefig(fig, fname)


# 15b: GFLOP/s vs. P, restricted to the 6 large matrices with a clear RCM
# P-scaling trend; cyclic/RCM/Block only (no Fennel).
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


# 16-17: partitioning comm-vs-compute breakdown, P=4, cuSPARSE.
# Missing method/matrix combos (OOM at this P) are drawn as an "OOM" label.
def fig_partitioning_breakdown(cyclic_paths, rcm_paths, fennel_paths, block_paths,
                                fname, title):
    series = {
        "cyclic": prototype_series(cyclic_paths, PARTITION_COMM_TAG["cyclic"]),
        "rcm": prototype_series(rcm_paths, PARTITION_COMM_TAG["rcm"]),
        "fennel": prototype_series(fennel_paths, PARTITION_COMM_TAG["fennel"]),
        "block": prototype_series(block_paths, PARTITION_COMM_TAG["block"]),
    }
    matrices = order_matrices_from_series(series)
    methods = ["cyclic", "rcm", "fennel", "block"]
    x = np.arange(len(matrices))
    width = 0.19
    fig, ax = plt.subplots(figsize=(max(11, 1.1 * len(matrices)), 6))

    ymax = 0.0
    for i, method in enumerate(methods):
        offset = (i - 1.5) * width
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
                ymax = max(ymax, d["compute_ms"] + d["comm_ms"])
        xs = x + offset
        ax.bar(xs, compute_vals, width, color=PARTITION_COLOR[method],
                label=PARTITION_LABEL[method])
        ax.bar(xs, comm_vals, width, bottom=compute_vals, color=PARTITION_COLOR[method],
                hatch="//", edgecolor=SURFACE, linewidth=0.4, alpha=0.65)
        for xi, miss in zip(xs, missing):
            if miss:
                ax.text(xi, 0, "OOM", rotation=90, va="bottom", ha="center",
                        fontsize=7, color=RED)

    ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in matrices], rotation=30, ha="right")
    ax.set_ylabel("time per SpMV call, ms (slowest rank, median over reps)")
    ax.set_title(title)

    method_handles = [plt.Rectangle((0, 0), 1, 1, color=PARTITION_COLOR[m])
                       for m in methods]
    texture_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=INK_MUTED),
                        plt.Rectangle((0, 0), 1, 1, facecolor=INK_MUTED, hatch="//",
                                      edgecolor=SURFACE, alpha=0.65)]
    leg1 = ax.legend(method_handles, [PARTITION_LABEL[m] for m in methods],
                      loc="upper left", fontsize=8.5, title="method")
    ax.add_artist(leg1)
    ax.legend(texture_handles, ["compute", "communication"],
              loc="upper right", fontsize=8.5)
    fig.tight_layout()
    savefig(fig, fname)


# 17b: horizontal-bar variant of fig 17, cyclic/RCM/Block only; mawi/webbase-2001
# excluded (their bars dwarf the rest on a linear axis).
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
    strong = load_csv("strong_scaling_summary.csv")
    weak = load_csv("weak_scaling_summary.csv", required=False)
    weak_sparse = load_csv("weak_scaling_sparse_summary.csv", required=False)
    weak_part = load_bonus_csv("weak_partitioned_summary.csv")
    weak_part_sparse = load_bonus_csv("weak_partitioned_summary_sparse.csv")
    baseline = load_csv("baseline_summary.csv", required=False)
    load_bal = load_csv("load_balance_summary.csv", required=False)
    comm_vol = load_csv("comm_volume_summary.csv", required=False)
    large = load_csv("large_matrices_summary.csv", required=False)
    large_cyclic = sorted(glob.glob("outputs/large_matrices/strong_large_night-*.out"))

    if strong is None:
        print("strong_scaling_summary.csv is required for most figures -- "
              "run scripts/analyze_strong_scaling.py first", file=sys.stderr)
        sys.exit(1)

    fig_strong_small_multiples(
        strong, "T_median_ms", "time per SpMV (ms)", "01_strong_scaling_time",
        "Strong scaling: execution time vs. P (1D ghost, cuSPARSE)")
    fig_strong_small_multiples(
        strong, "speedup", "speedup S(P) = T(1)/T(P)", "02_strong_scaling_speedup",
        "Strong scaling: speedup vs. P", ideal=lambda p: p)
    fig_strong_small_multiples(
        strong, "aggregate_effective_gflops", "effective GFLOP/s (aggregate)",
        "03_strong_scaling_gflops", "Strong scaling: aggregate GFLOP/s vs. P")

    fig_kernel_choice(strong)
    fig_nccl_speedup(
        strong, "05_nccl_vs_mpi_speedup_small",
        "1D ghost exchange: NCCL vs. MPI point-to-point, across P\n"
        "(small matrices; thin lines: one per matrix)")
    if large is not None:
        fig_nccl_speedup(
            large, "05_nccl_vs_mpi_speedup_large",
            "1D ghost exchange: NCCL vs. MPI point-to-point, across P\n"
            "(large matrices, up to 1.02B nnz; thin lines: one per matrix)",
            figsize=(8, 4), legend_fontsize=17)
    else:
        print("large_matrices_summary.csv missing -- skipping fig 05 (large)", file=sys.stderr)

    if comm_vol is not None:
        fig_comm_volume(comm_vol)
    if load_bal is not None:
        fig_load_balance(load_bal)
    if weak is not None:
        fig_weak_scaling(weak, weak_sparse, weak_part, weak_part_sparse)
    if baseline is not None:
        fig_baseline_speedup(baseline)

    fig_comm_compute_breakdown()

    if large is not None and large_cyclic:
        fig_structured_vs_unstructured(large, large_cyclic)
    else:
        print("large_matrices_summary.csv or outputs/large_matrices/ logs missing "
              "-- skipping fig 11", file=sys.stderr)

    fig_checkerboard_vs_1d(strong)

    if large is not None:
        fig_large_matrix_scaling(large)
    else:
        print("large_matrices_summary.csv missing -- skipping fig 13", file=sys.stderr)

    # Partitioning-prototype figures (RCM/Fennel/Block vs. baseline cyclic),
    # small (10) and large (big_matrices/) matrix passes.
    small_cyclic = sorted(glob.glob("outputs/strong_scaling/strong-*.out"))
    small_rcm = sorted(glob.glob(f"{BONUS_DIR}/outputs/rcm/rcm_cusparse-*.out"))
    small_fennel = sorted(glob.glob(f"{BONUS_DIR}/outputs/fennel/fennel_cusparse-*.out"))
    small_block = sorted(glob.glob(f"{BONUS_DIR}/outputs/block/block_cusparse-*.out"))
    if small_cyclic and (small_rcm or small_fennel or small_block):
        fig_partitioning_gflops(
            small_cyclic, small_rcm, small_fennel, small_block,
            "14_partitioning_gflops_small",
            "Partitioning strategies: GFLOP/s vs. P, small matrices (cuSPARSE)")
        fig_partitioning_breakdown(
            small_cyclic, small_rcm, small_fennel, small_block,
            "16_partitioning_breakdown_small",
            "Partitioning strategies: compute vs. communication, P=4, cuSPARSE (small matrices)")
    else:
        print("no small-matrix partitioning-prototype logs found -- skipping figs 14/16", file=sys.stderr)

    large_partitioned = sorted(glob.glob(f"{BONUS_DIR}/outputs/large_matrices/large_night-*.out"))
    if large_cyclic and large_partitioned:
        fig_partitioning_gflops(
            large_cyclic, large_partitioned, large_partitioned, large_partitioned,
            "15_partitioning_gflops_large",
            "Partitioning strategies: GFLOP/s vs. P, large matrices (up to 1.02B nnz, cuSPARSE)")
        fig_partitioning_gflops_large_variant(
            large_cyclic, large_partitioned, large_partitioned,
            "15b_partitioning_gflops_large_rcm_trend",
            "Partitioning strategies: GFLOP/s vs. P\n"
            "(matrices with a clear RCM P-scaling trend, cuSPARSE)")
        fig_partitioning_breakdown(
            large_cyclic, large_partitioned, large_partitioned, large_partitioned,
            "17_partitioning_breakdown_large",
            "Partitioning strategies: compute vs. communication, P=4, cuSPARSE (large matrices)")
        fig_partitioning_breakdown_large_variant(
            large_cyclic, large_partitioned, large_partitioned,
            "17b_partitioning_breakdown_large_log_horizontal",
            "Partitioning strategies: compute vs. communication, P=4, cuSPARSE\n"
            "(large matrices, log scale, mawi/webbase-2001 excluded)")
    else:
        print("no large-matrix partitioning-prototype logs found -- skipping figs 15/17", file=sys.stderr)

    print(f"\nAll figures written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
