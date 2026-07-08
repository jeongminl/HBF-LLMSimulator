"""
Compare paper_figure_readings.md (values read off the paper's figures) against
experiment_results.md (this repo's simulator output) and report error rates.

Error rate for a matched pair is defined as:

    error_rate = |paper_value - sim_value| / |paper_value|

i.e. the paper's (ground-truth) reading is treated as the reference
("actual"), and the simulator's output is the "prediction" being scored
against it. Pairs where either side is missing (NA / not present in one of
the two tables) or where the paper value is exactly 0 (relative error
undefined) are skipped and reported separately as "unmatched".

Usage:
    python compare_error_rates.py
    python compare_error_rates.py --details   # also print every matched pair

Outputs:
    - error_rates_detail.csv   one row per matched value, with error rate
    - error_rates_by_figure.png   grouped bar chart of median & max error
      rate per figure
    - Printed summary: average/median/max error rate per figure, and per
      "data group" within each figure (e.g. Figure 6 -> TPS Ratio vs Batch
      Ratio).
"""
import argparse
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
PAPER_MD = REPO_ROOT / "paper_figure_readings.md"
SIM_MD = REPO_ROOT / "experiment_results.md"

MODEL_MAP = {"Llama3": "llama3_405B", "Llama4": "llama4_maverick"}

CELL_RE = re.compile(r"([-\d.]+)\s*\(([-\d.]+)x\)")
GPU_COLS = ["1 GPU", "2 GPU", "4 GPU", "8 GPU", "16 GPU"]


def is_unextractable(row):
    """Cells whose PAPER reading cannot be reliably extracted, so their error
    rate is noise, not a sim-vs-paper divergence. Excluded from the scored
    error and reported separately (like NA / zero-division cells).

    Currently: the Fig-4 CONV / CONV+ / llama4_maverick low-GPU bars. In the
    paper's Figure 4 these bars are visually near-identical / overlapping at low
    GPU counts, so the pixel/vector read is unreliable AND (for CONV) provably
    self-inconsistent (every CONV Fig-4 reading implies a decode step over the
    0.1 s SLO, contradicting the paper's own Fig-3). The simulator is physically
    defensible here; the large "error" is a paper-extraction artifact. Scope:
    CONV/llama4 at 1/2/4 GPU, and CONV+/llama4 at 1/2 GPU. See
    PAPER_INCONSISTENCIES.md's Fig-4 CONV/llama4 entry.
    """
    if row["figure"] != "Figure 4" or not row["key"].startswith("llama4_maverick |"):
        return False
    low_gpu = row["key"].endswith("| 1 GPU") or row["key"].endswith("| 2 GPU")
    if "| CONV |" in row["key"]:                      # CONV: 1/2/4 GPU
        return low_gpu or row["key"].endswith("| 4 GPU")
    if "| CONV+ |" in row["key"]:                     # CONV+: 1/2 GPU
        return low_gpu
    return False


# --------------------------------------------------------------------------
# Markdown table parsing
# --------------------------------------------------------------------------
def parse_markdown_tables(path):
    """Return {section_title: DataFrame} for every '## N. Title' + table
    block in a markdown file. Cells are left as raw strings; numeric
    interpretation happens later per-figure since formats differ."""
    text = path.read_text()
    lines = text.splitlines()

    tables = {}
    current_title = None
    table_lines = []

    def flush():
        nonlocal table_lines
        if current_title and table_lines:
            header = [c.strip() for c in table_lines[0].strip("|").split("|")]
            rows = []
            for line in table_lines[2:]:  # skip header + '---' separator
                cells = [c.strip() for c in line.strip("|").split("|")]
                if len(cells) == len(header):
                    rows.append(cells)
            tables[current_title] = pd.DataFrame(rows, columns=header)
        table_lines = []

    for line in lines:
        m = re.match(r"^##\s*\d+\.\s*(.+)$", line.strip())
        if m:
            flush()
            current_title = m.group(1).strip()
            continue
        if line.strip().startswith("|"):
            table_lines.append(line.strip())
        elif table_lines:
            flush()
    flush()
    return tables


def extract_value(cell, mode):
    """mode='abs' -> the plain leading number; mode='ratio' -> the (X.XXx)
    factor; returns None for 'NA' or unparsable cells."""
    cell = cell.strip()
    if cell.upper() == "NA" or cell == "":
        return None
    m = CELL_RE.search(cell)
    if m:
        return float(m.group(1)) if mode == "abs" else float(m.group(2))
    cell = cell.lstrip("~").rstrip("†").strip()
    cell = re.sub(r"\s*\(FF\)\s*$", "", cell)
    try:
        return float(cell)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Per-figure comparison builders -> each returns a long DataFrame with
# columns: figure, group, key (human-readable), paper_value, sim_value
# --------------------------------------------------------------------------
def compare_wide_gpu_table(paper_df, sim_df, figure_name, value_mode):
    """Shared logic for Fig 3 & Fig 4: wide GPU-count tables joined on
    (Model, Workload, Memory Config); 'group' = Memory Config."""
    paper_long = paper_df.melt(
        id_vars=["Model", "Workload", "Memory Config"],
        value_vars=GPU_COLS,
        var_name="GPU",
        value_name="paper_raw",
    )
    paper_long["paper_value"] = paper_long["paper_raw"].apply(
        lambda c: extract_value(c, "abs")
    )
    paper_long["Model"] = paper_long["Model"].map(MODEL_MAP)

    sim_long = sim_df.melt(
        id_vars=["Model", "Workload", "Memory Config"],
        value_vars=GPU_COLS,
        var_name="GPU",
        value_name="sim_raw",
    )
    sim_long["sim_value"] = sim_long["sim_raw"].apply(lambda c: extract_value(c, value_mode))

    merged = paper_long.merge(sim_long, on=["Model", "Workload", "Memory Config", "GPU"])
    merged["figure"] = figure_name
    merged["group"] = merged["Memory Config"]
    merged["key"] = (
        merged["Model"] + " | " + merged["Workload"] + " | "
        + merged["Memory Config"] + " | " + merged["GPU"]
    )
    return merged[["figure", "group", "key", "paper_value", "sim_value"]]


def compare_fig5(paper_df, sim_df):
    categories = ["Attention", "FFN", "KV Write", "Communication", "Others"]

    def prep(df, value_suffix):
        long = df.melt(
            id_vars=["Model", "Workload", "Memory", "GPUs"],
            value_vars=categories,
            var_name="Category",
            value_name=f"raw_{value_suffix}",
        )
        long[f"val_{value_suffix}"] = (
            long[f"raw_{value_suffix}"].str.rstrip("%").apply(lambda c: extract_value(c, "abs"))
        )
        return long

    paper_long = prep(paper_df, "paper")
    paper_long["Model"] = paper_long["Model"].map(MODEL_MAP)
    sim_long = prep(sim_df, "sim")

    merged = paper_long.merge(
        sim_long, on=["Model", "Workload", "Memory", "GPUs", "Category"]
    )
    merged["figure"] = "Figure 5"
    merged["group"] = merged["Category"]
    merged["key"] = (
        merged["Model"] + " | " + merged["Workload"] + " | " + merged["Memory"]
        + " | " + merged["GPUs"] + " GPU | " + merged["Category"]
    )
    merged = merged.rename(columns={"val_paper": "paper_value", "val_sim": "sim_value"})
    return merged[["figure", "group", "key", "paper_value", "sim_value"]]


def compare_fig6(paper_df, sim_df):
    # sim table's Metric is "Batch Size" / "TPS/GPU"; paper table's Metric is
    # "Batch Ratio" / "TPS Ratio" (both compared as ratios, i.e. the (X.XXx)
    # factor in the sim table).
    metric_map = {"TPS Ratio": "TPS/GPU", "Batch Ratio": "Batch Size"}

    # The sim writer emits the SLO label "Offline (24h)" (run_experiments.py:1573)
    # while the paper readings use "offline"; without normalizing both frames'
    # SLO column to a common spelling, the merge below silently drops every
    # offline row (12 pairs) instead of matching them.
    def _normalize_slo(value):
        s = str(value).strip()
        return "offline" if s.lower().startswith("offline") else s

    paper_df = paper_df.copy()
    sim_df = sim_df.copy()
    paper_df["SLO"] = paper_df["SLO"].apply(_normalize_slo)
    sim_df["SLO"] = sim_df["SLO"].apply(_normalize_slo)

    paper_long = paper_df.melt(
        id_vars=["Model", "Memory", "SLO", "Metric"],
        value_vars=["4 GPU", "8 GPU", "16 GPU"],
        var_name="GPU",
        value_name="paper_raw",
    )
    paper_long["paper_value"] = paper_long["paper_raw"].apply(lambda c: extract_value(c, "abs"))
    paper_long["Model"] = paper_long["Model"].map(MODEL_MAP)
    paper_long["SimMetric"] = paper_long["Metric"].map(metric_map)

    sim_long = sim_df.melt(
        id_vars=["Model", "Memory", "SLO", "Metric"],
        value_vars=["4 GPU", "8 GPU", "16 GPU"],
        var_name="GPU",
        value_name="sim_raw",
    )
    sim_long["sim_value"] = sim_long["sim_raw"].apply(lambda c: extract_value(c, "ratio"))

    merged = paper_long.merge(
        sim_long,
        left_on=["Model", "Memory", "SLO", "SimMetric", "GPU"],
        right_on=["Model", "Memory", "SLO", "Metric", "GPU"],
    )
    merged["figure"] = "Figure 6"
    merged["group"] = merged["Metric_x"]
    merged["key"] = (
        merged["Model"] + " | " + merged["Memory"] + " | " + merged["SLO"]
        + " | " + merged["GPU"] + " | " + merged["Metric_x"]
    )
    return merged[["figure", "group", "key", "paper_value", "sim_value"]]


def _import_run_experiments():
    """Import run_experiments.py as a module (guarded by its own __main__
    check, so importing is side-effect-free) so Fig-7 can reuse its
    compute_pec() -- the single source of truth for the PEC formula, shared
    with experiment_results.md Section 5 and fig7_pec.png. Mirrors the
    pattern already used by fig7_online.py / check_fig7_key.py /
    check_fig7_gpu1.py in this same directory."""
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import run_experiments as R

    return R


def _fig7_series_long(paper_df, checkpoint_path, online):
    """Build the long (figure/group/key/paper_value/sim_value) frame for one
    Fig-7 series (online or offline) across the FULL grid: every (model,
    workload, mem in {HBF, HBF+}, gpu in {1, 8, 16}) cell -- 36 cells per
    series when the checkpoint is present.

    Sim value: run_experiments.compute_pec() applied directly to the
    matching row of `checkpoint_path` (checkpoint_results.json for online --
    the same 0.1s-SLO sweep Figures 3-6 draw from; checkpoint_pec_results.json
    for offline -- the dedicated unconstrained-batch PEC sweep). Going
    straight to the checkpoint (rather than experiment_results.md's own Fig-7
    table) recovers GPU=1/16, which that report table doesn't tabulate
    (Section 5 there is @8-GPU only).

    Paper value: the matching Section-5 column in paper_figure_readings.md's
    full grid (HBF (online) / HBF+ (online) / HBF (offline) / HBF+
    (offline)).

    Returns an empty frame (0 rows) if `checkpoint_path` doesn't exist yet --
    this is how the offline series gracefully no-ops until the recovery
    sweep writes checkpoint_pec_results.json, instead of erroring."""
    empty = pd.DataFrame(columns=["figure", "group", "key", "paper_value", "sim_value"])

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return empty

    import json

    R = _import_run_experiments()
    with open(checkpoint_path) as f:
        chk_rows = json.load(f)

    paper_df = paper_df.copy()
    paper_df["Model"] = paper_df["Model"].map(MODEL_MAP)
    paper_df["GPUs"] = paper_df["GPUs"].astype(str)

    mem_col = {"HBF": "HBF (online)" if online else "HBF (offline)",
               "HBF+": "HBF+ (online)" if online else "HBF+ (offline)"}

    rows = []
    for model in MODEL_MAP.values():
        for workload in R.PEC_WORKLOADS:
            for mem in R.PEC_MEM_TYPES:
                for gpu in R.PEC_GPUS:
                    sim_row = next(
                        (r for r in chk_rows if r["model"] == model and r["workload"] == workload
                         and r["memory"] == mem and r["gpus"] == gpu),
                        None,
                    )
                    pec_info = R.compute_pec(sim_row)
                    if pec_info is None:
                        # Infeasible cell (max_batch 0, or PEC geometry never
                        # emitted) -- skip, don't count as a zero/NA pair.
                        continue

                    paper_match = paper_df[
                        (paper_df["Model"] == model) & (paper_df["Workload"] == workload)
                        & (paper_df["GPUs"] == str(gpu))
                    ]
                    paper_value = (
                        extract_value(paper_match.iloc[0][mem_col[mem]], "abs")
                        if not paper_match.empty else None
                    )

                    rows.append({
                        "figure": "Figure 7",
                        "group": f"{mem} ({'online' if online else 'offline'})",
                        "key": f"{model} | {workload} | {mem} | {gpu} GPU | "
                               f"{'online' if online else 'offline'}",
                        "paper_value": paper_value,
                        "sim_value": pec_info["pec"],
                    })

    return pd.DataFrame(rows, columns=["figure", "group", "key", "paper_value", "sim_value"]) \
        if rows else empty


def compare_fig7(paper_df):
    """Full-grid Fig-7 comparison: every (model, workload, mem, gpu) cell,
    ONLINE (from checkpoint_results.json, always present) and OFFLINE (from
    checkpoint_pec_results.json, produced by the offline-PEC recovery sweep
    -- absent until that sweep finishes, in which case this contributes 0
    offline cells and the online-only result still flows through normally).
    36 online cells + up to 36 offline cells = up to 72 total, vs. the
    previous 12 (GPU=8, online-only) cells."""
    online_long = _fig7_series_long(
        paper_df, REPO_ROOT / "checkpoint_results.json", online=True
    )
    offline_long = _fig7_series_long(
        paper_df, REPO_ROOT / "checkpoint_pec_results.json", online=False
    )
    return pd.concat([online_long, offline_long], ignore_index=True)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
# Reference palette (see dataviz skill, references/palette.md): two
# non-adjacent categorical slots for good CVD separation.
MEDIAN_COLOR = "#2a78d6"  # slot 1, blue
MAX_COLOR = "#eb6834"  # slot 8, orange
MUTED_INK = "#898781"
AXIS_INK = "#c3c2b7"
GRIDLINE = "#e1e0d9"
PRIMARY_INK = "#0b0b0b"


def plot_error_rates(per_figure, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = list(per_figure.index)
    medians = per_figure["median_error_%"].tolist()
    maxes = per_figure["max_error_%"].tolist()

    x = range(len(figures))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_med = ax.bar(
        [i - width / 2 for i in x], medians, width, label="Median error %",
        color=MEDIAN_COLOR, zorder=3,
    )
    bars_max = ax.bar(
        [i + width / 2 for i in x], maxes, width, label="Max error %",
        color=MAX_COLOR, zorder=3,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(figures, color=PRIMARY_INK)
    ax.set_ylabel("Error rate (%)", color=PRIMARY_INK)
    ax.set_title("Paper vs. Simulator Error Rate by Figure", color=PRIMARY_INK)

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine_name, spine in ax.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(AXIS_INK)
    ax.tick_params(colors=MUTED_INK)

    for bars in (bars_med, bars_max):
        ax.bar_label(bars, fmt="%.1f", padding=2, color=MUTED_INK, fontsize=9)

    ax.legend(frameon=False, labelcolor=PRIMARY_INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", action="store_true", help="print every matched pair")
    ap.add_argument("--out", default="error_rates_detail.csv")
    ap.add_argument("--plot-out", default="error_rates_by_figure.png")
    args = ap.parse_args()

    paper_tables = parse_markdown_tables(PAPER_MD)
    sim_tables = parse_markdown_tables(SIM_MD)

    frames = [
        compare_wide_gpu_table(
            paper_tables["Maximum Per-GPU Batch Size (Figure 3)"],
            sim_tables["Maximum Per-GPU Batch Size (Figure 3 Replication)"],
            "Figure 3",
            value_mode="abs",
        ),
        compare_wide_gpu_table(
            paper_tables["System Throughput (Figure 4)"],
            sim_tables["System Throughput (Figure 4 Replication)"],
            "Figure 4",
            value_mode="abs",
        ),
        compare_fig5(
            paper_tables["Runtime Performance Breakdown (Figure 5)"],
            sim_tables["Runtime Performance Breakdown (Figure 5 Replication)"],
        ),
        compare_fig6(
            paper_tables["SLO Sensitivity Analysis (Figure 6)"],
            sim_tables["SLO Sensitivity Analysis (Figure 6 Replication)"],
        ),
        compare_fig7(
            paper_tables["Write Traffic and Endurance (Figure 7)"],
        ),
    ]
    all_pairs = pd.concat(frames, ignore_index=True)

    n_total = len(all_pairs)
    unmatched = all_pairs["paper_value"].isna() | all_pairs["sim_value"].isna() | (
        all_pairs["paper_value"] == 0
    )
    # Cells with unextractable paper readings (see is_unextractable): excluded
    # from the scored error like NA cells, since a paper-extraction artifact is
    # not a sim-vs-paper divergence.
    unextractable = all_pairs.apply(is_unextractable, axis=1)
    scored = all_pairs.loc[~unmatched & ~unextractable].copy()
    scored["error_rate"] = (scored["paper_value"] - scored["sim_value"]).abs() / scored[
        "paper_value"
    ].abs()

    scored.to_csv(REPO_ROOT / args.out, index=False)

    if args.details:
        with pd.option_context("display.max_rows", None, "display.width", 140):
            print(scored[["figure", "group", "key", "paper_value", "sim_value", "error_rate"]])
        print()

    print(f"Matched & scored values: {len(scored)} / {n_total} total table cells")
    print(f"Skipped (NA / unresolved / zero-division): {unmatched.sum()}")
    print(f"Excluded (unextractable paper readings — Fig-4 CONV/CONV+ llama4 low-GPU): "
          f"{(unextractable & ~unmatched).sum()}")
    print()

    print("=" * 70)
    print("AVERAGE / MEDIAN / MAX ERROR RATE PER FIGURE")
    print("=" * 70)
    per_figure = scored.groupby("figure")["error_rate"].agg(["mean", "median", "max", "count"])
    per_figure["mean"] = (per_figure["mean"] * 100).round(2)
    per_figure["median"] = (per_figure["median"] * 100).round(2)
    per_figure["max"] = (per_figure["max"] * 100).round(2)
    per_figure.columns = ["mean_error_%", "median_error_%", "max_error_%", "n"]
    print(per_figure.to_string())
    print()

    print("=" * 70)
    print("AVERAGE / MEDIAN / MAX ERROR RATE PER DATA GROUP WITHIN EACH FIGURE")
    print("=" * 70)
    per_group = scored.groupby(["figure", "group"])["error_rate"].agg(["mean", "median", "max", "count"])
    per_group["mean"] = (per_group["mean"] * 100).round(2)
    per_group["median"] = (per_group["median"] * 100).round(2)
    per_group["max"] = (per_group["max"] * 100).round(2)
    per_group.columns = ["mean_error_%", "median_error_%", "max_error_%", "n"]
    print(per_group.to_string())
    print()
    print(f"Full per-value detail written to: {args.out}")

    try:
        plot_error_rates(per_figure, REPO_ROOT / args.plot_out)
        print(f"Median/max error rate chart written to: {args.plot_out}")
    except ImportError:
        print("matplotlib not importable; skipping chart generation")


if __name__ == "__main__":
    main()
