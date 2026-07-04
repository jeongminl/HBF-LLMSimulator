"""
Compare paper_figure_readings.md (values read off the paper's figures) against
experiment_results.md (this repo's simulator output) and report error rates.

Error rate for a matched pair is defined as:

    error_rate = |paper_value - sim_value| / |sim_value|

i.e. the simulator's own output is treated as the reference ("actual"), and
the pixel-read paper value is the "prediction" being scored against it. Pairs
where either side is missing (NA / not present in one of the two tables) or
where the simulator value is exactly 0 (relative error undefined) are skipped
and reported separately as "unmatched".

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
    cell = cell.lstrip("~")
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


def compare_fig7(paper_df, sim_df):
    paper_df = paper_df.copy()
    paper_df["Model"] = paper_df["Model"].map(MODEL_MAP)
    paper_df["paper_value"] = paper_df["3-Year PEC (@8 GPU, online)"].apply(
        lambda c: extract_value(c, "abs")
    )

    sim_df = sim_df.copy()
    sim_df["sim_value"] = sim_df["3-Year PEC"].apply(lambda c: extract_value(c, "abs"))

    merged = paper_df.merge(sim_df, on=["Model", "Workload", "Memory"])
    merged["figure"] = "Figure 7"
    merged["group"] = merged["Memory"]
    merged["key"] = merged["Model"] + " | " + merged["Workload"] + " | " + merged["Memory"]
    return merged[["figure", "group", "key", "paper_value", "sim_value"]]


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
            sim_tables["Write Traffic and Endurance Assessment (Figure 7 Replication)"],
        ),
    ]
    all_pairs = pd.concat(frames, ignore_index=True)

    n_total = len(all_pairs)
    unmatched = all_pairs["paper_value"].isna() | all_pairs["sim_value"].isna() | (
        all_pairs["sim_value"] == 0
    )
    scored = all_pairs.loc[~unmatched].copy()
    scored["error_rate"] = (scored["paper_value"] - scored["sim_value"]).abs() / scored[
        "sim_value"
    ].abs()

    scored.to_csv(REPO_ROOT / args.out, index=False)

    if args.details:
        with pd.option_context("display.max_rows", None, "display.width", 140):
            print(scored[["figure", "group", "key", "paper_value", "sim_value", "error_rate"]])
        print()

    print(f"Matched & scored values: {len(scored)} / {n_total} total table cells")
    print(f"Skipped (NA / unresolved / zero-division): {unmatched.sum()}")
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
