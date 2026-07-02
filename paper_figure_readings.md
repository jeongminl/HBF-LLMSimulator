# Paper Figure Readings (Ground-Truth Extraction)

Values in this file were extracted directly from the figures of *"Exploring
High-Bandwidth Flash for Modern LLM Inference: Opportunities and Challenges"*
via calibrated pixel analysis (600 DPI render, axis-gridline calibration,
per-series color matching). They are **not** taken from the paper's text —
only from reading the plotted bars/markers themselves.

Conventions:
- `NA` = the paper's figure does not show a value here (either the
  configuration is infeasible at that GPU count — e.g. HBM4/CONV/CONV+ never
  show 1-2 GPU bars — or pixel extraction could not resolve the point because
  of marker/line overlap).
- `(FF)` in the prose below means "forward-filled": the paper's own figure
  convention (Fig. 3 caption) states that if a larger GPU count does not
  increase the plotted value, no new bar segment is drawn — i.e. the value is
  identical to the last GPU count that *was* drawn. These are real readings,
  not guesses, and are written out explicitly in the tables below.
- A handful of Fig. 4 / Fig. 6 points sit exactly where two series' markers
  visually overlap; these are linearly interpolated from neighboring SLO/GPU
  points and are marked with `~`.
- Model name mapping used for comparison: `Llama3` -> `llama3_405B`,
  `Llama4` -> `llama4_maverick`.

## 1. Maximum Per-GPU Batch Size (Figure 3)

Calibrated against the paper's own printed "w/ 8 GPUs" anchor values
(HBM4 @ 8 GPU per workload); recovered anchors matched the printed ones to
within ~1-2%.

| Model | Workload | Memory Config | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 16 GPU |
|---|---|---|---|---|---|---|---|
| Llama3 | SHORT | HBM4 | NA | NA | 104.4 | 195.5 | 195.5 |
| Llama3 | SHORT | HBF | 176.2 | 257.6 | 257.6 | 257.6 | 257.6 |
| Llama3 | SHORT | HBF+ | 329.4 | 329.4 | 329.4 | 329.4 | 329.4 |
| Llama3 | SHORT | CONV | NA | NA | 32.6 | 81.1 | 81.1 |
| Llama3 | SHORT | CONV+ | NA | NA | 57.8 | 98.6 | 98.6 |
| Llama3 | MID | HBM4 | NA | NA | 32.5 | 62.0 | 76.1 |
| Llama3 | MID | HBF | 70.0 | 146.9 | 146.9 | 146.9 | 146.9 |
| Llama3 | MID | HBF+ | 116.1 | 189.9 | 189.9 | 189.9 | 189.9 |
| Llama3 | MID | CONV | NA | NA | 12.2 | 35.6 | 35.6 |
| Llama3 | MID | CONV+ | NA | NA | 21.4 | 42.9 | 42.9 |
| Llama3 | LONG | HBM4 | NA | NA | 1.8 | 3.8 | 4.7 |
| Llama3 | LONG | HBF | 5.1 | 11.5 | 14.8 | 16.0 | 16.0 |
| Llama3 | LONG | HBF+ | 7.1 | 14.5 | 17.5 | 19.0 | 19.0 |
| Llama3 | LONG | CONV | NA | NA | 0.6 | 2.5 | 2.5 |
| Llama3 | LONG | CONV+ | NA | NA | 1.3 | 3.1 | 3.1 |
| Llama4 | SHORT | HBM4 | NA | NA | 256.7 | 463.7 | 592.5 |
| Llama4 | SHORT | HBF | NA | 1218.1 | 1526.3 | 1687.3 | 1687.3 |
| Llama4 | SHORT | HBF+ | 854.7 | 854.7 | 854.7 | 854.7 | 854.7 |
| Llama4 | SHORT | CONV | 49.7 | 49.7 | 118.7 | 321.1 | 417.7 |
| Llama4 | SHORT | CONV+ | 54.3 | 54.3 | 173.9 | 390.1 | 477.5 |
| Llama4 | MID | HBM4 | NA | NA | 84.5 | 152.7 | 192.1 |
| Llama4 | MID | HBF | 267.9 | 449.7 | 566.3 | 626.9 | 626.9 |
| Llama4 | MID | HBF+ | 352.7 | 561.8 | 686.0 | 745.1 | 745.1 |
| Llama4 | MID | CONV | 36.1 | 36.1 | 57.3 | 102.7 | 134.5 |
| Llama4 | MID | CONV+ | 40.6 | 40.6 | 72.4 | 125.4 | 157.3 |
| Llama4 | LONG | HBM4 | NA | NA | 16.7 | 31.3 | 39.0 |
| Llama4 | LONG | HBF | 96.4 | 109.7 | 127.4 | 140.4 | 145.0 |
| Llama4 | LONG | HBF+ | 113.1 | 131.1 | 151.5 | 164.9 | 169.5 |
| Llama4 | LONG | CONV | 17.9 | 17.9 | 17.9 | 24.4 | 28.2 |
| Llama4 | LONG | CONV+ | 20.7 | 20.7 | 24.7 | 28.5 | 32.8 |

Note: Llama4-SHORT/HBF+ is the figure's one "SRAM bound" (✕-marked) bar — flat
across all GPU counts by construction.

## 2. System Throughput (Figure 4)

Calibrated against the paper's own printed 8-GPU-HBM4 TPS anchors per
workload (3.3K / 1.8K / 147 / 19K / 6.3K / 1.3K); recovered HBM4@8GPU read
0.997-1.008x the anchor across all six panels (~1% calibration error).

| Model | Workload | Memory Config | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 16 GPU |
|---|---|---|---|---|---|---|---|
| Llama3 | SHORT | HBM4 | NA | NA | 3032.7 | 3290.1 | 3290.1 |
| Llama3 | SHORT | HBF | 1656.6 | 2448.6 | 2445.3 | 2448.6 | 2445.3 |
| Llama3 | SHORT | HBF+ | 3243.9 | 3243.9 | 3276.9 | ~3255 | ~3255 |
| Llama3 | SHORT | CONV | NA | NA | 214.5 | 693.0 | 693.0 |
| Llama3 | SHORT | CONV+ | NA | NA | 511.5 | 953.7 | 953.7 |
| Llama3 | MID | HBM4 | NA | NA | 1224.0 | 1794.6 | 1899.0 |
| Llama3 | MID | HBF | 675.0 | 1454.4 | 1454.4 | 1463.4 | 1454.4 |
| Llama3 | MID | HBF+ | 1180.8 | 1936.8 | 1936.8 | 1947.6 | ~1750 |
| Llama3 | MID | CONV | NA | NA | 55.8 | 284.4 | 284.4 |
| Llama3 | MID | CONV+ | NA | NA | 198.0 | 421.2 | 421.2 |
| Llama3 | LONG | HBM4 | NA | NA | 72.8 | 146.6 | 156.4 |
| Llama3 | LONG | HBF | 47.5 | 113.6 | 145.0 | 156.8 | ~163 |
| Llama3 | LONG | HBF+ | 73.2 | 145.0 | 176.8 | 189.5 | 190.0 |
| Llama3 | LONG | CONV | NA | NA | 2.1 | 19.7 | 19.7 |
| Llama3 | LONG | CONV+ | NA | NA | 12.6 | 30.3 | 30.3 |
| Llama4 | SHORT | HBM4 | NA | NA | 9823.0 | 18943.0 | 20900.0 |
| Llama4 | SHORT | HBF | 5491.0 | NA | 14497.0 | 16454.0 | 16454.0 |
| Llama4 | SHORT | HBF+ | 8227.0 | 12008.0 | 15675.0 | ~16311 | 18544.0 |
| Llama4 | SHORT | CONV | NA | 589.0 | 266.0 | 2413.0 | 3401.0 |
| Llama4 | SHORT | CONV+ | NA | 589.0 | 1634.0 | 3781.0 | 4731.0 |
| Llama4 | MID | HBM4 | NA | NA | 3439.8 | 6281.1 | 7396.2 |
| Llama4 | MID | HBF | 2576.7 | 4359.6 | 5550.3 | ~6021 | 6237.0 |
| Llama4 | MID | HBF+ | 3528.0 | 5625.9 | 6860.7 | 7471.8 | ~7180 |
| Llama4 | MID | CONV | 44.1 | 88.2 | 283.5 | 781.2 | 1102.5 |
| Llama4 | MID | CONV+ | 409.5 | 453.6 | 693.0 | 1234.8 | 1556.1 |
| Llama4 | LONG | HBM4 | NA | NA | 1036.1 | 1296.1 | 1511.9 |
| Llama4 | LONG | HBF | 927.0 | 1054.3 | 1232.4 | 1383.2 | 1406.6 |
| Llama4 | LONG | HBF+ | 1121.9 | 1290.9 | 1489.8 | 1621.1 | 1670.5 |
| Llama4 | LONG | CONV | 124.8 | 133.9 | 152.1 | 187.2 | 227.5 |
| Llama4 | LONG | CONV+ | 210.6 | 218.4 | 241.8 | 276.9 | 321.1 |

## 3. Runtime Performance Breakdown (Figure 5)

The paper's Fig. 5 only plots **HBM4** and **HBF+**, and only for the **MID**
and **LONG** workloads (no SHORT, no HBF/CONV/CONV+ breakdown). Values are %
of decode time; rows may sum to slightly less than 100% due to sub-pixel
rounding at segment boundaries.

| Model | Workload | Memory | GPUs | Attention | FFN | KV Write | Communication | Others |
|---|---|---|---|---|---|---|---|---|
| Llama3 | MID | HBM4 | 4 | 40.8 | 50.6 | 0.0 | 5.5 | 1.6 |
| Llama3 | MID | HBM4 | 8 | 49.9 | 29.5 | 0.0 | 14.6 | 4.5 |
| Llama3 | MID | HBM4 | 16 | 54.3 | 34.0 | 0.0 | 7.7 | 2.5 |
| Llama3 | LONG | HBM4 | 4 | 42.2 | 53.8 | 0.0 | 2.1 | 1.9 |
| Llama3 | LONG | HBM4 | 8 | 67.5 | 25.5 | 0.0 | 5.1 | 1.9 |
| Llama3 | LONG | HBM4 | 16 | 71.2 | 22.1 | 0.0 | 4.8 | 1.9 |
| Llama3 | MID | HBF+ | 4 | 54.3 | 31.7 | 9.3 | 2.1 | 2.6 |
| Llama3 | MID | HBF+ | 8 | 54.3 | 31.7 | 9.3 | 2.1 | 2.6 |
| Llama3 | MID | HBF+ | 16 | 54.3 | 31.7 | 9.3 | 2.1 | 2.6 |
| Llama3 | LONG | HBF+ | 4 | 77.5 | 13.5 | 6.4 | 0.0 | 2.6 |
| Llama3 | LONG | HBF+ | 8 | 81.9 | 6.8 | 7.1 | 2.3 | 1.9 |
| Llama3 | LONG | HBF+ | 16 | 81.9 | 6.8 | 7.1 | 2.3 | 1.9 |
| Llama4 | MID | HBM4 | 4 | 32.8 | 63.0 | 0.0 | 2.3 | 1.9 |
| Llama4 | MID | HBM4 | 8 | 60.4 | 36.1 | 0.0 | 0.0 | 3.5 |
| Llama4 | MID | HBM4 | 16 | 71.2 | 19.8 | 0.0 | 6.8 | 2.2 |
| Llama4 | LONG | HBM4 | 4 | 52.7 | 44.0 | 0.0 | 0.0 | 3.3 |
| Llama4 | LONG | HBM4 | 8 | 65.5 | 31.4 | 0.0 | 0.0 | 3.1 |
| Llama4 | LONG | HBM4 | 16 | 76.6 | 18.0 | 0.0 | 3.7 | 1.7 |
| Llama4 | MID | HBF+ | 4 | 66.8 | 18.3 | 13.0 | 0.0 | 1.9 |
| Llama4 | MID | HBF+ | 8 | 72.7 | 11.5 | 13.6 | 0.0 | 2.2 |
| Llama4 | MID | HBF+ | 16 | 72.7 | 7.8 | 13.7 | 3.9 | 1.9 |
| Llama4 | LONG | HBF+ | 4 | 75.3 | 16.1 | 6.7 | 0.0 | 1.9 |
| Llama4 | LONG | HBF+ | 8 | 82.1 | 8.9 | 7.1 | 0.0 | 1.9 |
| Llama4 | LONG | HBF+ | 16 | 84.4 | 5.0 | 7.5 | 0.0 | 3.1 |

## 4. SLO Sensitivity Analysis (Figure 6)

Measured in the **LONG** workload only. Both axes in the paper are
normalized to 8-GPU HBM4 for that workload, so values here are **ratios**
(directly comparable to the `(X.XXx)` figures in `experiment_results.md`),
not absolute units. `~` marks points linearly interpolated across a
marker/line overlap.

| Model | Memory | SLO | Metric | 4 GPU | 8 GPU | 16 GPU |
|---|---|---|---|---|---|---|
| Llama3 | HBM4 | 0.05s | TPS Ratio | 0.419 | 0.855 | 0.898 |
| Llama3 | HBM4 | 0.1s | TPS Ratio | 0.419 | 0.840 | ~0.897 |
| Llama3 | HBM4 | 0.2s | TPS Ratio | 0.422 | 0.842 | 0.895 |
| Llama3 | HBM4 | offline | TPS Ratio | 0.419 | 0.843 | 0.898 |
| Llama3 | HBF | 0.05s | TPS Ratio | 0.625 | 0.791 | 0.791 |
| Llama3 | HBF | 0.1s | TPS Ratio | 0.831 | 0.892 | 0.892 |
| Llama3 | HBF | 0.2s | TPS Ratio | 0.924 | ~0.930 | 0.945 |
| Llama3 | HBF | offline | TPS Ratio | 0.972 | 0.971 | 0.971 |
| Llama3 | HBF+ | 0.05s | TPS Ratio | 0.814 | 0.983 | 0.983 |
| Llama3 | HBF+ | 0.1s | TPS Ratio | 1.012 | 1.090 | ~1.060 |
| Llama3 | HBF+ | 0.2s | TPS Ratio | 1.110 | 1.142 | 1.142 |
| Llama3 | HBF+ | offline | TPS Ratio | 1.157 | 1.157 | 1.157 |
| Llama4 | HBM4 | 0.05s | TPS Ratio | ~0.674 | 0.849 | 0.971 |
| Llama4 | HBM4 | 0.1s | TPS Ratio | 0.674 | 0.834 | 0.988 |
| Llama4 | HBM4 | 0.2s | TPS Ratio | 0.673 | 0.842 | ~0.990 |
| Llama4 | HBM4 | offline | TPS Ratio | 0.673 | 0.843 | ~0.990 |
| Llama4 | HBF | 0.05s | TPS Ratio | 0.666 | 0.779 | 0.855 |
| Llama4 | HBF | 0.1s | TPS Ratio | 0.799 | 0.881 | 0.916 |
| Llama4 | HBF | 0.2s | TPS Ratio | 0.892 | 0.936 | 0.951 |
| Llama4 | HBF | offline | TPS Ratio | 0.936 | 0.959 | 0.962 |
| Llama4 | HBF+ | 0.05s | TPS Ratio | 0.828 | 0.956 | 1.026 |
| Llama4 | HBF+ | 0.1s | TPS Ratio | 0.968 | 1.055 | 1.084 |
| Llama4 | HBF+ | 0.2s | TPS Ratio | 1.064 | 1.108 | 1.119 |
| Llama4 | HBF+ | offline | TPS Ratio | 1.108 | 1.128 | 1.131 |
| Llama3 | HBM4 | 0.05s | Batch Ratio | 0.23 | 0.64 | 0.87 |
| Llama3 | HBM4 | 0.1s | Batch Ratio | 0.23 | 0.64 | 0.87 |
| Llama3 | HBM4 | 0.2s | Batch Ratio | 0.23 | 0.64 | 0.87 |
| Llama3 | HBM4 | offline | Batch Ratio | 0.23 | 0.64 | 0.87 |
| Llama3 | HBF | 0.05s | Batch Ratio | 1.05 | 1.40 | 1.40 |
| Llama3 | HBF | 0.1s | Batch Ratio | 3.14 | 3.37 | 3.37 |
| Llama3 | HBF | 0.2s | Batch Ratio | 7.21 | 7.38 | 7.38 |
| Llama3 | HBF | offline | Batch Ratio | 14.59 | 14.59 | 14.59 |
| Llama3 | HBF+ | 0.05s | Batch Ratio | 1.40 | 1.74 | 1.74 |
| Llama3 | HBF+ | 0.1s | Batch Ratio | 3.72 | 4.07 | 4.07 |
| Llama3 | HBF+ | 0.2s | Batch Ratio | 8.20 | 8.78 | 8.78 |
| Llama3 | HBF+ | offline | Batch Ratio | 16.86 | 16.86 | 16.86 |
| Llama4 | HBM4 | 0.05s | Batch Ratio | 0.23 | 0.64 | 0.87 |
| Llama4 | HBM4 | 0.1s | Batch Ratio | 0.23 | 0.64 | 0.87 |
| Llama4 | HBM4 | 0.2s | Batch Ratio | 0.23 | 0.64 | 0.87 |
| Llama4 | HBM4 | offline | Batch Ratio | 0.23 | 0.64 | 0.87 |
| Llama4 | HBF | 0.05s | Batch Ratio | 1.28 | 1.51 | 1.69 |
| Llama4 | HBF | 0.1s | Batch Ratio | 3.26 | 3.60 | 3.72 |
| Llama4 | HBF | 0.2s | Batch Ratio | 7.56 | 7.91 | 8.02 |
| Llama4 | HBF | offline | Batch Ratio | 15.06 | 15.47 | 15.70 |
| Llama4 | HBF+ | 0.05s | Batch Ratio | 1.57 | 1.86 | 1.98 |
| Llama4 | HBF+ | 0.1s | Batch Ratio | 3.90 | 4.30 | 4.36 |
| Llama4 | HBF+ | 0.2s | Batch Ratio | 8.84 | 9.24 | 9.24 |
| Llama4 | HBF+ | offline | Batch Ratio | 17.38 | 17.79 | 22.38 |

Note: the HBM4 batch-size bars are extremely short (near-zero) at this
scale, so the fixed ~4px bar-outline detection offset dominates their
reading — expect a larger relative error on the HBM4/Batch-Ratio group than
elsewhere in this figure; this is a known pixel-reading limitation, not a
transcription error.

## 5. Write Traffic and Endurance (Figure 7)

`experiment_results.md`'s Fig. 7 replication table reports one steady-state
value per (Model, Workload, Memory) with no explicit GPU count — its
TPS/GPU and Batch Size columns match this repo's own Fig. 3/4 tables at
**8 GPUs**, so that is the operating point used here. Only the **online**
(continuous-batching, steady-state) series applies to that comparison; the
paper's separate "offline" series has no counterpart in
`experiment_results.md`. Units: 3-year P/E-cycle count (raw count, matching
`experiment_results.md`'s scale — the paper plots this as x10^3).

| Model | Workload | Memory | 3-Year PEC (@8 GPU, online) |
|---|---|---|---|
| Llama3 | SHORT | HBF | 172600 |
| Llama3 | SHORT | HBF+ | 196000 |
| Llama3 | MID | HBF | 240500 |
| Llama3 | MID | HBF+ | 273900 |
| Llama3 | LONG | HBF | 189300 |
| Llama3 | LONG | HBF+ | 198200 |
| Llama4 | SHORT | HBF | 441000 |
| Llama4 | SHORT | HBF+ | 425400 |
| Llama4 | MID | HBF | 394200 |
| Llama4 | MID | HBF+ | 410900 |
| Llama4 | LONG | HBF | 200400 |
| Llama4 | LONG | HBF+ | 206000 |
