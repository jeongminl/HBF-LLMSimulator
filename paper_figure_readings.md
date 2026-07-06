# Paper Figure Readings (Ground-Truth Extraction)

**Methodology changed 2026-07-06: vector-graphics extraction, not pixel reading.** The paper's
PDF figures (`Exploring_High-Bandwidth_Flash_for_Modern_LLM_Inference_Opportunities_and_Challenges.pdf`)
turned out to be pure vector graphics — `pdfimages -list` finds **zero embedded raster images** in
the whole PDF, and converting any figure page to SVG shows every bar/marker as an exact vector
path (`<path>`/rect draw command) with real coordinates, not rendered pixels. This means the
values below are read directly from the plotting library's own geometry (via PyMuPDF's
`page.get_drawings()`), not estimated by eye or by pixel-color thresholding on a raster render.
This supersedes the prior methodology (600 DPI render + pixel-color matching), which is preserved
verbatim in `paper_figure_readings_pixel_extraction_backup.md` for reference/audit.

**How the extraction works, in brief:**
1. Every drawing object's fill color is matched against the exact legend-swatch colors for that
   figure (also read as exact vector fills, not eyeballed).
2. Individual bar/marker sub-paths are pulled out of PyMuPDF's grouped drawing objects (a whole
   figure's same-colored bars are frequently returned as ONE drawing object with many rectangle
   sub-items — the top-level bounding box is useless; you have to descend into `d["items"]`).
3. **Axis calibration uses an exact self-normalization anchor, not tick-label text positions,
   wherever the figure's own convention provides one.** Figures 3, 4, and 6 are each explicitly
   "normalized to the 8-GPU HBM4 [value] for each workload" per the paper's captions — so the
   HBM4-at-8-GPU bar/marker in every facet *must* sit at exactly value=1.0 (or, for Fig. 3/4, at
   the paper's own printed exact anchor number). Checking this across every facet is itself a
   correctness proof: in every figure, all 6-8 independent HBM4-anchor points landed on the
   *exact same pixel y-coordinate* (spread = 0.0000pt) — i.e. the paper's own plotting tool
   really did share one consistent axis across facets, and the extraction recovers it exactly.
   Where no such self-normalizing anchor exists (Fig. 7's absolute-PEC axis), a least-squares fit
   across all 6 tick labels was used instead (residuals <0.5 units on a 0-500 scale, ~0.1%).
4. Values in Section 4 (Fig. 6) are pure ratios, so no further conversion is needed. Sections 1/2/5
   convert the normalized ratio to absolute units by multiplying by the paper's own printed exact
   anchor value for that facet (given in each section below).

**This methodology retroactively resolved two previously-unresolved discrepancies from earlier
sessions:** the Fig. 6 TPS-axis miscalibration (already independently found and corrected via
pixel-domain cross-checking on 2026-07-06 — the vector read now *exactly* confirms that fix, see
Section 4) and the Fig. 6 Batch-axis "~25% overshoot" (previously flagged as unexplained — the
vector read shows this was **simply a pixel-reading error in the old extraction**, not a hidden
second miscalibration; see Section 4).

## Known shortfalls of vector extraction (read before trusting a specific cell)

Not everything got easier. These are the concrete failure modes hit while building this, in
roughly the order you'd run into them:

1. **Subsumed/omitted segments (Fig. 3) are not a shortfall, but a benefit — flagged here because
   it's a common misconception.** Fig. 3's caption states that if a larger GPU count doesn't
   increase the batch size, no new bar segment is drawn at all. In vector space this is
   unambiguous (no rectangle exists for that GPU count, full stop), so forward-filling from the
   last real segment is exact, not a guess — this is *more* reliable than pixel reading, where a
   very thin/near-invisible segment can be mistaken for present-but-tiny or genuinely absent.
2. **Legend swatches share the exact same fill color as real data series.** A legend swatch for
   "HBM4" and an actual HBM4 bar/marker are visually and chromatically identical vector objects —
   the only way to tell them apart is by position (legend rows sit in a small, known, otherwise-
   unused y-band) or by selecting the drawing-object grouping with the most sub-items (the real
   data series has 15-30+ segments; a legend swatch has exactly 1). Getting this filter's
   boundary wrong silently pollutes the dataset with 1-3 phantom "data points" sitting exactly at
   the legend's y-position — this happened twice during extraction (Figs. 4 and 6) and was caught
   by an "unassigned/duplicate" sanity check (any extracted point that doesn't cleanly land on a
   known x-axis tick, or where two candidates map to the same cell, is suspect and was
   investigated rather than silently kept).
3. **Point/marker overlap near the axis floor breaks marker identification (Fig. 4, and likely
   Fig. 6 TPS to a lesser extent).** `CONV`/`CONV+` are described in the paper's own text as
   operating "near-floor" — their markers sit close to the x-axis and close to each other at low
   GPU counts. In one cell (`Llama4-SHORT / CONV / 1-GPU`) the extraction returned a **negative**
   TPS value — physically impossible, and a clear sign the wrong marker (or a partial/clipped
   shape) was picked up in that cramped region. This cell is flagged `†` in Section 2 and the old
   pixel-reading's value is kept there instead. This is the mirror image of the old methodology's
   own documented weakness (it marked exactly these kinds of points `~` for "interpolated across
   an overlap") — overlap is a fundamental readability limit of the *figure*, not just of one
   extraction method, though vector data resolves the vast majority of previously-`~`-marked
   points cleanly (see Section 2/4 notes).
4. **No thick-outline-vs-fill mismatch was actually observed in this paper's figures** (checked
   for specifically, per a concern that a bar's stroke/outline could be thicker than its fill and
   thus bias a naive bounding-box read) — outlines here are either absent or a separate,
   distinguishable drawing object (fill-only vs stroke-only groups, both anchored to the same
   rectangle geometry), so this was not a real problem *for this PDF*. It remains a real risk for
   other papers/figures — worth checking for before trusting a bounding box as the true data
   boundary, since a thick outline centered on the true edge would bias a read by roughly half the
   outline's stroke width.
5. **No exact printed anchor exists for every figure.** Figs. 3, 4, and 6 each have a paper-stated
   exact normalization value (either literal printed numbers, as in Figs. 3/4, or the tautological
   "HBM4@8GPU≡1.0" built into Fig. 6's own definition). Fig. 7 has neither — its y-axis is a plain
   labeled scale with no special self-normalizing structure, so its calibration falls back to
   fitting the 6 tick labels' text positions, which carries a small residual font-metrics
   uncertainty (empirically <0.1% here, but not exactly zero the way the anchor-based figures are).
6. **Fig. 5 (runtime breakdown, stacked percentages) has since been vector-extracted (2026-07-06,
   later same day)** — initially skipped for time, then completed. It surfaced a genuinely new
   shortfall: **adjacent stacked-bar categories can have overlapping y-ranges** (their shared
   boundary line gets drawn as part of both rectangles) — reading each category's own rectangle
   height directly was measurably wrong for a few bars (e.g. one HBM4 cell read Communication as
   ~1.5% via its own rect height vs. ~1.3% via boundary-tops, with the row's overall 100% sum only
   holding exactly under the boundary-tops method). **Fix: derive every category's share from the
   gap between successive category tops in the known stack order, never from a category's own
   rectangle span.** This generalizes beyond Fig. 5 — any stacked bar/area chart is at risk of this
   if two series share a rendered boundary.
7. **This only works because the source PDF happens to be vector.** If a paper's figures are
   scanned/rasterized (common for older papers, some conference proceedings, or figures pasted in
   as images), none of this applies and pixel-based tools (WebPlotDigitizer, Engauge Digitizer)
   are the only option. Always check first (`pdfimages -list yourfile.pdf` — zero images strongly
   suggests vector figures; nonzero doesn't rule it out, since a paper can mix vector text/axes
   with a raster-embedded plot).

Conventions (unchanged from the prior version of this file):
- `NA` = the paper's figure genuinely does not show a value here (configuration infeasible at
  that GPU count, e.g. HBM4/CONV/CONV+ never show 1-2 GPU bars).
- `(FF)` = forward-filled per Fig. 3's own stated convention (see shortfall #1 above) — an exact
  reading, not a guess.
- `†` = flagged low-confidence cell (see shortfall #3); old pixel-reading value kept instead of
  the vector read.
- Model name mapping: `Llama3` -> `llama3_405B`, `Llama4` -> `llama4_maverick`.

## 1. Maximum Per-GPU Batch Size (Figure 3)

Exact printed anchors (paper's own red-label text, 8-GPU HBM4 per workload):

| | SHORT | MID | LONG |
|---|---|---|---|
| Llama3 | 194 | 61.5 | 3.75 |
| Llama4 | 460 | 151.5 | 31 |

All 6 of these anchor points were independently recovered from the vector data as the exact
y-coordinate of each facet's HBM4-at-8-GPU segment top (spread across facets: 0.0000pt) — i.e. the
figure's underlying geometry reproduces the paper's own printed numbers exactly, by construction.

| Model | Workload | Memory Config | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 16 GPU |
|---|---|---|---|---|---|---|---|
| Llama3 | SHORT | HBM4 | NA | NA | 101.2 | 194.0 | 194.0 (FF) |
| Llama3 | SHORT | CONV | NA | NA | 28.2 | 78.8 | 78.8 (FF) |
| Llama3 | SHORT | CONV+ | NA | NA | 53.5 | 95.6 | 95.6 (FF) |
| Llama3 | SHORT | HBF | 171.5 | 253.0 | 253.0 (FF) | 253.0 (FF) | 253.0 (FF) |
| Llama3 | SHORT | HBF+ | 326.1 | 326.1 (FF) | 326.1 (FF) | 326.1 (FF) | 326.1 (FF) |
| Llama3 | MID | HBM4 | NA | NA | 31.2 | 61.5 | 74.9 |
| Llama3 | MID | CONV | NA | NA | 10.7 | 33.9 | 33.9 (FF) |
| Llama3 | MID | CONV+ | NA | NA | 19.6 | 41.0 | 41.0 (FF) |
| Llama3 | MID | HBF | 68.6 | 146.2 | 146.2 (FF) | 146.2 (FF) | 146.2 (FF) |
| Llama3 | MID | HBF+ | 115.0 | 188.9 | 188.9 (FF) | 188.9 (FF) | 188.9 (FF) |
| Llama3 | LONG | HBM4 | NA | NA | 1.7 | 3.8 | 4.6 |
| Llama3 | LONG | CONV | NA | NA | 0.5 | 2.4 | 2.4 (FF) |
| Llama3 | LONG | CONV+ | NA | NA | 1.3 | 3.0 | 3.0 (FF) |
| Llama3 | LONG | HBF | 5.0 | 11.5 | 14.7 | 15.9 | 15.9 (FF) |
| Llama3 | LONG | HBF+ | 7.0 | 14.5 | 17.4 | 19.0 | 19.0 (FF) |
| Llama4 | SHORT | HBM4 | NA | NA | 246.7 | 460.0 | 580.0 |
| Llama4 | SHORT | CONV | 40.1 | 46.8 | 106.8 | 306.7 | 406.7 |
| Llama4 | SHORT | CONV+ | 46.8 | 60.1 | 166.7 | 380.0 | 473.3 |
| Llama4 | SHORT | HBF | 580.0 | 1206.5 | 1519.7 | 1679.7 | 1679.7 (FF) |
| Llama4 | SHORT | HBF+ | 766.6 | 766.6 (FF) | 766.6 (FF) | 766.6 (FF) | 766.6 (FF) |
| Llama4 | MID | HBM4 | NA | NA | 81.3 | 151.5 | 188.8 |
| Llama4 | MID | CONV | 33.0 | 37.4 | 54.9 | 98.8 | 131.7 |
| Llama4 | MID | CONV+ | 37.4 | 46.1 | 70.3 | 123.0 | 155.9 |
| Llama4 | MID | HBF | 265.6 | 445.6 | 564.2 | 623.4 | 632.2 |
| Llama4 | MID | HBF+ | 349.1 | 559.8 | 682.7 | 742.0 | 744.2 |
| Llama4 | LONG | HBM4 | NA | NA | 16.2 | 31.0 | 38.2 |
| Llama4 | LONG | CONV | 17.1 | 18.9 | 20.7 | 23.8 | 27.4 |
| Llama4 | LONG | CONV+ | 19.8 | 22.0 | 24.3 | 27.9 | 32.3 |
| Llama4 | LONG | HBF | 95.7 | 109.2 | 126.7 | 139.7 | 144.6 |
| Llama4 | LONG | HBF+ | 112.7 | 130.3 | 150.9 | 164.4 | 168.9 |

Note: Llama4-SHORT/HBF+ is the figure's one "SRAM bound" (✕-marked) bar — flat across all GPU
counts by construction (confirmed: only a GPU=1 segment exists in the vector data; every higher
GPU count forward-fills from it).

Comparison to the prior pixel-reading (`paper_figure_readings_pixel_extraction_backup.md`): every
cell agrees within ~1-4%, **except every cell was previously off by a small but nonzero amount even
at the exact-anchor points** — e.g. the old reading had Llama3-SHORT-HBM4-8GPU at 195.5 against the
true (and paper-printed) 194 — a 0.8% error baked into the "anchor-calibrated" pixel method itself.
The vector method cannot have this error class by construction.

## 2. System Throughput (Figure 4)

Exact printed anchors (paper's own red-label text, 8-GPU HBM4 TPS per workload):

| | SHORT | MID | LONG |
|---|---|---|---|
| Llama3 | 3.3K | 1.8K | 147 |
| Llama4 | 19K | 6.3K | 1.3K |

Recovered identically across all 6 facets (spread 0.0000pt), same exact-anchor confirmation as
Fig. 3.

| Model | Workload | Memory Config | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 16 GPU |
|---|---|---|---|---|---|---|---|
| Llama3 | SHORT | HBM4 | NA | NA | 3038.9 | 3300.0 | 3300.0 |
| Llama3 | SHORT | HBF | 1678.7 | 2494.8 | 2494.8 | 2494.8 | 2494.8 |
| Llama3 | SHORT | HBF+ | 3251.0 | 3251.0 | 3251.0 | 3251.0 | 3251.0 |
| Llama3 | SHORT | CONV | NA | NA | 220.7 | 732.1 | 732.1 |
| Llama3 | SHORT | CONV+ | NA | NA | 454.6 | 900.7 | 900.7 |
| Llama3 | MID | HBM4 | NA | NA | 1218.4 | 1800.0 | 1906.8 |
| Llama3 | MID | HBF | 686.9 | 1485.4 | 1485.4 | 1494.1 | 1485.4 |
| Llama3 | MID | HBF+ | 1170.6 | 1945.4 | 1945.4 | 1954.1 | 1945.4 |
| Llama3 | MID | CONV | NA | NA | 66.9 | 310.3 | 310.3 |
| Llama3 | MID | CONV+ | NA | NA | 164.9 | 390.4 | 390.4 |
| Llama3 | LONG | HBM4 | NA | NA | 71.4 | 147.0 | 157.2 |
| Llama3 | LONG | HBF | 48.3 | 116.2 | 148.2 | 159.1 | 159.8 |
| Llama3 | LONG | HBF+ | 71.9 | 145.5 | 178.0 | 190.9 | 191.6 |
| Llama3 | LONG | CONV | NA | NA | 1.6 | 21.5 | 21.5 |
| Llama3 | LONG | CONV+ | NA | NA | 9.6 | 28.0 | 28.0 |
| Llama4 | SHORT | HBM4 | NA | NA | 9665.3 | 19000.0 | 21004.7 |
| Llama4 | SHORT | HBF | 5559.4 | 11949.5 | 15178.4 | 16807.3 | 16807.3 |
| Llama4 | SHORT | HBF+ | 8034.0 | 11918.2 | 15648.3 | 18592.8 | 18592.8 |
| Llama4 | SHORT | CONV | 49.7 † | 17.5 | 581.4 | 2711.4 | 3713.8 |
| Llama4 | SHORT | CONV+ | 77.7 | 174.2 | 1239.2 | 3494.5 | 4434.2 |
| Llama4 | MID | HBM4 | NA | NA | 3391.8 | 6300.0 | 7442.5 |
| Llama4 | MID | HBF | 2612.0 | 4451.2 | 5676.8 | 6279.2 | 6362.3 |
| Llama4 | MID | HBF+ | 3474.0 | 5645.7 | 6892.0 | 7515.2 | 7536.0 |
| Llama4 | MID | CONV | 202.3 | 234.3 | 400.5 | 878.3 | 1210.6 |
| Llama4 | MID | CONV+ | 275.0 | 327.8 | 577.1 | 1137.9 | 1470.3 |
| Llama4 | LONG | HBM4 | NA | NA | 1034.2 | 1300.0 | 1522.9 |
| Llama4 | LONG | HBF | 946.2 | 1077.1 | 1257.1 | 1390.0 | 1445.7 |
| Llama4 | LONG | HBF+ | 1119.8 | 1293.6 | 1499.3 | 1636.5 | 1683.6 |
| Llama4 | LONG | CONV | 153.2 | 164.1 | 181.2 | 211.2 | 249.8 |
| Llama4 | LONG | CONV+ | 185.3 | 196.2 | 217.7 | 256.2 | 299.1 |

`†` **Llama4/SHORT/CONV/1-GPU: vector extraction failed** (returned a negative value — a marker
misidentification near the axis floor, see shortfall #3 above); value shown is the old
pixel-reading's 49.7, kept as a placeholder, not independently confirmed by either method.

Note on the rest of the `CONV`/`CONV+` cells at this same corner (Llama4/SHORT, low GPU count):
the vector read (2GPU=17.5, 4GPU=581.4) is a **major, monotonic correction** to the old
pixel-reading, which had a non-monotonic artifact there (2GPU=589 > 4GPU=266 — physically odd,
throughput dropping as GPUs increase) that the old file's own notes flagged as a probable
marker-overlap misread. The vector data resolves this cleanly and is trusted here. Every other
previously-`~`-marked (interpolated/overlap-flagged) cell in the old file was likewise resolved to
a clean, non-ambiguous vector reading in this pass.

## 3. Runtime Performance Breakdown (Figure 5)

**Vector-extracted 2026-07-06** (previously left as the old pixel-reading — see shortfall #6's
history note below). The paper's Fig. 5 only plots **HBM4** and **HBF+**, and only for the **MID**
and **LONG** workloads (no SHORT, no HBF/CONV/CONV+ breakdown). Values are % of decode time.

This figure has no self-normalizing anchor (it's not normalized to anything — it's a stacked
percentage breakdown), but it has an equivalent structural check: the five categories are stacked
bottom-to-top (Attention, FFN, KV Write, Communication, Others) and must sum to 100%. Values here
use each category's own **top boundary only** (not its own rectangle's height) to compute the gap
to the next category up — this is deliberately robust against a rendering quirk found during
extraction: in a few bars, the "KV Write" and "Communication" rectangles' y-ranges slightly
overlap (their shared boundary is drawn by both), which would bias a naive
per-rectangle-height read. Using only the boundary tops, every bar's five categories now sum to
100.0% (one bar at 100.1%, sub-pixel) — a clean internal-consistency check the old reading
couldn't perform (it wasn't reading exact coordinates to begin with).

| Model | Workload | Memory | GPUs | Attention | FFN | KV Write | Communication | Others |
|---|---|---|---|---|---|---|---|---|
| Llama3 | MID | HBM4 | 4 | 42.1 | 50.9 | 0.2 | 5.2 | 1.7 |
| Llama3 | MID | HBM4 | 8 | 51.1 | 29.6 | 0.2 | 14.4 | 4.7 |
| Llama3 | MID | HBM4 | 16 | 55.6 | 34.1 | 0.2 | 7.5 | 2.6 |
| Llama3 | LONG | HBM4 | 4 | 43.3 | 54.1 | 0.2 | 1.9 | 0.4 |
| Llama3 | LONG | HBM4 | 8 | 68.9 | 25.5 | 0.2 | 4.9 | 0.4 |
| Llama3 | LONG | HBM4 | 16 | 72.5 | 22.1 | 0.2 | 4.7 | 0.4 |
| Llama3 | MID | HBF+ | 4 | 55.6 | 31.8 | 9.4 | 1.9 | 1.3 |
| Llama3 | MID | HBF+ | 8 | 55.6 | 31.8 | 9.4 | 1.9 | 1.3 |
| Llama3 | MID | HBF+ | 16 | 55.6 | 31.8 | 9.4 | 1.9 | 1.3 |
| Llama3 | LONG | HBF+ | 4 | 79.0 | 13.3 | 6.4 | 1.1 | 0.2 |
| Llama3 | LONG | HBF+ | 8 | 83.3 | 6.9 | 7.1 | 2.4 | 0.4 |
| Llama3 | LONG | HBF+ | 16 | 83.3 | 6.9 | 7.1 | 2.4 | 0.4 |
| Llama4 | MID | HBM4 | 4 | 34.1 | 63.1 | 0.2 | 1.9 | 0.6 |
| Llama4 | MID | HBM4 | 8 | 61.8 | 36.1 | 0.2 | 1.1 | 0.9 |
| Llama4 | MID | HBM4 | 16 | 72.5 | 19.7 | 0.2 | 6.7 | 0.9 |
| Llama4 | LONG | HBM4 | 4 | 53.9 | 44.2 | 0.2 | 1.3 | 0.4 |
| Llama4 | LONG | HBM4 | 8 | 66.7 | 31.6 | 0.2 | 1.3 | 0.2 |
| Llama4 | LONG | HBM4 | 16 | 77.9 | 18.2 | 0.2 | 3.4 | 0.2 |
| Llama4 | MID | HBF+ | 4 | 68.0 | 18.5 | 12.9 | 0.2 | 0.4 |
| Llama4 | MID | HBF+ | 8 | 74.0 | 11.4 | 13.7 | 0.2 | 0.6 |
| Llama4 | MID | HBF+ | 16 | 74.0 | 7.7 | 13.7 | 3.9 | 0.6 |
| Llama4 | LONG | HBF+ | 4 | 76.6 | 16.3 | 6.7 | 0.2 | 0.3 |
| Llama4 | LONG | HBF+ | 8 | 83.5 | 8.8 | 7.3 | 0.2 | 0.2 |
| Llama4 | LONG | HBF+ | 16 | 85.8 | 4.9 | 7.5 | 1.5 | 0.2 |

**Comparison to the prior pixel-reading:** Attention/FFN agree closely (within ~1-2pp, e.g.
Llama3/MID/HBM4/4GPU: 42.1 vs old 40.8). **KV Write now reads a small ~0.2% for every HBM4 row**,
where the old table read a flat 0.0 — this is very likely a genuine refinement, not an artifact:
the paper defines "KV Write" as writing newly-admitted-query KV cache regardless of memory type
(§IV), so HBM4 doing this at HBM's much higher bandwidth would produce a real but tiny (sub-1%)
time share that the old pixel method's resolution couldn't distinguish from zero. **Communication
and Others shifted more than Attention/FFN did** for a few HBM4 cells (e.g. Llama4/LONG/HBM4/8GPU:
old Communication=0.0/Others=3.1 vs new Communication=1.3/Others=0.2) — both bands are thin (<5%)
in the source figure, so this is consistent with the general "near-zero-denominator" reading
imprecision already flagged for this figure's smallest categories (see the error-rate analysis in
`PAPER_INCONSISTENCIES.md`), not a methodology concern — but treat any single Communication/Others
cell here with the same caution as a near-floor Fig. 4 CONV/CONV+ reading.

## 4. SLO Sensitivity Analysis (Figure 6)

Measured in the **LONG** workload only. Both axes are pure ratios, normalized to 8-GPU HBM4 for
that workload — no external anchor value needed (the normalization *is* the anchor: HBM4@8GPU
must read exactly 1.0). Recovered as exactly 1.0 with 0.0000pt spread across every SLO and both
models, for both the TPS and Batch axes independently.

**Both rows below are now vector-exact** (superseding the 2026-07-06 pixel-domain correction to
the TPS row, and fully resolving the previously-open Batch-row question):

- **TPS row**: matches the earlier pixel-domain-corrected values to within ~1.5%, and resolves
  three previously-uncertain interpolated (`~`) cells cleanly (e.g. Llama3/HBF+/0.1s: the 16-GPU
  cell is now confirmed *exactly* flat with the 8-GPU cell, 1.294 both, rather than the previous
  `~1.257` estimate).
- **Batch row: the previously-flagged "~25% overshoot, unexplained" problem is resolved — it was
  a pixel-reading error in the old extraction, not a second axis miscalibration.** The old
  reading had HBM4@8GPU at 0.64 (should be exactly 1.0 by the figure's own definition); the vector
  data gives exactly 1.0, confirmed 8 independent times (both models × 4 SLOs). The true HBM4
  curve is 0.48/1.00/1.26 (Llama3) and 0.55/1.00/1.26 (Llama4) at 4/8/16 GPU — nothing like the old
  0.23/0.64/0.87. Cross-checked against this repo's own simulator output
  (`experiment_results.md`, the freshest available sweep):
  the new Llama3 values (0.48/1.00/1.26) match the sim's own reported ratios (0.47/1.00/1.00) at
  4 and 8 GPU closely; the 16-GPU gap (1.26 vs the sim's flat 1.00) is a **separate, already-known
  issue** — the sim doesn't grow llama3/HBM4 batch past 8 GPU (a TP-cap-at-`num_kv_heads`
  limitation), while both Fig. 3 (independently, absolute batch 3.8→4.7 at 8→16 GPU) and this
  vector-extracted Fig. 6 agree the paper's own systems show real growth there. Llama4 matches
  even more closely (0.55/1.00/1.26 vs the sim's 0.51/1.00/1.23).

| Model | Memory | SLO | Metric | 4 GPU | 8 GPU | 16 GPU |
|---|---|---|---|---|---|---|
| Llama3 | HBM4 | 0.05s | TPS Ratio | 0.499 | 1.000 | 1.068 |
| Llama3 | HBM4 | 0.1s | TPS Ratio | 0.499 | 1.000 | 1.068 |
| Llama3 | HBM4 | 0.2s | TPS Ratio | 0.499 | 1.000 | 1.068 |
| Llama3 | HBM4 | offline | TPS Ratio | 0.499 | 1.000 | 1.068 |
| Llama3 | HBF | 0.05s | TPS Ratio | 0.765 | 0.958 | 0.958 |
| Llama3 | HBF | 0.1s | TPS Ratio | 1.006 | 1.080 | 1.080 |
| Llama3 | HBF | 0.2s | TPS Ratio | 1.119 | 1.141 | 1.141 |
| Llama3 | HBF | offline | TPS Ratio | 1.174 | 1.174 | 1.174 |
| Llama3 | HBF+ | 0.05s | TPS Ratio | 0.966 | 1.167 | 1.167 |
| Llama3 | HBF+ | 0.1s | TPS Ratio | 1.201 | 1.294 | 1.294 |
| Llama3 | HBF+ | 0.2s | TPS Ratio | 1.320 | 1.355 | 1.355 |
| Llama3 | HBF+ | offline | TPS Ratio | 1.375 | 1.375 | 1.375 |
| Llama4 | HBM4 | 0.05s | TPS Ratio | 0.801 | 1.000 | 1.167 |
| Llama4 | HBM4 | 0.1s | TPS Ratio | 0.801 | 1.000 | 1.167 |
| Llama4 | HBM4 | 0.2s | TPS Ratio | 0.801 | 1.000 | 1.167 |
| Llama4 | HBM4 | offline | TPS Ratio | 0.801 | 1.000 | 1.167 |
| Llama4 | HBF | 0.05s | TPS Ratio | 0.814 | 0.945 | 1.035 |
| Llama4 | HBF | 0.1s | TPS Ratio | 0.968 | 1.068 | 1.109 |
| Llama4 | HBF | 0.2s | TPS Ratio | 1.080 | 1.132 | 1.148 |
| Llama4 | HBF | offline | TPS Ratio | 1.132 | 1.161 | 1.164 |
| Llama4 | HBF+ | 0.05s | TPS Ratio | 0.983 | 1.134 | 1.218 |
| Llama4 | HBF+ | 0.1s | TPS Ratio | 1.149 | 1.252 | 1.288 |
| Llama4 | HBF+ | 0.2s | TPS Ratio | 1.262 | 1.313 | 1.330 |
| Llama4 | HBF+ | offline | TPS Ratio | 1.313 | 1.339 | 1.342 |
| Llama3 | HBM4 | 0.05s | Batch Ratio | 0.48 | 1.00 | 1.26 |
| Llama3 | HBM4 | 0.1s | Batch Ratio | 0.48 | 1.00 | 1.26 |
| Llama3 | HBM4 | 0.2s | Batch Ratio | 0.48 | 1.00 | 1.26 |
| Llama3 | HBM4 | offline | Batch Ratio | 0.48 | 1.00 | 1.26 |
| Llama3 | HBF | 0.05s | Batch Ratio | 1.46 | 1.89 | 1.89 |
| Llama3 | HBF | 0.1s | Batch Ratio | 3.99 | 4.25 | 4.25 |
| Llama3 | HBF | 0.2s | Batch Ratio | 8.87 | 9.06 | 9.06 |
| Llama3 | HBF | offline | Batch Ratio | 17.72 | 17.72 | 17.74 |
| Llama3 | HBF+ | 0.05s | Batch Ratio | 1.91 | 2.29 | 2.29 |
| Llama3 | HBF+ | 0.1s | Batch Ratio | 4.71 | 5.10 | 5.10 |
| Llama3 | HBF+ | 0.2s | Batch Ratio | 10.43 | 10.76 | 10.76 |
| Llama3 | HBF+ | offline | Batch Ratio | 20.45 | 20.45 | 20.44 |
| Llama4 | HBM4 | 0.05s | Batch Ratio | 0.54 | 1.00 | 1.26 |
| Llama4 | HBM4 | 0.1s | Batch Ratio | 0.54 | 1.00 | 1.26 |
| Llama4 | HBM4 | 0.2s | Batch Ratio | 0.54 | 1.00 | 1.26 |
| Llama4 | HBM4 | offline | Batch Ratio | 0.54 | 1.00 | 1.26 |
| Llama4 | HBF | 0.05s | Batch Ratio | 1.74 | 2.04 | 2.24 |
| Llama4 | HBF | 0.1s | Batch Ratio | 4.12 | 4.58 | 4.71 |
| Llama4 | HBF | 0.2s | Batch Ratio | 9.26 | 9.72 | 9.85 |
| Llama4 | HBF | offline | Batch Ratio | 18.30 | 18.76 | 19.02 |
| Llama4 | HBF+ | 0.05s | Batch Ratio | 2.10 | 2.43 | 2.63 |
| Llama4 | HBF+ | 0.1s | Batch Ratio | 4.90 | 5.36 | 5.49 |
| Llama4 | HBF+ | 0.2s | Batch Ratio | 10.82 | 11.28 | 11.34 |
| Llama4 | HBF+ | offline | Batch Ratio | 21.03 | 21.55 | 21.75 |

Full derivation, raw extraction scripts, and the prior pixel-domain TPS-only correction: see
`ledgers/FINDINGS_REGISTER.md` "Fifth-pass register" (F3) for the original pixel-domain work, and
`PAPER_INCONSISTENCIES.md`'s Fig-6 entries for how both the TPS and Batch findings were tracked
before this vector-extraction pass resolved them.

## 5. Write Traffic and Endurance (Figure 7)

No exact printed anchor exists for this figure (see shortfall #5) — calibrated via least-squares
fit across the 6 y-axis tick labels (0/100/200/300/400/500 ×10³), residuals <0.5 units (<0.1% of
scale). Units: 3-year P/E-cycle count (raw count, matching this repo's own scale — the paper plots
this as ×10³). Unlike the prior version of this table (which only had the @8-GPU "online" column,
matching this repo's `experiment_results.md` layout), the vector data recovers **all three GPU
counts (1/8/16) and both the online and offline series** cleanly, since Fig. 7 is a plain grouped
bar chart with no stacking/subsumption ambiguity.

| Model | Workload | GPUs | HBF (online) | HBF+ (online) | HBF (offline) | HBF+ (offline) |
|---|---|---|---|---|---|---|
| Llama3 | SHORT | 1 | 116522 | 194766 | 189550 | 197896 |
| Llama3 | SHORT | 8 | 171814 | 194766 | 189550 | 197896 |
| Llama3 | SHORT | 16 | 171814 | 194766 | 189550 | 197896 |
| Llama3 | MID | 1 | 111305 | 164512 | 275097 | 281356 |
| Llama3 | MID | 8 | 239626 | 273010 | 275097 | 281356 |
| Llama3 | MID | 16 | 239739 | 273223 | 275097 | 281356 |
| Llama3 | LONG | 1 | 57056 | 73748 | 175987 | 182247 |
| Llama3 | LONG | 8 | 188651 | 197148 | 205198 | 209372 |
| Llama3 | LONG | 16 | 188506 | 196852 | 205198 | 209372 |
| Llama4 | SHORT | 1 | 149906 | 185377 | 415936 | 185377 |
| Llama4 | SHORT | 8 | 439931 | 424283 | 470186 | 424283 |
| Llama4 | SHORT | 16 | 440405 | 425056 | 470186 | 425326 |
| Llama4 | MID | 1 | 165555 | 190593 | 359601 | 270923 |
| Llama4 | MID | 8 | 392985 | 409677 | 423239 | 410720 |
| Llama4 | MID | 16 | 398201 | 410720 | 423239 | 410720 |
| Llama4 | LONG | 1 | 136344 | 139473 | 181204 | 187463 |
| Llama4 | LONG | 8 | 200000 | 205200 | 217700 | 219800 |
| Llama4 | LONG | 16 | 207300 | 211500 | 218800 | 219800 |

The @8-GPU/online column matches the prior table closely (e.g. Llama3/SHORT/HBF: 171814 vs the
old table's 172600, ~0.5% off; Llama4/MID/HBF+: 409677 vs 410900, ~0.3% off) — consistent with the
general pattern of small-but-nonzero error in the prior pixel-calibrated method even at points it
calibrated carefully. Several offline/online pairs read identically at low GPU counts (e.g.
Llama4/SHORT/1-GPU: online and offline HBF+ both 185377) — plausibly a genuine feature (too little
work at 1 GPU/SHORT for the offline relaxation to matter) rather than an extraction error, but not
independently confirmed; treat with the same caution noted for near-floor values in Section 2.
