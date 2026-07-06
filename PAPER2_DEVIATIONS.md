# Paper2 Replication — Honest Deviations Report

Consolidated, reader-facing summary of how this simulator extension reproduces
**Kyung, Moon, Cho, Ahn, "High-Bandwidth Flash for KV Caches: Endurance and
Performance Implications," IEEE CAL 2026** (Figures 4 and 5), what deviates, and
why. Every deviation below is either (a) a grounded code fix, or (b) an honest
modeling difference we deliberately did **not** patch because closing it would
require tuning-to-target or importing an unstated paper assumption. No number in
the simulator was curved to fit the paper.

Backing data: `paper2_results/fig4_table.md`, `fig5_table.md`; raw per-cell
markers in `paper2_results/fig4_results.jsonl` / `fig5_results.jsonl`. Full
decision trail in `PAPER2_NOTES.md`.

---

## 1. Headline verdict

- **Fig4 (60 cells: lifetime + TPOT):** the sim reproduces the paper's
  qualitative structure and normalized-lifetime trend well. Absolute lifetimes
  run systematically low (median −10%). The deviation is a ~8% central
  bandwidth-efficiency effect (§4a) *plus* a workload-dependent spread (§4b): the
  per-cell TPOT error is NOT flat — it ranges +3% to −26% and grows with input
  share and context. Both components are honest, neither is a bug.
- **Fig5 (throughput bars + TPOT + SRAM):** after correcting the harness's
  baseline denominator (§3), the sim's baseline throughput matches the paper to
  **−1.0…−3.2%** (all four model/context baselines), and the 19 offload-
  independent bar TPOTs match to a median **−8.5%** (range −4…−11%; these are all
  at the fixed 1:3 workload, where the efficiency effect is tightest). Most
  decisively, the paper's normalized-throughput **bar heights** — a quantity
  *nowhere printed as text*, recovered from the figure's vector geometry (§7) —
  match the sim's normalized throughput to a **median 0.8%** across all 32 bars.
  One real code bug (SRAM double-buffer over-report) was found and fixed (§5);
  the resulting SRAM sizing is the one metric that only lands to ±40%, scattered
  in both directions (§5) — but it is output-only and affects no headline result.
- **One qualitative edge case is NOT reproduced (§6):** the paper's distinctive
  "even ½-HBF becomes SLO-bound" case (R1 768GB/8K) is not SLO-bound in the sim
  (20 ms of headroom) — a direct downstream consequence of the ~8–10% TPOT
  underestimate placing that bar just under the 200 ms SLO.

Two harness/code fixes were applied (baseline denominator §3; SRAM DBUF §5). Two
systematic deviations were investigated and left as honest, documented
differences (TPOT efficiency §4a; Fig4 context-scaling §4b), plus one disclosed
qualitative mismatch (§6).

---

## 2. What matches the paper closely

| Quantity | Sim vs paper | Notes |
|---|---|---|
| KV bytes/token (Maverick) | 196,608 B, exact | 2·8·128·2·48, paper-exact |
| KV bytes/token (R1, BF16) | 70,272 B, exact | (512+64)·2·61 |
| **Fig5 normalized-throughput bar heights (32)** | **median 0.8%, max 5.3%** | vector-extracted (§7); the paper's own plotted quantity |
| Fig5 baseline throughput (all 4) | −1.0 … −3.2% | after §3 denominator fix + deterministic workload |
| Fig4 normalized lifetime trend | tracks within a few % | e.g. Mav CV0.1 (1.0, 1.99, 3.79) vs (1.0, 2.04, 4.14) |
| Fig4 DeepSeek-R1 @ 8K | +6% … −6% | near-exact at short context |
| Fig5 bar TPOTs (19, offload-independent) | median −8.5% (−4…−11%) | all at fixed 1:3 workload |
| Fig4 lifetime, 60 cells | median −10%, 87% within ±20% | worst −36% at one extreme corner |
| Fig4 TPOT, 60 cells | median −11%, range +3…−26% | workload-dependent, not flat (§4b) |
| First-activated-expert tR exposure | exact to the ns | 24×3µs Maverick / 58×3µs R1 per step |

Paper1 functionality is untouched: the byte-identity regression gate (three
paper1 configs) passes after every change — only the additive `P2_*` marker
lines differ.

---

## 3. Fixed — Fig5 baseline denominator (harness interpretation)

**Symptom.** The harness's original "baseline throughput" was +148…+181% over the
paper (24,488 vs 9,843 tok/s, Maverick 8K).

**Root cause (not a simulator bug).** An Opus bug-hunt proved the plain-HBM GQA
decode path is correct (TPOT scales linearly with batch; the MLA-only YAML flags
are byte-identically inert for GQA Maverick; hand-check reproduces the step
time). The harness had simply computed the *wrong operating point*: a synthetic
all-in-HBM, no-offload, batch=648 config. The paper's Fig5 "baseline" is the
**NVLink5.0 device_HBM @ 512 GB bar at the 200 ms SLO** — the leftmost bar each
normalized group sits against (≈1.0×).

**Evidence it is the right denominator.** Sim NVLink5.0-512GB = 9,687 vs paper
9,843 (−1.6%). The paper-implied served batch at the SLO for all four
(model,ctx) baselines (≈493 / 4,097 / 1,113 for Mav-32K / R1-8K / R1-32K)
matches the harness's `compute_batch` for those exact configs (480 / 4,360→
throttled / 1,088) to a few percent; and where it exceeds the SLO (R1), the
paper independently marks NVLink5.0 SLO-bound. This is validation, not fitting —
the sim's NVLink5.0 output lands within 1.6% of the paper's absolute number.

**Fix.** `run_paper2.py` `_sim_baseline_rec` + the baseline table + self-
consistent bar normalization now use the NVLink5.0-512GB bar; the synthetic
probe is retained only as a labeled diagnostic column.

---

## 4. Honest, NOT patched — the systematic lifetime/TPOT underestimate

### 4a. The ~8% decode-TPOT underestimate (the central bandwidth-efficiency effect)

**Scope note.** This ~8% is the *central tendency*, cleanest on the Fig5
offload-independent bars (all at the fixed 1:3 workload). It is NOT a flat
property of every cell: across the full Fig4 grid the per-cell TPOT error spans
+3% to −26% and is workload-dependent (ratio 1:3 mean −6%, ratio 15:1 mean −16%;
and it drifts with context per model). The workload-dependent spread on top of
this central effect is analyzed in §4b — do not read "8%" as holding uniformly
across all 60+19 cells.

**Observation (where it is tightest).** Sim/paper TPOT is a tight **constant
ratio ≈ 0.92** across three physically different configs at one workload point
(HBF 46.6/51, NVLink6.0 113.8/124, ½-HBF 92.8/100 for Maverick 8K/512GB 1:3),
while the absolute ms gaps vary 2.3× — a constant *fraction*, not a fixed
overhead. This constancy across bottleneck regimes is what points to a
bandwidth-efficiency cause.

**Diagnosis.** A full decode-step composition trace (KV-read-from-flash 67.6%,
MoE weight-read 27.1%, all-to-all comm 0.4%; total matches the sim's own
`latency` to 0.03 ms) ruled out every candidate missing term (comm is modeled
but negligible; all weight tensors charged; roofline max/sum correct). The
mechanism: the sim applies the paper's *stated peak* bandwidths (flash 8 TB/s,
½-HBF 4 TB/s, HBM 8 TB/s, C2C 1.8 TB/s) at 100% efficiency, whereas the paper's
roofline appears to fold in ~92% achievable-vs-peak efficiency it never
separately reports.

**Why not patched.** Because the bandwidths already match the paper exactly, a
0.92 derate would be pure tuning-to-target — the forbidden move. Documented as an
honest ~8% central effect.

### 4b. Fig4 context-dependent residual (on top of 4a)

**Observation.** The sim's normalized lifetime grows *less* with context than the
paper's (e.g. R1 CV0.1: 3.07× vs 3.37× at 32K); worst single corner Mav CV0.3
15:1 32K = −16.9% on the normalized value. Any context-uniform part of §4a
*cancels* in the normalized column (it is a ratio to the 8K case), so what
remains here is a genuinely separate, context-dependent effect.

**Diagnosis (two honest mechanisms, no bug).**
1. The decode-write byte term is **exact**: `batch · 196,608`, halving 4.00× per
   context doubling in all 20 cells.
2. **DeepSeek** sub-linearity is roofline physics: MLA decode is compute/batch-
   bound, so as batch shrinks 4× with context, per-iter time drops ~24% (the
   same peak-vs-achievable family as §4a). Even the paper's own R1 column is
   sub-linear (3.37 < 4).
3. **Maverick** is realized-stochastic-workload noise: the admission-byte
   scaling *sign-flips* across cells (overshoots the paper at CV0.3 1:3 = 4.53 >
   4.22; undershoots at 15:1) — the fingerprint of the truncNormal×Beta sampler
   + size-biased fill, not a monotone formula error.

**Why not patched.** Matching the paper's normalized column would require an
unstated paper assumption (constant decode throughput regardless of batch, or a
deterministic mean-based endurance population instead of a sampled one) that is
not better grounded than the current model.

---

## 5. Fixed — Fig5 HBF required-SRAM double-buffer over-report

**Symptom.** P2_REQUIRED_SRAM_BYTES_PER_DEVICE for the HBF bar (Maverick
512GB/8K) read ~500 MB vs the paper's 206 MB (~2.4×).

**Root cause.** Decomposition: ACT 33.4 MB + KVWRITE 47.2 MB (both correctly
batch-scaled) + a bogus **DBUF 419 MB**. Because `system.chunk_size==0` for
paper2, the read double-buffer collapsed to the full *physical staging capacity*
(2 × 40 MiB/stack × 5 = 400 MiB) — far more SRAM than needed to hide tR.

**Fix (grounded, not fitted).** The minimal buffer that hides one page-read
latency is the flash pipeline's bandwidth-delay product: `flash_read_bandwidth ×
tR` = 8 TB/s × 3 µs = 24 MB → **48 MB double-buffered**. This is exactly the
timing model's own break-even (a chunk with transfer ≥ tR exposes zero per-chunk
residual, `layer_impl.h:156`), so decode timing is **identical** either way
(independently confirmed by the §4a trace). Applied at the DBUF block in
`cluster.cpp`; `system.chunk_size` override + capacity cap preserved.

**Result (fixed-binary rerun, all 8 HBF cases).** The fix removes the flat
400 MiB floor; the metric is now driven by the batch-scaled ACT + KVWRITE terms
plus the 48 MB read buffer:

| Model | Budget | Ctx | Sim MB | Paper MB | %err |
|---|---|---|---|---|---|
| Maverick | 512 GB | 8K | 129 | 206 | −38% |
| Maverick | 768 GB | 8K | 182 | 308 | −41% |
| Maverick | 512 GB | 32K | 68 | 89 | −23% |
| Maverick | 768 GB | 32K | 81 | 114 | −29% |
| DeepSeek-R1 | 512 GB | 8K | 310 | 265 | +17% |
| DeepSeek-R1 | 768 GB | 8K | 523 | 431 | +21% |
| DeepSeek-R1 | 512 GB | 32K | 113 | 104 | +9% |
| DeepSeek-R1 | 768 GB | 32K | 167 | 145 | +15% |

**This is the weakest-matching metric, and it is honest scatter in BOTH
directions — not a uniform bias.** Maverick undershoots (−23…−41%) while R1
*overshoots* (+9…+21%). The fixed 48 MB read buffer ≈ the paper's ~51 MB
batch-independent SRAM offset, so the remaining gap is in the batch-scaled
ACT+KVWRITE per-sequence staging: it runs low for Maverick (GQA, small hidden
activation) and high for R1 (MLA, large hidden-dim activation working set at its
much larger per-device batch). Matching both models' absolute SRAM would require
the paper's exact (unpublished) staging-buffer accounting; we report the
simulator's own component sum. This is an **output-only** metric — it has zero
effect on TPOT, throughput, or lifetime — so it does not affect any headline
result. The DBUF fix itself is unambiguously correct (removes SRAM the timing
model provably does not need to hide tR); the residual per-model scatter is a
documented modeling difference, not fitted.

**Caveat — Fig4's SRAM markers are stale (pre-fix).** The SRAM table above is
from the fixed-binary Fig5 rerun. The `P2_SRAM_*` marker fields inside
`paper2_results/fig4_results.jsonl` were NOT re-run after the DBUF fix (Fig4's
checkpoints key on cell_id+config_hash, not binary version), so they still carry
the old ~419 MB DBUF constant. This is harmless — Fig4's lifetime/TPOT come from
`total_ns` / `P2_KV_BYTES_WRITTEN_TOTAL`, never from the SRAM markers — but do
not read SRAM out of the Fig4 jsonl; the corrected values are the Fig5 table
above only.

---

## 6. Qualitative mismatch NOT reproduced — one SLO-bound edge case

The paper highlights one distinctive edge case: for **DeepSeek-R1, 768 GB, 8K**,
*even* ½-HBF becomes SLO-bound (the only HBF/½-HBF bar in Fig5 that is). The sim
does **not** reproduce this: `deepseekR1|ctx8192|budget3x|1/2-HBF` measures
TPOT 178.5 ms — 21 ms *under* the 200 ms SLO, not bound.

This is an honest, disclosed consequence of §4a: the sim runs this R1 bar ~8–10%
faster than the paper (in line with R1's bar-TPOT deviations elsewhere,
−9.7…−10.7%). Applying the paper's implicit ~8–10% bandwidth derate would push
178.5 ms to ~196–200 ms — right at the SLO boundary, where the paper places it
just above and the sim places it just below. So the single binary SLO-bound flag
flips. We do not patch it (that would be the §4a tuning-to-target); we flag it as
the one qualitative disagreement with the paper. No other SLO-bound flag differs.

---

## 7. Vector bar-height validation (Fig5 normalized throughput)

Fig5's y-axis ("Normalized throughput", 0–6) is the paper's headline performance
quantity, but the bar heights are **never printed as text** — only visually
encoded. Fig5 is a fully vector figure (PDF page 3: 0 raster images, 36
rectangles), so the bar geometry is extracted directly and calibrated against the
printed 0–6 tick labels (≈4.83 pt/unit; bars baselined at the 0 gridline). This
recovers the paper's own normalized-throughput values and compares them to the
sim's normalized throughput (each bar ÷ the sim's NVLink5.0-512GB bar per
(model,ctx) — the same reference the paper's axis is built on).

Across all **32 bars: median |error| 0.8%, mean 1.4%, max 5.3%** — the tightest
agreement of any metric in the replication. Two internal-consistency checks pass:
all 8 NVLink5.0 bars read ≈1.01× (the ~1.0 reference by construction, validating
both the calibration and the §3 baseline-denominator choice), and the
HBF/½-HBF/NVLink6.0 heights track the sim per bar (e.g. Maverick 512GB/8K: paper
4.22 / 2.15 / 1.74 vs sim 4.26 / 2.14 / 1.74). The max 5.3% is the R1 768GB/8K
SLO-edge group (§6). This is the most direct throughput comparison available — no
intermediate assumptions — and it independently confirms the sim reproduces the
paper's plotted bars to ~1%.

**Absolute throughput (tokens/s).** Multiplying each vector bar height by the
paper's *printed* baseline (9,843 / 2,467 / 20,483 / 5,566 tok/s) gives the
paper's absolute throughput per bar, compared to the sim's measured
`throughput_tok_s`: across all 32 bars **median |error| 1.1%, mean 1.4%, max
5.6%**. Maverick lands within ±2.5% everywhere; DeepSeek-R1 runs 3–6% low on its
NVLink and 32K-HBF bars (its baseline throughput is ~4% under the paper and its
bar TPOTs sit at the −10% end of §4a). So the sim reproduces not just the
*shape* of Fig5 (normalized bars, ~1%) but the *absolute* tokens/s to ~1% median.

- **Fig5 workload = deterministic mean length (no dispersion).** Paper §VI-B:
  "…for each Lin and Lout configuration, we use the average sequence length within
  a batch…" — a single homogeneous mean-length batch. Dispersion (CV/κ) is a
  Fig4-only construct (its stochastic heterogeneous-request lifetime sweep). The
  harness runs Fig5 with `FIG5_CV=0, FIG5_KAPPA=0`, which triggers the sampler's
  exact deterministic path. (Verified negligible vs the earlier low-dispersion
  0.1/90 assumption: <1% TPOT, since CV=0.1 was already near-deterministic — but
  this matches the paper's stated method exactly.)
- **First-activated-expert exposure is now in the Fig4 dataset.** An earlier Fig4
  sweep binary predated that feature; its committed lifetimes/TPOTs were ~0.15%
  (Maverick, 24×3µs) / ~0.29% (R1, 58×3µs) too fast. Both figures were re-run on
  the final binary so the entire dataset is reproducible from committed source and
  includes every modeled paper2 mechanism. Aggregate deviations are essentially
  unchanged (the correction is marginal); it is a correctness/reproducibility fix,
  not a numbers change.
- **V_weight** taken from the simulator's own mapping accounting (Maverick
  999.73 GB, R1 1581.70 GB), not the paper's label-inversion (~910 / ~1454 GB).
  Consequence: our printed CPU/HBM ratios run ~1.95 where the paper prints ~1.8
  (Maverick 512GB). Kept faithful to the sim; not fudged.
- **GB vs GiB:** preset capacities in GiB (codebase convention); decimal GB in
  the Python TBW/batch math (paper Eq. 2 convention). ±7% sensitivity,
  documented, not tunable.
- **injection_rate = 0** is mandatory for paper2 (nonzero engages Poisson gating
  and starves the batch below the budget-derived size).
- **iRoPE disabled** for paper2 Maverick (full attention per paper §II); paper1's
  iRoPE Maverick path is untouched.

- **Attention-score/softmax tile placement in the required-SRAM total (open
  question, 2026-07-06).** Paper2's `P2_REQUIRED_SRAM_BYTES_PER_DEVICE`
  (`cluster.cpp:289`) correctly sums the three buffers the paper's §III-System
  assumes — activation (`P2_SRAM_ACT_BYTES` = `peakIntermediateBytes`), read
  double-buffer (`P2_SRAM_DBUF_BYTES`), and per-stage KV-write buffer
  (`P2_SRAM_KVWRITE_BYTES`). But where the `Q·Kᵀ` score/softmax matrix lives in
  this total is *asserted, not sized*: the comment at `cluster.cpp:236-239` says
  the score staging "shares the same physical double-buffer as P2_SRAM_DBUF_BYTES,"
  i.e. it is folded into the flash-read double-buffer (sized by the flash
  bandwidth-delay product, `2 × flash_read_bandwidth × tR`), and NOT added to the
  activation term. This is a *third, distinct* placement hypothesis vs. the two in
  paper1's `PAPER_INCONSISTENCIES.md` U7: (a) paper1-main charges the score tile
  **0** on HBF+/CONV+; (b) the rejected `ab-score-accounting` A/B charged the full
  **O(context)** matrix against the activation pool; (c) paper2 here folds it into
  the read double-buffer. None of the three sizes an actual FlashAttention-style
  **O(chunk)** compute tile — the physically-correct invariant (one chunk of `S`
  resident on-die, independent of both context length and flash-read timing).
  Consequence for paper2's Fig-5 SRAM sizing: because the score tile is folded into
  DBUF (which is flash-timing-sized, not batch-scaled), the reported required-SRAM
  is **batch-independent** in its score component — but a real O(chunk) score tile
  scales with batch (`batch × heads/tp × chunk × precision`). This may be part of
  the batch-scaled residual noted in §5 ("the remaining gap is in the batch-scaled
  term"). Not patched: resolving it requires the same dedicated compute-tile knob
  proposed for paper1 (`PAPER_INCONSISTENCIES.md` U7 "SEVENTH-PASS" recipe); doing
  it here without that shared knob would be an unstated assumption. Flagged for a
  joint paper1+paper2 score-tile pass.

---

## 9. Discipline

Every proposed change was adversarially pre-screened; the two systematic biases
(§4a, §4b) were each traced to a mechanism and left unpatched precisely because
the only way to close them was tuning-to-target or importing an unstated
assumption. The two applied fixes (§3 harness denominator, §5 SRAM DBUF) are
grounded in, respectively, the paper's own figure construction and the
simulator's own timing-model break-even — and both were validated independently
of the paper's target numbers.
