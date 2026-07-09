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
  run systematically low (median −11%, worst corner −29%). The deviation is a
  ~8% central efficiency-gap effect (§4a, irreducible) *plus* a context-dependent
  residual (§4b) — one part of which was a real **finite-window measurement bug
  now fixed** (Little's-law admission rate, §4b), the rest genuine roofline
  physics. The per-cell TPOT error is NOT flat (+3% to −26%). No number is tuned.
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
| Fig4 lifetime, 60 cells | median −11%, worst −29% | seed-independent after the §4b Little's-law fix |
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
`latency` to 0.03 ms) ruled out every candidate missing term. Two follow-up
adversarial bug-hunts closed the remaining hypotheses: **(a) compute MFU** —
decode is 20–50× under the compute frontier (MoE-GEMM compute/memory = 0.004), so
no MFU derate moves TPOT, and the paper states *peak* 2,250 TFLOPS regardless;
**(b) MoE activated-expert count under Zipfian** — saturated (all 16 local
experts active every step regardless of skew 0.8), so skewness-insensitive
(<0.001% TPOT change at skew 0 vs 0.8). The paper explicitly uses **peak**
8 TB/s / 2,250 TFLOPS with no stated efficiency derate, and its 16.8 TB/s
on-chip bandwidth is a *sufficiency* assumption, not a throttle (½-HBF, with half
the on-chip demand, shows the same deviation).

**Why not patched — and it is NOT an "efficiency" the paper folds in.** An
earlier draft of this report rationalized the gap as the paper assuming ~92%
achievable-vs-peak bandwidth. **That is wrong** — the paper states *peak*
bandwidths, so there is no derate to match, and inventing a 0.92 factor would be
tuning-to-target. The paper's tool is a *modified* version of the same base
simulator this codebase derives from (its ref [1] = `scale-snu/LLMSimulator`), so
the ~8% is the residual between **two HBF reimplementations of the same
simulator**, both at peak — an irreducible implementation-detail gap we cannot
close without their code. Documented as such.

### 4b. Fig4 context-dependent residual — one FIXED measurement bug + one honest residual

**Observation.** The sim's normalized lifetime grows *less* with context than the
paper's (e.g. R1 CV0.1: 3.07× vs 3.37× at 32K). This decomposed into a real,
fixable measurement bug **plus** a genuine residual.

**FIXED — non-converged admission write-rate (finite-window bug).** Lifetime ∝
1/write_rate; write_rate = (admission + decode KV bytes)/elapsed. The decode term
is exact every step, but the *admission* term was accumulated only when a
sequence completes — and at ctx≥16K a sequence's output (2K–24K steps) exceeds
the 1000-step timed window, so few complete and the admission byte-rate was a
sample-starved, **seed-dependent transient** (~6–11% swings at ctx≥16K; the
"sign-flip across cells" an earlier draft mistook for benign sampler noise was
*this* measurement artifact). **Fix (grounded, no tuning):** replace the windowed
admission counter with the steady-state **Little's-law** rate. In continuous
batching admissions/sec = completions/sec, so
`admission_byte_rate = decode_byte_rate × E[Lin]/E[Lout]`, with E[Lin]/E[Lout]
from a one-time Monte-Carlo of the sampler (the unbiased *arrival* distribution,
not the size-biased in-flight one). It is exact and seed-independent, and reduces
to E[Lin]/E[Lout] because paper2 disables iRoPE (uncapped admission) — assert-
guarded against any iRoPE-on reuse (`run_paper2.py` `analytic_admission_decode_ratio`
/ `fig4_metrics`; `_fig4_worker`). Effect: worst corner **−36% → −29%**, and the
sign-flipped cross-CV inconsistencies collapse (e.g. Mav 1:1/32K windowed 14.3
(CV0.1) vs 19.6 (CV0.3) → analytic 16.8 vs 17.0). Applied as a post-processing
recompute of the exact stored decode bytes — no re-sweep needed.

**Honest residual (survives the fix).** After convergence,
`normalized_lifetime(32K) = 4 × (t_step_32K / t_step_8K)`: the write RATE is
coupled to TPOT (steps/sec), so the DeepSeek MLA sub-linearity (per-iter time
drops ~24% as batch shrinks 4× with context) and the flat §4a effect propagate
into the lifetime scaling. That is genuine roofline physics; if the paper's
endurance model used a fixed reference throughput rather than live TPOT it would
be an unstated model difference — not patched (it's not better-grounded than the
live-TPOT model).

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

**This is the weakest-matching metric, and it is a systematic, architecture-
correlated DEFECT — not "honest scatter" (an earlier draft's mischaracterization).**
Maverick undershoots (−23…−41%) while R1 *overshoots* (+9…+21%) because the ACT
term (`P2_SRAM_ACT_BYTES` = `peakIntermediateBytes`, `footprint.h:175`) reuses a
conservative **capacity-gate** footprint — which *sums* all attention sub-tensors
as concurrently-live — as the paper's I/O-activation *staging* metric. Real
chunked/pipelined execution releases those transients as they're consumed, so the
sum over-counts: for R1's MLA attention the true peak-of-chain liveness is **45.8%
below** the coded sum (a genuine concurrent-liveness error). **But an adversarial
refutation showed no grounded fix unifies both models:** applying the rigorous
peak-liveness *flips* R1 from +17% to −20% — the paper's figure sits *between* the
naive sum and the true peak, so it embeds an unstated staging assumption — and it
cannot touch Maverick, whose ACT is FFN-bound (gate+up genuinely coexist). A
separate weight-staging buffer was refuted (`layer_impl.h:39-51` already stages
weight reads through the *same* double-buffer as KV — a second one double-counts),
as was an I/O-double-buffer redefinition (~15.6 MB for R1, catastrophic
undershoot). So it is a real definitional defect with **no non-tuning fix**; the
DBUF term itself (above) is correct, and the `Q·Kᵀ` score tile's home is likewise
unspecified by the paper (§8). **Output-only** — zero effect on TPOT/throughput/
lifetime — so it touches no headline result; documented, not patched. (A graph-
based peak-liveness pass would make the number *physically defensible* but still
not match the paper, and is output-only, so it is deferred rather than built.)

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
−9.7…−10.7%). Closing that ~8–10% implementation gap would push 178.5 ms to
~196–200 ms — right at the SLO boundary, where the paper places it just above and
the sim places it just below. So the single binary SLO-bound flag flips. We do
not patch it (§4a has no locatable term to fix, so any change would be
tuning-to-target); we flag it as the one qualitative disagreement with the paper.
No other SLO-bound flag differs.

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

Every claim here was put through an adversarial investigate-then-refute pass
(Opus bug-hunts, then Sonnet refuters attacking each verdict). That process
overturned two of my own earlier calls: it retracted the "~92% efficiency"
rationalization for §4a (the paper uses *peak*), and it reclassified the SRAM
gap from "honest scatter" to a real concurrent-liveness defect — while also
downgrading the Fig4 finding from a "~45% bug" to a ~6–11% finite-window variance.

Net, after vetting: **three grounded fixes applied** — §3 harness baseline
denominator (the paper's own figure construction), §5 SRAM DBUF (the timing
model's own bandwidth-delay break-even), and §4b the Fig4 Little's-law admission
rate (steady-state flow balance) — each validated independently of the paper's
target numbers. **Two residuals left honestly unpatched** because no locatable
term or non-tuning redefinition closes them: the §4a ~8% (an irreducible gap
between two reimplementations of the same base simulator, both at peak) and the
§5/§8 SRAM defect (output-only; the paper's staging assumption is unpublished and
sits between the sim's over-count and the rigorous peak-liveness). No number was
tuned to fit the paper.
