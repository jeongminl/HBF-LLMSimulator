# Paper Inconsistencies — HBF-LLMSimulator vs. Son et al. (IEEE CAL 2026)

Tracker for sim-vs-paper divergences that are **still open** or **explained (not a bug)**. Resolved
code fixes — including the many paper-comparison findings that turned out to be real bugs — live in
`CHANGES.md`; the raw per-pass adjudication trail lives in `ledgers/FINDINGS_REGISTER.md`. This doc
holds only the two live categories.

**Conventions**
- **Ground truth:** `paper_figure_readings.md`, rebuilt 2026-07-06 from the paper PDF's own vector
  drawing commands (the PDF has zero raster images), so every figure's HBM4-at-8-GPU self-
  normalization anchor reproduces at 0.0000pt spread. Older **pixel-sampled** readings (e.g. the
  SHORT/HBF+ bar 854.7, MID 745.1) are **superseded** by the vector values (766.6, 742.0); a
  2026-07-06 audit corrected every item that had cited a stale pixel value — see U7 below.
- **Error metric:** `error = |paper − sim| / sim` (`compare_error_rates.py:7`).
- **Item format:** each item is **Divergence** (exact sim vs paper figures) → **Current
  explanation** → **Disproved arguments** (what was tried and did not hold, incl. stale-data claims).

---

## Still open

### Fig-4 — plain-HBF / llama3_405B / SHORT: flat ~25.3% high, 2–16 GPU
**Divergence.** Paper flat **2494.8** across 2/4/8/16 GPU; sim flat **3342.1** — a consistent
**25.3%** at every GPU count ≥2. The 1-GPU cell is close (sim 1781.96 vs paper 1678.7, 5.8%). Plain
**HBF** — outside U7's HBF+/CONV+ scope.

**Current explanation.** None confirmed. Both curves are flat vs GPU count, so it reads like a fixed
per-GPU ceiling on both sides calibrated ~25% apart (weight-stream/capacity-floor class) — but this
has not been traced to a specific term.

**Disproved arguments.** The "same mechanism class as U5" framing is weak: U5's floor is worst at
1-GPU and eases with GPU count; here it is the reverse (1-GPU is the *close* cell, the gap opens at
2-GPU and stays flat) — so a simple U5-class explanation does not fit.

### Fig-5 — llama3_405B / HBM4 / MID: paper's Communication share drops at 16-GPU, sim's doesn't
**Divergence.** Communication share: 4-GPU 5.1% paper / 5.2% sim, 8-GPU 14.4% / 14.5%, but **16-GPU
paper drops to 7.5% while sim stays 14.5%** (its own 8-GPU value) — 0.48× error.

**Current explanation.** None. The paper's own non-monotonic (up-then-down) Communication share at
16-GPU for a dense HBM4 model is not explained by any documented mechanism, and would require the
paper's winning parallelism to change character at 16-GPU in a way the sim's doesn't.

**Disproved arguments.** (none yet — not investigated.)

### Fig-6 — llama3_405B, 0.05s SLO: HBF-vs-HBM4 TPS ordering flips at 8/16-GPU
**Divergence.** An **ordering** flip, not just magnitude. Paper (HBM4@8GPU≡1.0): 4-GPU HBF 0.765 >
HBM4 0.499, but 8-GPU HBM4 1.000 > HBF 0.958, 16-GPU HBM4 1.068 > HBF 0.958. Sim: HBF leads at
**every** GPU count (8-GPU HBF 1.048 vs HBM4 1.000; 16-GPU 1.048 vs 0.9999) — never reproduces HBM4
overtaking HBF. Sim's HBM4 curve matches paper closely (≤6.4%); its HBF curve runs ~9% high at
8/16-GPU.

**Current explanation.** The flip is driven by HBF reading ~9% too high (or HBM4 too low) at higher
GPU count under a tight SLO. Candidate: the deferred MFU/score-traffic pair — a tight 0.05s SLO is
exactly where compute/MFU-sensitive terms bite hardest.

**Disproved arguments.** (none — flagged as candidate-linked to the deferred pair, not yet traced.)

### Fig-7 — 36-cell online PEC: HBF+ errors elevated, model-dependent GPU trend
**Divergence.** Plain-HBF online PEC errors mostly single digits to ~26%; **HBF+ errors 4.9%–37.8%**,
generally higher than the paired HBF cell. But the GPU-count trend splits by model:
llama3/SHORT/HBF+ error **grows** (20.4%→37.8%→37.8% at 1/8/16-GPU) while llama4/SHORT/HBF+ **shrinks**
(25.1%→22.6%→15.4%).

**Current explanation.** Extends U7's HBF+/CONV+ divergence to the PEC metric (which folds batch and
tpot). The opposite GPU trends across the two models are unresolved — plausibly the U7 escape
dynamics (the throughput-max search reselecting tp/pp differently per model as GPU count grows).

**Disproved arguments.** Cannot yet be filed as a clean U7 extension because of the model-split trend.

---

## Explained (not bugs)

### Residual-1 — llama4 / HBM4 / 8-GPU SHORT: batch high, TPS low
**Divergence.** Sim's throughput-max winner is TP=2 at **488.5/GPU** (batch 3908); the DP-pure config
gives **460/GPU** (batch 3680) = the paper's printed anchor exactly.

**Current explanation.** The cell is **capacity-bound** (tpot ≈26 ms ≪ SLO), so more batch fits
until memory fills. TP=2 wins by TP-sharding the non-expert weights into KV headroom (+~6%),
yielding a higher batch than DP-pure. Not a bug — the sim explores the full TP/PP/DP/EP space the
paper's own §III describes ("combining data, tensor, pipeline, and expert parallelism"). The DP-pure
ceiling 3680 = 8×460 is real, verified via CSV bisection (`data/p4_f8c_pure_m460` completes;
m461/m462/m464 OOM) and reproduces only under **288 GiB** capacity (decimal GB → ~428), settling
GB-vs-GiB in the code's favor.

**Disproved arguments.** The inference "⇒ the paper restricted this cell to DP-pure" is an
**unfalsifiable rationalization** (no evidence, and it contradicts the doc's own use of the same §III
quote in U7, where §III proves the paper *does* search TP/PP/EP). An untested alternative that would
produce the identical symptom without any paper-side restriction — the sim's own "TP-headroom"
mechanism being overestimated — was never ruled out. So the capacity match is solid; the causal
story "the paper restricted the search" is not established.

### U7 — HBF+/CONV+ per-GPU batch grows with GPU count instead of staying flat
**Divergence.** The sim's per-GPU batch **grows** with GPU count and is **SLO-bound** (llama4/HBF+/
SHORT: ~730 → 1370 → 1527 → 1330 → 1289 across 1/2/4/8/16 GPU); the paper's Fig-3 bars are **flat**
and **SRAM-bound** at **766.6/GPU** (llama4) and **326.1/GPU** (llama3). The sim's SRAM ceiling
(~2400/GPU at tp1) sits *above* the SLO-permitted batch, so cells never hit the 320 MB logic-SRAM
tier the paper's do.

**Current explanation.** The sim's intermediate-data SRAM gate (`peakIntermediateBytes`,
`footprint.h`) is **looser** than the paper's. It is a hand-coded, layer-scoped `max(attn, ffn)`
formula that **excludes the attention score matrix** (charged **0 bytes** on HBF+/CONV+, because
`getAttentionMemoryDuration` staging-chunks only `kv_read_size`, not `act_size`) and **omits the
KV-write on-chip staging buffer and the LM-head logits**. The physically-correct model is a
liveness-aware **step-level** peak, not a layer-scoped max. **KV-write-staging A/B — RUN 2026-07-06,
reproduces the paper's SHORT bars.** Charging the context-independent, layer-scaled KV-write burst
(`batch × 2·num_kv_heads·head_dim × layers/stage`, behind `system.kv_write_sram_gate`) under the full
optimizer+sim search reproduces all three signatures of the paper's SRAM-bound SHORT bars:

| HBF+/SHORT, side B | 1-GPU | 8-GPU | 16-GPU | paper | ratio (mav:l3) |
|---|---|---|---|---|---|
| llama4_maverick | 999 sram | 999 sram | 915 sram | 766.6 | — |
| llama3_405B | 404 slo | 390 sram | 390 sram | 326.1 | — |
| ratio per GPU | 2.47 | 2.56 | 2.35 | **2.35** | ✓ |

(1) cells flip SLO→**SRAM**-bound; (2) per-GPU batch goes **flat** across GPU count (vs side A's
~730→2149 growth); (3) the cross-model ratio lands on the paper's **2.35** even at 16-GPU where the
search could escape via tp/pp. Absolute values run a consistent **~20–30% high** (single-buffer 1×
offset; maverick/1g = 999 is byte-exact `base 136 + kvwrite 192 KB/seq`). The score-tile O(chunk) fix
is a separate correctness improvement but only ~13–19% at tp1. Full inventory, magnitudes, A/B setup,
and recipes are in `CHANGES.md`. **Disposition: leading, A/B-supported mechanism** for the paper's
SHORT SRAM-bound bars (shape + regime + cross-model ratio reproduced); two open items — the ~25%
absolute overshoot (double-buffering would over-correct, pulling below the paper) and MID/LONG (a
different, non-SRAM-bound regime, untested). The earlier "paper-internal inconsistency, disposition
FINALIZED" headline is **retracted**.

**Disproved arguments.**
- **(a) O(context) full-score-matrix charge.** A/B tested (`ab-score-accounting` worktree): it
  reproduces the one flat llama4/SHORT bar but regresses llama3-MID / llama3-LONG / llama4-MID by
  2–3× — no context-scaled score charge fits all six HBF+ bars.
- **(b) The "impossibility proof."** As originally stated it used **stale pixel bars** (SHORT 854.7,
  MID 745.1). With vector-exact values (766.6, 742.0) the arithmetic is sound and the margin is **~6×**
  (not the "1.5× near-miss" a sixth-pass self-critique claimed) — BUT the proof only excludes
  O(context)-**linear** charges. It does **not** exclude a **context-independent** term, which is
  exactly what KV-write staging is. So it does not prove "no accounting fits" and cannot finalize U7.
- **(c) The KV-write *analytic* argument was over-stated (refuter panel) — but the *A/B run*
  vindicates the mechanism.** The panel's methodological critiques stand: the 187/492 MB formula was
  mis-cited to `getKVWriteDuration` (context-*dependent*; the correct layer-scaled form is
  `kvWriteStagingBytes`, now implemented); the cross-model ratio is driven by `num_layers` (48 vs 126,
  per-layer KV bytes identical); and it is SHORT-only. What the panel could not do — and the A/B now
  does — is TEST it: charging the term reproduces the paper's SHORT flat/SRAM-bound bars and the 2.35
  ratio under the real search (see the table above). So "num_layers drives the ratio" is the
  *mechanism*, not a flaw, and the standing caveats are the ~25% absolute overshoot and MID/LONG — not
  "unverified." NOTE (correction): paper1 does NOT explicitly state that the KV-write staging buffer
  shares the 320-MB intermediate-data SRAM — its footnote 1 ("writing intermediate data increases
  write traffic ... every decode iteration") uses "intermediate data" for the short-lived activations
  it keeps in SRAM, and refers to the flash-written KV as "KV cache" (footnote 2), i.e. as distinct
  terms. Charging the KV-write burst against the 320-MB pool is a physically-reasonable MODELING
  CHOICE (the KV must buffer on-chip somewhere before the flash write); the evidence that it matches
  the paper's tool is the A/B reproduction above, NOT the footnote.
- **(d) "327 reconciled exactly."** `320 MiB/(2·40·6399·2 B) = 327.7` is arithmetically right but
  overfit — llama3's own SHORT/HBF+ bar (326.1) is a zero-parameter explanation comparably close.
- **(e) The maverick-tp8 SRAM-ceiling impact numbers** carried in the 7th/8th-pass tables were wrong
  (a reproduction-script bug — missing `/e_tp` on the MoE term); the tp1 numbers were correct.

### Fig-5 — llama3_405B / MID / HBF+: runtime-breakdown share mismatch — the MFU=1.0 assumption (deferred pair)
**Divergence.** Shares (flat 4/8/16 GPU): FFN 18.3% sim vs 31.8% paper, Comm 7.2 vs 1.9, Attention
64.2 vs 55.6, KV-write 10.2 vs 9.4 (matches).

**Explanation (investigator + refuter pair, 2026-07-06).** NOT an FFN bug — FFN's absolute time is
roofline-EXACT (18.26 ms sim = hand-computed, compute-bound). The driver is the **MFU=1.0 assumption**
(`linear_impl.cpp::effectiveMFU`, default 1.0 — the sim runs every GEMM at 100% peak, but real decode
GEMMs run ~0.5). Setting the realistic **MFU≈0.5** (share-implied 0.479) moves all four to near-
agreement: FFN→31.4% (31.8), Attention→58.7% (55.6), Comm→2.2% (1.9, via the winner flipping
TP=4→TP=2), KV→7.6% (9.4). A single lever explains the figure. This is the **deferred score/MFU pair**
(below) — not applied by the no-tuning rule (the MFU value would be fitted). Same mechanism likely
underlies the Fig-6 ordering-flip item.

**Disproved arguments.** "FFN absolute time is under-counted (a coding bug)" — refuted (roofline-exact
at MFU=1.0). "Three independent mismatches" — refuted (one MFU lever moves all four).

### Fig-4 — CONV/CONV+ / llama4_maverick, low GPU: sim TPS 55–89% "off" — paper-extraction artifact
**Divergence.** At low GPU counts CONV/CONV+/llama4 per-GPU TPS diverges 55–89% (e.g. CONV/MID/1-GPU:
sim batch 39 at tpot 99.47 ms).

**Explanation.** A non-divergence for two reasons: (1) the sim is physically defensible — decode is
flash-weight-stream-bound, the breakdown sums exactly to the 99.474 ms step, CONV's HBM read is masked
under `std::max()` (`layer_impl.h:72,141`) so it's if anything conservative; residual genuine gap
≤~20% (sim batch and the paper's own Fig-3 batch agree to 0–22%). (2) **The paper reading is
unextractable** — at low GPU the CONV/CONV+ Fig-4 bars are visually near-identical / overlapping, so
the read is unreliable and the "error" is a paper-extraction artifact. The 15 affected cells
(CONV/llama4 at 1/2/4 GPU + CONV+/llama4 at 1/2 GPU, × SHORT/MID/LONG) are **excluded from
`compare_error_rates.py`** (`is_unextractable`); dropping them cuts Fig-4 max error 87%→70% (remainder
is a *llama3* CONV cell, outside scope).

**Disproved arguments.** The earlier "paper is provably self-contradictory, CONV+ is clean" proposal
used **stale pixel data** — with vector-exact data "8.2× over SLO" becomes 1.63× and "CONV+ is clean"
is false (12 cells >25%, split 6 CONV / 6 CONV+). Only "the sim is not the problem" survives.

### Fig-7 — online≈offline "duplicate" (1-GPU): closed by the intermediate-data gate
**Divergence.** The paper prints identical online/offline PEC for llama4/SHORT/HBF+ at 1-GPU
(185377=185377) and 8-GPU (424283=424283). Pre-merge baseline sim: 8-GPU near-duplicate (1.2% apart),
but 1-GPU a real **57%** gap (online SLO-bound batch 1071 vs offline SRAM-bound batch 2824).

**Explanation, CONFIRMED on the merged default build (2026-07-06 — was out-of-sample on a branch,
now reproduced on `main` itself after the faithful-intermediate-gate merge, CHANGES.md item 76/78).**
With the full intermediate-data accounting now the default (unconditional) behavior, the 1-GPU online
cell is **SRAM-bound at 844**, so online == offline exactly at both 1-GPU (844=844) and 8-GPU
(6752=6752) — the 57% gap collapses to 0%, reproducing the paper's exact duplicates at both GPU
counts. A pure capacity change (KV-write *timing* share byte-identical). So "1-GPU is a paper
misprint" is **rejected** — it's a real SRAM-saturation effect the sim now reproduces by default.

---

## Deferred — real, quantified, deliberately NOT applied (avoid tuning-to-target)

### Score/query HBM-traffic removal + GEMM-efficiency (MFU<1) — both or neither
The non-flash attention memory charge includes `m·n·G` (score) + `m·k·G` (query) bytes
(`attention_gen_impl.cpp`). With FlashAttention the score matrix never materializes in HBM, so it is
physically over-charged — a constant ~11% (llama3) / 3.76% (llama4) fraction of attention memory.
Removing it alone drives 3 cells to −8…−11%; pairing it with a fitted MFU≈0.45–0.5 (infra exists:
`mfu_max`/`mfu_m_half`) is expected to bring every cell within a few % — but the MFU would be fitted
to Fig-5, so **deferred**. Notes: score bytes are **0 on HBF+/CONV+** and sub-`max()` on HBF (live
only on HBM4's sum path); `use_flash_attention` is structurally MLA-only, so charging score bytes on
the GQA path is plausibly paper-conformant; the MFU knob's per-op-`m` miskey was fixed (`CHANGES.md`
item 64), a no-op at defaults but a precondition for any future MFU work.

### Expert-routing skewness (`config.yaml: skewness: 0.8`, unstated by the paper)
Every Maverick MoE cell runs with a Zipf skew of 0.8 the paper never states. The effect is a
**coupon-collector hump** (not monotone): peaks ~13–15% around batch 100–500 and saturates to ~0% by
batch 1500–2500 (all experts active once the system batch is large, since tokens pool across all DP
replicas). Real exposure is confined to llama4/LONG at low GPU count (+8.6% at 1-GPU/batch 100).
**No paper ground truth exists** to fix toward 0.0 (uniform) or 0.8 — both unanchored — so surface-
only, no code change. (Distinct from the already-fixed hot-expert *placement* bug, `CHANGES.md` item
50.)

---

## Moved to CHANGES.md / ledgers (2026-07-06 cleanup)
- **Resolved paper-comparison items** (U1, U2, U4, U5, U6, U8, U9, Residuals 2–4, Fig-6 TPS+Batch
  extraction) — full resolution records in `CHANGES.md` ("Paper-comparison items resolved"). U5 was
  resolved to a verified *agreement* (no code fix), moved 2026-07-06.
- **U9 reproduction recipe** and the **"Ruled out as causes"** list → `CHANGES.md` (consolidation
  section, same date).
- **U7 7th/8th/9th-pass detail** (score-tile O(chunk) implementation recipe, full uncounted-
  intermediate inventory with magnitudes, six-agent refuter verdicts) → `CHANGES.md`; raw per-pass
  adjudication → `ledgers/FINDINGS_REGISTER.md`.
- The verbose per-session-note history that used to head this file was process narrative; the
  substantive fixes it described are all in `CHANGES.md` items 30–75.
