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

### Fig-4 — CONV / llama4_maverick, low GPU count: sim TPS 55–89% off
**Divergence.** At low GPU counts CONV/llama4 per-GPU TPS diverges 55–89% (e.g. CONV/MID/1-GPU: sim
batch 39 at tpot 99.47 ms). The gap is specific to CONV (not CONV+) at 1–2 GPU.

**Current explanation.** The simulator is very likely correct — decode is flash-weight-streaming-
bound and the sim's throughput is physically defensible (breakdown at CONV/MID/1-GPU sums exactly to
the 99.474 ms step; `expert_ffn` 70.7%; CONV's 1-HBM read bandwidth is masked under `std::max()`
against the dominant flash weight-stream, `layer_impl.h:72,141`, so if anything the sim is mildly
conservative). The residual genuine gap is bounded ≤~20% (the sim's batch and the paper's OWN Fig-3
batch agree to 0–22%, both SLO-bound). What is NOT established is *why* the paper's Fig-4 CONV curve
sits where it does. Not covered by any resolved item.

**Disproved arguments.** A drop-in "RESOLVED: paper figures are provably self-contradictory, CONV+ is
clean" proposal (drafted in the removed `bughunt-fifth-pass` worktree) was **rejected** — it used
**stale pixel readings**. With vector-exact data its evidence collapses: "CONV implies 1.2×–8.2× over
SLO" becomes **1.09×–1.84×** (its "worst case" MID/1-GPU 8.2× used tps 44.1, actually 202.3 →
1.63×); and "CONV+ is clean" is **false** — 12 cells exceed 25% error, split 6 CONV / 6 CONV+. So the
"paper is provably self-contradictory" framing does not hold; only Pillar 1 (sim is not the problem)
survives.

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

### Fig-5 — llama3_405B / MID / HBF+: runtime-breakdown share mismatch
**Divergence.** Breakdown shares (flat across 4/8/16 GPU): **FFN** 18.3% sim vs 31.8% paper (sim
under-counts), **Communication** 7.2% vs 1.9% (sim over), **Attention** 64.2% vs 55.6%; **KV-write**
matches (10.2% vs 9.4%).

**Current explanation.** Most likely **one** root cause — FFN absolute time under-counted by ~40% —
mechanically inflating the other shares (shares sum to 100%). Candidate: the deferred MFU under-
charge, but its leverage on HBF+ is weak (HBF+ FFN is flash-bandwidth-bound, not GEMM-bound, so an
MFU<1 correction that only touches FLOP-bound time moves it little). Not yet root-caused.

**Disproved arguments.** The "three independent mismatches (FFN + Communication + Attention)" framing
overstates it — they are not independent; correcting FFN alone would move all three shares.

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

### Fig-7 — online≈offline "duplicate": 8-GPU genuine, 1-GPU open
**Divergence.** The paper prints **identical** online/offline PEC for llama4/SHORT/HBF+ at 1-GPU
(185377 = 185377) and 8-GPU (424283 = 424283). Sim: **8-GPU** near-duplicate (online 23743 vs offline
24026 tps/GPU, 1.2% apart); **1-GPU** a real **57%** gap (online 10711 slo-bound batch 1071 vs offline
16810 sram-bound batch 2824).

**Current explanation.** 8-GPU = genuine SRAM saturation (explained — loosening the SLO barely moves
a batch already near the ceiling). 1-GPU = open: either a paper misprint, **or** an under-modeled
SRAM term. If the KV-write staging term (U7) were charged, the 1-GPU online ceiling would collapse
to ~559–999/GPU and could flatten all four operating points into the paper's duplicate without any
misprint.

**Disproved arguments.** "8-GPU is sram-bound in both directions" is **wrong** — the probe log shows
8-GPU *online* is `bound=slo`. "1-GPU is definitely a paper misprint" is **premature** — the KV-write
A/B (U7, in progress) is a competing sim-side explanation that hasn't been run to conclusion.

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

### U5 — "1-GPU HBF/HBF+ per-GPU batch > 8-GPU HBM4, in most cases"
**Divergence.** The paper claims 1-GPU flash beats 8-GPU HBM4 per-GPU batch "in most cases." The sim
matches **11 of 12** comparable cells; the single miss is llama3_405B/SHORT/HBF (sim 171.5 < paper
194.0).

**Current explanation.** The weight-reread floor (`getLinearMemoryDuration`: weight time =
`k·n·precision`, no batch factor) means flash's larger capacity lets a 1-GPU flash node hold more
batch than an 8-GPU HBM4 node whose weights are replicated. The one miss is consistent with the
paper's own "in most cases" hedge. Blind re-derivation matched the mechanism to 0.3%.

**Disproved arguments.** The earlier "U5 is a real miss" reading dissolved — the current binary
matches the paper's OWN 1-GPU bars (the old comparison table was stale).

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
  "unverified." Paper1's own footnote 1 calls KV-cache writes "intermediate data," so this is arguably
  the paper's stated mechanism, not a new one.
- **(d) "327 reconciled exactly."** `320 MiB/(2·40·6399·2 B) = 327.7` is arithmetically right but
  overfit — llama3's own SHORT/HBF+ bar (326.1) is a zero-parameter explanation comparably close.
- **(e) The maverick-tp8 SRAM-ceiling impact numbers** carried in the 7th/8th-pass tables were wrong
  (a reproduction-script bug — missing `/e_tp` on the MoE term); the tp1 numbers were correct.

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
- **Resolved paper-comparison items** (U1, U2, U4, U6, U8, U9, Residuals 2–4, Fig-6 TPS+Batch
  extraction) — full resolution records in `CHANGES.md` ("Paper-comparison items resolved").
- **U9 reproduction recipe** and the **"Ruled out as causes"** list → `CHANGES.md` (consolidation
  section, same date).
- **U7 7th/8th/9th-pass detail** (score-tile O(chunk) implementation recipe, full uncounted-
  intermediate inventory with magnitudes, six-agent refuter verdicts) → `CHANGES.md`; raw per-pass
  adjudication → `ledgers/FINDINGS_REGISTER.md`.
- The verbose per-session-note history that used to head this file was process narrative; the
  substantive fixes it described are all in `CHANGES.md` items 30–75.
