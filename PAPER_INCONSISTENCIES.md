# Paper Inconsistencies — HBF-LLMSimulator vs. Son et al. (IEEE CAL 2026)

Single-source tracker for every place our simulator's results have been checked against the
paper's reported numbers/claims. Supersedes `INCONSISTENT_WITH_PAPER.md` and
`INCONSISTENCY_POSSIBLE_FIXES.md` (both merged in here and removed). For full fix-implementation
detail on any resolved item, see the referenced `CHANGES.md` item number. For the original
external audit this work grew out of, see `FAIRNESS_AUDIT.md`. The primary source
(`Exploring_High-Bandwidth_Flash_for_Modern_LLM_Inference_Opportunities_and_Challenges.pdf`, in
this repo) was read in full when compiling this document — claims sourced from it are cited
directly rather than through intermediary docs.

## Resolved

Each item: what the mismatch was, its root cause, how it was fixed, one line each.

1. **HBF+ produced a LOWER total batch than HBF** (paper: HBF+ should be larger, "+24% on
   average"). Root cause: the scarce-tier (SRAM/HBM) capacity gate summed every intermediate
   tensor as if co-resident instead of taking peak concurrently-live footprint; the dominant
   (wrongly-summed) term was seq-length-scaled data that should never have counted at all. Fixed
   via a shared `peakIntermediateBytes()` helper taking `max(attention-phase, FFN-phase)`.
   → `CHANGES.md` item 14.
2. **Reported per-GPU batch size was wrong** (divided by DP-replica count instead of total GPU
   count, penalizing configs for using DP more effectively). Fixed: divide by total GPU count in
   all 4 call sites in `run_experiments.py`, matching `INSTRUCTIONS.md`'s own TPS-metric
   convention. → `CHANGES.md` item 15.
3. **llama3_405B wrongly showed 1-/2-GPU HBM4 as feasible** (paper: "no 1-/2-GPU segments in all
   HBM4 bars"). Root cause: `precision_byte` was set to FP8 (~405 GB) instead of BF16 (~810 GB);
   FP8 fits under 2×288 GB, BF16 doesn't. Fixed the model preset. → `CHANGES.md` item 11.
4. **llama4_maverick's HBM4/8-GPU/LONG batch anchor was 68% low** (10.0 vs. paper's 31). Root
   cause: every attention layer was modeled as full/global; real Llama-4 uses "iRoPE" interleaved
   local/global attention (only every 4th layer is global; others use an 8192-token window) —
   over-counted KV footprint ~3x at long context. Fixed by modeling the windowing throughout the
   capacity, KV-cache-allocation, and runtime KV-read paths. Now 33.0 vs. paper's 31 (~6% off).
   → `CHANGES.md` item 16.
5. **Flash page-read latency was charged once per SRAM-staging chunk instead of once total**
   (double-buffering should hide all but the first chunk's latency). Inflated HBF+'s KV-read time
   ~49% above HBM4's despite identical nominal bandwidth. Fixed to expose exactly one page latency
   (pipeline-fill) plus any residual where a chunk's transfer time is shorter than the page
   latency. → `CHANGES.md` item 17.
6. **MoE "experts per device" activation term divided by total device count instead of
   devices-per-pipeline-stage** (8x undercount for degenerate one-device-per-stage configs). Fixed
   to divide by `devices_per_stage`, matching the (already-correct) weight-term pattern. Confirmed
   this does NOT explain the still-open U1 anchor gap below.  → `CHANGES.md` item 18.
7. **Linear weight-read charged page-read latency once per op with no reference to chunk
   size/SRAM capacity at all**, unlike the KV-read path, which does chunk/double-buffer (audit
   F2). The paper's methodology describes double-buffering generically for "read data," not
   scoped to KV — leaning toward this being a real (if minor) inconsistency with the paper's
   stated model. Fixed: `getLinearMemoryDuration` (`src/hardware/layer_impl.h`) now uses the same
   chunked double-buffer formula as KV reads. **Verified numerically as a no-op on every current
   preset** (chunk-transfer time at full SRAM capacity exceeds page latency for all 4 flash
   presets, so exposed latency reduces to exactly one page latency either way) — confirmed
   empirically via byte-identical before/after runs on both HBF (`0.0966962731654616s`) and CONV,
   the highest-page-latency preset (`0.09089014626032631s`). Kept for structural correctness and
   forward-safety if config constants ever change, even though it moves no current number.

## Still open

### U1 — llama4_maverick's HBM4/8-GPU SHORT & MID batch anchors ~9-12% high

SHORT: our 514.9 vs. paper's 460 (×1.12). MID: our 164.8 vs. paper's 151.5 (×1.09). Unaffected by
the iRoPE fix (item 4 above — both contexts are shorter than the 8192-token local-attention
window). Two hypotheses ruled out: collision-reduced active experts (numerically negligible idle
experts at these batch sizes) and an artificial capacity-reservation "fix" (rejected as circular —
no such convention exists in the paper).

**Root-caused to a structural inconsistency in the paper's own reported ratio, not a fixable bug
in our formulas.** Under any pure weight+KV capacity model, `batch_per_gpu = (C − weight_per_gpu)
/ (K · ctx)`, where weight is context-independent — so the SHORT:MID batch ratio must exactly
equal `ctx_MID/ctx_SHORT = 6399/2033 = 3.148`, for *any* model or parameter values (the weight and
capacity terms cancel algebraically). Our simulator's ratio (514.9/164.8 = 3.124) is close to that
mandated value; **the paper's own ratio (460/151.5 = 3.036) is not** — no choice of capacity,
weight, or KV values under a pure-capacity model can produce it. Independently, our weight
footprint for llama4_maverick (746.4 GiB, computed bottom-up from all 128 resident experts) is an
exact match to the paper's own stated 746 GB, and the workload token lengths (`⟨1660,373⟩`,
`⟨5900,499⟩`) are confirmed verbatim against the paper's Methodology section (§III) — ruling out
workload-definition mismatch as the cause.

The paper's own text supplies a candidate mechanism: §IV's Fig. 3 discussion states short-context
batch sizes are limited by "(i) inter-GPU communication increas[ing] almost linearly with batch
size... under the TPOT SLO; (ii)... FFN execution shift[ing] toward compute-bound GEMM
operations... under the SLO" — i.e., the paper's own systems (including HBM4) are latency-bound by
communication/compute effects, not purely capacity-bound. **A batch sweep at this exact operating
point (500→5000, executed this pass) confirms our simulator does not exhibit this mechanism**:
TPOT at the anchor batch uses only ~26% of the SLO budget (deeply capacity-bound, hitting OOM at
batch≈4300 long before TPOT would approach the SLO), communication stays ≤2% throughout, and FFN's
*share* of decode time actually **shrinks** with batch (76.5%→34.4%) — the opposite of the paper's
described compute-bound-FFN mechanism. Our comm/FFN-compute cost models are structurally milder
than whatever produces the paper's numbers.

**No safe fix identified.** Inflating communication or FFN-compute cost terms purely to match the
paper's two numbers would repeat the same "reverse-engineer a fudge factor from the answer"
problem already rejected for the capacity-reservation proposal. A real fix would need either a
documented real-hardware basis for stronger communication/compute costs at scale (kernel launch
overhead, real collective-communication overhead beyond pure bandwidth+latency, imperfect compute
utilization — none currently modeled or sourced here), or direct access to the paper's own
methodology section, neither available in this repo.

### U2 — "4-GPU HBF+ TPS is 15% higher than 8-GPU HBM4" residual (llama4, LONG workload)

Paper: HBF+/4-GPU per-GPU TPS should be ~1.15× of HBM4/8-GPU's. The page-read-latency
double-buffering fix (item 5 above) moved the measured ratio from 0.53× to 0.72× — real progress,
but the paper's 1.15× is still not reached.

**Leading hypothesis tested and refuted this pass.** Both operating points are ~70% attention
time, and the decode attention KV-read is provably symmetric in code between HBM4 and HBF+ (same
windowed KV term, identical 12.8 TB/s bandwidth, page latency negligible post-fix) — yet HBF+
sits at ~1362 attention-TPS/GPU vs. HBM4's ~2586, a gap the symmetric read model alone can't
produce. The leading candidate was a parallel-config asymmetry: the optimizer's `"max"` latency
model scales every per-layer cost term by `1/pp` with no penalty, even though the simulator
doesn't deliver real cross-step pipeline overlap for single-token decode — if HBM4 were forced
into low-PP by its tight capacity limit while HBF+ (capacity-rich) picked high-PP, HBF+ would pay
for an unrealized "1/pp" benefit its analytic estimate assumed.

**Confirmation run (executed this pass): REFUTED.** Captured the optimizer's actual chosen
configs: HBF+/4-GPU picked `TP=1, PP=4, DP=1, EP=1`; HBM4/8-GPU picked `TP=1, PP=8, DP=1, EP=1` —
both are max-PP *relative to their own GPU count* (the same structural choice), not the
asymmetric low-PP/high-PP split the hypothesis needed. Neither pays communication (both are 100%
pipeline-parallel with zero tensor/data/expert sharding). The KV-write penalty (see U6, below)
remains a confirmed, real, ~14.8%-of-decode-time contributor, but was already known to be
insufficient alone to close the gap. **The actual mechanism behind the residual 0.72×→1.15× gap
is unknown** — this pass eliminated one well-argued candidate but found no replacement. No code
fix applied for U2.

### U4 — "HBF+ always outperforms HBM4 at 16 GPUs, across all SLOs" (llama4, LONG workload)

Paper: at 16 GPUs, LONG workload, HBF+'s per-GPU TPS should exceed 8-GPU HBM4 across every tested
SLO (0.05s, 0.1s, 0.2s, offline). The page-latency fix (item 5) moved the 0.1s-SLO ratio from
0.75× to 1.01× — but that check covered only one of the four SLOs.

**Full 4-SLO sweep result:**

| SLO | 0.05s | 0.1s | 0.2s | Offline |
|---|---|---|---|---|
| norm. TPS (HBF+/HBM4) | 0.96× | **1.01×** | **1.04×** | 0.85× |

**Holds at 0.1s and 0.2s only; misses at the tightest (0.05s, −4%) and loosest (offline, −15%)
ends** — narrower than the paper's "always outperforms" claim. Not root-caused further this pass.

### U8 — llama4_maverick/HBF+'s TPS edge over HBM4 shrinks at the offline SLO instead of growing (new finding)

Surfaced by the same full-SLO sweep as U4, not one of the originally-tracked items. The paper's
own Fig. 6 narrative states "the more relaxed the SLO, the greater HBF's benefit" (monotonically
increasing), citing its own example of 4.1%→14.8% at 16 GPUs for Llama4. Our measured data
contradicts monotonicity specifically for this one combination:

| Series (16-GPU, normalized to 8-GPU HBM4) | 0.05s | 0.1s | 0.2s | Offline |
|---|---|---|---|---|
| **llama4/HBF+ (the anomaly)** | 0.96× | 1.01× | 1.04× | **0.85×** |
| llama4/HBF, for contrast | 0.82× | 0.87× | 0.89× | **0.90×** |
| llama3/HBF+, for contrast | 1.15× | 1.34× | 1.43× | **1.47×** |

Only llama4×HBF+ dips at the offline end; the same model's HBF (1 reserved HBM stack) and the
same memory type's dense model (llama3) both climb toward offline as expected. The anomaly is
specific to the **MoE model × all-flash (no reserved HBM) config** combination at very large
(offline-unlocked) batch sizes. A second oddity alongside it: llama4/HBF+'s offline-SLO per-GPU
batch is *lower* at 16 GPUs (580.8) than at 8 GPUs (676.4) — batch dropping as GPU count
increases, itself unusual and possibly the same underlying mechanism (a capacity-split or
parallelism-ranking artifact unique to this combination at extreme batch — not yet investigated).
**Not root-caused.**

## Explained — not bugs

### U3 — Degenerate one-device-per-stage optimizer choices (e.g. PP=8/EP=1/TP=1)

The optimizer can select configs like llama4_maverick's 8-GPU choice (PP=8/EP=1/TP=1) that
trivially zero every communication term by construction, raising the question of whether the
latency ranking is biased toward such degenerate configs relative to a middle-ground split (e.g.
PP=2/EP=4).

**A/B check executed this pass: ranking validated, not a bug.** Forced both PP=8/EP=1 (the
optimizer's actual choice) and PP=2/EP=4 (the rejected alternative) via direct `distribution.*`
overrides at llama4/HBM4/8-GPU/SHORT. At the literal reported anchor batch, PP=2/EP=4 doesn't
even fit (OOM) — PP=8/EP=1 is the only feasible option there. At a smaller batch where both fit:

| Config | Weight | Cache | Measured TPOT |
|---|---|---|---|
| PP=8/EP=1/TP=1 (optimizer's choice) | 94.75 GB | 176.82 GB | **0.02449 s** |
| PP=2/EP=4/TP=1 (rejected alternative) | 103.21 GB | 176.82 GB | 0.02563 s |

Cache is identical between the two (confirming the EP/PP weight-cancellation algebra); PP=2/EP=4's
weight is ~8.9% higher (confirming that non-expert weight scales with `layers_per_stage`, which is
4× larger for PP=2, and isn't reduced by EP). Neither pays any communication. **The optimizer's
chosen config really is faster in the live simulator (~4.6%), not just by its own analytic
estimate** — the ranking is validated for this pair. This narrows but doesn't fully close the
general question of whether the ranking could still be biased for some other operating point.

### U5 — "1-GPU HBF/HBF+ per-GPU batch > 8-GPU HBM4, in most cases" (llama3_405B miss)

Holds 5/6 for llama4_maverick but 0/6 (pre-fix) for llama3_405B — e.g. SHORT: 1-GPU HBF = 75 vs.
8-GPU HBM4 = 194.8 (−61%). **Root-caused as physically correct, not a bug**, and confirmed
airtight by the simulator's own instrumentation. **Full derivation: see
`WHY_HBF_1GPU_BATCH_IS_LOW.md`.** Summary: a >400B-parameter dense model at 1 GPU (TP=PP=1, no
sharding possible) must re-stream its entire weight every decode step, consuming ~63-76% of the
100ms online SLO before any per-sequence cost — an SLO-bound, not capacity-bound, operating point
(capacity ceiling ≈2,956 sequences vs. the reported 75). The paper's own text states this exact
distinction generally: *"HBM4's per-GPU TPS and batch size hardly change across SLOs due to the
memory-capacity bottleneck, whereas HBF and HBF+ significantly benefit from relaxed SLOs by
leveraging their large capacity"* (§IV, "Effect of SLO"). No code change proposed.

### U6 — HBF+ KV-write "unhidden" overhead (14.7%@4-GPU → 19.8%@16-GPU of decode time; 0% for HBM4)

**Confirmed directly consistent with the paper, not inferred.** The paper's own footnote 2 (p. 3)
states verbatim: *"Writing the KV cache newly generated during the decode phase can be overlapped
with computation in the attention layer"* — exactly the mechanism our code implements (hide
against attention-only compute). The paper's Fig. 5 discussion further states KV-cache writes
account for "5–13.9% of the execution time in Llama4," increasing "with more GPUs, shorter
contexts, and the MoE model" — qualitatively matching our measured range (somewhat higher in
absolute magnitude, not alarming given different exact cells compared). Not a bug; a confirmed
contributing factor to U2's residual, exactly as the paper's own Takeaway 3 frames it.

## Ruled out as causes (for completeness)

Flash bandwidth constants and llama4's weight footprint (746.4 GiB) match the paper's Table I and
stated figure exactly. Routing skew (`skewness: 0.8`) doesn't reach U1's capacity-bound operating
point. Parallelism-ranking communication cost is ≤0.6% of decode time in every checked config
(confirmed further by the U3 A/B check: zero communication either way). The MoE
activation-divisor bug (resolved item 6 above) is real but confirmed NOT the cause of U1 (SHORT/
MID anchors byte-identical before/after that fix). `BUGS_HIDDEN_BY_FLAGS.md` entries are all
masked by pinned config flags (`decode_mode: on`, `disagg_system: off`, `parallel_execution: off`,
`chunk_size: 0`) and confirmed to have zero effect on any current paper comparison.
