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
8. **Batch-size search non-monotonicity bug — `run_experiments.py` could substantially
   under-report the true max batch** (surfaced while root-causing U8 below; the "676→580 batch
   drop" flagged there was this bug, not a real effect). A single failing batch probe near the
   search boundary was trusted as proof of infeasibility, but the discrete divisibility
   constraint on parallelism configs (`batch_size % dp == 0`) makes feasibility non-monotonic in
   batch at the integer level — an unlucky probe value can route to an inferior config and fail
   even though neighboring, equally-large values succeed with the true-best config. Fixed by
   replacing every single-point probe with an 8-value window scan. llama4_maverick/HBF+/16-GPU/
   LONG/offline: reported per-GPU batch corrected from 580.75 to 684.12 (now flat vs. the 8-GPU
   value of 676.38, as physically expected — see U8 below). → `CHANGES.md` item 21.
9. **No compute-utilization (MFU) factor existed anywhere in the roofline compute term** (every
   compute-bound GEMM was implicitly charged at 100% of peak FLOPs). Added a saturating
   `MFU(M) = mfu_max*M/(M+mfu_m_half)` factor (default no-op) across every compute-duration call
   site. This *by itself* does not resolve any open item (see U1 below) but is a real, defensible
   gap the audit against the paper's own stated mechanism (Fig. 3's aspect (ii)) surfaced.
   → `CHANGES.md` item 19.
10. **Optimizer's TP all-reduce term used a flat, non-ring communication formula**, diverging
    from the live simulator's proper ring all-reduce (`2*(N-1)/N` bandwidth term, `2*(N-1)`
    latency hops). Aligned the optimizer to the simulator's exact formula. Ranking-only fix (F1) —
    verified to not move any TP=1 paper anchor, since every anchor investigated in this document
    happens to use TP=1 (zero communication) once EP/PP are accounted for.  → `CHANGES.md` item 20.

## Still open

### U1 — llama4_maverick's HBM4/8-GPU SHORT & MID batch anchors ~9-12% high

SHORT: our 514.9 vs. paper's 460 (×1.12). MID: our 164.8 vs. paper's 151.5 (×1.09). Unaffected by
the iRoPE fix (item 4 above — both contexts are shorter than the 8192-token local-attention
window). Two hypotheses ruled out: collision-reduced active experts (numerically negligible idle
experts at these batch sizes) and an artificial capacity-reservation "fix" (rejected as circular —
no such convention exists in the paper).

**CORRECTION (this pass): the previously-recorded root cause below was wrong. Retracted.**

The earlier analysis in this file claimed: *"Under any pure weight+KV capacity model,
`batch_per_gpu = (C − weight_per_gpu) / (K · ctx)`... the SHORT:MID batch ratio must exactly
equal `ctx_MID/ctx_SHORT = 3.148`... the paper's own ratio (460/151.5 = 3.036) is not [reachable
under a pure-capacity model] — no choice of capacity, weight, or KV values under a pure-capacity
model can produce it,"* concluding this was *"a structural inconsistency in the paper's own
reported ratio, not a fixable bug in our formulas."*

That reasoning is **only valid if both SHORT and MID are capacity-bound in the paper**. They are
not. The paper is explicit that short-context batch scaling is **latency/SLO-bound**, not
capacity-bound. §IV (the "Fourth" observation), verbatim, in full (the earlier analysis quoted
only two of three clauses):

> "In general, LLM-serving systems can support larger batches for shorter-context queries, which
> limits HBF's benefits in three aspects; (i) the inter-GPU communication increases almost
> linearly with batch size, making it difficult to further scale the already large batch sizes in
> short-context workloads under the TPOT SLO; (ii) in the dense model, FFN execution shifts toward
> compute-bound GEMM (general matrix multiplication) operations, which also limits further
> batch-size scaling under the SLO; (iii) a larger batch size also linearly increases the peak
> size of intermediate data, so the limited SRAM-buffer capacity (e.g., 40 MB) bottlenecks
> batch-size scaling in HBF+."

The paper's *only* "capacity-bottleneck" statement is scoped to the **LONG** workload / SLO sweep,
not SHORT/MID. §IV "Effect of SLO", verbatim: *"HBM4's per-GPU TPS and batch size hardly change
across SLOs due to the memory-capacity bottleneck, whereas HBF and HBF+ significantly benefit from
relaxed SLOs by leveraging their large capacity."* The earlier analysis conflated this LONG-only
statement with the SHORT/MID anchors and concluded the paper contradicts itself. Since the paper
itself says SHORT is SLO-bound, its SHORT:MID ratio is under no obligation to equal the
context-length ratio — 3.036 is perfectly consistent with the paper's own framing, and there is no
paper self-inconsistency to point at.

**Corrected root cause: our simulator's cost model is too optimistic in exactly the two ways
the paper's own mechanism (i)/(ii) name**, so our SHORT/MID anchors never leave the capacity-bound
regime the paper describes only for LONG:

- **(i) communication**: our comm bandwidth term does scale linearly with batch (confirmed by
  code inspection — `communication.cpp`'s `AllReduce`, `parallelism_optimizer.cpp`'s aligned ring
  term, item 10 above), but the optimizer's chosen 8-GPU config for this operating point is
  TP=1/PP=8/DP=1 (confirmed via direct stdout trace — the same degenerate one-device-per-stage
  choice already investigated in U3 below), which zeroes every communication term by construction
  (`AllReduce` costs nothing at TP=1; the PP send/recv term is a fixed per-transition cost, not
  the batch-linear all-reduce mechanism (i) describes). Comm contributes ~0.
- **(ii) compute**: confirmed via item 9's diagnostic — **there was no compute-utilization factor
  anywhere**; every compute-bound op was charged at ideal 100% of peak FLOPs, so FFN never
  "shifts toward compute-bound GEMM under the SLO" the way the paper describes; the roofline
  `max(compute, memory)` keeps decode memory-bound far longer than real hardware would. (Note:
  the paper scopes (ii) explicitly to "the dense model" — llama3_405B, not the MoE llama4_maverick
  this item concerns — so this is general roofline realism rather than a literal citation match.)

At the SHORT anchor (batch=4119 total / 514.9 per-GPU, TP=1/PP=8/DP=1), TPOT=25.9ms is only ~26%
of the 100ms SLO, and communication is exactly 0 (confirmed via CSV breakdown). A hand-derived
roofline check (Rubin peak=8.75 PFLOP/s FP16, HBM4 12.8 TB/s, model dims from `model_config.h`)
shows the MoE routed-expert GEMMs (~4 tokens/expert at top_k=1/128 sparsity) are deeply
memory-bound (compute ≈170x smaller than memory), while the always-active shared-expert FFN
(full 514.9-token batch) sits closest to the roofline crossover (compute≈65% of memory at
`mfu_max=1.0`) — the first op that would flip compute-bound as MFU degrades.

*(Bookkeeping: after `CHANGES.md` item 21's search-boundary-window fix — found later, while
investigating U8 — the SHORT anchor itself shifts slightly, 514.9→518.3 batch/GPU (4119→4146
total, ~0.65%), the same divisibility-driven search artifact as U8, just a much smaller magnitude
at this operating point. Does not change any conclusion above; all MFU-sweep numbers below predate
item 21 and use the pre-fix 514.9/164.8 baseline for comparability across the sweep.)*

**MFU sensitivity sweep (this pass, item 9's model, `mfu_m_half=128` — a stated tensor-core-tile-
granularity assumption, not tuned to any target number):**

| `mfu_max` | SHORT batch/GPU | SHORT TPOT | MID batch/GPU | MID TPOT |
|---|---|---|---|---|
| 1.0 (baseline) | 514.9 | 0.0259s | 164.8 | 0.0237s |
| 0.7 | 518.3 | 0.0319s | 164.8 | 0.0289s |
| 0.6 | 518.3 | 0.0358s | 164.8 | 0.0325s |
| 0.5 | 518.3 | 0.0414s | 164.8 | 0.0374s |

TPOT rises meaningfully and the derivative visibly steepens around `mfu_max≈0.6-0.7` — exactly
where the shared-expert FFN's hand-derived compute/memory ratio (~65% at mfu_max=1.0) crosses the
roofline threshold (65%/0.6=108%). This confirms the MFU model is doing real, physically-consistent
work, not a no-op. **But even at `mfu_max=0.5` (an aggressive derating), TPOT only reaches ~37-41%
of the SLO — nowhere near binding, and the batch anchor itself barely moves** (SHORT even ticks up
slightly, within parallelism-selection noise; MID stays exactly flat). A defensible MFU value
alone is **not sufficient** to make these anchors SLO-bound at the paper's smaller 460/151.5
values; that would require either a far more aggressive (indefensible, unsourced) derating, or a
communication/compute mechanism this model doesn't capture at all (e.g. the paper's SHORT/MID
points may use a different — possibly non-pure-DP — parallelism choice than our optimizer's
latency-ranking selects, for reasons outside a pure cost-model fix).

**Status: root cause corrected (was NOT a paper inconsistency); gap NOT closed.** No further
tuning attempted, per this investigation's standing rule against calibrating to match the paper's
number without independent justification.

### U2 — "4-GPU HBF+ TPS is 15% higher than 8-GPU HBM4" residual (llama4, LONG workload)

Paper: HBF+/4-GPU per-GPU TPS should be ~1.15× of HBM4/8-GPU's. The page-read-latency
double-buffering fix (item 5) moved the measured ratio from 0.53× to 0.72× — real progress,
but the paper's 1.15× is still not reached.

**Leading hypothesis tested and refuted this pass (prior session).** Both operating points are ~70%
attention time, and the decode attention KV-read is provably symmetric in code between HBM4 and
HBF+ (same windowed KV term, identical 12.8 TB/s bandwidth, page latency negligible post-fix) —
yet HBF+ sits at ~1362 attention-TPS/GPU vs. HBM4's ~2586, a gap the symmetric read model alone
can't produce. The parallel-config-asymmetry hypothesis (optimizer's `"max"` model over-crediting
PP overlap) was tested via forced-config comparison and **refuted**: both HBM4/8-GPU and HBF+/
4-GPU pick max-PP relative to their own GPU count (the same structural choice), paying zero
communication either way.

**This pass: re-verified unaffected by items 9/10/21.** `HBF+/4gpu` TPS/GPU = 1308.87,
`HBM4/8gpu` TPS/GPU = 1811.67 (both at LONG/0.1s SLO) — ratio = **0.722×**, numerically identical
to the pre-this-pass value (0.72×). Neither the MFU model (item 9), the ring all-reduce alignment
(item 10), nor the search-boundary fix (item 21) moved this point at all — confirming the gap is
not an artifact of any bug found and fixed in this investigation. The KV-write penalty (U6, below)
remains a confirmed, real, ~14.8%-of-decode-time contributor, but was already known to be
insufficient alone to close the gap.

**No safe fix identified — status unchanged.** Inflating communication or FFN-compute cost terms
purely to match the paper's two numbers would repeat the same "reverse-engineer a fudge factor
from the answer" problem already rejected for the capacity-reservation proposal and (this pass)
for U1. A real fix would need either a documented real-hardware basis for stronger communication/
compute costs at scale (kernel launch overhead, real collective-communication overhead beyond pure
bandwidth+latency, imperfect compute utilization at a magnitude well beyond what a defensible MFU
value provides — see U1), or direct access to the paper's own methodology section, neither
available in this repo.

### U4 — "HBF+ always outperforms HBM4 at 16 GPUs, across all SLOs" (llama4, LONG workload)

Paper: at 16 GPUs, LONG workload, HBF+'s per-GPU TPS should exceed 8-GPU HBM4 across every tested
SLO (0.05s, 0.1s, 0.2s, offline). The page-latency fix (item 5) moved the 0.1s-SLO ratio from
0.75× to 1.01× — but that check covered only one of the four SLOs.

**Full 4-SLO sweep, re-verified this pass with the search-boundary fix (item 21) applied:**

| SLO | 0.05s | 0.1s | 0.2s | Offline |
|---|---|---|---|---|
| norm. TPS (HBF+/HBM4), pre-item-21 | 0.96× | 1.01× | 1.04× | 0.85× |
| norm. TPS (HBF+/HBM4), post-item-21 | 0.959× | 1.012× | 1.038× | 0.846× |

**Statistically identical before and after fixing the batch-search bug (item 21/U8).** This is an
important negative result: it confirms the "misses at the tightest (0.05s) and loosest (offline)
ends" pattern is a **genuine throughput-ratio gap**, not an artifact of the batch-size search bug
that was corrupting the *offline* per-GPU batch number specifically (see U8). The batch number
was wrong (580.75, corrected to 684.12); the *derived throughput* (TPS/GPU = batch/(tpot×gpu)) was
essentially unaffected, because TPOT scaled up proportionally with the corrected batch in this
memory-bound regime. **Holds at 0.1s and 0.2s only; misses at 0.05s (−4%) and offline (−15%) — not
further root-caused this pass** (same open residual as U2 — see that item's "no safe fix
identified" discussion, which applies identically here).

### U8 — llama4_maverick/HBF+'s TPS edge over HBM4 shrinks at the offline SLO instead of growing

**RESOLVED (batch-size component) / STILL OPEN (throughput component) — split into two distinct
findings this pass.**

The original finding conflated two things that turned out to have different causes:

1. **The per-GPU BATCH dropping 8→16 GPUs (676.4 → 580.8) — was flagged as "itself unusual...not
   yet investigated."** Root-caused and **fixed** this pass: this was `CHANGES.md` item 21's
   search-algorithm non-monotonicity bug, not a real effect. Forced-distribution testing (holding
   PP=4 fixed, matching the optimizer's actual auto-picked PP at both GPU counts) showed TP=2
   unlocks 684.19 batch/GPU at 16 GPUs — *higher* than the auto-search's reported TP=1 ceiling of
   580.75, and matching the 8-GPU value (676.38) almost exactly. Tracing the live (unforced)
   optimizer directly at batches just above the old reported ceiling confirmed the mechanism:
   `b=9292` (divisible by 4, TP=1/PP=4/DP=4, feasible to ~10947) succeeds; the very next probed
   value `b=9293` (divisible by neither 4 nor 2) routes to TP=4/PP=4/DP=1 — a config with a much
   lower ceiling (~8192) — and fails there, even though `b=9296` (divisible by 4 again) succeeds
   with TP=1 same as 9292. The single-adjacent-probe boundary check took this spurious failure as
   proof of a true ceiling at 9292. After the fix (item 21), the corrected offline per-GPU batch is
   **684.12 at 16 GPUs vs. 676.38 at 8 GPUs — essentially flat, matching the paper's own stated
   mechanism (iii)** ("a larger batch size also linearly increases the peak size of intermediate
   data, so the limited SRAM-buffer capacity... bottlenecks batch-size scaling in HBF+" — §IV,
   quoted in full under U1 above): the SRAM tier is a fixed **per-GPU** capacity, so its ceiling is
   roughly GPU-count-independent once TP is chosen to shard activations appropriately, which is
   exactly what the corrected numbers now show. (Note: the *earlier* version of this document's
   "Finding 2" proposed mechanism (iii) as a *direct* explanation for the batch drop, without
   realizing the drop was actually a search bug. Mechanism (iii) is real and does explain why the
   *offline batch ceiling is roughly flat, not growing*, but the specific 676→580 *drop* was
   entirely the search bug, now fixed — the two should not be conflated.)

2. **The offline-SLO TPS ratio (0.85× / now 0.846× post-fix) still falling short of "always
   outperforms," and doing so most at the loosest SLO** — re-verified this pass (see U4's table)
   and **confirmed unchanged by the batch-search fix**: TPS/GPU barely moved (1535.06→1532.61,
   <0.2%) because TPOT scaled up proportionally with the corrected (larger, more accurate) batch.
   This is the **same open residual as U2/U4** (not a new, offline-specific mechanism) — the
   paper's own Fig. 6 "the more relaxed the SLO, the greater HBF's benefit" claim is not reproduced
   at the offline end specifically for this one combination (MoE model × all-flash config), for the
   same unexplained reason U2/U4's gap persists generally. Not further root-caused this pass beyond
   ruling out the search-bug explanation.

**Contrast rows, refreshed this pass with the fixed search** (16-GPU, normalized to 8-GPU HBM4,
0.05s/0.1s/0.2s/offline):

| Series | 0.05s | 0.1s | 0.2s | Offline |
|---|---|---|---|---|
| llama4/HBF+ (the anomaly) | 0.959× | 1.012× | 1.038× | 0.846× |
| llama4/HBF, for contrast | 0.819× | 0.870× | 0.895× | 0.906× |
| llama3/HBF+, for contrast (not re-verified this pass) | 1.15× | 1.34× | 1.43× | 1.47× |

llama4/HBF (1 reserved HBM stack, not all-flash) climbs monotonically toward offline as expected;
only llama4×HBF+ (all-flash) dips at the offline end — still specific to the **MoE model ×
all-flash (no reserved HBM) config** combination, as originally observed. **Not root-caused.**

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
general question of whether the ranking could still be biased for some other operating point —
and indeed U8's search-boundary bug (item 21) shows the search *can* miss a strictly-better config
at some operating points, though via a different mechanism (a divisibility-driven blind spot in
the *batch* search, not a latency-ranking bias at a fixed batch).

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
`chunk_size: 0`) and confirmed to have zero effect on any current paper comparison. **This pass:**
a compute-utilization (MFU) factor, even at a defensible `mfu_max=0.5`, does not by itself close
U1's gap (see U1). The optimizer's TP all-reduce ring-formula misalignment (item 10) does not
explain U2/U4's residual (both operating points are TP=1, zero communication, unaffected by the
fix). The batch-search non-monotonicity bug (item 21) does not explain U2/U4's TPS-ratio residual
(confirmed via full before/after re-sweep — see U4's table) — it only affected the *reported
batch-size number* at U8's specific operating point, not the throughput metrics U2/U4/U8's TPS
components are about.

## Harness note

`run_flash_only.py` (the fast 8-GPU-only smoke check, distinct from `run_experiments.py`, the
paper-comparison harness used for every number in this document) previously used non-canonical
workload token lengths (`short=(1024,1024)`, `long=(104600,1660)`, no MID) that don't match the
paper's SHORT/MID/LONG definitions used everywhere in this document — it could not have reproduced
any of the anchors above even before the other fixes in this pass. Corrected to use the same
canonical `WORKLOADS` as `run_experiments.py`, and its `"Total: "` stdout parser (which could
previously mis-parse if a future memory-report line's format changed) was made explicit about
excluding the `"...GB"`-suffixed memory-report lines. It remains an 8-GPU-only fast check, not a
substitute for `run_experiments.py` for any paper comparison.
