# Paper Inconsistencies — HBF-LLMSimulator vs. Son et al. (IEEE CAL 2026)

Tracker for every place our simulator's results have been checked against the paper's reported
numbers/claims and found **still open** or **explained (not a bug)** — i.e. every inconsistency
that isn't a plain resolved code fix. Resolved code fixes (including several that originated as
paper-comparison findings here) live in `CHANGES.md`; this doc only holds items that still need
attention or whose resolution is "the paper and the simulator genuinely differ for an understood,
non-bug reason," not "a bug was found and fixed." The primary source
(`Exploring_High-Bandwidth_Flash_for_Modern_LLM_Inference_Opportunities_and_Challenges.pdf`, in
this repo) was read in full when compiling this document — claims sourced from it are cited
directly rather than through intermediary docs.

## Still open

### U1 — llama4_maverick's HBM4/8-GPU SHORT & MID batch anchors ~9-12% high

**UPDATE (this session): partially closed by items 22-23 (pipeline-latency propagation +
throughput-ranked optimizer selection), gap not fully closed.** SHORT re-verified: 500.0 vs.
paper's 460 (×1.087, down from ×1.119/514.9). MID re-verified: 158.9 vs. paper's 151.5 (×1.049,
down from ×1.088/164.8). The optimizer's chosen 8-GPU config for this operating point now
genuinely pays communication instead of trivially defaulting to a zero-comm degenerate
one-device-per-stage config (see `CHANGES.md` item 26 for the full correction) — the mechanism-(i) analysis further down
this section (communication near-zero because of that exact degenerate choice) no longer applies
as literally stated; the *reason* that config previously looked cheap (an ~8× undercounted
pipeline latency, item 22) is now fixed. The residual gap is smaller but real: this operating
point remains capacity-bound in this simulator, not the paper's comm/compute-bound SHORT regime
described by mechanism (i)/(ii) below. The MFU-sensitivity analysis and mechanism (i)/(ii)
reasoning below predate items 22-23 and are kept for historical context — the root-cause framing
they establish (our cost model is too optimistic in the ways the paper's own mechanism names)
still stands; items 22-23 closed part of the (i) communication gap specifically, not the (ii)
compute-utilization gap.

SHORT: our (pre-items-22/23) 514.9 vs. paper's 460 (×1.12). MID: our 164.8 vs. paper's 151.5
(×1.09). Unaffected by the iRoPE fix (CHANGES.md item 16 — both contexts are shorter than the
8192-token local-attention window). Two hypotheses ruled out: collision-reduced active experts
(numerically negligible idle experts at these batch sizes) and an artificial capacity-reservation
"fix" (rejected as circular — no such convention exists in the paper).

**CORRECTION (an earlier pass): the previously-recorded root cause below was wrong. Retracted.**

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
  term, CHANGES.md item 20), but the optimizer's chosen 8-GPU config for this operating point is
  TP=1/PP=8/DP=1 (confirmed via direct stdout trace — the same degenerate one-device-per-stage
  choice investigated in `CHANGES.md` item 26), which zeroes every communication term by construction
  (`AllReduce` costs nothing at TP=1; the PP send/recv term is a fixed per-transition cost, not
  the batch-linear all-reduce mechanism (i) describes). Comm contributes ~0.
- **(ii) compute**: confirmed via CHANGES.md item 19's diagnostic — **there was no compute-utilization factor
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

**MFU sensitivity sweep (this pass, CHANGES.md item 19's model, `mfu_m_half=128` — a stated tensor-core-tile-
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

**Status: root cause corrected (was NOT a paper inconsistency); gap improved (items 22-23) but NOT
fully closed** (SHORT ×1.119→×1.087, MID ×1.088→×1.049). No further tuning attempted, per this
investigation's standing rule against calibrating to match the paper's number without independent
justification.

### U2 — "4-GPU HBF+ TPS is 15% higher than 8-GPU HBM4" residual (llama4, LONG workload)

**UPDATE (this session): substantially closed by items 22-24 (pipeline-latency propagation +
throughput-ranked optimizer selection + KV-write iRoPE windowing).** Re-measured from a clean
baseline (both HBF+ and the HBM4 comparison point re-run post-fix, since the old HBM4 number was
itself potentially undercounted by the same pipeline-latency bug if it picked a `PP>1` config):
`HBF+/4gpu` TPS/GPU = 1403.90, `HBM4/8gpu` TPS/GPU = 1402.17 (both LONG/0.1s SLO) — ratio =
**1.001×** (paper: 1.15×), up from the prior 0.722×. HBF+ now genuinely matches/edges HBM4 at this
comparison, crossing the paper's qualitative claim ("HBF+ outperforms HBM4"), though short of its
exact 1.15× magnitude. The analysis below (all pre-this-session) is kept for historical context;
its "refuted" leading hypothesis (symmetric KV-read, parallel-config-asymmetry) was itself tested
using simulator TPOT measurements now known to have been affected by item 22's bug for any `PP>1`
config in the comparison — those specific refutations should be treated as unverified rather than
re-asserted, though the qualitative conclusion (KV-read is symmetric by code inspection, not a
timing-model divergence) is a static-code-analysis claim unaffected by the runtime bug and still
holds.

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

**This pass: re-verified unaffected by items 19/20/21.** `HBF+/4gpu` TPS/GPU = 1308.87,
`HBM4/8gpu` TPS/GPU = 1811.67 (both at LONG/0.1s SLO) — ratio = **0.722×**, numerically identical
to the pre-this-pass value (0.72×). Neither the MFU model (CHANGES.md item 19), the ring all-reduce alignment
(CHANGES.md item 20), nor the search-boundary fix (item 21) moved this point at all — confirming the gap is
not an artifact of any bug found and fixed in this investigation. The KV-write penalty (U6, below)
remains a confirmed, real, ~14.8%-of-decode-time contributor, but was already known to be
insufficient alone to close the gap.

**(Historical, pre-items-22/24) No safe fix identified at the time.** Inflating communication or
FFN-compute cost terms purely to match the paper's two numbers would repeat the same
"reverse-engineer a fudge factor from the answer" problem already rejected for the
capacity-reservation proposal and for U1 — this concern remains valid and was not violated by the
actual fix found (items 22-24 are independently-motivated correctness/methodology fixes, not
numeric calibration; see their write-ups above). **Status: substantially closed (0.722×→1.001×,
paper 1.15×), residual not further root-caused this session** — the same "no fudge factor" rule
applies to the remaining ~13% gap; it stays open rather than being tuned away.

### U4 — "HBF+ always outperforms HBM4 at 16 GPUs, across all SLOs" (llama4, LONG workload)

**UPDATE (this session): re-swept post items 22-24.** Tight-SLO cells improved cleanly and
substantially: 0.05s 0.959×→**1.076×**, 0.1s 1.012×→**1.171×** — both now more comfortably above
1.0×, strengthening the "HBF+ outperforms" claim at these SLOs. The 0.2s/offline cells produced
**9.261×/9.376×** — technically "even more true," but this magnitude is a striking outlier vs. the
paper's own reported range (nothing above ~1.5×) and should **not** be read as an equivalent,
directly-comparable measurement to the 0.05s/0.1s cells above. Investigated and found to be driven
by a confirmed-real (not a search artifact) capacity cliff specific to the **HBM4 comparison
point**, not HBF+: HBM4/8-GPU's fastest config (`TP=4/PP=2/EP=4`) hits an exact capacity ceiling at
batch≈259 (`Out of Memory` confirmed at 260); past that, every valid `PP=4` variant (4 combinations
force-tested directly against the live simulator, not just the optimizer's top pick) also OOMs by
batch 262-264, leaving `TP=1/PP=8/EP=1` as the *only* capacity-feasible fallback — a discrete,
~7.8× latency jump (0.023s→0.179s) with no intermediate option, because this simulator only
supports powers-of-2 pipeline depths at 8 GPUs. This is a genuine consequence of item 22's fix (the
old buggy simulator could never show this — `PP=8` used to look artificially cheap, so the
capacity-driven fallback's latency penalty was invisible). Whether this reflects a real
architectural insight the old bug was hiding, or whether it reflects this simulator's discrete `PP`
granularity being coarser than whatever the paper's real system supports (smoother intermediate
parallelism options, different capacity assumptions), is **not resolved this session** — flagged
as a new, distinct open question rather than folded into the same throughput-ratio narrative as the
0.05s/0.1s results.

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
memory-bound regime. **(Historical, pre-items-22/24) Held at 0.1s and 0.2s only; missed at 0.05s
(−4%) and offline (−15%).** Superseded by the "UPDATE (this session)" block above — post items
11-13, 0.05s/0.1s both now exceed 1.0× (1.076×/1.171×); 0.2s/offline's new numbers (9.261×/9.376×)
are a distinct capacity-cliff finding, not a like-for-like re-measurement of this same residual —
see above.

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

2. **The offline-SLO TPS ratio — (historical, pre-items-22/24) 0.85×/0.846×, falling short of
   "always outperforms."** **UPDATE (this session):** re-measured post items 22-24 as part of
   U4's fresh 4-SLO sweep — offline is now **9.376×**, technically far exceeding "always
   outperforms," but for the capacity-cliff reason explained in U4 above (a discrete HBM4-side
   parallelism-fallback penalty), not because the original offline-specific throughput mechanism
   was found and fixed. **Still not root-caused in the sense the original finding meant** (why
   HBF+'s own relative benefit doesn't monotonically grow with looser SLOs the way the paper's
   Fig. 6 claims) — the new number is real but answers a different question than the one this
   finding originally asked; see U4's caveat before citing this ratio.

**Contrast rows** (16-GPU, normalized to 8-GPU HBM4, 0.05s/0.1s/0.2s/offline). Only the
llama4/HBF+ row was re-verified this session (post items 22-24); the other two rows are unchanged
from the pre-items-22/24 pass and may also be affected by the same fixes — not yet re-checked:

| Series | 0.05s | 0.1s | 0.2s | Offline |
|---|---|---|---|---|
| llama4/HBF+ (the anomaly), **re-verified this session** | 1.076× | 1.171× | 9.261×‡ | 9.376×‡ |
| llama4/HBF, for contrast (not re-verified this session) | 0.819× | 0.870× | 0.895× | 0.906× |
| llama3/HBF+, for contrast (not re-verified this session) | 1.15× | 1.34× | 1.43× | 1.47× |

‡ See U4's capacity-cliff caveat above — not directly comparable to the 0.05s/0.1s cells or to the
other two rows.

llama4/HBF (1 reserved HBM stack, not all-flash) climbs monotonically toward offline as expected;
only llama4×HBF+ (all-flash) dips at the offline end — still specific to the **MoE model ×
all-flash (no reserved HBM) config** combination, as originally observed. **Not root-caused.**

## Explained — not bugs

*(Note: an item formerly tracked here, "Degenerate one-device-per-stage optimizer choices," was
found this session to actually be a real, now-fixed bug — see `CHANGES.md` item 26 and `BUGS.md`'s
former item 8, both consolidated there. Removed from this section since it's no longer a genuine
"explained, not a bug" finding.)*

### U5 — "1-GPU HBF/HBF+ per-GPU batch > 8-GPU HBM4, in most cases" (llama3_405B miss)

Holds 5/6 for llama4_maverick but 0/6 (pre-fix) for llama3_405B — e.g. SHORT: 1-GPU HBF = 75 vs.
8-GPU HBM4 = 194.8 (−61%). **Root-caused as physically correct, not a bug**, and confirmed
airtight by the simulator's own instrumentation.

**The claim being checked.** The paper states "1-GPU HBF/HBF+ per-GPU batch exceeds 8-GPU HBM4's,
in most cases." The full sweep shows this holds 5/6 for llama4_maverick and 3/6 for
llama3_405B — HBF+ crosses the anchor in all 3 workloads, plain HBF in none:

| Workload | 8-GPU HBM4 | 1-GPU HBF | 1-GPU HBF+ |
|---|---|---|---|
| SHORT | 194.8 | 97.0 (−50%) | 235.0 (+21%) |
| MID | 61.9 | 39.0 (−37%) | 81.0 (+31%) |
| LONG | 3.8 | 2.0 (−47%) | 5.0 (+33%) |

**HBF+ does not exhibit the problem at all** (later fixes shrank the SLO-bound floor enough that
HBF+ beats the 8-GPU HBM4 anchor everywhere). **Plain HBF still misses at every workload**, and the
rest of this section explains why plain HBF specifically remains SLO-bound.

**Four pieces of evidence, together airtight.**

1. *The weight-reread floor is real, large, and correctly modeled.* `getLinearMemoryDuration`
   (`src/hardware/layer_impl.h:17-63`) charges weight-read time as `weight_size /
   flash_read_bandwidth` plus one exposed flash page-read latency, where `weight_size = k*n*
   precision` — **no batch-size factor**. This is physically required: in decode, weights are
   streamed fresh from memory every step regardless of how many sequences are batched together
   (the batch only changes the *activation* volume, not the weight volume read). llama3_405B
   (hidden=16384, 126 layers, 128 heads, 8 KV heads, intermediate=53248, BF16) has 401.6B
   parameters = **803 GB** of weight. At 1 GPU, TP=PP=1 is the *only* legal configuration (nothing
   else to shard across) — so this entire 803 GB must be re-read from flash every decode step: HBF
   (1 reserved HBM stack + 7 flash stacks, 11.2 TB/s flash read) ≈ **72 ms**; HBF+ (8 flash stacks,
   12.8 TB/s) ≈ **63 ms**. Both consume **62–73% of the 100 ms TPOT SLO** before a single
   sequence's marginal cost is added — not a hidden or synthetic penalty, just the direct
   consequence of a model this large needing sharding to fit any real compute budget.
2. *The same formula is independently validated by the anchor that DOES match.* The concern with
   (1) is "maybe the formula over-charges weight cost, and 8-GPU somehow escapes it by luck." It
   doesn't: the identical formula at 8-GPU HBM4 (weight sharded 8× via TP×PP, 12.8 TB/s, zero page
   latency) gives a first-principles capacity ceiling of **≈197 sequences/GPU**, against the
   simulator's reported **194.8/GPU** — a match to <2%. A bug that inflated the 1-GPU weight cost
   would also have to leave the independently-matching 8-GPU number untouched, which is not how a
   shared formula works.
3. *The 1-GPU HBF operating point is SLO-bound, not capacity-bound — confirmed three ways.*
   Direct capacity math: HBF's 1-GPU flash pool is 3,620 GB; subtracting the 803 GB weight leaves
   ~2,817 GB for KV, enough for **≈2,956 sequences** at SHORT context (KV ≈953 MB/seq at
   ~1660+187 avg tokens) — the reported batch (97) is about **3.3% of the capacity ceiling**.
   SLO-sensitivity sweep: at 4-GPU, llama3/HBF's batch scales **4.8 → 14.5 → 33.8 → 68.0** across
   SLO 0.05s/0.1s/0.2s/offline — a ~14× swing driven purely by relaxing the latency budget, vs.
   HBM4's exactly SLO-invariant 1.8 at every SLO (the signature of capacity-bound, not
   latency-bound). The harness's own `classify_failure()` distinguishes SRAM-bound from SLO-bound
   points precisely because the codebase already expects different points to bind on different
   constraints.
4. *Independent first-principles reproduction lands in the same regime.* A back-of-envelope model
   (weight-reread + compute + KV-read + KV-write, correctly sharded/batch-scaled) puts the 1-GPU
   HBF SLO-limited batch at roughly 100-150 sequences; the simulator's 97 sits at the low end of
   that range — pricing in attention/softmax compute, HBF's single reserved 1.6 TB/s activation
   stack (vs. HBF+'s free SRAM activations), and per-op page-read latencies, every one of which
   further tightens the SLO budget. This also explains why HBF+ (235) beats HBF (97) at 1 GPU:
   higher bandwidth (12.8 vs 11.2 TB/s) and zero activation-bandwidth cost both shave milliseconds
   off an SLO budget where weight-read is the binding constraint.

**Why llama4_maverick doesn't show this problem.** llama4_maverick is MoE with top_k=1 of 128
experts. At 1 GPU its per-step weight-read floor is only the active experts' weight (~1/128th the
routed-expert pool) plus attention — an order of magnitude smaller than a fully-dense
403B-parameter reread. Its weight floor never dominates the SLO budget the way llama3's does, so
its 1-GPU batch is governed by the same capacity/compute trade-off as its 8-GPU case, and the
"1-GPU beats 8-GPU" pattern holds in 5/6 of its tested combinations.

**Conclusion.** This is a **binding-regime mismatch, not a bug**: HBM4's reported batch is
capacity-bound (matches the paper); a fully-dense 400B+-parameter model's 1-GPU batch is throttled
by an un-shardable weight-reread floor under a strict 100ms online SLO — a real architectural
consequence of trying to serve a model this large from a single GPU, which the SLO (not capacity)
exposes. The model's true capacity advantage (2,956 vs. 197 sequences — nearly 15×) is real and
reappears once the SLO is relaxed. No code change proposed for this item.

**Confirmed by the simulator's own instrumentation.** Running `build/run` at this operating point
(`memory_type: HBF`, `num_device: 1`, `model_name: llama3_405B`, SHORT workload, `max_batch_size:
75`, `optimize_parallelism: true` — which correctly re-derives TP=PP=DP=EP=1, the only legal
1-GPU config) and extracting the live per-component decode-step breakdown from the CSV output:
measured TPOT was 0.09670s — 96.7% of the SLO budget, confirming the operating point is
SLO-saturating. Breakdown of that 96.7ms: FFN weight-read (gate/up/down) 63.2%, QKV/O-proj
weight-read 13.5%, KV-write (unhidden) 14.5%, attention KV-read + compute 7.4%, others 1.4%.
**Combined weight-read = 76.7% — the single dominant cost**, confirming this section's thesis
directly from the simulator's own instrumentation (the hand estimate above, ~72.6%, is the same
order/mechanism — the gap is expected since it didn't separately account for attention-projection
weight-read as distinct from FFN weight-read). KV-write (14.5%) is the clear second-largest term.
(This breakdown was captured at the older max_batch_size=75 anchor value, before later KV-read/MoE
fixes moved the converged max batch for this cell to 97 — not re-captured at 97, but the
mechanism it demonstrates is a property of the model/hardware pair near the SLO boundary, not tied
to the exact batch number, so it remains valid as illustrative evidence.)

### U6 — HBF+ KV-write "unhidden" overhead (was 14.7%@4-GPU → 19.8%@16-GPU pre-item-24; now ~10-11% post-item-24, 0% for HBM4)

**Confirmed directly consistent with the paper, not inferred — and quantitatively tightened this
session (item 24).** The paper's own footnote 2 (p. 3) states verbatim: *"Writing the KV cache
newly generated during the decode phase can be overlapped with computation in the attention
layer"* — exactly the mechanism our code implements (hide against attention-only compute, kept
unchanged this session — see item 24's note on why the hiding-budget alternative was rejected).
The paper's Fig. 5 discussion further states KV-cache writes account for "5–13.9% of the execution
time in Llama4," increasing "with more GPUs, shorter contexts, and the MoE model." **Pre-item-13,
our measured range (14.7-19.8% at the real near-SLO batch anchors) sat above the paper's stated
range** — item 24 found and fixed the reason (KV-write size wasn't windowed for Llama-4's iRoPE
local-attention layers, unlike the already-windowed KV-read/capacity paths). **Post-item-24
(combined with items 22-23), the measured range moved to ~9-11%** (measured jointly with the
pipeline-latency fix, so not a clean isolated before/after at the identical operating point — see
items 24/23's write-up in `CHANGES.md` for the two measurement passes) — now comfortably inside the
paper's stated 5-13.9% range. Not a bug, and now a better-calibrated (not just qualitatively
consistent) contributing factor to U2's residual, exactly as the paper's own Takeaway 3 frames it.

### U7 — HBF+/CONV+ per-GPU batch grows with GPU count instead of staying flat — not a bug, a documented model divergence from the paper's tool

**The claim being checked.** Fig. 3 of the paper marks HBF+/CONV+ bars as **SRAM-bound**: per-GPU
batch size does not increase once the 320-MB logic-die SRAM ceiling binds, so a larger GPU count
"does not appear" as a new bar segment (paper's own Fig. 3 caption convention). The paper quotes
~327 seq/GPU as an example (Llama 4 Maverick). Our full sweep instead shows
`llama4_maverick / HBF+ / SHORT` growing monotonically with GPU count and tagged
`bound_reason=slo`, not `sram`:

| GPUs | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| per-GPU batch (current sim) | 730 | 1557.5 | 1964.5 | 2149.0 | 1870.8 |
| bound_reason | slo | slo | slo | sram | sram |

Same pattern for `llama3_405B / HBF+` and, less dramatically, `llama4_maverick / CONV+`.

**Root cause: two deliberate changes to the intermediate-data footprint formula.**
`peakIntermediateBytes()` (`src/model/footprint.h`, added during the fairness-audit sprint,
`CHANGES.md` item 14) replaced an older formula that lived directly in
`src/hardware/cluster.cpp::checkMemorySize` (present as far back as the repo's initial commit,
predating any HBF-specific work). The two formulas diverge in exactly two ways: (1)
`max(attention-phase, FFN-phase)` vs. `sum(attention-phase, FFN-phase)` — the current model
assumes tensors are produced and consumed sequentially within a layer (attention scratch freed
before FFN scratch is allocated), so the peak resident set is whichever phase is larger, not their
total; the original formula summed both, assuming no intra-layer buffer reuse. (2) Excluding vs.
including the attention-score matrix (`Q·Kᵀ`) — in decode this term is `∝ batch × num_heads ×
total_context_length`, the only term in the whole footprint that scales with sequence length; the
current model excludes it entirely (flash/chunked attention never materializes the full score
matrix — it streams a chunk at a time through a separate, already-budgeted 3.13-MB/stack staging
buffer), while the original formula counted the full score matrix against the 320-MB pool.
Analytically, for `llama4_maverick` (hidden=5120, num_heads=40, BF16), the score term is roughly
93% of the original formula's per-sequence footprint at MID-length context — removing it (plus the
max/sum change) collapses the per-sequence footprint from hundreds of KB down to ~74 KB, pushing
the SRAM ceiling from ~300–800 seq/GPU up to ~4,200 seq/GPU. At that ceiling the 0.1s TPOT SLO
binds first instead, which is why the current sim reports `slo`, not `sram`, and why per-GPU batch
grows with GPU count (more GPUs → more sharding → lower per-token latency → more batch fits under
the fixed SLO).

**A/B verification.** Both formulas were built as separate binaries in isolated git worktrees and
run through the same batch-search harness. Old-style results, `llama4_maverick / HBF+`, 0.1s TPOT
SLO: SHORT 725/725/725/725/725 (sram-bound, 5/5 GPU counts), MID 288×5 (sram-bound, 5/5), LONG
19.0/19.5/19.8/19.8/19.8 (sram-bound, 5/5). Restoring the original formula's two changes is
sufficient to flip every one of these 15 cells from `slo`/growing to `sram`/flat — exactly the
paper's Fig. 3 regime. `llama3_405B / HBF+ / SHORT` showed the same effect even more cleanly (226
seq/GPU, flat across all 5 GPU counts, `sram`-bound throughout). The `HBF` control (scarce tier =
the 1 reserved HBM stack, not the 320 MB SRAM pool) was untouched, as expected, since that gate
never calls `peakIntermediateBytes`.

One second-order effect: `llama3_405B / HBF+` (and, more mildly, `llama4_maverick / HBF+ / LONG`)
shows a small batch increase specifically at the 1→2 GPU transition, not perfect flatness from GPU
1. Root cause, confirmed via the emitted `TP*_DP*` distribution: `ParallelismOptimizer`
opportunistically switches from `TP=1` to `TP=2` once a second GPU makes it available, because
several terms in `peakIntermediateBytes` are sharded by `tp` (query/key/value projections, the
score term) while others are not (the residual stream, the out-projection, and — for MoE models —
the routed-expert FFN term). This is **not** a bug: the paper's own §III states "each evaluated
system selects the parallelism configuration that maximizes the achievable system throughput
subject to all constraints ... by combining data, tensor, pipeline, and expert parallelism" —
exactly what `ParallelismOptimizer` does.

**Is the current (max, no-score) model more accurate?** Yes, on physical grounds independent of
which one matches the paper: the score-matrix exclusion is correct for flash/chunked attention
(FlashAttention provably keeps peak score memory at `O(chunk)`, not `O(context)`; charging the full
matrix against the 320 MB pool double-counts memory that is never resident in that form and
already has its own 3.13-MB/stack budget), and `max`-not-`sum` matches production runtime behavior
(activation buffers are recycled as tensors die, not held for the whole layer — the paper's own
prose, "much of the intermediate data has a short lifetime, which enables quick release of
SRAM-buffer space," describes exactly this, even though the paper's own tool evidently did not
implement it that way per this A/B test). The important caveat: Son et al.'s actual extended
simulator was never open-sourced (the paper says only "we will open-source our extended LLM
simulator"). The formula reverted for this test was reverse-engineered from Duplex-era (MICRO'24
LLMSimulator) bookkeeping code that happened to still be present in `cluster.cpp`, under the
assumption that the paper's authors extended that pre-existing formula rather than replacing it
outright — plausible (same codebase lineage, same GQA/MLA math) but unconfirmed, which is also why
the old-style magnitudes (725/288/19.8 seq/GPU) land in the same order of magnitude as, but not
exactly on, the paper's quoted 327.

**Conclusion.** This is a **real, understood model divergence**, not a bug in either direction.
The current simulator's SLO-bound, growing-per-GPU-batch result for HBF+/CONV+ is the direct and
correctly modeled consequence of two deliberate, physically-motivated departures from the formula
the paper's own (unpublished) tool most likely used. The A/B experiment confirms the mechanism
precisely — reverting those two changes reproduces the paper's SRAM-bound, flat-per-GPU-batch
regime across every tested model and workload — without establishing that the paper's formula is
the more physically correct one. Given flash/chunked attention and realistic buffer reuse, the
current model's SRAM ceiling is arguably looser than the paper reports for legitimate reasons; no
code change is proposed as a result of this investigation. Related to (and separately confirmed
alongside) U5's SLO-vs-capacity-bound distinction above.

## Ruled out as causes (for completeness)

Flash bandwidth constants and llama4's weight footprint (746.4 GiB) match the paper's Table I and
stated figure exactly. Routing skew (`skewness: 0.8`) doesn't reach U1's capacity-bound operating
point. Parallelism-ranking communication cost is ≤0.6% of decode time in every checked config
(confirmed further by the forced-config A/B check in `CHANGES.md` item 26: zero communication either way). The MoE
activation-divisor bug (`CHANGES.md` item 18) is real but confirmed NOT the cause of U1 (SHORT/
MID anchors byte-identical before/after that fix). `BUGS_HIDDEN_BY_FLAGS.md` entries are all
masked by pinned config flags (`decode_mode: on`, `disagg_system: off`, `parallel_execution: off`,
`chunk_size: 0`) and confirmed to have zero effect on any current paper comparison. **This pass:**
a compute-utilization (MFU) factor, even at a defensible `mfu_max=0.5`, does not by itself close
U1's gap (see U1). The optimizer's TP all-reduce ring-formula misalignment (`CHANGES.md` item 20) does not
explain U2/U4's residual (both operating points are TP=1, zero communication, unaffected by the
fix). The batch-search non-monotonicity bug (`CHANGES.md` item 21) does not explain U2/U4's
TPS-ratio residual (confirmed via full before/after re-sweep — see U4's table) — it only affected
the *reported batch-size number* at U8's specific operating point, not the throughput metrics
U2/U4/U8's TPS components are about.
