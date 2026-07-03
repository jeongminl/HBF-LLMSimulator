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

> **Session note (2026-07-03): reorganized.** U1, U2, U4, U8 and U6 are RESOLVED; their full
> resolution records (original inconsistency, root cause, fix references, final verified numbers)
> now live in `CHANGES.md`'s "Paper-comparison items resolved (2026-07-03)" section, with one-line
> pointers under "Resolved — see CHANGES.md" below. The dominant root causes were search/objective
> defects (optimizer ranking metric under-ranking DP by dp×; max-batch instead of §III
> max-throughput reporting), an NVLink generation mismatch (gen 5 vs the paper's stated 1,800 GB/s =
> gen 6 on Rubin), a KV-write page-program latency 48×-overcount, analytic/recorded weight-footprint
> drift (MoE router), and an accidental pipeline-stage-overlap timing bug for tp≥2/pp>1 —
> `CHANGES.md` items 30-35. This doc now holds only still-open items (U7, U9), the "Open residuals"
> pick-up list (exact operating points + reproduction recipes for every sub-5-13% delta that
> survived the fixes, deliberately NOT tuned), "explained — not a bug" items (U5), and the "Ruled
> out as causes" list. A parallel investigator should start from U9 + the pick-up list (each entry
> names its winner config, measured vs paper numbers, and suspects) and U7's reproduction block.
> The full-sweep regeneration of `experiment_results.md` is pending (on hold per user).

> **Session note (2026-07-03, second pass): U9 and residuals 2-4 RESOLVED; U5 reclassified as
> matching.** A four-agent investigation root-caused and fixed (CHANGES.md items 37-40): U9 = the
> AllReduce ring-latency overcharge (item 37, two agents converged independently); residual-4
> (CONV+ 4-GPU −33%) = per-op flash page-read latency on the weight stream (item 38); residual-3
> (llama3 batch anchors) = replicated (un-TP-sharded) embedding/LM-head weights (item 39); U2's
> ratio residual closed as a by-product (1.145 vs paper 1.15). U5's "miss" dissolved — the
> current binary matches the paper's OWN 1-GPU bars (the old table below was stale). U7's
> explanation SURVIVED an adversarial re-audit, strengthened by the "327-is-a-mislabel" finding.
> Two physically-motivated changes were DELIBERATELY DEFERRED by user decision (score-matrix
> traffic + MFU pair; steady-state context seeding) — see "Deferred by decision" below.

## Resolved — see CHANGES.md

Full resolution records (original inconsistency, root cause, fix references, final verified
numbers) are in `CHANGES.md`'s "Paper-comparison items resolved (2026-07-03)" section and the
"Paper-comparison fixes, second pass" section (items 37-40).

- **U9** — llama3/HBM4/8-GPU LONG TPS −13%: resolved, AllReduce ring-latency overcharge
  (CHANGES.md item 37). Post-fix: batch 3.75/GPU (−1.3%), TPS/GPU 138.2 (−5.8%, was −13.0%);
  comm share 5.4% vs paper Fig-5 5.1%. NOT the KV-read/context suspects originally listed —
  those were measured matching (attention share 64.9% vs paper 67.5%).
- **Residual-4** — llama4/CONV+/SHORT 4-GPU −33%: resolved, weight-stream page-latency
  amortization (item 38). 4-GPU now 189.25/GPU vs paper 173.9 (+8.8%), 8-GPU 383 vs 390.1
  (−1.8%). The 1/2-GPU bars (59/88 vs printed 54.3/54.3) sit below the figure's line-thickness
  resolution — reading tolerance, not a divergence (user-confirmed).
- **Residual-3** — llama3/HBM4 batch anchors −2.4%/−4.7%: resolved, vocab-parallel
  embedding/LM-head sharding (item 39). SHORT +1.2%, MID +1.4%, LONG −1.3%.
- **Residual-2** — U2 ratio 1.116 vs paper 1.15: resolved as a by-product of items 37-39:
  ratio now **1.145** (numerator batch +4.0%, denominator batch +0.6%).

- **U1** — llama4_maverick HBM4/8-GPU SHORT/MID batch anchors ~9-12% high: resolved, see CHANGES.md
  items 30-34 + resolution record (winner TP=2/PP=1/DP=4; SHORT 483.5/GPU ×1.051, MID 153.5/GPU ×1.013).
- **U2** — llama4 LONG "4-GPU HBF+ TPS 15% higher than 8-GPU HBM4" residual: resolved, see CHANGES.md
  items 30-31 + 33 + resolution record (ratio 1.116× vs paper 1.15×).
- **U4** — llama4 LONG "HBF+ always outperforms HBM4 at 16 GPUs across all SLOs": resolved, see
  CHANGES.md items 30-31 + resolution record (~9× outlier gone; monotone 1.193→1.333, within 1.6% of Fig-6).
- **U8** — llama4 HBF+ TPS edge over HBM4 shrinking at offline SLO: resolved, see CHANGES.md items
  21 + 30-31 + resolution record (now monotone 1.193→1.333; per-GPU batch flat 676→684, 8→16 GPU).
- **U6** — HBF+ KV-write "unhidden" overhead above the paper's 5-13.9% range: resolved, see CHANGES.md
  items 24 + 33 + resolution record (validated vs all 12 Fig-5 readings, mean |error| ≈ 0.7pp).

## Still open

### Residual-1 — llama4/HBM4/8-GPU SHORT: batch high / TPS low (only surviving open residual)

Winner `TP=2/PP=1/EP=1/DP=4`. Pre-second-pass: batch +5.1% (483.5/GPU vs 460), TPS −4.9%. The
comm model prices TP=2 marginally better than the DP-pure layout whose batch (460.0/GPU) equals
the paper's printed anchor exactly; the fix-37 latency change does NOT alter N=2 (ring == log at
2 ranks), and fix-39 frees ~1 GB/GPU of vocab weight on this capacity-bound cell, so expect the
batch side to drift slightly HIGHER — an accepted, documented entanglement (the compensating
lever is the deferred MFU item below, not comm tuning). Suspects if reopened: MoE cross-replica
all-to-all volume at dp>1 vs the single attention all-reduce at TP=2 (static audit found the
ring formula itself correct and the optimizer-vs-live MoE comm drift to be ranking-only).

### [RESOLVED 2026-07-03 second pass] U9 — llama3_405B/HBM4/8-GPU LONG TPS −13%

Resolved as the AllReduce ring-latency overcharge — see "Resolved" list above and CHANGES.md
item 37. The original record (kept for the reproduction recipe; its candidate suspects (a)-(c)
were all MEASURED NOT-CAUSAL — attention/KV-read shares match the paper):

Fresh measurement on the fixed binary (`CHANGES.md` items 30-35): batch **3.62/GPU** (29 total,
winner `TP=8/PP=1/EP=1/DP=1`, capacity-bound ✓, `bound=flash`) vs the paper's 3.8 (−4.7%), but
TPS/GPU **127.5** vs the paper's 146.6 (−13.0%) — measured tpot **0.0284s** vs the paper-implied
0.0259s (= 3.8×8/146.6/8). SHORT and MID at the SAME winner config match the paper to ±3% on both
metrics (SHORT 190.9/GPU & 3393 TPS at tpot 0.0563; MID 60.6/GPU & 1780 TPS at tpot 0.0341), so
the divergence is specific to the very-long-context regime (input 103500 / output 1100).

**Candidate suspects** (untested): (a) the attention KV-read path at 103.5K context — llama3 has
no iRoPE window and no attn_chunk, so every layer reads the FULL context's KV each step; check
whether the per-step KV-read volume uses steady-state average occupancy (input + output/2 ≈
104,050 tokens) vs. full input+output (104,600) vs. current-length growth over the decode, and
whether the paper's tool may average differently; (b) attention score COMPUTE at this context
(the GPU roofline max(compute, memory) per attention op — 128 heads × 104K ctx); (c) HBM4's
attention path goes through the non-flash branch (`getAttentionMemoryDuration` returns 0 for
HBM4; timing comes from the plain roofline/ramulator path) — diff that path's context scaling
against the flash branch's.

**How to reproduce / investigate** (~1 minute per run): call
`run_experiments.find_max_batch_size("llama3_405B", "HBM4", 8, 103500, 1100, tpot_slo=0.1,
temp_cfg_name=..., worker_tag=...)`, or force the exact operating point via
`run_simulation(..., batch_size=29, distribution={"tp": 8, "pp": 1, "ep": 1})`. The winner's CSV
(path printed after the numeric `Total:` line) has per-component times in `type=="t2t"` rows
(columns: qkvgen/atten_sum/atten_gen/o_proj/ffn/expert_ffn/communication/kv_write/...; average
across rows; `latency` column = tpot in ns). Diff this cell's component shares against the SHORT
(batch 1527) and MID (batch 485) cells at the same TP=8 config: the −13% must live in whichever
component grows superlinearly with context relative to the paper's implied scaling.

### Open residuals — pick-up list status after the second pass

Items 2/3/4 of the former pick-up list are RESOLVED (see the "Resolved" list above: U2 ratio
1.145; llama3 anchors ±1.4%; CONV+ 4-GPU +8.8%). Item 1 (llama4/HBM4 SHORT) remains open and is
now tracked as Residual-1 under "Still open" above.

## Explained — not bugs

*(Note: an item formerly tracked here, "Degenerate one-device-per-stage optimizer choices," was
found this session to actually be a real, now-fixed bug — see `CHANGES.md` item 26 and `BUGS.md`'s
former item 8, both consolidated there. Removed from this section since it's no longer a genuine
"explained, not a bug" finding.)*

### U5 — "1-GPU HBF/HBF+ per-GPU batch > 8-GPU HBM4, in most cases" (llama3_405B miss)

**RECLASSIFIED (2026-07-03 second pass): the miss no longer exists — current binary MATCHES the
paper's own 1-GPU bars.** Two independent measurements on the current binary (before the second-
pass fixes) gave llama3 1-GPU HBF = 181/74/5 (SHORT/MID/LONG) vs the paper's own Fig-3 1-GPU
bars 176.2/70.0/5.1 (+2.7/+5.7/−2%), and HBF+ = 362/125/7 vs 329.4/116.1/7.1 (+10/+7.7/−1.4%);
after fix-38 (weight-stream page-fill amortization) the HBF SHORT cell reads 185 (+5.0%). The
table below (97/39/2, 235/81/5) is STALE — it predates the CHANGES.md item-17/18/30-35 fixes and
was never re-measured when this section was written. Two errors in the original record: (a)
stale numbers, and (b) it benchmarks 1-GPU HBF against the 8-GPU HBM4 anchor (194.8) — the
paper's claim compares against the paper's OWN 1-GPU bars. The weight-reread-floor MECHANISM
below was independently re-verified (weight charged exactly once per op, `k·n·precision`, no
batch factor; CSV: FFN+QKV+O-proj weight-read ≈ 77% of the 100 ms budget at the SLO-saturated
point) and stands as the correct physical explanation of WHY the 1-GPU cell is SLO-bound — it
just no longer needs to explain away any discrepancy. Historical record kept below.

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

### U7 — HBF+/CONV+ per-GPU batch grows with GPU count instead of staying flat — not a bug, a documented model divergence from the paper's tool

**ADVERSARIAL RE-AUDIT (2026-07-03 second pass): explanation SURVIVES, strengthened.** Two new
findings: (1) the paper-text "327 seq/GPU (Llama 4 Maverick)" does not match the llama4 SHORT
bar under any accounting (score-free ≈ 4,200, score-inclusive-max ≈ 800-900 — the SHORT bar
854.7 IS the score-inclusive-max ceiling; exact footprint replica: 361.7 KB/seq → 906). Two
candidate readings of "327", neither changing the disposition: (a) a model mislabel — 327 ≈
llama3's own SHORT/HBF+ bar (329.4); or (b) **(user-preferred, 2026-07-03)** the llama4 MID
1-GPU HBF+ bar — read as 352.7 in paper_figure_readings.md, but the figure's scale makes 327 an
equally valid visual read, and notably the score-inclusive accounting's implied MID ceiling
(~300/GPU: 40 heads × ~6.4K ctx × 2B × 2 buffers ≈ 1 MB/seq against 320 MB) lands within ~8% of
327 — which would make the paper's text internally consistent for llama4 at that one cell.
(2) The dual-constraint hunt (a single accounting fitting llama4's ~374 KB/seq AND llama3's
~995 KB/seq, SHORT and MID simultaneously) has no solution — adopting score-inclusive as the
gate breaks llama4 MID at 8/16 GPUs (314 vs 745), llama3 MID (98 vs 190) and llama3 LONG (6 vs
19) by 2-3×, and empirical SRAM_DIAG ceilings confirm it would UNDERSHOOT the paper's own
growing MID/LONG bars. (Reading (b) above sharpens rather than resolves this: if 327 = the MID
1-GPU score-inclusive ceiling, the paper's own MID series would be SRAM-capped at ~327/GPU at
1 GPU yet reads 745.1 at 8/16 GPUs — per-GPU SRAM ceilings don't loosen with GPU count to first
order, so the six bars remain mutually inconsistent with any single O(ctx) formula.) The
current score-exclusive max model matches 10/12 bars. No fix. Original record below.

**UPDATE (2026-07-03 session — napkin-math quantification of the divergence, and why the paper's
SRAM-bound bar is flat across GPU counts at all).** Two additions to the analysis below:

1. **Implied per-sequence footprint.** The paper's ✕-marked (SRAM-bound) Fig-3 bar is
   llama4/SHORT/HBF+ flat at 854.7 seq/GPU, so its tool's effective intermediate footprint is
   320 MB / 854.7 ≈ **374 KB/seq**. Hand-summing llama4 decode tensors (BF16, unsharded):
   residual+projections ≈ 44 KB, dense-FFN phase ≈ 76 KB, MoE routed+shared ≈ 120 KB, and the
   context-length-scaled **attention-score/softmax buffers ≈ 163 KB each (heads × ctx × 2B at
   SHORT ctx≈2000)**. Score-free models land at ~76 KB/seq (ceiling ≈ 4,200 — SLO/capacity bind
   first, never SRAM); score-inclusive sums land at ~400-500 KB/seq (ceiling ≈ 650-900). Nothing
   in between falls out naturally — **374 KB/seq is only reachable by charging O(ctx) attention
   scratch against the 320 MB pool**, confirming (independently of the A/B below) that the paper's
   tool counts it.
2. **The SRAM ceiling in per-GPU-batch units is parallelism-invariant to first order** — which is
   why a flat bar across GPU counts is even coherent. Score and head-dimension terms shard by tp,
   but per-GPU batch (total/GPUs) then also covers proportionally fewer sequences at full per-seq
   cost, so the ratio cancels; DP replicas hold their own B/dp at full per-seq cost — same
   cancellation. The binding question is therefore NOT the tp/pp/dp/ep config, purely what counts
   toward the footprint. (Second-order: the residual stream and shared-expert terms don't
   tp-shard — the small 1→2 GPU bump documented below.)

**Full A/B experiment (2026-07-03; KEPT OPEN as "explainable" per user decision — the explanation
may yet prove to hide a bug).** Side A = current model (score excluded); side B = a git worktree
(`ab-score-accounting`, left in place at `/home/arcuser/jeongmin/HBF-ab-score`) identical except
`peakIntermediateBytes` also charges the chunked-attention score/softmax working set (2 buffers x
heads/tp x min(ctx, attn_chunk_size) x precision). Same fixed binary base (`CHANGES.md` items
30-35), same throughput-max search, 0.1s SLO.

SHORT sweep (batch/GPU, bound):

| llama4/HBF+ | 1 | 2 | 4 | 8 | 16 | paper |
|---|---|---|---|---|---|---|
| A (current) | 823 slo | 1593 slo | 2004 slo | 2170 slo | (converging†) | 854.7 FLAT, ✕ sram |
| B (variant) | 823 slo | 906 sram | 906 sram | 906 sram | 777 sram | |

† the A-side 16-GPU search was still bisecting `TP=8/DP=2` at ~1,315-1,560/GPU when this was
written — already well above both B (777) and the paper (854.7), so the shape conclusion (A grows,
paper flat) is settled regardless of the exact figure; it will be appended when converged.

llama4/CONV+: A = B = 52/48/116/352/457 per GPU, all slo (paper 54.3/54.3/173.9/390.1/477.5 —
growth shape ✓; the score charge never binds on CONV+, whose SLO binds far below the SRAM
ceiling). llama3/HBF+ (B): 285/270/244/244/244, all sram (paper 329.4 flat but NOT ✕-marked).

MID/LONG check (B, 8/16 GPU — side A matches the paper on all of these):
- llama4 MID: **308/297 sram** vs paper 745.1 (not sram-marked; A: 804 slo) — ~2.5x regression
- llama3 MID: 86.6/86.6 sram vs paper 189.9 (A: 238.8 slo) — ~2.2x regression
- llama3 LONG: 6.1/6.1 sram vs paper 19.0 (A: 19.2 slo ✓) — ~3x regression
- llama4 LONG: 166.5/172.2 slo vs paper 164.9/169.5 — matches, only because the 8192 iRoPE chunk
  caps the score charge there.

**Structural conclusion: no context-scaled score charge fits the paper's own bars simultaneously.**
The ✕ bar implies ~374 KB/seq at ctx≈2000 while the (non-✕) MID bar requires ≤430 KB/seq at
ctx≈6400 — nearly context-independent, which no physical O(ctx) score accounting produces
(checked: 1 and 2 buffers, chunk-capped, staging-slice-sized, and the literal old Duplex
sum-formula, which also fails MID at 288 vs 745 per the A/B below). Matching the one ✕ bar
necessarily breaks three MID/LONG bars.

**Decision (user, 2026-07-03): keep the current score-exclusive model** (matches 10/12 HBF+ batch
bars incl. every MID/LONG cell; the two SHORT HBF+ bars remain a documented divergence). A
**diagnostic-only** score-inclusive footprint is now logged per run so both accountings stay
visible without changing any reported metric: `footprint.h::scoreInclusiveIntermediateBytes`,
`SRAM_DIAG_SCORE_INCLUSIVE_ACT_BYTES` / `SRAM_DIAG_CEILING_BATCH_PER_GPU` markers (cluster.cpp),
surfaced as `sram_diag_ceiling_per_gpu` in `[config-search]` sweep lines. Future passes on this
item should hunt for accountings whose per-seq footprint is ~context-independent (~374-430 KB/seq
for llama4) rather than iterating further on score terms.

**Reproduction for follow-up investigators.** Side B lives in the worktree
`/home/arcuser/jeongmin/HBF-ab-score` (branch `ab-score-accounting`, built binary at `build/run`,
sweep drivers `ab_sweep.py` / `ab_sweep_midlong.py`, raw logs `ab_sweep*.log`, per-cell CSVs under
its `data/ab*`); the variant is the marked block in its `src/model/footprint.h` (grep
`AB-VARIANT`). Cells are one call each: `run_experiments.find_max_batch_size(model, "HBF+"|
"CONV+", gpus, input_len, output_len, tpot_slo=0.1, temp_cfg_name=<unique>, worker_tag=<unique>)`
from the respective tree (SHORT=1660/373, MID=5900/499, LONG=103500/1100); unique
temp_cfg_name/worker_tag per concurrent call is REQUIRED (shared build dir + CSV names). The
per-config analytic ceilings come from `system.analytic_configs_only: true`
(`ANALYTIC_CONFIG:` stdout lines). Paper ground truth: `paper_figure_readings.md` section 1.

---

**UPDATE (paper-inconsistencies pass, current binary — re-measured after `CHANGES.md` items 27
(scheduler `seq_len` clamp) and 28 (`num_heads`→`num_kv_heads` footprint fix); the "explained, not a
bug" conclusion holds and is now a *closer* match to the paper's flat/SRAM-bound shape.)** Fresh
race-free sequential sweep, `llama4_maverick / HBF+ / SHORT / 0.1s`:

| GPUs | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| per-GPU batch (current binary) | 730 | 1370 | 1527 | 1330 | 1288 |
| bound_reason | slo | slo | slo | slo | sram |

(Re-confirmed clean this pass, race-free single-process: GPU 1/2/4/8 = 730 (`TP=1`) / 1370 (`TP=2`) /
1527.2 (`TP=4`) / 1329.5 (`TP=8/PP=1`), all `slo`; GPU 16 ≈ 1289 (`TP=8/PP=2`, `sram`) — converged to
batch ~20636 before the exact-integer boundary refinement was stopped as unnecessary for this table.
The curve **peaks at 4 GPU and declines** to an SRAM-bound ceiling by 16 GPU.) **The 4→8 turnover
happens within *pure tensor parallelism* (`TP=4`→`TP=8`, no PP yet), so it is not a pipeline effect —
it is the TP all-reduce cost growing with the TP degree: doubling GPUs 4→8 raises total batch only
×1.74 (6109→10636), not ×2, because each all-reduce over `N` ranks pays `2(N−1)` hops whose fixed
per-hop latency grows linearly with `N` and eats the fixed 0.1s SLO budget. Past `TP=num_kv_heads=8`
the extra GPUs must go to PP (`CHANGES.md` item 3 caps TP at `num_kv_heads`), and by 16 GPU the SRAM
ceiling binds.** Two things changed vs.
the older table below (730 / 1557.5 / 1964.5 / 2149.0 / 1870.8, peaking at 8 GPU): (i) the scheduler
`seq_len` crash (`CHANGES.md` item 27) was previously capping/​distorting the high-GPU-count batch
search — removing that artifact lowers the mid-range and makes 16 GPU reach `sram` cleanly; (ii) the
`num_kv_heads` footprint fix (`CHANGES.md` item 28) tightens the SRAM ceiling slightly. **Net effect:
the current sim's HBF+ curve is now qualitatively closer to the paper's flat/SRAM-bound Fig. 3 shape
(a peak-then-decline into an SRAM ceiling, rather than unbounded monotonic growth) — but this is the
automatic consequence of fixing a crash artifact and a head-count over-count, NOT a tuned match.** The
underlying model divergence analysed below (max-vs-sum footprint + score-matrix exclusion) is
unchanged and still explains why the *absolute* ceiling remains looser than the paper's ~327 seq/GPU.
The old table and A/B analysis are retained below for the mechanism.

---

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

## Deferred by decision (2026-07-03 second pass — real, quantified, deliberately NOT applied)

Two physically-motivated model changes were identified, measured, and deferred by explicit user
decision, to avoid any calibrated constant ("no fudging" rule). Recorded here so a future pass
can pick them up as a PAIR:

1. **FlashAttention score/query HBM-traffic removal + GEMM-efficiency (MFU<1) — both or
   neither.** The non-flash attention memory charge includes `m·n·G` (score matrix) + `m·k·G`
   (query) bytes per (seq, kv-head) (`attention_gen_impl.cpp` Scoring/Context; MLA gen has the
   same term; the flash branch charges the same bytes on the HBM-stack act path). With
   `use_flash_attention: on` the score matrix never materializes in HBM, so this is physically
   wrong — but it is a CONSTANT ~11% fraction of attention memory in every cell (llama3 G=16:
   1.9/1.86/2.05 ms at LONG/MID/SHORT; llama4 G=5: ~0.6 ms), and it currently cancels an
   opposite-sign divergence: §III implies roofline max (「FFN … compute-bound GEMM」— so max is
   the PAPER-ALIGNED convention), yet Fig-5's FFN shares imply the paper's GEMM time is ~2× our
   100%-peak time, i.e. an unmodeled MFU ≈ 0.45-0.5 (infra exists: `mfu_max`/`mfu_m_half`,
   currently 1.0/0). Measured prediction table (post-fix-37 binary, tpot err vs paper): applying
   score-removal alone drives MID/SHORT/llama4-LONG to −8…−11% (breaks 3 cells); applying both
   (score-removal + calibrated MFU) is expected to bring every cell within a few %, but the MFU
   value would be fitted to Fig-5 — hence deferred.
   **Honest post-fix state**: with the ring-latency overcharge (item 37) gone, llama3 SHORT/MID
   tpot reads ~4-6% FAST vs paper (TPS high; e.g. MID TPS/GPU 1904.7). This is the exposed
   compute-under-charge, no longer masked by accidental cancellation.
2. **Steady-state decode context seeding.** Decode tpot is measured over 10 iterations seeded at
   `current_len = input_len` (start-of-generation), while steady-state serving averages
   `input + output/2`. Raises tpot ~2-3% on SHORT/MID, ~0.2% on LONG; helps the post-fix-37
   fast cells, slightly worsens llama4 SHORT. Deferred with the MFU item (same cells, same
   direction); the CAPACITY convention stays at full `input+output` — steady-state there was
   REFUTED (overshoots SHORT by +7%).

## Ruled out as causes (for completeness)

**CORRECTION (2026-07-03 second pass):** the entry below that ruled out communication cost was
only ever measured at TP=1/TP=2 operating points — it did NOT cover the TP=8 regime, where the
ring-latency term was in fact U9's root cause (CHANGES.md item 37). Scope rulings to the
configs actually measured. Additionally ruled out this pass: steady-state KV CAPACITY sizing
(overshoots, wrong direction — twice confirmed), decimal-GB capacity constants (wrong direction;
paper's pools are consistent with GiB), and the KV fork's apparent "7.4% optimizer↔live weight
drift" (a GiB-vs-GB units artifact in the comparison itself, not a real drift).

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
TPS-ratio residual (confirmed via full before/after re-sweep — see CHANGES.md's U4 resolution
record) — it only affected
the *reported batch-size number* at U8's specific operating point, not the throughput metrics
U2/U4/U8's TPS components are about.
