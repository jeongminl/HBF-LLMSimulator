# Known unresolved bugs and brittle code

Bugs and fragile spots that were investigated but **not fixed** (either because a live repro
requires a currently-untriggerable config, because they're genuinely low-priority/dormant, or
because a decision was explicitly deferred to the user). See `CHANGES.md` for what *was* fixed —
including the 2026-07-02 bug-fixes session (items 66-74) that resolved everything else that used
to live in this file and in the now-deleted `BUGS_HIDDEN_BY_FLAGS.md`. Full investigation/
verification detail for that session lives in `ledgers/BUGS_FIXES.md`. Listed roughly in order of
how likely each is to actually bite someone.

> **2026-07-08 — correctness-first campaign, landed.** This campaign abandoned paper-number
> reproduction as a goal and prioritised technical correctness. A codebase-wide
> internal-inconsistency audit (17 items, see `ledgers/CODEBASE_INCONSISTENCIES.md`) plus a
> CB1↔worktree source diff surfaced items **#28-#33** below and re-scoped several existing ones.
> Items marked **FIX SCOPED (2026-07-08)** are now merged to `main` (I3/I8/I4, the CB1 fixes,
> and the `use_flash_attention` master-flag refactor — see `ledgers/FA_FLAG_SPEC.md` — all landed
> and validated; see `CHANGES.md` for the commit history and full-sweep verification results). The
> FA-flag family (score/softmax-materialization coherence) is resolved by that refactor and is NOT
> tracked as a bug here.

## 1. Disaggregated path (`disagg_system=on`) lost its prefill→decode KV transfer term

An earlier session deleted a block in `cluster.cpp::runIterationSumGenSplit` (the
`disagg_system=on` execution path, currently unused — `config.yaml` always sets `disagg_system:
off`) that modeled a one-time interconnect transfer cost when a sequence moves from a prefill node
to a decode node. It was removed because (a) it double-counted against the correct, separate
per-decode-step KV-write overlap logic that already exists in `attention_gen_impl.cpp` and fires on
*both* execution paths, and (b) it computed a flash-write-bandwidth latency but then discarded it
in favor of interconnect bandwidth, so it never actually modeled a flash write despite appearing to.
Net effect: the disagg path's per-step KV-write penalty is intact (confirmed, both paths share the
same kernel), but the *disagg-specific one-time transfer* event is no longer modeled at all.

**Not re-implemented in the 2026-07-02 bug-fixes session either**: `disagg_system` is dead-in-
practice in every config this repo ships, and re-adding a "fixed" version with no disagg
integration test to verify it against would risk repeating the same class of mistake under a
different guise.

**Intended fix, for the record** (do this only alongside a disagg-system integration test): a
one-time transfer cost, sized on the transferred sequence's `input_len` (not the per-token KV-write
size — this is a single one-shot handoff, not a recurring write), charged at **interconnect**
bandwidth (`device_ict_bandwidth`/`node_ict_bandwidth` depending on whether the prefill and decode
nodes are co-located, matching `communication.cpp`'s existing `AllReduce`/`PipelineStage` bandwidth
selection pattern), applied exactly once per sequence at the moment `updateSchedulerSumGenSplit`
first moves it from the sum queue to the gen queue — **not** added to the per-decode-step
`getKVWriteDuration` overlap logic already in `attention_gen_impl.cpp`, which must remain untouched
to avoid reintroducing the double-count.

## 2. `runIterationMixed` has no structural defensive check for `decode_mode`

Decode-only correctness (i.e., never letting prefill compute time leak into the reported
decode-focused TPOT) currently depends entirely on `decode_mode: on` staying set by convention in
`config.yaml`. `Cluster::runIterationMixed` itself has no internal guard — it unconditionally
accumulates every iteration's device time into `total_time`, with no check for whether that
iteration happened to process prefill ("sum") sequences. This is safe today only because
`decode_mode: on` guarantees no sequence ever enters the prefill/"sum" bucket in the first place
(verified: `scheduler.cpp`'s `pushDummySeq` forces `current_len = input_len` at creation). If
`decode_mode` were ever turned off while using `runIterationMixed` (i.e., without also setting
`disagg_system: on`), prefill compute would silently contaminate the decode critical path.

**Mitigated, not structurally fixed, in the 2026-07-02 bug-fixes session** (`CHANGES.md` item 73):
a one-time runtime warning now fires on the exact contamination condition
(`!config.disagg_system && scheduler->hasSumSeq()`), so the failure mode is no longer silent — but
`total_time`'s accumulation itself is unchanged, and no hard guard prevents the contamination from
happening.

## 3. `checkHeteroMemorySize()` is confirmed dead code with a latent capacity-math bug

Separate from `Cluster::checkMemorySize()` (which received the F8/item-58 fix), `checkHeteroMemorySize()`
in `cluster.cpp` has its own capacity-check logic, including a hardcoded `3.3 * 1024^3` ("3.3 GB",
labeled "Non MoE weight") magic-number subtraction, and does not fold activation size into its
capacity comparison at all (activation is only factored into a separate `avail_capacity`
batch-shrink calculation further down, not this function's initial gate) — inconsistent with
`checkMemorySize()`'s scarce-tier gate (`footprint.h`'s `checkCapacity`/
`scarceTierActivationLimit`), which both the optimizer and the live simulator's main gate share.

**Audit resolved in the 2026-07-02 bug-fixes session**: `grep -rn checkHeteroMemorySize` across the
entire tree (excluding its own declaration/definition) returns **zero call sites** — this is dead
code, not merely under-exercised or "unclear whether used." Left in place (not deleted), with an
audit comment added directly above the function. Its capacity-math inconsistency remains
unaddressed and should not be trusted if the function is ever wired back up; the decision of
delete-vs-repair-vs-keep is left to the user now that its dead-code status is unambiguous.

## 4. MLA prefill attention on flash never models KV-read timing — **RESOLVED (2026-07-08, commit 295dbeb): audit item I15**

`MultiLatentAttentionSumExecutionGPU/Logic/PIM` and `AbsorbMLASumExecutionGPU/Logic/PIM` (6
functions, `attention_sum_impl.cpp`) never route flash KV-read timing through
`getAttentionMemoryDuration`, unlike the base GQA `AttentionSumExecutionGPU`. This requires an MLA
model *and* prefill mode together — doubly dormant, since no sweep in this repo uses either. Each
of the 6 functions has its own memory-size accounting (including a distinct FlashAttention
SRAM-tiling algorithm branch, `use_flash_attention`, orthogonal to the HBF flash-memory tiering),
and none of them are currently exercised by any config this repo runs end-to-end.

**Fixed:** `MultiLatentAttentionSumExecutionGPU`'s Scoring/Context loops now accumulate
`total_kv_read_size`/`total_act_size` and call `getAttentionMemoryDuration` once post-loop,
mirroring GQA-Sum. Byte-identical on all paper-1 GQA cells (verified); structural-only for MLA
(no paper-1 MLA model to numerically test against).

**Not implemented**: given the size (6 large, intricate functions across GPU/Logic/PIM ×
MLA/Absorb × flash-algorithm/non-flash) and the total absence of a way to verify correctness
against a known-good reference (MLA+prefill has no test coverage anywhere in this codebase),
implementing this blind was judged higher-risk than valuable — a wrong "fix" here could silently
corrupt a currently-inert code path in a way nobody would notice. **Recommended follow-up:** mirror
the `AttentionSumExecutionGPU` pattern — accumulate `total_kv_read_size`/`total_act_size` across
the seq×kv_head loop, call `getAttentionMemoryDuration` once post-loop — for each of the 6
functions, but only after a real MLA+prefill test config exists to validate against.

## 5. `logic_x`/`pim_x` speedup multiplier is inconsistent between the routing heuristic and the timing model

`logic_x`/`pim_x` are used by the expert high/low assignment heuristic (`route.cpp`'s
`getIdxHigh`/`getIdxHighOptimal`) to predict a speedup for the "low" (LOGIC/PIM) processor, but the
actual LOGIC/PIM kernel timing (`*_impl.cpp`) never applies that multiplier — the routing heuristic
and the timing model disagree about how fast LOGIC/PIM actually is. This is a design
inconsistency, not a crash, and only matters once `parallel_execution: on` is used (off in every
current sweep). Changing kernel timing based on a routing-heuristic constant is a modeling
decision for the user, not a clear-cut bug fix — left as-is, flagged here for a deliberate
decision.

## 6. Non-llama3/llama4 model presets' precision assumptions are unverified

This session and the 2026-07-02 bug-fixes session together verified and corrected `llama3_405B`'s
precision (FP8→BF16) and confirmed `llama4_maverick`'s (already correct BF16) against the paper's
explicit constraints. The other model presets in `model_config.h` — `mixtral`, `openMoE`,
`llama7bMoE`, `grok1`, `deepseekV3`, `llama4_scout` — were not cross-checked against any external
reference for their `precision_byte` defaults (`deepseekV3`'s `precision_byte=1`/FP8 does match
DeepSeek-V3's published FP8 training/inference precision, but this wasn't independently
re-verified beyond what's already stated in `CHANGES.md`). If any of these are ever used in a
reported sweep, their precision assumption should be verified the same way (footprint math + any
available capacity-feasibility constraint from a reference source), not just trusted as-is.

## 7. `page_size_bytes` is dead/vestigial config surface (not a bug)

After an earlier fix to the non-chunked KV-read overflow branch (which now chunks by SRAM staging
capacity, matching the plane-parallel physical model, instead of charging one page-read latency
per literal 4KiB page), `page_size_bytes` is no longer read anywhere in the timing formulas
(confirmed: grepped all of `src/hardware/layer_impl.h` and the `attention_*_impl.cpp` files;
`src/dram/hbf_memory_config.h` still defines and documents it as hardware geometry). Left in
place — removing unused config surface is a cleanup, not a bug fix, and doing so risks confusing
future extensions of the flash timing model that might want to reintroduce page-granular
accounting.

## 8. Non-chunked-attention fallback branch is currently a dead no-op (not a bug)

`use_chunked_attention`'s non-chunked branch (`layer_impl.h`'s `getAttentionMemoryDuration`) is,
under the pinned default `chunk_size: 0`, byte-for-byte identical to the chunked branch (both chunk
by full SRAM staging capacity) — confirmed by reading both branches side-by-side. Flipping the flag
today changes nothing; its real fallback behavior (one page-latency per full KV read instead of
per chunk) has never actually been exercised by any config in this repo.

## 9. SRAM intermediate-data capacity gate omits several resident tensors

`peakIntermediateBytes()` (`src/model/footprint.h`), the layer-scoped `max(attention, FFN)` gate
that dictates HBF+/CONV+ SRAM-boundness, does not charge several tensors that are genuinely resident
in the 320-MB pool: the **KV-write on-chip staging burst** (the dominant one — timed via
`getKVWriteDuration` but given zero byte reservation), the **attention score/softmax tile**
(charged 0 on HBF+/CONV+ because `getAttentionMemoryDuration` staging-chunks only `kv_read_size`),
the **LM-head logits** (a step-level tensor the layer-scoped formula can't see), and the TP
all-reduce scratch + MoE dispatch/GateOut. This is the mechanism behind `PAPER_INCONSISTENCIES.md`'s
U7 (sim HBF+ batch grows/SLO-bound vs the paper's flat/SRAM-bound bars).

**RESOLVED (2026-07-06): merged to `main`, unconditional, no flag.** The `faithful-intermediate-gate`
branch (`footprint.h::intermediateExtrasBytes` — KV-write staging + score tile + all-reduce scratch +
MoE dispatch/GateOut + tiled LM-head logits, all single-buffered) was merged into `main`. It first
landed behind `system.faithful_intermediate_gate` (default false, then default true), but since this
is the paper-conformant accounting rather than an experimental A/B toggle, the flag itself (and the
now-fully-subsumed older `kv_write_sram_gate`) was removed entirely — `cluster.cpp` and
`parallelism_optimizer.cpp` both add `intermediateExtrasBytes` unconditionally on top of
`peakIntermediateBytes`. Verified: clean rebuild, smoke-run completes without error. **Still not the
architecturally-correct version** — this is a hand-coded formula, not the liveness-maximum sweep over
the op graph (see U7 and the companion over-count note below); that more rigorous alternative exists
in the separate `liveness-sweep` branch (`agent-aaedd3e8f91fe883f` worktree,
`TopModuleGraph::livenessPeakActBytes()`) but is NOT yet merged. **Consequence: every figure
regeneration, checkpoint, and error-rate comparison produced earlier in this session (Figs 3–7,
`error_rates_detail.csv`, and this session's six new `PAPER_INCONSISTENCIES.md` findings) was computed
under the OLD under-counting behavior and is now stale for any cell where this gate changes the
winning config or bound reason — re-sweep before trusting those numbers against this build.**

**Companion over-count found by the paper2-extension team (2026-07-06,
`.claude/worktrees/paper2-extension/FINDING_peakIntermediateBytes_attention_overcount.md`):** the
same function also SUMS every attention-phase intermediate as if concurrently resident
(`footprint.h:186-200`), when in reality they form a def→last-use dataflow chain and the true peak
is the max over that chain, not the sum. Architecture-dependent — badly over-counts MLA (many
attention intermediates: DeepSeek-R1 measured 204,864 summed vs. 114,240 true peak-of-chain, a
1.79× / 45.8% over-count, batch/context-independent), barely affects GQA (llama3/llama4, this repo's
paper1 models — few attention intermediates, and both are FFN-bound anyway so the over-count is
inert for U7's current scope). Replacing the sum with the rigorous peak-liveness value does NOT by
itself converge on paper2's reported SRAM figure — it flips R1 from +21% to −20% — because the
under-count inventory above (items this section already tracks) is simultaneously in effect; the two
errors partially cancel, which is very likely why paper figures land between a naive sum and a
rigorous peak. The two fixes are not independent — the full architecturally-correct answer is a
graph-based per-tensor liveness pass (max-concurrency sweep over def/last-use intervals, using the
module graph's existing producer→consumer edges — see the separate `liveness-sweep` branch,
`TopModuleGraph::livenessPeakActBytes()`, NOT yet merged) that both (a) stops over-counting
attention's dataflow chain as a sum and (b) adds the currently-missing tensors (LM-head logits, score
tile, KV-write staging) as their own liveness intervals. FFN is NOT over-counted (gate/up genuinely
coexist for the SiLU multiply) — this finding is attention-specific and orthogonal to the FFN-side
accounting.

**RESOLVED (2026-07-06): the over-count itself is fixed on `main`, independent of the liveness-sweep
migration.** The paper2 team's own derivation gives an exact, hand-derivable peak-of-chain formula
for the `use_absorb` (MLA) branch of `peakIntermediateBytes` — no graph tracing required. Fixed
directly in `footprint.h`: replaced the 5-term sum (`common_prefix` + `tr_k_up` + `attn_context` +
`v_up` + `out_proj`) with `max(pre_score_peak, post_score_peak)`, where `pre_score_peak` = H (residual
carry) + c_kv + rope + query_proj(+rope) + tr_k_up (the set alive up to score computation, correctly
EXCLUDING `q_lora`/c_q, which dies once query_proj is produced — the previous formula's biggest
single over-count source) and `post_score_peak` = H + attn_context + v_up + out_proj (the set alive
after scores, which never coexists with the pre-score set). **Verified:** hand-computed against the
paper2 finding's own DeepSeek-R1 numbers — reproduces 204,864 (old sum) → 114,240 (new peak) exactly,
ratio 1.7933 ≈ their cited 1.79×. Clean rebuild; a smoke run on the default `deepseekV3`/`use_absorb`
config shows `ACT:` drop from 0.0105839 GB (old) to 0.00847626 GB (new) — confirms the binary picks
it up. llama3 (GQA, untouched `else` branch) is **byte-identical** before/after (`ACT: 0.105263GB`
both times) — this fix is a no-op for every current paper1 preset, exactly as scoped. **Not done:**
the `compressed_kv`-without-absorb branch has the same class of issue in principle but no verified
derivation exists for it and no current preset reaches it (only `deepseekV3` sets `use_absorb`),
so it was left alone rather than guessing; the missing-tensor UNDER-count above is still open and
requires the separate liveness-sweep merge (or hand-adding the missing terms) to close.

**Note (2026-07-07, paper-1 bug-hunt campaign):** the gate merge above fixed Fig-3's batch match for
llama3/HBF+ (batch error down to +6.4%) but this in turn EXPOSED a separate tpot-realism gap in
Fig-7's online PEC metric — llama3/HBF+ PEC error jumped from a stale-low 37.8% to a freshly-measured
**60.5%** at 8/16-GPU, because PEC folds batch error together with the sim-vs-paper tpot ratio, and
fixing the batch side alone left the tpot side (an MFU-class deferred lever, not this gate) fully
exposed. See `PAPER_INCONSISTENCIES.md`'s Fig-7 entry for the full zero-free-parameter decomposition.

## 10. `LmHead::forward` bypasses `device->execution()` — its FLOPs/DRAM-energy are uncounted

`src/module/lm_head.cpp` writes timing straight into `device->status` (`lm_head.cpp:67-74`) instead
of routing through the normal `device->execution()`/`ExecStatus` path, so the LM head's compute
FLOPs and DRAM-access energy never enter the energy totals (latency itself is correct). Affects
every model's last PP stage. Not distinctive to LmHead — the same direct-`device->status` idiom is
used by `communication.cpp`, `rope.cpp`, `residual.cpp`, `layernorm.cpp` — but LmHead is the largest
op that skips energy accounting.

## 11. `Embedding::forward` charges zero time and zero energy

`src/module/embedding.cpp` allocates its output and returns — no `device->execution()` call, no
compute or memory-time charge at all, for the vocab-row gather at the start of every step. Dormant
for latency (the gather is cheap) but it means the embedding contributes nothing to energy either.

## 12. `AttentionMixedExecutionLogic` silently returns 0 duration on `ProcessorType::LOGIC`

`src/hardware/attention_mixed_impl.cpp` stubs to a 0-duration return if ever routed to
`ProcessorType::LOGIC`, instead of failing loudly. Only reachable under `parallel_execution: on`
(off in every current sweep), but a silent 0 is a landmine if that path is enabled.

## 13. Attention gen/sum impls are ~9× near-duplicated (the `#R1` refactor was never done)

`attention_gen_impl.cpp` (~2790 lines) and `attention_sum_impl.cpp` (~3068 lines) each hand-roll ~9
near-identical GPU/Logic/PIM × {base, MLA, absorbed-MLA} implementations. The Linear/Activation
shared-core refactor was applied elsewhere but never to attention. This is the largest maintenance
hazard in the codebase: any attention-side change (e.g. declaring the score tile as a real
intermediate for the item-9 gate fix, or the item-4 MLA-prefill fix) must be replicated ~9× per
file, which is error-prone and blocks several other fixes.

## 14. `ParallelismOptimizer::EvaluateConfig` duplicates the timing formulas by hand

`src/optimizer/parallelism_optimizer.cpp` re-implements the simulator's weight/KV/activation/latency
formulas as a hand-written parallel copy (contract-by-comment, not shared code with `layer_impl.h`/
`expert.cpp`/`footprint.h`). This has already drifted and been re-synced multiple times (see the
"lock-step"/"parity" corrections in `CHANGES.md`); every future timing or footprint change must be
mirrored in both places or the optimizer's config search silently diverges from the live simulator.

## 15. MoE `Activation` allocates a `GateOut` buffer that is computed but never consumed

`src/module/activation.cpp` (used by `FeedForward3Way`, `ffn_way==3`) allocates two intermediate-
width tensors, `ActOut` and `GateOut`; `layer.cpp:436-440` feeds `up_proj_out` to the down-proj
directly and never reads `GateOut`. It is charged for bandwidth timing but its value is discarded —
verify this is a benign 2-buffer artifact and not a mis-wire that drops the gated activation.

## 16. RMSNorm allocates a distinct output tensor (the "in-place/free" capacity treatment is an assumption)

`src/module/layernorm.cpp` creates a separate `layer_norm_output` tensor rather than normalizing in
place. The intermediate-data gate treats layernorm as free (folded into the residual/input term) on
the physical argument that a real kernel can normalize in place — a defensible modeling assumption,
but it is inconsistent with how the code actually allocates, and with how the all-reduce/GateOut
buffers (item 9) are treated as distinct. Flag for consistency when the item-9 gate is reworked.

## 17. Dead, non-compiling `src/dram/data_object.{h,cpp}`

Absent from `CMakeLists.txt` sources; header and impl disagree on member names and ctor signature.
Safe to delete; kept only because no one has.

## 18. `SystemConfig` hardware presets are built via ~30-argument positional constructors

`src/hardware/hardware_config.h`'s six presets are constructed with long positional argument lists
(no named fields), which is easy to misalign on future edits. A designated-initializer or builder
refactor would remove a whole class of silent-misalignment bugs. Cleanup, not a live bug.

## 19. Runtime PP time-propagation charges 3× stage-time for a 2-stage pipeline — **FIX IN PROGRESS on branch `bughunt-paper1-campaign`**

**Confirmed, byte/ratio-exact (2026-07-07, paper-1 bug-hunt campaign, Hunt B + refuter).** The
runtime charges exactly **3× stage_time** for a pp=2 pipeline (a correct serialized 2-stage pipeline
should charge 2×). Discriminator (forced tp8/pp2/dp1, batch 16/64/128/256/400): measured/stage ratio
= **2.999/3.005/3.009/3.014/3.017** — a flat +50% over-count (one whole extra stage-time), not noise.
Physical hop cost is negligible (0.021 ms); the large "comm" bucket seen on pp>1 runs is this
propagation artifact being stamped into the Comm bucket (`cluster.cpp:943`), not real communication
(a refuter's initial "18.1ms real hop" reading was itself this same artifact).

**Root cause:** CHANGES-35 flipped `PipelineStage::forward`'s propagation from `max()` to `+=`
(`communication.cpp:587-588`). With tp≥2, the destination stage has already accumulated its own work
by the time this fires, and `sync_devices()`'s max-broadcast (`module_graph.cpp:73`) then spreads the
already-bumped clock across the destination stage's own AR group on every layer — compounding to a
net 3-stage total instead of 2. History: pre-35 undercounted at ~1×stage; item-35 overshot to 3×,
skipping the correct 2× entirely. Affects every tp≥2, pp>1 forced or optimizer-selected run, and
retroactively invalidates CHANGES-26's PP measurements (which measured this same artifact).

**Fix (two parts, under review):** (A) the accounting bug itself — 3S→2S in the
PipelineStage/lock-step propagation path (uncontroversial once isolated). (B) a paired
physical-realism model change — decode PP with continuous batching should overlap microbatches
(steady-state period `W + K·B/pp` + hop, not the naive sum of stages); the paper is silent on PP
scheduling (verified by PDF grep), so (B) is argued from physics, not paper-conformance, and (A) alone
is not sufficient to reproduce the paper's 16-GPU Fig-3/Fig-4 growth (a correct-serialized 2S still
loses to dp2 everywhere). See `PAPER_INCONSISTENCIES.md`'s Fig-5 entry for the interaction with the
16-GPU communication-share item, and `ledgers/FINDINGS_REGISTER.md` (2026-07-07 section) for the full
verdict trail.

## 20. Optimizer's analytic activation-gate reads a stale `e_tp_dg` for `ep>1` configs — **FIX IN PROGRESS on branch `bughunt-paper1-campaign`**

**Confirmed, byte-exact (2026-07-07, paper-1 bug-hunt campaign, Hunt E).** `peakIntermediateBytes`
(`footprint.h:231`) and `intermediateExtrasBytes` (`footprint.h:367`) both read `model.e_tp_dg` from
the optimizer's const `ModelConfig` (always **1** in the analytic sweep), while
`num_routed_expert_per_device` (`parallelism_optimizer.cpp:323-324`) correctly uses the swept `ep` —
so the `×ep`/`÷e_tp` cancellation the formula relies on fails for every `ep>1` config, and the analytic
capacity ceiling undershoots the true simulator ceiling. Byte-exact repro at ep2: 43.51 MB over-charge
(18.68 MB peak + 24.83 MB GateOut, the un-cancelled `e_tp=2` term); analytic cap comes out 3027
vs a true 3476. The live simulator itself is correct (it writes `e_tp` into its own config before
running) — this is an optimizer-analytic-only bug.

**Consequences:** (1) a latent pruning-invariant violation — the deflated `seed_tps` for `ep>1`
configs could prune a borderline true `ep>1` winner unverified (no confirmed case of an actual winner
being dropped in audited cells, but the invariant is violated in principle); (2) this is also the
dominant driver of PEC offline-sweep cost — ~94% of the probe runs in maverick multi-GPU cells are
`ep>1` probe walks caused by this bug (~82% of those cells' total cost), because the deflated ceiling
forces the exponential prune-verification search to walk many more candidate configs than necessary.

**Fix (surgical, under review):** pass a local `ModelConfig` copy with `mc.e_tp_dg` set to the swept
`ep` into both `footprint.h` calls inside `EvaluateConfig`, instead of reading the shared const
config's default. Audited blast radius: `e_tp_dg` is the *only* stale sweep field in these formulas
(dense/1-GPU cells are already byte-exact and untouched by this fix). See
`ledgers/FINDINGS_REGISTER.md` (2026-07-07 section) for the full verification trail.

## 21. `classify_failure` mislabels HBM-capacity OOM failures as `"flash"` (cosmetic)

`run_experiments.py:481`'s `classify_failure` maps an "HBM capacity exceeded" failure message into
the `"flash"` bucket — semantically it should be `"capacity"`. Affects only Fig-3's failure-cause
hatching/labeling in generated plots; no numeric consequence for any reported figure. Low priority,
cosmetic-label fix only.

## 22. CHANGES#61 documents the gen-path `fill_amortize` fix but not the still-open sum/MLA residual — **RESOLVED (2026-07-08, commit 2810d8f): audit item I3**

`CHANGES.md` item 61 documents fixing the gen-path K/V double-fill (`fill_amortize`), but the same
class of asymmetry remains on the sum (prefill) and MLA-gen paths, which still effectively run at
`fill_amortize=1` rather than the gen-path's corrected value. Confirmed dormant for paper-1: the sum
path never executes under `decode_mode: on` (this repo's config for every reported sweep), and MLA-gen
is not exercised by either paper-1 model (llama3/llama4 are both GQA). Doc-gap only — worth a note so
a future paper-2/prefill or MLA-model sweep doesn't rediscover this as a "new" bug.

**Fixed:** added `fill_amortize_calls=2` to GQA-Sum's Scoring/Context and MLA-Gen's Score/Context
calls, matching GQA-Gen's existing convention. Verified byte-identical on all published decode
cells (4-cell sanity + full 60-cell sweep).

## 23. `attention_sum_impl.cpp` memory_util divisor drift (print-only)

`attention_sum_impl.cpp:358`'s `memory_util` diagnostic computation uses a divisor that has drifted
from the value it's nominally supposed to track. This field is print/logging-only — it never feeds
into any timing, capacity, or energy calculation — so the drift has no effect on any reported number.
Cosmetic; flagged so a future diagnostic-trust exercise doesn't need to re-derive this from scratch.

## 24. Cross-node AllReduce is flat (no hierarchical NCCL model) — **RESOLVED (2026-07-08, commit 32b53a0): audit item I8 (hierarchical AllReduce)**

`communication.cpp`'s AllReduce charges the full ring volume `2(n−1)/n · size` over ONE link — the
inter-node IB link (100 GB/s) whenever the group spans nodes — instead of a hierarchical
NVLink-reduce-scatter → IB-exchange(size/gpus_per_node) → NVLink-all-gather decomposition (~8× cheaper
on the IB leg at 2×8 GPUs). Attention-TP and MoE-gather AR groups are contiguous `ne_tp_dg` blocks and
never span nodes for the swept models (tp ≤ num_kv_heads = 8). The one reachable cross-node case is
`all_reduce_for_e_tp` (`expert.cpp:106-108`, group sized by `e_tp_dg` — bounded by devices-per-stage,
NOT num_kv_heads): maverick ep16 at 16 GPU spans both nodes. Both the analytic ranker and the runtime
price it identically flat (verified — no ranker/runtime skew, pruning invariant holds), so such
configs self-suppress consistently. Measured (refF_etp16, b=512): ep16 loses to ep1 by 6.07% as-is;
a hand-corrected hierarchical AR would flip it to +5.15% at that batch — but at the figures'
throughput-max batches ep16 loses by ~26%, far beyond any AR correction (the n=16 latency floor
doesn't shrink), so **no paper-figure winner is suppressed**. Fix (only if a num_kv_heads ≥ 16 model
or small-batch cross-node ep study is ever swept): hierarchical AR in both `communication.cpp` and the
optimizer's comm term, in lock-step.

**Fixed:** `AllReduce::forward` now models a two-phase hierarchical cost (intra-node reduce-scatter
+ all-gather on `device_ict`/NVLink, inter-node exchange on `node_ict`/IB), reducing exactly to the
prior formula when the group fits in one node. Verified byte-identical on all gpu≤8 (single-node)
cells and the full 60-cell sweep at gpu=16 (no paper-1 group spans nodes, so zero published-cell
movement) — the hierarchical model is now available for any future config where one does.

## 25. `logic_memory_bandwidth`/`pim_memory_bandwidth` go stale after the Table-I memory override — disagg/PIM-only

`SystemConfig`'s ctor computes `logic_memory_bandwidth = memory_bandwidth × logic_x` (and pim
equivalent) ONCE at construction (`hardware_config.h:90-93`) from the GPU preset's ctor argument
(e.g. Rubin's non-load-bearing 22 TB/s). `eval/test.cpp:144-145` later overwrites
`system_config.memory_bandwidth` from the memory preset (Table-I values, e.g. HBM4's 12.8 TB/s) but
never recomputes the logic/pim derivatives — they stay Rubin-derived on every preset. Dead for this
campaign: only the Logic/PIM execution variants read them (`attention_gen_impl.cpp:250,402,…`), and
`processor_type: GPU` means those paths never run. Would bite any future GPU+LOGIC/PIM or
disagg sweep. Fix when needed: recompute the derivatives after the override (or make them accessors).

**Related — CB1 source diff (2026-07-08):** independent of the stale-value issue above, the
worktree ctor also **dropped CB1's `/2.0`** on both derivatives — CB1 (`LLMSimulator_HBF`,
`hardware_config.h:100-101`) computes `logic_memory_bandwidth = memory_bandwidth * logic_x / 2.0`
(and pim likewise), so the worktree's are **2× CB1's**. Same PIM/LOGIC-only dormancy. Reconcile
against the paper's Table-I hardware model when either path is exercised — determine whether the 2×
is an intentional worktree change or an accidental drop before trusting it.

## 26. `batch % dp != 0` strands up to dp−1 sequences permanently (unreachable via the sweep driver)

`batch_size_per_dp = total_batch_size / dp_degree` floors (scheduler.cpp:17), every shard caps at the
floored value, but `fillSequenceQueue`/`fillRunningQueue` compute the refill gap against the
un-floored `total_batch_size` — so when `total_batch_size % dp_degree != 0`, up to `dp−1` sequences
can never be placed and pile up in `sequence_queue` forever (a PERMANENT running-batch shortfall vs
what the caller believes it configured, growing queue memory). Unreachable through
`run_experiments.py` (its search only ever probes batch values that are exact multiples of the
config's dp — verified in code and empirically in the CSV corpus), so no paper-comparison number is
affected. Bites anyone driving the binary directly with a non-divisible batch. Fix when touched:
either assert `batch % dp == 0` loudly at scheduler construction, or compute the gap against
`dp_degree × batch_size_per_dp`. (Found by the Hunt-H refuter, 2026-07-07.)

## 27. `all_reduce_for_e_tp` message sized on the full pre-routing batch — e_tp>1-only over-charge

`expert.cpp:188` passes ExpertFFN's own `input` (shape `[batch_per_dp, hidden]`) to the expert-TP
AllReduce, but the tensor that physically needs reducing is the e_tp-group's partial down-proj
outputs, whose true row-count is the tokens routed to this device's resident experts. Over-charge
factor ≈ `parallel_num/e_tp_dg`: EP1 → AllReduce is an algebraic no-op (n_ranks=1), bug inert;
EP2/TP4 → exactly 2× (measured: correct 5,335,040 B vs used 10,670,080 B per MoE layer); EP4/TP4 →
coincidentally exact (one group holds all tokens). NOT an optimizer/runtime divergence — the
optimizer's comment (`parallelism_optimizer.cpp:663-666`) documents deliberately replicating this
exact sizing, so ranker and runtime agree and the pruning invariant is unaffected; the disagreement
is with the true MoE architecture. Live exposure: only e_tp>1 configs, which currently lose the
searches (partly *because* of this over-charge — fixing it makes ep2-class configs slightly
cheaper). Fix when touched: size the AR from the routed-token count (Σ tokens over resident
experts), and update the optimizer mirror in lock-step. (Hunt-J refuter, 2026-07-07;
`refJ_etp_check.py`.)

## 28. LM-head logits materialized in timing but tiled in capacity (audit item I2, deferred)

`lm_head.cpp:58,92` charge the full `m·(n_vocab/tp)` logits **write** as memory traffic, while the
capacity gate (`footprint.h:390`) counts only a 2048-wide argmax tile. Greedy-argmax decode never
needs the full vocab row resident, so timing over-charges a tensor capacity treats as a bounded
tile — the same "materialized-in-timing / tiled-in-capacity" pattern as the attention score
(handled by the `use_flash_attention` flag), but a *different* physical mechanism (argmax logits,
not attention) so it is NOT covered by that flag. Low numeric impact: on flash configs the
hidden×vocab weight read dominates under `max()`, and on HBM4 it's a small additive term. Deferred
this campaign. Fix: charge only the argmax-tile write in timing to match capacity.

## 29. MoE GateUpdate aggregation leader gated on the wrong device (PP bug, CB1 diff) — **RESOLVED (2026-07-08, commit d8fdf45)**

`route.cpp:157` reads `if (device->device_total_rank == 0)`; CB1 uses `== device_list.front()`
("With PP, device 0 may not belong to this stage's MoE layer"). Under `pp_dg>1`, any MoE layer on a
pipeline stage whose `device_list` excludes global rank 0 has no device satisfying `==0`, so
`aggregate_expert` (the per-layer `update_expert` regen + batch-sum of
`num_token_in_expert`/`local_num_token_in_expert` + the token-conservation `assertTrue`) silently
never runs for that layer — it reuses whatever draw rank 0 last produced for a different layer, and
MoEScatter/MoEGather then size the all-to-all off stale counts. Timing impact is bounded
(same-distribution draw) but it is incorrect and it disables the invariant check for non-zero
stages. Only bites llama4 + pp>1. The MOE_TAG_FIX_SPEC §4.5 micro-vector block (`route.cpp:193-202`)
is inside the same `==0` block and inherited the defect (fixed alongside — same commit).
**Fixed:** `route.cpp:157` now compares against `device_list.front()`.

## 30. Sequence token accessors are `int`, should be `int64_t` (CB1 diff) — **RESOLVED (2026-07-08, commits d8fdf45 + e4b0264)**

`src/scheduler/sequence.h`'s `get_gen_process_token()`/`get_sum_process_token()`/
`get_total_sequence_length()` return `int`; CB1 uses `int64_t`. These feed comm-size and footprint
byte/FLOP math. Safe at paper-1 scale (batch × seq_len ≈ 5×10⁷ ≪ 2³¹) but a large-batch/long-context
sweep can overflow 2³¹ and silently corrupt sizing.

**Fixed:** widened to `int64_t` in `sequence.h` (d8fdf45). Note: the matching `sequence.cpp`
out-of-line definitions were initially left uncommitted in d8fdf45 (a header/definition type
mismatch that would fail to compile from a clean clone) — caught and fixed in e4b0264.

## 31. KV-write reuse / seeded-steady-state model absent vs CB1 (dormant divergence)

The worktree folds KV-write inline with HBF flash physics (page-program latency + double-buffer
hiding) and amortizes only the prompt write; CB1 has a standalone `decode_kv_write.cpp` modeling
prompt-prefix reuse / seeded steady-state bulk writes. Both are GQA/TP-correct. Moot for paper-1
(`reuse_kv_cache: off`), but if reuse or seeded steady-state is ever enabled the worktree lacks that
accounting — a conscious divergence, flagged so it is not rediscovered as a "bug."

## 32. PIM/LOGIC GQA attention never model the score matrix (documented `use_flash_attention` gap)

The `use_flash_attention` master flag gates score/softmax materialization on the GPU attention
paths, but the PIM/LOGIC GQA roofline variants (`attention_gen_impl.cpp` G2/G3,
`attention_sum_impl.cpp` S2/S3) never modeled the score in either branch and were deliberately left
as-is (ruling, 2026-07-08 — see `ledgers/FA_FLAG_SPEC.md`). Consequence: the flag's `OFF`
(materialized) semantics are **not modeled on PIM/LOGIC** — those paths behave as always-`ON`
there. **Verified as the regression tripwire during implementation**: grep-confirmed zero
`use_flash_attention` references in these functions, and both are byte-identical across `ON`/`OFF`.
Only matters for hetero/disaggregated runs, never the paper-1 GPU sweeps.

## 33. Missing `#include <cstdint>` in ramulator2 `utils.h` (GCC-13 build break)

`src/dram/ramulator2/src/base/utils.h` uses `uint64_t` without including `<cstdint>`, which fails
to compile under GCC-13. The submodule + `patch/ramulator2_pim.patch` otherwise handle the
ramulator2 setup correctly — this single missing include is the only outstanding build fault. Add
the include.

---

Resolved fixes and the (now historical) paper-vs-simulator comparison items live in `CHANGES.md`.
This campaign's internal-inconsistency audit is in `ledgers/CODEBASE_INCONSISTENCIES.md`; the
`use_flash_attention` flag design is in `ledgers/FA_FLAG_SPEC.md`. `PAPER_INCONSISTENCIES.md` is
superseded (2026-07-08, paper-conformance abandoned) — its dispositions were absorbed into
`CHANGES.md`.
