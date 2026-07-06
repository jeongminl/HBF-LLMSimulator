# Known unresolved bugs and brittle code

Bugs and fragile spots that were investigated but **not fixed** (either because a live repro
requires a currently-untriggerable config, because they're genuinely low-priority/dormant, or
because a decision was explicitly deferred to the user). See `CHANGES.md` for what *was* fixed —
including the 2026-07-02 bug-fixes session (items 66-74) that resolved everything else that used
to live in this file and in the now-deleted `BUGS_HIDDEN_BY_FLAGS.md`. Full investigation/
verification detail for that session lives in `ledgers/BUGS_FIXES.md`. Listed roughly in order of
how likely each is to actually bite someone.

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

## 4. MLA prefill attention on flash never models KV-read timing

`MultiLatentAttentionSumExecutionGPU/Logic/PIM` and `AbsorbMLASumExecutionGPU/Logic/PIM` (6
functions, `attention_sum_impl.cpp`) never route flash KV-read timing through
`getAttentionMemoryDuration`, unlike the base GQA `AttentionSumExecutionGPU`. This requires an MLA
model *and* prefill mode together — doubly dormant, since no sweep in this repo uses either. Each
of the 6 functions has its own memory-size accounting (including a distinct FlashAttention
SRAM-tiling algorithm branch, `use_flash_attention`, orthogonal to the HBF flash-memory tiering),
and none of them are currently exercised by any config this repo runs end-to-end.

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

**Partially addressed:** KV-write staging is implemented behind `system.kv_write_sram_gate` (default
off) and an A/B run reproduces the paper's SHORT SRAM-bound bars (shape + cross-model ratio, ~+20-30%
absolute). **Intended full fix** (recipe in `CHANGES.md`'s "doc cleanup" section): add KV-write
(single-buffered — paper1 sizes by peak occupancy, no double-buffer), score tile (O(chunk),
hardware-set), all-reduce scratch (`batch×hidden/tp`), and MoE dispatch/GateOut; model logits as a
small argmax-consistent tile (NOT full vocab); mirror in `parallelism_optimizer.cpp`. The
architecturally-correct version is a liveness-maximum sweep over the op graph (see U7).

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

---

Paper-vs-simulator numeric/qualitative comparison items live in `PAPER_INCONSISTENCIES.md`
(still-open/explained-not-bugs) and `CHANGES.md` (resolved), not here.
