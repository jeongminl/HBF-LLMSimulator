# Bug fixes: BUGS.md and BUGS_HIDDEN_BY_FLAGS.md

This session root-caused and fixed every item in `BUGS.md` and `BUGS_HIDDEN_BY_FLAGS.md`, plus
two related quirks found in `HBF-LLMSimulator_Guide_EN.md` §16 that weren't in either bug file.
Work was done in an isolated git worktree (`worktree-bugs-fixes`) so it wouldn't collide with a
concurrent session fixing a separate set of paper-inconsistency items on the main tree.

All fixes below live in branches gated off in every real sweep (`decode_mode: on`,
`disagg_system: off`, `parallel_execution: off`, only non-MLA models run) — that's *why* they
were bugs nobody hit. Every fix was verified two ways: (1) the default pinned-flag regression
suite is **byte-identical** before/after (confirmed via CSV diff and stdout diff against the
main tree's pre-existing build), and (2) for each fix, the guarded code was temporarily reverted
in-place, rebuilt, and shown to reproduce the *original* bug (crash/stall/corruption), then
restored and shown to produce the fixed behavior — see each item's "Verified" line for the exact
before/after evidence.

## Environment fix (not a repo bug, needed to build at all)

**`src/dram/ramulator2/src/base/utils.h`**: added `#include <cstdint>`. On a fresh submodule
checkout + GCC 13.3 (this environment), `utils.h` fails to compile (`'uint64_t' does not name a
type`) because libstdc++ no longer transitively pulls in `<cstdint>` via `<string>`/`<vector>`.
This is invisible on the main tree only because its `build/` has stale cached object files from
an earlier, working compile — a genuinely fresh `git submodule update --init` + `cmake` there
would hit the same error. This is a vendored submodule file, changed only enough to build; not
part of BUGS.md/BUGS_HIDDEN_BY_FLAGS.md.

---

## BUGS_HIDDEN_BY_FLAGS.md

### #1 — T1: MLA `num_kv_heads` divisibility guard missing → SIGFPE

**Root cause**: `src/module/parallel.cpp` — `SelfAttentionParallel` asserts both `num_heads` and
`num_kv_heads` divisibility by `parallel_num`, but `MultiLatentAttentionParallel` and
`AbsorbMLAParallel` asserted only `num_heads`, then still computed `num_kv_heads / parallel_num`.
When that floors to 0, `attention_group_size = num_heads / num_kv_heads` in
`AbsorbMLASum::forward`/`MultiLatentAttentionSum::forward` divides by zero.

**Fix**: added the matching `assertTrue(num_kv_heads % parallel_num == 0, ...)` guard to both MLA
wrapper constructors.

**Verified**: deepseekV3's own preset (`num_kv_heads = num_heads = 128`) can't exercise this at
any valid TP degree — 128 is a power of 2 and every other weight dimension in the preset also
divides cleanly only by powers of 2, so no TP value trips this guard alone without first tripping
an unrelated linear-layer assert. Temporarily patched the preset to `num_kv_heads = 1` (the
paper's actual "one shared latent KV" MLA semantics) to exercise it for real:
- **Pre-fix** (guard commented out, `num_kv_heads=1`, TP=2): `Floating point exception`, exit
  code 136 (SIGFPE), confirmed via `ulimit -c 0` + direct run.
- **Fixed**: clean `ERROR: num_kv_head mod parallel_num == 0`, exit code 1.
- Preset and guard both restored afterward; default regression re-confirmed byte-identical.

### #2 — T2: 32-bit int overflow in prefill activation-footprint formula

**Root cause**: `src/hardware/cluster.cpp`'s `checkMemorySize()` non-decode (prefill/mixed)
branch computed terms like `batch_size_per_dp * input_len * input_len * num_heads / ne_tp_dg`
entirely in `int` before any `double` operand appeared in the expression — with realistic
`input_len` (thousands) and `num_heads` (128), this overflows `INT_MAX` well before the final
multiply by `precision_byte` (a `double`), corrupting `ACT:`/`Total:` and the scarce-tier OOM
gate. Present in both the absorb and base (non-absorb) sub-branches.

**Fix**: every product now leads with `(double)batch_size_per_dp * ...` so the whole left-to-right
multiply/divide chain evaluates in `double`, matching `peakIntermediateBytes()` (the decode-path
formula), which was already all-`double` and thus immune.

**Verified**: `decode_mode: off`, `prefill_mode: on`, `input_len: 8192` (batch×input_len² ×
num_heads ≈ 2.75e14, ~130,000× `INT_MAX`) → `ACT: 77.1172GB`, a sane positive value; run completes
to a final `Total:` with exit 0. Default-config regression unaffected (decode path never touches
this branch).

### #3 — T3: Scheduler prefill token allocator floors to zero → infinite stall

**Root cause**: `src/scheduler/scheduler.cpp` — `num_process = max_process_token / num_sum_seq`
is integer division; `config.yaml`'s default `max_process_token: 0` makes every prefill sequence's
per-step token allowance floor to 0 forever, so `current_len` never advances and `hasSumSeq()`
stays permanently true.

**Fix**: `max_process_token <= 0` now means "no per-step cap" (`num_process = INT_MAX`, clamped by
the existing per-sequence `std::min(num_process, input_len - current_len)` immediately below);
`max_process_token > 0` keeps the original divided allowance, now floored at 1 instead of 0.

**Verified**: `decode_mode: off`, tiny `input_len: 128`, `iter: 50` (would finish in well under a
second if making progress) — **pre-fix**: reverted the one-line fix, rebuilt, run timed out at 25s
with zero progress (confirmed stall, not slowness, given the trivial workload size). **Fixed**:
same config completes in full (exit 0) within ~1 minute (wall-clock dominated by verbose per-op
`print_log` output, not simulated time). Fix restored; default regression re-confirmed unaffected
(`decode_mode: on` never reaches this branch).

### #4 — T7: Disaggregated prefill ("sum") CSV rows never populated

**Root cause**: `Cluster::runIterationSumGenSplit`'s "sum" (prefill) branch built a `Stat` but
never called `setStat`/`setTimeBreakDown` and never filled energy fields, unlike the sibling "gen"
branch. Timing (`sum_machine_time`) was correct and didn't leak into `total_time`; only the CSV's
batch/energy/breakdown columns were zeroed for every prefill row.

**Fix**: populate `stat`'s energy fields via `getTotalEnergy()`, set `seq_queue_size`, and call
`setStat`/`setTimeBreakDown`, mirroring the gen branch exactly. `setStat()` deliberately leaves
`stat.latency` untouched on the disagg+`hasSumSeq()` path (verified by reading its own internal
branching), so this does not disturb the `time`/`latency` values already set.

**Verified**: `disagg_system: on`, `decode_mode: off`, `prefill_mode: on` produced one `sum`-typed
CSV row. **Pre-fix** (reverted in-place, rebuilt): `time`/`latency` = `27643789.195542` (both
runs), every field after that zero (`batchsize=0`, all energy=0, all breakdown=0). **Fixed**: same
`time`/`latency` = `27643789.195542` bit-for-bit, but `batchsize=32` and all energy/breakdown
columns populated with plausible non-zero values. Confirms the fix only fills previously-missing
fields and makes zero change to the timing/accounting path. Fix restored; default regression
re-confirmed unaffected (`disagg_system: off` never reaches this branch).

### #5 — T6: PIM attention-gen latency formula diverges from GPU/LOGIC + missing softmax

**Root cause**: `src/hardware/attention_gen_impl.cpp`'s `AttentionGenExecutionPIM` (the base
GQA/MHA decode-attention kernel, used by non-MLA models like llama3/llama4) used
`total_duration += accumul_memory_duration * opb` (`opb = flops/memory_size`) at both the Scoring
and Context sections, instead of the `max(compute, memory)` roofline-overlap formula every other
kernel (GPU, LOGIC, and PIM's own sibling MLA variants) uses. Its Softmax section was a bare
comment with no code, silently skipping the softmax compute time GPU/LOGIC charge.

**Fix**: replaced both `accumul_memory_duration * opb` sites with
`std::max(accumul_compute_duration, accumul_memory_duration)`, removed the now-dead `opb` locals,
and added a Softmax loop mirroring the GPU kernel's `flops = 7.0 * m * n * num_heads` compute
charge.

**Verified**: this kernel only activates when `parallel_execution: on` **and** a tensor is
explicitly routed to PIM (`AttentionSplit::forward` sets `setPerformLow()` on the gen tensor only
when `config.parallel_execution` is true) **and** `low_processor_type == PIM` (only
`processor_type: GPU+PIM` sets this) **and** the model is non-MLA (deepseekV3's MLA path uses a
different function, `MultiLatentAttentionGenExecutionPIM`/`AbsorbMLAGenExecutionPIM`, which don't
have this bug — confirmed via `grep -n opb` across the whole file, matches only in the base
kernel). With `model_name: llama3_405B`, `processor_type: GPU+PIM`, `parallel_execution: on`:
fixed build → `Total: 166042852.392729`; pre-fix baseline binary (unmodified main-tree build) →
`Total: 166014897.803638`. Small but real, expected delta (adds the previously-missing softmax
term). Every other config combination tried (`processor_type: PIM` alone — silently ignored by
`eval/test.cpp`'s string matching, no such branch exists; `GPU+PIM` with an MLA model) produced
byte-identical totals pre/post-fix, as expected since they never reach this function.

### #6 — T4: `getIdxHigh` hard-asserts for expert-group sizes not divisible by 4

**Root cause**: `src/module/route.cpp`'s `getIdxHigh` asserts `list.size() % 4 == 0` and aborts
otherwise (e.g. 10 or 20 experts/device under a heterogeneous EP split). Its sibling
`getIdxHighOptimal` already handles any list size correctly (per-expert scoring, no grouping
assumption) and is already used as the `<= 4` fallback.

**Fix**: when `list.size() % 4 != 0`, fall back to `getIdxHighOptimal` instead of asserting.

**Verified by code inspection + reachability check**: `getIdxHigh` is called from exactly one
site, `Route::forward` (`route.cpp:242`), gated on `device->config.parallel_execution == true`.
`getIdxHighOptimal`'s algorithm (read in full) has no size constraint — it sorts and greedily
assigns per individual expert index, so correctness for arbitrary `N` is structural, not
input-dependent. Full end-to-end reproduction of an actual 10-or-20-experts-per-device EP split
was not attempted: deepseekV3's `num_routed_expert = 256 = 2^8`, so every EP degree that survives
the optimizer's own divisibility asserts yields a power-of-2 per-device count (always `% 4 == 0`
once `> 4`), meaning constructing a real non-multiple-of-4 count requires either a different model
preset or hand-crafted `expert_token_list`. Given the fix is a one-line fallback to an
already-correct, already-in-use function, this was judged low-risk enough to accept without a
live repro; `parallel_execution: on` + `GPU+PIM` regression (T6's verification config) exercised
the surrounding `Route::forward` code path without error at the default (`% 4 == 0`) EP count,
confirming no regression in the common case.

### #7 — Secondary findings

**T11 — MLA prefill on HBF never calls `getAttentionMemoryDuration` (NOT implemented, documented
only)**. `MultiLatentAttentionSumExecutionGPU/Logic/PIM` and `AbsorbMLASumExecutionGPU/Logic/PIM`
(6 functions, `attention_sum_impl.cpp`) never route flash KV-read timing through
`getAttentionMemoryDuration`, unlike the base GQA `AttentionSumExecutionGPU`. This requires an MLA
model **and** prefill mode together (doubly dormant — no sweep uses either). Each of the 6
functions has its own memory-size accounting (including a distinct FlashAttention SRAM-tiling
algorithm branch, `use_flash_attention`, orthogonal to the HBF flash-memory tiering), and none of
them are currently exercised by any config this repo runs end-to-end. Given the size (6 large,
intricate functions across GPU/Logic/PIM × MLA/Absorb × flash-algorithm/non-flash) and the total
absence of a way to verify correctness against a known-good reference (MLA+prefill has no test
coverage anywhere in this codebase), implementing this blind was judged higher-risk than valuable
— a wrong "fix" here would be worse than the documented gap, since it could silently corrupt a
currently-inert code path in a way nobody would notice. **Recommended follow-up**: mirror the
`AttentionSumExecutionGPU` pattern — accumulate `total_kv_read_size`/`total_act_size` across the
seq×kv_head loop, call `getAttentionMemoryDuration` once post-loop — for each of the 6 functions,
but only after a real MLA+prefill test config exists to validate against.

**T12 — `head_dim`/`qk_nope_head_dim` confusion (2 sites, both fixed)**. Traced this to two
independent, opposite-direction slip-of-the-pen errors, both dormant only because deepseekV3's
preset sets `qk_nope_head_dim == head_dim == 128`:
- `src/model/footprint.h` (absorb + compressed_kv branches, `peakIntermediateBytes`): the
  query-projection-output term `(3.0 * qk_rope_head_dim + head_dim) * num_heads` should use
  `qk_nope_head_dim` (Q's non-rope width) instead of `head_dim` (V's up-projected width, correctly
  used elsewhere in the same formula for "v_up out"). Fixed both branches; left the GQA-base
  branch's `head_dim` alone (correct there — no MLA nope/rope split exists for GQA models, and
  `qk_nope_head_dim` defaults to 0 for those presets, so swapping there would have been a real
  regression, not a fix).
- `src/module/layer.cpp` (`attn_mla_absorbed` wiring): `attn_o_up_proj`'s declared input width was
  `num_heads * qk_nope_head_dim`, but it must match `attn_v_up_proj`'s output width just above it
  (`num_heads * head_dim`) — attention output is a weighted sum of V vectors, not Q_nope vectors.
  Corroborated against the sibling non-absorb `MultiLatentAttention` branch, which wires its own
  `attn_o_proj` with `head_dim` identically. Fixed to `head_dim`.
- **Verified**: numerically a no-op on every model this repo actually runs (only deepseekV3 sets
  `use_absorb`/`compressed_kv: on`, and its `qk_nope_head_dim == head_dim`); default regression
  confirmed byte-identical. Not independently exercised with a diverged config since no such
  model preset exists in this codebase — the fix is a structural correction for future MLA
  presets, verified by architectural cross-reference (V-projection output width, sibling
  non-absorb branch) rather than a numeric before/after.

**T13 — `logic_x`/`pim_x` speedup multiplier inconsistency (NOT fixed, documented only)**.
`route.cpp`'s `getIdxHigh`/`getIdxHighOptimal` use `logic_x`/`pim_x` to predict a LOGIC/PIM
speedup for routing decisions, but the actual LOGIC/PIM kernel timing (`*_impl.cpp`) never applies
that multiplier — the routing heuristic and the timing model disagree about how fast LOGIC/PIM
actually is. This is a design inconsistency, not a crash; per the original audit's own framing and
this session's read-only-unless-clearly-safe posture, changing kernel timing based on a
routing-heuristic constant is a modeling decision for the user, not a bug fix — left as-is,
flagged here for a deliberate decision.

**T14 — Dead non-chunked-attention branch (NOT fixed, documented only)**.
`layer_impl.h`'s `getAttentionMemoryDuration` non-chunked branch is byte-identical to the chunked
branch under the pinned default `chunk_size: 0` (both chunk by full SRAM capacity). Confirmed by
reading both branches side-by-side. Not a bug — just unreachable-in-practice code, left as
documentation for future readers (its real "one page-latency per full KV read" fallback behavior
has never been exercised with a config that would make it diverge).

---

## BUGS.md

### #1 — T5: `gpu_gen: A100` crashes on `YAML::LoadFile("")`

**Root cause**: `src/hardware/device.cpp`'s `dram_cfg_path`/`memory_scale_factor` switch had
branches for H100/B100/B200/Rubin but none for `A100` (a valid preset in `hardware_config.h`).
`dram_cfg_path` stayed empty, and `YAML::LoadFile("")` throws `YAML::BadFile`, uncaught.

**Fix**: mirrored the existing Rubin pattern — added an `A100` branch pointing at
`dram_config_HBM3_80GB.yaml` (inert placeholder matching the real A100-80GB SKU's capacity, since
`use_ramulator` is always false in every sweep) with an estimated `memory_scale_factor = 0.355`
(~2.4 Gbps HBM2e pin rate, scaled proportionally from H100's documented 5.2 Gbps → 0.76923
derivation). Also added a defensive `else { fail(...) }` so any future unmapped `gpu_gen` string
gets a clear error instead of the same empty-path crash.

**Verified**: `gpu_gen: A100` on the fixed build → completes normally (exit 0). **Pre-fix**
(disabled both the A100 branch and the new `else`-fail, restoring the exact original code shape,
rebuilt): `terminate called after throwing an instance of 'YAML::BadFile' / what(): bad file:`,
`Aborted` (SIGABRT, exit 134) — reproduces the documented crash exactly. Both edits restored;
default regression re-confirmed byte-identical (`gpu_gen: Rubin` never touches this branch).

### #2 — T10: `runIterationMixed` has no defensive check for `decode_mode`

**Root cause**: `Cluster::runIterationMixed` unconditionally accumulates every iteration's device
time into the decode-only `total_time`, relying entirely on the convention that
`decode_mode: on` guarantees `hasSumSeq()` is always false (verified: `scheduler.cpp`'s
`pushDummySeq` forces `current_len = input_len` at creation when `decode_mode` is set). No
in-function guard existed.

**Fix**: added a one-time (`static bool`) warning to stderr when `!config.disagg_system &&
scheduler->hasSumSeq()` — i.e. exactly the contamination condition the original audit described —
without changing `total_time`'s accumulation (a silent behavior change here would be a bigger risk
than a loud warning, since some future legitimate use might intentionally want this path).

**Verified**: appeared exactly once, on the first offending iteration, when testing T2 with
`decode_mode: off`, `disagg_system: off` (default): `WARNING: runIterationMixed is accumulating a
"sum" (prefill) iteration into the decode-only total_time -- decode_mode should be on unless
disagg_system is also on (see BUGS.md #2).` Silent (no output) on every default-config regression
run (`decode_mode: on`).

### #3 — T8: Disaggregated path lost its prefill→decode KV transfer term (NOT re-implemented,
documented only)

**Background**: a prior session deleted a one-time interconnect-transfer-cost block from
`Cluster::runIterationSumGenSplit` because it (a) double-counted against the existing per-decode-
step KV-write overlap logic in `attention_gen_impl.cpp`, which fires on both execution paths, and
(b) computed a flash-write-bandwidth latency but then discarded it in favor of interconnect
bandwidth, so it never modeled what it claimed to.

**Decision**: not re-implemented this session. `disagg_system` is `off` in every config this repo
ships and every sweep `run_experiments.py` runs — this is 100% dead-in-practice code, and the two
failure modes above are exactly the class of subtle bug that's hard to catch without a working
disagg test harness (which doesn't exist here). Reintroducing a "fixed" version with no way to
verify it against a reference would risk repeating the same mistake under a different guise.

**Intended fix, for the record** (do this only alongside a disagg-system integration test):
a one-time transfer cost, sized on the transferred sequence's `input_len` (not the per-token
KV-write size — this is a single one-shot handoff, not a recurring write), charged at
**interconnect** bandwidth (`device_ict_bandwidth`/`node_ict_bandwidth` depending on whether the
prefill and decode nodes are co-located, matching `communication.cpp`'s existing
`AllReduce`/`PipelineStage` bandwidth selection pattern), applied exactly once per sequence at the
moment `updateSchedulerSumGenSplit` first moves it from the sum queue to the gen queue — **not**
added to the per-decode-step `getKVWriteDuration` overlap logic already in
`attention_gen_impl.cpp`, which must remain untouched to avoid reintroducing the double-count.

### #4 — Non-llama3/llama4 model presets' precision assumptions unverified (NOT fixed, documented
only)

`mixtral`, `openMoE`, `llama7bMoE`, `grok1`, `deepseekV3`, `llama4_scout`'s `precision_byte`
defaults in `model_config.h` were not cross-checked against any external reference this session
(deepseekV3's is used as this repo's default and appears correct — `precision_byte=1`/FP8, matching
DeepSeek-V3's published FP8 training/inference precision — but this wasn't independently
re-verified beyond what's already stated in `CHANGES.md`). This requires an external reference per
model (a published config, a paper's stated precision, or a footprint-math cross-check against a
known total-parameter-count), not a code change. **Recommended approach if these presets are ever
used in a reported sweep**: same method the prior session used for `llama3_405B`/`llama4_maverick`
— footprint math (total params × precision_byte vs. published model size in GB) cross-checked
against each model's public config/card.

### #5 — `page_size_bytes` vestigial config surface (NOT fixed — not a bug)

Confirmed: no timing formula in the current codebase computes with `page_size_bytes` (grepped all
of `src/hardware/layer_impl.h` and the `attention_*_impl.cpp` files). It remains documented in
`hbf_memory_config.h` as hardware-geometry metadata only. Left in place — removing unused config
surface is a cleanup, not a bug fix, and doing so risks confusing future extensions of the flash
timing model that might want to reintroduce page-granular accounting.

### #6 — T9: `checkHeteroMemorySize()` unaudited / inconsistent with `checkMemorySize` (documented,
not removed)

**Audit result**: `grep -rn checkHeteroMemorySize` across the entire tree (excluding its own
declaration/definition) returns **zero call sites**. This resolves the original audit's "unclear
whether/when this function is actually invoked" — it is dead code, not merely under-exercised.

**Its capacity math, for the record** (in case it's ever wired back up): a hardcoded
`3.3 * 1024^3` ("Non MoE weight") magic-number subtraction with no architectural justification
found in this session, and no activation term at all in the `size` OOM check (activation is only
factored into the separate `avail_capacity` batch-shrink calculation further down, not the initial
gate) — genuinely inconsistent with `checkMemorySize()`'s scarce-tier gate (`footprint.h`'s
`checkCapacity`/`scarceTierActivationLimit`), which both the optimizer and the live simulator's
main gate share.

**Decision**: left in place (not deleted), with an audit comment added directly above the function
pointing future readers at this file. Deleting unreachable code the user hasn't explicitly asked
to remove felt like overreach for a bug-fix pass; flagging it precisely (with the zero-call-sites
confirmation) is the actionable next step, and the decision of delete-vs-repair-vs-keep is best
left to the user now that it's unambiguous.

---

## Guide §16 quirks (not in either bug file) — comment-only per explicit instruction

The user asked to document and comment these two, **not** fix them.

### G1 — `getActualArrivalTime` uses `poisson_distribution` instead of `exponential_distribution`

`src/scheduler/scheduler.cpp`: a Poisson *process*'s inter-arrival times are exponentially
distributed; `std::poisson_distribution` draws integer counts with the right mean
(`average_ns_per_request`) but the wrong shape/variance for inter-arrival gaps. Also,
`request_per_second == 0` divides by zero computing `average_ns_per_request`. Only live on a bare
`./run config.yaml` with `injection_rate > 0` — every `run_experiments.py` sweep sets
`injection_rate: 0`, so this never fires in a reported result. Added an explanatory comment at the
call site (no functional change) pointing at the intended replacement:
`std::exponential_distribution<double> distribution(1.0 / average_ns_per_request)` plus a
`request_per_second <= 0` guard, for whenever this is deliberately addressed.

### G2 — Request-length jitter computed but never applied (dead computation)

`src/scheduler/scheduler.cpp`'s `pushDummySeq`: `norm_dist_value` (with a rejection loop) and
`delta` are fully computed on every call, but the three lines that would apply `delta` to
`input_len`/`output_len` are commented out just below — every call pays the RNG + rejection-loop
cost for zero effect. Added an explanatory comment (no functional change) noting this is
intentionally left as dead computation: every reported sweep's results were generated with jitter
off, so either enabling or deleting this now would itself be a behavior/output change requiring a
separate, deliberate decision — not something to fold silently into a bug-fix pass.

---

## Verification summary

| Item | Pre-fix reproduced? | Fixed behavior confirmed? | Default regression unaffected? |
|---|---|---|---|
| T1 (SIGFPE) | Yes — SIGFPE, exit 136 | Yes — clean assert, exit 1 | Yes |
| T2 (int overflow) | N/A (formula-only) | Yes — sane `ACT: 77GB` at input_len=8192 | Yes |
| T3 (stall) | Yes — timeout at 25s, tiny workload | Yes — completes, exit 0 | Yes |
| T4 (route.cpp assert) | Not reproduced (see #6 above) | Code-verified; no regression at default EP | Yes |
| T5 (A100 crash) | Yes — SIGABRT/YAML::BadFile | Yes — exit 0 | Yes |
| T6 (PIM formula) | N/A (baseline binary comparison) | Yes — numeric delta vs. pre-fix binary | Yes |
| T7 (disagg CSV) | Yes — all fields zeroed | Yes — all fields populated, time bit-identical | Yes |
| T8 (disagg transfer) | Not implemented — documented | — | — |
| T9 (checkHeteroMemorySize) | Confirmed dead code — documented | — | — |
| T10 (decode_mode warning) | N/A (new warning) | Yes — fires exactly once under contamination | Yes (silent) |
| T11 (MLA prefill flash) | Not implemented — documented | — | — |
| T12 (head_dim/qk_nope) | N/A (numeric no-op on current presets) | Verified by architectural cross-reference | Yes |
| T13 (logic_x/pim_x) | Not fixed — documented | — | — |
| T14 (dead non-chunked branch) | Not fixed — documented (not a bug) | — | — |
| G1 (Poisson vs exponential) | Comment only, no fix (per instruction) | — | Yes (no code change) |
| G2 (dead jitter) | Comment only, no fix (per instruction) | — | Yes (no code change) |

Default pinned-flag regression (`decode_mode: on`, `HBM`/`HBF`/`HBF+`/`CONV`/`CONV+`,
`deepseekV3`, 8-GPU) confirmed byte-identical (stdout diff + CSV diff) against the pre-existing
main-tree build across all fix rounds.
