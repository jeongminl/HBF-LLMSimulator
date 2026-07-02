# Known minor bugs and brittle code

Bugs and fragile spots noticed during this session's fairness-audit/paper cross-reference work
that were **not** fixed (either genuinely low-priority, currently dormant, or explicitly out of
scope). See `CHANGES.md` for what *was* fixed. Listed roughly in order of how likely each is to
actually bite someone.

## 1. `gpu_gen: A100` would crash

`src/hardware/device.cpp`'s `dram_cfg_path` / `memory_scale_factor` switch has branches for
`H100`, `B100`, `B200`, and `Rubin` (the last added this session) — but **no branch for
`A100`**, even though `A100` is a defined preset in `hardware_config.h` and a valid
`gpu_gen` string elsewhere in the codebase. Setting `gpu_gen: A100` in config would leave
`dram_cfg_path` empty and `YAML::LoadFile("")` would throw `YAML::BadFile` — the exact crash
class this session fixed for the newly-added `Rubin` preset. Latent and dormant: no current
sweep or config.yaml default uses A100. Fix would mirror the Rubin fix (add an A100 branch,
pointing at a suitable existing or new dram config file).

## 2. `runIterationMixed` has no defensive check for `decode_mode`

Decode-only correctness (i.e., never letting prefill compute time leak into the reported
decode-focused TPOT) currently depends entirely on `decode_mode: on` staying set by convention
in `config.yaml`. `Cluster::runIterationMixed` itself has no internal guard — it unconditionally
accumulates every iteration's device time into `total_time`, with no check for whether that
iteration happened to process prefill ("sum") sequences. This is safe today only because
`decode_mode: on` guarantees no sequence ever enters the prefill/"sum" bucket in the first
place (verified: `scheduler.cpp`'s `pushDummySeq` forces `current_len = input_len` at creation).
If `decode_mode` were ever turned off while using `runIterationMixed` (i.e., without also
setting `disagg_system: on`), prefill compute would silently contaminate the decode critical
path. A code comment documenting this dependency was added at the relevant line
(`cluster.cpp`, `runIterationMixed`'s `total_time += time`) this session; no structural fix was
made.

## 3. Disaggregated path (`disagg_system=on`) lost its prefill→decode KV transfer term

This session deleted a block in `cluster.cpp::runIterationSumGenSplit` (the `disagg_system=on`
execution path, currently unused — `config.yaml` always sets `disagg_system: off`) that modeled
a one-time interconnect transfer cost when a sequence moves from a prefill node to a decode
node. It was removed because (a) it double-counted against the correct, separate per-decode-step
KV-write overlap logic that already exists in `attention_gen_impl.cpp` and fires on *both*
execution paths, and (b) it computed a flash-write-bandwidth latency but then discarded it in
favor of interconnect bandwidth, so it never actually modeled a flash write despite appearing
to. Net effect: the disagg path's per-step KV-write penalty is intact (confirmed, both paths
share the same kernel), but the *disagg-specific one-time transfer* event is no longer modeled
at all. If `disagg_system` is ever enabled for real experiments, this one-time transfer term
should be reintroduced correctly — sized on the transferred sequence's `input_len`, using
interconnect (not flash) bandwidth, without double-counting against the existing per-step
write-overlap logic.

## 4. Non-llama3/llama4 model presets' precision assumptions are unverified

This session verified and corrected `llama3_405B`'s precision (FP8→BF16) and confirmed
`llama4_maverick`'s (already correct BF16) against the paper's explicit constraints. The other
model presets in `model_config.h` — `mixtral`, `openMoE`, `llama7bMoE`, `grok1`, `deepseekV3`,
`llama4_scout` — were not cross-checked against any external reference for their
`precision_byte` defaults. If any of these are ever used in a reported sweep, their precision
assumption should be verified the same way (footprint math + any available capacity-feasibility
constraint from a reference source), not just trusted as-is.

## 5. `page_size_bytes` is now dead/vestigial config surface

After this session's earlier fix to the non-chunked KV-read overflow branch (which now chunks
by SRAM staging capacity, matching the plane-parallel physical model, instead of charging one
page-read latency per literal 4KiB page), `page_size_bytes` is no longer read anywhere in the
timing formulas (`src/dram/hbf_memory_config.h` still defines and documents it as hardware
geometry, but nothing computes with it). Not a bug, just unused config surface worth knowing
about if extending the flash timing model further.

## 6. `checkHeteroMemorySize()` was not audited this session

Separate from `Cluster::checkMemorySize()` (which received the F8 fix this session),
`checkHeteroMemorySize()` in `cluster.cpp` has its own capacity-check logic, including a
hardcoded `3.3 * 1024^3` ("3.3 GB", labeled "Non MoE weight") magic-number subtraction, and does
not appear to fold activation size into its capacity comparison at all. It's unclear from this
session's investigation whether/when this function is actually invoked by the sweeps this
project runs (as opposed to `checkMemorySize()`). Flagged as unaudited and potentially
inconsistent with the F8 fix, not confirmed broken.

Paper-vs-simulator numeric/qualitative comparison items live in `PAPER_INCONSISTENCIES.md`
(still-open/explained-not-bugs) and `CHANGES.md` (resolved), not here.
