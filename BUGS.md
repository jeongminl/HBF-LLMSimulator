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

## 4. `run_flash_only.py` was not audited or fixed this session

This is an older, separate sweep driver with its own two-phase batch-size search, distinct from
(and not routed through) `run_experiments.py`'s `find_max_batch_size`, which received all of
this session's fixes (audit F1's capacity/latency separation, P1b's per-GPU division, etc.).
`run_flash_only.py` almost certainly still carries every one of those bugs. Should be updated to
share the fixed logic (or share `run_experiments.py`'s functions directly) or retired, whichever
is intended, before being relied on for any reported numbers.

## 5. Non-llama3/llama4 model presets' precision assumptions are unverified

This session verified and corrected `llama3_405B`'s precision (FP8→BF16) and confirmed
`llama4_maverick`'s (already correct BF16) against the paper's explicit constraints. The other
model presets in `model_config.h` — `mixtral`, `openMoE`, `llama7bMoE`, `grok1`, `deepseekV3`,
`llama4_scout` — were not cross-checked against any external reference for their
`precision_byte` defaults. If any of these are ever used in a reported sweep, their precision
assumption should be verified the same way (footprint math + any available capacity-feasibility
constraint from a reference source), not just trusted as-is.

## 6. `page_size_bytes` is now dead/vestigial config surface

After this session's earlier fix to the non-chunked KV-read overflow branch (which now chunks
by SRAM staging capacity, matching the plane-parallel physical model, instead of charging one
page-read latency per literal 4KiB page), `page_size_bytes` is no longer read anywhere in the
timing formulas (`src/dram/hbf_memory_config.h` still defines and documents it as hardware
geometry, but nothing computes with it). Not a bug, just unused config surface worth knowing
about if extending the flash timing model further.

## 7. `checkHeteroMemorySize()` was not audited this session

Separate from `Cluster::checkMemorySize()` (which received the F8 fix this session),
`checkHeteroMemorySize()` in `cluster.cpp` has its own capacity-check logic, including a
hardcoded `3.3 * 1024^3` ("3.3 GB", labeled "Non MoE weight") magic-number subtraction, and does
not appear to fold activation size into its capacity comparison at all. It's unclear from this
session's investigation whether/when this function is actually invoked by the sweeps this
project runs (as opposed to `checkMemorySize()`). Flagged as unaudited and potentially
inconsistent with the F8 fix, not confirmed broken.

## 8. Open question (not a confirmed bug): are degenerate one-device-per-stage parallelism choices actually optimal?

The optimizer's ranking can select configurations like llama4_maverick's 8-GPU choice
(PP=8, EP=1, TP=1 — one device per pipeline stage) that trivially drive every communication
term (TP all-reduce, EP scatter/gather) to exactly zero, simply because there's no other device
in the same stage to communicate with. This is a legitimate consequence of the ranking
correctly preferring lower communication cost — but it also means each such device must hold
100% of the routed experts' weight for its assigned layers, with zero expert-parallel sharding.
Not yet investigated: whether the weight-read/compute cost formula fully and correctly accounts
for holding all experts unsharded in this configuration, or whether it under-counts that cost
relative to a real alternative (e.g. PP=2/EP=4) that would incur communication but shard
expert weight much more thinly per device. If under-counted, this could bias the ranking toward
these degenerate configs more than is truly latency-optimal. Related to (but distinct from) the
now-resolved ~8-9x batch-size gap noted in `CHANGES.md` items 14-15.

**Partial finding (this session's llama4-mismatch investigation, see `CHANGES.md` item 16):**
routed-expert weight per GPU is invariant to the TP/PP/EP split (`parallelism_optimizer.cpp`'s EP
degree cancels out of the weight formula algebraically), so the degenerate PP=8/EP=1 choice does
NOT under-count *capacity* — it only affects the *latency ranking* (by zeroing the EP
scatter/gather and TP all-reduce communication terms). Still not confirmed whether the latency
ranking itself is biased toward this config more than is truly optimal.

**[LARGELY RESOLVED] A/B check executed (`PAPER_INCONSISTENCIES.md`'s U3):** forced both
PP=8/EP=1 (optimizer's actual choice) and PP=2/EP=4 (rejected alternative) at
llama4/HBM4/8-GPU/SHORT and compared real simulator-measured TPOT. PP=8/EP=1 genuinely wins
(~4.6% faster; at the literal anchor batch, PP=2/EP=4 doesn't even fit — OOM), with the win coming
entirely from PP=2/EP=4's confirmed-real extra non-expert weight cost (`layers_per_stage` 4× larger,
not reduced by EP), not from any under-counted communication term (both configs pay zero
communication, by construction). **The ranking is validated for this specific pair — not a bug.**
This narrows but doesn't fully close the general question for every operating point; see
`PAPER_INCONSISTENCIES.md` for full detail.

## 9. Paper-comparison items moved

All paper-vs-simulator numeric/qualitative comparison items (formerly items 9-10 here, plus the
content of `INCONSISTENT_WITH_PAPER.md` and `INCONSISTENCY_POSSIBLE_FIXES.md`) have been merged
into a single canonical tracker — see **`PAPER_INCONSISTENCIES.md`** for current status
(resolved/still-open/explained-not-bugs) of every such item.
