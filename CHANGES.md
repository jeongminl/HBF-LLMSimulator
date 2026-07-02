# Changes — HBF-LLMSimulator fairness/correctness fixes and paper cross-reference

This session fixed a series of correctness/fairness bugs surfaced by an external audit
(`FAIRNESS_AUDIT.md`, independently vetted before acting on it) and by directly cross-
referencing the codebase against the source paper ("Exploring High-Bandwidth Flash for
Modern LLM Inference," Son et al.). It also corrected a model-precision assumption that closed
a large chunk of an observed batch-size discrepancy against the paper's own reported numbers.

## Fixes applied and verified

1. **Analytic search no longer rejects a batch on the latency estimate alone (audit F1).**
   `eval/test.cpp`'s `analytic_sweep_only` mode now emits a separate
   `ANALYTIC_CAP_FEASIBLE_AT_1` marker, distinguishing "capacity/SRAM genuinely infeasible"
   (no simulator run needed) from "capacity is fine, only the analytic latency estimate
   rejected it" (must still ask the real simulator). `run_experiments.py`'s
   `find_max_batch_size` now falls through to a real `verify(1)` call in the latter case
   instead of silently returning batch=0 without ever consulting the simulator. This was a
   direct violation of the "simulator is the sole SLO arbiter" principle (`INSTRUCTIONS.md`
   §6) that this same session had written and enforced everywhere else — an ironic regression
   in the very code implementing that principle.

2. **Analytic search direction fixed to match the approved design.** `eval/test.cpp`'s
   analytic sweep now computes the capacity/SRAM ceiling directly (exact and monotonic, no
   search needed) and binary-searches downward from it using the `max()`-model latency
   estimate, instead of exponentially doubling from B=1. Functionally equivalent result under
   the proven envelope-monotonicity argument, but matches the originally agreed design and is
   more efficient.

3. **`tp > num_kv_heads` crash risk fixed (F4).** `parallelism_optimizer.cpp`'s TP sweep guard
   previously allowed proposing `tp > num_kv_heads` configs (intending KV-head replication),
   but the simulator (`src/module/parallel.cpp`) hard-asserts `num_kv_heads % parallel_num ==
   0` and crashes otherwise. The guard now requires full divisibility unconditionally, capping
   TP at `num_kv_heads` (8 for both llama3_405B and llama4_maverick); DP/PP make up the rest of
   the GPU count. Previously this could crash 16-GPU sweeps (and any config where the
   optimizer's best candidate exceeded num_kv_heads), silently reported as "OOM/Crash" and
   under-reporting that GPU count for *all* memory presets equally (not a cross-config fairness
   issue per se, but a real correctness bug threatening the accuracy of high-GPU-count results).

4. **KV-write overlap now uses attention-only compute as the hiding budget (F5).**
   `parallelism_optimizer.cpp`'s unhidden-KV-write calculation previously divided *total*
   per-layer compute (attention + FFN) as the "hidden behind compute" budget; the actual
   simulator (`attention_gen_impl.cpp`) only hides the write behind *attention* kernel compute
   (FFN is a separate kernel). The optimizer's analytic model now derives an attention-only
   compute time from the already-tracked `attn_weights_params`, matching the simulator's basis
   and tightening the drift-harness comparison. Simulator-verified metrics were never affected
   (this only changes the optimizer's ranking/estimate), but this reconciles a real divergence.

5. **`chunk_size` unit sanity guard added (F7).** `eval/test.cpp` now rejects
   `0 < chunk_size < page_size_bytes` (4096) at config-parse time, since such a value is
   almost certainly a tokens-vs-bytes unit mistake (the field is byte-valued) rather than an
   intentional sub-page micro-chunk request, which would otherwise silently multiply the flash
   page-read-latency count by ~1000x.

6. **Activation no longer double-charged against flash capacity (F8).** `cluster.cpp`'s
   `checkMemorySize()` lumped capacity gate (`activation + weight + kv > memory_capacity`)
   previously applied to flash systems too, even though activation already has its own,
   correct scarce-tier gate (1 reserved HBM stack for HBF/CONV, 320MB logic SRAM for
   HBF+/CONV+) shared with the optimizer's `checkCapacity()`. The lumped gate now excludes
   activation whenever a scarce tier exists, matching the optimizer's model and removing the
   double-charge (magnitude was small — activation ≤320MB/36GB against a 3.6-4TB pool — but a
   real, fixable divergence).

7. **Redundant `Optimize()` re-derivation eliminated in verification.** `run_analytic_sweep`
   now parses the discovered `ANALYTIC_TP/PP/EP/DP` and threads them into the subsequent
   simulator-verification call via explicit `distribution.*` overrides
   (`optimize_parallelism: False`), skipping a redundant (cheap, but unnecessary) in-process
   re-derivation of the same deterministic config. Only applies to the analytically-known
   batch/boundary checks; fallback narrow searches (when the analytic estimate diverges from
   the simulator) still let the optimizer re-derive, since there's no pre-known config there.

8. **Metric 3 & 4 GPU-count scope now matches the paper's figures (P2).** The paper's
   performance-breakdown (Fig. 5) and SLO-sensitivity (Fig. 6) figures both cover only
   {4, 8, 16} GPUs — HBM4 can't load either evaluated model at 1-2 GPUs, and both figures
   normalize to HBM4. Metric 3 (breakdown) previously computed only at 8 GPUs (a single point,
   missing the paper's GPU-count-dependence story entirely); it now computes at {4, 8, 16}.
   Metric 4 (SLO sensitivity) previously swept the full {1,2,4,8,16} (including GPU counts
   where the HBM4 normalization baseline is infeasible); it now restricts to {4, 8, 16}.

9. **Rubin GPU compute preset added, matching the paper's cited reference hardware.** The
   paper states all five evaluated memory configs "share the same state-of-the-art GPU-core
   architecture," citing NVIDIA DGX Rubin NVL8 (an unreleased, next-generation part). The
   codebase previously defaulted to `gpu_gen: B100` (a real, current-generation, but different
   and slower part — no Rubin preset existed at all). Added a `Rubin` preset to
   `hardware_config.h` with `compute_peak_flops` estimated from the DGX Rubin NVL8 datasheet's
   published "FP8/FP6 training: 140 PFLOPS" aggregate figure (÷8 GPUs ÷2, since no direct FP16
   figure is published — clearly commented as an estimate, not a directly-sourced value).
   `config.yaml`'s default `gpu_gen` switched to `Rubin`. This required also fixing a crash the
   new preset exposed: `device.cpp`'s Ramulator2 DRAM-config-file lookup (used regardless of
   whether Ramulator2 is actually invoked) had no branch for the new preset name, throwing
   `YAML::BadFile`; it now reuses B100/B200's HBM3E config file as an inert placeholder (this
   lookup's contents don't affect simulated results, since `use_ramulator` is always false in
   every sweep this project runs). Confirmed empirically (and via first-principles reasoning
   about compute-bound batch scaling) that this GPU-speed difference was *not* the explanation
   for the batch-size discrepancy investigated below — see "Open items."

10. **Batch-size metrics now correctly report per-GPU, not total (P1b) — later corrected, see
    item 15.** The value the optimizer/simulator internally call "batch_size" is a TOTAL,
    global quantity across all data-parallel (DP) replicas (`scheduler.cpp`: `batch_size_per_dp
    = total_batch_size / dp_degree`; `parallelism_optimizer.cpp`: same pattern).
    `run_experiments.py`'s Throughput metric already correctly treated this as a total (dividing
    by GPU count to get a per-GPU rate), but the "Maximum Per-GPU Batch Size" metric (and the
    SLO-sensitivity batch rows) was reporting the same raw total *undivided* — a real bug, fixed
    at the time by reporting `max_batch / dp`. **This divisor was itself wrong** (dividing by DP
    replica count instead of total GPU count) and was corrected later in this same session —
    see item 15 for why, and for the evidence that this was the dominant cause of the "residual
    gap" noted as unresolved below.

11. **llama3_405B precision corrected: FP8 → BF16.** `model_config.h`'s `llama3_405B` preset
    previously set `precision_byte=1` (FP8, ~405GB total), an assumption from an earlier
    session that was never actually verified against the paper. Directly confirmed as wrong by
    the paper's own explicit text: "HBM4 requires at least four GPUs to serve both LLMs
    (no 1-/2-GPU segments in all HBM4 bars)" — this requires llama3_405B's real footprint to
    exceed 2×288GB=576GB, which FP8 (~405GB) does not satisfy (it fits comfortably under 2
    GPUs' worth of HBM4 capacity, and indeed the pre-fix simulator wrongly showed 2-GPU HBM4 as
    feasible for llama3_405B, contradicting the paper), while BF16 (~810GB, matching how a
    "Llama 3 405B" checkpoint is natively released, and matching the precision already used for
    llama4_maverick per its own explicit 746GB figure in the paper) is consistent. After the
    fix: 1-GPU and 2-GPU HBM4 are now correctly infeasible for llama3_405B (matching
    llama4_maverick's already-correct behavior and the paper); the SHORT/8-GPU max batch
    dropped from 3907 to 1555, narrowing the gap against the paper's reported 194 from ~20.1x
    to ~8.0x.

12. **MLA-absorb double-division bug fixed.** `attention.cpp`'s `AbsorbMLAGen`/`AbsorbMLASum`
    constructors (used only by DeepSeek-family/MLA-absorb models) received an already
    TP-sharded `num_kv_heads` from their caller (`parallel.cpp` passes
    `num_kv_heads/parallel_num`), but then divided by `parallel_num` a *second* time when
    sizing their `"attn_output"` activation tensor — for `num_kv_heads=8, TP=8` this computes
    `1/8=0` via integer truncation, a zero-sized activation tensor. Fixed by removing the
    redundant second division in both constructors (and the now-unused local `parallel_num`
    variable). Confined to the activation-output tensor, not the actual MLA KV cache (which is
    the separate, correctly-unsharded `latent_kv_cache`/`latent_pe_cache`). Dormant for
    llama3_405B/llama4_maverick (neither uses MLA-absorb); affects any future DeepSeek/MLA
    sweep.

13. **`FAIRNESS_AUDIT.md` corrected.** Two of the external audit's claims were independently
    re-verified and found factually incorrect (annotated in place in that file, not deleted, so
    a future reader sees both the original claim and the correction):
    - **F6** ("disagg path lost its only KV-write critical-path term"): refuted. Both
      execution-loop implementations (`runIterationMixed`, `runIterationSumGenSplit`) call the
      identical kernel-dispatch chain that contains the correct per-step KV-write overlap
      logic — the deleted block was actually a *different*, disagg-specific one-time
      prefill→decode KV *transfer* term (and never modeled a flash write even before deletion,
      contrary to the audit's framing).
    - **F7**'s supporting claim ("`use_chunked_attention` defaults false and is never set"):
      refuted. `module/attention.cpp` sets it to `true` unconditionally at 5 call sites,
      predating this session.

14. **HBF+/CONV+ scarce-tier (SRAM) capacity gate now uses PEAK intermediate-data footprint,
    not a sum.** Scoped verification (below) found HBF+ producing a *lower* total max batch
    than plain HBF in every case tested — a full inversion of the paper, which claims HBF+
    should be larger. Root cause: both `checkCapacity()` (`src/model/footprint.h`) and
    `Cluster::checkMemorySize()` (`src/hardware/cluster.cpp`) gated the 320-MB logic-SRAM scarce
    tier (HBF+/CONV+) and the 36-GB HBM-stack scarce tier (HBF/CONV) against `activation_size`,
    a **sum** of every intermediate tensor a decode-step layer touches (input, q/k/v
    projections, attention-score matrix, context, out-proj, FFN) as if all simultaneously
    resident. The paper is explicit this is wrong: *"much of the intermediate data has a short
    lifetime, which enables quick release of SRAM-buffer space"* and *"a larger batch size ...
    linearly increases the **peak size** of intermediate data"* — tensors are produced and
    freed sequentially (attention phase, then FFN phase), so the true resident set at any
    instant is `max(attention-phase, FFN-phase)`, never their sum. Worse, the summed formula's
    dominant term was the sequence-length-scaled Q·Kᵀ attention-score matrix
    (`batch·2·total_len·num_heads/tp`) — per `INSTRUCTIONS.md`'s Read Prefetching policy, that
    data is chunked through a *separate* 3.13-MB/stack double-buffer and is never resident in
    the 320-MB intermediate-data pool at all. Because this term scaled with sequence length, the
    320-MB gate bit hardest at *long* context — backwards from both the paper and
    `INSTRUCTIONS.md`'s original "particularly in short-context workloads" framing — and because
    HBF's 36-GB scarce tier never binds regardless, only HBF+/CONV+ were penalized, producing
    the inversion. Fixed by adding a single shared `peakIntermediateBytes()` helper
    (`src/model/footprint.h`), used identically by the optimizer's `checkCapacity` call and the
    simulator's Part-C scarce-tier gate, that: (a) excludes the streamed attention-score/
    decompressed-KV terms entirely, (b) takes `max(attention-phase, FFN-phase)` instead of a
    sum, and (c) adds the dense-FFN transient term that the old formula omitted entirely for
    non-MoE layers (its `ffn_act` term was gated on `num_routed_expert > 0`, silently zeroing
    for dense models like llama3_405B). `INSTRUCTIONS.md`'s Section 2 "Intermediate Data"
    paragraph was rewritten to state the peak/short-lifetime model explicitly, per the paper
    (trusted over the prior generic "map all intermediate data" wording where they conflicted).
    Effect: HBF+ total batch at MID/8-GPU roughly doubled for both models (llama3: 756→1444,
    llama4: 2336→4562), flipping the HBF+-vs-HBF total-batch inversion the right way round.

15. **Per-GPU batch-size divisor corrected: total GPU count, not DP replica count (supersedes
    item 10/"P1b").** After fix 14, HBF+'s *total* batch exceeded HBF's, as the paper requires —
    but the *reported* "per-GPU" figure (item 10's `max_batch / dp`) still showed HBF+ *below*
    HBF, because HBF+'s optimal config now sometimes uses `dp=2` (two independent 4-GPU
    replicas) where HBF stays at `dp=1`. Dividing by `dp` penalizes a config for splitting into
    *more* independent, hardware-efficient replicas — backwards, since a DP replica still
    consumes real GPU hardware whether or not it's counted in the denominator.
    `INSTRUCTIONS.md`'s own explicit definition of the neighboring TPS metric — "(Total Output
    Tokens) / (Total Simulation Time) / (**Number of GPUs**)" — already divides by total GPU
    count, not DP, and `run_experiments.py`'s TPS formula (`max_b / (tpot * gpu)`) already
    implemented it that way; only the batch-size divisor (4 call sites in `run_experiments.py`)
    used `dp` instead, an internal inconsistency within the same file. Corrected all 4 sites to
    `max_batch / gpu_count`. This is a reversal of a decision defended earlier in this same
    session (originally argued that TP/PP GPUs "collaboratively serving the same sequences"
    justified not dividing by them) — that reasoning describes a real quantity (batch size per
    *replica*), but not the quantity `INSTRUCTIONS.md`/the paper mean by "per-GPU": a
    hardware-normalized metric for comparing across GPU counts, which must divide by all GPUs
    consumed regardless of whether they're organized via TP or DP. Confirmed by recomputing
    already-completed simulator runs with the corrected divisor (no re-run needed, since the
    totals aren't affected by this reporting-only change) — see "Residual gap resolved" below.

16. **Llama-4 interleaved local/global ("iRoPE") attention now modeled — root cause of
    `llama4_maverick`-specific LONG-context mismatches.** After the full paper-replication sweep
    (`run_experiments.py`, Figs 3-7), `llama4_maverick`'s HBM4/8-GPU/LONG per-GPU batch anchor
    came out as 10.0 vs. the paper's 31 (a 68% undershoot), while `llama3_405B`'s equivalent
    anchor matched almost exactly (194.8 vs. 194) — the mismatch was specific to `llama4_maverick`
    at LONG context (input+output ≈ 104,600 tokens). Root cause: the simulator modeled every
    attention layer as full/global (attending over the entire context), for every model
    uniformly. Real Llama 4 uses Meta's "iRoPE" architecture — only every 4th layer is
    full/global ("NoPE"); the other 3 use a fixed 8192-token local attention window — confirmed
    independently against Meta's actual released `Llama-4-Maverick-17B-128E-Instruct` `config.json`
    (`attention_chunk_size=8192` explicit; `no_rope_layer_interval` defaults to 4 and isn't
    overridden) rather than relying on recalled model-architecture facts. `llama3_405B` is a
    genuinely dense, full-global-attention model, so the pre-existing formula was already correct
    for it — exactly why it matched and llama4 didn't.
    - Added `attn_chunk_size`/`attn_global_interval` fields to `ModelConfig`
      (`src/model/model_config.h`), defaulting to 0/1 (full global attention — byte-identical for
      every existing preset), plus shared helpers `isGlobalAttentionLayer()`/`effectiveKvLen()`/
      `effectiveKvLenSumAllLayers()`. Set `attn_chunk_size=8192, attn_global_interval=4` on
      `llama4_maverick` and `llama4_scout` (Scout isn't one of the paper's own evaluated models,
      so treat its exact values as lower-confidence than Maverick's independently-verified ones).
    - Fixed the analytic KV-cache-per-GPU formula (`parallelism_optimizer.cpp`'s
      `kv_cache_per_gpu`, feeding `kv_read_size`/`kv_read_time` — the batch-size search's actual
      arbiter) to sum each layer's *effective* KV length instead of `layers_per_stage *
      sequence_length` uniformly.
    - Fixed the SAME uniform-full-context assumption in the live simulator's KV-cache TENSOR
      ALLOCATION (`module/attention.cpp`'s `SelfAttentionGen` constructor allocates `k_cache`/
      `v_cache` shaped `{context_len, head_dim}` per layer at construction time — this is what
      actually drives `cluster.cpp::checkMemorySize`'s ground-truth capacity gate via
      `Module::get_size()`'s "cache"-tagged tensor sum, NOT the separate `kv_cache_size_per_seq`
      hand-rolled formula in `checkMemorySize`'s dormant `mem_cap_limit` branch, which is
      unreachable for any sweep since `run_experiments.py` always sets
      `exit_out_of_memory=True`). Threaded a `layer_idx` parameter through the module
      construction chain (`llm.cpp`'s per-layer loop → `decoder.{cpp,h}` → `layer.{cpp,h}`'s
      `Attention` constructor, which computes the effective window and passes it down through
      `parallel.{cpp,h}`'s `SelfAttentionParallel` — only to its `SelfAttentionGen` sub-module,
      not the prefill-path `AttentionSplit`/`SelfAttentionSum`/`AttentionMerge`, deferred per
      item 2 in `BUGS.md` since `decode_mode: on` means prefill is never exercised by any current
      sweep). `SelfAttentionGen` now also stores this value and sets a new
      `LayerInfo::local_attention_window` field (`hardware/base.h`) at `forward()` time.
    - Fixed the corresponding RUNTIME KV-read latency computation
      (`hardware/attention_gen_impl.cpp`'s `AttentionGenExecutionGPU`, the GQA decode path both
      llama3/llama4 use) to cap `n`/`k` at `layer_info.local_attention_window` in the Scoring,
      Softmax, and Context loops — only the GPU variant was touched (llama4's actual decode
      path); the Logic/PIM GQA variants and the MLA variants (deepseek, always full-global) were
      left unchanged.
    - Verified: llama4_maverick's HBM4/8-GPU/LONG anchor moved from 10.0 to **33.0** (paper: 31,
      ~6% off — down from a 68% miss). SHORT (context=2033) and MID (context=6399) anchors are
      **exactly unchanged** (514.88, 164.75 — both below the 8192-token chunk size, so the fix is
      correctly a no-op there, confirming it's precisely targeted at the LONG-context regime).
      `llama3_405B`'s HBM4/8-GPU SHORT (194.8→194.8, total 1558) and LONG (3.8→3.75) anchors are
      byte-identical before/after, confirming zero behavior change for non-Llama-4 models.
    - **Two smaller, independently-confirmed bugs fixed in the same pass**
      (`parallelism_optimizer.cpp`'s MoE-layer-count block, lines ~156-175): (a) `first_k_dense`
      handling was gated to `model_name == "deepseekV3"` only, silently ignoring it for any other
      model with `first_k_dense>0`; (b) the `expert_freq>1` branch rounded an evenly-averaged
      per-stage MoE-layer count instead of computing each stage exactly. Both replaced with a
      single exact per-layer count using `isMoELayer()` (moved to the shared
      `model/model_config.h` so `llm.cpp`'s module-construction path and the optimizer can never
      disagree on which layers are MoE), taking the heaviest stage's count (matches the
      pipeline-bottleneck/capacity-gate semantics the rest of the function already assumes). Note:
      this does NOT noticeably move `llama4_maverick`'s numbers by itself (the optimizer evaluates
      one representative/heaviest stage, so `round(1.5)=2` already happened to equal the true
      heaviest stage's count at maverick's pp=16) — the interleaved-attention fix above accounts
      for essentially all of the improvement. This bug fix DOES change `deepseekV3`'s
      weight/capacity estimate at `pp>1` (the old code's blanket
      `dense_in_stage=min(first_k_dense,layers_per_stage)` under-counted MoE weight for any stage
      that doesn't include the dense prefix — a genuine, separate latent bug for `deepseekV3`,
      not a regression; `deepseekV3` isn't part of the paper's own evaluated models so this
      wasn't previously exercised by any paper-comparison check).
    - **Two of the four originally-reported llama4-specific mismatches are NOT explained by this
      fix** — see `INCONSISTENT_WITH_PAPER.md` for the still-open residual (the 4-GPU/16-GPU
      HBF+-vs-HBM4 TPS claims at LONG context actually moved further from the paper's claimed
      values, apparently because HBM4 benefits from the corrected KV/capacity ceiling at least as
      much as HBF+ does at this workload, since HBM4 was also artificially capacity-constrained
      before this fix).

## Residual batch-size gap: resolved (was open; see items 14 and 15 above)

Fixes 14 and 15 together resolve what was tracked as an unexplained "~8-9x batch-size gap"
against the paper. Recomputed from already-completed simulator runs with the corrected total/
GPU-count divisor:

| Check | Ours (corrected) | Paper | Verdict |
|---|---|---|---|
| HBM4 8-GPU/SHORT, llama3_405B | 194.4 | 194 | match (ratio 1.002) |
| HBM4 8-GPU/SHORT, llama4_maverick | 511.6 | 460 | close (ratio 1.11) |
| HBF+/HBF ratio, MID/8-GPU, avg | +25.6% | "+24% on average" | match |
| llama3 HBF/HBM4, SHORT/8-GPU | 1.52x | 1.3–4.5x | in range |
| llama3 HBF+/HBM4, SHORT/8-GPU | 1.93x | 1.7–5.3x | in range |
| llama4 HBF/HBM4, SHORT/8-GPU | 2.00x | 1.3–4.5x | in range |
| llama4 HBF+/HBM4, SHORT/8-GPU | 2.05x | 1.7–5.3x | in range |
| 1-GPU HBF/HBF+ > 8-GPU HBM4 (4 cases) | 1 holds, 2 close (-14%/-17%), 1 clear miss (-61%, llama3 HBF) | "in most cases" | partial — see below |

The 194.4-vs-194 match in particular is essentially exact and was obtained purely from the
divisor correction (the underlying total, 1555, was unchanged). This was previously investigated
at length and several explanations were **ruled out** before the actual cause (the divisor) was
found:
- GPU compute-preset/speed mismatch (a faster GPU would increase our numbers, not close the gap
  — ruled out by direction-of-effect argument and the DGX Rubin NVL8 datasheet).
- A TP/DP-sharding bug in KV-cache-per-GPU accounting (checked twice, independently; correct and
  consistent between the optimizer and live simulator).
- "The optimizer doesn't account for communication overhead" (it does; confirmed via code and
  real measured CSV data).
- Confirmed the search was capacity-gated, not latency-gated, at the relevant operating points.

17. **Flash page-read latency was charged once per SRAM-staging chunk instead of being hidden by
    double-buffering (root cause of `INCONSISTENT_WITH_PAPER.md` items 1 & 2).**
    `getAttentionMemoryDuration` (`src/hardware/layer_impl.h`) charged HBF/HBF+ KV reads as
    `kv_read_size/bandwidth + num_chunks * flash_page_read_latency_ns`, i.e. one full page-read
    latency for *every* SRAM-staging chunk (`num_chunks = ceil(kv_read_size / 25.04MB)` for an
    8-stack config). HBM4 (`num_flash_stacks==0`) never enters this branch and pays zero latency,
    even though HBF+ has *identical* 12.8 TB/s nominal read bandwidth (confirmed vs. the paper's
    Table I) — so this term alone inflated HBF+'s effective KV-read time by ~49% over HBM4's,
    which is why HBF+ could never beat HBM4 on decode TPS despite equal bandwidth. The paper's own
    text says each stack "prefetch[es] and double-buffer[s] read data, thereby minimizing the
    performance impact of µs-scale page-read latency": double-buffering means chunk *N+1*'s
    page-read latency overlaps chunk *N*'s transfer and is hidden — for every chunk except the
    first, which has nothing to overlap with (pipeline fill) and always exposes one full page
    latency. Fixed to `exposed = page_latency + (num_chunks-1) * max(0, page_latency -
    chunk_transfer_time)`: exactly one page latency remains (not eliminated — an instantaneous
    first access is unphysical), and the `max(0, …)` term degrades gracefully back toward the
    old fully-exposed behavior if a chunk is ever configured small enough that its transfer time
    drops below the page latency. Mirrored identically in
    `parallelism_optimizer.cpp`'s `kv_read_time` (the analytic candidate proposer) so the two
    formulas stay in lock-step, matching this codebase's established pattern for shared formulas
    (e.g. `peakIntermediateBytes`). Also reworded `INSTRUCTIONS.md`'s Read-Prefetching paragraph,
    which previously said to "pay the page-read latency only once per chunk" — a literal
    description of the bug — to state the corrected double-buffer model instead.
    **Scope/safety:** only executes when `config.use_hbf && num_flash_stacks > 0`; verified HBM4
    and every dense/llama3-HBM4 anchor is byte-identical before/after (tpot matched to 15 decimal
    places). **Verified effect:** llama4_maverick LONG-workload HBF+ per-GPU TPS rose ~34-36%
    uniformly across 4/8/16 GPUs (e.g. 8-GPU: 130.6→176.1 batch/GPU, 1306.6→1761.7 tok/s/GPU).
    Item 1 ("4-GPU HBF+ TPS 15% higher than 8-GPU HBM4") moved from 0.53x to **0.72x** measured
    (paper: 1.15x) — real progress, not fully closed. Item 2 ("HBF+ always outperforms HBM4" at
    16 GPUs) moved from 0.75x to **1.01x** measured at the 0.1s SLO — **now holds** at that SLO
    point (other SLOs not reverified this pass). llama3_405B's 1-GPU LONG anchors (item 3) also
    improved via the same general fix (not llama4-specific): HBF 1→2 batch (vs. 8-GPU HBM4's 3.75,
    still a miss), HBF+ 3→5 batch (now *exceeds* 3.75 — this specific combination now holds,
    though the broader "1-GPU HBF/HBF+ > 8-GPU HBM4 in most cases" claim was never about a single
    data point and the underlying reason llama3 struggles at 1 GPU, below, is unaffected by this
    fix).

18. **MoE "experts per device" activation term divided by total device count instead of
    devices-per-pipeline-stage (found investigating `INCONSISTENT_WITH_PAPER.md` item 4; does
    NOT explain item 4's gap — see that file).** `cluster.cpp`'s `checkMemorySize` and
    `parallelism_optimizer.cpp`'s activation-sizing code both computed
    `num_routed_expert * e_tp_dg / num_total_device`, dividing by the whole cluster's device
    count. The neighboring, already-correct WEIGHT term (`parallelism_optimizer.cpp:123`)
    computes the identical semantic quantity as `num_routed_expert / devices_per_stage`
    (`devices_per_stage = total_gpus/pp`, the devices actually sharing one pipeline stage's
    expert allotment) — not the whole cluster. For llama4_maverick's degenerate PP=8/EP=1
    8-GPU config (`devices_per_stage=1`), the old formula computed `128*1/8=16` "experts per
    device" instead of the correct `128*1/1=128`, an 8x undercount specific to that
    configuration. Fixed both call sites to divide by `devices_per_stage`, matching the weight
    term's pattern exactly. Feeds only `peakIntermediateBytes`'s MoE-FFN activation term, which
    scales with `expert_batch_size` (~4 tokens/expert at these batch sizes) — as expected, this
    moved HBM4's llama4 SHORT/MID anchors by 0 (514.875/164.75 batch-per-GPU, byte-identical
    before/after), confirming (per item 4's write-up) that this bug is real but does not explain
    that item's ~9-12% gap. **Verified against HBF+ (the 320-MB scarce SRAM tier this matters
    more for):** llama4 SHORT/8-GPU HBF+ (SRAM-bound) dropped from 1051.0 to 1024.5 batch/GPU
    (-2.5%) — the fix correctly *tightens* a previously-undercounted activation term against the
    small 320 MB tier, i.e. the old number was slightly too optimistic; this is the fix working
    as intended, not a regression. llama4 MID/8-GPU HBF+ (SLO-bound, not SRAM-bound at this point)
    rose from 570.25 to 793.25 batch/GPU (+39%), dominated by item 17's KV-read-latency fix
    (same mechanism as the LONG-workload gains above), confirming Fix B has no effect once the
    SRAM tier isn't the binding constraint.

**Still open, smaller residual:**
- llama4_maverick's HBM4/8-GPU/SHORT anchor is ~11% high (511.6 vs. paper's 460) — much smaller
  than the earlier ~8-9x gap, not yet root-caused. Possibly model-specific (MoE routing/expert
  weight footprint assumptions) rather than a shared mechanism with the batch-size fixes above.
- The 1-GPU HBF/HBF+ vs. 8-GPU HBM4 comparison (paper: "holds in most cases") now holds in 1 of
  4 cases and is close (within ~15-17%) in 2 more, but llama3_405B's HBF (non-plus) case is a
  clear miss (75 vs. 194.4, -61%). Not yet investigated further — worth a separate pass focused
  specifically on 1-GPU HBF (not HBF+) behavior for the dense model.
- **Open, untested hypothesis (unchanged from before):** whether degenerate
  one-device-per-pipeline-stage optimizer choices (e.g. llama4_maverick's PP=8/EP=1, which
  trivially zeroes every communication term by construction) are genuinely latency/
  capacity-optimal, or an artifact of how completely they eliminate communication cost in the
  ranking relative to a middle-ground split (e.g. PP=2/EP=4) that would trade some communication
  for a much smaller per-device expert-weight footprint.
- See `BUGS.md` for additional minor/dormant bugs and brittle code noticed but not fixed this
  session.
