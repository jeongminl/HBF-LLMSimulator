# Changes — HBF-LLMSimulator fairness/correctness fixes and paper cross-reference

This session fixed a series of correctness/fairness bugs surfaced by an external audit
(a cross-configuration fairness review, independently vetted before acting on it; its findings
are tagged "audit F1"-"F8" throughout this doc) and by directly cross-referencing the codebase
against the source paper ("Exploring High-Bandwidth Flash for Modern LLM Inference," Son et al.,
IEEE CAL 2026 — the ground-truth spec for this simulator; see the PDF in this repo). It also
corrected a model-precision assumption that closed a large chunk of an observed batch-size
discrepancy against the paper's own reported numbers.

## Fixes applied and verified

1. **Analytic search no longer rejects a batch on the latency estimate alone (audit F1).**
   `eval/test.cpp`'s `analytic_sweep_only` mode now emits a separate
   `ANALYTIC_CAP_FEASIBLE_AT_1` marker, distinguishing "capacity/SRAM genuinely infeasible"
   (no simulator run needed) from "capacity is fine, only the analytic latency estimate
   rejected it" (must still ask the real simulator). `run_experiments.py`'s
   `find_max_batch_size` now falls through to a real `verify(1)` call in the latter case
   instead of silently returning batch=0 without ever consulting the simulator. This was a
   direct violation of this codebase's own optimizer/simulator separation-of-concerns design
   (derived from the paper's own methodology: the parallelism optimizer only "ranks and
   proposes" candidate configurations analytically, while the discrete-event simulator is the
   sole arbiter of actual SLO satisfaction and every reported metric) that this same session had
   written and enforced everywhere else — an ironic regression in the very code implementing
   that principle.

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

13. **Two of the external fairness audit's claims corrected.** (The audit document itself,
    `FAIRNESS_AUDIT.md`, was deleted in a later documentation cleanup once every finding it
    raised — F1-F8 — was confirmed addressed here; this item preserves the correction.)
    Independently re-verified and found factually incorrect:
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
    (`batch·2·total_len·num_heads/tp`) — per the paper's Read Prefetching double-buffering
    description (each flash stack's dedicated 3.13-MB SRAM staging buffer), that data is chunked
    through a *separate* buffer and is never resident in the 320-MB intermediate-data pool at
    all. Because this term scaled with sequence length, the 320-MB gate bit hardest at *long*
    context — backwards from the paper's intent that this be primarily a short-context,
    large-batch constraint — and because HBF's 36-GB scarce tier never binds regardless, only
    HBF+/CONV+ were penalized, producing the inversion. Fixed by adding a single shared
    `peakIntermediateBytes()` helper
    (`src/model/footprint.h`), used identically by the optimizer's `checkCapacity` call and the
    simulator's Part-C scarce-tier gate, that: (a) excludes the streamed attention-score/
    decompressed-KV terms entirely, (b) takes `max(attention-phase, FFN-phase)` instead of a
    sum, and (c) adds the dense-FFN transient term that the old formula omitted entirely for
    non-MoE layers (its `ffn_act` term was gated on `num_routed_expert > 0`, silently zeroing
    for dense models like llama3_405B). Effect: HBF+ total batch at MID/8-GPU roughly doubled for
    both models (llama3: 756→1444,
    llama4: 2336→4562), flipping the HBF+-vs-HBF total-batch inversion the right way round.

15. **Per-GPU batch-size divisor corrected: total GPU count, not DP replica count (supersedes
    item 10/"P1b").** After fix 14, HBF+'s *total* batch exceeded HBF's, as the paper requires —
    but the *reported* "per-GPU" figure (item 10's `max_batch / dp`) still showed HBF+ *below*
    HBF, because HBF+'s optimal config now sometimes uses `dp=2` (two independent 4-GPU
    replicas) where HBF stays at `dp=1`. Dividing by `dp` penalizes a config for splitting into
    *more* independent, hardware-efficient replicas — backwards, since a DP replica still
    consumes real GPU hardware whether or not it's counted in the denominator.
    this project's own TPS metric definition — "(Total Output Tokens) / (Total Simulation Time) /
    (**Number of GPUs**)," consistent with the paper's per-GPU-normalized throughput reporting
    convention — already divides by total GPU count, not DP, and `run_experiments.py`'s TPS
    formula (`max_b / (tpot * gpu)`) already implemented it that way; only the batch-size divisor
    (4 call sites in `run_experiments.py`) used `dp` instead, an internal inconsistency within the
    same file. Corrected all 4 sites to `max_batch / gpu_count`. This is a reversal of a decision
    defended earlier in this same session (originally argued that TP/PP GPUs "collaboratively
    serving the same sequences" justified not dividing by them) — that reasoning describes a real
    quantity (batch size per *replica*), but not what "per-GPU" means for the paper's
    cross-GPU-count comparisons: a hardware-normalized metric that must divide by all GPUs
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
    (e.g. `peakIntermediateBytes`).
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

19. **Compute-utilization (MFU) derating added to the roofline compute term (was implicitly
    100% of peak).** `linear_impl.cpp`, `activation_impl.cpp`, `attention_{gen,sum,mixed}_impl.cpp`,
    and `parallelism_optimizer.cpp` all computed `compute_duration = flops/compute_peak_flops`
    with no efficiency factor — every compute-bound GEMM was charged at ideal 100% of peak
    FLOPs, which no real GEMM achieves (tensor-core tile/wave quantization, epilogue overhead).
    Added `SystemConfig::mfu_max`/`mfu_m_half` (`hardware_config.h`, parsed from config.yaml's
    `simulation.mfu_max`/`mfu_m_half`) implementing a saturating curve `MFU(M) =
    mfu_max*M/(M+mfu_m_half)` (M = the GEMM's row/token count at that call site) via a shared
    `effectiveMFU()` helper, applied at every compute-duration call site. Defaults
    (`mfu_max=1.0`, `mfu_m_half=0.0`) make `MFU(M) == 1.0` for all M — an **exact no-op**
    verified byte-identical against a pre-change baseline across 7 anchor points. Investigated
    while root-causing `PAPER_INCONSISTENCIES.md`'s U1 (see that file): a sensitivity sweep
    (`mfu_max` in {1.0, 0.7, 0.6, 0.5}, `mfu_m_half=128` as a stated tensor-core-tile-granularity
    assumption, not tuned to any target number) confirms the effect is real and physically
    consistent — TPOT rises meaningfully as `mfu_max` drops (e.g. llama4_maverick/HBM4/8-GPU/
    SHORT: 0.0259s→0.0414s), with the steepest change right where the shared-expert MoE FFN's
    compute/memory ratio (~65% at mfu_max=1.0, hand-derived) crosses the roofline threshold
    around mfu_max≈0.6-0.7 — but even at mfu_max=0.5 the SHORT/MID batch anchors barely move
    (TPOT only reaches ~37-41% of the SLO), so this alone does not close U1's gap to the paper's
    reported numbers. Kept as a real, defensible, non-tuned addition to the cost model.

20. **Optimizer's TP all-reduce term aligned to the live simulator's ring-collective formula.**
    `parallelism_optimizer.cpp`'s analytic TP-communication estimate used a flat
    `2*layers*(device_ict_latency + full_message/device_ict_bandwidth)` — no `(N-1)/N` bandwidth
    factor, no per-hop latency multiply, full unsplit message per hop. The live simulator's
    `AllReduce::forward` (`communication.cpp`) instead models a proper ring all-reduce: for a
    group of size `N`, `2*(N-1)` hops, each hop paying one `device_ict_latency` plus `1/N` of the
    message. The optimizer now uses the identical ring formula (`per_allreduce = 2*(tp-1)*
    device_ict_latency + (2*(tp-1)/tp)*(message/device_ict_bandwidth)`, two all-reduces per
    layer). This is a ranking-heuristic-only fix (the simulator's measured TPOT remains the sole
    SLO arbiter, per audit F1) — verified to not move any of the tp=1 (pure-DP, zero-comm) paper
    anchors, but removes a latent optimizer/simulator drift relevant whenever TP>1 is in play.

21. **Batch-size search non-monotonicity bug fixed — `run_experiments.py`'s `find_max_batch_size`
    could substantially under-report the true max batch (found investigating
    `PAPER_INCONSISTENCIES.md`'s U8).** The search's boundary/exponential/binary-search steps
    each probed a single batch value and treated a failure there as proof that batch is
    infeasible from that point on. But parallelism-config selection is gated by an exact
    divisibility constraint (`batch_size % dp == 0`, `parallelism_optimizer.cpp`), so the set of
    TP/PP/DP/EP candidates available to the optimizer's ranking changes discontinuously between
    adjacent integer batch values — a batch value that happens to only be divisible for an
    inferior config (e.g. one with a much lower capacity ceiling) can spuriously "fail" even
    though slightly larger or smaller batches, divisible for the true-best config, succeed.
    Confirmed empirically: llama4_maverick/HBF+/16-GPU/LONG/offline — `b=9292` (divisible by 4,
    TP=1/PP=4/DP=4 selected, feasible up to ~10947) succeeds; `b=9293` (divisible by neither 4
    nor 2, only TP=4/PP=4/DP=1 remains divisibility-eligible, ceiling ~8192) spuriously fails;
    `b=9296` (divisible by 4 again) succeeds with TP=1 same as 9292. The single-probe boundary
    check took 9293's failure as final, under-reporting the true ~10947 ceiling as 9292 (a
    per-GPU batch of 580.75 instead of the true 684.12 — the exact spurious "676→580 batch drop"
    `PAPER_INCONSISTENCIES.md`'s U8 had flagged as "itself unusual...not yet investigated").
    Fixed by replacing every single-point probe (the boundary check, the exponential-doubling
    loop's failure point, and both binary searches) with `probe_window()`, which scans up to 8
    consecutive batch values (covering the TP-degree range this codebase's model presets use,
    `num_kv_heads`-capped) before concluding infeasibility. **Impact:** the reported per-GPU
    offline batch for llama4_maverick/HBF+/16-GPU jumps from 580.75 to 684.12 — essentially flat
    relative to the 8-GPU value (676.38), matching the paper's mechanism (iii) expectation
    (SRAM-tier capacity is a roughly fixed per-GPU ceiling, largely independent of total GPU
    count when TP is chosen appropriately) rather than the previously-reported drop. **Crucially,
    the corresponding per-GPU *throughput* (TPS/GPU) barely changed** (1535.06→1532.61,
    <0.2%) because TPOT scaled up proportionally with the corrected batch in this memory-
    bound regime — confirming the U4/U8 TPS-ratio gap (paper vs. simulator) is a *distinct*,
    still-open phenomenon, not an artifact of this search bug. Re-verified the full llama4/HBF+
    16-GPU LONG SLO sweep post-fix: TPS ratios vs. 8-GPU HBM4 are 0.959x/1.012x/1.038x/0.846x
    (0.05s/0.1s/0.2s/offline) — statistically identical to the pre-fix 0.96x/1.01x/1.04x/0.85x,
    confirming the fix is a pure batch/TPOT-reporting correction with no effect on the
    already-reported throughput ratios. The same artifact also produces a smaller shift on other
    anchors already reported in `PAPER_INCONSISTENCIES.md`'s U1: llama4_maverick/HBM4/8-GPU/SHORT
    moves 514.9→518.3 batch/GPU (~0.65%), TP=1/PP=8/DP=1 both before and after — noted there for
    bookkeeping; does not change any conclusion.

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

22. **Pipeline-parallel decode latency was never propagated across stages — a ~pp× undercount of
    every `PP>1` config's measured TPOT, and the root cause of `PAPER_INCONSISTENCIES.md`'s U1
    and the inverted U3 conclusion.** `PipelineStage` (`src/module/communication.cpp`) is the only
    cross-device module in the decode critical path constructed `sync=false` with a single-device
    `device_list` — unlike `AllReduce`/`MoEScatter` (`sync=true`, reconciled via
    `ModuleGraph::sync_devices()`'s max-and-broadcast), whose `device_list` only ever spans one
    pipeline stage's TP/EP group, never a stage boundary. `PipelineStage::forward` only added
    `comm_time` to the *source* device's own `status.device_time`, never touching the destination
    stage's clock; `LLM::forward` (`llm.cpp`) carries no data dependency across stages either
    (each stage starts from the original graph input, not the previous stage's output). So every
    pipeline stage's device independently accumulated only its own local layer work from 0 every
    iteration. `cluster.cpp` then read `get_device(0)->status.device_time` (stage 0 only) as "the"
    per-iteration decode-step time at 3 call sites (`runIterationMixed`, `runIterationSumGenSplit`,
    `setStat`) — for any `PP>1` config, this reports one stage's local time, not the true
    sequentially-summed cross-stage critical path. **Fix:** `PipelineStage::forward` now propagates
    `max(dst.device_time, src.device_time-after-comm)` to the destination device; `cluster.cpp`
    gained `Cluster::maxDeviceTime()` (max `status.device_time` across all devices) and all 3 call
    sites read this instead of `get_device(0)`. **Verified two ways:** (a) forced-distribution A/B
    at llama3_405B/HBF+/8-GPU/LONG/batch=200 — `PP=1` (TP=8) is byte-identical before/after
    (`0.1398461194712427` both, exact — no `PipelineStage` module is ever created when `pp_dg==1`);
    `PP=8` (TP=1) jumps `0.1153667652361051`→`0.9693988488543821` (~8.4×, matching the predicted
    `~pp×` undercount). (b) llama4_maverick/HBM4/8-GPU/SHORT, forced `TP=1/PP=8`: last stage's
    `device_time` was ~21ms pre-fix vs. the true ~164ms post-fix (8 sequential ~20.5ms stages) —
    at this exact operating point, 164ms **violates the 100ms SLO by 64%**, meaning the simulator
    had been silently reporting SLO-violating PP-heavy configs as passing.

23. **Optimizer's analytic latency estimate had the identical single-stage bug as item 22, and its
    candidate-selection objective was argmin(latency) instead of the paper's stated
    maximize-throughput.** `parallelism_optimizer.cpp`'s `total_latency_ns` only ever reflected one
    pipeline stage's terms (compute/weight/KV + that stage's own TP all-reduce + MoE scatter/
    gather), never multiplied by `pp`. **Fix:** restructured so the per-stage total is multiplied by
    `pp` (all stages run sequentially; one simulator iteration is one full forward pass through
    every stage) plus `(pp-1)` inter-stage hop costs, replacing the previous single-hop-only add.
    Candidate selection changed from `argmin(estimated_latency_ms)` to
    `argmax(batch_size_per_gpu / estimated_latency_ms)`, matching the paper's §III: "each evaluated
    system selects the parallelism configuration that maximizes the achievable system throughput
    subject to all constraints, including SLO requirements." **No SLO veto was added** —
    `estimated_latency_ms` remains ranking-only; `checkCapacity()` (capacity/SRAM) remains the only
    hard veto in `Optimize()`, and `run_experiments.py`'s live-simulator `verify()` remains the sole
    SLO arbiter, preserving the "optimizer proposes, simulator decides" principle throughout.
    Companion: `run_experiments.py`'s `BOUNDARY_WINDOW` widened from a fixed `8` to
    `max(num_device, 8)` — with `dp` no longer implicitly biased toward small values by a
    latency-minimizing objective, the divisibility-cycle window (item 21) must scale with GPU count
    to still guarantee catching a hit. **Verified impact:**
    - llama4_maverick/HBM4/8-GPU/SHORT (U1): batch/GPU 514.9→500.0 (paper: 460; miss shrank from
      ×1.119 to ×1.087). MID: 164.8→158.9 (paper: 151.5; ×1.088→×1.049). Neither fully closed.
    - U2 (HBF+/4-GPU vs. HBM4/8-GPU, LONG, TPS/GPU ratio): 0.722×→**1.001×** (paper: 1.15×) — HBF+
      now genuinely matches/edges HBM4, crossing the paper's qualitative claim, though short of its
      magnitude.
    - U4/U8 (HBF+/16-GPU vs. HBM4/8-GPU, LONG, 4-SLO sweep): 0.05s 0.959×→1.076×, 0.1s
      1.012×→1.171×. The 0.2s/offline cells surfaced a separate, confirmed-real finding — see the
      new U4/U8 write-up in `PAPER_INCONSISTENCIES.md` for the capacity-cliff mechanism (HBM4's
      best-latency config hits a hard capacity ceiling around batch 259 and must fall back to
      `PP=8`, the only larger-capacity option, at a steep latency cost) rather than restating it
      here.
    - **U3 inverted, not just corrected.** Forced A/B re-check at the exact original operating
      point (llama4/HBM4/8-GPU/SHORT, `PP=8/EP=1` vs. `PP=2/EP=4`, total batch=800): `PP=8/EP=1`
      tpot=0.0862s, `PP=2/EP=4` tpot=0.0121s — **`PP=2/EP=4` is ~7.1× faster**, not 4.6% slower as
      previously concluded. Decomposed via a batch-per-GPU-matched control (since `dp` is derived
      as `total_gpus/(tp*pp)`, `PP=2` implies `dp=4` vs. `PP=8`'s `dp=1` at the same total batch):
      holding per-GPU load constant (`PP=8` total=800/per_gpu=800 vs. `PP=2` total=3200/per_gpu=800)
      isolates a **~3.64× pure pipeline-depth effect** (matches the naive `pp`-count hand-estimate,
      ~3.8×, closely); the remaining **~1.96× is the DP-replica-parallelism effect** (4 concurrent
      replicas of the batch vs. 1). Both multiply out to the observed ~7.1×
      (3.64×1.96≈7.13). 2× batch stability check (`PP=8` @ total=1600): tpot scales sub-linearly
      (1.36× for 2× batch, consistent with the fixed weight-reread cost dominating this regime) —
      confirms the finding isn't a batch-800-specific artifact. Also directly force-tested all 4
      valid `PP=4` combinations (`TP∈{1,2}, EP∈{1,2}`) at the original comparison's capacity-cliff
      batch range (262, 264) — all 4 genuinely OOM in the live simulator, confirming `PP=8` really
      is the only capacity-feasible fallback there, not a search gap. See
      `PAPER_INCONSISTENCIES.md`'s U3 and `BUGS.md` item 8 for the corrected write-up.

24. **KV-write charged the full `input_len` for every layer, including Llama-4 iRoPE local layers
    whose retained KV is capped at `attn_chunk_size` — contradicting the already-windowed KV-read
    and KV-capacity paths (item 16), and inflating U6's measured KV-write overhead well above the
    paper's own stated range.** `getKVWriteDuration` (`src/hardware/layer_impl.h`) and the
    optimizer's mirror (`parallelism_optimizer.cpp`) used `model_config.input_len` directly in
    `kv_write_size` regardless of layer type. For Llama-4 (`attn_chunk_size=8192,
    attn_global_interval=4`), 3 of every 4 layers are local — their true persisted KV never
    exceeds `min(input_len, 8192)`, so charging the full `input_len` (103,500 at LONG) claimed to
    write ~3.2× more KV than the capacity path says the cache is ever sized to hold. **Fix:**
    `getKVWriteDuration` gained a `local_attention_window` parameter (default `0` = no-op), capping
    `effective_input_len = min(input_len, window)`; the 4 simulator call sites
    (`attention_gen_impl.cpp:211,871,1949`, `attention_mixed_impl.cpp:124`) now pass
    `layer_info.local_attention_window` (already computed per-layer for the KV-read path). The
    optimizer's version sums the (nonlinear — `unhidden = max(0, write - attn_compute)`) per-layer
    write cost over all layers using `effectiveKvLen()` (the same shared helper item 16's KV-read
    term uses) and divides by `pp`, mirroring `effectiveKvLenSumAllLayers()/pp`'s existing pattern.
    Deliberately did **not** adopt an alternative fix considered (widening the hiding budget from
    attention-compute-only to `max(compute, memory)`) — the paper's own footnote 2 (p. 3) states
    verbatim *"[KV writes] can be overlapped with **computation** in the attention layer,"*
    explicitly scoped to compute, not memory/KV-read time; widening it would contradict the paper's
    stated methodology while happening to move numbers toward its target ratios, the "reverse-
    engineer a fudge factor from the answer" pattern this investigation has repeatedly rejected.
    **Backward-compat verified:** `attn_chunk_size==0` (every non-Llama-4 model) makes the cap a
    no-op — `llama3_405B` and `deepseekV3` (MLA/compressed-KV call site) both byte-identical
    before/after (`tpot` matched to full double precision). **Verified effect:** llama4_maverick/
    HBF+/LONG measured KV-write share (U6) dropped from 19.7-26.6% (pre-fix, at the real near-SLO
    batch anchors, 4/8/16-GPU) toward the paper's Fig. 5 range (~6.7-7.5%) once combined with items
    22-23's fixes; MID (context below the 8192-token window, so windowing is a no-op there) stayed
    close to its pre-fix value (~15%) both times, consistent with the mechanism being
    windowing-specific rather than a general recalibration.

**Trailing "Still open" notes above are superseded by items 22-24** (the HBM4/8-GPU/SHORT anchor,
the degenerate-PP-config hypothesis, and the U3 reference are all addressed above with corrected,
verified numbers) — left in place for historical record rather than deleted, per this doc's
established practice of retaining superseded analysis with a pointer to the correction.

25. **Linear weight-read charged page-read latency once per op with no reference to chunk
    size/SRAM capacity at all** (audit F2), unlike the KV-read path, which does chunk/double-buffer.
    The paper's methodology describes double-buffering generically for "read data," not scoped to
    KV — leaning toward this being a real (if minor) inconsistency with the paper's stated model.
    Fixed: `getLinearMemoryDuration` (`src/hardware/layer_impl.h`) now uses the same chunked
    double-buffer formula as KV reads. **Verified numerically as a no-op on every current preset**
    (chunk-transfer time at full SRAM capacity exceeds page latency for all 4 flash presets, so
    exposed latency reduces to exactly one page latency either way) — confirmed empirically via
    byte-identical before/after runs on both HBF (`0.0966962731654616s`) and CONV, the
    highest-page-latency preset (`0.09089014626032631s`). Kept for structural correctness and
    forward-safety if config constants ever change, even though it moves no current number.
    (Originally documented only in `PAPER_INCONSISTENCIES.md`'s Resolved section, consolidated
    here as part of the documentation cleanup that trimmed that doc to open/explained findings
    only.)

26. **Degenerate one-device-per-pipeline-stage optimizer choices were never actually optimal — a
    prior "ranking validated, not a bug" conclusion was itself an artifact of item 22's bug**
    (consolidates `BUGS.md`'s former item 8 and `PAPER_INCONSISTENCIES.md`'s former U3, both
    removed as part of the same cleanup — this is now the single authoritative write-up). The
    optimizer's ranking can select configs like llama4_maverick's 8-GPU choice (`PP=8/EP=1/TP=1`,
    one device per pipeline stage) that trivially zero every communication term by construction.
    An earlier A/B check (forcing `PP=8/EP=1` — the optimizer's actual choice — against
    `PP=2/EP=4` — a rejected alternative — at llama4/HBM4/8-GPU/SHORT, comparing real
    simulator-measured TPOT) concluded `PP=8/EP=1` genuinely won by ~4.6%, "validating" the
    ranking. **That conclusion was wrong — not just imprecise, but backwards.** The live simulator
    itself had item 22's bug (`PipelineStage::forward` never propagated elapsed time across
    pipeline-stage boundaries), so the A/B was comparing stage-0's local time for both configs —
    undercounted by roughly each config's own `pp` factor (~8× for `PP=8`, ~2× for `PP=2`). Since
    the undercount differs between the two candidates, it does not cancel in the comparison; the
    original near-tie (0.02449s vs. 0.02563s) was actually one stage's cost 8× smaller than reality
    vs. one stage's cost only 2× smaller, making `PP=8` look artificially competitive.
    **Re-running the identical forced A/B post-fix (item 23) gives the opposite answer:**
    `PP=8/EP=1/TP=1` tpot=**0.0862s**, `PP=2/EP=4/TP=1` tpot=**0.0121s** at the same operating
    point (total batch=800) — **`PP=2/EP=4` is ~7.1× faster**, not 4.6% slower. Decomposed into two
    independent, multiplicative effects via a batch-per-GPU-matched control (since
    `dp = total_gpus/(tp*pp)` is derived, not independently chosen, `PP=2` implies `dp=4` vs.
    `PP=8`'s `dp=1` at the same *total* batch — so the unmatched comparison conflates pipeline
    depth with DP-replica count):

    | Comparison | PP=8/EP=1 tpot | PP=2/EP=4 tpot | Ratio |
    |---|---|---|---|
    | Unmatched (both at total batch=800; PP=2 → dp=4, per-GPU load=200) | 0.0862s | 0.0121s | 7.12× |
    | **Matched per-GPU load** (PP=8 @ total=800/per_gpu=800; PP=2 @ total=3200/per_gpu=800) | 0.0862s | 0.0237s | **3.64×** |

    The matched comparison isolates a **~3.64× pure pipeline-depth effect** (close to the naive
    `pp`-count hand-estimate of ~3.8×, from 8 vs. 2 sequential stages' own weight-restream/compute
    time plus the newly-modeled inter-stage hop costs) — the remaining **~1.96× is the
    DP-replica-parallelism effect** (4 concurrent replicas each handling 1/4 the batch, vs. 1
    replica handling all of it). `3.64 × 1.96 ≈ 7.13`, consistent with the unmatched ratio.
    **Verified this isn't a batch-800-specific artifact:** a 2× batch stability check (`PP=8` @
    total=1600) shows tpot scaling sub-linearly (1.36× for 2× batch — consistent with the fixed
    weight-reread cost dominating this regime, not a coincidence). **Verified this isn't a
    missed-config search gap:** all 4 valid `PP=4` combinations (`TP∈{1,2}, EP∈{1,2}` — the only
    intermediate discrete option between `PP=2` and `PP=8` at 8 GPUs) were force-tested directly
    against the live simulator at the original comparison's capacity-cliff batch range (262, 264)
    — all 4 genuinely `Out of Memory`, confirming `PP=8` really was the only capacity-feasible
    fallback there, not an artifact of the optimizer's own candidate ranking skipping a better
    option. **The degenerate configs the optimizer used to favor were never actually optimal —
    they only looked cheap because their true multi-stage latency was hidden by item 22's bug**,
    which also explains, mechanistically, why the pre-fix optimizer kept selecting them (their
    `estimated_latency_ms` was artificially divided by `pp`). With both the propagation bug (item
    22) and the ranking objective (item 23: `argmin(latency)`→`argmax(throughput)`) now fixed, the
    optimizer's config selection is grounded in a real, correctly-computed comparison going
    forward.

27. **`num_max_batched_token` hardcoded to 8192 made every batch > 8192 crash on a zero-length
    `Sequence`, masquerading as a real capacity ceiling.** `eval/test.cpp:507` constructs the
    scheduler with `num_max_batched_token = 8192` (a fixed literal, independent of the batch under
    test). `Scheduler::getMaxMetadata` (`src/scheduler/scheduler.cpp:261`) then derives each
    sequence's length as `int seq_len = num_max_batched_token / batch_size_per_dp;`. For any
    `batch_size_per_dp > 8192` this integer-divides to **0**, and the immediately-following
    `Sequence::Create(seq_len, seq_len)` trips `sequence.cpp:19`'s `assertTrue(input_len > 0, ...)`
    ("input len is 0"), aborting the process with a non-zero exit. `run_experiments.py`'s
    `run_simulation` catches any non-zero exit as `"OOM/Crash"` (line 154) and `classify_failure`
    returns `"unknown"` (no capacity marker present), so `find_max_batch_size` reads the crash as a
    genuine capacity ceiling and stops the batch search at 8192 — under-reporting the true feasible
    batch for any cell whose real ceiling exceeds 8192/dp. **Fix:** clamp `if (seq_len < 1) seq_len
    = 1;` — provably a no-op for every batch ≤ 8192 (where `seq_len` was already ≥ 1), and for
    larger batches it lets the sequence be built with the minimum valid length so the run proceeds
    to the real SLO/capacity check instead of aborting. Verified batches 8193/9000/15000 now
    terminate on a legitimate SLO violation or capacity rejection rather than the assert. Same bug
    *class* as the sibling worktree's activation-overflow fixes (`BUGS_FIXES.md` T2), but a distinct
    site (scheduler metadata construction, not the activation-size formula). **Impact on the paper
    comparison:** materially changed U7's HBF+/CONV+ shape (the old monotonic-growth-to-2149 anchor
    was partly the crash capping the search at 8192; post-fix the per-GPU batch peaks at 4 GPU and
    declines to an SRAM-bound ceiling by 16 GPU — a much closer match to the paper's flat/SRAM-bound
    curve; see `PAPER_INCONSISTENCIES.md` U7).

28. **GQA "current-token k,v" activation term used `num_heads` where the projection produces
    `num_kv_heads` — an over-count of the resident k/v footprint.** `src/model/footprint.h`'s
    `peakIntermediateBytes` GQA-base branch (~line 190) sized the current decode step's freshly
    projected key/value tensors as `batch_per_dp * 2.0 * head_dim * num_heads / tp`. Under grouped-
    query attention the k/v projections emit `num_kv_heads` heads, not `num_heads` (confirmed against
    `src/module/attention.cpp`'s actual k/v tensor shapes) — for llama4_maverick/llama3_405B
    (`num_heads=128`, `num_kv_heads=8`) this over-counted that one term 16×. **Fix:** `num_heads` →
    `num_kv_heads` in that term only. Isolated to the GQA-base branch (the MLA `use_absorb` /
    `compressed_kv` branches never execute it), so it is a **no-op for deepseekV3-style MLA models**.
    **Numerically inert on every current anchor tested**, for two independent reasons: (a) for
    llama4_maverick the FFN-phase MoE term dominates `max(attn_phase, ffn_phase)`, so the attention
    term never binds the scarce-tier gate; (b) for llama3_405B none of the tested SHORT/MID GPU
    counts are SRAM-bound. Notably this fix makes the HBF+/CONV+ SRAM ceiling *tighter* (moves U7's
    numbers slightly *further* from the paper's looser reported ceiling), which is itself evidence it
    is a real correctness fix and not a paper-matching adjustment. Kept regardless of its inertness:
    the two footprint gates (`peakIntermediateBytes` here and `cluster.cpp`'s Part C) share this one
    definition, so the term must be correct for any future config where `num_heads != num_kv_heads`
    and the attention phase does bind.

29. **Optimizer's HBM-path MoE `weight_for_latency` used an unjustified `sparse_ratio` discount that
    contradicted the live simulator's own per-expert dispatch.** `parallelism_optimizer.cpp`'s
    latency estimate (Fix 2c) charged routed-expert weight-read bandwidth differently by memory tier:
    the **flash** branch used the physically-grounded `e_active` model (the number of on-device
    experts that receive ≥1 token, each streaming its full `k·n` weight), but the **HBM** branch
    applied `sparse_ratio = top_k / num_routed_expert` to the *per-device* routed weight — at the
    SHORT anchor this charged the equivalent of ~1 expert's weight where the flash path (same
    operating point) charges ~16–32. Two independent audits (the second explicitly tasked with
    refuting the first) agreed it was wrong on its own merits: (a) it contradicted the flash branch
    for the identical operating point; (b) it contradicted the live simulator's actual dispatch —
    `ExpertFFN::forward` (`src/module/expert.cpp`) loops per expert, charging each full `k·n` weight
    via `getLinearMemoryDuration`, skipped only when a device receives zero tokens for that expert;
    this is memory-tier-agnostic, and there is **no grouped-GEMM / active-row HBM-specific
    optimization anywhere in the dispatch code**; (c) the "accounts for compute-memory overlap"
    rationale double-counts — the default `optimizer_latency_model=="max"` already credits overlap via
    `max(compute, weight_mem)`; (d) the paper (Son et al., §III) frames the HBM-vs-flash distinction
    as purely a read/write-bandwidth asymmetry, with large-batch benefit coming from amortizing
    active-expert weight-read across queries — exactly what `e_active` captures and a fixed
    `sparse_ratio` does not. **Fix:** both tiers now use the `e_active` model (the HBM `else` branch
    is deleted). **Provably cannot regress dense models** (llama3_405B never enters the
    `num_routed_expert > 0` block). **Measured before/after on the current binary: byte-identical on
    every llama4_maverick anchor** — HBM4/8-GPU SHORT (500.0/GPU), MID (158.88/GPU), and LONG across
    all four SLOs (258/258/264/264 total; TPS/GPU 1401/1401/184/184) are unchanged to full precision.
    The reason is regime, not coincidence: in the `"max"` latency model these operating points are
    compute-bound (SHORT/MID) or capacity/SLO-pinned (LONG), so raising the routed-weight estimate
    ~128× never makes `max(compute, weight_mem)` flip to weight-bound and the ranking/selection is
    untouched. Applied as a ranking-only correctness/consistency fix (it never gates SLO pass/fail —
    `checkCapacity` and the live-simulator TPOT remain the only hard/authoritative signals) that
    unifies the two memory paths and hardens the estimate for any future constants where the weight
    term would bind; see `PAPER_INCONSISTENCIES.md` U1/U2.

30. **Optimizer ranking metric under-ranked DP configs by exactly a factor of `dp` — zero-communication
    DP+EP layouts could never win selection.** Item 23 changed the selection to "throughput" ranking via
    `(batch_size / dp) / estimated_latency_ms`, but `batch_size / dp` is the per-REPLICA batch, so that
    expression is per-replica throughput, not system throughput: a dp-way config's `dp` replicas each
    carry `batch/dp` and run concurrently, so system TPS = `batch / L` and per-GPU TPS =
    `batch / (L · total_gpus)` — within one `Optimize()` call `batch` and `total_gpus` are constants
    across candidates, so the correct rank is **argmin estimated_latency_ms**. The spurious `1/dp`
    penalized DP-heavy candidates by `dp`×, which made the zero-all-reduce "attention-DP + experts
    count-sharded across the stage" layout structurally unable to beat TP-heavy layouts no matter how
    much all-reduce time the latter pay. This config is exactly the paper's llama4 operating point:
    forced `tp=1/pp=1/dp=8/ep=1` on llama4_maverick/HBM4/8-GPU verifies end-to-end (recorded weight
    116.4 GiB = replicated non-expert + 16-of-128 experts/GPU; capacity ceiling 3680 total = 460/GPU =
    the paper's printed Fig-3 anchor exactly; measured tpot 25.8 ms ≈ the paper's Fig-4-implied
    24.3 ms; zero comm = the paper's Fig-5 llama4 comm=0.0% rows). **Fix:**
    `parallelism_optimizer.cpp`'s reduction is now argmin-latency over non-OOM candidates (comment
    carries the derivation). Capacity/SRAM remain the only hard vetoes; no SLO gate added.

31. **Sweep objective aligned to the paper's §III: per-parallelism-config max-batch search, report the
    argmax-TPS config (was: maximize batch across ALL configs, report that batch's TPS).** §III
    verbatim: "each evaluated system selects the parallelism configuration that maximizes the
    achievable system throughput subject to all constraints." The old global-max-batch objective could
    publish a low-throughput config that squeezed out a few more sequences of capacity — the exact
    mechanism behind PAPER_INCONSISTENCIES U4/U8's ~9× Fig-6 outlier (llama4/HBM4/8-GPU at loose SLOs
    crossing from TP4/PP2, batch 258 / TPS-per-GPU 1401, to TP1/PP8, batch 264 / TPS 184, for +6
    sequences) and part of U1's batch-anchor overshoot. **Implementation:** (a)
    `ParallelismOptimizer` refactored into `EnumerateCandidates()` (structural gates; optional
    batch%dp) + `EvaluateConfig()` (one fixed tuple at one batch) with `Optimize()` = enumerate +
    item-30 reduction, behavior-preserving; (b) new `analytic_configs_only` mode in `eval/test.cpp`
    emits per capacity-feasible config its analytic capacity ceiling and SLO-latency hint
    (`ANALYTIC_NUM_CONFIGS` / `ANALYTIC_CONFIG: tp= pp= ep= dp= cap_batch= slo_hint_batch=
    est_lat_min_ms= est_lat_hint_ms=`); (c) `run_experiments.py::find_max_batch_size` rewritten (same
    return-tuple shape, all downstream consumers untouched): per config, simulator-verified bisection
    over multiples of its own dp seeded by the hint, jumping to the analytic capacity ceiling
    (post-item-32 it tracks the recorded footprint to <0.01%), then probing upward until failure (the
    est≤measured invariant is audited, not guaranteed); winner = argmax verified
    `batch/(tpot·total_gpus)`; its batch/tpot/CSV/bound_reason become the cell's operating point, and
    the old objective's answer is logged as `legacy_global_max_batch`. Within a fixed config
    feasibility is monotone in batch (no batch%dp config-switching), so item 21's
    `probe_window`/`BOUNDARY_WINDOW` machinery is structurally unnecessary in the per-config search —
    this also removes U1's "crash band" search artifact. Pruning: configs are simulated in descending
    `seed_tps = slo_hint_batch/(est_lat_hint·G)` order and skipped once their seed_tps (an upper bound
    on achievable TPS under the audited estimate≤measured invariant) cannot beat the best verified
    TPS; `HBF_DISABLE_CONFIG_PRUNING=1` simulates every feasible config. Also:
    `Cluster::checkMemorySize`'s main capacity gate now prints "HBM/Flash capacity exceeded" so
    forced-distribution OOMs classify as a capacity bound instead of "unknown" in Fig-3's
    bound_reason. **Measured:** llama4/HBM4/8-GPU/SHORT/0.1s went from batch 500/GPU (×1.087 vs the
    paper's 460) at TPS/GPU ~12.0K (paper 18.9K, −37%) with a 26%-comm TP8/EP8 config, to batch
    483.5/GPU (×1.051) at TPS/GPU 18.0K (−4.9%) — with item 34's NVLink fix; under NVLink-5 the winner
    was the pure-DP config at exactly 460.0/GPU. llama3 anchors (below) also verified.

32. **Analytic weight footprint reached parity with the simulator's recorded tensors; the recorded
    side itself double-counted the MoE router.** Two halves: (a) the optimizer's `weight_per_gpu`
    omitted weights the live simulator records — the MoE router/gate projection (`expert.cpp`'s
    `gate_fn`, `hidden × num_routed × precision` per MoE layer, replicated per device: 24 × 1.31 MB =
    31.46 MB for maverick = **precisely the ~0.01% optimizer-vs-simulator capacity drift** behind U1's
    batch-4001–4095 crash band), LayerNorm gammas (2/decoder layer), and the un-TP-sharded embedding
    (PP stage 0) and LM head (last stage; device-0 parity ⇒ counted when pp==1) — all now added to the
    capacity weight, and the streamed subset (router/LN/LM-head, not embedding whose forward charges
    no memory op) to `weight_for_latency`; (b) `Route::Route` (`route.cpp`) created a dead
    `route_weight` tensor of the identical `{hidden, num_routed}` shape, tagged "weight" and never
    referenced again — a duplicate record of the same physical router matrix that `gate_fn` already
    records; removed. **Verified:** at the item-30 validation point the pred-vs-recorded weight drift
    fell from 0.025% (pre) to <0.01% (the Part-E harness stays silent at a 0.0001 threshold);
    recorded weight dropped 116.413 → 116.384 GiB (the removed duplicate).

33. **KV-write page-program latency amortized to one exposed tail per iteration's write stream (was: a
    flat 100 µs per attention layer per iteration — 48× overcounted for llama4).**
    `getKVWriteDuration` (`layer_impl.h`) added `flash_page_program_latency_ns` on every per-layer
    call: 48 × 100 µs = 4.8 ms/iteration of pure program latency for llama4 before hiding.
    Hand-verified decomposition at U2's cell (llama4/LONG/HBF+/4-GPU, batch 523): bytes term
    5.85 ms/GPU + 4.8 ms latency ≈ the measured 10.6% kv_write share, vs the paper's Fig-5 reading of
    6.7% ≈ bytes + ~one latency. Grounding (not calibration): the KV write is a fire-and-forget
    background stream — the model already charges only the unhidden remainder vs the attention
    kernel's own compute, nothing per-layer ever waits on program completion, successive layers'
    admitted-KV writes queue back-to-back into the same flash write path, and the aggregate
    `flash_write_bandwidth` (0.128 TB/s = 8 stacks × 16 GB/s) already embodies sustained multi-plane
    program throughput — so the program latency is a pipeline-fill/tail cost of the per-iteration
    stream, exposed once, not once per layer. **Fix:** `getKVWriteDuration` takes
    `program_latency_amortize_calls` (= layers per PP stage, passed at all four call sites); the
    optimizer's mirror term divides by `layers_per_stage` identically. **Measured at U2's cell:**
    kv_write share 10.6% → 6.08% (paper 6.7%), TPOT 99.9 → 90.1 ms at the same batch/config.

34. **NVLink generation aligned to the paper's §III hardware spec: gen 6 (Rubin), 1,800 GB/s per
    direction (was gen 5, 900 GB/s — a Blackwell rate on a Rubin system).** §III verbatim: "NVLink
    (1,800 GB/s) for intra-node GPU communication", on systems whose GPU-core architecture cites the
    DGX Rubin NVL8. This file's convention is unidirectional (its gen-4 constant is 450 GB/s = H100's
    NVLink-4 per-direction rate), so the paper's 1,800 GB/s = NVLink-6's 3.6 TB/s bidirectional ÷ 2 —
    added as `nvlink_gen: 6` (`eval/test.cpp`) and set in `config.yaml`.
    *Acknowledged ambiguity:* NVIDIA's marketing convention quotes NVLink bidirectionally, under
    which "1,800 GB/s" would read as NVLink-5 (Blackwell) = 900 GB/s per direction — i.e. the old
    setting. That reading was rejected on three grounds: (a) the paper's own platform is Rubin,
    whose interconnect is NVLink-6 (3.6 TB/s bidir / 1,800 per direction) — quoting bidirectional
    for their own cited hardware would have read 3,600; (b) the simulator's ring all-reduce sends
    and receives concurrently per link, so its `device_ict_bandwidth` is a per-direction rate by
    construction, whatever the paper's quoting convention; (c) empirically, 900 GB/s per direction
    cannot reproduce the paper's llama3 anchors under any config (tp8 comm = 39.9% of decode; the
    throughput-max winner flips to tp4/dp2 at 95 batch/GPU vs the paper's 195.5), while 1,800
    reproduces SHORT/MID batch AND TPS to ±3%. If the paper's number is later confirmed
    bidirectional, its §III spec is internally inconsistent with its cited hardware and its own
    published anchors under this simulator's collective model. **Root-caused from, and
    verified against, the llama3 SHORT anchor:** at `nvlink_gen: 5` the tp=8 config measured
    tpot 68.5 ms with communication = 39.9% of decode (27.3 ms), making the item-31 search correctly
    prefer a tp=4/dp=2 config at 95/GPU — half the paper's batch anchor; at gen 6 tp=8 measures
    56.3 ms and wins, restoring batch 190.9/GPU (paper 195.5, −2.4%) at TPS/GPU 3393 (paper 3290,
    +3.1%). llama3 MID: 60.6/GPU & 1780 TPS (paper 62.0 & 1795, −2.2%/−0.8%); LONG: 3.62/GPU & 127.5
    TPS (paper 3.8 & 146.6, −4.7%/−13% — TPS residual still open). The stale +18%/+14% llama3
    SHORT/MID TPS errors in the old experiment tables are gone.

35. **Pipeline-stage decode timing serialized correctly for tp≥2 — the item-22 propagation silently
    produced OPPOSITE (fully-overlapped) semantics for every tp≥2/pp>1 config.** Item 22's fix bumps
    the destination stage's clock with `max(dst, src_cumulative)` in `PipelineStage::forward`
    (`communication.cpp`). That is only correct when the destination device has not executed anything
    yet — which was an accident of `Cluster::run`'s round-robin scheduler: with tp==1 no intra-stage
    sync module blocks, each device drains its whole graph in one pass in rank order, so dst really
    was at 0 (stages serialized; the ~21 ms vs ~164 ms llama4 TP1/PP8 verification in item 22). With
    tp≥2 the per-layer AllReduce blocks every device each layer, all stages advance in lock-step,
    and by the time the bump lands dst has already accumulated ~its full stage work — `max()` then
    reports TPOT ≈ ONE stage instead of the sum, under-counting pp>1 decode TPOT by up to pp× for
    every tp≥2 config. Root-caused with a controlled tp×pp sweep at fixed batch (dp=1, tp·pp=8):
    only the tp=1 column serialized (~5× per-stage), for BOTH dense llama3 and MoE maverick —
    dense-vs-MoE is irrelevant; the intra-stage sync barrier is the discriminating variable.
    Physically, serialization is the correct decode accounting: autoregressive dependency means a
    request's token t+1 cannot enter stage 0 before token t leaves the last stage, and micro-batch
    pipelining leaves the steady-state rate at batch/Σstages regardless. **Fix:** the bump is now an
    ADDITION (`dst.device_time += src_cumulative`) — exact for both orderings since clocks reset per
    iteration and upstream bumps land in rank order (= pipeline order) before a stage's own
    PipelineStage runs. tp==1 behavior is bit-identical (dst contributes 0). **Measured:**
    llama3/HBM4/8GPU/SHORT tp4/pp2 batch 1559: 45.9 ms (overlap bug) → ~90 ms serialized, matching
    the analytic ×pp model (90.45 ms) that item 23 built — the two now agree, which also restores
    the estimate≤measured invariant that the sweep's pruning bound and the F1 search design rely on
    (a pruning A/B on this exact cell had exposed the violation: the analytic model "over-estimated"
    tp4/pp2 by 2× only because the simulator was under-counting it by 2×). None of the previously
    verified paper-comparison anchors move: every winning config in the U1/U2/U4/llama3-anchor
    verifications is pp=1.

## Paper-comparison items resolved (2026-07-03)

Resolution records for the paper-comparison findings formerly tracked as open in
`PAPER_INCONSISTENCIES.md` (U1, U2, U4, U8, U6). Each began as a discrepancy against Son et al.'s
reported numbers and is now resolved by the correctness/objective fixes recorded as items 30-35
above (plus items 21/24 for the older components). The technical mechanism of each fix lives in
those items; this section records, per finding, the original inconsistency, the root cause(s), the
fix references, and the final verified numbers. Multi-pass historical narratives were condensed
away — only the final before/after evidence a future reader needs to trust the resolution is kept.

### U1 — llama4_maverick's HBM4/8-GPU SHORT & MID batch anchors resolved

**Original inconsistency:** the SHORT and MID per-GPU batch anchors ran ~9-12% high vs the paper
(SHORT ×1.087, MID ×1.049), and the reported config paid ~26% all-reduce communication where the
paper's Fig-5 llama4 rows read comm=0.0%. Earlier framing blamed an optimistic cost model; the real
cause was that the search could never select the paper's operating point.

**Root cause(s):** (i) the optimizer's ranking metric divided by `dp`, under-ranking zero-comm DP
layouts by exactly `dp×` (item 30); (ii) the sweep reported the max-batch config instead of the
§III max-throughput config (item 31); (iii) `nvlink_gen: 5` (900 GB/s Blackwell) instead of the
paper's §III 1,800 GB/s Rubin rate (item 34); plus the MoE router/gate weight missing from the
optimizer footprint and double-recorded on the simulator side, which produced the batch-4001–4095
"crash band" (item 32).

**Final verified numbers (fixed binary, items 30-34):** winner `TP=2/PP=1/DP=4`.

| Anchor | our batch/GPU | paper | ratio | our TPS/GPU | paper | Δ |
|---|---|---|---|---|---|---|
| SHORT | 483.5 | 460 | ×1.051 | 18,016 | 18,943 | −4.9% |
| MID | 153.5 | 151.5 | ×1.013 | 6,301 | 6,281 | +0.3% |

Under NVLink-5 the winner was instead the pure-DP `tp=1/pp=1/dp=8` layout at exactly 460.0/GPU (the
paper's printed anchor). The crash band is structurally gone (item 32). SHORT residual (+5% batch,
−5% TPS) is the tp2-vs-dp8 selection margin under our comm constants; not tuned further.

### U2 — llama4 LONG "4-GPU HBF+ TPS 15% higher than 8-GPU HBM4" resolved to within 3%

**Original inconsistency:** the paper's Takeaway is that HBF+/4-GPU per-GPU TPS is ~1.15× of
HBM4/8-GPU's on the llama4 LONG workload; the simulator measured a ratio of only 1.002×, so HBF+
merely edged HBM4 rather than beating it by the paper's margin.

**Root cause(s):** (i) the flat 100-µs KV-write page-program latency was charged once per attention
layer per iteration (48× for llama4 = 4.8 ms/iter) instead of once per iteration's fire-and-forget
write stream (item 33); (ii) the sweep reported max-batch rather than the §III throughput-max
config for both cells (items 30-31).

**Final verified numbers (fixed binary):** ratio **1.116×** (paper 1.15×), both cells now matching
the paper's own Fig-3/Fig-4 readings to ~1-4%.

| Cell | winner | batch/GPU (paper) | tpot / bound | TPS/GPU (paper) |
|---|---|---|---|---|
| HBF+/4gpu | TP=2/PP=1/DP=2 | 150.0 (151.5, −1.0%) | 0.0997s, slo | 1503.9 (1489.8, +0.9%) |
| HBM4/8gpu | TP=4/PP=1/EP=8/DP=2 | 31.0 (31.3, −1.0%) | 0.023s, capacity | 1347.2 (1296.1, +3.9%) |

Post-fix kv_write share at this cell: **6.28% vs the paper's Fig-5 reading of 6.7%**. The residual
−3% on the ratio comes entirely from HBM4's +3.9% TPS (its tpot 0.023s vs the paper-implied
0.0242s); not tuned further.

### U4 — llama4 LONG "HBF+ always outperforms HBM4 at 16 GPUs across all SLOs" resolved

**Original inconsistency:** the 4-SLO sweep produced a ~9× outlier at the 0.2s/offline SLOs (and an
earlier 0.85× shrink at offline), so the HBF+/16-GPU-vs-HBM4/8-GPU ratio was neither monotone nor
within the paper's Fig-6 range (nothing above ~1.5×).

**Root cause(s):** the max-batch objective on the HBM4 denominator — at loose SLOs the max-batch
search crossed from config `TP=4/PP=2/EP=4` (tpot 0.023s, TPS/GPU 1401) into `TP=1/PP=8/EP=1` (tpot
0.179s, TPS/GPU 184) to gain just 6 sequences of capacity, cratering the reported denominator. Fixed
by the §III per-config throughput-max objective plus the DP-ranking correction (items 30-31); the
config-switch itself is a real capacity cliff, no longer reported because the search now maximizes
throughput, not batch.

**Final verified numbers (fixed binary, 4-SLO sweep):** ~9× outlier gone; monotone and within 1.6%
of Fig-6 at every SLO. Baseline SLO-invariant (winner stays `TP=4/PP=1/EP=8/DP=2`, batch 248/GPU,
TPS 1347 at every SLO — the paper's "HBM4 hardly changes across SLOs").

| SLO | HBM4/8gpu (batch/GPU, TPS/GPU) | HBF+/16gpu (batch/GPU, TPS/GPU, bound) | ratio | paper Fig-6 (unwrapped†) |
|---|---|---|---|---|
| 0.05s | 31.0, 1347.2 | 80.25, 1606.8, slo | **1.193** | ≈1.21 |
| 0.1s | 31.0, 1347.2 | 172.25, 1724.1, slo | **1.280** | ≈1.30 |
| 0.2s | 31.0, 1347.2 | 355.0, 1775.3, slo | **1.318** | ≈1.33 |
| offline | 31.0, 1347.2 | 682.75, 1795.9, flash | **1.333** | ≈1.34 |

† paper Fig-6 readings divided pairwise (HBF+-16GPU / HBM4-8GPU-same-SLO), since the figure's own
HBM4-8GPU series reads ~0.84, not 1.0, against its stated normalization. The offline HBF+ point is
capacity-bound (bound=flash) at 682.75 batch/GPU vs the paper's ≈700 (−2.5%).

### U8 — llama4 HBF+ TPS edge over HBM4 now grows monotonically with looser SLOs

**Original inconsistency:** HBF+'s relative benefit shrank at the offline SLO (0.85×) instead of
growing the way Fig-6 claims; a later pass instead saw a ~9× spike. Also, the per-GPU batch appeared
to drop 8→16 GPUs (676.4 → 580.8).

**Root cause(s):** the batch-drop component was the batch-search non-monotonicity bug (item 21),
already fixed (corrected to 684.12 at 16 GPUs vs 676.38 at 8 GPUs — essentially flat, matching the
paper's SRAM-per-GPU mechanism iii). The remaining TPS-shrink/spike was the same max-batch-objective
defect on the HBM4 denominator described under U4 (items 30-31).

**Final verified numbers:** once the sweep reports the §III throughput-max operating point, the
ratio is strictly monotone **1.193 → 1.280 → 1.318 → 1.333** (paper-unwrapped ≈1.21/1.30/1.33/1.34;
see U4's table), with the offline point capacity-bound at 682.75 batch/GPU vs the paper's ≈700. Both
the earlier ~9× "growth" and the 0.85× "shrink" were artifacts of the max-batch objective, not real
HBF+ behavior.

### U6 — HBF+ KV-write "unhidden" overhead now inside the paper's stated range

**Original inconsistency:** measured HBF+ KV-write share ran 14.7%@4-GPU → 19.8%@16-GPU, above the
paper's Fig-5 stated range of "5–13.9% of the execution time in Llama4."

**Root cause(s):** the flat 100-µs flash page-program latency was charged per attention layer per
iteration (48× for llama4 ≈ 4.8 ms/iter) instead of once per iteration's fire-and-forget write
stream (item 33); additionally the KV-write size wasn't windowed for llama4's iRoPE local-attention
layers, unlike the already-windowed KV-read/capacity paths (item 24). The hiding budget itself
(attention-only compute, per the paper's footnote 2) was left unchanged — deliberately not widened,
to avoid fitting to the target.

**Final verified numbers:** post-fix, validated against all 12 of the paper's Fig-5 HBF+ KV-write
readings at each cell's own throughput-max operating point (0.1s SLO):

| Series | our KV-write share (4/8/16 GPU) | paper Fig-5 |
|---|---|---|
| llama4 MID | 12.86 / 14.18 / 13.40% | 13.0 / 13.6 / 13.7 |
| llama4 LONG | 6.28 / 6.95 / 7.19% | 6.7 / 7.1 / 7.5 |
| llama3 MID | 10.15% (all counts) | 9.3 |
| llama3 LONG | 5.20 / 5.54 / 5.54% | 6.4 / 7.1 / 7.1 |

Mean |error| ≈ 0.7pp with mixed signs (llama4 within ±0.6pp everywhere) — a physical correction, not
a fit to any one cell. Not a bug; a better-calibrated contributing factor to U2's residual.

36. **Diagnostic-only score-inclusive SRAM footprint added (never gates).** Outcome of the U7
    score-accounting A/B (see PAPER_INCONSISTENCIES.md U7): the user decided to keep the
    score-exclusive activation model (matches 10/12 of the paper's HBF+ batch bars) and log the
    paper-style score-inclusive accounting alongside it. `footprint.h::scoreInclusiveIntermediateBytes`
    computes peakIntermediateBytes plus the chunked-attention score/softmax working set
    (2 buffers x heads/tp x min(ctx, attn_chunk_size) x precision); `Cluster::checkMemorySize`
    prints `SRAM_DIAG_SCORE_INCLUSIVE_ACT_BYTES` and `SRAM_DIAG_CEILING_BATCH_PER_GPU` on
    scarce-tier decode runs; `run_experiments.py` parses the ceiling into run results and appends
    `sram_diag_ceiling_per_gpu` to `[config-search]` lines. No reported metric changes.
