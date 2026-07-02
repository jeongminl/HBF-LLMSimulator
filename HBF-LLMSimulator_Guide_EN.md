# HBF-LLMSimulator Structure & Functionality Guide

> This document is a structure/behavior guide written after directly reading and verifying the
> code of the `HBF-LLMSimulator` repository — a fork of `LLMSimulator` (the cycle-approximate
> simulator from MICRO 2024's *"Duplex"* paper) extended with **High-Bandwidth Flash (HBF)**
> memory modeling and a constraint-aware **parallelism optimizer**, implementing the evaluation
> methodology of Son et al., *"Exploring High-Bandwidth Flash for Modern LLM Inference: Opportunities
> and Challenges"* (IEEE CAL 2026). Every core claim is backed by a `file:line` reference against
> the current working tree (including its substantial **uncommitted** fairness/correctness fix
> sprint — see `CHANGES.md`/`BUGS.md`), so it can be read alongside the code.
>
> This guide is a companion to `LLMSimulator_Guide_EN.md` (the upstream guide, still accurate for
> the base simulator). Section 16 explicitly re-verifies every quirk from that document's §13
> against this fork and records which are fixed vs. still present. Sections 7–8 and the HBF/
> optimizer content woven through the others are new relative to the upstream guide.

---

## Table of Contents

1. [At a Glance](#1-at-a-glance)
2. [Background: Two Problems Layered Together](#2-background-two-problems-layered-together)
3. [Top-Level Architecture and Object Graph](#3-top-level-architecture-and-object-graph)
4. [End-to-End Execution Flow](#4-end-to-end-execution-flow)
5. [Core Timing Model (Roofline, Inherited)](#5-core-timing-model-roofline-inherited)
6. [The HBF Memory Model (NEW)](#6-the-hbf-memory-model-new)
7. [The Parallelism Optimizer (NEW)](#7-the-parallelism-optimizer-new)
8. [Hardware / Cluster Layer](#8-hardware--cluster-layer)
9. [Model / Module Layer](#9-model--module-layer)
10. [Scheduler Layer (Continuous Batching)](#10-scheduler-layer-continuous-batching)
11. [DRAM / Ramulator / PIM Layer](#11-dram--ramulator--pim-layer)
12. [Configuration Reference (config.yaml)](#12-configuration-reference-configyaml)
13. [Outputs: CSV · stdout markers · experiment_results.md](#13-outputs-csv--stdout-markers--experiment_resultsmd)
14. [Build & Run](#14-build--run)
15. [File Map (Quick Reference)](#15-file-map-quick-reference)
16. [Modeling Simplifications · Known Quirks — Re-Verified Against the Fork](#16-modeling-simplifications--known-quirks--re-verified-against-the-fork)
17. [Appendix: Verification Method](#17-appendix-verification-method)

---

## 1. At a Glance

`HBF-LLMSimulator` still predicts LLM inference latency/energy via the same **roofline
accumulation model** as upstream (§5), but adds an entire hierarchy of flash-timing and
capacity logic (§6) plus an analytic parallelism optimizer that proposes and bounds
configurations before the discrete-event simulator verifies them (§7).

The central research question: **can flash-backed memory (NAND stacks colocated with the HBM
die) substitute for HBM to serve very large LLMs at scale, and under what SLO constraints does it
remain competitive?**

Three things now define the codebase (extending the upstream three):

| Concept | One-line summary | Evidence |
|---|---|---|
| **Roofline + time accumulation** (inherited) | `total_duration = max(compute, memory)`, serially accumulated into `device_time`; sync points reconcile via `max`. | `linear_impl.cpp:74,107`, `module_graph.cpp:231,73-96` |
| **Flash is a 2-tier memory** (new) | Weights + KV-cache live on flash (page-latency + bandwidth); "scarce" activation/staging tiers (1 HBM stack, or 320 MB logic-die SRAM) are separately capacity-gated and separately timed. | `src/dram/hbf_memory_config.h`, `src/hardware/layer_impl.h`, `src/model/footprint.h` |
| **Optimizer proposes, simulator decides** (new) | The analytic optimizer sweeps TP×PP×DP×EP and ranks candidates by an estimated latency; only *capacity/SRAM* infeasibility is a hard veto. SLO pass/fail is decided **exclusively** by the real simulator's measured TPOT. | `src/optimizer/parallelism_optimizer.cpp:440-445`, `INSTRUCTIONS.md` §6 |

Processor-type mapping (unchanged from upstream): `GPU` = xPU, `LOGIC` = Logic-PIM (4× BW, Op/B
8), `PIM` = bank-PIM (16× BW, Op/B 1).

---

## 2. Background: Two Problems Layered Together

**Problem 1 (inherited, Duplex/MICRO'24)**: LLM inference alternates compute-bound prefill
("sum") and memory-bound decode ("gen"); MoE/Attention decode has low Op/B, so routing it to a
Logic-PIM/bank-PIM unit (higher bandwidth, lower peak flops) can beat a GPU.

**Problem 2 (new, HBF/CAL'26)**: Large MoE/dense models increasingly cannot fit in HBM capacity at
low GPU counts. This fork asks whether replacing some/all HBM stacks with flash (NAND on the same
package) can serve these models at acceptable latency, given that flash has:
- **Asymmetric read/write bandwidth** (fast reads, much slower writes — flash wear).
- **A page-read latency floor** (µs-scale, amortized by chunked double-buffering).
- **No safe place for short-lived intermediate data** (writing activations to flash would wear it
  out fast), so activations are diverted to whatever scarce fast tier exists (1 reserved HBM stack,
  or a small logic-die SRAM if there's no HBM at all).

`INSTRUCTIONS.md` is the literal spec this fork implements (5 memory presets, data-placement
policy, disaggregated prefill/decode semantics, KV-write penalty, and the optimizer/simulator
separation of concerns). Everything in §6–§7 below traces back to it.

---

## 3. Top-Level Architecture and Object Graph

Two subsystems are new relative to upstream's five (`scheduler`, `hardware`, `model`, `module`,
`dram`):

```
                 ┌─────────────────────────── eval/test.cpp (main, "run") ───────────────────────────┐
                 │  config.yaml → SystemConfig(+HBFMemoryConfig) + ModelConfig                        │
                 │       → [optional: ParallelismOptimizer::Optimize / analytic_sweep_only]           │
                 │       → Scheduler → Cluster → Model → runIteration                                 │
                 └────────────────────────────────────────────────────────────────────────────────────┘
        ┌──────────────┬──────────────┬───────────────┬───────────────┬───────────────┬───────────────┐
        ▼              ▼              ▼               ▼               ▼               ▼
  ┌───────────┐  ┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────────┐
  │ scheduler │  │   hardware   ││    model     ││    module    ││  optimizer   ││       dram       │
  │           │  │ + layer_impl ││+ footprint.h ││              ││  (NEW)       ││+ hbf_memory_cfg  │
  │           │  │  flash timing││ scarce-tier  ││              ││ TP×PP×DP×EP  ││  (NEW) 5 presets │
  └───────────┘  └──────────────┘└──────────────┘└──────────────┘└──────────────┘└──────────────────┘
```

`SystemConfig` (`hardware_config.h:17-191`) now embeds an `HBFMemoryConfig hbf_config` and a
`bool use_hbf` (`:161-163`); `ParallelConfig` (`parallelism_optimizer.h:10-26`) is the optimizer's
output struct, carrying not just `tp/pp/dp/ep` but predicted footprint fields
(`pred_weight/kv/act/total_bytes`) consumed by a drift harness in `cluster.cpp`/`eval/test.cpp`.

The **two-layer mental model from upstream is unchanged**: the module graph is built once
(symbolic forward pass with worst-case metadata) and re-walked every iteration with the real
batch/seqlen substituted in.

---

## 4. End-to-End Execution Flow

### 4.1 Setup (once) — `eval/test.cpp`

```
Load config.yaml
 ├─ gpu_gen: A100/H100/B100/B200/Rubin → SystemConfig preset               (hardware_config.h:194-395)
 ├─ nvlink_gen/infiniband_gen → override device_ict_*/node_ict_*
 ├─ injection_rate > 0 → use_inject_rate=true, request_per_second=rate     (eval/test.cpp:39,102-104)
 ├─ system.memory_type: HBM|HBM4|HBF|HBF+|CONV|CONV+                       (eval/test.cpp:107-131)
 │   ├─ "HBM" → use_hbf=false (plain lumped-capacity model, upstream behavior)
 │   └─ else  → use_hbf=true; hbf_config = {hbm4,hbf,hbf_plus,conv,conv_plus}_preset
 │              memory_capacity  = hbf_config.total_capacity_bytes         (:127)
 │              memory_bandwidth = flash_read_bandwidth (or hbm_read_bandwidth if no flash stacks) (:128-129)
 ├─ system.chunk_size (bytes; 0=auto) → unit-mistake guard: reject 0<chunk_size<4096  (eval/test.cpp:134-147)
 ├─ [optional] system.analytic_sweep_only: true → run the analytic batch-search and exit (§7.3, :294-398)
 ├─ [optional] optimize_parallelism: true → ParallelismOptimizer::Optimize(...) picks TP/PP/DP/EP
 ├─ Scheduler::Create / Cluster::Create / Model(...) — same as upstream
 ├─ cluster->checkMemorySize() — now routes through footprint.h's scarce-tier logic when use_hbf (§6.3)
 └─ cluster->set_dependency()
```

### 4.2 Simulation Loop — unchanged shape, new content inside kernels

`Cluster::runIteration` still dispatches `runIterationMixed` (colocated) vs.
`runIterationSumGenSplit` (disagg) based on `config.disagg_system` (`cluster.cpp:470-474`), and
still accumulates `total_time += time` each step (`:519`, `:593`). What's new is *what happens
inside* each device's forward pass: linear/attention/activation kernels now consult
`config.hbf_config` via `layer_impl.h` helpers (§6.2) to add flash page-latency and split
weight-vs-activation timing across tiers, and attention kernels additionally accumulate a
**KV-write penalty** (§6.2, "flash writes") that gets folded into the per-step latency.

### 4.3 Wrap-up

Same CSV/gantt export as upstream, plus (when `analytic_sweep_only` is set) a set of
`ANALYTIC_*` stdout markers (§13) consumed by `run_experiments.py`'s two-phase batch search (§7.3).

---

## 5. Core Timing Model (Roofline, Inherited)

The roofline core is **byte-for-byte unchanged**:

```cpp
// linear_impl.cpp:74  (linearCore, shared by GPU/LOGIC/PIM via LinearExecutionGPU etc.)
time_ns compute_duration = total_flops / desc.compute_peak_flops * 1000*1000*1000;
// linear_impl.cpp:107
exec_status.total_duration = std::max(exec_status.compute_duration, exec_status.memory_duration);
```

The only change is **where `memory_duration` comes from**. Previously `bytes/bandwidth`
everywhere; now (`linear_impl.cpp:75-77`):

```cpp
time_ns memory_duration = getLinearMemoryDuration(config, m, k, n, weight->precision_byte,
                                                   total_memory_size, desc.memory_bandwidth,
                                                   num_heads, duplicated_input);
```

`getLinearMemoryDuration` (`layer_impl.h:17-42`) only takes the flash-aware branch when
`config.use_hbf && hbf_config.num_flash_stacks > 0`; otherwise it falls through to the original
`total_memory_size / memory_bandwidth` (`:39-40`) — so plain-`HBM4`/A100-H100-B100-B200 runs are
numerically identical to upstream's model. Full flash-timing details in §6.

Time accumulation (`module_graph.cpp`), the high/low parallel tracks, `sync_devices()`'s
max-barrier, and the executor's try-both-pick-min processor selection (`executor.cpp:135-155`) are
all unchanged from upstream — see the base guide §5.3–§5.4 for full detail. One addition: a
`kv_write_time` accumulator tracks the new KV-write penalty separately from other timing
(`module_graph.cpp:219`).

---

## 6. The HBF Memory Model (NEW)

### 6.1 The Five Presets — `src/dram/hbf_memory_config.h`

`HBFMemoryConfig` (`:5-52`) fields: `num_hbm_stacks`, `num_flash_stacks`, `total_capacity_bytes`
(flash pool), `hbm_read/write_bandwidth`, `flash_read/write_bandwidth`,
`flash_page_read/program_latency_ns`, `sram_per_stack_bytes` (3.13 MB double-buffer staging, all
flash presets), `logic_sram_bytes` (320 MB, "+" presets only), `page_size_bytes` (4096, now mostly
vestigial — see §16 item 13), `hbm_per_stack_bytes`.

| Preset | HBM stacks | Flash stacks | Total capacity | Flash read BW | Page-read latency | Staging/stack | Logic SRAM | HBM/stack |
|---|---|---|---|---|---|---|---|---|
| `hbm4_preset` | 8 | 0 | 288 GB | — | — | — | — | 36 GB |
| `hbf_preset` | 1 | 7 | 3,620 GB | 11.2 TB/s | 1 µs | 3.13 MB | 0 | 36 GB |
| `hbf_plus_preset` | 0 | 8 | 4,096 GB | 12.8 TB/s | 1 µs | 3.13 MB | 320 MB | 0 |
| `conv_preset` | 1 | 7 | 3,620 GB | 2.45 TB/s | 3 µs | 3.13 MB | 0 | 36 GB |
| `conv_plus_preset` | 0 | 8 | 4,096 GB | 2.80 TB/s | 3 µs | 3.13 MB | 320 MB | 0 |

(`hbf_memory_config.h:55-137`.) All flash writes are asymmetric and much slower than reads (e.g.
HBF+: 128 GB/s write vs. 12.8 TB/s read); all presets share a 100 µs page-program latency.
Selected from `config.yaml: system.memory_type` in `eval/test.cpp:107-129` (§4.1).

### 6.2 Flash Timing Kernels — `src/hardware/layer_impl.h`

Three inline helpers, all gated on `config.use_hbf && hbf_config.num_flash_stacks > 0`:

- **`getLinearMemoryDuration`** (`:17-42`) — weight (k×n, possibly ×`num_heads` for batched
  linear) is charged flash-read bandwidth **plus one page-read latency per call**
  (`:26`); activations are charged HBM bandwidth **only if `num_hbm_stacks > 0`** — on HBF+/CONV+
  (`num_hbm_stacks==0`) activation time is **0 (modeled as infinite-bandwidth logic-die SRAM)**
  (`:34-37`). Returns `max(weight_read_time, act_time)` (`:38`). Called from
  `linear_impl.cpp:75-77`.
- **`getAttentionMemoryDuration`** (`:48-89`) — models **double-buffered chunked KV prefetch**:
  when `use_chunked_attention`, the KV read is split into chunks sized by
  `min(chunk_size_override or config.chunk_size, sram_per_stack_bytes × num_flash_stacks)`
  (`:56-64`), and **one page-read latency is paid per chunk, not per byte or per 4 KiB page**
  (`:65-69`) — this is what makes flash's µs-scale latency tolerable at scale. The non-chunked
  fallback branch still double-buffers by SRAM-sized chunks when the read fits in SRAM, and
  degrades to per-page latency only when it doesn't (`:70-80`). Same SRAM-infinite-bandwidth rule
  for activations (`:83-86`). Called from all three `attention_*_impl.cpp` kernels.
- **`getKVWriteDuration`** (`:91-105`) — models the "**Critical Fix**" from `INSTRUCTIONS.md`:
  writing a newly-admitted query's *entire* prefill KV-cache (sized by `input_len`, not the 1
  decode token) to flash, at flash-write bandwidth plus one page-program latency (`:102`).
  Compressed-KV (MLA) and standard-GQA branches both handled (`:97-101`).

**KV-write overlap**: the call site (`attention_gen_impl.cpp:864-865`,
`unhidden_write = std::max(0, kv_write - exec_status.compute_duration)`) only subtracts time
already hidden behind the *attention* kernel's own compute — never FFN — matching
`INSTRUCTIONS.md`'s "subtract any write time overlapped with the attention layer" requirement.
The optimizer's analytic model mirrors this exactly (§7.2).

The **SRAM-infinite-bandwidth assumption is intentional, not a bug**: `activation_impl.cpp:47-50`
implements the identical rule directly (`memory_duration = num_hbm_stacks>0 ? bytes/hbm_bw : 0`).
It reflects that HBF+/CONV+'s 320 MB logic-die SRAM buffer is fast enough, relative to the
per-decode-step activation volume, that its bandwidth is not the bottleneck being modeled — but
(critically) it is still **capacity**-gated, separately, in §6.3.

### 6.3 Scarce-Tier Capacity Model — `src/model/footprint.h`

Shared by the optimizer and `Cluster::checkMemorySize()` (dependency policy: only includes
`hardware_config.h`, to avoid an include cycle with `cluster.h`, `:1-7`).

- **`scarceTierActivationLimit`** (`:37-47`) returns: HBM-stack capacity
  (`num_hbm_stacks × hbm_per_stack_bytes`) if any exist, else `logic_sram_bytes` (HBF+/CONV+), else
  (plain HBM) the full `memory_capacity`.
- **`hasScarceTier`** (`:51-53`): true whenever flash stacks exist.
- **`checkCapacity`** (`:59-97`) — the actual gate: weights + KV vs. `total_capacity_bytes` (flash
  pool, `:66-72`); activations vs. `scarceTierActivationLimit` (`:75-82`, error message names the
  tier — `"HBM"` or `"Logic SRAM"`, `:77`); plain-HBM path uses the old lumped
  `act+weight+kv > memory_capacity` check (`:85-94`).
- **`peakIntermediateBytes`** (`:144-205`) — this is the **peak concurrently-live** intermediate
  footprint, not a sum of every tensor a layer touches. Per `INSTRUCTIONS.md` §2: attention and
  FFN phases execute sequentially within a layer, so the resident set at any instant is
  `max(attention-phase, FFN-phase)`, and this **scales with batch size, not sequence length** —
  the seq-length-scaled Q·Kᵀ score matrix and any decompressed KV are explicitly excluded (they
  stream through the separate 3.13 MB/stack double-buffer of §6.2, never resident in this pool).
  Handles MLA-absorb, compressed-KV, and GQA-base attention variants (`:155-188`) plus MoE/dense
  FFN (`:190-202`).

This gate is why **HBF+/CONV+'s 320 MB logic SRAM can bind batch size much harder than HBF's 36 GB
HBM stack**, even though HBF+ has *more* raw flash capacity and higher bandwidth on every other
axis. This was previously an observed HBF+-worse-than-HBF batch-size inversion (paper claims
HBF+ should be *larger*) — **root-caused and fixed** (`CHANGES.md` item 14): the gate previously
summed every intermediate tensor a layer touches (as if all simultaneously resident) instead of
taking `peakIntermediateBytes`'s `max(attention-phase, FFN-phase)`, and the summed formula's
dominant term was the seq-length-scaled Q·Kᵀ score matrix, which streams through the separate
double-buffer and should never have counted against this pool at all. After the fix, HBF+
correctly exceeds HBF on total batch (see §16 "Open HBF-specific items" for what remains open).

---

## 7. The Parallelism Optimizer (NEW)

### 7.1 What It Sweeps — `src/optimizer/parallelism_optimizer.cpp:Optimize()` (`:8-480`)

Nested nested loop over **TP × PP × DP × EP**:

- **TP** (`:17-26`) doubles from 1; **hard-capped at `num_kv_heads`** — `parallel.cpp` asserts
  `num_kv_heads % parallel_num == 0` and crashes otherwise, so the optimizer unconditionally skips
  `tp` values that don't divide `num_kv_heads` (this guard is itself a fix — see §16 quirk-check
  addendum below and `CHANGES.md` item 3).
- **PP** (`:27-30`): `tp*pp ≤ total_gpus`, evenly divides `total_gpus` and `num_layers`.
- **DP** = `total_gpus/(tp*pp)` (`:32`); requires `batch_size % dp == 0` (`:40`).
- **EP** (`e_tp_dg`, `:42-61`) — an **independent** degree from TP: routed-expert weight can be
  tensor-split across a different device count than attention/dense weight, bounded by
  `devices_per_stage = total_gpus/pp` and mirroring `expert.cpp`'s own validity asserts
  (`:57-60`) so the optimizer never proposes a config the simulator would reject.

Per candidate, it computes: TP-aware attention/MLP weight bytes (`:79-139`, MLA vs. GQA branches),
MoE/dense layer-mix handling for `first_k_dense`/`expert_freq` patterns (`:143-175`), an
`E_active` estimate of concurrently-hot experts per device (`:180-196`), KV-cache bytes
(compressed vs. GQA, `:234-237`), and `peakIntermediateBytes`-based activation size (`:249-259`).

### 7.2 Capacity Gate vs. Analytic Latency — the Separation of Concerns

- **Hard gate**: `checkCapacity(...)` (`:264-269`) — capacity/SRAM is exact and monotonic; it's
  the *only* thing that sets `config.oom`.
- **Ranking-only estimate**: `estimated_latency_ms` (`:271-445`) — weight-read time (flash or HBM
  bandwidth + per-op page-read latency, counting distinct linear-op calls per layer to match the
  live kernel, `:281-315`), compute time from FLOPs (`:317-323`), chunked KV-read time mirroring
  `getAttentionMemoryDuration` (`:325-345`), and KV-write time **hidden behind attention-only
  compute** (`:347-374`, matching `attention_gen_impl.cpp`'s basis exactly — using total per-layer
  compute here would over-credit hiding). Two selectable models via
  `system_config.optimizer_latency_model` (`:377-399`): `"max"` (tighter, `max(compute,
  weight_mem)+kv` per layer) vs. `"sum"` (conservative, additive, with a flash-only DMA/compute
  overlap credit). Communication terms — TP all-reduce (`:401-407`), PP send/receive
  (`:409-420`), MoE scatter/gather (`:422-438`, only when `e_tp_dg < devices_per_stage`) — are
  added on top. The comment at `:440-444` is explicit: *this must never set `config.oom`* — the
  simulator's measured TPOT is the sole SLO arbiter (`INSTRUCTIONS.md` §6).
- Selection (`:458-479`): among non-OOM candidates, pick minimum `estimated_latency_ms`.

Note `tpot_slo_ms` is a parameter of `Optimize()` but **not used inside it** for filtering — SLO
comparison happens in the *caller* (`eval/test.cpp`, §7.3).

### 7.3 Two-Phase Batch-Size Search — `eval/test.cpp: analytic_sweep_only` (`:294-398`)

Implements `INSTRUCTIONS.md` §6's two-phase design:

1. **`ANALYTIC_CAP_FEASIBLE_AT_1`** (`:323-331`) — if batch=1 is capacity-infeasible, the whole
   GPU count is infeasible; print `0` and stop (no simulator call needed — capacity is exact).
2. **Phase 0 (capacity ceiling)**: since capacity is exact and monotonic in batch size, compute
   `b_cap` directly via exponential-then-binary search (`:340-365`) with no per-batch simulator
   calls.
3. **Phase 1 (SLO-guided binary search)**: binary-search *downward* from `b_cap` using the
   analytic latency estimate against `tpot_slo_ms` (`:367-387`) to propose a candidate batch and
   its TP/PP/EP/DP.
4. Emits `ANALYTIC_MAX_BATCH`, `ANALYTIC_TP/PP/EP/DP`, `ANALYTIC_ESTIMATED_LATENCY_MS` (`:389-398`).

`run_experiments.py`'s `find_max_batch_size` then does **Phase 2 (simulator verification)**:
confirms the candidate and the batch immediately above it (boundary safety check) with the real
simulator; if the simulator disagrees with the analytic estimate in either direction, it falls
back to a simulator-driven binary search over the narrowed range. This closes a historical bug
(audit F1, `CHANGES.md` item 1) where a pure analytic-latency rejection at batch=1 was
indistinguishable from genuine capacity infeasibility and could silently return batch=0 without
ever consulting the simulator.

---

## 8. Hardware / Cluster Layer

### 8.1 Topology & GPU Presets

Topology (`Cluster`/`Node`/`Device`, `get_device`, NVLink=intra-node/`device_ict_*`,
InfiniBand=inter-node/`node_ict_*`) is unchanged from upstream. `hardware_config.h` now has
**five** `SystemConfig` presets instead of four: **A100, H100, B100, B200** (unchanged,
`:194-344`) plus **Rubin** (`:359-395`, added this fork to match the HBF paper's cited reference
GPU, NVIDIA DGX Rubin NVL8). Rubin's `compute_peak_flops` is explicitly commented as an
**estimate** (`:346-358`): the datasheet gives only an aggregate FP8/FP6 training figure (140
PFLOPS/8 GPUs), halved per this file's FP16-base convention. Its `memory_bandwidth`/
`memory_capacity` fields are placeholders and **not load-bearing** — `eval/test.cpp:127-129`
always overwrites them from the selected `hbf_config` preset whenever `memory_type` is set (true
in every sweep this project runs).

`Device`'s constructor (`device.cpp:21-75`) applies the same override at construction time
(`:29-34`, `use_hbf && num_flash_stacks>0 → memory_capacity/bandwidth = hbf_config.*`).

**A100 crash still latent** (see §16 item 12): `device.cpp:39-52`'s Ramulator DRAM-config-path
switch has branches for `H100` and `B100/B200/Rubin` only; `A100` falls through to an empty path
and `YAML::LoadFile("")` throws. Dormant because no sweep in this project uses `gpu_gen: A100`.

### 8.2 Communication

AllReduce ring and All-to-All expert scatter/gather (`communication.cpp:30-37,61-428`) are
structurally unchanged from upstream. One correctness fix: every node-index derivation
(`src_node`/`dst_node`) previously hardcoded `/8`; now uses `device->config.num_device`
(`communication.cpp:83,97,104,159,171,269,283,290,352,364,499-500`) — so runs with `num_device !=
8` (e.g. the 16-GPU / 2-node sweep points) compute node membership correctly. NVLink/InfiniBand
bandwidths are set via `nvlink_gen`/`infiniband_gen` in `eval/test.cpp:72-73,84-85`
(`nvlink_gen: 5 → 900 GB/s` unidirectional per-link, matching `INSTRUCTIONS.md`'s 1,800 GB/s
bidirectional-total figure; `infiniband_gen: 800 → 100 GB/s`).

### 8.3 Cluster Execution Loop

`runIteration` dispatches `runIterationMixed` vs. `runIterationSumGenSplit` on
`config.disagg_system` (`cluster.cpp:470-474`); both still accumulate `total_time` unconditionally
per step (`:519`, `:593`, gated on `!hasSumSeq()` for the disagg gen-only timeline). Two capacity
checks:

- **`checkMemorySize`** (F8 fix, `cluster.cpp:262`): `size_for_capacity_gate = hasScarceTier(config)
  ? (size - activation_size) : size` — activation is excluded from the lumped weight+KV pool gate
  whenever a scarce tier exists (it already has its own, correct, separate gate via
  `scarceTierActivationLimit`/§6.3), removing a small double-charge that existed before this fix.
- **`checkHeteroMemorySize`** (`cluster.cpp:333-334`) — a *separate*, unaudited capacity check with
  a hardcoded `3.3 GB` "Non MoE weight" magic-number subtraction and no activation term at all;
  flagged in `BUGS.md` #7 as not confirmed broken but inconsistent with the F8 fix.

The KV-write penalty (§6.2) is folded into the breakdown without double-counting: attention
duration is split as `atten_sum/gen += (total - kv_write); kv_write += kv_write` at the relevant
stamp-processing sites in `cluster.cpp`.

Attention timing kernels (`AttentionSum`/`AttentionGen`/`AttentionMixed`, MLA/Absorb variants) are
structurally the same three-regime split as upstream — see the base guide §6.5 — with the flash
calls into `layer_impl.h` (§6.2) as the only change, plus the still-present PIM
`memory×opb` non-roofline quirk (§16 item 9).

---

## 9. Model / Module Layer

Graph assembly (`LLM` = Embedding → N×Decoder/MoEDecoder → LmHead), the `Tensor` abstraction, and
attention geometry (MHA/GQA/MQA/MLA, `attention_group_size = num_heads/num_kv_heads`) are
structurally the same as upstream, with three fixes now in place (fully re-verified in §16):

- **`isMoELayer`** helper (`llm.cpp:16-21`) generalizes the old hardcoded `expert_freq`/
  DeepSeek-specific branch: honors `first_k_dense` (forced-dense prefix, any model) **and**
  `expert_freq`, so MLA-specific/DeepSeek-specific handling is derived from the model's own
  architecture parameters rather than a global flag (`INSTRUCTIONS.md`'s explicit requirement).
- **Decoder chaining fixed**: `LLM::forward` now threads `temp = out` between decoders
  (`llm.cpp:114-115`) instead of feeding every decoder the raw embedding output.
- **Decoder residual wiring fixed**: `post_attn_layer_norm` now consumes `res_1_out` (the actual
  residual sum), not raw `attention_out` (`decoder.cpp:82-86`); `MoEDecoder`'s residual slots are
  now real `Residual` modules, not `LayerNorm` (`decoder.cpp:129-144`).

Model presets (`model_config.h`): `llama3_405B` precision corrected **FP8 → BF16**
(`precision_byte=2`, `:116-118`) after cross-referencing the paper's explicit "HBM4 needs ≥4 GPUs
for both LLMs" constraint (`CHANGES.md` item 11); `llama4_maverick` (`:124-126`, already BF16,
128 experts) is the fork's other primary evaluation model. `deepseekV3`, `mixtral`, and other
presets' precision assumptions are **unverified** against any external reference (`BUGS.md` #5).

---

## 10. Scheduler Layer (Continuous Batching)

Sequence lifecycle, `BatchedSequence`, and expert routing (uniform/Zipfian) are unchanged from
upstream. One real behavioral change: **`injection_rate` is no longer a dead key**.
`eval/test.cpp:39,102-104` now reads `simulation.injection_rate` and, when `> 0`, sets
`use_inject_rate=true` + `request_per_second`, activating the open-loop Poisson-arrival path
(`scheduler.cpp:373,457`). `config.yaml` currently ships `injection_rate: 20` — **this means the
system now runs open-loop by default**, a behavioral change from upstream's always-closed-loop
default. See §16 items 3–4 for the interaction this creates.

---

## 11. DRAM / Ramulator / PIM Layer

Unchanged from upstream (still gated behind `use_ramulator: off` by default; every sweep in this
project runs the pure analytical path described in §5–§6, not real Ramulator cycles). See the
base guide §9 for the full Ramulator/PIM-kernel/address-mapping description — it applies
identically here.

---

## 12. Configuration Reference (config.yaml)

New/changed keys relative to upstream (current values shown are what `config.yaml` ships):

| Section | Key | Meaning | Current value |
|---|---|---|---|
| system | `gpu_gen` | now includes `Rubin` | `Rubin` |
| system | `memory_type` | `HBM` (plain, upstream-compatible) / `HBM4` / `HBF` / `HBF+` / `CONV` / `CONV+` | `HBM` |
| system | `chunk_size` | chunked-attention chunk size in **bytes**; `0`=auto (full SRAM staging capacity) | `0` |
| system | `tpot_slo` | target TPOT in **seconds** (converted ×1000 internally to `tpot_slo_ms`, `eval/test.cpp:302`) | `0.1` |
| system | `optimize_parallelism` | run `ParallelismOptimizer::Optimize` to pick TP/PP/DP/EP instead of `distribution.*` | `false` |
| system.distribution | `expert_tensor_degree` / `none_expert_tensor_degree` | `e_tp_dg` / `ne_tp_dg` (unchanged) | `1` / `8` |
| simulation | `precision_byte` | `0` = use model preset's own precision | `0` |
| simulation | `injection_rate` | **now live** (was dead upstream); `>0` enables open-loop Poisson arrivals | `20` |
| simulation | `validate_optimizer` | `off`/`warn`/`strict` — cross-checks optimizer prediction vs. live simulator footprint | `warn` |
| simulation | `validate_optimizer_threshold` | relative divergence threshold for the above | `0.10` |
| simulation | `optimizer_latency_model` | `"max"` (tighter, tracks simulator overlap) or `"sum"` (conservative additive) — §7.2 | `max` |
| simulation | `latency_margin` | multiplicative safety margin on the optimizer's estimated latency | `1.0` |

All other keys (`num_node`/`num_device`/`processor_type`, `optimization.*`, `serving.*`,
`log.*`) are unchanged from upstream — see the base guide §10 for the full reference.

---

## 13. Outputs: CSV · stdout markers · experiment_results.md

CSV columns, `type` values, and Gantt/Timeboard export are unchanged from upstream (base guide
§11) — one addition: the per-stage breakdown now separately reports the KV-write penalty
(accumulated via `module_graph.cpp:219`'s `kv_write_time`) rather than folding it silently into
attention time.

**New stdout markers** (only emitted under `system.analytic_sweep_only: true`, §7.3):
`ANALYTIC_CAP_FEASIBLE_AT_1`, `ANALYTIC_CAP_BATCH`, `ANALYTIC_MAX_BATCH`,
`ANALYTIC_TP/PP/EP/DP`, `ANALYTIC_ESTIMATED_LATENCY_MS` (`eval/test.cpp:323-398`). Also
`PEC_KV_BYTES_PER_TOKEN`/`PEC_FLASH_CAPACITY_BYTES` markers (parsed by `run_experiments.py`) for
the write-endurance metric.

`run_experiments.py`'s `main()` writes **five paper-defined metrics** to `experiment_results.md`:
(1) Maximum Per-GPU Batch Size (`max_batch/gpu_count` — total GPU count, **not** DP replica
count; `CHANGES.md` item 15 corrects an earlier "P1b" fix that divided by `dp` instead, which
penalized configs for using DP *more* effectively since a DP replica still consumes real GPU
hardware — matches `INSTRUCTIONS.md`'s own TPS formula, which was already implemented this way),
(2) System Throughput, (3) Runtime Performance Breakdown (restricted to `{4,8,16}` GPUs — a P2
fix, matching the paper's actual figure scope), (4) SLO Sensitivity (also `{4,8,16}`), (5) Write
Traffic / 3-Year PEC vs. the 100K SLC cycle limit.

---

## 14. Build & Run

Build steps are unchanged from the base guide §12 (Ramulator 2.0 submodule + PIM patch, cmake,
`make -j`, output `build/run`). Two sweep drivers now exist:

- **`run_experiments.py`** — the maintained, fixed driver. Full sweep: GPU counts `{1,2,4,8,16}` ×
  presets `{HBM4,HBF,HBF+,CONV,CONV+}` × workloads `{SHORT,MID,LONG}` × SLOs
  `{0.05,0.1,0.2,offline}`. Implements the two-phase batch search (§7.3) with capacity/latency
  separation (F1) and per-GPU batch reporting (P1b). Baseline = 8-GPU HBM4.
- **`run_flash_only.py`** — an older, narrower driver (`gpus=[8]` only, different workload
  lengths, flash presets only) that predates and was **not updated with** any of this session's
  fixes: no analytic phase, no capacity/latency separation, reports raw (not per-GPU) batch size.
  **Do not trust its output for reported numbers without auditing it first** (`BUGS.md` #4).

```bash
./run config.yaml > output.log                       # single run
conda run -n fluidlab python3 run_experiments.py      # full fixed sweep → experiment_results.md
conda run -n fluidlab python3 run_flash_only.py        # older, unfixed fast check — audit before trusting
```

---

## 15. File Map (Quick Reference)

New/changed files relative to the upstream file map (base guide §14):

| File | Role |
|---|---|
| `src/dram/hbf_memory_config.h` | `HBFMemoryConfig` struct + 5 memory presets (§6.1) |
| `src/hardware/layer_impl.h` | Flash timing helpers: `getLinearMemoryDuration`, `getAttentionMemoryDuration`, `getKVWriteDuration` (§6.2) |
| `src/model/footprint.h` | Shared scarce-tier capacity/footprint logic: `checkCapacity`, `scarceTierActivationLimit`, `peakIntermediateBytes` (§6.3) |
| `src/optimizer/parallelism_optimizer.{h,cpp}` | `ParallelismOptimizer::Optimize` — TP×PP×DP×EP sweep (§7) |
| `eval/test.cpp` | Entry point; now also hosts `analytic_sweep_only` mode and memory-preset/chunk-size parsing (§4.1, §7.3) |
| `run_experiments.py` | Maintained, fixed sweep driver (§14) |
| `run_flash_only.py` | Older, unfixed sweep driver — audit before use (§14) |
| `INSTRUCTIONS.md` | The literal HBF research spec this fork implements |
| `CHANGES.md` | This session's fixes, with rationale, cross-referenced against the paper |
| `BUGS.md` | Known unfixed/dormant bugs and open investigations |
| `FAIRNESS_AUDIT.md` | External audit that seeded the `CHANGES.md` fix sprint (with corrections noted in place) |

All other files match the upstream file map exactly (base guide §14).

---

## 16. Modeling Simplifications · Known Quirks — Re-Verified Against the Fork

Every quirk from `LLMSimulator_Guide_EN.md` §13 was re-checked directly against this fork's
current (including uncommitted) code.

| # | Upstream quirk | Verdict here | Evidence |
|---|---|---|---|
| 1 | "mixed" not truly mixed — any pending prefill forces the whole step to prefill, decode gets 0 tokens | **STILL PRESENT** | `scheduler.cpp:267-275` (`hasSumSeq() → process_gen=false`), `sequence.cpp:45-46` |
| 2 | Synthetic request-length jitter computed but never applied | **STILL PRESENT** | `scheduler.cpp:52-62` (`delta` computed, apply lines commented out) |
| 3 | `injection_rate` config key dead; `use_inject_rate` never enabled → always closed-loop | **FIXED** | `eval/test.cpp:39,102-104` now reads it; `config.yaml:46` ships `injection_rate: 20` |
| 4 | Inter-arrival times drawn from `poisson_distribution` (should be exponential for a Poisson *process*) | **STILL PRESENT — and now live** | `scheduler.cpp:492-494`. Because #3 is fixed and the shipped config has `injection_rate: 20 > 0`, this bug is now on the **active default path**, not dormant as it was upstream |
| 5 | Decoder residual wiring: `post_attn_layer_norm` received `attention_out` instead of `res_1_out` | **FIXED** | `decoder.cpp:80-86` — comment at `:83-85` documents the correction |
| 6 | `MoEDecoder`'s residual slots were actually `LayerNorm`, not `Residual` | **FIXED** | `decoder.cpp:129-144` now uses `Residual::Create` for both slots |
| 7 | `LLM::forward` fed every decoder the raw embedding output (no chaining) | **FIXED** | `llm.cpp:108-116` — `temp = out` threads the previous decoder's output forward |
| 8 | `attention_group_size` inconsistent: gen used `num_heads/num_kv_heads`, sum/mixed/MLA-sum used `head_dim/num_heads` | **FIXED** | `attention.cpp:57,108,151,415,547` — all five sites now `num_heads/num_kv_heads` |
| 9 | PIM decode-attention uses `memory_duration × opb` instead of `max(compute, memory)` | **STILL PRESENT** | `attention_gen_impl.cpp:449,499` (contrast with `:647,719,848,...` which correctly use `std::max`) |
| 10 | Dead code: `data_object.*`, `pimBankgroupEnergy`, most PIM kernels (CKKS legacy), `getNumInjection`/`getPoissondistribution` (no callers) | **STILL PRESENT** | `power.h:40`, `pim_kernel.h` CKKS declarations, `scheduler.cpp:187,195` (no call sites found) |
| 11 | Memory-limited batch shrinking (`mem_cap_limit`) happens once at startup, not dynamically at runtime | **STILL PRESENT** | `cluster.cpp:262-267` (invoked once, `model/test.cpp:36`) |
| 12 | `gpu_gen: A100` has no Ramulator DRAM-config branch in `device.cpp` → would crash | **STILL PRESENT** (Rubin branch added alongside it) | `device.cpp:39-52` — H100/B100/B200/Rubin branches exist, no A100 branch |

**Net effect of the #3/#4 interaction**: fixing the dead `injection_rate` key (a real
correctness improvement — it makes a documented config key actually work) exposes the
long-dormant Poisson-vs-exponential approximation on the default path for the first time. Anyone
running QPS-style experiments (rather than the closed-loop batch-saturation sweeps this project's
own harnesses use) should be aware inter-arrival times are approximately, not exactly,
Poisson-process-distributed.

### Open, HBF-specific items (from `BUGS.md`, not yet resolved)

These are new to this fork and have no upstream analogue:

- **[RESOLVED] HBF+ underperforming HBF on total achievable batch size** — the ~8-9x
  HBM4/SHORT/8-GPU batch-size gap vs. the paper's reported numbers, and the HBF+-worse-than-HBF
  inversion, are now both resolved (`CHANGES.md` items 14-15, `BUGS.md` #9). Root causes: the
  scarce-tier gate summed instead of taking peak intermediate-data footprint (§6.3), and
  `run_experiments.py`'s per-GPU batch divisor used DP replica count instead of total GPU count.
  Recomputing already-completed simulator runs with both fixes matches the paper's own anchors
  closely (e.g. HBM4/8-GPU/SHORT llama3_405B: 194.4 vs. paper's 194; HBF+/HBF ratio at MID/8-GPU:
  +25.6% vs. paper's "+24% on average"). GPU-speed mismatch, TP/DP KV-sharding bugs, missing
  communication-overhead accounting, and (per a follow-up investigation) the `injection_rate`/
  Poisson-open-loop quirk (#3/#4 above) were all investigated and ruled out along the way — the
  last of these is neutralized for every `run_experiments.py`-driven number by the explicit
  `injection_rate = 0` override at `run_experiments.py:57` ("Continuous batching"); it only
  affects a bare `./run config.yaml` invocation, never a sweep-driven one.
- **Two smaller residuals remain** after the above fixes (`BUGS.md` #10), not yet root-caused:
  llama4_maverick's HBM4/8-GPU/SHORT anchor is ~11% high (511.6 vs. paper's 460, vs. llama3_405B's
  near-exact match) — possibly a MoE-specific routing/footprint assumption; and the "1-GPU
  HBF/HBF+ beats 8-GPU HBM4 in most cases" claim now holds/is-close in 3 of 4 tested combinations
  but llama3_405B's plain-HBF (not HBF+) case undershoots by ~61%, not yet investigated.
- **Degenerate one-device-per-pipeline-stage optimizer choices** (e.g. PP=8/EP=1/TP=1) trivially
  zero every communication term by construction — legitimate per the ranking, but not yet verified
  whether the per-device all-experts-unsharded weight cost is fully/correctly counted against a
  communication-incurring alternative (`BUGS.md` #8).
- **`run_flash_only.py` carries none of this session's fixes** (§14) — should not be used for
  reported numbers without auditing or retiring it (`BUGS.md` #4).
- **`page_size_bytes` is now vestigial config surface**: after the chunked/SRAM-staging fix (§6.2),
  nothing computes with the literal 4 KiB page size anymore except as a documentation/hardware-
  geometry field (`BUGS.md` #6).
- **Disaggregated path (`disagg_system=on`, currently unused by any sweep)** lost its one-time
  prefill→decode KV-transfer term when a double-counting/never-actually-modeled block was removed;
  the correct per-decode-step KV-write penalty (§6.2) is unaffected and present on both execution
  paths (`BUGS.md` #3).
- **`checkHeteroMemorySize()`** has its own capacity logic (hardcoded 3.3 GB subtraction, no
  activation term) and was not audited or reconciled with the F8 `checkMemorySize` fix (§8.3,
  `BUGS.md` #7).

---

## 17. Appendix: Verification Method

Every `file:line` reference in this guide was directly read from the current working tree
(`git status` at the time of writing showed 15 modified tracked files plus several new untracked
docs — this guide describes that *uncommitted* state, not a prior commit). Three parallel
subsystem sweeps (HBF-specific additions; inherited-core-mechanism confirmation + Python harness;
quirk-by-quirk re-verification against `LLMSimulator_Guide_EN.md` §13) were cross-checked against
each other and against a second, manual read-through of the highest-traffic files
(`layer_impl.h`, `footprint.h`, `hbf_memory_config.h`, `parallelism_optimizer.cpp`,
`decoder.cpp`, `llm.cpp`, `attention.cpp`, `device.cpp`, `hardware_config.h`, `scheduler.cpp`,
`config.yaml`) before this document was written, to catch any line-number drift between the two
passes. `CHANGES.md`, `BUGS.md`, and `INSTRUCTIONS.md` were used as corroborating (not
primary) sources — every claim sourced from them was cross-checked against the actual code they
describe.
