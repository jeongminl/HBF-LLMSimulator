# HBF-LLMSimulator Structure & Functionality Guide

> This document describes the **current** structure and behavior of the `HBF-LLMSimulator`
> codebase — a fork of `LLMSimulator` (the cycle-approximate simulator from MICRO 2024's
> *"Duplex"* paper) extended with **High-Bandwidth Flash (HBF)** memory modeling and a
> constraint-aware **parallelism optimizer**, implementing the evaluation methodology of Son et
> al., *"Exploring High-Bandwidth Flash for Modern LLM Inference: Opportunities and Challenges"*
> (IEEE CAL 2026; the PDF is in this repo and is the ground-truth spec). This guide describes
> what the simulator does today, not how it got there — for the history of bugs found and fixed,
> see `CHANGES.md`; for currently-known bugs and open paper-comparison gaps, see `BUGS.md` and
> `PAPER_INCONSISTENCIES.md`. Raw investigation/adjudication trails (superseded by the docs above,
> kept for full derivations) live under `ledgers/`.
>
> This guide is a companion to `LLMSimulator_Guide_EN.md` (the upstream guide, still accurate for
> the base simulator's unmodified subsystems). Sections 6–7 (HBF memory model, parallelism
> optimizer) are entirely new relative to the upstream guide; the rest of this document notes
> where behavior has changed from upstream and where it hasn't.

---

## Table of Contents

1. [At a Glance](#1-at-a-glance)
2. [Background](#2-background)
3. [Top-Level Architecture and Object Graph](#3-top-level-architecture-and-object-graph)
4. [End-to-End Execution Flow](#4-end-to-end-execution-flow)
5. [Core Timing Model (Roofline)](#5-core-timing-model-roofline)
6. [The HBF Memory Model](#6-the-hbf-memory-model)
7. [The Parallelism Optimizer](#7-the-parallelism-optimizer)
8. [Hardware / Cluster Layer](#8-hardware--cluster-layer)
9. [Model / Module Layer](#9-model--module-layer)
10. [Scheduler Layer (Continuous Batching)](#10-scheduler-layer-continuous-batching)
11. [DRAM / Ramulator / PIM Layer](#11-dram--ramulator--pim-layer)
12. [Configuration Reference (config.yaml)](#12-configuration-reference-configyaml)
13. [Outputs: CSV, stdout markers, experiment_results.md](#13-outputs-csv-stdout-markers-experiment_resultsmd)
14. [Build & Run](#14-build--run)
15. [File Map](#15-file-map)
16. [Known Modeling Simplifications](#16-known-modeling-simplifications)

---

## 1. At a Glance

The simulator predicts LLM inference latency via a **roofline accumulation model**
(`total_duration = max(compute, memory)`, accumulated serially into each device's `device_time`),
inherited from upstream, plus a hierarchy of flash-timing and capacity logic, plus an analytic
parallelism optimizer that proposes and bounds configurations before the discrete-event simulator
verifies them.

The central research question: **can flash-backed memory (NAND stacks colocated with the HBM
die) substitute for HBM to serve very large LLMs at scale, and under what SLO constraints does it
remain competitive?**

| Concept | Summary | Where |
|---|---|---|
| **Roofline + time accumulation** | `total_duration = max(compute, memory)`, serially accumulated into `device_time`; cross-device sync points reconcile via `max`. | `linear_impl.cpp`, `module_graph.cpp` |
| **Flash is a 2-tier memory** | Weights + KV-cache live on flash (page-latency + bandwidth); "scarce" activation/staging tiers (1 HBM stack, or 320 MB logic-die SRAM) are separately capacity-gated and separately timed. | `src/dram/hbf_memory_config.h`, `src/hardware/layer_impl.h`, `src/model/footprint.h` |
| **Optimizer proposes, simulator decides** | The analytic optimizer sweeps TP×PP×DP×EP and ranks capacity-feasible candidates by estimated throughput. Only capacity/SRAM infeasibility is a hard veto. SLO pass/fail is decided **exclusively** by the real simulator's measured TPOT. | `src/optimizer/parallelism_optimizer.cpp` |
| **Pipeline-latency propagation** | A decode token's true per-step latency is the full sequential traversal of all `PP` pipeline stages; the cluster reads the slowest-finishing device's clock (`Cluster::maxDeviceTime()`), not any single stage's local time. | `src/hardware/cluster.{h,cpp}`, `src/module/communication.cpp` |
| **Compute-utilization (MFU) derating** | Every `compute_duration = flops/compute_peak_flops` site is divided by an additional saturating-curve factor `effectiveMFU(config, m)`. Defaults to an exact no-op; only active if `config.yaml` sets `simulation.mfu_max`/`mfu_m_half`. | `src/hardware/hardware_config.h` |

Processor-type mapping (unchanged from upstream): `GPU` = xPU, `LOGIC` = Logic-PIM (4× BW, Op/B
8), `PIM` = bank-PIM (16× BW, Op/B 1).

---

## 2. Background

**Problem 1 (inherited, Duplex/MICRO'24)**: LLM inference alternates compute-bound prefill
("sum") and memory-bound decode ("gen"); MoE/Attention decode has low Op/B, so routing it to a
Logic-PIM/bank-PIM unit (higher bandwidth, lower peak flops) can beat a GPU.

**Problem 2 (HBF/CAL'26)**: Large MoE/dense models increasingly cannot fit in HBM capacity at low
GPU counts. This simulator evaluates whether replacing some/all HBM stacks with flash (NAND on
the same package) can serve these models at acceptable latency, given that flash has:
- **Asymmetric read/write bandwidth** (fast reads, much slower writes — flash wear).
- **A page-read latency floor** (µs-scale, amortized by chunked double-buffering).
- **No safe place for short-lived intermediate data** (writing activations to flash would wear it
  out fast), so activations are diverted to whatever scarce fast tier exists (1 reserved HBM
  stack, or a small logic-die SRAM if there's no HBM at all).

The source paper is the literal spec this simulator implements (5 memory presets, data-placement
policy, disaggregated prefill/decode semantics, KV-write penalty, and the optimizer/simulator
separation of concerns). §6–§7 below trace back to it directly.

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
  │           │  │ + layer_impl ││+ footprint.h ││              ││              ││+ hbf_memory_cfg  │
  │           │  │  flash timing││ scarce-tier  ││              ││ TP×PP×DP×EP  ││  5 presets       │
  └───────────┘  └──────────────┘└──────────────┘└──────────────┘└──────────────┘└──────────────────┘
```

`SystemConfig` (`src/hardware/hardware_config.h`) embeds an `HBFMemoryConfig hbf_config` and a
`bool use_hbf`. `ParallelConfig` (`src/optimizer/parallelism_optimizer.h`) is the optimizer's
output struct, carrying `tp/pp/dp/ep` plus predicted footprint fields
(`pred_weight/kv/act/total_bytes`) consumed by a drift-detection harness (`validate_optimizer` in
`config.yaml`, cross-checking the optimizer's prediction against the live simulator's footprint).

The two-layer mental model from upstream is unchanged: the module graph is built once (symbolic
forward pass with worst-case metadata) and re-walked every iteration with the real batch/seqlen
substituted in.

---

## 4. End-to-End Execution Flow

### 4.1 Setup (once) — `eval/test.cpp`

```
Load config.yaml
 ├─ gpu_gen: A100/H100/B100/B200/Rubin → SystemConfig preset
 ├─ nvlink_gen/infiniband_gen → override device_ict_*/node_ict_*
 ├─ injection_rate > 0 → use_inject_rate=true, request_per_second=rate (open-loop Poisson arrivals)
 ├─ system.memory_type: HBM|HBM4|HBF|HBF+|CONV|CONV+
 │   ├─ "HBM" → use_hbf=false (plain lumped-capacity model, upstream behavior)
 │   └─ else  → use_hbf=true; hbf_config = one of the five presets (§6.1)
 │              memory_capacity  = hbf_config.total_capacity_bytes
 │              memory_bandwidth = flash_read_bandwidth (or hbm_read_bandwidth if no flash stacks)
 ├─ system.chunk_size (bytes; 0=auto) → unit-mistake guard: reject 0<chunk_size<4096
 ├─ [optional] system.analytic_sweep_only: true → run the analytic batch-search and exit (§7.3)
 ├─ [optional] optimize_parallelism: true → ParallelismOptimizer::Optimize(...) picks TP/PP/DP/EP
 ├─ Scheduler::Create / Cluster::Create / Model(...) — same as upstream
 ├─ cluster->checkMemorySize() — routes through footprint.h's scarce-tier logic when use_hbf (§6.3)
 └─ cluster->set_dependency()
```

### 4.2 Simulation Loop

`Cluster::runIteration` dispatches `runIterationMixed` (colocated) vs. `runIterationSumGenSplit`
(disagg) based on `config.disagg_system`, and accumulates each step's per-token latency into
`total_time`. Inside each device's forward pass: linear/attention/activation kernels consult
`config.hbf_config` via `layer_impl.h` helpers (§6.2) to add flash page-latency and split
weight-vs-activation timing across tiers; attention kernels additionally accumulate a **KV-write
penalty** (§6.2) that gets folded into the per-step latency. For pipeline-parallel configs
(`PP>1`), each stage's elapsed time is propagated forward to the next stage's device before the
step's true latency is read back out (§8.3) — a decode token's real per-step latency is the full
sequential traversal of all `PP` stages, not any one stage's local compute alone.

### 4.3 Wrap-up

Same CSV/gantt export as upstream, plus (when `analytic_sweep_only` is set) a set of
`ANALYTIC_*` stdout markers (§13) consumed by `run_experiments.py`'s two-phase batch search (§7.3).

---

## 5. Core Timing Model (Roofline)

The roofline core is inherited unchanged:

```cpp
// linear_impl.cpp (linearCore, shared by GPU/LOGIC/PIM via LinearExecutionGPU etc.)
time_ns compute_duration = total_flops / (desc.compute_peak_flops * effectiveMFU(config, m)) * 1e9;
exec_status.total_duration = std::max(exec_status.compute_duration, exec_status.memory_duration);
```

The only structural change is **where `memory_duration` comes from**. It's computed by
`getLinearMemoryDuration` (`layer_impl.h`), which only takes the flash-aware branch when
`config.use_hbf && hbf_config.num_flash_stacks > 0`; otherwise it falls through to the original
`total_memory_size / memory_bandwidth` — so plain-`HBM4`/A100-H100-B100-B200 runs are numerically
identical to upstream's model. Full flash-timing details in §6.

Time accumulation (`module_graph.cpp`), the high/low parallel tracks, `sync_devices()`'s
max-barrier, and the executor's try-both-pick-min processor selection are unchanged from
upstream — see the base guide for full detail. One addition: a `kv_write_time` accumulator tracks
the KV-write penalty separately from other timing (`module_graph.cpp`).

**Compute-utilization (MFU) derating.** The roofline `compute_duration` formula assumes every
compute-bound op hits 100% of `compute_peak_flops`, which no real GEMM does (tensor-core
tile/wave quantization, epilogue/reduction overhead). `SystemConfig` carries two fields, `mfu_max`
(default `1.0`) and `mfu_m_half` (default `0.0`), plus a free function `effectiveMFU(config, m)`
(`hardware_config.h`) implementing a saturating curve in the GEMM row count `m` (batch×tokens for
that specific op): `MFU(M) = mfu_max * M / (M + mfu_m_half)`. Every `compute_duration` call site
divides `compute_peak_flops` by this factor: linear, activation, all attention-kernel compute
sites, and the optimizer's mirror (§7.2). The defaults make `MFU(M) ≡ 1.0` for every `M > 0`, so
this is an exact no-op unless `config.yaml` explicitly sets `simulation.mfu_max`/`mfu_m_half`
(unset by default — see §12).

---

## 6. The HBF Memory Model

### 6.1 The Five Presets — `src/dram/hbf_memory_config.h`

`HBFMemoryConfig` fields: `num_hbm_stacks`, `num_flash_stacks`, `total_capacity_bytes` (flash
pool), `hbm_read/write_bandwidth`, `flash_read/write_bandwidth`,
`flash_page_read/program_latency_ns`, `sram_per_stack_bytes` (3.13 MB double-buffer staging, all
flash presets), `logic_sram_bytes` (320 MB, "+" presets only), `page_size_bytes` (4096 — hardware
geometry documentation only; no timing formula currently computes with it), `hbm_per_stack_bytes`.

| Preset | HBM stacks | Flash stacks | Total capacity | Flash read BW | Page-read latency | Staging/stack | Logic SRAM | HBM/stack |
|---|---|---|---|---|---|---|---|---|
| `hbm4_preset` | 8 | 0 | 288 GB | — | — | — | — | 36 GB |
| `hbf_preset` | 1 | 7 | 3,620 GB | 11.2 TB/s | 1 µs | 3.13 MB | 0 | 36 GB |
| `hbf_plus_preset` | 0 | 8 | 4,096 GB | 12.8 TB/s | 1 µs | 3.13 MB | 320 MB | 0 |
| `conv_preset` | 1 | 7 | 3,620 GB | 2.45 TB/s | 3 µs | 3.13 MB | 0 | 36 GB |
| `conv_plus_preset` | 0 | 8 | 4,096 GB | 2.80 TB/s | 3 µs | 3.13 MB | 320 MB | 0 |

All flash writes are asymmetric and much slower than reads (e.g. HBF+: 128 GB/s write vs. 12.8
TB/s read); all presets share a 100 µs page-program latency. Selected from `config.yaml:
system.memory_type` in `eval/test.cpp`.

### 6.2 Flash Timing Kernels — `src/hardware/layer_impl.h`

Three inline helpers, all gated on `config.use_hbf && hbf_config.num_flash_stacks > 0`:

- **`getLinearMemoryDuration`** — weight (k×n, possibly ×`num_heads` for batched linear) reads
  stage through the same per-stack SRAM double-buffer as KV reads (below). Activations are
  charged HBM bandwidth **only if `num_hbm_stacks > 0`** — on HBF+/CONV+ (`num_hbm_stacks==0`)
  activation time is **0** (modeled as effectively-infinite-bandwidth logic-die SRAM — see the
  note at the end of this section). Returns `max(weight_read_time, act_time)`. Called from
  `linear_impl.cpp`.
- **`getAttentionMemoryDuration`** — models double-buffered chunked KV prefetch: the KV read is
  split into chunks sized by `min(chunk_size_override or config.chunk_size, sram_per_stack_bytes ×
  num_flash_stacks)`. Double-buffering hides all but one page-read latency: while chunk *N*
  streams out of one SRAM buffer, chunk *N+1*'s page read is issued into the other buffer and its
  latency overlaps chunk *N*'s transfer — hidden for every chunk except the first (pipeline
  fill), which always exposes one full page latency; if a chunk's transfer time is shorter than
  the page latency, the residual `(latency − transfer)` per chunk is still exposed on top. Net:
  `kv_read_time = kv_read_size/flash_bandwidth + page_latency + (num_chunks-1)·max(0, page_latency
  − chunk_transfer_time)` — under every currently-shipped preset this reduces to one exposed page
  latency total, not one per chunk. The non-chunked fallback branch double-buffers by SRAM-sized
  chunks identically. Same SRAM-infinite-bandwidth rule for activations. Called from all three
  `attention_*_impl.cpp` kernels; the identical exposed-latency formula is mirrored in the
  optimizer's `kv_read_time` (§7.2) so the two stay in lock-step.
- **`getKVWriteDuration`** — models writing a newly-admitted query's prefill KV-cache to flash, at
  flash-write bandwidth plus one page-program latency. Compressed-KV (MLA) and standard-GQA
  branches both handled. **Window-aware for Llama-4's interleaved local/global ("iRoPE")
  attention** (§9.1): takes an optional `local_attention_window` parameter — for a local
  (non-global) attention layer, the write length is capped at `min(input_len, window)`, matching
  the fact that a local layer never retains more KV than its window regardless of how long the
  original prompt was. This keeps the write path consistent with the KV-read and KV-capacity
  paths, which already cap at the same window. A window of `0` (every non-iRoPE model, and global
  layers) is a no-op — the full `input_len` is charged, matching the pre-windowing behavior
  exactly.

**KV-write overlap.** Each attention kernel's call site computes
`unhidden_write = max(0, kv_write - exec_status.compute_duration)` — only time already hidden
behind the *attention* kernel's own compute counts as overlapped, never FFN. The optimizer's
analytic model mirrors this exactly (§7.2).

**The SRAM-infinite-bandwidth assumption is intentional, not a bug**: `activation_impl.cpp`
implements the identical rule directly (`memory_duration = num_hbm_stacks>0 ? bytes/hbm_bw : 0`).
It reflects that HBF+/CONV+'s 320 MB logic-die SRAM buffer is fast enough, relative to the
per-decode-step activation volume, that its bandwidth is not the bottleneck being modeled — but it
is still **capacity**-gated, separately, in §6.3.

### 6.3 Scarce-Tier Capacity Model — `src/model/footprint.h`

Shared by the optimizer and `Cluster::checkMemorySize()`.

- **`scarceTierActivationLimit`** returns: HBM-stack capacity (`num_hbm_stacks ×
  hbm_per_stack_bytes`) if any exist, else `logic_sram_bytes` (HBF+/CONV+), else (plain HBM) the
  full `memory_capacity`.
- **`hasScarceTier`**: true whenever flash stacks exist.
- **`checkCapacity`** — the actual gate: weights + KV vs. `total_capacity_bytes` (flash pool);
  activations vs. `scarceTierActivationLimit` (error message names the tier — `"HBM"` or `"Logic
  SRAM"`); plain-HBM path uses the old lumped `act+weight+kv > memory_capacity` check.
- **`peakIntermediateBytes`** — the **peak concurrently-live** intermediate footprint, not a sum
  of every tensor a layer touches. The paper is explicit that attention and FFN phases execute
  sequentially within a layer, so the resident set at any instant is `max(attention-phase,
  FFN-phase)`, and this scales with batch size, not sequence length — the seq-length-scaled Q·Kᵀ
  score matrix and any decompressed KV are explicitly excluded (they stream through the separate
  3.13 MB/stack double-buffer of §6.2, never resident in this pool). Handles MLA-absorb,
  compressed-KV, and GQA-base attention variants, plus MoE/dense FFN.

This gate is why HBF+/CONV+'s 320 MB logic SRAM can bind batch size much harder than HBF's 36 GB
HBM stack, even though HBF+ has more raw flash capacity and higher bandwidth on every other axis.

---

## 7. The Parallelism Optimizer

### 7.1 What It Sweeps — `src/optimizer/parallelism_optimizer.cpp::Optimize()`

Nested loop over **TP × PP × DP × EP**:

- **TP** doubles from 1; **hard-capped at `num_kv_heads`** — `parallel.cpp` asserts `num_kv_heads
  % parallel_num == 0` and crashes otherwise, so the optimizer unconditionally skips `tp` values
  that don't divide `num_kv_heads`.
- **PP**: `tp*pp ≤ total_gpus`, evenly divides `total_gpus` and `num_layers`.
- **DP** = `total_gpus/(tp*pp)`; requires `batch_size % dp == 0`.
- **EP** (`e_tp_dg`) — an independent degree from TP: routed-expert weight can be tensor-split
  across a different device count than attention/dense weight, bounded by `devices_per_stage =
  total_gpus/pp` and mirroring `expert.cpp`'s own validity asserts so the optimizer never
  proposes a config the simulator would reject.

Per candidate, it computes: TP-aware attention/MLP weight bytes (MLA vs. GQA branches,
routed-expert weight divided by `devices_per_stage`, not total device count), MoE/dense
layer-mix handling for `first_k_dense`/`expert_freq` patterns via the shared `isMoELayer()`
helper (§9), an `E_active` estimate of concurrently-hot experts per device (also
`devices_per_stage`-based), KV-cache bytes (compressed vs. GQA, iRoPE-window-aware via
`effectiveKvLenSumAllLayers()`, §9.1), and `peakIntermediateBytes`-based activation size.

### 7.2 Capacity Gate vs. Analytic Latency Estimate

- **Hard gate**: `checkCapacity(...)` — capacity/SRAM is exact and monotonic; it's the *only*
  thing that sets `config.oom`.
- **Ranking-only estimate**: `estimated_latency_ms`, built from weight-read time (flash or HBM
  bandwidth + exposed page-read latency), compute time from FLOPs divided by `effectiveMFU(...)`
  (a no-op under shipped defaults), chunked KV-read time mirroring `getAttentionMemoryDuration`'s
  double-buffer exposed-latency formula (§6.2), and KV-write time hidden behind attention-only
  compute — that attention-only compute term is itself also divided by `effectiveMFU(...)`,
  matching the live kernel's basis exactly. Two selectable per-layer aggregation models via
  `system_config.optimizer_latency_model`: `"max"` (tighter, `max(compute, weight_mem)+kv` per
  layer) vs. `"sum"` (conservative, additive, with a flash-only DMA/compute overlap credit).
  **TP all-reduce** is a ring all-reduce model matching the live simulator's `AllReduce::forward`
  exactly: for a TP group of size `N`, cost = `2*(N-1)` hops, each hop paying one
  `device_ict_latency` plus `1/N` of the message size over `device_ict_bandwidth`; two all-reduces
  per layer (attention + FFN). PP send/receive and MoE scatter/gather (only when `e_tp_dg <
  devices_per_stage`) are added on top.
  **Per-stage total is multiplied by `pp`**, matching the live simulator: a decode token traverses
  all `pp` pipeline stages sequentially (no micro-batch-level pipeline overlap is modeled), so the
  true per-token estimate is `pp` stages' worth of the per-stage total (compute/weight/KV +
  that stage's own TP all-reduce + MoE scatter/gather) plus `(pp-1)` inter-stage send/receive
  hops — mirroring `Cluster::maxDeviceTime()`'s live-simulator accounting (§8.3).
  `estimated_latency_ms` is used only for ranking; it never sets `config.oom`, and the live
  simulator's measured TPOT remains the sole SLO arbiter (§7.3, §12).
- **Candidate selection**: among capacity-feasible (non-OOM) candidates, pick the one maximizing
  `batch_size_per_gpu / estimated_latency_ms` — i.e. **estimated throughput**, matching the
  paper's own stated objective ("selects the parallelism configuration that maximizes the
  achievable system throughput subject to all constraints"). No latency-based veto is applied at
  any point in `Optimize()`.

Note `tpot_slo_ms` is a parameter of `Optimize()` but not used inside it for filtering — SLO
comparison happens in the caller (`eval/test.cpp`, §7.3).

### 7.3 Two-Phase Batch-Size Search — `eval/test.cpp: analytic_sweep_only`

1. If batch=1 is capacity-infeasible, the whole GPU count is infeasible — no simulator call
   needed, since capacity is exact.
2. **Phase 0 (capacity ceiling)**: since capacity is exact and monotonic in batch size, compute
   the capacity ceiling directly via exponential-then-binary search with no per-batch simulator
   calls.
3. **Phase 1 (SLO-guided binary search)**: binary-search downward from the capacity ceiling using
   the analytic latency estimate against `tpot_slo_ms` to propose a candidate batch and its
   TP/PP/EP/DP.
4. Emits `ANALYTIC_MAX_BATCH`, `ANALYTIC_TP/PP/EP/DP`, `ANALYTIC_ESTIMATED_LATENCY_MS` on stdout.

`run_experiments.py`'s `find_max_batch_size` then does **Phase 2 (simulator verification)**:
confirms the candidate and a window of batches immediately above it (a boundary safety check —
sized to cover the DP-divisibility cycle at the current GPU count, since feasibility is
non-monotonic at the integer level) with the real simulator; if the simulator disagrees with the
analytic estimate in either direction, it falls back to a simulator-driven binary search over the
narrowed range. The analytic phase never itself declares a batch infeasible purely on the latency
estimate — a real simulator call is always the final word.

---

## 8. Hardware / Cluster Layer

### 8.1 Topology & GPU Presets

Topology (`Cluster`/`Node`/`Device`, `get_device`, NVLink=intra-node/`device_ict_*`,
InfiniBand=inter-node/`node_ict_*`) is unchanged from upstream. `hardware_config.h` has five
`SystemConfig` presets: **A100, H100, B100, B200** (upstream) plus **Rubin** (matching the HBF
paper's cited reference GPU, NVIDIA DGX Rubin NVL8). Rubin's `compute_peak_flops` is explicitly
commented as an estimate: the datasheet gives only an aggregate FP8/FP6 training figure (140
PFLOPS/8 GPUs), halved per this file's FP16-base convention. Its `memory_bandwidth`/
`memory_capacity` fields are placeholders and not load-bearing — `eval/test.cpp` always overwrites
them from the selected `hbf_config` preset whenever `memory_type` is set (true in every sweep this
project runs).

`Device`'s constructor applies the same override at construction time
(`use_hbf && num_flash_stacks>0 → memory_capacity/bandwidth = hbf_config.*`).

`gpu_gen: A100` has no Ramulator DRAM-config-path branch in `device.cpp` — H100/B100/B200/Rubin
branches exist, A100 does not, and setting it would crash on `YAML::LoadFile("")` (see `BUGS.md`
item 1 — dormant, no sweep in this project uses it).

### 8.2 Communication

AllReduce ring and All-to-All expert scatter/gather (`communication.cpp`) are structurally
unchanged from upstream. Every node-index derivation (`src_node`/`dst_node`) uses
`device->config.num_device` (not a hardcoded constant), so runs with `num_device != 8` (e.g. the
16-GPU / 2-node sweep points) compute node membership correctly. NVLink/InfiniBand bandwidths are
set via `nvlink_gen`/`infiniband_gen` in `eval/test.cpp` (`nvlink_gen: 5 → 900 GB/s` unidirectional
per-link; `infiniband_gen: 800 → 100 GB/s`).

**Pipeline-stage time propagation.** `PipelineStage` (`communication.cpp`) is the module
representing a pipeline-parallel stage transition. Unlike `AllReduce`/`MoEScatter` (which
reconcile time across their device group via `ModuleGraph::sync_devices()`'s max-and-broadcast,
since they operate within one pipeline stage's TP/EP group), `PipelineStage` is the *only*
cross-stage-boundary module in the decode critical path, and its `forward()` explicitly
propagates elapsed time to the destination device: after computing the inter-stage transfer
`comm_time` and adding it to the source device's own `status.device_time`, it sets
`dst_device->status.device_time = max(dst_device->status.device_time, device->status.device_time)`
— a token cannot begin the next stage's compute before the current stage's output has physically
arrived. This makes the *last* stage of whichever pipeline finishes last hold the true cumulative
per-token latency across the entire pipeline traversal.

### 8.3 Cluster Execution Loop

`runIteration` dispatches `runIterationMixed` vs. `runIterationSumGenSplit` on
`config.disagg_system`; both read the per-iteration decode-step time via
`Cluster::maxDeviceTime()` — the maximum `status.device_time` across every device in the cluster,
not any single device's own time. This is what correctly captures the propagated pipeline-stage
latency from §8.2 (for `PP==1`, no `PipelineStage` module is ever created, so this reduces
identically to reading the one device's own time — a no-op there). `setStat` reads the same value.

Two capacity checks exist:

- **`checkMemorySize`**: `size_for_capacity_gate = hasScarceTier(config) ? (size -
  activation_size) : size` — activation is excluded from the lumped weight+KV pool gate whenever a
  scarce tier exists, since it already has its own, separate gate via
  `scarceTierActivationLimit`/§6.3.
- **`checkHeteroMemorySize`** — a separate, unaudited capacity check with a hardcoded `3.3 GB`
  "Non MoE weight" magic-number subtraction and no activation term at all (`BUGS.md` item 6).

The KV-write penalty (§6.2) is folded into the breakdown without double-counting: attention
duration is split as `atten_sum/gen += (total - kv_write); kv_write += kv_write` at the relevant
stamp-processing sites in `cluster.cpp`.

Attention timing kernels (`AttentionSum`/`AttentionGen`/`AttentionMixed`, MLA/Absorb variants) are
structurally the same three-regime split as upstream — see the base guide — with the flash calls
into `layer_impl.h` (§6.2) as the main change, plus the still-present PIM `memory×opb`
non-roofline quirk (§16).

---

## 9. Model / Module Layer

Graph assembly (`LLM` = Embedding → N×Decoder/MoEDecoder → LmHead), the `Tensor` abstraction, and
attention geometry (MHA/GQA/MQA/MLA, `attention_group_size = num_heads/num_kv_heads`) are
structurally the same as upstream.

`isMoELayer()` (`model/model_config.h`) is a shared helper used by both the module-construction
path (`llm.cpp`) and the parallelism optimizer, so the two can never disagree on which layers are
MoE. Honors `first_k_dense` (forced-dense prefix, any model) and `expert_freq`, so MLA-specific/
DeepSeek-specific handling is derived from the model's own architecture parameters rather than a
global flag.

### 9.1 Llama-4 "iRoPE" Interleaved Local/Global Attention

Real Llama 4 uses Meta's "iRoPE" architecture: only every 4th layer is full-context ("NoPE"); the
other 3 use a fixed 8192-token local attention window. `ModelConfig` carries
`attn_chunk_size`/`attn_global_interval` fields, defaulting to `0`/`1` (full global attention —
a no-op for every model that doesn't set them). `llama4_maverick` and `llama4_scout` set
`attn_chunk_size=8192, attn_global_interval=4` (independently verified against Meta's released
`Llama-4-Maverick-17B-128E-Instruct config.json`; Scout isn't one of the paper's own evaluated
models, so its exact values are lower-confidence than Maverick's).

Three shared helpers in `model_config.h`:
- `isGlobalAttentionLayer(mc, layer)` — 0-indexed layer `L` is global when `(L+1) %
  attn_global_interval == 0`.
- `effectiveKvLen(mc, layer, context_len)` — global layers see the full context; local layers are
  capped at `attn_chunk_size` (a deliberate flat-cap simplification vs. Llama-4's real
  sawtoothing per-token local window).
- `effectiveKvLenSumAllLayers(mc, context_len)` — sums `effectiveKvLen` across all layers; the
  quantity that replaces `num_layers * context_len` everywhere. Reduces exactly to the old
  formula when `attn_chunk_size==0`.

This windowing is wired into every path that touches KV volume, kept in lock-step by design:

- **Live-simulator KV-cache tensor allocation**: a `layer_idx` parameter threads through
  `llm.cpp`'s per-layer loop → `decoder.{cpp,h}` → `layer.{cpp,h}`'s `Attention` constructor
  (`gen_seq_len = effectiveKvLen(...)`) → `parallel.{cpp,h}`'s `SelfAttentionParallel` → its
  `SelfAttentionGen` sub-module (the decode path every current sweep exercises; the prefill-path
  modules were deliberately left unwindowed, since `decode_mode: on` means prefill is never
  exercised — `BUGS.md` item 2). `SelfAttentionGen` stores the effective window as a
  `LayerInfo::local_attention_window` field at `forward()` time — this drives
  `cluster.cpp::checkMemorySize`'s ground-truth capacity gate, since it sizes the `k_cache`/
  `v_cache` tensors themselves.
- **Runtime KV-read latency**: `attention_gen_impl.cpp`'s `AttentionGenExecutionGPU` (the GQA
  decode path both llama3/llama4 use) caps `n`/`k` at `layer_info.local_attention_window` in the
  Scoring/Softmax/Context loops. Only the GPU variant is windowed (llama4's actual decode path);
  Logic/PIM GQA and the MLA variants (DeepSeek, always full-global) are unaffected.
- **Runtime KV-write latency**: `getKVWriteDuration` (§6.2) caps the write length at the same
  per-layer window.
- **Optimizer's analytic KV-cache-bytes and KV-write formulas** (§7.1, §7.2) both use
  `effectiveKvLen`/`effectiveKvLenSumAllLayers` to match.

Model presets (`model_config.h`): `llama3_405B` uses BF16 precision (`precision_byte=2`);
`llama4_maverick` (also BF16, 128 experts) is the other primary evaluation model.
`deepseekV3`, `mixtral`, and other presets' precision assumptions are unverified against any
external reference (`BUGS.md` item 4).

---

## 10. Scheduler Layer (Continuous Batching)

Sequence lifecycle, `BatchedSequence`, and expert routing (uniform/Zipfian) are unchanged from
upstream. `simulation.injection_rate` (`eval/test.cpp`), when `> 0`, sets `use_inject_rate=true` +
`request_per_second`, activating the open-loop Poisson-arrival path (`scheduler.cpp`) instead of
closed-loop continuous batching. `run_experiments.py`'s sweep driver explicitly overrides
`injection_rate = 0` for every measurement it runs ("Continuous batching" — the paper's steady-
state saturated-batch model), so this only affects a bare `./run config.yaml` invocation with the
shipped `config.yaml`'s default (`injection_rate: 20`), never a sweep-driven measurement.

---

## 11. DRAM / Ramulator / PIM Layer

Unchanged from upstream (still gated behind `use_ramulator: off` by default; every sweep in this
project runs the pure analytical path described in §5–§6, not real Ramulator cycles). See the
base guide for the full Ramulator/PIM-kernel/address-mapping description — it applies identically
here.

---

## 12. Configuration Reference (config.yaml)

Keys relative to upstream that are new or behave differently (current shipped values shown):

| Section | Key | Meaning | Current value |
|---|---|---|---|
| system | `gpu_gen` | now includes `Rubin` | `Rubin` |
| system | `memory_type` | `HBM` (plain, upstream-compatible) / `HBM4` / `HBF` / `HBF+` / `CONV` / `CONV+` | `HBM` |
| system | `chunk_size` | chunked-attention chunk size in **bytes**; `0`=auto (full SRAM staging capacity) | `0` |
| system | `tpot_slo` | target TPOT in **seconds** (converted ×1000 internally to `tpot_slo_ms`) | `0.1` |
| system | `optimize_parallelism` | run `ParallelismOptimizer::Optimize` to pick TP/PP/DP/EP instead of `distribution.*` | `false` |
| system.distribution | `expert_tensor_degree` / `none_expert_tensor_degree` | `e_tp_dg` / `ne_tp_dg` | `1` / `8` |
| simulation | `precision_byte` | `0` = use model preset's own precision | `0` |
| simulation | `injection_rate` | `>0` enables open-loop Poisson arrivals (see §10) | `20` |
| simulation | `validate_optimizer` | `off`/`warn`/`strict` — cross-checks optimizer prediction vs. live simulator footprint | `warn` |
| simulation | `validate_optimizer_threshold` | relative divergence threshold for the above | `0.10` |
| simulation | `optimizer_latency_model` | `"max"` (tighter, tracks simulator overlap) or `"sum"` (conservative additive) — §7.2 | `max` |
| simulation | `latency_margin` | multiplicative safety margin on the optimizer's estimated latency | `1.0` |
| simulation | `mfu_max` | asymptotic (large-batch) achieved FLOPs fraction of peak, `(0,1]`; `1.0` = no derating (§5) | *(unset → defaults to `1.0`, exact no-op)* |
| simulation | `mfu_m_half` | GEMM row-count at half-saturation of the MFU curve; `0` disables the ramp (§5) | *(unset → defaults to `0.0`, exact no-op)* |

All other keys (`num_node`/`num_device`/`processor_type`, `optimization.*`, `serving.*`,
`log.*`) are unchanged from upstream — see the base guide for the full reference.

---

## 13. Outputs: CSV, stdout markers, experiment_results.md

CSV columns, `type` values, and Gantt/Timeboard export are unchanged from upstream — one
addition: the per-stage breakdown separately reports the KV-write penalty (accumulated via
`module_graph.cpp`'s `kv_write_time`) rather than folding it silently into attention time.

**Stdout markers** (only emitted under `system.analytic_sweep_only: true`, §7.3):
`ANALYTIC_CAP_FEASIBLE_AT_1`, `ANALYTIC_CAP_BATCH`, `ANALYTIC_MAX_BATCH`,
`ANALYTIC_TP/PP/EP/DP`, `ANALYTIC_ESTIMATED_LATENCY_MS`. Also
`PEC_KV_BYTES_PER_TOKEN`/`PEC_FLASH_CAPACITY_BYTES` markers (parsed by `run_experiments.py`) for
the write-endurance metric.

`run_experiments.py`'s `main()` writes **five paper-defined metrics** to `experiment_results.md`
(currently stale relative to the fixes described in `CHANGES.md`'s later items — regenerating it
is a distinct, heavy step, not run automatically):
(1) Maximum Per-GPU Batch Size (`max_batch/gpu_count` — total GPU count, not DP replica count),
(2) System Throughput,
(3) Runtime Performance Breakdown (restricted to `{4,8,16}` GPUs, matching the paper's figure
scope),
(4) SLO Sensitivity (also `{4,8,16}`),
(5) Write Traffic / 3-Year PEC vs. the 100K SLC cycle limit.

---

## 14. Build & Run

```bash
git clone <this-repo>
cd HBF-LLMSimulator
git submodule update --init --recursive

# Apply the Ramulator 2.0 PIM patch
cd src/dram/ramulator2
git apply ../../../patch/ramulator2_pim.patch
cd ../../..

mkdir build && cd build
cmake ..
make -j
cd ..
```

**Single run** — edit `config.yaml`, then:
```bash
./run config.yaml > output.log
```
Exit code 0 = success; 1 = OOM (capacity or SRAM limit exceeded).

**Full experiment sweep** — all GPU counts (1/2/4/8/16), all 5 memory presets, all workloads and
SLOs:
```bash
python3 run_experiments.py
```
Implements the two-phase batch search (§7.3) with capacity/latency separation and per-GPU batch
reporting. Baseline for normalized metrics = 8-GPU HBM4 under the same SLO. Simulation CSV outputs
go to `data/`.

**`compare_error_rates.py`** — compares `paper_figure_readings.md` (values read off the paper's
own figures) against `experiment_results.md` (this repo's simulator output) and reports error
rates per figure/data-group.

---

## 15. File Map

New/changed files relative to the upstream file map (base guide has the full upstream reference):

| File | Role |
|---|---|
| `src/dram/hbf_memory_config.h` | `HBFMemoryConfig` struct + 5 memory presets (§6.1) |
| `src/hardware/layer_impl.h` | Flash timing helpers: `getLinearMemoryDuration`, `getAttentionMemoryDuration`, `getKVWriteDuration` (§6.2) |
| `src/model/footprint.h` | Shared scarce-tier capacity/footprint logic: `checkCapacity`, `scarceTierActivationLimit`, `peakIntermediateBytes` (§6.3) |
| `src/optimizer/parallelism_optimizer.{h,cpp}` | `ParallelismOptimizer::Optimize` — TP×PP×DP×EP sweep (§7) |
| `src/hardware/cluster.{h,cpp}` | `Cluster::maxDeviceTime()` — cross-stage-propagated per-token latency read (§8.3) |
| `src/module/communication.cpp` | `PipelineStage::forward` — cross-stage time propagation (§8.2) |
| `eval/test.cpp` | Entry point; hosts `analytic_sweep_only` mode and memory-preset/chunk-size parsing (§4.1, §7.3) |
| `run_experiments.py` | Sweep driver — the paper-comparison harness (§14) |
| `compare_error_rates.py` | Reusable comparison tool: `paper_figure_readings.md` vs. `experiment_results.md` (§14) |
| `CHANGES.md` | Full history of bugs found and fixed, with rationale and paper cross-references |
| `BUGS.md` | Currently-known unfixed/dormant bugs (including those formerly masked by pinned config flags) |
| `PAPER_INCONSISTENCIES.md` | Still-open and explained-not-bug paper-comparison gaps |
| `experiment_results.md` | This repo's simulator output, in the paper's figure format |
| `paper_figure_readings.md` | Values read directly off the paper's own figures |
| `PAPER_GROUND_TRUTH.md` | Non-figure paper ground truth: hardware constants, methodology, prose claims |
| `ledgers/` | Raw investigation/adjudication trails (bug-hunt passes, orchestrator ledgers) |

All other files match the upstream file map exactly (base guide has the full reference).

---

## 16. Known Modeling Simplifications

Simplifications and quirks inherited from upstream that remain present in this fork (see the base
guide's own quirks section for the original description of each):

- "Mixed" scheduling isn't truly mixed — any pending prefill forces the whole step to prefill;
  decode gets 0 tokens that step (`scheduler.cpp`).
- Synthetic request-length jitter is computed but never applied (`scheduler.cpp`).
- Inter-arrival times under `injection_rate > 0` are drawn from a `poisson_distribution` (should
  be exponential for a Poisson *process*) — not exercised by any sweep in this project, since
  `run_experiments.py` always sets `injection_rate = 0` (§10), but live on the default path for a
  bare `./run config.yaml` invocation.
- PIM decode-attention uses `memory_duration × opb` instead of `max(compute, memory)`
  (`attention_gen_impl.cpp`), inconsistent with every other kernel's roofline formula. Only
  matters if attention-gen is ever dispatched to PIM/LOGIC.
- Dead code: `data_object.*`, `pimBankgroupEnergy`, most PIM kernels (CKKS legacy),
  `getNumInjection`/`getPoissondistribution` (no call sites).
- Memory-limited batch shrinking (`mem_cap_limit`) happens once at startup, not dynamically at
  runtime (`cluster.cpp`).
- `gpu_gen: A100` previously had no Ramulator DRAM-config branch in `device.cpp` and would crash
  if selected — **fixed 2026-07-02** (`CHANGES.md` item 70); an A100 branch now exists, mirroring
  the Rubin pattern.

HBF-specific simplifications, new to this fork:

- **iRoPE local-attention windowing uses a flat cap**, not Llama-4's real per-token sawtooth
  local-window trajectory (§9.1) — a deliberate simplification that empirically tracks the paper's
  reported anchors more closely than the sawtooth's lower average would.
- **`page_size_bytes` is vestigial config surface**: no timing formula currently computes with the
  literal 4 KiB page size; it remains as hardware-geometry documentation only (`BUGS.md` item 7).
- **Disaggregated path (`disagg_system=on`, currently unused by any sweep)** does not model a
  one-time prefill→decode KV-transfer event; the per-decode-step KV-write penalty (§6.2) is
  present and correct on both execution paths regardless (`BUGS.md` item 1).
- **`checkHeteroMemorySize()`** (§8.3) has its own, separately-maintained capacity-check logic,
  not reconciled with `checkMemorySize`'s scarce-tier gate — confirmed dead code (zero call sites)
  with a latent capacity-math bug if ever revived (`BUGS.md` item 3).
- **`runIterationMixed` has no internal structural guard** that `decode_mode: on` is set — decode-
  only TPOT correctness currently depends on this convention holding; a runtime warning was added
  (`CHANGES.md` item 73) but no hard guard exists (`BUGS.md` item 2).
- **MLA prefill attention on flash never models KV-read timing** (`BUGS.md` item 4) and
  **`logic_x`/`pim_x` speedup multipliers are inconsistent between the routing heuristic and the
  timing model** (`BUGS.md` item 5) — both dormant, requiring `parallel_execution: on` and/or an
  MLA model in prefill mode.

See `PAPER_INCONSISTENCIES.md` for simulator-vs-paper numeric gaps (both still-open and
explained-as-not-a-bug), and `BUGS.md` for the full current bug list.
