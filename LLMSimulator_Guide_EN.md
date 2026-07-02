# LLMSimulator Structure & Functionality Guide

> This document is a structure/behavior guide written after directly reading and verifying the code
> of the `LLMSimulator` repository (= the cycle-approximate simulator used for evaluation in the MICRO
> 2024 paper *"Duplex: A Device for Large Language Models with Mixture of Experts, Grouped Query
> Attention, and Continuous Batching"*, arXiv:2409.01141). Every core claim is backed by a
> `file:line` reference, so it can be read alongside the code.
>
> Target commit: `main` (Add Llama-4 support). Since the paper, the code has been extended with
> DeepSeek-V3 (MLA), Llama-4, Zipfian expert skew, disaggregated serving, and more, so keep in mind
> that **the code is a superset of the paper**.

---

## Table of Contents

1. [At a Glance](#1-at-a-glance)
2. [Background: The Problem the Simulator Models](#2-background-the-problem-the-simulator-models)
3. [Top-Level Architecture and Object Graph](#3-top-level-architecture-and-object-graph)
4. [End-to-End Execution Flow](#4-end-to-end-execution-flow)
5. [Core Timing Model (the Simulator's Heart)](#5-core-timing-model-the-simulators-heart)
6. [Hardware / Cluster Layer](#6-hardware--cluster-layer)
7. [Model / Module Layer](#7-model--module-layer)
8. [Scheduler Layer (Continuous Batching)](#8-scheduler-layer-continuous-batching)
9. [DRAM / Ramulator / PIM Layer](#9-dram--ramulator--pim-layer)
10. [Configuration Reference (config.yaml)](#10-configuration-reference-configyaml)
11. [Outputs: CSV Columns · Gantt/Timeboard](#11-outputs-csv-columns--ganttimeboard)
12. [Build & Run](#12-build--run)
13. [Modeling Simplifications · Known Quirks (Important)](#13-modeling-simplifications--known-quirks-important)
14. [File Map (Quick Reference)](#14-file-map-quick-reference)

---

## 1. At a Glance

`LLMSimulator` is a C++ simulator that **predicts LLM inference serving in terms of latency and
energy**. The README calls it "cycle-accurate," but its actual default behavior is an
**analytical roofline model** (a closed-form expression based on compute/data volume); memory time
is only computed cycle-by-cycle via Ramulator 2.0 when the `use_ramulator` option is enabled (§5, §9).

The simulator's purpose is to quantitatively evaluate the paper's **Duplex** idea — sending
high-Op/B operations to a GPU-like **xPU**, and low-Op/B (memory-bound) operations such as
MoE/Attention to **Logic-PIM** or bank-PIM on the HBM logic die — across many combinations of
model, batch, and hardware.

The whole structure boils down to three things:

| Concept | One-line summary | Evidence |
|---|---|---|
| **Two components** | Exactly as in Section VI of the paper: `Scheduler` (continuous batching) + `Cluster` (devices execute ops and emit timing). | `eval/test.cpp:224-343` |
| **Graph built once, re-walked every step** | The model is symbolically traced **exactly once** into a per-device `TopModuleGraph` (a flattened list of ops), and every iteration re-walks this graph to accumulate time. | `model.h:22-52`, `module_graph.cpp:149-157` |
| **Roofline + time accumulation** | Latency of one op = `max(compute_time, memory_time)`. This is simply **added** to `device_time` (no event queue). Across devices, sync points reconcile time via `max`. | `linear_impl.cpp:61-62`, `module_graph.cpp:249, 74-97` |

Processor-type mapping (paper terminology ↔ code):

| Code `ProcessorType` | Paper concept | Characteristics (default) |
|---|---|---|
| `GPU` | xPU (high Op/B) | Uses `compute_peak_flops`, `memory_bandwidth` as-is |
| `LOGIC` | Logic-PIM | Bandwidth ×4 (`logic_x`), Op/B 8 (`logic_op_b`) |
| `PIM` | bank-PIM | Bandwidth ×16 (`pim_x`), Op/B 1 (`pim_op_b`) |

`processor_type` is evaluated as combinations of `GPU`, `LOGIC`, `GPU+LOGIC` (= Duplex), and
`GPU+PIM` (`eval/test.cpp:140-157`).

---

## 2. Background: The Problem the Simulator Models

A short summary of the paper (see `docs/2409.01141v1.pdf` for details):

- **LLM inference** consists of a single **prefill** stage (called *sum*, summarization, in the
  code) followed by many **decode** stages (called *gen*, generation, in the code). Continuous
  batching groups requests at the **stage level** rather than the request level to increase
  throughput.
- **Op/B (arithmetic intensity)**: MoE/Attention layers have low Op/B during the decode stage
  (memory-bound), so GPU utilization drops below 10%. Conversely, FC (GEMM) and prefill have high
  Op/B (compute-bound).
- **Duplex**: routes high-Op/B work to the xPU (GPU) and low-Op/B work to Logic-PIM, using
  expert/attention co-processing to use both units simultaneously.
- **MoE / GQA / MLA**: MoE performs top-k expert FFN; GQA reduces K/V heads by a factor of
  `deg_grp` to cut KV-cache/bandwidth; MLA (DeepSeek) compresses K/V into a latent representation.

The simulator represents each of these elements as a **module** and feeds each module's compute/data
volume into the roofline model to derive timing. In short, it is a
**"model structure → tensor size → roofline latency"** pipeline.

---

## 3. Top-Level Architecture and Object Graph

The source is split into 5 subsystems (build library units, `src/CMakeLists.txt`).

```
                 ┌─────────────────────────── eval/test.cpp (main, "run") ───────────────────────────┐
                 │  config.yaml → SystemConfig + ModelConfig → Scheduler → Cluster → Model → runIteration
                 └───────────────────────────────────────────────────────────────────────────────────┘
                                     │                    │                 │
        ┌────────────────────────────┼────────────────────┼─────────────────┼──────────────────────────┐
        ▼                            ▼                    ▼                 ▼                          ▼
  ┌───────────┐            ┌──────────────────┐   ┌──────────────┐  ┌──────────────────┐     ┌──────────────────┐
  │ scheduler │            │     hardware     │   │    model     │  │      module      │     │       dram       │
  │ (1.1k LOC)│            │    (9.6k LOC)    │   │  (0.4k LOC)  │  │    (6.1k LOC)    │     │    (1.3k LOC)    │
  ├───────────┤            ├──────────────────┤   ├──────────────┤  ├──────────────────┤     ├──────────────────┤
  │ Sequence  │            │ Cluster/Node/    │   │ LLM graph    │  │ Attention/Expert │     │ DRAMInterface    │
  │ Batched   │            │ Device/Executor  │   │ assembly     │  │ Linear/Route/... │     │ +Ramulator patch │
  │ Continuous│            │ *_impl.cpp       │   │ ModelConfig  │  │ TopModuleGraph   │     │ PIM kernels      │
  │ batching  │            │ = roofline timing│   │ presets      │  │ Tensor/Timeboard │     │ (GEMV/Read/Write)│
  └───────────┘            └──────────────────┘   └──────────────┘  └──────────────────┘     └──────────────────┘
```

The core objects that live at runtime, and their ownership relationships:

```
Scheduler ── running_queue: vector<BatchedSequence>   (one per DP rank)
   │             └── BatchedSequence ── vector<Sequence>   (requests in the batch)
   │
Cluster ── node: vector<Node>
   │           └── Node ── device: vector<Device>
   │                          └── Device
   │                                ├── config: SystemConfig     (peak FLOPS, bandwidth, capacity...)
   │                                ├── top_module_graph: TopModuleGraph  ← this device's "program"
   │                                │       ├── module_graph: vector<ModuleGraph>  (flattened op list)
   │                                │       └── TimeBoard  (per-op timestamp/energy tree)
   │                                ├── status: StatusBoard  (device_time / high_time / low_time / energy accumulator)
   │                                ├── dram_interface: DRAMInterface  (Ramulator wrapper)
   │                                └── mmap_controller  (address mapping/allocation)
   │
Model (inherits Module) ── builds one LLM per device via symbolic forward, filling in the graph
Executor (owned by Cluster) ── LayerType×ProcessorType → table of timing function pointers
```

**Two-layer mental model** (most important):

1. **Build time (once)**: The `Model` constructor builds an `LLM` module tree for each device and
   runs a symbolic forward pass once using `getMaxMetadata` (worst-case batch). Every time
   `Module::operator()` is invoked, a matching push/pop pair is recorded into the
   `TopModuleGraph::module_graph` vector, completing a **flattened, ordered list of ops**
   (`model.h:37-53`, `module.cpp:10-17`, `module_graph.cpp:118-140`).
2. **Runtime (every iteration)**: `Cluster::run` walks that vector from start to end, adding each
   op's `total_duration` to `device_time`. The actual batch size/sequence length is read from the
   current `BatchedSequence` (metadata) at that moment, updating tensor shapes and recomputing the
   roofline each time.

---

## 4. End-to-End Execution Flow

`main()` lives in `eval/test.cpp`. The resulting executable is named `run` (`eval/CMakeLists.txt:1`).

### 4.1 Setup (once)

```
Load config.yaml                                                             eval/test.cpp:15-22
 ├─ Select SystemConfig: gpu_gen(A100/H100/B100/B200) → preset struct                :42-56
 │   └─ Override device_ict_* / node_ict_* via nvlink_gen / infiniband_gen           :59-89
 │   └─ processor_type string → {GPU}/{LOGIC}/{GPU,LOGIC}/{GPU,PIM}                  :140-157
 │   └─ Optimization flags (parallel_execution, disagg_system, use_ramulator,
 │       compressed_kv, use_absorb, use_flash_*, prefill/decode_mode ...)            :99-202
 ├─ Select ModelConfig: model_name → preset (mixtral, deepseekV3, ...)               :163-182
 │   └─ Override e_tp_dg/ne_tp_dg, precision_byte, skewness, input/output_len        :184-213
 │   └─ If precision_byte==1 (FP8/INT8), compute_peak_flops ×2                       :197-199
 ├─ Scheduler::Create(system, model, expert_path, max_batch_size, 8192, max_process_token)  :224-226
 ├─ Cluster::Create(system, scheduler)  → creates Node/Device                        :228
 ├─ Model(model, cluster, scheduler)    → symbolically builds the per-device graph   :230
 ├─ cluster->checkMemorySize()          → checks per-device memory footprint         :232
 └─ cluster->set_dependency()           → wires cross-device dependencies for sync modules  :233
```

- `SystemConfig` presets are hardcoded as static structs (`hardware_config.h:163-313`).
- `checkMemorySize` compares the sum of `activation + weight + KV cache` against
  `memory_capacity`. On overflow it either triggers `exit_out_of_memory` (abort) or
  `mem_cap_limit` (shrink the batch) (§8.5).
- **Note**: `max_batch_size` comes from config, `8192` is a hardcoded
  `num_max_batched_token` (used mainly for graph sizing), and `max_process_token` is the
  prefill token budget (§8.3).

### 4.2 Simulation Loop

```
scheduler->getActualArrivalTime(iter)   Pre-generates Poisson arrival times          eval/test.cpp:342
cluster->runIteration(iter, file_name)                                               eval/test.cpp:343
 ├─ Write CSV header                                                                 cluster.cpp:428-436
 ├─ fillSequenceQueue(); fillRunningQueue();  Initial fill of queue/running batch     cluster.cpp:440-441
 ├─ hittingQueue(10000)   Warm-up (reach steady state)                                cluster.cpp:444
 └─ disagg_system ? runIterationSumGenSplit : runIterationMixed                       cluster.cpp:446-450

runIterationMixed(iter):  for i in [0, iter):                                        cluster.cpp:467-522
   metadata = scheduler->setMetadata()       Decide this step's batch/tokens          :476
   run(metadata)                             Run the model once end-to-end → device_time  :477
   time = get_device(0)->status.device_time  This step's wall-clock time              :478
   total_time += time                                                                 :486
   Build Stat (getTotalEnergy, setTimeBreakDown)                                      :488-510
   token_list = scheduler->updateScheduler(time)  Advance current_len / pop completed  :517
   addLatency(...)                           Record e2e/t2ft for completed/first-token seqs  :518
   fillSequenceQueue(time,total_time); fillRunningQueue();  Refill the batch           :520-521
```

- **One iteration = one model forward pass** = generates 1 token for decode requests, and a
  `num_process_token`-sized chunk for prefill requests. `iter` is this step count (not the number
  of layers).
- Inside `run(metadata)`: after `restartModuleGraph()`, `while(check_module_graph_remain()) { run
  each node/device's graph }` — this is a **multi-pass drain loop** that breaks out when blocked at
  a sync point and loops again (`cluster.cpp:1084-1092`). Each device executes with the metadata of
  its `dp_rank = device_total_rank / ne_tp_dg` shard (`device.cpp:71-74`).
- If there are no tokens to process (`getNumProcessToken()==0`), it inserts 20ms and continues
  (`cluster.cpp:481-484`).

### 4.3 Wrap-up

- The `stat_list` returned by `cluster->runIteration` is streamed out as CSV (the filename encodes
  model/length/processor/N/D/TP/DP/batch/token/iter/skew/precision/parallel,
  `eval/test.cpp:247-336`).
- If `export_gantt` is set, per-device timeboard trees are dumped; if `print_log` is set, device
  0's timeboard is printed to stdout (`eval/test.cpp:345-355`).

---

## 5. Core Timing Model (the Simulator's Heart)

### 5.1 Roofline: Latency of a Single Op

All GEMM-family ops share the same form. `LinearExecutionGPU` (`linear_impl.cpp:16-75`) is the
canonical implementation:

```cpp
double m = input->shape[0], k = input->shape[1], n = weight->shape[1];
total_flops       = 2.0 * m * k * n;                                   // GEMM FLOPs
total_memory_size = (m*k + k*n + m*n) * weight->precision_byte;        // in + weight + out (bytes)

compute_duration  = total_flops / compute_peak_flops * 1e9;            // ns
memory_duration   = total_memory_size / memory_bandwidth * 1e9;        // ns
total_duration    = std::max(compute_duration, memory_duration);       // ← roofline
opb               = total_flops / total_memory_size;                   // arithmetic intensity
```

- `total_duration = max(compute, memory)` is the exact equivalent of the roofline model: if Op/B
  exceeds the ridge point (`peak_flops/bandwidth`) the op is compute-bound; otherwise it is
  memory-bound.
- `compute_util` and `memory_util` are the achieved utilization rates (`linear_impl.cpp:64-67`).
- `precision_byte` directly scales the data volume. FP8/INT8 (`=1`) uses half the memory of FP16
  (`=2`).

### 5.2 Substituting Peak/Bandwidth per Processor

The same roofline formula is reused with **different constants** to represent xPU/PIM.

| Function | `compute_peak_flops` | `memory_bandwidth` | Evidence |
|---|---|---|---|
| `LinearExecutionGPU` | `config.compute_peak_flops` | `config.memory_bandwidth` | `linear_impl.cpp:20-21` |
| `LinearExecutionLogic` | `logic_memory_bandwidth × logic_op_b` = (BW×4)×8 | `logic_memory_bandwidth` = BW×4 | `linear_impl.cpp:81-83` |
| `LinearExecutionPIM` | `pim_memory_bandwidth × pim_op_b` = (BW×16)×1 | `pim_memory_bandwidth` = BW×16 | `linear_impl.cpp:145-146` |

In other words, **Logic-PIM is a unit with 4× bandwidth and Op/B of 8**, and **bank-PIM is a unit
with 16× bandwidth and Op/B of 1**. `logic_memory_bandwidth = memory_bandwidth × logic_x` and
`pim_memory_bandwidth = memory_bandwidth × pim_x` are derived values
(`hardware_config.h:90-91`). LOGIC/PIM also get peak ×2 when `precision_byte==1`.

`Activation` (e.g. SwiGLU gating) is treated as `flops=0` (pure memory traffic) on GPU, but on
LOGIC/PIM it is treated as `flops=memory_size` (1 flop/byte), allowing it to become compute-bound
on low-Op/B units (per agent analysis; `activation_impl.cpp`).

### 5.3 Time Accumulation & Parallel Tracks (No Event Queue)

This simulator has **no discrete-event queue and no global event clock.** Instead:

- Whenever an op is popped, `set_pop_status()` performs `status.device_time +=
  exec_status.total_duration`, **serially accumulating** the device clock
  (`module_graph.cpp:249`).
- Ops under `parallel_execution` (GPU‖PIM overlap) are added separately to either the `high_time`
  (GPU) or `low_time` (LOGIC/PIM) track (`module_graph.cpp:239-247`). When a non-parallel op is
  encountered, the two tracks are merged via `device_time = max(device_time, high_time, low_time)`
  (`module_graph.cpp:196-218`). This is the time-overlap model for co-processing.
- **Cross-device synchronization**: `communication`/`sync` modules have `sync==true`, so before
  execution `sync_devices()` takes the **maximum** of `get_time()` across dependent devices and
  writes it into all of them (a barrier; `module_graph.cpp:74-97`).

In summary: **latency = "serial sum of per-op max(compute, memory), except GPU‖PIM sections which
progress in parallel on two tracks, with a max-barrier at device boundaries."**

### 5.4 Processor Selection: "Try Both, Pick the Faster One"

When an op can run on multiple processors (e.g. `GPU+LOGIC`), `Executor::execution` measures the
time for each candidate in `layer_info.processor_type` and picks the **minimum duration**
(`executor.cpp:145-152`):

```cpp
for (auto type : layer_info.processor_type) {
  status = executePType(..., type, ...);
  if (duration == 0 || duration > status.total_duration) { optimal = status; duration = ...; }
}
```

The paper's notion of "selecting a processor by Op/B" is not implemented as an explicit threshold,
but **emerges naturally from this min()**: memory-bound decode-attention wins on LOGIC (4×
bandwidth), while compute-bound GEMM wins on GPU.

However, a module can also **narrow** the candidate set. Under `parallel_execution` mode,
`SelfAttentionGen` is pinned to `low_processor_type` (`attention.cpp:52`), while
`SelfAttentionSum` is pinned to `high_processor_type` (`attention.cpp:102`) — implementing the
paper's "decode→PIM, prefill→xPU" split. Results are memoized by the key
`(layer_type, processor_type, dram_request_type, size)` (`device.cpp:98-105`).

### 5.5 Energy

`set_pop_status()` accumulates energy from the number of DRAM commands an op issues × per-command
energy (`module_graph.cpp:254-274`): `act/read/write` and their all-bank variants
`all_act/all_read/all_write`, plus `mac_energy = flops × kMAC`. The per-command energy tables are
`gpuEnergy/logicEnergy/pimEnergy` in `power.h`, scaled by the number of pseudo-channels (§9.7).
Command counts come from either Ramulator (ON) or `run_ideal` (OFF, analytical).

---

## 6. Hardware / Cluster Layer

### 6.1 Topology

- `Cluster::CreateNode` creates `num_node` `Node`s, and each `Node` creates `num_device` `Device`s
  (`cluster.cpp:1105-1109`, `node.cpp:45-51`).
- `num_total_device = num_device × num_node`. `get_device(rank)`: `node_id = rank/num_device`
  (`cluster.cpp:19,22-25`).
- **Interconnect semantics**: `device_ict_*` = NVLink (intra-node, GPU↔GPU),
  `node_ict_*` = InfiniBand (inter-node). (There's a confusing naming quirk: `Node` stores its own
  intra-node link in a member named `node_ict_*`, but the value assigned is actually the NVLink
  value — however, actual communication code reads `device->config.*_ict_*` directly, so this is
  harmless.)

### 6.2 Hardware Presets (`hardware_config.h`)

Four static `SystemConfig` presets. Bandwidth in B/s, latency in ns, FLOPS in FLOP/s, capacity in
bytes.

| Field | A100 | H100 | B100 | B200 |
|---|---|---|---|---|
| `compute_peak_flops` (FP16) | 312 T | 989.4 T | 1750 T | 2250 T |
| `memory_bandwidth` | 2.039 TB/s | 3.352 TB/s | 8.0 TB/s | 8.0 TB/s |
| `memory_capacity` | 80 GiB | 80 GiB | 192 GiB | 192 GiB |
| `device_ict_bandwidth` (NVLink) | 150 GB/s | 450 GB/s | 900 GB/s | 900 GB/s |
| `device_ict_latency` | 3.0 µs | 0.8 µs | 0.8 µs | 0.8 µs |
| `node_ict_bandwidth` (InfiniBand) | 50 GB/s | 50 GB/s | 50 GB/s | 50 GB/s |
| `node_ict_latency` | 130 ns | 130 ns | 130 ns | 130 ns |
| `logic_x` / `logic_op_b` | 4 / 8 | 4 / 8 | 4 / 8 | 4 / 8 |
| `pim_x` / `pim_op_b` | 16 / 1 | 16 / 1 | 16 / 1 | 16 / 1 |
| `num_cube` / `num_logic_cube` | 5 / 5 | 5 / 5 | 8 / 8 | 8 / 8 |
| Default `num_node`,`num_device` | 1, 2 | 1, 2 | 1, 2 | 1, 2 |

(Evidence: `hardware_config.h:163-313`.) The `nvlink_gen`/`infiniband_gen` settings can override
`device_ict_*`/`node_ict_*` at runtime (`eval/test.cpp:59-89`).

### 6.3 Weight Distribution (Parallelism)

Two tensor-parallel degrees plus data-parallel:

- `ne_tp_dg` = non-expert TP degree, `e_tp_dg` = expert TP degree (`model_config.h:83-84`,
  config `distribution.*_tensor_degree`).
- `dp_degree = num_total_device / ne_tp_dg` (`scheduler.cpp:14-15`); each device runs the shard
  `dp_rank = device_total_rank / ne_tp_dg` (`device.cpp:72`).
- TP splitting happens inside the parallel modules: `ColumnParallelLinear` divides the output
  dimension by `parallel_num`, `RowParallelLinear` divides the input dimension by `parallel_num`,
  and attention divides the number of heads by `parallel_num` (`parallel.cpp`).
- Expert parallelism: experts per device = `num_routed_expert / (parallel_num / e_tp_dg)`; each
  device instantiates only its own share of expert FFNs (`expert.cpp:48-86`).

### 6.4 Communication Time

- **AllReduce** (ring): `hop=(P-1)×2`, per-hop = `device_ict_latency +
  (size/P)/device_ict_bandwidth` (`communication.cpp:18-49`). Always uses NVLink — TP spanning
  nodes is modeled optimistically.
- **All-to-All** (expert): `MoEScatter`/`MoEGather` compute per-destination token volume from
  `local_num_token_in_expert`, separating intra-node (NVLink) from inter-node (InfiniBand) via a
  node-id comparison. Prefill uses `max(intra,inter)`, decode uses a single path. DeepSeek uses
  FP8 dispatch / BF16 combine, so the combine size is doubled (`communication.cpp:100-465`).
- **Sync/Sync__**: A pure barrier (§5.3).

### 6.5 Attention Timing Kernels (`*_impl.cpp`)

Three kernel variants per regime (each with GPU/LOGIC/PIM versions, function-pointer table in
`executor.cpp:15-133`):

| Kernel | When | What it measures |
|---|---|---|
| `AttentionSum` | prefill (*sum*) | QK^T·softmax·context over the entire prompt. Per sequence, `m=num_process_token, n=current_len+m`. Softmax is compute-only (zero memory traffic). |
| `AttentionGen` | decode (*gen*) | Loops per KV head, modeling the accumulated KV cache as a single bulk read (memory-bound). Reflects GQA group replication. |
| `AttentionMixed` | colocated single-pass | Computes score+context in one pass (no softmax term). The LOGIC version is an **empty stub (returns 0)**. |

MLA variants are separate as `MultiLatentAttention*`, `AbsorbMLA*`. `use_flash_mla` takes a branch
that avoids materializing the score matrix, reducing memory volume (the graph shape is unchanged,
only timing changes).

> ⚠️ **PIM decode-attention is not a proper roofline calculation**: `AttentionGenExecutionPIM` uses
> `total_duration += accumul_memory_duration × opb` (`attention_gen_impl.cpp:424,474`).
> Since PIM has `op_b=1`, this effectively equals the compute-bound time, but because it isn't a
> `max()`, it can underestimate the memory lower bound at extremely low Op/B. GPU/LOGIC decode use
> the normal `max()`.

### 6.6 Stat / Output

`struct Stat` (`stat.h`) holds timing (`time/latency/queueing_delay/arrival_time`), batch
counters, a **per-stage latency breakdown** (qkv_gen, atten_sum, atten_gen, o_proj, ffn,
expert_ffn, communication, rope, layernorm, residual, plus MLA details), and energy (FC/Attn/MoE ×
DRAM/COMP). `setTimeBreakDown` scrapes timestamps by name from device 0's timeboard, aggregates
them, and multiplies by `num_total_device` (`cluster.cpp:684-1082`).

---

## 7. Model / Module Layer

### 7.1 Graph Assembly

The `LLM` constructor (`llm.cpp:10-62`): `Embedding` → N×(`Decoder` or `MoEDecoder`) → `LmHead`.
- Regular models: only layers where `expert_freq != 0 && layer % expert_freq == 0` are MoE
  (`llm.cpp:24`).
- DeepSeek-V3: the first `first_k_dense` (=3) layers are dense `Decoder`s, the rest are
  `MoEDecoder`s (`llm.cpp:42-54`).

`Model` builds an `LLM` for each device and runs a single symbolic forward pass with
`getMaxMetadata` (worst-case batch) to finalize the graph (`model.h:22-52`). The graph is not an
explicit DAG object but a `TopModuleGraph::module_graph` vector (a push/pop trace); data
dependencies are represented via the `Tensor::ready` flag, and cross-device dependencies via a
sync module's `dependency_tensor_list`.

### 7.2 The Tensor Abstraction

`Tensor` (`tensor.h`) has a `shape`, `tag` (`"act"`/`"weight"`/`"cache"`), `precision_byte`,
`ready`, and a backing `MemoryObject`. Byte size = `precision_byte × Π(shape)`
(`tensor.cpp:23-29`). This byte size/shape is exactly the roofline's input. `Module::get_size()`
returns per-tag sums of `[act, weight, cache]` (`module.cpp:60-93`), which `checkMemorySize` uses.

### 7.3 Anatomy of One Decoder Layer

**Dense `Decoder`** (`decoder.cpp:12-90`):
```
input_layer_norm(RMS)
  → Attention                      (regular if qk_rope_head_dim==0, else MultiLatentAttention)
      qkv_proj(ColumnParallelLinear, width = head_dim·(num_heads+2·num_kv_heads))  ← reflects GQA
      self_attention:  Split → { Sum(prefill,high) ‖ Gen(decode,low) } → Merge
      o_proj(RowParallelLinear)
      all_reduce
  → residual
  → post_attn_layer_norm(RMS)
  → feedforward(FeedForward2Way or 3Way = gated SwiGLU)
  → all_reduce
  → residual
```
**`MoEDecoder`** (`decoder.cpp:92-162`): the feedforward slot is replaced with `ExpertFFN`.
(Note: the residual slot in MoEDecoder is actually constructed as a `LayerNorm`, not a `Residual`
— see §13 quirk.)

### 7.4 Attention: MHA / GQA / MQA / MLA

- Head geometry: group degree = `num_heads / num_kv_heads`. MHA (`==num_heads`), GQA (in
  between), MQA (`==1`). Since the QKV projection width is
  `head_dim·(num_heads + 2·num_kv_heads)`, the group degree automatically shrinks the K/V
  projection and KV cache (`layer.cpp:16-19`).
- Prefill/decode split: `SelfAttentionSum` (prompt tokens, high/GPU) ‖ `SelfAttentionGen`
  (1 token/seq, low/PIM). A `Split → Sum‖Gen → Merge` subgraph (`parallel.cpp`).
- **KV cache**: `SelfAttentionGen` allocates a `{max_seq_len, head_dim}` k/v cache per (batch
  sequence, kv-head), tagged `"cache"` (`attention.cpp:24-37`). Since this is a worst-case
  allocation, it becomes the cache term in `checkMemorySize`.
- **MLA (DeepSeek)**: q/kv are down-projected into a latent space (`q_lora_rank`/`kv_lora_rank`).
  With `use_absorb`, the K up-projection is absorbed into the query
  (`attn_tr_k_up_proj`), so attention operates directly on the compressed latent. Absorb-gen
  caches only **`latent_kv{max_seq,512}` + `latent_pe{max_seq,64}`** per sequence
  (`attention.cpp:455-469`) — this is MLA's KV compression. `compressed_kv` toggles whether the
  K/V cache is physically allocated.

### 7.5 MoE Routing

- Gate: `ColumnParallelLinear(hidden → num_routed_expert)`, computed in full on every device
  (`expert.cpp:29-32`).
- **Top-k assignment and per-expert token counts** are decided per processed token in
  `BatchedSequence::update_expert` (`sequence.cpp:152-206`):
  - Measured trace (`get_expert_from_list`): replays the expert list from a CSV.
  - `skewness > 0`: `getZipfianRandomExpert` (weight `w_i ~ 1/(i+1)^skewness`, discrete
    distribution) — Zipfian skew.
  - Otherwise: `getRandomExpert` (uniform). (All are seeded with a fixed seed of 777, for
    reproducibility.)
- `Route::forward` builds a `{num_token_in_expert[e], hidden}` tensor per expert
  (`route.cpp:227-233`) — **skew directly translates into uneven GEMM sizes**, which feeds into
  timing.
- Shared experts (DeepSeek/Llama4) always run over the full token set (`expert.cpp:199-202`).
  Expert parallelism is modeled via `MoEScatter` (dispatch)/`MoEGather` (combine) all-to-all
  movement.

### 7.6 Model Presets (`model_config.h`)

Constructor argument order: `hidden, head_dim, layers, heads, kv_heads, max_seq, interm,
expert_interm, act_factor, precision, n_routed_expert, n_shared_expert, expert_freq,
top_k, ffn_way, first_k_dense, q_lora, kv_lora, qk_nope, qk_rope, n_vocab, compressed_kv,
use_absorb, skewness, name` (`model_config.h:12-22`).

| Model | L | hidden | interm | heads | kv | grp | N_ex | shared | top_k | freq | MLA | precision | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **mixtral** (default) | 32 | 4096 | 14336 | 32 | 8 | 4 | 8 | 0 | 2 | 1 | – | FP16 | Matches paper Table I |
| openMoE | 32 | 3072 | 12288 | 24 | 24 | 1(MHA) | 32 | 0 | 2 | 4 | – | FP16 | act_factor 2 |
| llama7bMoE | 32 | 4096 | 11008 | 32 | 32 | 1(MHA) | 16 | 0 | 2 | 1 | – | FP16 | expert_interm 688 |
| **grok1** | 64 | 6144 | 32768 | 48 | 8 | 6 | 8 | 0 | 2 | 1 | – | FP16 | Matches paper Table I |
| glam | 32 | 4096 | 16384 | 32 | 32 | 1(MHA) | 64 | 0 | 2 | 2 | – | FP16 | ffn_way 2 |
| **deepseekV3** | 60 | 7168 | 18432 | 128 | 128 | 1 | 256 | 1 | 8 | 1 | q1536/kv512/nope128/rope64 | **FP8** | first_k_dense 3, absorb |
| llama3_405B | 126 | 16384 | 53248 | 128 | 8 | 16 | 0(dense) | 0 | 0 | 0 | – | FP16 | No MoE |
| llama4_scout | 48 | 5120 | 16384 | 40 | 8 | 5 | 16 | 1 | 1 | 1 | – | FP16 | max_seq 10M |
| llama4_maverick | 48 | 5120 | 16384 | 40 | 8 | 5 | 128 | 1 | 1 | 2 | – | FP16 | max_seq 1M |

(Evidence: `model_config.h:91-121`.) Verified in the code that this exactly matches Mixtral
(deg_grp=32/8=4) and Grok1 (48/8=6) in the paper's Table I.

---

## 8. Scheduler Layer (Continuous Batching)

Terminology: **sum = prefill (summarization)**, **gen = decode (generation)**.

### 8.1 Sequence and Its Lifecycle

Core fields of `Sequence` (`sequence.h`, `sequence.cpp:6-31`):

| Field | Meaning |
|---|---|
| `input_len` / `output_len` | Prompt length / number of generated tokens |
| `total_len` | Completion target = `input_len + output_len - 1` |
| `current_len` | Context length processed so far. **The prefill↔decode discriminator**: `< input_len` → sum, `>= input_len` → gen |
| `num_process_token` | Tokens processed this iteration. Reset to 0 after `update()` |
| `sum_stage` | True during prefill, becomes false at `current_len==input_len` |
| `arrival_time`/`first_token_time`/`end_token_time` | Accumulate arrival / TTFT / E2E latency |

Lifecycle: `Create` → `sequence_queue` (waiting) → joins a `BatchedSequence` (running) via
`fillRunningQueue` → `update` each step (advances `current_len`) → first token (t2ft) recorded at
`current_len==input_len` → completion pop (e2e) at `current_len==total_len`.

### 8.2 Continuous Batching and "Prefill-First" Behavior

- `running_queue` = one `BatchedSequence` per DP rank (`scheduler.cpp:33-40`).
- `fillRunningQueue` pulls from the front of `sequence_queue` every iteration and round-robins
  requests into the batch (`scheduler.cpp:357-381`). Slots freed by completion are refilled
  immediately = continuous batching.
- **Key point (and caveat)**: In colocated mode (`disagg_system==false`), `setMetadata` sets
  `process_gen=false` whenever `hasSumSeq()` is true (i.e. if any DP batch has even one prefill
  request left) (`scheduler.cpp:267-275`). This means gen sequences don't receive
  `num_process_token` this step (the previous `update` already reset it to 0,
  `sequence.cpp:46`), yielding 0 tokens.
  → **This simulator's "mixed" mode is not vLLM-style simultaneous prefill+decode batching, but
  rather a strict alternation: if any prefill remains, the entire step is prefill; otherwise the
  entire step is decode.** (The CSV's `mixed` column merely indicates whether this step had
  prefill mixed in.) Keep this firmly in mind when interpreting results.

### 8.3 Batch/Token Limits

- `max_batch_size` (config) → `total_batch_size`, `batch_size_per_dp = total_batch_size /
  dp_degree`. The upper bound on concurrent sequences.
- `max_process_token` (config, defaults to `65536×8` if 0) → **prefill token budget per
  iteration**, split evenly across sum sequences: `min(max_process_token/num_sum_seq,
  input_len-current_len)` (`scheduler.cpp:296-304`) = chunked prefill.
- `8192` (hardcoded `num_max_batched_token`) → mainly used for graph sizing
  (`getMaxMetadata`, `scheduler.cpp:251-256`). Does not limit the runtime prefill batch.

### 8.4 Request Arrival / Injection

- `getActualArrivalTime` samples inter-arrival times from a `std::poisson_distribution` (mean
  `1e9/request_per_second` ns) and builds cumulative arrival times (`scheduler.cpp:489-509`).
- However, **the default path is closed-loop (saturated)**: since `use_inject_rate` is not
  enabled in config (statically defaults to false), `fillSequenceQueue` unconditionally refills
  `total_batch_size - getBatchSize()` requests every step (`scheduler.cpp:471-474`). (The config
  key `injection_rate` is a dead key never read by the code.)
- `use_inject_rate` must be enabled to activate arrival-time-based injection (for QPS
  experiments).

### 8.5 Memory Capacity Limit

`Cluster::checkMemorySize` (runs once at startup) checks whether `activation+weight+KV cache >
memory_capacity`:
- If so, `exit_out_of_memory` → returns true, and main aborts (`eval/test.cpp:338-341`).
- Otherwise, `mem_cap_limit` → recomputes a feasible `max_batch_size` as `avail_capacity /
  kv_cache_size_per_seq`, shrinking the scheduler's batch **once** (`cluster.cpp:280-288`).
  (Note this is a one-time adjustment at startup, not dynamic runtime OOM handling.)

### 8.6 Prefill / Decode / Disagg Modes

- `prefill_mode`: sets `output_len=input_len, total_len=input_len`, producing a prefill-only
  workload (`scheduler.cpp:84-87`).
- `decode_mode`: starts with `current_len=input_len`, going straight into a decode workload
  (`scheduler.cpp:88-90`).
- `disagg_system` (Splitwise-style): `setMetadata` fills sum and gen simultaneously, and the
  cluster accounts for the sum machine and gen machine on separate timelines
  (`sum_machine_time` vs `total_time`) via `runIterationSumGenSplit` (`cluster.cpp:527-607`).

### 8.7 Scheduler ↔ Cluster Interface

`setMetadata()` returns a `std::vector<BatchedSequence::Ptr>` (= `running_queue`), which
`cluster->run(metadata)` consumes. Each device streams its `dp_rank` shard's `BatchedSequence`
into the graph, and modules read `get_sum_process_token()`/`get_gen_process_token()`/
`current_len`/`num_token_in_expert` to determine tensor shapes → roofline input.

---

## 9. DRAM / Ramulator / PIM Layer

> This layer only affects timing **when `use_ramulator` is enabled**. The default is `off`
> (config.yaml), in which case memory time is computed via the analytical formula in §5
> (`bytes/bandwidth`), and the DRAM layer only produces **command counts for energy purposes**
> analytically (`run_ideal`). The following describes the path when enabled.

### 9.1 Ramulator Integration Boundary

- Ramulator 2.0 is a git submodule (`.gitmodules`, `src/dram/ramulator2`) and is not checked out
  by default — it must be checked out via `git submodule update` and patched before building
  (§12).
- `DRAMInterface` (`dram_interface.{h,cpp}`) is the sole wrapper; it holds `IFrontEnd`/
  `IMemorySystem` pointers and instantiates them from `dram_config*.yaml` (impl:
  `PIMController`/`PIMDRAMSystem`/`HBM3`/`PIM_DRAM_controller`/`PIM_Scheduler`/`RoBaRaCoCh`).
- Request flow: a higher-level op (e.g. GEMV) is lowered via `GeneratePIMCommand` (PIM kernel)
  into a `PIMRequest` (= list of PIMCommands) → sent via `frontend->send` →
  `memory_system->tick()` repeated until `is_finished()` → the resulting `end-start` cycle count
  is multiplied by `memory_scale_factor` (= tCK ns) to yield `memory_duration`; ACT/READ/WRITE/
  ALL_* command counts are recovered for energy purposes.

### 9.2 What the Patch Does (`patch/ramulator2_pim.patch`)

Extends Ramulator to support Logic-PIM's "simultaneously read an entire bank bundle" mechanism
(summary):
- Adds a **rank-level** view and 4 PIM commands (`ALL-ACT/ALL-PRE/ALL-RD/ALL-WR`) to `HBM3`. A
  single `ALL-ACT` opens the same row across all banks in a rank, and `ALL-RD` reads them all at
  once.
- `PIM_DRAM_controller` routes `AllRead/AllWrite` to a separate PIM buffer, ending the drain via
  `is_finished()`.
- Embeds `llm_system::PIMCommand` into `Request`, and returns command counts upstream via the
  `issued_dram_cmd` histogram.

### 9.3 PIM Kernels and the Bandwidth Model

- `pim_kernel.cpp:18-40` **registers only `kRead`/`kWrite`/`kGEMV`** (the remaining CKKS/HE-family
  kernels are commented out — a legacy of this codebase).
- `GEMV.cpp` models the bandwidth multiplier via **address masking**: it iterates over 32B
  bundles of the weight `MemoryObject`, issuing a `kMAC` command only for bundles whose
  `addr_vec` matches the mask.
  - `bandwidth_x==16` (bank-PIM): keeps only the 1 case where `addr_vec[0..4]==0` → parallel
    across all banks/bank-groups/ranks/channels.
  - `==8` (bank-group): additionally requires `addr[4]%2==0`.
  - `==4` (Logic-PIM family): requires `addr[0..3]==0`; LOGIC uses a `getAddrVec(idx, LOGIC)`
    hash.
  - The reduced command count shortens the Ramulator drain time → effective bandwidth increases.
    (In the analytical path this is achieved equivalently via `bandwidth × {logic_x, pim_x}`.)

### 9.4 The Three PIM Architectures

| Architecture | Code | Bandwidth multiplier | Energy table |
|---|---|---|---|
| bank-PIM | `PIM` + `pim_x=16` | 16× | `pimEnergy` |
| bank-group-PIM | `PIM` + `pim_x=4/8` | 4/8× | (no dedicated table → shares pimEnergy) |
| Logic-PIM | `LOGIC` + `logic_x=4` | 4× | `logicEnergy`, uses a separate LOGIC address hash |

### 9.5~9.7 Data Structures · Address Mapping · Energy

- `MemoryObject` (tensor allocation), `DRAMRequest` (higher-level op), `PIMRequest/PIMCommand`
  (lowered command stream), `MMapController` (RoBaRaCoCh address decomposition + bump allocation).
  `memory_config.h` holds HBM geometry presets (`hbm3_80GB`, `hbm3e_192GB`); `power.h` holds
  per-command energy.
- Energy = command count × per-command energy (scaled by the number of pseudo-channels). All-bank
  commands represent wide reads via `×8` (logic) / `×32` (pim) multipliers.
- `data_object.*` is dead code excluded from the build (ignore it).

---

## 10. Configuration Reference (config.yaml)

| Section | Key | Meaning |
|---|---|---|
| model | `model_name` | Selects a preset: `mixtral`/`grok1`/`deepseekV3`/`llama4_scout`, etc. |
| system | `gpu_gen` | `A100`/`H100`/`B100`/`B200` → SystemConfig preset |
| | `nvlink_gen` / `infiniband_gen` | Override device_ict / node_ict bandwidth & latency |
| | `num_node` / `num_device` | Topology |
| | `processor_type` | `GPU`/`LOGIC`/`GPU+LOGIC`(Duplex)/`GPU+PIM` |
| distribution | `expert_tensor_degree` / `none_expert_tensor_degree` | `e_tp_dg` / `ne_tp_dg` |
| optimization | `parallel_execution` | Simultaneous GPU‖PIM execution (separate high/low tracks) |
| | `hetero_subbatch` | Heterogeneous sub-batch distribution |
| | `disagg_system` | Splits prefill/decode (Splitwise) → SumGenSplit loop |
| | `use_low_unit_moe_only` | Restricts MoE to low-Op/B units only |
| | `use_ramulator` | Compute memory time via Ramulator cycles (default off = analytical) |
| | `compressed_kv` / `use_absorb` | MLA options (K/V compressed cache / weight absorb) |
| | `use_flash_mla` / `use_flash_attention` | Flash timing branches |
| | `reuse_kv_cache` / `kv_cache_reuse_rate` | Adjusts `current_len` initial value via prefix sharing |
| | `prefill_mode` / `decode_mode` | Pure prefill / pure decode workload (mutually exclusive) |
| serving | `max_batch_size` | Upper bound on concurrent sequences |
| | `max_process_token` | Prefill token budget/iteration (0 → default 65536×8) |
| simulation | `data` | `synthesis` (synthetic) or a trace name |
| | `input_len` / `output_len` | Synthetic request lengths |
| | `precision_byte` | 1 (FP8/INT8) or 2 (FP16); if 1, peak FLOPS ×2 |
| | `skewness` | Zipfian expert skew (0=uniform) |
| | `iter` | Number of simulation steps |
| | `injection_rate` | (⚠️ currently a dead key, not read by the code) |
| | `exit_out_of_memory` / `mem_cap_limit` | Abort on OOM / shrink batch |
| log | `print_log` / `export_gantt` | Print timeboard to stdout / dump gantt |
| | `output_directory` / `gantt_directory` | Output paths |

---

## 11. Outputs: CSV Columns · Gantt/Timeboard

**CSV**: one row = one step (or a completion/first-token event). Header (`cluster.cpp:428-436`):

```
iter_info, split, type, time, latency, queueing_delay, arrival_time, seq_queue_size,
input_len, output_len, num_sum_iter, mixed, batchsize, numtoken, num_sum_seq, num_gen_seq,
seqlen, sum_attention_opb,
qkvgen, q_down_proj, kv_down_proj, kr_proj, q_up_proj, qr_proj, kv_up_proj, tr_k_up_proj,
v_up_proj, atten_sum, atten_gen, o_proj, ffn, expert_ffn, communication, rope, layernorm,
residual,                                    ← per-stage latency breakdown up to here (ns)
act_energy, read_energy, write_energy, all_act_energy, all_read_energy, all_write_energy,
mac_energy, total_energy,                     ← energy (nJ)
fc_dram, fc_comp, attn_dram, attn_comp, moe_dram, moe_comp,   ← FC/Attn/MoE × DRAM/COMP breakdown
OOM
```

- `type`: `t2t` (token-to-token, per step), `t2ft` (time-to-first-token), `e2e` (end-to-end),
  `sum` (the sum machine in disagg mode).
- **Gantt/Timeboard**: when `export_gantt` is set, per-device the entire hierarchical tree of ops
  (indentation + duration, start–end µs, tensor shape, util/Op-B/processor) is dumped as text
  (`timeboard.cpp`). This is not a graphical Gantt chart but a **hierarchical trace**. `print_log`
  prints device 0's trace to stdout.

---

## 12. Build & Run

Per the README:

```bash
git clone https://github.com/scale-snu/LLMSimulator.git && cd LLMSimulator
git submodule update --init --recursive        # Checks out Ramulator 2.0
cd src/dram/ramulator2
git apply ../../../patch/ramulator2_pim.patch   # PIM extension patch
cd ../../../
mkdir build && cd build
cmake .. && make -j                             # Output: build/run
```

To run, edit the `config.yaml` copied into `build/`, then:

```bash
./run > test.log          # or  ./run <other_config.yaml>
```

- Compiler: g++ 11.4.0, cmake, clang++ (per README).
- With `use_ramulator: off`, the simulator runs purely on the analytical path, no submodule/patch
  required (most performance/energy sweeps use this path).
- CMake copies `dram_config_HBM3_80GB.yaml`, `dram_config_HBM3E_192GB.yaml`, and `config.yaml`
  into the build directory (`CMakeLists.txt:28-30`).

---

## 13. Modeling Simplifications · Known Quirks (Important)

Points that **could distort result interpretation** when using the simulator as a research tool.
All confirmed directly against the code.

1. **"mixed" is not truly mixed (prefill takes priority)** — In colocated mode, if even one
   prefill remains, the entire step is prefill and decode gets 0 tokens. This is not vLLM-style
   simultaneous batching (`scheduler.cpp:267-275`, `sequence.cpp:46`). See §8.2.
2. **Jitter on synthetic request lengths is commented out** — `pushDummySeq` computes a Gaussian
   delta but the line applying it is commented out (`scheduler.cpp:59-62`). This means all
   synthetic sequences have **identical** input/output lengths.
3. **The `injection_rate` config key is dead**, and `use_inject_rate` is also never enabled in
   config → the default is closed-loop saturation (§8.4). QPS experiments require a code change.
4. **Arrival model**: inter-arrival times are drawn using `poisson_distribution` (strictly
   speaking this should be exponential for a Poisson process). Since the mean is large, this is
   effectively close to equally spaced.
5. **Decoder residual wiring bug** — `Decoder::forward` calls `residual_1` twice and never uses
   `residual_2`; `post_attn_layer_norm` receives `attention_out` instead of `res_1_out`
   (`decoder.cpp:82-86`). **No timing impact** (two residuals are still traced at the same cost).
   This only affects qualitative correctness.
6. **`MoEDecoder`'s residual slot is actually a `LayerNorm`** (the dense `Decoder` uses
   `Residual`) — MoE layers incur RMSNorm cost instead of an add (`decoder.cpp:124,138`).
7. **`LLM::forward` passes the embedding output directly to every decoder** (it does not chain the
   previous decoder's output, `llm.cpp:80`). Since all hidden-state shapes are identical, this has
   **no timing impact**.
8. **`attention_group_size` inconsistency**: gen uses `num_heads/num_kv_heads` (correct);
   sum/mixed/MLA-sum use `head_dim/num_heads` (`attention.cpp:107,149,420,548`). The sum kernel
   uses `num_heads` directly so this is largely harmless, but it's a source of confusion.
9. **PIM decode-attention uses `memory×opb` instead of `max()`** (§6.5,
   `attention_gen_impl.cpp:424`).
10. **Dead code**: `data_object.*` (excluded from build), `pimBankgroupEnergy` (never
    registered), most PIM kernels (CKKS legacy), `getNumInjection`/`getPoissondistribution`
    (no callers).
11. **Memory-limited batch shrinking happens once at startup**, not as dynamic runtime OOM
    handling (§8.5).
12. **A100 has no DRAM YAML branch** — the combination of `use_ramulator` + A100 may not have been
    intended (the analytical path is recommended, `device.cpp:34-57`).

---

## 14. File Map (Quick Reference)

### scheduler (`src/scheduler/`)
| File | Role |
|---|---|
| `sequence.{h,cpp}` | `Sequence` (request) · `BatchedSequence` (batch) · expert routing (uniform/Zipfian) |
| `scheduler.{h,cpp}` | Continuous batching, setMetadata (decides batch/tokens), arrival/injection, warm-up |

### hardware (`src/hardware/`) — timing core
| File | Role |
|---|---|
| `hardware_config.h` | `SystemConfig` + A100/H100/B100/B200 presets |
| `base.h` | `ProcessorType` enum, etc. |
| `cluster.{h,cpp}` | Topology, runIteration loop, checkMemorySize, timebreakdown, CSV/gantt |
| `node.{h,cpp}` / `device.{h,cpp}` | Node (intra-node) · Device (owns graph/status/dram, runs DP shard) |
| `executor.{h,cpp}` | LayerType×ProcessorType function table, min-duration selection |
| `linear_impl.cpp` | **The canonical roofline implementation** (GPU/LOGIC/PIM, including batched) |
| `activation_impl.cpp` | Activation timing |
| `attention_{sum,gen,mixed}_impl.cpp` | Prefill/decode/colocated attention (+MLA/absorb) |
| `layer_impl.{h,cpp}` | Shared helpers such as `issueRamulator`/`getIdealMemoryStatus` |
| `stat.h` | `Stat` output record |

### model (`src/model/`)
| File | Role |
|---|---|
| `model_config.h` | `ModelConfig` + 9 model presets |
| `model.h` | `Model`: symbolically builds LLM per device |
| `llm.{h,cpp}` | Assembles Embedding + N×Decoder/MoEDecoder + LmHead |

### module (`src/module/`)
| File | Role |
|---|---|
| `module.{h,cpp}` / `base.h` | `Module` base (trace push/pop, get_size) |
| `module_graph.{h,cpp}` | `TopModuleGraph` (op list, time accumulation, sync barrier, energy) |
| `tensor.{h,cpp}` | `Tensor` (shape/tag/precision → bytes) |
| `timeboard.{h,cpp}` | Per-op timestamp/energy tree, gantt dump |
| `attention.{h,cpp}` / `layer.{h,cpp}` | Attention variants (MHA/GQA/MLA/absorb, Split/Sum/Gen/Merge) |
| `expert.{h,cpp}` / `route.{h,cpp}` | MoE expert FFN, gating · top-k · expert distribution |
| `linear.{h,cpp}` / `parallel.{h,cpp}` | Linear/BatchedLinear, TP (Column/Row parallel) |
| `communication.{h,cpp}` | AllReduce/All-to-All/Sync communication time |
| `decoder.{h,cpp}` | Dense/MoE decoder blocks |
| `embedding/lm_head/layernorm/rope/residual/activation/compressed_kv_restore.{h,cpp}` | Remaining sub-modules |

### dram (`src/dram/`)
| File | Role |
|---|---|
| `dram_interface.{h,cpp}` | Ramulator wrapper (request lowering → tick → recover cycle/energy) |
| `pimkernel/{pim_kernel,GEMV,Read,Write}.cpp` | Higher-level op → PIM commands (address-masking bandwidth model) |
| `memory_object/mmap_controller/*_request.{h,cpp}` | Tensor allocation · RoBaRaCoCh address mapping · request structures |
| `memory_config.h` / `power.h` | HBM geometry presets / per-command energy |
| `../../patch/ramulator2_pim.patch` | Adds PIM commands and all-bank read to Ramulator |

### Entry point / Configuration
| File | Role |
|---|---|
| `eval/test.cpp` | `main()` = the `run` executable |
| `config.yaml` | Runtime configuration |
| `dram_config_HBM3_80GB.yaml` / `dram_config_HBM3E_192GB.yaml` | Ramulator DRAM config (H100 / B100·B200) |
| `src/*/test.cpp` | Per-subsystem unit tests (not `main`) |

---

### Appendix: Verification Method

The claims in this document were verified after 4 sub-agents each analyzed the code by subsystem,
with the author directly cross-checking the following core paths against the source: roofline
timing (`linear_impl.cpp` in full), hardware presets (`hardware_config.h` in full), time
accumulation/sync (`module_graph.cpp` in full), processor selection (`executor.cpp`),
prefill-first batching (`scheduler.cpp`+`sequence.cpp`), attention pinning/KV cache
(`attention.cpp`), graph assembly (`model.h`+`llm.cpp`), the execution loop (`cluster.cpp`), PIM
kernels/attention quirks (`GEMV.cpp`, `pim_kernel.cpp`, `attention_gen_impl.cpp`), and
model/decoder wiring (`model_config.h`, `decoder.cpp`). The agents' reports were confirmed to
match the actual code, and all quirks in §13 were confirmed to be reproducible.
