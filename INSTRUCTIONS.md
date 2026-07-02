1. Hardware & Memory Configurations
Implement an asymmetric memory timing model that individually configures read and write bandwidths for NAND flash
. Create the following five memory presets per GPU:
HBM4: 8 HBM stacks, 288 GB total capacity
. Symmetric bandwidth: 12.8 TB/s read, 12.8 TB/s write
.
HBF: 1 HBM stack and 7 HBF stacks, 3,620 GB total capacity
. Bandwidth: 11.2 TB/s HBF read, 0.112 TB/s HBF write
. Assume 1-µs 4-KiB page-read latency and 100-µs page-program latency
.
HBF+: 0 HBM stacks and 8 HBF stacks, 4,096 GB total capacity
. Bandwidth: 12.8 TB/s HBF read, 0.128 TB/s HBF write
. Assume 1-µs 4-KiB page-read latency and 100-µs page-program latency
.
CONV: 1 HBM stack and 7 flash stacks, 3,620 GB total capacity
. Bandwidth: 2.45 TB/s flash read, exactly 0.073 TB/s flash write
. Assume 3-µs page-read latency and 100-µs page-program latency
.
CONV+: 0 HBM stacks and 8 flash stacks, 4,096 GB total capacity
. Bandwidth: 2.80 TB/s flash read, exactly 0.084 TB/s flash write
. Assume 3-µs page-read latency and 100-µs page-program latency
.
2. Data Placement and Buffer Policies
Model Weights & KV-Cache: Because they require massive capacity and are read-dominant, these must be placed directly on the flash stacks in the HBF, HBF+, CONV, and CONV+ configurations
.
Intermediate Data (Crucial Constraint): This short-lived data must NEVER be written to the flash memory to avoid severe wear and performance degradation
. For HBF and CONV, map all intermediate data to the 1 reserved HBM stack
. For HBF+ and CONV+, map it to a dedicated 40-MB SRAM buffer located on the logic die of each HBF stack
. This means an 8-stack HBF+/CONV+ setup has a total of 320 MB of SRAM per GPU dedicated to intermediate data
. This 320-MB SRAM limit will act as a hard constraint on the maximum batch size for these configurations
.
Gate this constraint on the PEAK concurrently-live intermediate-data footprint for a single decode step, not the sum of every intermediate tensor a layer touches
. The source paper (Son et al., IEEE CAL 2026) is explicit that "much of the intermediate data has a short lifetime, which enables quick release of SRAM-buffer space" and that "a larger batch size ... linearly increases the peak size of intermediate data" — i.e. tensors are produced and consumed sequentially within a layer (attention phase, then FFN phase), so the resident set at any instant is the residual stream plus whichever phase is currently executing, never the sum of every op's output
. Concretely: take max(attention-phase footprint, FFN-phase footprint), not their sum, and this peak scales linearly with batch size, independent of sequence length
. Crucially, the query·key^T attention-score matrix (and any decompressed KV) must NOT be counted against this 320-MB pool: per the Read Prefetching policy below, that data is chunked through the separate 3.13-MB/stack double-buffer staging pool, so it is never resident in the intermediate-data SRAM regardless of context length — charging it here would make the 320-MB limit scale (incorrectly) with sequence length instead of batch size
. In practice the SRAM limit still tends to bind hardest at shorter-context workloads, but because those workloads can otherwise sustain much larger batch sizes before other constraints (compute, communication) cap them — not because the intermediate-data footprint formula itself grows with context length
.
Read Prefetching: Every individual flash stack (in all flash configurations) has a dedicated 3.13-MB SRAM staging buffer on its logic die
. For an 8-stack configuration, this totals 25.04 MB of SRAM per GPU
. Implement this strictly as a double-buffer: while one buffer streams the current chunk, the other buffer's page-read latency for the next chunk is issued concurrently and so overlaps that transfer, hiding it — except for the very first chunk (pipeline fill), which has nothing to overlap with and always exposes one full page-read latency
. This is not "zero latency after chunking" — exactly one page-read latency remains on the critical path; if a chunk's own transfer time is ever shorter than the page-read latency (e.g. an explicitly small configured chunk size), double-buffering can no longer fully hide it and the residual (latency − transfer) must be exposed per chunk as well, degrading gracefully back toward the un-hidden case
.
3. Execution Modeling & Pipeline Details
Disaggregated Prefill-Decode: Do not simulate the internal computation of prefill nodes, as they are compute-bound and benefit little from expanded memory capacity
. Focus the simulation strictly on the decode nodes
.
Continuous Batching: Implement continuous batching where the query arrival rate exactly matches the completion rate to maintain a fully saturated batch in a steady state
. Do not use microbatching.
KV-Cache Write Transfer Penalty (Critical Fix): Do not calculate write overhead based on the single token generated during decode, as that is easily overlapped with attention computation
. Instead, when newly admitted queries are transferred from the prefill nodes to the decode nodes, calculate the time required to write their entire newly generated prefill KV-cache tensors (based on input_len) to the decode node's flash memory
. Subtract any write time that can be successfully overlapped with parallel computation in the attention layer
. Only log this 'unhidden' write latency and add it to the simulation's critical path
.
Parallelism Features: Implement pipeline parallelism and chunked attention, enabling them globally across all memory configurations
. Make the chunk size a configurable parameter, not a hardcoded value.
Interconnects: Model intra-node communication via NVLink (1,800 GB/s bidirectional total bandwidth, connecting up to 8 GPUs per node) and inter-node communication via InfiniBand (100 GB/s, for scaling beyond 8 GPUs)
. (Note: The bidirectional nature of the 1,800 GB/s NVLink is standard architecture specification explicitly requested for this implementation. Unidirectionally, it should be 900 GB/s.).
4. Workload Profiles and Optimizations
The simulator must validate against two models: Llama 3 405B (dense) and Llama 4 Maverick (MoE, 746 GB)
. Run simulations across GPU counts of 1, 2, 4, 8, and 16
. Test across three specific application workloads based on their average input/output token lengths (⟨L 
IN
​
 ,L 
OUT
​
 ⟩):
SHORT (shareGPT): ⟨1660,373⟩
MID (LongBench): ⟨5900,499⟩
LONG (English summarization): ⟨103500,1100⟩
 Target SLO constraints for sweeps: 0.05s, 0.1s, 0.2s, and Offline (no constraint, max 24 hours/86400s)
.
5. Metric Calculations and Output Processing
Implement a constraint-aware optimizer that sweeps combinations of data, tensor, pipeline, and expert parallelism to maximize throughput subject to GPU memory capacity and SRAM limits, using an analytic latency estimate against the target SLO only to rank and propose candidate configurations — see Section 6 for how final SLO satisfaction must actually be determined
. For all normalized metrics, use the 8-GPU HBM4 configuration as the baseline (value = 1.0)
. Process the simulation output to calculate the following five metrics:
1. Maximum Per-GPU Batch Size: Do NOT cap the batch size sweep at an arbitrary number like 128 (e.g., HBF+ can reach 327)
. Use a dynamic two-phase search:
Phase 1 (Capacity Bound): Compute the capacity/SRAM ceiling directly — the largest batch size for which some parallelism configuration fits the total GPU-memory capacity and the specific intermediate data limit (either the 1 HBM stack capacity for HBF/CONV or the 320-MB total logic SRAM limit for HBF+/CONV+, gated on the PEAK intermediate-data footprint per Section 2, not a sum). This is exact and monotonic in batch size, so it does not require an exponential search from B=1. Starting from that ceiling, binary-search downward using the optimizer's analytic latency estimate against the target TPOT SLO (see Section 6) to propose a candidate maximum batch size and its parallelism configuration
.
Phase 2 (Simulator Verification): Confirm the candidate batch size with the actual simulator, and confirm the batch immediately above it fails, as a boundary safety check. If the simulator's measured TPOT disagrees with the analytic estimate in either direction, fall back to a simulator-driven binary search over the narrowed range until the exact maximum integer batch size is pinpointed (see Section 6).
2. System Throughput: In steady-state continuous batching using the max batch size, calculate Per-GPU Tokens Per Second (TPS) as: (Total Output Tokens) / (Total Simulation Time in Seconds) / (Number of GPUs)
.
3. Runtime Performance Breakdown: Under the 0.1s SLO, log the exact fraction of total decode execution time spent on: Attention, FFN, KV Write (unhidden transfer penalty from prefill only), Communication, and Others (layer norm, residual, LM head)
.
4. SLO Sensitivity Analysis: Using the LONG workload, evaluate how relaxing the SLO (0.05s, 0.1s, 0.2s, Offline) affects Per-GPU TPS and Batch Size
. Normalize to the 8-GPU HBM4 baseline running under the exact same SLO
.
5. Write Traffic and Endurance (PEC): Calculate the 3-Year Program/Erase-cycle Count (PEC) using steady-state write traffic: 3-Year PEC = (Bytes Written Per Second * 3600 * 24 * 365 * 3) / (System Flash Capacity)
. Evaluate this resulting figure against the 100K physical P/E cycle limit of single-level cell (SLC) NAND flash
. Ensure this accurately accounts for the fact that every newly processed token generates new KV-cache tensors that must be written to flash
.
6. Optimizer/Simulator Separation of Concerns
The parallelism optimizer (Section 5's data/tensor/pipeline/expert sweep) must never itself determine a reported metric or an SLO pass/fail decision — only the discrete-event simulator may do so. GPU-memory capacity and the SRAM/intermediate-data limit are exact and may be enforced analytically by the optimizer. Latency and SLO satisfaction may not: the optimizer's analytic latency estimate is a ranking and search heuristic only, used to propose the fastest parallelism configuration for a given batch size and to bound the batch-size search efficiently. It must never itself reject a batch size or produce a value that appears in a reported metric.
Implement the batch-size search (Metric 1) as two cooperating phases. An analytic phase estimates the maximum feasible batch size and its best parallelism configuration using only the optimizer's in-process model, without running the full simulator on every candidate batch. A simulator-verification phase then confirms this candidate — and the batch immediately above it, as a boundary safety check — by actually running the simulator. The simulator's measured TPOT is the sole SLO arbiter. If verification disagrees with the analytic estimate in either direction, fall back to a simulator-driven search over the narrowed range until the true boundary is confirmed. Continuously audit the analytic model against measured simulator latency and flag whenever it directionally overestimates (analytic latency > measured latency), since that is the direction that can cause the search to silently reject a batch size the simulator would have accepted.
Model-architecture-dependent optimizations (e.g. MLA-specific compressed KV-cache handling) must be derived from the selected model's own architecture parameters, not from a global configuration flag applied uniformly across all models. Applying MLA-specific handling to a non-MLA model corrupts its KV-cache and activation sizing.
