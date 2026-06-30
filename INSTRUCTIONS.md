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
. This 320-MB SRAM limit will act as a hard constraint on the maximum batch size for these configurations, particularly in short-context workloads
.
Read Prefetching: Every individual flash stack (in all flash configurations) has a dedicated 3.13-MB SRAM staging buffer on its logic die
. For an 8-stack configuration, this totals 25.04 MB of SRAM per GPU
. Implement this strictly as a double-buffer to prefetch read data by chunking the KV cache to fit within these buffers, allowing you to pay the µs-scale page-read latency only once per chunk to hide the delay
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
. (Note: The bidirectional nature of the 1,800 GB/s NVLink is standard architecture specification explicitly requested for this implementation).
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
Implement a constraint-aware optimizer that sweeps combinations of data, tensor, pipeline, and expert parallelism to maximize throughput subject to GPU memory capacity, SRAM limits, and SLO constraints
. For all normalized metrics, use the 8-GPU HBM4 configuration as the baseline (value = 1.0)
. Process the simulation output to calculate the following five metrics:
1. Maximum Per-GPU Batch Size: Do NOT cap the batch size sweep at an arbitrary number like 128 (e.g., HBF+ can reach 327)
. Use a dynamic two-phase search:
Phase 1 (Exponential Bounds): Start at B=1 and double iteratively (1,2,4,8,16…) until the simulation fails. A failure occurs if NO parallelism configuration can simultaneously satisfy the target TPOT SLO, the total GPU-memory capacity, and the specific intermediate data limit (either the 1 HBM stack capacity for HBF/CONV or the 320-MB total logic SRAM limit for HBF+/CONV+)
.
Phase 2 (Binary Search): Once you find the first failing batch size (B 
fail
​
 ) and the last successful one (B 
success
​
 ), perform a binary search between them to pinpoint the exact maximum integer batch size.
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
