#pragma once
#include <string>
#include <vector>

#include "common/type.h"
#include "hardware/base.h"
#include "dram/hbf_memory_config.h"

namespace llm_system {

struct PIMHWConfig {
  ProcessorType type = ProcessorType::GPU;

  int bandwidth_x = 0;
};

class SystemConfig {

  public:
    // default initilizing with H100 config
    SystemConfig(std::string gpu_gen ="H100",
                 int num_node = 1, int num_device = 2, 
                 hw_metric node_ict_latency = 0.5 * 1000,
                 hw_metric node_ict_bandwidth = 400.0 * 1000 * 1000 * 1000,
                 hw_metric device_ict_latency = 3.0 * 1000,
                 hw_metric device_ict_bandwidth = 450.0 * 1000 * 1000 * 1000, 
                 hw_metric compute_peak_flops = 989.4 * 1000 * 1000 * 1000 * 1000,
                 hw_metric memory_bandwidth = 3.352 * 1000 * 1000 * 1000 * 1000,
                 hw_metric memory_capacity = 80.0 * 1024 * 1024 * 1024,\
                 int logic_x = 4,
                 hw_metric logic_op_b = 8,                 
                 int pim_x = 16,
                 hw_metric pim_op_b = 1,
                 std::vector<ProcessorType> processor_type = {},
                 bool parallel_execution = false,
                 bool hetero_subbatch = false,
                 ProcessorType high_processor_type = ProcessorType::GPU,
                 ProcessorType low_processor_type = ProcessorType::LOGIC,
                 bool communication_hiding = false,            
                 bool disagg_system = false,
                 bool use_low_unit_moe_only = false,
                 bool use_ramulator = false,
                 bool exit_out_of_memory = true,
                 bool mem_cap_limit = false,               
                 bool use_flash_mla = true,
                 bool use_flash_attention = true,
                 bool reuse_kv_cache = true,
                 hw_metric kv_cache_reuse_rate = 0.5,
                 bool prefill_mode = false,
                 bool decode_mode = false,
                 bool use_inject_rate = false,
                 int request_per_second = 10,
                 int num_cube = 5,
                 int num_logic_cube = 5
                )
      : gpu_gen(gpu_gen),
        num_node(num_node),
        num_device(num_device),
        node_ict_latency(node_ict_latency),
        node_ict_bandwidth(node_ict_bandwidth),
        device_ict_latency(device_ict_latency),
        device_ict_bandwidth(device_ict_bandwidth),
        compute_peak_flops(compute_peak_flops),
        memory_bandwidth(memory_bandwidth),
        memory_capacity(memory_capacity),
        logic_x(logic_x),
        logic_op_b(logic_op_b),
        pim_x(pim_x),
        pim_op_b(pim_op_b),
        processor_type(processor_type),
        parallel_execution(parallel_execution),
        hetero_subbatch(hetero_subbatch),
        high_processor_type(high_processor_type),
        low_processor_type(low_processor_type),
        communication_hiding(communication_hiding),
        disagg_system(disagg_system),
        use_low_unit_moe_only(use_low_unit_moe_only),
        use_ramulator(use_ramulator),
        exit_out_of_memory(exit_out_of_memory),
        mem_cap_limit(mem_cap_limit),
        use_flash_mla(use_flash_mla),
        use_flash_attention(use_flash_attention),
        reuse_kv_cache(reuse_kv_cache),
        kv_cache_reuse_rate(kv_cache_reuse_rate),
        prefill_mode(prefill_mode),
        decode_mode(decode_mode),
        use_inject_rate(use_inject_rate),
        request_per_second(request_per_second),
        num_cube(num_cube),
        num_logic_cube(num_logic_cube){
          logic_memory_bandwidth = memory_bandwidth * logic_x;
          pim_memory_bandwidth = memory_bandwidth * pim_x;
        };

    SystemConfig& operator=(const SystemConfig& rhs) = default;

  std::string gpu_gen;

  // Device number
  int num_node;
  int num_device;

  // Cluster specification
  hw_metric node_ict_latency;   // ns
  hw_metric node_ict_bandwidth; // B/s

  // Node specification
  hw_metric device_ict_latency;    // ns, 
  hw_metric device_ict_bandwidth;  // B/s

  // Device specification
  hw_metric compute_peak_flops;
  hw_metric memory_bandwidth;

  hw_metric memory_capacity;

  // Logic specification
  int logic_x;
  hw_metric logic_memory_bandwidth = memory_bandwidth * logic_x;
  hw_metric logic_op_b;

  // PIM specifiaction
  int pim_x;
  hw_metric pim_memory_bandwidth = memory_bandwidth * pim_x;
  hw_metric pim_op_b;

  std::vector<ProcessorType> processor_type = {};

  bool parallel_execution = false;
  bool hetero_subbatch = false;
  ProcessorType high_processor_type = ProcessorType::GPU;
  ProcessorType low_processor_type = ProcessorType::LOGIC;

  bool communication_hiding = false;

  bool disagg_system = true;
  bool use_low_unit_moe_only = false;
  bool use_ramulator = false;
  
  bool exit_out_of_memory = false;
  bool mem_cap_limit = false;

  bool use_flash_mla = true; 
  bool use_flash_attention = true; 
  bool reuse_kv_cache = true;
  hw_metric kv_cache_reuse_rate; 
  // this rate includes, 
  // 1) how long does prompt share tokens with cached KV 
  // 2) does prompt share tokens with cached KV
  // because we select rate between [0, kv_cache_reuse_rate * 2), kv_cache_reuse_rate must be max 0.5

  bool prefill_mode = false; 
  bool decode_mode = false;

  bool use_inject_rate = false;  // injection random number of sequence
  int request_per_second;

  int num_cube; //8: for HBM3E (B100), 5 for HBM3 (H100)
  int num_logic_cube;
  // Device
  std::string memory_type = "HBM";
  HBFMemoryConfig hbf_config;
  bool use_hbf = false;

  // Optimizer alignment flags (parsed from config.yaml: simulation section)
  // validate_optimizer: "off" | "warn" | "strict"
  //   off    – comparison code skipped entirely (zero output / overhead)
  //   warn   – print [OptValidation] lines when divergence exceeds threshold (default)
  //   strict – call fail() instead of printing when over threshold (for CI)
  std::string validate_optimizer = "warn";
  double validate_optimizer_threshold = 0.10;  // relative divergence threshold (10%)

  // optimizer_latency_model: "sum" | "max"
  //   sum – conservative: compute + memory (never under-predicts, but diverges
  //         further from the simulator's own std::max()-based overlap model)
  //   max – tighter: max(compute, memory) per stage, hidden behind compute; tracks
  //         the simulator's overlap behavior closely, which is what the batch-size
  //         search (run_experiments.py) relies on to rank candidates before
  //         simulator verification (default: this is a ranking heuristic only —
  //         the simulator's measured tpot is the sole SLO arbiter)
  std::string optimizer_latency_model = "max";
  double latency_margin = 1.0;  // multiplicative safety margin on the estimated latency

  // Chunked-attention chunk granularity, in BYTES of KV staged per chunk.
  //   0  – auto: use the full per-stack SRAM staging capacity
  //        (sram_per_stack_bytes * num_flash_stacks), i.e. the legacy behavior.
  //   >0 – explicit chunk size; physically clamped to the SRAM staging capacity
  //        (a double-buffer cannot stage more KV than the SRAM holds).
  // Parsed from config.yaml: system.chunk_size.
  int chunk_size = 0;

  // Compute-utilization (MFU, Model FLOPs Utilization) derating of GEMM compute time.
  // The inherited roofline model (compute_duration = total_flops/compute_peak_flops) assumes
  // every compute-bound op hits 100% of peak FLOPs, which no real GEMM does (tensor-core
  // tile/wave quantization, epilogue/reduction overhead). This is modeled as a saturating
  // curve in the GEMM row count M (batch*tokens for that specific op):
  //   MFU(M) = mfu_max * M / (M + mfu_m_half)
  //   compute_duration = total_flops / (compute_peak_flops * MFU(M)) * 1e9
  // Defaults (mfu_max=1.0, mfu_m_half=0) make MFU(M) == 1.0 for all M > 0, i.e. an EXACT
  // no-op matching the pre-existing behavior unless explicitly configured.
  //   mfu_max    – asymptotic (large-M) achieved FLOPs fraction, in (0, 1].
  //   mfu_m_half – GEMM row count at which MFU(M) == mfu_max/2 (curve's half-saturation
  //                point); 0 disables the ramp so MFU(M) == mfu_max for every M > 0.
  // Parsed from config.yaml: simulation.mfu_max / simulation.mfu_m_half.
  double mfu_max = 1.0;
  double mfu_m_half = 0.0;
};

// MFU(M) = mfu_max * M / (M + mfu_m_half) -- see SystemConfig::mfu_max/mfu_m_half above.
// m <= 0 (degenerate/empty op) returns mfu_max rather than 0 to avoid a divide-by-zero
// blow-up in callers that divide compute_peak_flops by this value.
inline double effectiveMFU(const SystemConfig& config, double m) {
  if (m <= 0.0 || config.mfu_m_half <= 0.0) return config.mfu_max;
  return config.mfu_max * m / (m + config.mfu_m_half);
}


static SystemConfig A100 = SystemConfig(
                 "A100",                            // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7 https://www.fs.com/products/161048.html?attribute=106827&id=3941024
                 50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 3.0 * 1000,                        // device_ict_latency, nvlink 3
                 150.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 312.0 * 1000 * 1000 * 1000 * 1000, // compute_peak_flops, FP16
                 2.039 * 1000 * 1000 * 1000 * 1000, // memory_bandwidth
                 80.0 * 1024 * 1024 * 1024,         // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system 
                 false,                             // use_low_unit_moe_only
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 5,                                 // num_cube
                 5                                  // int num_logic_cube
                 );

static SystemConfig H100 = SystemConfig(
                 "H100",                            // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7
                 50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency
                 450.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth, nvlink 4
                 989.4 * 1000 * 1000 * 1000 * 1000, // compute_peak_flops, FP16
                 3.352 * 1000 * 1000 * 1000 * 1000, // memory_bandwidth
                 80.0 * 1024 * 1024 * 1024,         // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 5,                                 // num_cube
                 5                                  // int num_logic_cube
                 );

static SystemConfig B100 = SystemConfig(
                  "B100",                            // gpu gen
                  1,                                 // num_node 
                  2,                                 // num_device
                  130.0,                             // node_ict_latency, connectx-7 
                  50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                  0.8 * 1000,                        // device_ict_latency, nvlink 5.0
                  900.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                  1750.0 * 1000 * 1000 * 1000 * 1000,// compute_peak_flops, FP16
                  8.000 * 1000 * 1000 * 1000 * 1000, // memory_bandwidth
                  192.0 * 1024 * 1024 * 1024,        // memory_capacity 
                  4,                                 // logic_x 
                  8,                                 // logic_op_b                 
                  16,                                // pim_x
                  1,                                 // pim_op_b
                  {},                                // processor_type
                  false,                             // parallel_execution
                  false,                             // hetero_subbatch
                  ProcessorType::GPU,                // high_processor_type
                  ProcessorType::LOGIC,              // low_processor_type
                  false,                             // communication_hiding
                  false,                             // disagg_system
                  false,                             // use_low_unit_moe_only 
                  false,                             // use_ramulator
                  true,                              // exit_out_of_memory
                  false,                             // mem_cap_limit
                  true,                              // use_flash_mla
                  true,                              // use_flash_attention
                  false,                             // reuse_kv_cache
                  0.0,                               // kv_cache_reuse_rate
                  false,                             // prefill_mode
                  false,                             // decode_mode
                  false,                             // use_inject_rate
                  10,                                // request_per_second
                  8,                                 // num_cube
                  8                                  // int num_logic_cube
                  );
                  
static SystemConfig B200 = SystemConfig(
                 "B200",                            // gpu gen
                 1,                                 // num_node 
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7 
                 50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency, nvlink 5.0
                 900.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 2250.0 * 1000 * 1000 * 1000 * 1000,// compute_peak_flops, FP16
                 8.000 * 1000 * 1000 * 1000 * 1000, // memory_bandwidth
                 192.0 * 1024 * 1024 * 1024,        // memory_capacity 
                 4,                                 // logic_x 
                 8,                                 // logic_op_b                 
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only 
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 8,                                 // num_cube
                 8                                  // int num_logic_cube
                 );

// Rubin: models the NVIDIA DGX Rubin NVL8 reference GPU that the "Exploring High-Bandwidth
// Flash for Modern LLM Inference" paper (Son et al.) cites as its "state-of-the-art GPU-core
// architecture" for all five evaluated memory configs. compute_peak_flops is ESTIMATED, not
// directly sourced: the datasheet publishes only "NVFP4 inference: 400 PFLOPS" and "FP8/FP6
// training (dense): 140 PFLOPS" aggregate across the system's 8 GPUs -- no FP16 figure is
// given. Per-GPU dense FP8/FP6 = 140 PFLOPS / 8 = 17.5 PFLOPS; halving that (matching this
// file's existing FP16-base-then-double-for-FP8 convention, see eval/test.cpp's
// "precision_byte==1: compute_peak_flops *= 2") gives an estimated FP16 base of 8.75 PFLOPS.
// memory_bandwidth/memory_capacity below are placeholders derived the same way (176 TB/s / 8,
// 2.3 TB / 8) but are NOT load-bearing: eval/test.cpp always overwrites
// system_config.memory_capacity/memory_bandwidth from the selected hbf_memory_config.h preset
// once system.memory_type is set (true in every sweep), so only compute_peak_flops (and
// device_ict_bandwidth, itself separately overridden by nvlink_gen) actually affect results.
static SystemConfig Rubin = SystemConfig(
                 "Rubin",                           // gpu gen
                 1,                                 // num_node
                 2,                                 // num_device
                 130.0,                             // node_ict_latency, connectx-7
                 50.0 * 1000 * 1000 * 1000,         // node_ict_bandwidth
                 0.8 * 1000,                        // device_ict_latency, nvlink 6.0
                 900.0 * 1000 * 1000 * 1000,        // device_ict_bandwidth
                 8750.0 * 1000 * 1000 * 1000 * 1000,// compute_peak_flops, FP16 (ESTIMATED, see comment above)
                 22.0 * 1000 * 1000 * 1000 * 1000,  // memory_bandwidth (176 TB/s / 8, not load-bearing)
                 287.5 * 1024 * 1024 * 1024,        // memory_capacity (2.3 TB / 8, not load-bearing)
                 4,                                 // logic_x
                 8,                                 // logic_op_b
                 16,                                // pim_x
                 1,                                 // pim_op_b
                 {},                                // processor_type
                 false,                             // parallel_execution
                 false,                             // hetero_subbatch
                 ProcessorType::GPU,                // high_processor_type
                 ProcessorType::LOGIC,              // low_processor_type
                 false,                             // communication_hiding
                 false,                             // disagg_system
                 false,                             // use_low_unit_moe_only
                 false,                             // use_ramulator
                 true,                              // exit_out_of_memory
                 false,                             // mem_cap_limit
                 true,                              // use_flash_mla
                 true,                              // use_flash_attention
                 false,                             // reuse_kv_cache
                 0.0,                               // kv_cache_reuse_rate
                 false,                             // prefill_mode
                 false,                             // decode_mode
                 false,                             // use_inject_rate
                 10,                                // request_per_second
                 8,                                 // num_cube
                 8                                  // int num_logic_cube
                 );

}  // namespace llm_system