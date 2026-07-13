#pragma once

namespace llm_system {

struct HBFMemoryConfig {
  HBFMemoryConfig& operator=(const HBFMemoryConfig& rhs) = default;

  HBFMemoryConfig(int num_hbm_stacks = 8,
                  int num_flash_stacks = 0,
                  unsigned long long total_capacity_bytes = 288ULL * 1024 * 1024 * 1024,
                  double hbm_read_bandwidth = 12.8e12,
                  double hbm_write_bandwidth = 12.8e12,
                  double flash_read_bandwidth = 0.0,
                  double flash_write_bandwidth = 0.0,
                  unsigned long long flash_page_read_latency_ns = 0,
                  unsigned long long flash_page_program_latency_ns = 0,
                  unsigned long long sram_per_stack_bytes = 0,
                  unsigned long long logic_sram_bytes = 0,
                  unsigned long long page_size_bytes = 4096,
                  unsigned long long hbm_per_stack_bytes = 0,
                  double logic_sram_bandwidth = 12.8e12)
      : num_hbm_stacks(num_hbm_stacks),
        num_flash_stacks(num_flash_stacks),
        total_capacity_bytes(total_capacity_bytes),
        hbm_read_bandwidth(hbm_read_bandwidth),
        hbm_write_bandwidth(hbm_write_bandwidth),
        flash_read_bandwidth(flash_read_bandwidth),
        flash_write_bandwidth(flash_write_bandwidth),
        flash_page_read_latency_ns(flash_page_read_latency_ns),
        flash_page_program_latency_ns(flash_page_program_latency_ns),
        sram_per_stack_bytes(sram_per_stack_bytes),
        logic_sram_bytes(logic_sram_bytes),
        page_size_bytes(page_size_bytes),
        hbm_per_stack_bytes(hbm_per_stack_bytes),
        logic_sram_bandwidth(logic_sram_bandwidth) {}

  int num_hbm_stacks;
  int num_flash_stacks;
  unsigned long long total_capacity_bytes;
  double hbm_read_bandwidth;
  double hbm_write_bandwidth;
  double flash_read_bandwidth;
  double flash_write_bandwidth;
  unsigned long long flash_page_read_latency_ns;
  unsigned long long flash_page_program_latency_ns;
  unsigned long long sram_per_stack_bytes;
  unsigned long long logic_sram_bytes;
  unsigned long long page_size_bytes;
  // Capacity of each individual HBM stack in bytes.
  // Total HBM capacity = num_hbm_stacks * hbm_per_stack_bytes.
  // Used by checkCapacity() and the parallelism optimizer; absent from
  // total_capacity_bytes which tracks flash-pool capacity on HBF presets.
  unsigned long long hbm_per_stack_bytes;
  // Logic-die SRAM / interconnect bandwidth; models the ~12.8 TB/s link that
  // even SRAM-resident activation traffic traverses (= HBM4 bandwidth, so
  // SRAM-tier traffic is charged the same per-byte cost as HBM regardless of
  // tier). Read only on the num_hbm_stacks==0 (HBF+/CONV+) branch of the
  // per-token activation memory-duration calculations.
  double logic_sram_bandwidth;

  // paper2 (Kyung et al., IEEE CAL 2026) §IV: "we assume sufficient on-chip
  // SRAM" -- the paper treats on-chip SRAM capacity as an assumed-sufficient
  // resource whose REQUIRED size is an output of the analysis, not an input
  // constraint. When true, the activation scarce-tier gate (footprint.h's
  // scarceTierActivationLimit) must never bind, regardless of
  // sram_per_stack_bytes/logic_sram_bytes. NOT a constructor arg (deliberately
  // kept out of the positional ctor to avoid disturbing every existing
  // preset's argument list); assigned after construction in the preset
  // definitions below. Defaults false so every paper1 preset (hbm4_preset,
  // hbf_preset, hbf_plus_preset, conv_preset, conv_plus_preset) is completely
  // unaffected.
  bool unbounded_sram_gate = false;
  // H3 (Ha et al., IEEE CAL 2026) topology: the HBF stacks are daisy-chained
  // BEHIND the HBM base dies, so HBF traffic reaches the GPU through the same
  // GPU<->HBM-base D2D link that carries HBM traffic (the paper sets link BW =
  // HBM core BW). When true, getAttentionMemoryDuration adds a third roofline
  // ceiling ((kv_read + act) / hbm_read_bandwidth) so the two tiers' bandwidths
  // are not additive. Like unbounded_sram_gate above: member initializer only
  // (NOT a constructor arg), defaults false so every existing preset -- all of
  // which model flash as direct-attached -- is completely unaffected.
  bool flash_behind_hbm = false;
};

// Preset 1: HBM4 (8 HBM stacks, 288 GB, 12.8 TB/s symmetric)
static HBFMemoryConfig hbm4_preset = HBFMemoryConfig(
    8,                                           // num_hbm_stacks
    0,                                           // num_flash_stacks
    288ULL * 1024 * 1024 * 1024,                // total_capacity_bytes
    12.8e12,                                     // hbm_read_bandwidth
    12.8e12,                                     // hbm_write_bandwidth
    0.0,                                         // flash_read_bandwidth
    0.0,                                         // flash_write_bandwidth
    0,                                           // flash_page_read_latency_ns
    0,                                           // flash_page_program_latency_ns
    0,                                           // sram_per_stack_bytes
    0,                                           // logic_sram_bytes
    4096,                                        // page_size_bytes
    36ULL * 1024 * 1024 * 1024                  // hbm_per_stack_bytes (288GB / 8 stacks)
);

// Preset 2: HBF (1 HBM stack, 7 HBF stacks, 3620 GB)
static HBFMemoryConfig hbf_preset = HBFMemoryConfig(
    1,                                           // num_hbm_stacks
    7,                                           // num_flash_stacks
    3620ULL * 1024 * 1024 * 1024,               // total_capacity_bytes (COMBINED Table-I: HBM reserve + flash; pool = 3620 - 36 = 3584 GB, footprint.h checkCapacity)
    1.6e12,                                      // hbm_read_bandwidth (1/8 of HBM4)
    1.6e12,                                      // hbm_write_bandwidth
    11.2e12,                                     // flash_read_bandwidth (7 * 1.6 TB/s)
    0.112e12,                                    // flash_write_bandwidth (7 * 16 GB/s)
    1000,                                        // flash_page_read_latency_ns (1 us)
    100000,                                      // flash_page_program_latency_ns (100 us)
    (unsigned long long)(3.13 * 1024 * 1024),   // sram_per_stack_bytes
    0,                                           // logic_sram_bytes
    4096,                                        // page_size_bytes
    36ULL * 1024 * 1024 * 1024                  // hbm_per_stack_bytes (1 stack x 36 GB)
);

// Preset 3: HBF+ (0 HBM stacks, 8 HBF stacks, 4096 GB)
static HBFMemoryConfig hbf_plus_preset = HBFMemoryConfig(
    0,                                           // num_hbm_stacks
    8,                                           // num_flash_stacks
    4096ULL * 1024 * 1024 * 1024,               // total_capacity_bytes (flash pool)
    0.0,                                         // hbm_read_bandwidth
    0.0,                                         // hbm_write_bandwidth
    12.8e12,                                     // flash_read_bandwidth (8 * 1.6 TB/s)
    0.128e12,                                    // flash_write_bandwidth (8 * 16 GB/s)
    1000,                                        // flash_page_read_latency_ns (1 us)
    100000,                                      // flash_page_program_latency_ns (100 us)
    (unsigned long long)(3.13 * 1024 * 1024),   // sram_per_stack_bytes
    320ULL * 1024 * 1024,                       // logic_sram_bytes (320 MB)
    4096,                                        // page_size_bytes
    0                                            // hbm_per_stack_bytes (no HBM stacks)
);

// Preset 4: CONV (1 HBM stack, 7 Flash stacks, 3620 GB)
static HBFMemoryConfig conv_preset = HBFMemoryConfig(
    1,                                           // num_hbm_stacks
    7,                                           // num_flash_stacks
    3620ULL * 1024 * 1024 * 1024,               // total_capacity_bytes (COMBINED Table-I: HBM reserve + flash; pool = 3620 - 36 = 3584 GB, footprint.h checkCapacity)
    1.6e12,                                      // hbm_read_bandwidth
    1.6e12,                                      // hbm_write_bandwidth
    2.45e12,                                     // flash_read_bandwidth (7 * 350 GB/s)
    0.073e12,                                    // flash_write_bandwidth (exactly 0.073 TB/s)
    3000,                                        // flash_page_read_latency_ns (3 us)
    100000,                                      // flash_page_program_latency_ns (100 us)
    (unsigned long long)(3.13 * 1024 * 1024),   // sram_per_stack_bytes
    0,                                           // logic_sram_bytes
    4096,                                        // page_size_bytes
    36ULL * 1024 * 1024 * 1024                  // hbm_per_stack_bytes (1 stack x 36 GB)
);

// Preset 5: CONV+ (0 HBM stacks, 8 Flash stacks, 4096 GB)
static HBFMemoryConfig conv_plus_preset = HBFMemoryConfig(
    0,                                           // num_hbm_stacks
    8,                                           // num_flash_stacks
    4096ULL * 1024 * 1024 * 1024,               // total_capacity_bytes (flash pool)
    0.0,                                         // hbm_read_bandwidth
    0.0,                                         // hbm_write_bandwidth
    2.80e12,                                     // flash_read_bandwidth (8 * 350 GB/s)
    0.084e12,                                    // flash_write_bandwidth (exactly 0.084 TB/s)
    3000,                                        // flash_page_read_latency_ns (3 us)
    100000,                                      // flash_page_program_latency_ns (100 us)
    (unsigned long long)(3.13 * 1024 * 1024),   // sram_per_stack_bytes
    320ULL * 1024 * 1024,                       // logic_sram_bytes (320 MB)
    4096,                                        // page_size_bytes
    0                                            // hbm_per_stack_bytes (no HBM stacks)
);

// ---------------------------------------------------------------------------
// paper2 presets (Kyung et al., "..." IEEE CAL 2026).
// These are ADDITIVE to the paper1 (Son et al.) presets above: no existing
// preset's field values are touched, and every new field defaults false/inert
// for paper1 configs (see unbounded_sram_gate in the struct above).
// ---------------------------------------------------------------------------

// paper2 preset A: device_HBM -- paper2 §IV: 256 GB of HBM4 at B200-class
// 8 TB/s per-direction bandwidth. All-HBM (num_flash_stacks=0), so every
// flash-only code path (weight/KV-on-flash, chunked read, page latencies)
// stays off, exactly like the paper1 hbm4_preset above.
static HBFMemoryConfig paper2_hbm_preset = HBFMemoryConfig(
    8,                                           // num_hbm_stacks
    0,                                           // num_flash_stacks
    256ULL * 1024 * 1024 * 1024,                // total_capacity_bytes
    8e12,                                        // hbm_read_bandwidth
    8e12,                                        // hbm_write_bandwidth
    0.0,                                         // flash_read_bandwidth
    0.0,                                         // flash_write_bandwidth
    0,                                           // flash_page_read_latency_ns
    0,                                           // flash_page_program_latency_ns
    0,                                           // sram_per_stack_bytes
    0,                                           // logic_sram_bytes
    4096,                                        // page_size_bytes
    32ULL * 1024 * 1024 * 1024                  // hbm_per_stack_bytes (256GB / 8 stacks)
);

// paper2 preset B: device_HBF -- paper2 §II/§IV: 5 Z-NAND-class flash stacks,
// 1.6 TB/s read / 48 GB/s write each (48 = 1600 * tR/tPROG = 1600 * 3us/100us
// per paper2 §II's tR=3us / tPROG=100us Z-NAND timings), 2,560 GB aggregate
// capacity. sram_per_stack_bytes is kept at the paper1 3.13-MB staging value
// for STRUCTURAL consistency with the shared chunked-read code path (so the
// paper1 double-buffer arithmetic in layer_impl.h/parallelism_optimizer.cpp
// still has a well-defined chunk size to divide by) -- but is INERT under
// paper2_mode, because unbounded_sram_gate=true below means the activation
// scarce-tier capacity gate this SRAM would otherwise back is never checked,
// and paper2_mode (SystemConfig, set by test.cpp's P2_* branches) zeroes the
// exposed chunked-read latency entirely (layer_impl.h), so the chunk count
// this value implies has no effect on modeled latency either. logic_sram_bytes
// is 0 (paper2 has no HBM stack backing an activation tier the way HBF/CONV
// do; unbounded_sram_gate is what makes that fine).
static HBFMemoryConfig paper2_hbf_preset = []() {
  HBFMemoryConfig cfg(
      0,                                          // num_hbm_stacks
      5,                                          // num_flash_stacks
      2560ULL * 1024 * 1024 * 1024,              // total_capacity_bytes (flash pool)
      0.0,                                        // hbm_read_bandwidth
      0.0,                                        // hbm_write_bandwidth
      8e12,                                       // flash_read_bandwidth (5 * 1.6 TB/s)
      0.24e12,                                     // flash_write_bandwidth (5 * 48 GB/s)
      3000,                                       // flash_page_read_latency_ns (tR = 3us, Z-NAND)
      100000,                                     // flash_page_program_latency_ns (tPROG = 100us)
      (unsigned long long)(40.0 * 1024 * 1024),  // sram_per_stack_bytes (inert under paper2_mode)
      0,                                          // logic_sram_bytes
      4096,                                       // page_size_bytes
      0                                           // hbm_per_stack_bytes (no HBM stacks)
  );
  // paper2 treats on-chip SRAM as an assumed-sufficient resource whose
  // REQUIRED size is an OUTPUT of the analysis (§IV "we assume sufficient
  // on-chip SRAM"), not an input capacity constraint -- so the activation
  // SRAM gate must never bind under this preset.
  cfg.unbounded_sram_gate = true;
  return cfg;
}();

// paper2 preset C: device_HBF, "1/2-HBF" -- paper2 Fig5 / §V-B: "read/write
// bandwidth is reduced by half" relative to device_HBF, capacity unchanged.
static HBFMemoryConfig paper2_hbf_half_preset = []() {
  HBFMemoryConfig cfg(
      0,                                          // num_hbm_stacks
      5,                                          // num_flash_stacks
      2560ULL * 1024 * 1024 * 1024,              // total_capacity_bytes (flash pool, unchanged)
      0.0,                                        // hbm_read_bandwidth
      0.0,                                        // hbm_write_bandwidth
      4e12,                                       // flash_read_bandwidth (half of paper2_hbf_preset)
      0.12e12,                                     // flash_write_bandwidth (half of paper2_hbf_preset)
      3000,                                       // flash_page_read_latency_ns (tR = 3us, Z-NAND)
      100000,                                     // flash_page_program_latency_ns (tPROG = 100us)
      (unsigned long long)(40.0 * 1024 * 1024),  // sram_per_stack_bytes (inert under paper2_mode)
      0,                                          // logic_sram_bytes
      4096,                                       // page_size_bytes
      0                                           // hbm_per_stack_bytes (no HBM stacks)
  );
  cfg.unbounded_sram_gate = true;  // see paper2_hbf_preset's comment above
  return cfg;
}();

}  // namespace llm_system
