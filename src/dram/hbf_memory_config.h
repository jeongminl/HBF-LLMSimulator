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
    3620ULL * 1024 * 1024 * 1024,               // total_capacity_bytes (flash pool)
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
    3620ULL * 1024 * 1024 * 1024,               // total_capacity_bytes (flash pool)
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

}  // namespace llm_system
