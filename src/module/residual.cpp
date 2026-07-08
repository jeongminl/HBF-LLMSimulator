#include "module/residual.h"
#include "scheduler/scheduler.h"

#include "common/assert.h"
// AllReduce //

namespace llm_system {

Residual::Residual(std::string& prefix, std::string& name,
                     int hidden_dim, std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true) { // sync == true, thererfore need to be synced
    std::vector<int> shape = {1, 1};

    Tensor::Ptr residual_out = Tensor::Create("residual_out", shape, "act", device, device->model_config.precision_byte);
    add_tensor(residual_out);
}

Tensor::Ptr Residual::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  int m = input->shape[0];
  int k = input->shape[1];

  Tensor::Ptr output = get_activation("residual_out", input->shape);

  long size = input->getSize();
  if (size == 0) {
    return output;
  }

  hw_metric compute_peak_flops = device->config.compute_peak_flops;

  double flops, memory_size;
  flops = m * k * 1;
  memory_size = (3.0 * m * k) * input->precision_byte; // input x 2, store x 1

  time_ns compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
  // Residual traffic is pure activation data: on flash presets it lives on the
  // intermediate tier (the reserved HBM stack for HBF/CONV, 0-cost logic-die
  // SRAM for HBF+/CONV+), never on flash — same tier selection as
  // activationCore (activation_impl.cpp). The device-wide memory_bandwidth
  // scalar equals the FLASH read bandwidth on these presets and priced this
  // op ~7x too fast on HBF.
  time_ns memory_duration;
  auto& hw_config = device->config;
  if (hw_config.use_hbf && hw_config.hbf_config.num_flash_stacks > 0) {
    memory_duration = (hw_config.hbf_config.num_hbm_stacks > 0)
        ? memory_size / hw_config.hbf_config.hbm_read_bandwidth * 1000 * 1000 * 1000
        : memory_size / hw_config.hbf_config.logic_sram_bandwidth * 1000 * 1000 * 1000;
  } else {
    memory_duration = memory_size / hw_config.memory_bandwidth * 1000 * 1000 * 1000;
  }

  time_ns total_time = std::max(compute_duration, memory_duration);

  if (input->parallel_execution) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;
  // PP_FIX_SPEC.md §3.3: Residual manipulates device_time directly (never
  // goes through ExecStatus/set_pop_status), so it needs its own
  // device_time_dep mirror. Elementwise op: flops/bytes are proportional to m
  // -- entirely batch-dependent.
  device->status.device_time_dep += total_time;

  return output;
}

}  // namespace llm_system