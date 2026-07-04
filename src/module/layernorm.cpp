#include "module/layernorm.h"
#include "scheduler/scheduler.h"
#include "common/assert.h"
#include "hardware/layer_impl.h"

namespace llm_system {

LayerNorm::LayerNorm(std::string& prefix, std::string& name,
                     int hidden_dim, std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true) {
    std::vector<int> shape = {hidden_dim, 1};

    Tensor::Ptr layer_norm_weight = Tensor::Create("layer_norm_weight", shape, "weight", device, device->model_config.precision_byte);
    Tensor::Ptr layer_norm_out = Tensor::Create("layer_norm_output", shape, "act", device, device->model_config.precision_byte);
    add_tensor(layer_norm_weight);
    add_tensor(layer_norm_out);
}

Tensor::Ptr LayerNorm::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  int m = input->shape[0];
  int k = input->shape[1];
  int n = 1;

  Tensor::Ptr layer_norm_weight = tensor_list.at("layer_norm_weight");
  Tensor::Ptr output = get_activation("layer_norm_output", input->shape);

  if (k != layer_norm_weight->shape[0]) {
    std::cerr << "RMSNorm/LayerNorm Error: Input shape[1] (k) = " << k 
              << ", weight shape[0] = " << layer_norm_weight->shape[0] 
              << ", input shape[0] (m) = " << input->shape[0] << std::endl;
  }
  assertTrue(k == layer_norm_weight->shape[0], "input shape doesn't match with layer norm weight (= hidden dim)");

  long size = input->getSize();
  if (size == 0) {
    return output;
  }

  hw_metric compute_peak_flops = device->config.compute_peak_flops;
  hw_metric memory_bandwidth = device->config.memory_bandwidth;

  double flops, memory_size;
  flops =  4.0 * m * k; // layer norm (e.g. RMSNorm, Square (1) + Root (1) + Mean (2))
  memory_size = (2.0 * m * k + 1.0 * k * n) * input->precision_byte;

  time_ns compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration;
  const auto& sys_config = device->config;
  if (sys_config.use_hbf && sys_config.hbf_config.num_flash_stacks > 0) {
    const auto& hbf = sys_config.hbf_config;
    // LayerNorm weight (gamma/beta, shape [k,1]) lives on flash. Route through
    // the same amortized weight-stream helper Linear/QKVO use (m=0 isolates the
    // weight-read term: LayerNorm's activation read/write is width k on both
    // sides, not the Linear (m,k)->(m,n) shape getLinearMemoryDuration assumes,
    // so it's priced separately below rather than via this call's act terms).
    time_ns weight_read = getLinearMemoryDuration(sys_config, 0, k, n, input->precision_byte,
                                                   0, memory_bandwidth);
    // Activations (input read + output write) are on the scarce tier.
    // When num_hbm_stacks==0 (HBF+/CONV+), activations stage in logic-die SRAM
    // (~320 MB, fast) → modeled as free to match all other ops (e.g. activation_impl.cpp).
    double act_bytes = 2.0 * (double)m * k * input->precision_byte;
    double act_time = (hbf.num_hbm_stacks > 0) ? act_bytes / hbf.hbm_read_bandwidth * 1e9 : 0.0;
    memory_duration = (time_ns)std::max(weight_read, act_time);
  } else {
    memory_duration = (time_ns)(memory_size / memory_bandwidth * 1000 * 1000 * 1000);
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

  return output;
}

}  // namespace llm_system