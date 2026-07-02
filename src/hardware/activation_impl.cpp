#include <memory>

#include "common/type.h"
#include "dram/dram_interface.h"
#include "dram/dram_request.h"
#include "hardware/layer_impl.h"
#include "module/tensor.h"

namespace llm_system {
class DRAMRequest;
class Tensor;

using Tensor_Ptr = std::shared_ptr<Tensor>;
using DRAMRequest_Ptr = std::shared_ptr<DRAMRequest>;

// Per-variant descriptor for activation execution.
// Activation is always memory-bound (total_flops = total_memory_size, so compute
// time equals memory time).  The three variants differ in bandwidth source,
// fp8 compute doubling, and which DRAM request types are issued.
struct ActivationDesc {
  hw_metric compute_peak_flops;
  hw_metric memory_bandwidth;
  bool fp8_double;  // Logic/PIM: double compute_peak_flops when precision_byte==1
  // Ramulator operand types (gate_output, input use read_req/read_pim; output uses write_req/write_pim)
  ProcessorType proc;
  DRAMRequestType read_req;
  PIMOperandType  read_pim;
  DRAMRequestType write_req;
  PIMOperandType  write_pim;
};

static ExecStatus activationCore(
    Device_Ptr device, Tensor_Ptr gate_output, Tensor_Ptr input, Tensor_Ptr output,
    bool use_ramulator, ActivationDesc desc)
{
  auto& config = device->config;

  if (desc.fp8_double && input->precision_byte == 1)
    desc.compute_peak_flops *= 2;

  hw_metric total_memory_size = gate_output->getSize() + input->getSize() + output->getSize();
  // Activation is elementwise: FLOPs ≈ memory bytes read (memory-bound); total_flops=total_memory_size.
  hw_metric total_flops = total_memory_size;

  time_ns compute_duration = total_flops /
      (desc.compute_peak_flops * effectiveMFU(config, input->shape[0])) * 1000 * 1000 * 1000;
  time_ns memory_duration;
  if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
    memory_duration = (config.hbf_config.num_hbm_stacks > 0)
        ? total_memory_size / config.hbf_config.hbm_read_bandwidth * 1000 * 1000 * 1000
        : 0;
  } else {
    memory_duration = total_memory_size / desc.memory_bandwidth * 1000 * 1000 * 1000;
  }

  ExecStatus exec_status;
  if (input->getSize() == 0) return exec_status;

  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status += issueRamulator(device, LayerType::ACTIVATION,
        desc.proc, desc.read_req,  desc.read_pim,  gate_output);
    exec_status += issueRamulator(device, LayerType::ACTIVATION,
        desc.proc, desc.read_req,  desc.read_pim,  input);
    exec_status += issueRamulator(device, LayerType::ACTIVATION,
        desc.proc, desc.write_req, desc.write_pim, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, desc.proc, DRAMRequestType::kRead,  gate_output);
    exec_status += getIdealMemoryStatus(device, desc.proc, DRAMRequestType::kRead,  input);
    exec_status += getIdealMemoryStatus(device, desc.proc, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration = std::max(exec_status.compute_duration, exec_status.memory_duration);

  double eff_bw = (config.use_hbf && config.hbf_config.num_flash_stacks > 0)
      ? (config.hbf_config.num_hbm_stacks > 0 ? config.hbf_config.hbm_read_bandwidth : 1e13)
      : desc.memory_bandwidth;
  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             desc.compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util  = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                             eff_bw / exec_status.total_duration;
  exec_status.flops        = total_flops;
  exec_status.memory_size  = total_memory_size;
  exec_status.opb          = total_flops / total_memory_size;  // always 1.0 for activation

  return exec_status;
}

// ---- Public entry points --------------------------------------------------

ExecStatus ActivationExecutionGPU(Device_Ptr device, Tensor_Ptr gate_output,
                                  Tensor_Ptr input, Tensor_Ptr output,
                                  bool use_ramulator) {
  auto& c = device->config;
  return activationCore(device, gate_output, input, output, use_ramulator, {
      c.compute_peak_flops, c.memory_bandwidth, /*fp8_double=*/false,
      ProcessorType::GPU, DRAMRequestType::kRead, PIMOperandType::kDRAM,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM
  });
}

ExecStatus ActivationExecutionLogic(Device_Ptr device, Tensor_Ptr gate_output,
                                    Tensor_Ptr input, Tensor_Ptr output,
                                    bool use_ramulator) {
  auto& c = device->config;
  // Logic ramulator: all three operands use kGEMV/kSrc (incl. output write).
  return activationCore(device, gate_output, input, output, use_ramulator, {
      c.logic_memory_bandwidth * c.logic_op_b, c.logic_memory_bandwidth, /*fp8_double=*/true,
      ProcessorType::LOGIC, DRAMRequestType::kGEMV, PIMOperandType::kSrc,
                            DRAMRequestType::kGEMV, PIMOperandType::kSrc
  });
}

ExecStatus ActivationExecutionPIM(Device_Ptr device, Tensor_Ptr gate_output,
                                  Tensor_Ptr input, Tensor_Ptr output,
                                  bool use_ramulator) {
  auto& c = device->config;
  return activationCore(device, gate_output, input, output, use_ramulator, {
      c.pim_memory_bandwidth * c.pim_op_b, c.pim_memory_bandwidth, /*fp8_double=*/true,
      ProcessorType::PIM, DRAMRequestType::kRead,  PIMOperandType::kDRAM,
                          DRAMRequestType::kWrite, PIMOperandType::kDRAM
  });
}

}  // namespace llm_system
