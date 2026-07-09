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

// Per-variant descriptor: the only things that differ between GPU/Logic/PIM.
struct LinearDesc {
  hw_metric compute_peak_flops;
  hw_metric memory_bandwidth;
  bool fp8_double;           // Logic/PIM double compute_peak_flops when precision_byte==1
  bool set_opb;              // GPU sets exec_status.opb; Logic/PIM do not
  ProcessorType weight_ramul_proc;  // ProcessorType for issueRamulator weight op
  DRAMRequestType weight_ramul_req; // DRAMRequestType for issueRamulator weight op
  PIMOperandType weight_ramul_pim;  // PIMOperandType for issueRamulator weight op
  ProcessorType weight_ideal_proc;  // ProcessorType for getIdealMemoryStatus weight op
};

// Shared core for Linear and BatchedLinear across all three processor variants.
// When num_heads > 1, operates on 3D tensors (batched-linear): shape is
// [num_heads, m, k] for input and [num_heads, k, n] for weight; all three variants
// use the same 3D logic (Logic and PIM must not ignore num_heads or duplicated_input).
static ExecStatus linearCore(
    Device_Ptr device, Tensor_Ptr input, Tensor_Ptr weight, Tensor_Ptr output,
    bool use_ramulator, LinearDesc desc,
    int num_heads = 1, bool duplicated_input = false)
{
  auto& config = device->config;

  if (num_heads > 1) {
    assertTrue(input->shape.size()  == 3, "input tensor is not 3D tensor");
    assertTrue(weight->shape.size() == 3, "weight tensor is not 3D tensor");
    assertTrue(output->shape.size() == 3, "output tensor is not 3D tensor");
  }

  if (desc.fp8_double && input->precision_byte == 1)
    desc.compute_peak_flops *= 2;

  // Extract m/k/n from the appropriate shape dimensions.
  double m, k, n;
  std::vector<int> input_orig, weight_orig, output_orig;
  if (num_heads > 1) {
    input_orig  = input->shape;
    weight_orig = weight->shape;
    output_orig = output->shape;
    m = input->shape[1];
    k = input->shape[2];
    n = weight->shape[2];
  } else {
    m = input->shape[0];
    k = input->shape[1];
    n = weight->shape[1];
  }

  hw_metric total_flops = 2.0 * m * k * n * num_heads;
  hw_metric total_memory_size;
  if (num_heads > 1 && duplicated_input)
    total_memory_size = (m * k + (k * n + m * n) * num_heads) * weight->precision_byte;
  else
    total_memory_size = (m * k + k * n + m * n) * num_heads * weight->precision_byte;

  // paper2 §IV first-activated-expert exposure: one-shot arm/consume flag on
  // the weight tensor (see tensor.h's exposeFirstExpertPageLatency and
  // module/expert.cpp's arming logic). Read and reset UNCONDITIONALLY here,
  // before the zero-size early return below, so the flag can never persist
  // into a later call on this same weight tensor regardless of which code
  // path this call takes.
  bool expose_first_expert_latency = weight->exposeFirstExpertPageLatency;
  weight->exposeFirstExpertPageLatency = false;

  ExecStatus exec_status;
  if (input->getSize() == 0) return exec_status;

  time_ns compute_duration = total_flops /
      (desc.compute_peak_flops * effectiveMFU(config, m)) * 1000 * 1000 * 1000;
  time_ns memory_duration  = getLinearMemoryDuration(config, m, k, n, weight->precision_byte,
                                                      total_memory_size, desc.memory_bandwidth,
                                                      num_heads, duplicated_input,
                                                      expose_first_expert_latency);
  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    if (num_heads > 1) {
      // Flatten 3D tensors to 2D shapes for the ramulator path.
      input->setShape(duplicated_input
          ? std::vector<int>{(int)m, (int)k}
          : std::vector<int>{(int)m, (int)(k * num_heads)});
      weight->setShape({(int)(k * num_heads), (int)n});
      output->setShape({(int)m, (int)(n * num_heads)});
    }
    exec_status += issueRamulator(device, LayerType::LINEAR,
        ProcessorType::GPU, DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status += issueRamulator(device, LayerType::LINEAR,
        desc.weight_ramul_proc, desc.weight_ramul_req, desc.weight_ramul_pim, weight);
    exec_status += issueRamulator(device, LayerType::LINEAR,
        ProcessorType::GPU, DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
    if (num_heads > 1) {
      input->setShape(input_orig);
      weight->setShape(weight_orig);
      output->setShape(output_orig);
    }
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU,     DRAMRequestType::kRead,  input);
    exec_status += getIdealMemoryStatus(device, desc.weight_ideal_proc, DRAMRequestType::kRead,  weight);
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU,     DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration = std::max(exec_status.compute_duration, exec_status.memory_duration);
  // Batch-dependence tag (PP_FIX_SPEC.md §3.3, refined past the doc's literal
  // "whole branch" rule): compute_duration is always fully batch-dependent
  // (2*m*k*n flops, exactly linear in m). memory_duration is NOT uniformly
  // one or the other:
  //  - Non-flash (HBM-only): memory_duration = total_memory_size/bandwidth is
  //    an ADDITIVE sum of a fixed weight-read term (k*n bytes, m-independent)
  //    and an m-scaled activation term (m*k [+ m*n] bytes) -- tagging the
  //    WHOLE memory-bound duration as indep (the doc's literal rule) discards
  //    the activation term's real linear scaling. For hidden_dim-scale models
  //    this is usually small (weight >> activation) but not always negligible
  //    enough for the pp>1 reconstruction's +/-1% target -- measured ~2-3%
  //    excess via PP_FIX_SPEC.md's battery item A before this refinement.
  //    Since duration is linear in bytes here, split it EXACTLY by byte
  //    fraction instead of all-or-nothing.
  //  - Flash (HBF/HBF+/CONV): getLinearMemoryDuration's flash branch is a
  //    max() of two terms that are EACH already purely one-or-the-other
  //    (weight_read_time has no m-term at all; act_time is purely linear in
  //    m) -- so the whole-branch tag IS exact there; recompute act_time with
  //    the identical formula to see which side of the max won.
  if (exec_status.compute_duration > exec_status.memory_duration) {
    exec_status.batch_dependent_duration = exec_status.total_duration;
  } else if (config.use_hbf && config.hbf_config.num_flash_stacks > 0) {
    const auto& hbf = config.hbf_config;
    hw_metric act_read_size = m * k * weight->precision_byte * (duplicated_input ? 1 : num_heads);
    hw_metric act_write_size = m * n * weight->precision_byte * num_heads;
    double act_time = (hbf.num_hbm_stacks > 0)
        ? (act_read_size + act_write_size) / hbf.hbm_read_bandwidth * 1e9
        : (act_read_size + act_write_size) / hbf.logic_sram_bandwidth * 1e9;
    exec_status.batch_dependent_duration =
        (act_time >= exec_status.memory_duration) ? exec_status.total_duration : 0.0;
  } else {
    hw_metric weight_bytes = (k * n) * num_heads * weight->precision_byte;
    hw_metric act_bytes = total_memory_size - weight_bytes;
    exec_status.batch_dependent_duration = (total_memory_size > 0)
        ? exec_status.memory_duration * (act_bytes / total_memory_size) : 0.0;
  }

  // MOE_TAG_FIX_SPEC §4.2: routed-expert pipeline-parallel correction. A cold-at-micro
  // expert (activated at full B, zero tokens in the first B/pp) is not dispatched
  // in the pp>1 microbatch steady state, so this call must contribute ZERO to the
  // reconstructed per-token time. reportedIterationTime() (cluster.cpp) computes
  // this op's contribution as total_duration - batch_dependent_duration*(pp-1)/pp;
  // setting batch_dependent_duration = total_duration*pp/(pp-1) makes that 0
  // EXACTLY, for both memory-bound and (pathologically) compute-bound calls,
  // using this call's own total_duration -- no separate weight-stream formula.
  // Overrides the roofline tag above ON PURPOSE for these calls only.
  {
    int pp = device->model_config.pp_dg;
    if (pp > 1 && input->cold_at_micro && exec_status.total_duration > 0.0) {
      exec_status.batch_dependent_duration =
          exec_status.total_duration * (double)pp / (double)(pp - 1);
    }
  }
  // Propagate the flag down this expert's FFN chain (mirrors linear.cpp:39's
  // perform_with_optimal). ALWAYS assign (overwrite) so a pooled/reused output
  // tensor can never carry a stale flag from a prior iteration.
  output->cold_at_micro = input->cold_at_micro;

  double eff_bw = (config.use_hbf && config.hbf_config.num_flash_stacks > 0)
      ? config.hbf_config.flash_read_bandwidth : desc.memory_bandwidth;
  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             desc.compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util  = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                             eff_bw / exec_status.total_duration;
  exec_status.flops        = total_flops;
  exec_status.memory_size  = total_memory_size;
  if (desc.set_opb) exec_status.opb = total_flops / total_memory_size;

  return exec_status;
}

// ---- Public entry points --------------------------------------------------
// Each just builds a LinearDesc and delegates to linearCore.

ExecStatus LinearExecutionGPU(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator) {
  auto& c = device->config;
  return linearCore(device, input, weight, output, use_ramulator, {
      c.compute_peak_flops, c.memory_bandwidth,
      /*fp8_double=*/false, /*set_opb=*/true,
      ProcessorType::GPU, DRAMRequestType::kRead, PIMOperandType::kDRAM,
      ProcessorType::GPU
  });
}

ExecStatus LinearExecutionLogic(Device_Ptr device, Tensor_Ptr input,
                                Tensor_Ptr weight, Tensor_Ptr output,
                                bool use_ramulator) {
  auto& c = device->config;
  return linearCore(device, input, weight, output, use_ramulator, {
      c.logic_memory_bandwidth * c.logic_op_b, c.logic_memory_bandwidth,
      /*fp8_double=*/true, /*set_opb=*/false,
      ProcessorType::LOGIC, DRAMRequestType::kGEMV, PIMOperandType::kSrc,
      ProcessorType::LOGIC
  });
}

ExecStatus LinearExecutionPIM(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator) {
  auto& c = device->config;
  return linearCore(device, input, weight, output, use_ramulator, {
      c.pim_memory_bandwidth * c.pim_op_b, c.pim_memory_bandwidth,
      /*fp8_double=*/true, /*set_opb=*/false,
      ProcessorType::PIM, DRAMRequestType::kGEMV, PIMOperandType::kSrc,
      ProcessorType::PIM
  });
}

ExecStatus BatchedLinearExecutionGPU(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator, bool duplicated_input) {
  auto& c = device->config;
  int num_heads = input->shape[0];
  return linearCore(device, input, weight, output, use_ramulator, {
      c.compute_peak_flops, c.memory_bandwidth,
      /*fp8_double=*/false, /*set_opb=*/true,
      ProcessorType::GPU, DRAMRequestType::kRead, PIMOperandType::kDRAM,
      ProcessorType::GPU
  }, num_heads, duplicated_input);
}

ExecStatus BatchedLinearExecutionLogic(Device_Ptr device, Tensor_Ptr input,
                                Tensor_Ptr weight, Tensor_Ptr output,
                                bool use_ramulator, bool duplicated_input) {
  auto& c = device->config;
  int num_heads = input->shape[0];
  return linearCore(device, input, weight, output, use_ramulator, {
      c.logic_memory_bandwidth * c.logic_op_b, c.logic_memory_bandwidth,
      /*fp8_double=*/true, /*set_opb=*/false,
      ProcessorType::LOGIC, DRAMRequestType::kGEMV, PIMOperandType::kSrc,
      ProcessorType::LOGIC
  }, num_heads, duplicated_input);
}

ExecStatus BatchedLinearExecutionPIM(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator, bool duplicated_input) {
  auto& c = device->config;
  int num_heads = input->shape[0];
  return linearCore(device, input, weight, output, use_ramulator, {
      c.pim_memory_bandwidth * c.pim_op_b, c.pim_memory_bandwidth,
      /*fp8_double=*/true, /*set_opb=*/false,
      ProcessorType::PIM, DRAMRequestType::kGEMV, PIMOperandType::kSrc,
      ProcessorType::PIM
  }, num_heads, duplicated_input);
}

}  // namespace llm_system
