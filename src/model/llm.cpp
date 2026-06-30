#include "model/llm.h"

#include "module/decoder.h"
#include "module/embedding.h"
#include "module/lm_head.h"
#include "module/layer.h"
#include "module/communication.h"

namespace llm_system {

// Returns true if the given layer index is a MoE (routed-expert) layer.
// Honors both first_k_dense (the first N layers are always dense regardless of
// expert_freq) and expert_freq (every expert_freq-th layer is MoE).
// Works for ALL models, not just deepseekV3: deepseekV3 uses first_k_dense only,
// llama4/mixtral use expert_freq only, dense models have expert_freq==0.
static bool isMoELayer(const ModelConfig& mc, int layer) {
  if (mc.num_routed_expert == 0) return false;              // dense model
  if (mc.first_k_dense > 0 && layer < mc.first_k_dense) return false; // forced-dense prefix
  if (mc.expert_freq == 0) return true;                     // all remaining layers are MoE
  return (layer % mc.expert_freq == 0);
}

LLM::LLM(const ModelConfig& model_config, Cluster::Ptr cluster,
         Scheduler::Ptr scheduler, Device::Ptr device)
    : Module("", "LLM", device), model_config(model_config) {
  int ne_tp_dg = model_config.ne_tp_dg;
  std::vector<int> device_list;

  set_device_list(device_list, 0, cluster->num_total_device);

  int pp_stage = (device->device_total_rank / ne_tp_dg) % model_config.pp_dg;
  int layers_per_stage = model_config.num_layers / model_config.pp_dg;
  int start_layer = pp_stage * layers_per_stage;
  int end_layer = (pp_stage == model_config.pp_dg - 1) ? model_config.num_layers : (start_layer + layers_per_stage);

  std::vector<int> stage_device_list;
  for (int r = 0; r < cluster->num_total_device; r++) {
    if ((r / ne_tp_dg) % model_config.pp_dg == pp_stage) {
      stage_device_list.push_back(r);
    }
  }

  std::vector<int> stage_0_device_list;
  for (int r = 0; r < cluster->num_total_device; r++) {
    if ((r / ne_tp_dg) % model_config.pp_dg == 0) {
      stage_0_device_list.push_back(r);
    }
  }

  std::vector<int> stage_last_device_list;
  for (int r = 0; r < cluster->num_total_device; r++) {
    if ((r / ne_tp_dg) % model_config.pp_dg == model_config.pp_dg - 1) {
      stage_last_device_list.push_back(r);
    }
  }

  if (pp_stage == 0) {
    auto embedding_layer = Embedding::Create(module_map_name, "Embedding_layer",
                                             model_config, stage_0_device_list, device);
    add_module(embedding_layer);
  }

  for (int layer = start_layer; layer < end_layer; layer++) {
    if (isMoELayer(model_config, layer)) {
      auto moe_decoder = MoEDecoder::Create(
          module_map_name, "MoE_decoder_" + std::to_string(layer),
          model_config, scheduler, stage_device_list, device);
      add_module(moe_decoder);
    } else {
      auto decoder =
          Decoder::Create(module_map_name, "decoder_" + std::to_string(layer),
                          model_config, scheduler, stage_device_list, device);
      add_module(decoder);
    }
  }

  if (model_config.pp_dg > 1 && pp_stage < model_config.pp_dg - 1) {
    auto pipeline_stage = PipelineStage::Create(
        module_map_name, "pipeline_stage", device->device_total_rank,
        device->device_total_rank + ne_tp_dg, device);
    add_module(pipeline_stage);
  }

  if (pp_stage == model_config.pp_dg - 1) {
    auto lm_head = LmHead::Create(module_map_name, "lm_head",
      model_config, stage_last_device_list, device);
    add_module(lm_head);
  }
}

Tensor::Ptr LLM::forward(const Tensor::Ptr input,
                         BatchedSequence::Ptr sequences_metadata) {
  Module::Ptr decoder;
  Tensor::Ptr temp = input;
  Tensor::Ptr out = input;

  int ne_tp_dg = model_config.ne_tp_dg;
  int pp_stage = (device->device_total_rank / ne_tp_dg) % model_config.pp_dg;
  int layers_per_stage = model_config.num_layers / model_config.pp_dg;
  int start_layer = pp_stage * layers_per_stage;
  int end_layer = (pp_stage == model_config.pp_dg - 1) ? model_config.num_layers : (start_layer + layers_per_stage);

  if (pp_stage == 0) {
    Module::Ptr embedding = get_module("Embedding_layer");
    temp = (*embedding)(input, sequences_metadata);
  }

  for (int layer = start_layer; layer < end_layer; layer++) {
    if (isMoELayer(model_config, layer)) {
      decoder = get_module("MoE_decoder_" + std::to_string(layer));
    } else {
      decoder = get_module("decoder_" + std::to_string(layer));
    }
    out = (*decoder)(temp, sequences_metadata);
    temp = out;
  }

  if (model_config.pp_dg > 1 && pp_stage < model_config.pp_dg - 1) {
    Module::Ptr pipeline_stage = get_module("pipeline_stage");
    out = (*pipeline_stage)(out, sequences_metadata);
  }

  if (pp_stage == model_config.pp_dg - 1) {
    Module::Ptr lm_head = get_module("lm_head");
    out = (*lm_head)(out, sequences_metadata);
  }

  return out;
}

}  // namespace llm_system
