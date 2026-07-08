#pragma once
#include <cassert>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "module/module.h"
#include "module/tensor.h"
#include "scheduler/sequence.h"

namespace llm_system {

class SelfAttentionGen : public Module {
 public:
  using Ptr = std::shared_ptr<SelfAttentionGen>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  int max_seq_len, int batch_size, int qk_rope_head_dim,
                                  bool use_flash_attention,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new SelfAttentionGen(prefix, name, head_dim, num_heads,
                                       num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                       use_flash_attention, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  SelfAttentionGen(std::string& prefix, std::string& name, int head_dim,
                   int num_heads, int num_kv_heads, int max_seq_len,
                   int batch_size, int qk_rope_head_dim, bool use_flash_attention,
                   std::vector<int> device_list,
                   Device::Ptr device);
  SelfAttentionGen() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

 private:
  int rank;
  int head_dim;
  int num_heads;
  int num_kv_heads;
  int qk_rope_head_dim;
  // FlashAttention master flag (SystemConfig::use_flash_attention). ON = score/P
  // never materialize (on-chip); OFF = materialized score + charged softmax memory.
  // Threaded through to LayerInfo in forward() so the GQA gen kernel
  // (AttentionGenExecutionGPU) can gate the score/softmax terms. See FA_FLAG_SPEC.md.
  bool use_flash_attention;
  // max_seq_len as constructed with (may be < the model's full context length for
  // Llama-4-style local/chunked attention layers -- see model/model_config.h's
  // effectiveKvLen()). Stored so forward() can populate LayerInfo::local_attention_window,
  // capping the decode-phase KV read at this layer's effective window.
  int max_seq_len;
  // std::vector<int> device_list;
};

class SelfAttentionSum : public Module {
 public:
  using Ptr = std::shared_ptr<SelfAttentionSum>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  int max_seq_len, int batch_size, int qk_rope_head_dim,
                                  bool use_flash_attention,
                                  std::vector<int> device_list,
                                  Device::Ptr device, int gen_max_seq_len = -1) {
    Ptr ptr = Ptr(new SelfAttentionSum(prefix, name, head_dim, num_heads,
                                       num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                       use_flash_attention, device_list, device, gen_max_seq_len));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  SelfAttentionSum(std::string& prefix, std::string& name, int head_dim,
                   int num_heads, int num_kv_heads, int max_seq_len,
                   int batch_size, int qk_rope_head_dim, bool use_flash_attention,
                   std::vector<int> device_list,
                   Device::Ptr device, int gen_max_seq_len = -1);
  SelfAttentionSum() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

 private:
  int rank;
  int head_dim;
  int num_heads;
  int num_kv_heads;
  int qk_rope_head_dim;
  // FlashAttention master flag (see SelfAttentionGen). ON = on-chip score/P;
  // OFF = materialized score + charged softmax memory. Threaded to LayerInfo in
  // forward() so AttentionSumExecutionGPU can gate its score/softmax terms.
  bool use_flash_attention;
  // Per-layer local-attention window (Llama-4-style iRoPE), mirroring
  // SelfAttentionGen::max_seq_len / model/model_config.h's effectiveKvLen(): caller
  // (module/parallel.cpp) passes resolved_gen_max_seq_len here. Stored SEPARATELY
  // from max_seq_len (unlike Gen) because the Sum output tensor must stay sized at
  // the full prefill sequence length -- only the scoring/context dimension is
  // capped, in forward(). <=0 (the default) or == max_seq_len means "global
  // attention, no cap" (every model except llama4_maverick/llama4_scout).
  int local_attention_window;
  // std::vector<int> device_list;
};

class SelfAttentionMixed : public Module {
 public:
  using Ptr = std::shared_ptr<SelfAttentionMixed>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  int max_seq_len, int batch_size, int qk_rope_head_dim,
                                  bool use_flash_attention,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new SelfAttentionMixed(prefix, name, head_dim, num_heads,
                                         num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                         use_flash_attention, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  SelfAttentionMixed(std::string& prefix, std::string& name, int head_dim,
                     int num_heads, int num_kv_heads, int max_seq_len,
                     int batch_size, int qk_rope_head_dim, bool use_flash_attention,
                     std::vector<int> device_list,
                     Device::Ptr device);
  SelfAttentionMixed() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

 private:
  int rank;
  int head_dim;
  int num_heads;
  int num_kv_heads;
  int qk_rope_head_dim;
  // FlashAttention master flag (see SelfAttentionGen). DEAD CODE path (Mixed
  // family never instantiated -- parallel.cpp builds only Sum/Gen), kept flag-aware
  // for internal consistency per FA_FLAG_SPEC.md ruling #5.
  bool use_flash_attention;
  // std::vector<int> device_list;
};

class AttentionSplit : public Module {
 public:
  using Ptr = std::shared_ptr<AttentionSplit>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  int max_seq_len, int batch_size, int qk_rope_head_dim,
                                  int kv_lora_rank, bool use_absorb, std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr =
        Ptr(new AttentionSplit(prefix, name, head_dim, num_heads, num_kv_heads,
                               max_seq_len, batch_size, qk_rope_head_dim, kv_lora_rank, use_absorb, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  AttentionSplit(std::string& prefix, std::string& name, int head_dim,
                 int num_heads, int num_kv_heads, int max_seq_len,
                 int batch_size, int qk_rope_head_dim, int kv_lora_rank,
                 bool use_absorb, std::vector<int> device_list,
                 Device::Ptr device);
  AttentionSplit() = default;

  TensorVec forward(const TensorVec input,
                    BatchedSequence::Ptr sequences_metadata) override;

 private:
  int rank;
  int head_dim;
  int num_heads;
  int num_kv_heads;
  int qk_rope_head_dim;
  int kv_lora_rank;
  bool use_absorb;
  // std::vector<int> device_list;
};

class AttentionMerge : public Module {
 public:
  using Ptr = std::shared_ptr<AttentionMerge>;
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  int max_seq_len, int batch_size, int qk_rope_head_dim, 
                                  int kv_lora_rank, bool use_absorb, std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr =
        Ptr(new AttentionMerge(prefix, name, head_dim, num_heads, num_kv_heads,
                               max_seq_len, batch_size, qk_rope_head_dim, kv_lora_rank, use_absorb, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  AttentionMerge(std::string& prefix, std::string& name, int head_dim,
                 int num_heads, int num_kv_heads, int max_seq_len,
                 int batch_size, int qk_rope_head_dim, int kv_lora_rank, bool use_absorb, std::vector<int> device_list,
                 Device::Ptr device);
  AttentionMerge() = default;

  TensorVec forward(const TensorVec input,
                    BatchedSequence::Ptr sequences_metadata) override;

 private:
  int rank;
  int head_dim;
  int num_heads;
  int num_kv_heads;
  int qk_rope_head_dim;
  int kv_lora_rank;
  bool use_absorb;
  // std::vector<int> device_list;
};

class MultiLatentAttentionGen : public Module {
  public:
   using Ptr = std::shared_ptr<MultiLatentAttentionGen>;
   [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                   int head_dim, int num_heads, int num_kv_heads,
                                   int max_seq_len, int batch_size, int qk_rope_head_dim,
                                   bool compressed_kv, bool use_flash_mla, bool use_flash_attention,
                                   std::vector<int> device_list,
                                   Device::Ptr device) {
     Ptr ptr = Ptr(new MultiLatentAttentionGen(prefix, name, head_dim, num_heads,
                                        num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                        compressed_kv, use_flash_mla, use_flash_attention, device_list, device));
     ptr->set_tensor_module();
     return ptr;
   };

  private:
    MultiLatentAttentionGen(std::string& prefix, std::string& name, int head_dim,
                    int num_heads, int num_kv_heads, int max_seq_len,
                    int batch_size, int qk_rope_head_dim, bool compressed_kv, bool use_flash_mla,
                    bool use_flash_attention, std::vector<int> device_list,
                    Device::Ptr device);
    MultiLatentAttentionGen() = default;

   Tensor::Ptr forward(const Tensor::Ptr input,
                       BatchedSequence::Ptr sequences_metadata) override;

  private:
   int rank;
   int head_dim;
   int num_heads;
   int num_kv_heads;
   int qk_rope_head_dim;
   bool compressed_kv;
   bool use_flash_mla;
   // FA master flag: gates the materialized (use_flash_mla==false) else-branch's
   // score/softmax presence in MultiLatentAttentionGenExecution* (ruling #3).
   bool use_flash_attention;
   // std::vector<int> device_list;
 };
 
 class MultiLatentAttentionSum : public Module {
  public:
   using Ptr = std::shared_ptr<MultiLatentAttentionSum>;
   [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                   int head_dim, int num_heads, int num_kv_heads,
                                   int max_seq_len, int batch_size, int qk_rope_head_dim, 
                                   bool use_flash_attention, std::vector<int> device_list,
                                   Device::Ptr device) {
     Ptr ptr = Ptr(new MultiLatentAttentionSum(prefix, name, head_dim, num_heads,
                                        num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                        use_flash_attention, device_list, device));
     ptr->set_tensor_module();
     return ptr;
   };
 
  private:
    MultiLatentAttentionSum(std::string& prefix, std::string& name, int head_dim,
                    int num_heads, int num_kv_heads, int max_seq_len,
                    int batch_size, int qk_rope_head_dim, bool use_flash_attention, 
                    std::vector<int> device_list, Device::Ptr device);
    MultiLatentAttentionSum() = default;
 
   Tensor::Ptr forward(const Tensor::Ptr input,
                       BatchedSequence::Ptr sequences_metadata) override;
  private:
   int rank;
   int head_dim;
   int num_heads;
   int num_kv_heads;
   int qk_rope_head_dim;
   bool use_flash_attention;
   // std::vector<int> device_list;
 };
 
 class AbsorbMLAGen : public Module {
  public:
   using Ptr = std::shared_ptr<AbsorbMLAGen>;
   [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                   int head_dim, int num_heads, int num_kv_heads,
                                   int max_seq_len, int batch_size, int qk_rope_head_dim,
                                   int kv_lora_rank, bool compressed_kv, bool use_flash_mla,
                                   bool use_flash_attention, std::vector<int> device_list,
                                   Device::Ptr device) {
     Ptr ptr = Ptr(new AbsorbMLAGen(prefix, name, head_dim, num_heads,
                                        num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                        kv_lora_rank, compressed_kv, use_flash_mla, use_flash_attention, device_list, device));
     ptr->set_tensor_module();
     return ptr;
   };

  private:
    AbsorbMLAGen(std::string& prefix, std::string& name, int head_dim,
                    int num_heads, int num_kv_heads, int max_seq_len,
                    int batch_size, int qk_rope_head_dim, int kv_lora_rank, bool compressed_kv,
                    bool use_flash_mla, bool use_flash_attention, std::vector<int> device_list, Device::Ptr device);
    AbsorbMLAGen() = default;

   Tensor::Ptr forward(const Tensor::Ptr input,
                       BatchedSequence::Ptr sequences_metadata) override;

  private:
   int rank;
   int head_dim;
   int num_heads;
   int num_kv_heads;
   int qk_rope_head_dim;
   int kv_lora_rank;
   bool compressed_kv;
   bool use_flash_mla;
   // FA master flag: gates the materialized else-branch's score/softmax presence
   // in AbsorbMLAGenExecution* (ruling #3).
   bool use_flash_attention;
   // std::vector<int> device_list;
 };
 
 class AbsorbMLASum : public Module {
  public:
   using Ptr = std::shared_ptr<AbsorbMLASum>;
   [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                   int head_dim, int num_heads, int num_kv_heads,
                                   int max_seq_len, int batch_size, int qk_rope_head_dim,
                                   int kv_lora_rank, bool use_flash_attention, std::vector<int> device_list,
                                   Device::Ptr device) {
     Ptr ptr = Ptr(new AbsorbMLASum(prefix, name, head_dim, num_heads,
                                        num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                        kv_lora_rank, use_flash_attention, device_list, device));
     ptr->set_tensor_module();
     return ptr;
   };

  private:
    AbsorbMLASum(std::string& prefix, std::string& name, int head_dim,
                    int num_heads, int num_kv_heads, int max_seq_len,
                    int batch_size, int qk_rope_head_dim, int kv_lora_rank, bool use_flash_attention,
                    std::vector<int> device_list,
                    Device::Ptr device);
    AbsorbMLASum() = default;

   Tensor::Ptr forward(const Tensor::Ptr input,
                       BatchedSequence::Ptr sequences_metadata) override;
  private:
   int rank;
   int head_dim;
   int num_heads;
   int num_kv_heads;
   int qk_rope_head_dim;
   int kv_lora_rank;
   // FA master flag: gates the materialized score/softmax presence in
   // AbsorbMLASumExecution* (ruling #3 / Absorb-Sum wrapper).
   bool use_flash_attention;
   // std::vector<int> device_list;
 };
}  // namespace llm_system