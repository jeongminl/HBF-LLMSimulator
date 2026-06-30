#include <yaml-cpp/yaml.h>
#include <filesystem>

#include <algorithm>
#include <cmath>
#include <iostream>

#include "common/assert.h"
#include "hardware/stat.h"
#include "model/model.h"
#include "model/util.h"
#include "module/layer.h"
#include "module/module_graph.h"
#include "optimizer/parallelism_optimizer.h"

using namespace llm_system;

int main(int argc, char *argv[]) {
  YAML::Node config;
  if(argc > 1){
    std::string config_path = argv[1];
    config = YAML::LoadFile(config_path);
  }
  else{
    config = YAML::LoadFile("config.yaml");
  }

  std::string model_name = config["model"]["model_name"].as<std::string>();
  std::string processor_type =
      config["system"]["processor_type"].as<std::string>();

  int num_node = config["system"]["num_node"].as<int>();
  int num_device = config["system"]["num_device"].as<int>();

  std::string data_name = config["simulation"]["data"].as<std::string>();
  int input_len = config["simulation"]["input_len"].as<int>();
  int output_len = config["simulation"]["output_len"].as<int>();
  int iter = config["simulation"]["iter"].as<int>();
  int injection_rate = config["simulation"]["injection_rate"].as<int>();

  int max_batch_size = config["serving"]["max_batch_size"].as<int>();

  int max_process_token = config["serving"]["max_process_token"].as<int>();
  std::string output_path = config["log"]["output_directory"].as<std::string>();
  std::filesystem::create_directories(output_path);

  SystemConfig system_config;
  if(config["system"]["gpu_gen"].as<std::string>() == "A100"){
    system_config = A100;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "H100"){
    system_config = H100;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "B100"){
    system_config = B100;
  }
  else if (config["system"]["gpu_gen"].as<std::string>() == "B200"){
    system_config = B200;
  }
  else{
    fail("No GPU generation information");
  }

  // NVLink Config // 
  if(config["system"]["nvlink_gen"].as<int>() == 4){
    system_config.device_ict_bandwidth = 450.0 * 1000 * 1000 * 1000; // B/s NVLink 4th Gen (H100)
    system_config.device_ict_latency = 0.8 * 1000; // ns
  }
  else if(config["system"]["nvlink_gen"].as<int>() == 5){
    system_config.device_ict_bandwidth = 900.0 * 1000 * 1000 * 1000; // B/s NVLink 5th Gen (B100, B200)
    system_config.device_ict_latency = 0.8 * 1000; // ns
  }else{
    fail("Not support NVLink generation");
  }

  // InfiniBand Config // 
  if(config["system"]["infiniband_gen"].as<int>() == 400){
    system_config.node_ict_bandwidth = 50.0 * 1000 * 1000 * 1000; // B/s Infiniband NDR
    system_config.node_ict_latency = 0.13 * 1000; // ns
  }
  else if(config["system"]["infiniband_gen"].as<int>() == 800){
    system_config.node_ict_bandwidth = 100.0 * 1000 * 1000 * 1000; // B/s InfiniBand XDR
    system_config.node_ict_latency = 0.13 * 1000; // ns
  }
  else if(config["system"]["infiniband_gen"].as<int>() == 3600){
    system_config.node_ict_bandwidth = 450.0 * 1000 * 1000 * 1000; // B/s NVLink 4th Gen
    system_config.node_ict_latency = 0.8 * 1000; // ns
  }
  else if(config["system"]["infiniband_gen"].as<int>() == 7200){
    system_config.node_ict_bandwidth = 900.0 * 1000 * 1000 * 1000; // B/s NVLink 5th Gen
    system_config.node_ict_latency = 0.8 * 1000; // ns
  }
  else{
    fail("Not support InfiniBand generation");
  }
  
  system_config.num_node = num_node;
  system_config.num_device = num_device;
  if (injection_rate > 0) {
    system_config.use_inject_rate = true;
    system_config.request_per_second = injection_rate;
  }

  if (config["system"]["memory_type"]) {
    std::string mem_type = config["system"]["memory_type"].as<std::string>();
    system_config.memory_type = mem_type;
    if (mem_type == "HBM") {
      system_config.use_hbf = false;
    } else {
      system_config.use_hbf = true;
      if (mem_type == "HBM4") {
        system_config.hbf_config = hbm4_preset;
      } else if (mem_type == "HBF") {
        system_config.hbf_config = hbf_preset;
      } else if (mem_type == "HBF+") {
        system_config.hbf_config = hbf_plus_preset;
      } else if (mem_type == "CONV") {
        system_config.hbf_config = conv_preset;
      } else if (mem_type == "CONV+") {
        system_config.hbf_config = conv_plus_preset;
      } else {
        fail("Unsupported memory_type: " + mem_type);
      }
      system_config.memory_capacity = system_config.hbf_config.total_capacity_bytes;
      system_config.memory_bandwidth = (system_config.hbf_config.num_flash_stacks > 0) ?
          system_config.hbf_config.flash_read_bandwidth : system_config.hbf_config.hbm_read_bandwidth;
    }
  }

  // Chunked-attention chunk size (bytes); 0 = auto (per-stack SRAM staging capacity).
  if (config["system"]["chunk_size"]) {
    system_config.chunk_size = config["system"]["chunk_size"].as<int>();
  }


  system_config.high_processor_type = ProcessorType::GPU;
  system_config.low_processor_type = ProcessorType::LOGIC;


  system_config.parallel_execution =
      config["system"]["optimization"]["parallel_execution"].as<bool>();
  system_config.hetero_subbatch =
      config["system"]["optimization"]["hetero_subbatch"].as<bool>();
  system_config.disagg_system =
      config["system"]["optimization"]["disagg_system"].as<bool>(); 
  system_config.use_low_unit_moe_only =
      config["system"]["optimization"]["use_low_unit_moe_only"].as<bool>();      
  system_config.use_ramulator =
      config["system"]["optimization"]["use_ramulator"].as<bool>();

  system_config.use_flash_mla =
      config["system"]["optimization"]["use_flash_mla"].as<bool>();
  system_config.use_flash_attention =
      config["system"]["optimization"]["use_flash_attention"].as<bool>();

  // kv cache reuse
  system_config.reuse_kv_cache =
      config["system"]["optimization"]["reuse_kv_cache"].as<bool>();
  system_config.kv_cache_reuse_rate =
      config["system"]["optimization"]["kv_cache_reuse_rate"].as<double>();
  
  // prefill mode or decode mode
  system_config.prefill_mode =
      config["system"]["optimization"]["prefill_mode"].as<bool>();
  system_config.decode_mode =
      config["system"]["optimization"]["decode_mode"].as<bool>();
  assertTrue((system_config.prefill_mode == false) || (system_config.decode_mode == false), 
            "prefill mode and decode mode is incompatible");

  assertTrue((system_config.parallel_execution == false) || (system_config.use_low_unit_moe_only == false),
            "parallel_execution & use_low_unit_moe_only are not compatible");
            
  if(system_config.prefill_mode){
    std::cout << "[Prefill Mode] Output Length is modified into 1" << std::endl;
  }

  if(system_config.decode_mode){
    std::cout << "[Decode Mode] Current Length of sequences is modified into input_len" << std::endl;
  }

  if (!processor_type.compare("GPU")) {
    system_config.processor_type = {ProcessorType::GPU};
    system_config.high_processor_type = ProcessorType::GPU;
    system_config.low_processor_type = ProcessorType::GPU;
  } else if (!processor_type.compare("LOGIC")) {
    system_config.processor_type = {ProcessorType::LOGIC};
    system_config.high_processor_type = ProcessorType::LOGIC;
    system_config.low_processor_type = ProcessorType::LOGIC;
  } else if (!processor_type.compare("GPU+LOGIC")) {
    system_config.processor_type = {ProcessorType::GPU, ProcessorType::LOGIC};
    system_config.high_processor_type = ProcessorType::GPU;
    system_config.low_processor_type = ProcessorType::LOGIC;
    // system_config.parallel_execution = true;
  } else if (!processor_type.compare("GPU+PIM")) {
    system_config.processor_type = {ProcessorType::GPU, ProcessorType::PIM};
    system_config.high_processor_type = ProcessorType::GPU;
    system_config.low_processor_type = ProcessorType::PIM;
  }

  std::string expert_file_path;

  ModelConfig model_config;

  if (!model_name.compare("mixtral")) {
    model_config = mixtral;
  } else if (!model_name.compare("openMoE")) {
    model_config = openMoE;
  } else if (!model_name.compare("llama7bMoE")) {
    model_config = llama7bMoE;
  } else if (!model_name.compare("llama3_405B")) {
    model_config = llama3_405B;
  } else if (!model_name.compare("grok1")) {
    model_config = grok1;
  } else if (!model_name.compare("deepseekV3")) {
    model_config = deepseekV3;
  }else if (!model_name.compare("llama4_scout")) {
    model_config = llama4_scout;
  }else if (!model_name.compare("llama4_maverick")) {
    model_config = llama4_maverick;
  } 
  else {
    fail("No model configuration of " + model_name);
  }

  model_config.e_tp_dg =
      config["system"]["distribution"]["expert_tensor_degree"].as<int>();
  model_config.ne_tp_dg =
      config["system"]["distribution"]["none_expert_tensor_degree"].as<int>();

  bool optimize_parallelism = false;
  if (config["system"]["optimize_parallelism"]) {
    optimize_parallelism = config["system"]["optimize_parallelism"].as<bool>();
  }

  // Precision: the model preset encodes the intended precision_byte.  Only
  // override it when the config explicitly sets a non-zero value; 0 means
  // "use model preset" (e.g. Maverick=2 → 746 GiB, llama3_405B=1 → 405 GB).
  {
    int cfg_precision = config["simulation"]["precision_byte"].as<int>();
    if (cfg_precision > 0) {
      model_config.precision_byte = cfg_precision;
    }
    // else: keep the model preset's precision_byte unchanged.
  }
  if (model_config.precision_byte == 1) {  // FP8 / INT8: double effective flops
    system_config.compute_peak_flops *= 2;
  }

  // Part D/E: parse optimizer alignment flags (all optional; fall back to in-class defaults)
  if (config["simulation"]["validate_optimizer"]) {
    system_config.validate_optimizer =
        config["simulation"]["validate_optimizer"].as<std::string>();
  }
  if (config["simulation"]["validate_optimizer_threshold"]) {
    system_config.validate_optimizer_threshold =
        config["simulation"]["validate_optimizer_threshold"].as<double>();
  }
  if (config["simulation"]["optimizer_latency_model"]) {
    system_config.optimizer_latency_model =
        config["simulation"]["optimizer_latency_model"].as<std::string>();
  }
  if (config["simulation"]["latency_margin"]) {
    system_config.latency_margin =
        config["simulation"]["latency_margin"].as<double>();
  }

  // Part E: hold predictions outside the if(optimize_parallelism) scope so
  // checkMemorySize and the latency harness can access them.
  ParallelConfig opt_pred;
  bool have_opt_pred = false;

  if (optimize_parallelism) {
    std::cout << "[Parallelism Optimizer] Searching optimal 3D parallelism strategy..." << std::endl;
    int total_gpus = num_node * num_device;
    double tpot_slo_ms = 100.0;
    if (config["system"]["tpot_slo"]) {
      tpot_slo_ms = config["system"]["tpot_slo"].as<double>() * 1000.0;
    }
    ParallelConfig opt = ParallelismOptimizer::Optimize(model_config, system_config, total_gpus, max_batch_size, input_len + output_len, tpot_slo_ms);
    opt_pred = opt;
    have_opt_pred = true;
    if (opt.oom) {
      // Spec: a batch size FAILS when NO parallelism configuration can satisfy
      // GPU-memory capacity, the SRAM / intermediate-data limit, AND the TPOT
      // SLO simultaneously (all three are enforced inside Optimize()).  Emit an
      // "Out of Memory" marker and exit non-zero so the batch-size sweep treats
      // this as a hard failure instead of silently continuing with pp=1.
      std::cout << "[Parallelism Optimizer] Out of Memory: no parallelism configuration "
                << "satisfies capacity/SRAM/SLO constraints. Reason: " << opt.oom_reason << std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      std::cout << "[Parallelism Optimizer] Found optimal configuration: TP=" << opt.tp
                << ", PP=" << opt.pp << ", DP=" << opt.dp << ", EP=" << opt.ep
                << " (Estimated Latency: " << opt.estimated_latency_ms << " ms)" << std::endl;
      model_config.ne_tp_dg = opt.tp;
      model_config.e_tp_dg = opt.ep;  // expert parallelism swept independently of non-expert TP
      model_config.pp_dg = opt.pp;
      std::cout << "[Parallelism Optimizer] Overriding none_expert_tensor_degree to " << opt.tp
                << ", expert_tensor_degree to " << opt.ep
                << " and pp_dg to " << opt.pp << std::endl;
    }
  } else {
    if (config["system"]["distribution"]["pipeline_degree"]) {
      model_config.pp_dg = config["system"]["distribution"]["pipeline_degree"].as<int>();
    } else {
      model_config.pp_dg = 1;
    }
  }
  
  model_config.compressed_kv =
      config["system"]["optimization"]["compressed_kv"].as<bool>();
  model_config.use_absorb =
      config["system"]["optimization"]["use_absorb"].as<bool>();
  model_config.skewness =
      config["simulation"]["skewness"].as<double>();
  
  // precision_byte was applied before the optimizer (above). No re-assignment
  // here — doing so would lose the preset value when cfg_precision == 0.

  system_config.exit_out_of_memory = config["simulation"]["exit_out_of_memory"].as<bool>();
  system_config.mem_cap_limit = config["simulation"]["mem_cap_limit"].as<bool>();

  model_config.dataset = data_name;

  if (!data_name.compare("synthesis")) {
    expert_file_path = "none";
    model_config.input_len = input_len;
    model_config.output_len = output_len;
  } else {
    expert_file_path =
        "../expert_data/experts_" + model_name + "_" + data_name + ".csv";
  }

  if((system_config.decode_mode == true) && (model_config.output_len <= 1)){
    fail("[Decode Mode] Output length must be larger than 1");
  }

  // long max_batch_size = 128;
  if (max_process_token == 0) {
    // max_process_token = 8192 * 16;
    max_process_token = 65536 * 8;
  }
  Scheduler::Ptr scheduler =
      Scheduler::Create(system_config, model_config, expert_file_path,
                        max_batch_size, 8192, max_process_token);

  Cluster::Ptr cluster = Cluster::Create(system_config, scheduler);

  Model model(model_config, cluster, scheduler);

  // Part E: pass optimizer predictions to checkMemorySize for drift comparison.
  bool out_of_memory = have_opt_pred
      ? cluster->checkMemorySize(opt_pred.pred_weight_bytes,
                                  opt_pred.pred_kv_bytes,
                                  opt_pred.pred_act_bytes,
                                  opt_pred.pred_total_bytes)
      : cluster->checkMemorySize();
  cluster->set_dependency();

  std::cout << "-----------------------------------" << std::endl;
  std::cout << "-------------start-----------------" << std::endl;
  std::cout << "-----------------------------------" << std::endl;

  std::vector<Stat> stat_list;
  int total_iter = iter;

  int ne_tp_dg = model_config.ne_tp_dg;
  int ne_dp_dg = system_config.num_device * num_node / ne_tp_dg;

  int precision_bytes = model_config.precision_byte;

  std::string file_name;
  if(system_config.prefill_mode){
    if(system_config.use_ramulator){
      file_name = output_path + "/" + model_name + "_" + data_name +
                            "_" + std::to_string(input_len) + "_" +
                            std::to_string(output_len) + "_" + processor_type +
                            "_N" + std::to_string(num_node) + "_D" + 
                            std::to_string(num_device) + "_TP" +
                            std::to_string(ne_tp_dg) + "_DP" +
                            std::to_string(ne_dp_dg) + "_maxbatch" +
                            std::to_string(max_batch_size) + "_maxprocess" +
                            std::to_string(max_process_token) + "_iter" +
                            std::to_string(total_iter) + 
                            "_skew"+ std::to_string(int(model_config.skewness*10)) +                            
                            "_precision_byte" + std::to_string(precision_bytes) + "_parallel_execution" + std::to_string(system_config.parallel_execution) + "_ramul_prefill.csv";
    }
    else{
      file_name = output_path + "/" + model_name + "_" + data_name +
                            "_" + std::to_string(input_len) + "_" +
                            std::to_string(output_len) + "_" + processor_type +
                            "_N" + std::to_string(num_node) + "_D" + 
                            std::to_string(num_device) + "_TP" +
                            std::to_string(ne_tp_dg) + "_DP" +
                            std::to_string(ne_dp_dg) + "_maxbatch" +
                            std::to_string(max_batch_size) + "_maxprocess" +
                            std::to_string(max_process_token) + "_iter" +
                            std::to_string(total_iter) + 
                            "_skew"+ std::to_string(int(model_config.skewness*10)) +                            
                            "_precision_byte" + std::to_string(precision_bytes) + "_parallel_execution" + std::to_string(system_config.parallel_execution) + "_prefill.csv";
    }
  }
  else if(system_config.decode_mode){
    if(system_config.use_ramulator){
      file_name = output_path + "/" + model_name + "_" + data_name +
                            "_" + std::to_string(input_len) + "_" +
                            std::to_string(output_len) + "_" + processor_type +
                            "_N" + std::to_string(num_node) + "_D" + 
                            std::to_string(num_device) + "_TP" +
                            std::to_string(ne_tp_dg) + "_DP" +
                            std::to_string(ne_dp_dg) + "_maxbatch" +
                            std::to_string(max_batch_size) + "_maxprocess" +
                            std::to_string(max_process_token) + "_iter" +
                            std::to_string(total_iter) + 
                            "_skew"+ std::to_string(int(model_config.skewness*10)) +                            
                            "_precision_byte" + std::to_string(precision_bytes) + "_parallel_execution" + std::to_string(system_config.parallel_execution) + "_ramul_decode.csv";
    }
    else{
      file_name = output_path + "/" + model_name + "_" + data_name +
                            "_" + std::to_string(input_len) + "_" +
                            std::to_string(output_len) + "_" + processor_type +
                            "_N" + std::to_string(num_node) + "_D" + 
                            std::to_string(num_device) + "_TP" +
                            std::to_string(ne_tp_dg) + "_DP" +
                            std::to_string(ne_dp_dg) + "_maxbatch" +
                            std::to_string(max_batch_size) + "_maxprocess" +
                            std::to_string(max_process_token) + "_iter" +
                            std::to_string(total_iter) + 
                            "_skew"+ std::to_string(int(model_config.skewness*10)) +                            
                            "_precision_byte" + std::to_string(precision_bytes) + "_parallel_execution" + std::to_string(system_config.parallel_execution) + "_decode.csv";
    }
  }
  else{
    if(system_config.use_ramulator){
      file_name = output_path + "/" + model_name + "_" + data_name +
                            "_" + std::to_string(input_len) + "_" +
                            std::to_string(output_len) + "_" + processor_type +
                            "_N" + std::to_string(num_node) + "_D" + 
                            std::to_string(num_device) + "_TP" +
                            std::to_string(ne_tp_dg) + "_DP" +
                            std::to_string(ne_dp_dg) + "_maxbatch" +
                            std::to_string(max_batch_size) + "_maxprocess" +
                            std::to_string(max_process_token) + "_iter" +
                            std::to_string(total_iter) + 
                            "_skew"+ std::to_string(int(model_config.skewness*10)) +                            
                            "_precision_byte" + std::to_string(precision_bytes) + "_parallel_execution" + std::to_string(system_config.parallel_execution) + "_ramul.csv";
    }
    else{
      file_name = output_path + "/" + model_name + "_" + data_name +
                            "_" + std::to_string(input_len) + "_" +
                            std::to_string(output_len) + "_" + processor_type +
                            "_N" + std::to_string(num_node) + "_D" + 
                            std::to_string(num_device) + "_TP" +
                            std::to_string(ne_tp_dg) + "_DP" +
                            std::to_string(ne_dp_dg) + "_maxbatch" +
                            std::to_string(max_batch_size) + "_maxprocess" +
                            std::to_string(max_process_token) + "_iter" +
                            std::to_string(total_iter) + 
                            "_skew"+ std::to_string(int(model_config.skewness*10)) +                            
                            "_precision_byte" + std::to_string(precision_bytes) + "_parallel_execution" + std::to_string(system_config.parallel_execution) + ".csv";
    }
  }
  if (out_of_memory && config["simulation"]["exit_out_of_memory"].as<bool>()) {
    std::cout << "Out of Memory: " << file_name << std::endl;
    return 0;
  }

  // Emit PEC geometry so Python post-processing never needs to duplicate model
  // dimensions or precision.  Matches getKVWriteDuration() exactly:
  //   standard KV: 2 * num_layers * num_kv_heads * head_dim * precision_byte
  //   compressed KV (MLA): (kv_lora_rank + qk_rope_head_dim) * num_layers * precision_byte
  {
    double kv_bytes_per_token = model_config.compressed_kv
        ? (double)(model_config.kv_lora_rank + model_config.qk_rope_head_dim)
            * model_config.num_layers * model_config.precision_byte
        : 2.0 * model_config.num_layers * model_config.num_kv_heads
            * model_config.head_dim * model_config.precision_byte;
    double flash_capacity = (system_config.use_hbf && system_config.hbf_config.num_flash_stacks > 0)
        ? (double)system_config.hbf_config.total_capacity_bytes
        : (double)system_config.memory_capacity;
    std::cout << "PEC_KV_BYTES_PER_TOKEN: " << kv_bytes_per_token << std::endl;
    std::cout << "PEC_FLASH_CAPACITY_BYTES: " << flash_capacity << std::endl;
  }

  scheduler->getActualArrivalTime(total_iter);
  stat_list = cluster->runIteration(total_iter, file_name);

  // Part E: latency drift harness — compare optimizer prediction vs measured.
  // measured_latency_ms = total simulation time / num_iterations / 1e6.
  // Uses a looser threshold (2x) since the predictive model is coarse-grained.
  if (have_opt_pred && system_config.validate_optimizer != "off") {
    double measured_latency_ms = (total_iter > 0)
        ? (double)scheduler->total_time / ((double)total_iter * 1e6)
        : 0.0;
    if (measured_latency_ms > 0.0) {
      double div = std::abs(opt_pred.estimated_latency_ms - measured_latency_ms) /
                   measured_latency_ms;
      double lat_thresh = system_config.validate_optimizer_threshold * 2.0;
      if (div > lat_thresh) {
        std::string msg = "[OptValidation] latency: pred=" +
                          std::to_string(opt_pred.estimated_latency_ms) + "ms" +
                          " actual=" + std::to_string(measured_latency_ms) + "ms" +
                          " div=" + std::to_string(div * 100.0) + "% WARN";
        if (system_config.validate_optimizer == "strict") {
          fail(msg);
        } else {
          std::cout << msg << std::endl;
        }
      }
    }
  }

  // TopModuleGraph::Ptr top1 = cluster->get_device(8)->top_module_graph;
  std::string gantt_file_path =
      config["log"]["gantt_directory"].as<std::string>();

  if(config["log"]["export_gantt"].as<bool>()){
    cluster->exportGantt(gantt_file_path);
  }

  if(config["log"]["print_log"].as<bool>()){
    TopModuleGraph::Ptr top0 = cluster->get_device(0)->top_module_graph;
    top0->print_timeboard();
  }

  return 0;
}
