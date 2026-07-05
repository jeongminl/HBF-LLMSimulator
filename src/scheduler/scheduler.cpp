#include "scheduler/scheduler.h"
#include <numeric>
#include <cmath>
#include <random>

namespace llm_system {
Scheduler::Scheduler(SystemConfig system_config, ModelConfig& model_config,
                     std::string expert_file_path, int total_batch_size,
                     int num_max_batched_token,  // per dp, total
                     int max_process_token)      // per dp, in sum
    : system_config(system_config),
      workload_rng(system_config.workload_seed),
      model_config(model_config),
      expert_file_path(expert_file_path),
      total_batch_size(total_batch_size),
      num_max_batched_token(num_max_batched_token),
      max_process_token(max_process_token) {
  dp_degree =
      system_config.num_device * system_config.num_node / (model_config.ne_tp_dg * model_config.pp_dg);
  batch_size_per_dp = total_batch_size / dp_degree;
  total_seq_num = 0;
  total_time = 0;
  real_data = false;
  disagg_system = system_config.disagg_system;
  initExpertList(expert_file_path);
};

bool Scheduler::hasSumSeq() {
  for (auto batchseq : running_queue) {
    if (batchseq->get_sum().size() != 0) {
      return true;
    }
  }
  return false;
}

void Scheduler::initRunningQueue() {
  running_queue.resize(0);
  for (int batch_idx = 0; batch_idx < dp_degree; batch_idx++) {
    BatchedSequence::Ptr sequences_metadata = BatchedSequence::Create(
        model_config.num_routed_expert, model_config.top_k, getPtr());
    running_queue.push_back(sequences_metadata);
  }
}

void Scheduler::pushDummySeq(int input_len, int output_len, int idx, int count) {
  // QUIRK (guide §16 / not in BUGS.md or BUGS_HIDDEN_BY_FLAGS.md): this whole block
  // -- norm_dist_value, the rejection loop below, and delta -- computes a per-sequence
  // length-jitter value that is NEVER applied: the three lines that would consume it
  // (input_len = input_len - delta; output_len = output_len + delta;) are commented
  // out just below. Every call pays the RNG + rejection-loop cost for no effect on
  // input_len/output_len. Left as dead computation intentionally, not removed: every
  // reported sweep's results were generated with jitter off, so deleting or enabling
  // this now would both be behavior changes that need a separate, deliberate decision
  // (documented, not fixed, per explicit instruction -- see BUGS_FIXES.md).
  double norm_dist_value = getNormaldistribution();

  int delta = std::min(256, input_len);
  delta = std::min(delta, output_len) - 1;

  while (norm_dist_value <= -0.95 || norm_dist_value >= 0.95) {
    norm_dist_value = getNormaldistribution();
  }

  delta = delta * norm_dist_value;
  if (delta < 0) {
    delta += 1;
  } else {
    delta -= 1;
  }

  // to give sequence some randomness, you can insert delat value in to length by uncommenting below
  // delta = 0;
  // input_len = input_len - delta;
  // output_len = output_len + delta;

  if (output_len == 0) {
    return;
  }

  Sequence::Ptr new_seq = Sequence::Create(0, input_len, output_len);
  if(total_time > 0){
    try {
      new_seq->arrival_time = actual_arrival_time.at(cur_arrival_time_idx++);
    } catch (...) {
      new_seq->arrival_time = 1;
    }
  }
  if(system_config.reuse_kv_cache){
    static unsigned int seed = 777;
    static std::mt19937 generator(seed);
    static std::uniform_real_distribution<> distribution(0, system_config.kv_cache_reuse_rate);

    new_seq->current_len = input_len * system_config.kv_cache_reuse_rate;
  }

  if(system_config.prefill_mode){
    new_seq->output_len = input_len; // for prefill mode
    new_seq->total_len = input_len;
  }
  else if(system_config.decode_mode){
    // Steady-state decode: under continuous batching with arrival rate matching
    // completion rate (paper SS-III), in-flight queries are uniformly distributed
    // over their output lifetime, so the POPULATION's mean context is input +
    // output/2 at every tick (matching the optimizer's steady_ctx,
    // parallelism_optimizer.cpp:404-408) -- not start-of-generation (input), which
    // under-charged the context-linear attention cost by output/2 tokens (~10% of
    // context on SHORT, ~4% MID, ~0.5% LONG).
    //
    // The INITIAL fill must realize that population mean as a uniform SNAPSHOT
    // across the generation lifetime, not a single cohort all seeded at the
    // midpoint: seeding every dummy at exactly input+output/2 makes them all
    // advance in lockstep, so hittingQueue(N)'s periodic sampling always lands on
    // the same phase of the cycle instead of the intended steady-state average
    // (measured drift: SHORT's sampled context read 1988-1996 instead of the
    // intended ~1846). Staggering idx/count sequences evenly across [0, output_len)
    // reproduces the uniform-snapshot population at t=0 and keeps it uniform
    // forever after (steady-state refills below enter at start-of-generation,
    // maintaining the same age distribution as sequences complete and are replaced).
    //
    // The output_len-2 clamp keeps the stagger strictly inside the sequence's
    // lifetime: updateScheduler's completion check (current_len == total_len) runs
    // AFTER advancing current_len by one step, so a dummy seeded at exactly
    // total_len-1 would complete on its very first step, but one seeded at
    // total_len would never satisfy the equality test post-increment and would
    // become a zombie that never gets collected.
    if (!initial_fill_done) {
      long long offset = (long long)idx * output_len / std::max(count, 1);
      offset = std::min(offset, (long long)(output_len - 2));
      if (offset < 0) offset = 0;
      new_seq->current_len = input_len + offset;
    } else {
      new_seq->current_len = input_len; // steady-state refill: enters at start-of-generation
    }
  }
  new_seq->get_expert_from_list = false;
  sequence_queue.push_back(new_seq);
}

// paper2 stochastic workload sampler (paper2 Sec. V-A). Draws a total context
// length C and an output/context ratio r independently, then derives
// (input_len, output_len) from their product. Determinism: this is the only
// place that advances workload_rng, and it is only ever called from pushSeq's
// per-seq loop (i = 0..num_seq-1, in order) when system_config.paper2_workload
// is set, so for a fixed workload_seed the sequence of draws -- and therefore
// every sampled (input_len, output_len) -- is fully determined by the
// scheduler's call order, independent of anything else in the simulator
// (workload_rng is never touched by getRandomExpert/getNormaldistribution/
// getPoissondistribution's static-local generators, and vice versa).
void Scheduler::sampleWorkloadLengths(int& input_len, int& output_len, bool size_biased) {
  double mu = system_config.workload_context_mean;
  double sigma = system_config.workload_context_cv * mu;
  double trunc_sigmas = system_config.workload_context_trunc_sigmas;
  // Truncated support for C: paper2 Sec. V-A draws context length from a
  // Normal(mu, sigma) truncated to mu +/- trunc_sigmas*sigma. sigma above uses
  // workload_context_cv as the PRE-truncation coefficient of variation;
  // conditioning on this truncated window shrinks the REALIZED CV to roughly
  // 0.88x the nominal cv at trunc_sigmas=2 (a truncated normal's variance is
  // always <= the untruncated variance). The same ~0.88x shrink factor
  // applies to both the paper's low- and high-dispersion cv settings, so the
  // *relative* low-vs-high dispersion contrast the paper studies is preserved
  // even though neither realized CV is exactly the configured value.
  double lo = mu - trunc_sigmas * sigma;
  double hi = mu + trunc_sigmas * sigma;
  std::normal_distribution<double> context_dist(mu, sigma);

  // r = Lout / (Lin + Lout) = Lout / C, modeled as Beta(alpha, beta) via the
  // mean/concentration parametrization: alpha = mu_r*kappa, beta =
  // (1-mu_r)*kappa, so that E[r] = mu_r and kappa = alpha+beta is paper2's
  // "concentration parameter" (larger kappa -> r concentrates tighter around
  // mu_r). mu_r = workload_lout_mean_ratio is the only Beta-compatible
  // reading of the paper's Lin:Lout ratio: Beta's support is [0,1], and
  // Lout/(Lin+Lout) is the natural quantity confined to [0,1] -- unlike
  // Lout/Lin, which is unbounded above and not Beta-distributable.
  double mu_r = system_config.workload_lout_mean_ratio;
  double kappa = system_config.workload_lout_beta_kappa;
  double alpha = mu_r * kappa;
  double beta = (1.0 - mu_r) * kappa;
  // Beta(alpha, beta) == g1/(g1+g2) for independent g1 ~ Gamma(alpha, 1),
  // g2 ~ Gamma(beta, 1) -- standard Gamma-ratio construction of the Beta
  // distribution.
  std::gamma_distribution<double> gamma_a(alpha, 1.0);
  std::gamma_distribution<double> gamma_b(beta, 1.0);

  // SIZE-BIASED mode (initial fill only, size_biased=true): under continuous
  // batching at steady state, the population of sequences in flight at a
  // random instant is length-biased -- the renewal/inspection paradox --
  // because a longer-Lout sequence occupies "in flight" status for longer, so
  // P(a given in-flight slot holds a sequence of a given Lout) is proportional
  // to that Lout (paper2 measures ~+9% mean Lout for the in-flight population
  // vs the raw arrival population at cv=0.3). Long-Lout sequences also
  // contribute almost no completions during a bounded simulated run, so an
  // unbiased initial population would never self-correct toward the true
  // steady-state length mix within the run's horizon -- the bias has to be
  // baked into the initial snapshot directly. Realize it by rejection
  // sampling: draw (C, r) unbiased as below, accept with probability
  // Lout/Lout_max, where Lout_max is the largest Lout the truncated support
  // can produce (C capped at hi, r capped below 1, so Lout = C*r < hi always
  // -- this bounds acceptance probability strictly below 1 without needing to
  // clamp it). Steady-state refills (size_biased=false) draw UNBIASED and
  // enter at current_len=input_len (age 0, see pushDummySeq's decode_mode
  // else-branch) -- an unbiased age-0 arrival stream combined with the
  // length-biased initial snapshot is exactly the stationary in-flight
  // distribution (matches pushDummySeq's decode_mode comment on why initial
  // fill and refills are seeded differently).
  double lout_max = hi;
  std::uniform_real_distribution<double> accept_dist(0.0, 1.0);

  double C = mu;
  double r = mu_r;
  while (true) {
    do {
      C = context_dist(workload_rng);
    } while (C < lo || C > hi);

    double g1 = gamma_a(workload_rng);
    double g2 = gamma_b(workload_rng);
    double denom = g1 + g2;
    r = (denom > 0.0) ? (g1 / denom) : mu_r;

    if (!size_biased) break;
    double lout_candidate = C * r;
    double accept_prob = (lout_max > 0.0) ? (lout_candidate / lout_max) : 1.0;
    if (accept_dist(workload_rng) <= accept_prob) break;
  }

  // Lower clamp 2: pushDummySeq's decode_mode stagger needs output_len-2 >= 0,
  // and updateScheduler's completion check (current_len == total_len, checked
  // AFTER advancing current_len) needs total_len = input_len+output_len >
  // input_len, i.e. output_len >= 1 -- clamping to 2 keeps a full margin for
  // the stagger. Upper clamp (round(C)-1): guarantees input_len >= 1.
  long long C_round = std::llround(C);
  output_len = (int)std::clamp<long long>(std::llround(C * r), 2, C_round - 1);
  input_len = (int)(C_round - output_len);
}

void Scheduler::pushSeq(int num_seq) {
  if (!real_data) {
    if (system_config.paper2_workload) {
      // paper2 stochastic workload: each dummy sequence draws its own
      // (input_len, output_len) via sampleWorkloadLengths instead of every
      // sequence reusing the single uniform model_config.input_len/output_len
      // pair below. This branch is only taken when paper2_workload is
      // explicitly set true (default false), so the uniform paper1 path
      // (assert + pushDummySeq loop in the else-branch) is untouched and
      // remains bit-identical.
      //
      // size_biased = !initial_fill_done: initial_fill_done is false only
      // during the very first fillSequenceQueue()/fillRunningQueue() call in
      // Cluster::runIteration (cluster.cpp, before it flips true at line
      // ~560), so every sequence pushSeq generates for that initial
      // population is drawn size-biased; every subsequent steady-state
      // refill (called from fillSequenceQueue inside hittingQueue/
      // runIterationMixed's per-iteration loop) is drawn unbiased. See
      // sampleWorkloadLengths's size-biased-mode comment for why.
      bool size_biased = !initial_fill_done;
      for (int i = 0; i < num_seq; i++) {
        int sampled_input_len = 0;
        int sampled_output_len = 0;
        sampleWorkloadLengths(sampled_input_len, sampled_output_len, size_biased);

        // Safety-net resample (mirrors the paper1 branch's max_seq_len guard
        // above, generalized to per-sample C): the truncated support
        // guarantees C <= workload_context_mean * (1 + trunc_sigmas * cv),
        // which is far below max_seq_len for every paper2 cell (e.g. mean
        // 8192, cv 0.3, trunc_sigmas 2 -> C <= 8192*1.6 = 13107 << 32768), so
        // this loop is expected to never execute. Resample rather than
        // clamp/truncate so a misconfigured preset can't silently produce a
        // statistically-biased sequence at the boundary; the guard counter
        // bounds worst-case cost if a preset ever violates the assumption.
        int guard = 0;
        while (sampled_input_len + sampled_output_len >= model_config.max_seq_len &&
               guard < 1000) {
          sampleWorkloadLengths(sampled_input_len, sampled_output_len, size_biased);
          guard++;
        }

        pushDummySeq(sampled_input_len, sampled_output_len, i, num_seq);
      }
      return;
    }

    assertTrue(model_config.input_len < model_config.max_seq_len, "Invalid input_len (= "
               + std::to_string(model_config.input_len) + ")" );
    if(model_config.input_len + model_config.output_len > model_config.max_seq_len){
      model_config.output_len = model_config.max_seq_len - model_config.input_len;
      std::cout << "output_len is modfied to " << model_config.output_len << std::endl;
    }
    for(int i = 0; i < num_seq; i ++){
      pushDummySeq(model_config.input_len, model_config.output_len, i, num_seq);
    }
  } else {
    pushRealSeq(num_seq);
  }
}

void Scheduler::clear() {
  sequence_queue.clear();
  running_queue.clear();
  total_token_in_expert.resize(0);
}

// num_sequences_metadata is the same as the data parallelism degree
void Scheduler::initializeDummyInput(int num_seq, int input_len,
                                     int output_len) {

  for (int batch_idx = 0; batch_idx < dp_degree; batch_idx++) {
    int each_num_seq = (num_seq + dp_degree - 1) / dp_degree;

    BatchedSequence::Ptr sequences_metadata = running_queue.at(batch_idx);

    sequences_metadata->add_dummy_sequence(each_num_seq, input_len, output_len);
  }
}

int Scheduler::getRandomExpertSeqId() {
  static unsigned int seed = 777;
  static std::mt19937 generator(seed);
  static std::uniform_int_distribution<int> distribution(0, 4096 - 1);

  int seq_id = distribution(generator);
  seq_id %= total_seq_num;

  return seq_id;
}

std::set<int> Scheduler::getRandomExpert(int top_k) {
  static unsigned int seed = 777;
  static std::mt19937 generator(seed);
  static std::uniform_int_distribution<int> distribution(0, model_config.num_routed_expert - 1);

  std::set<int> route;

  while (route.size() < top_k) {
    int seq_id = (distribution(generator) % model_config.num_routed_expert);
    route.insert(seq_id);
  }
  return route;
}

std::set<int> Scheduler::getEquallyDistributedExpert(int token_id, int top_k) {
  std::set<int> route;

  while (route.size() < top_k) {
    int seq_id = token_id++ % model_config.num_routed_expert;
    route.insert(seq_id);
  }
  return route;
}

std::set<int> Scheduler::getZipfianRandomExpert(std::vector<double> weight, int top_k) {
  static unsigned int seed = 777;
  static std::mt19937 generator(seed);
  static std::discrete_distribution<int> distribution(weight.begin(), weight.end());

  // The Zipf weights are index-monotone (expert 0 hottest) and expert->device
  // placement is contiguous (expert.cpp: experts [i*16, i*16+16) -> device i),
  // so drawing the raw index would colocate ALL hot experts on device 0 —
  // an accidental artifact, not a modeled placement policy. At low batch
  // (llama4 LONG, ~31 active experts) that inflates the bottleneck device's
  // expert weight-stream count ~3x vs any balanced placement, since
  // per-iteration time is max-over-devices and expert weight-read cost is
  // token-count-independent. Decorrelate hotness from placement with a fixed
  // coprime-stride permutation of the drawn hotness rank: hot ranks 0..7 land
  // on 8 distinct devices (round-robin), preserving the skew distribution
  // itself and per-seed determinism.
  int n = model_config.num_routed_expert;
  static const int kStrideCandidates[] = {17, 19, 23, 29, 31, 1};
  int stride = 1;
  for (int s : kStrideCandidates) {
    // Coprimality alone admits s == identity whenever s ≡ 1 (mod n) (e.g. stride 17
    // with n in {8, 16}): that leaves hot experts contiguous, defeating the whole
    // point of this permutation. Reject those candidates too.
    if (std::gcd(s, n) == 1 && (s % n) != 1) { stride = s; break; }
  }

  std::set<int> route;

  while (route.size() < top_k) {
    int hotness_rank = (distribution(generator) % n);
    int expert_id = (int)(((long)hotness_rank * stride) % n);
    route.insert(expert_id);
  }
  return route;
}

double Scheduler::getNormaldistribution() {
  static unsigned int seed = 777;
  static std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(0.0, 0.4);
  double number = distribution(generator);
  return number;
}

int Scheduler::getPoissondistribution(int request_per_second) {
  static unsigned int seed = 777;
  static std::default_random_engine generator(seed);
  std::poisson_distribution<> distribution(request_per_second);
  int number = distribution(generator);
  return number;
}

int Scheduler::getNumInjection() {
  static unsigned int seed = 777;
  static std::mt19937 generator(seed);
  static std::uniform_int_distribution<int> distribution(0, 4);
  static std::uniform_int_distribution<int> get_seq(0, 4);

  int random = distribution(generator);
  int num_injection = 0;
  if (random == 0) {
    num_injection = get_seq(generator);
  }
  return num_injection;
}

void Scheduler::initializeRealInput(int num_seq) {
  for (int batch_idx = 0; batch_idx < dp_degree; batch_idx++) {
    int each_num_seq = (num_seq + dp_degree - 1) / dp_degree;

    BatchedSequence::Ptr sequences_metadata = running_queue.at(batch_idx);
    std::vector<int> seq_ids;
    for (int seq_id = 0; seq_id < each_num_seq; seq_id++) {
      seq_ids.push_back(getRandomExpertSeqId());
    }
    sequences_metadata->add_sequence(seq_ids);
  }
}

void Scheduler::pushRealSeq(int num_seq) {
  assertTrue(sequences_info.size() != 0, "Input trace is not set");
  for (int temp = 0; temp < num_seq; temp++) {
    int seq_id = 0;
    int input_len = 0;
    int output_len = 0;
    do {
      seq_id = getRandomExpertSeqId();
      input_len = sequences_info.at(seq_id)->Lin;
      output_len = sequences_info.at(seq_id)->Lout;
    } while (output_len == 512);

    Sequence::Ptr new_seq = Sequence::Create(seq_id, input_len, output_len);
    sequence_queue.push_back(new_seq);
  }
}

std::vector<BatchedSequence::Ptr> Scheduler::getAllMetadata() {
  return running_queue;
}

BatchedSequence::Ptr Scheduler::getMetadata(int dp_rank) {
  return running_queue.at(dp_rank);
}

BatchedSequence::Ptr Scheduler::getMaxMetadata(int num_expert, int top_k,
                                               Ptr scheduler) {
  BatchedSequence::Ptr sequences_metadata =
      BatchedSequence::Create(num_expert, top_k, scheduler);
  int seq_len = num_max_batched_token / batch_size_per_dp;
  // Same bug class as the committed T3 fix (BUGS_FIXES.md #3): integer division
  // floors to 0 once batch_size_per_dp > num_max_batched_token (8192, hardcoded at
  // eval/test.cpp's Scheduler::Create call), which makes Sequence::Create(0, 0)
  // trip sequence.cpp's assertTrue(input_len > 0, ...) -> process exit. run_experiments.py
  // classifies that crash as "OOM/Crash" (indistinguishable from a real capacity OOM),
  // so the batch search reads batch_size_per_dp=8192 as a spurious hard ceiling for any
  // cell whose true ceiling is actually higher. No-op below the wall (seq_len>=1 already).
  if (seq_len < 1) seq_len = 1;
  for (int seq_idx = 0; seq_idx < batch_size_per_dp; seq_idx++) {
    Sequence::Ptr seq = Sequence::Create(seq_len, seq_len);
    seq->num_process_token = seq_len;
    sequences_metadata->add(seq);
  }
  return sequences_metadata;
}

std::vector<BatchedSequence::Ptr> Scheduler::setMetadata() {
  // paper2 admission-token counter: reset at the top of the step boundary --
  // see the member's doc comment in scheduler.h. Unconditional (paper1 and
  // paper2 alike); harmless since nothing reads the counter yet.
  admitted_prefill_tokens_this_step = 0;

  // paper2 CPU-memory/NVLink-C2C KV offload tier: reservation-based offloaded
  // byte fraction, recomputed once per step per dp-shard (running batch).
  // Each request reserves a contiguous KV region for its FULL context
  // (total_len = input_len + output_len - 1, Sequence's ctor) at admission;
  // batches fill the local HBM-KV budget first, and the remainder is modeled
  // as living in CPU memory. Guarded by cpuKvOffloadActive() (hardware_config.h)
  // so paper1 and every non-offload/HBF paper2 config leaves
  // BatchedSequence::p2_kv_offload_fraction at its default-initialized 0.0 --
  // a complete no-op.
  if (cpuKvOffloadActive(system_config)) {
    // ne_tp_degree_of_the_shard: paper2's mapping has model_config.ne_tp_dg==1,
    // so this reduces to exactly "one device's HBM minus its resident
    // weights". The multiplier generalizes to TP>1, where a dp-shard's KV
    // cache is sharded across the whole ne_tp group backing it (dp_degree's
    // own derivation above, ctor line 18-19, shows the same ne_tp_dg driving
    // the dp/TP split) -- paper2 itself never exercises TP>1 (ne_tp=1).
    double hbm_kv_budget =
        (system_config.memory_capacity - system_config.weight_bytes_per_device) *
        (double)model_config.ne_tp_dg;
    for (auto& batch : running_queue) {
      double reserved_total = 0.0;
      for (auto& seq : batch->get_seq()) {
        reserved_total +=
            perSeqAdmissionKvBytes(model_config, (double)seq->total_len);
      }
      double f_off = (reserved_total > 0.0)
          ? std::max(0.0, reserved_total - hbm_kv_budget) / reserved_total
          : 0.0;
      batch->p2_kv_offload_fraction = f_off;
      if (p2_byte_counting_enabled) {
        p2_offload_fraction_sum += f_off;
        p2_offload_fraction_samples += 1;
      }
    }
  }

  bool process_gen = true;
  bool process_sum = true;

  std::vector<Sequence::Ptr> execution_queue;
  execution_queue.resize(running_queue.size());

  if (disagg_system == false) {
    if (hasSumSeq()) {
      process_gen = false;
      process_sum = true;
    } else {
      process_gen = true;
      process_sum = false;
    }
  }

  for (int batch_idx = 0; batch_idx < running_queue.size(); batch_idx++) {
    BatchedSequence::Ptr batch = running_queue.at(batch_idx);

    int num_gen_seq = batch->get_gen().size();
    std::vector<Sequence::Ptr> gen_seq = batch->get_gen();
    for (auto& seq : gen_seq) {
      if (process_gen) {
        if (seq->gen_start_time == 0) {
          seq->num_process_token = 1;
        } else if (seq->gen_start_time <= total_time) {
          seq->num_process_token = 1;
        }
      }
    }

    int process_token = num_max_batched_token - batch->get_gen_process_token();

    std::vector<Sequence::Ptr> sum_seq = batch->get_sum();
    if (sum_seq.size() != 0) {
      int num_sum_seq = sum_seq.size();
      // max_process_token <= 0 (config.yaml's default) means "no per-step cap" --
      // integer-dividing by num_sum_seq used to floor to 0, permanently stalling
      // every prefill sequence's current_len (BUGS_HIDDEN_BY_FLAGS #3). INT_MAX is
      // then clamped per-sequence by the std::min below, same as an explicit cap.
      int num_process = (max_process_token > 0)
          ? std::max(1, max_process_token / num_sum_seq)
          : INT_MAX;
      for (int seq_idx = 0; seq_idx < num_sum_seq; seq_idx++) {
        if (process_sum) {
          sum_seq[seq_idx]->num_process_token =
              std::min(num_process, sum_seq[seq_idx]->input_len -
                                        sum_seq[seq_idx]->current_len);
        }
      }
    }
  }
  return running_queue;
}

std::vector<Sequence::Ptr> Scheduler::updateScheduler(time_ns time) {
  std::vector<Sequence::Ptr> token_list;

  for (int batch_idx = 0; batch_idx < running_queue.size(); batch_idx++) {
    BatchedSequence::Ptr batch = running_queue.at(batch_idx);

    // paper2 KV-bytes accountant -- decode hook. MUST run BEFORE
    // batch->update(time) below: that call both advances current_len AND
    // zeroes num_process_token (Sequence::update), so this is the only point
    // where "did this sequence just generate a token this step" is still
    // observable. get_gen() classifies by current_len >= input_len, i.e.
    // "already past prefill" at this pre-update instant -- the very step a
    // sequence's prefill *completes* (current_len reaches input_len via THIS
    // update call) is still classified "sum" going into it, so it is
    // correctly excluded here: that admission's prefill KV bytes were
    // already counted once, at admission time, in fillRunningQueue(). The
    // num_process_token>0 guard excludes gen-classified sequences that
    // haven't actually started generating yet (gen_start_time > total_time
    // in setMetadata(), whose num_process_token stays 0) -- they wrote no KV
    // entry this step.
    if (p2_byte_counting_enabled) {
      for (auto& seq : batch->get_gen()) {
        if (seq->num_process_token > 0) {
          p2_decode_kv_bytes +=
              model_config.num_layers * kvBytesPerLayerToken(model_config);
        }
      }
    }

    batch->update(time);

    // remove seq of which generation is done
    for (auto seq : batch->get_seq()) {
      if (seq->current_len == seq->total_len) {
        batch->pop(seq);
        token_list.push_back(seq);
      } else if (seq->current_len == seq->input_len) {
        if (!seq->record) {
          token_list.push_back(seq);
          seq->record = true;
        }
      }
    }
  }
  return token_list;
}

std::vector<Sequence::Ptr> Scheduler::updateSchedulerSumGenSplit(time_ns time) {
  std::vector<Sequence::Ptr> token_list;

  for (int batch_idx = 0; batch_idx < running_queue.size(); batch_idx++) {
    BatchedSequence::Ptr batch = running_queue.at(batch_idx);
    // remove seq of which generation is done
    for (auto seq : batch->get_sum()) {
      seq->update(time);
      seq->setGenStartTime(total_time + time);
      if (seq->current_len == seq->total_len) {
        batch->pop(seq);
        token_list.push_back(seq);
      } else if (seq->current_len == seq->input_len) {
        if (!seq->record) {
          token_list.push_back(seq);
          seq->record = true;
        }
      }
    }
  }
  return token_list;
}

void Scheduler::fillRunningQueue(time_ns time) {

  static int rotate = 0;
  int num_empty_seq = total_batch_size - getBatchSize();
  num_empty_seq = std::min(num_empty_seq, int(sequence_queue.size()));
  for (int i = 0; i < num_empty_seq; i++) {
    auto seq = sequence_queue.begin();

    Sequence::Ptr seqPtr = *seq;

    for (int i = 0; i < running_queue.size(); i++) {
      BatchedSequence::Ptr batch =
          running_queue[rotate++ % running_queue.size()];

      if (batch->get_num_seq() < batch_size_per_dp) {
        sequence_queue.erase(seq);
        if (system_config.use_inject_rate) {
          seqPtr->queueing_delay = total_time - seqPtr->arrival_time;
        }
        batch->add(seqPtr);
        // paper2 admission-token counter: seqPtr just transitioned from
        // sequence_queue into a running batch, i.e. it was just admitted --
        // this is the sole place that happens. Unconditional upkeep (see
        // scheduler.h doc comment); a plain integer add cannot alter timing.
        admitted_prefill_tokens_this_step += seqPtr->input_len;
        // paper2 KV-bytes accountant -- admission hook. Same site as the
        // admission-token counter just above (sole place a sequence
        // transitions sequence_queue -> running_queue). Guarded by
        // p2_byte_counting_enabled (see scheduler.h doc comment); a plain
        // double add cannot perturb timing.
        if (p2_byte_counting_enabled) {
          p2_admission_kv_bytes +=
              perSeqAdmissionKvBytes(model_config, (double)seqPtr->input_len);
        }
        break;
      }
    }
  }
}

void Scheduler::printStatus() {
  BatchedSequence::Ptr batseq = running_queue.at(0);
  std::cout << "Current status: " << std::to_string(batseq->get_process_token())
            << " | Sum: " << std::to_string(batseq->get_sum().size()) << ", "
            << std::to_string(batseq->get_sum_process_token())
            << " | Gen : " << std::to_string(batseq->get_gen().size()) << ", "
            << std::to_string(batseq->get_gen_process_token()) << ", average: "
            << std::to_string(batseq->get_average_sequence_length())
            << std::endl;
}

int Scheduler::getBatchSize() {
  int size = 0;
  for (auto batch_seq : running_queue) {
    size += batch_seq->get_num_seq();
  }
  return size;
}

int Scheduler::getSumSize() {
  int size = 0;
  for (auto batch_seq : running_queue) {
    auto seq_list = batch_seq->get_sum();
    for (auto seq : seq_list) {
      if (seq->num_process_token != 0) {
        size++;
      }
    }
  }
  return size;
}

int Scheduler::getGenSize() {
  int size = 0;
  for (auto batch_seq : running_queue) {
    auto seq_list = batch_seq->get_gen();
    for (auto seq : seq_list) {
      if (seq->num_process_token != 0) {
        size++;
      }
    }
  }
  return size;
}

  int Scheduler::getAverageSeqlen() {
    int len = 0;
    for (auto batch_seq : running_queue) {
      len += batch_seq->get_total_sequence_length();
    }

    int total_batch = getBatchSize();

    if (total_batch != 0) {
      len /= total_batch;
    } else {
      len = 0;
    }

    return len;
  }

  int Scheduler::getNumProcessToken() {
    int num_token = 0;
    for (auto batch_seq : running_queue) {
      num_token += batch_seq->get_process_token();
    }
    return num_token;
  };

  void Scheduler::fillSequenceQueue(time_ns iter_time, time_ns total_time) {
    // add reqeust
    static int iter = 0;
    int num_request_to_inject = 0;
    if (system_config.use_inject_rate) {
      if (iter_time != 0) {
        time_ns start_time = total_time - iter_time;
        time_ns end_time = total_time;
        for (int i = 0; i < actual_arrival_time.size(); i++) {
          if (actual_arrival_time[i] >= start_time &&
              actual_arrival_time[i] < end_time) {
            num_request_to_inject++;
          }
        }
      } else {
        num_request_to_inject = total_batch_size * 0.7 - getBatchSize();
        num_request_to_inject = std::max(num_request_to_inject, 0);
      }
    } else {
      num_request_to_inject = total_batch_size - getBatchSize();
    }
    pushSeq(num_request_to_inject);
  }

  void Scheduler::hittingQueue(int iter) {
    for (int i = 0; i < iter; i++) {
      setMetadata();
      // run
      updateScheduler(0);

      // fill
      fillSequenceQueue();
      fillRunningQueue();
    }
  }

  void Scheduler::getActualArrivalTime(int num_iter) {
    // QUIRK (guide §16 / not in BUGS.md or BUGS_HIDDEN_BY_FLAGS.md): a Poisson
    // *process*'s inter-arrival times are exponentially distributed, not
    // Poisson-distributed -- std::poisson_distribution below draws integer counts
    // with the right mean (average_ns_per_request) but the wrong shape/variance
    // for inter-arrival gaps. Also note request_per_second == 0 divides by zero at
    // average_ns_per_request just below. Not exercised by any run_experiments.py
    // sweep (injection_rate is always 0 there); live only on a bare
    // `./run config.yaml` with injection_rate > 0. Documented, not fixed, per
    // explicit instruction -- see BUGS_FIXES.md for the intended
    // exponential_distribution(1.0 / average_ns_per_request) replacement.
    static unsigned int seed = 777;
    static std::default_random_engine generator(seed);
    double request_per_second = system_config.request_per_second;
    int average_ns_per_request = 1e+9 / request_per_second;
    std::poisson_distribution<> distribution(average_ns_per_request);

    std::vector<time_ns> inter_arrival_times(num_iter * 100);  // temp *10
    for (time_ns& time : inter_arrival_times) {
      time = distribution(generator);
    }

    std::vector<time_ns> actual_arrival_times(num_iter * 100);
    std::partial_sum(inter_arrival_times.begin(), inter_arrival_times.end(),
                     actual_arrival_times.begin());

    for (const time_ns& time : actual_arrival_times) {
      actual_arrival_time.push_back(time);  // second -> ns
    }
    cur_arrival_time_idx = 0;
  }

  void Scheduler::initExpertList(std::string expert_file_path) {
    std::ifstream openFile;

    if (!expert_file_path.compare("none")) {
      std::cout << "Using synthesis trace" << std::endl;
      real_data = false;
      real_expert_data = false;
      return;
    }

    real_data = true;
    real_expert_data = true;

    openFile.open(expert_file_path);
    if (!openFile.is_open()) {
      std::cout << "Using trace of mixtral, use random expert selection"
                << std::endl;
      real_expert_data = false;
      std::string file_path;
      file_path =
          "../expert_data/experts_mixtral_" + model_config.dataset + ".csv";

      openFile.open(file_path);
      if (!openFile.is_open()) {
        std::cout << file_path << std::endl;
        std::cout << "Cannot open mixtral, use random dataset" << std::endl;
        real_data = false;
        return;
      }
    }

    SequenceInfo::Ptr cur_seq = SequenceInfo::Ptr(new SequenceInfo);
    std::string line;

    int num_layer = model_config.expert_freq
        ? model_config.num_layers / model_config.expert_freq
        : model_config.num_layers;

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Loading sequences Start" << std::endl;

    bool process_sum = false;
    std::vector<std::vector<int>> sum_expert_list;

    int idx = 0;

    while (getline(openFile, line)) {
      std::istringstream iss(line);
      std::string key;

      int top_1, top_2;

      iss >> key;

      if (key == "Lin") {
        iss >> cur_seq->Lin;
      } else if (key == "Lout") {
        iss >> cur_seq->Lout;
        sequences_info.push_back(cur_seq);
        cur_seq = SequenceInfo::Ptr(new SequenceInfo);
      } else if (real_expert_data && key == "Sum") {
        cur_seq->expert_list.resize(cur_seq->Lin * num_layer);
        process_sum = true;
        idx = 0;
      } else if (real_expert_data && key == "Gen") {
        process_sum = false;
      } else if (real_expert_data && !(iss >> top_2).fail()) {
        top_1 = std::stoi(key);
        if (process_sum) {
          int token_id = idx % cur_seq->Lin;
          int layer_id = idx / cur_seq->Lin;
          cur_seq->expert_list.at(token_id * num_layer + layer_id) = {top_1,
                                                                      top_2};
          idx++;

        } else {
          cur_seq->expert_list.push_back({top_1, top_2});
        }
      } else if (key.substr(0, 4) == "seq_") {
        total_seq_num += 1;
      }
    }

    if (!real_expert_data) {
      for (auto& seq_info : sequences_info) {
        for (int token_id = 0; token_id < seq_info->Lin + seq_info->Lout;
             token_id++) {
          for (int layer_id = 0; layer_id < model_config.num_layers;
               layer_id++) {
            std::set<int> expert_list = getRandomExpert(model_config.top_k);
            std::vector<int> expert_vec;
            for (auto idx : expert_list) {
              expert_vec.push_back(idx);
            }
            seq_info->expert_list.push_back(expert_vec);
          }
        }
      }
    }
  };

}  // namespace llm_system