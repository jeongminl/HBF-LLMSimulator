# Bugs hidden by pinned config flags

Every reported sweep pins several flags to one value (`decode_mode: on`, `disagg_system: off`, `use_chunked_attention` hardcoded `true` at all 5 call sites in `src/module/attention.cpp`, `parallel_execution: off`, and only non-MLA models – llama3_405B/llama4_maverick – actually get run despite `deepseekV3` being the `config.yaml` default). That leaves the branches these flags gate untested, and bugs are hiding in several of them. Found via a targeted read-only audit (three parallel agents, each independently verified against the source); **not fixed** – this is a findings list only. Ranked most-severe first. See `BUGS.md` for the prior session's separate (non-preset-related) bug list.

## ## 1. `num_kv_heads` divisibility guard missing in the MLA parallel wrappers, masked by `deepseekV3`'s own preset values

`src/module/parallel.cpp`'s `SelfAttentionParallel` (used for GQA/non-MLA models) asserts both `num_heads % parallel_num == 0` *and* `num_kv_heads % parallel_num == 0` (Lines 144–145) before dividing. Its MLA siblings, `MultiLatentAttentionParallel` (Line 223) and `AbsorbMLAParallel` (Line 291), assert only `num_heads` – the `num_kv_heads` guard was dropped – yet still compute `num_kv_heads / parallel_num` and pass it down. `AbsorbMLASum::forward` (`attention.cpp:547`) and `MultiLatentAttentionSum::forward` (`attention.cpp:415`) then compute `attention_group_size = num_heads / num_kv_heads` unconditionally on the first decode step, so a `num_kv_heads/parallel_num` that floors to 0 is an immediate `SIGFPE`.

This is invisible today only because `model_config.h`'s `deepseekV3` preset sets `num_kv_heads = num_heads = 128`, which trivially divides by any TP degree tried – it does not model MLA's actual *"one shared latent KV"* semantics (which would be `num_kv_heads = 1`). The moment `deepseekV3` (or any new MLA model) is corrected to a physically accurate `num_kv_heads` at TP > 1, this crashes.

## ## 2. 32-bit integer overflow in the prefill/mixed-mode activation-footprint formula, masked by `decode_mode: on`

`src/hardware/cluster.cpp`'s `checkMemorySize()` non-decode (`else`) branch (Lines ~120–165) computes terms like `batch_size_per_dp * input_len * input_len * num_heads / ne_tp_dg` (line 154, base model; line 130, absorb) entirely in `int` arithmetic – `batch_size_per_dp`, `input_len`, `num_heads`, `ne_tp_dg` are all declared `int` – before the result is ever multiplied by a `double` literal. With realistic `input_len` (thousands) and `num_heads` (128), this overflows `INT_MAX` for ordinary batch sizes, corrupting the printed `ACT:`/`Total:` values and the scarce-tier OOM gate that consumes `activation_size` (`cluster.cpp:196–205`).

The decode-mode formula (`peakIntermediateBytes` in `footprint.h`) is immune because every operand there is `double`. Fully masked because `decode_mode: on` is pinned in every real run; flip it off (or set `prefill_mode: on`) and this fires.

## ## 3. Scheduler prefill token allocator can floor to zero and stall the run forever, masked by `decode_mode: on`

`src/scheduler/scheduler.cpp:298`: `num_process = max_process_token / num_sum_seq` is integer division. `config.yaml`'s default `max_process_token: 0`, so any prefill ("sum") sequence gets `num_process_token = 0` forever – `current_len` never advances, `hasSumSeq()` stays permanently true, and `runIterationMixed` spins with zero progress for the whole `iter` loop the moment prefill sequences actually exist.

## ## 4. Disaggregated-path ("sum"/prefill) CSV rows are never populated, masked by `disagg_system: off`

In `Cluster::runIterationSumGenSplit` (`cluster.cpp:624-636`), the prefill branch builds a `Stat` but never calls `setStat`/`setTimeBreakDown` and never fills the energy fields, unlike the gen branch (`:601-614`). Timing itself (`sum_machine_time`) is correct and does not leak into `total_time`, but every prefill row would report zeroed batch/energy/breakdown columns the moment `disagg_system: on` is used for a real run.

## ## 5. PIM attention-gen duration formula diverges from the GPU/LOGIC formula, masked by `parallel_execution: off`

GPU (`attention_gen_impl.cpp:109,187`) and LOGIC (`:299,347`) both compute `total_duration += std::max(compute_duration, memory_duration)`. The PIM variant instead does `total_duration += accumul_memory_duration * opb` where `opb = total_flops / total_memory_size` (lines 449, 499) – inflating PIM decode-attention latency by roughly the compute/memory ratio instead of taking the overlap-max. Its Softmax section is also just a bare `// Softmax //` comment with no code (Line 451), so LOGIC/PIM silently skip the softmax time GPU charges (`:111–127`). Only matters if attention-gen is ever dispatched to PIM/LOGIC, which requires `parallel_execution`/`hetero_subbatch` or a `processor_type` list including them.

## ## 6. `getIdxHigh` hard-asserts (crash) for expert-per-device counts >4 not divisible by 4, masked by `parallel_execution: off`

`src/module/route.cpp:85`: `assertTrue(list.size() % 4 == 0, ...)`. `list.size()` is `num_expert_per_device` (`expert.cpp:51`); any EP split producing e.g. 10 or 20 experts per device aborts the moment expert routing runs under a heterogeneous GPU/PIM assignment.

## ## 7. Lower-confidence / secondary findings (same root cause: pinned flags hiding untested branches)

- `logic_x`/`pim_x` are used by the expert high/low assignment heuristic (`route.cpp`'s `getIdxHigh`/`getIdxHighOptimal`) to predict a speedup for the "low" (LOGIC/PIM) processor, but the actual LOGIC/PIM kernel timing never applies that multiplier – inconsistent, masked by `parallel_execution: off`.
- MLA prefill attention on HBF (`attention_sum_impl.cpp`'s `MultiLatentAttentionSumExecutionGPU`/`Logic`) never calls `getAttentionMemoryDuration`, so flash KV-read timing isn't modeled for MLA prefill – double-dark, since it needs an MLA model *and* prefill mode together.
- `head_dim` is used where `qk_nope_head_dim` is meant in the absorb `attn_o_proj` wiring (`layer.cpp:130–133`, `footprint.h:161`) – a real bug for any MLA config where `head_dim != qk_nope_head_dim`, though real DeepSeek-V3 (and this codebase's preset) also has them equal, so it isn't strictly "hidden by an artificial preset value," just untested for models where they'd diverge.
- `use_chunked_attention`'s non-chunked branch (`layer_impl.h:70–80`) is currently a dead no-op: with `chunk_size: 0` (the pinned default) it computes byte-for-byte the same result as the chunked branch, so flipping the flag today changes nothing – its real fallback behavior (one page-latency per full KV read instead of per chunk) has never actually been exercised.
