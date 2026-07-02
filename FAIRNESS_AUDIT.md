# HBF-LLMSimulator — Cross-Configuration Fairness Bug Audit

**Purpose.** Identify bugs that unfairly penalize (or favor) one of the five memory presets
(HBM4, HBF, HBF+, CONV, CONV+) in cross-configuration comparisons. This document is written
for another AI/reviewer to **vet independently** — every finding lists exact `file:line`,
what the code does, why it is (or is not) a bug, the config it skews, a severity, and a
verification hook. Findings I personally confirmed against source are marked **[verified]**;
items I take from subagent audits pending independent re-check are marked **[reported]**.

**Method.** Three parallel read-only audits (memory/timing model; parallelism optimizer +
capacity/SRAM; scheduler/accounting/PEC/normalization) plus first-hand reads of
`layer_impl.h`, `hbf_memory_config.h`, `footprint.h`, `eval/test.cpp`,
`parallelism_optimizer.cpp` (latency + capacity paths), and `run_experiments.py`
(batch-size search). Spec of record: `INSTRUCTIONS.md` (esp. §5 Metric 1 two-phase search,
§6 optimizer/simulator separation of concerns).

---

## Severity summary

| ID | Severity | Bug | Skews |
|----|----------|-----|-------|
| **F1** | **High** | Analytic latency estimate can reject a config (batch→0) with **no simulator run**, violating §6 "simulator is sole SLO arbiter" | Penalizes CONV/CONV+ and any flash preset under tight SLO (0.05s) on LONG |
| F2 | Low–Med | Linear/FFN weight-read charges flash page-read latency **once per op**, not per SRAM chunk — inconsistent with attention KV-read (per-chunk) | Favors flash, most CONV/CONV+ |
| F3 | Low | Analytic `max`-model adds KV read/write on top of `max(compute,weight)`, can over-predict flash latency; this is the *mechanism* feeding F1 | Penalizes flash in Phase-1 seeding |
| F4 | Low | `tp > num_kv_heads`: simulator integer-divides `num_kv_heads/tp → 0` (zeros KV) while optimizer keeps fractional | Memory-type agnostic; opt↔sim divergence |
| F5 | Low | Analytic KV-write overlap subtracts *total* per-layer compute (incl. FFN); simulator subtracts only attention compute | Optimistic for flash (safe direction), but a divergence |
| F6 | Info | Disagg path (`disagg_system=on`) lost its KV-write critical-path term | Unused in current sweeps; would understate flash if enabled |
| F7 | Info | `chunk_size` is interpreted in **bytes**, no unit guard; a token-valued config would shatter chunking | Would massively penalize flash if misconfigured |
| F8 | Info | Optimizer flash capacity check excludes activation from flash pool; simulator includes it | Negligible; slight over-strictness on sim side for flash |

**Considered and dismissed** (not bugs): SRAM modeled as infinite-bandwidth / 0-ns activation
time on HBF+/CONV+ (**intentional by design**, confirmed by user); the 5 presets match spec
numerically; HBM4 correctly exempt from flash/SRAM limits and KV-write penalty; KV-write
penalty is not double-counted; PEC units, baseline normalization, steady-state batching,
and batch-sweep uncapping are correct. Details in the last section.

---

## F1 — [verified] Analytic latency can veto a configuration without the simulator (High)

**Spec violated.** `INSTRUCTIONS.md` §6: "GPU-memory capacity and the SRAM/intermediate-data
limit are exact and may be enforced analytically by the optimizer. Latency and SLO
satisfaction may not... [the analytic estimate] must never itself reject a batch size or
produce a value that appears in a reported metric." The simulator's measured TPOT is the
sole SLO arbiter.

**Code path.**
1. `eval/test.cpp:290-298` — analytic Phase-1 feasibility test:
   ```cpp
   auto feasible = [&](int b, ParallelConfig& out) -> bool {
     out = ParallelismOptimizer::Optimize(...);
     return !out.oom && out.estimated_latency_ms <= tpot_slo_ms;   // <-- latency gate
   };
   ...
   if (!feasible(1, best)) { std::cout << "ANALYTIC_MAX_BATCH: 0"; return 0; }
   ```
   `out.oom` is pure capacity/SRAM (`parallelism_optimizer.cpp:498-503` — `Optimize()` never
   sets `oom` from latency **[verified]**). So the second conjunct,
   `estimated_latency_ms <= slo`, is an **independent latency gate**. If the analytic latency
   at batch=1 exceeds the SLO, `feasible(1)` is false and the binary emits
   `ANALYTIC_MAX_BATCH: 0`.
2. `run_experiments.py:184-186` — Python consumes that:
   ```python
   b_analytic = run_analytic_sweep(...)
   if b_analytic <= 0:
       return 0, 0.0, None, None, None    # <-- NO simulator call
   ```
   Returns `max_batch=0` → TPS=0, PEC N/A, normalized metric 0, **without ever constructing
   the Scheduler/Cluster/Model or running the discrete-event simulator.**

**Why it is a bug.** `b_analytic == 0` conflates two distinct causes:
- *Capacity/SRAM OOM at B=1* — a legitimate analytic rejection (exact constraint). Fine.
- *Analytic latency > SLO at B=1* — an **illegitimate** rejection: the simulator, which
  models SRAM double-buffered read overlap the analytic model treats additively (see F3),
  is never given the chance to accept B=1.

The `b_analytic > 0` path is spec-compliant: it verifies `b_analytic`, boundary-checks
`b_analytic+1`, and searches upward or downward with the simulator
(`run_experiments.py:194-255` **[verified]**). Only the `== 0` short-circuit bypasses the
simulator. The binary even prints an `OVERESTIMATE` warning
(`eval/test.cpp:611-616`) acknowledging "analytic > measured ... the direction that can
cause the search to silently reject a batch the simulator would accept" — but the `==0`
short-circuit does exactly that with no recovery.

**Config skew.** Differentially zeroes the highest-latency presets — **CONV/CONV+**
(2.45/2.80 TB/s read, 3 µs page-read) and any flash preset under the 0.05 s SLO on LONG
(103500-token) — precisely the cases whose analytic per-layer latency at B=1 can exceed the
SLO while the simulator (with read-overlap) might accept. For a B=1 LONG sequence this is a
*latency*, not a capacity, rejection (weight+KV of one sequence fits the 4096 GB pool
easily). HBM4 (fast, symmetric) essentially never trips it, so the **baseline is unaffected
while flash configs get zeroed** — the worst kind of asymmetry for a comparison paper.

**Candidate remediation (for vetting, not yet implemented).**
- Make the analytic Phase-1 lower bound depend on **capacity/SRAM only**; use the latency
  estimate solely to pick the candidate batch, never to force 0. Concretely, in
  `eval/test.cpp` separate "capacity-feasible at B=1" from "analytic-latency-feasible" and
  emit a distinct marker (e.g. `ANALYTIC_CAP_FEASIBLE: 1`), **or**
- In `run_experiments.py`, when `b_analytic <= 0`, still call `verify(1)` with the simulator
  before declaring the config infeasible — the simulator remains the arbiter and the cost is
  a single sim run.
The second is the smaller change and directly enforces "simulator has final say."

**Verification hook.** Run CONV+ / llama3_405B / LONG / 0.05 s / 8-GPU through
`run_analytic_sweep`; confirm it emits `ANALYTIC_MAX_BATCH: 0`, then manually run the
simulator at B=1 (`optimize_parallelism: true`, `analytic_sweep_only: false`) and check
whether measured TPOT ≤ 50 ms. If the simulator would pass B=1, F1 is demonstrated.

---

## F2 — [verified] Linear weight-read charges page-read latency once per op, not per chunk (Low–Med)

`src/hardware/layer_impl.h:26` (in `getLinearMemoryDuration`):
```cpp
double weight_read_time = (weight_size / hbf.flash_read_bandwidth * 1e9)
                          + hbf.flash_page_read_latency_ns;   // <-- single latency
```
The attention KV-read path in the same file (`layer_impl.h:69, 79`) correctly charges
`num_chunks * flash_page_read_latency_ns` where
`num_chunks = ceil(kv_read_size / sram_capacity)` — one page-read latency per 3.13 MB×N
double-buffer chunk. A FFN/projection weight tensor is many times larger than the ~25 MB
SRAM staging pool, so if weights stage through the same buffer they should also incur
multiple page-read latencies. Charging one flattens flash's latency cost.

**Note / nuance.** The optimizer is *internally consistent* with the simulator here: it also
charges one page-read latency **per linear op** via `page_lat_total`
(`parallelism_optimizer.cpp:360-380` **[verified]**), not per chunk. So this is an
**internal modeling inconsistency (weight-read vs KV-read), not an opt↔sim mismatch**.
Magnitude is bounded by (chunks_per_weight_op − 1) × page_lat (1–3 µs) × ops × layers;
low relative to bandwidth time but larger for CONV/CONV+ (3 µs, low BW). Whether weights
even stage through the double-buffer identically to KV is a modeling choice worth
confirming with the author before "fixing."

**Config skew.** Favors all flash configs, most CONV/CONV+.

---

## F3 — [verified] Analytic `max`-model over-predicts flash latency (Low; mechanism behind F1)

`parallelism_optimizer.cpp:435-443` (default `optimizer_latency_model = max`):
```cpp
double layer_ns = std::max(compute_per_layer, weight_mem_per_layer)
                  + kv_read_per_layer + kv_write_per_layer;   // KV added, not overlapped
```
For flash, `kv_read_per_layer` includes additive page-read latency
(`:406-407`) and `kv_write_per_layer` is flash-only (`:413-433`). The simulator models
attention memory as `max(kv_read_time, act_time)` and overlaps KV-write with compute
(`layer_impl.h:86`, `attention_gen_impl.cpp:864`). So the analytic estimate can **over**-
predict flash latency relative to the simulator. On its own this only over-conservatively
*seeds* Phase-1 (safe, since the simulator upward-fallback corrects it) — **except** when it
drives `b_analytic` to 0, at which point F1's short-circuit makes it a silent rejection. F3
is thus the mechanism that makes F1 bite flash specifically.

**Config skew.** Penalizes flash in analytic ranking/seeding.

---

## F4 — [reported, spot-checked] `tp > num_kv_heads` integer-division divergence (Low)

Simulator `src/module/parallel.cpp:150` computes `num_kv_heads / parallel_num` in **integer**
arithmetic → 0 when `tp > num_kv_heads`, zeroing KV geometry; optimizer
`parallelism_optimizer.cpp:232` keeps a fractional `num_kv_heads/(double)tp`. Affects GQA
models at very high TP. **Memory-type agnostic** (same across presets), so not a fairness-
across-config bug, but a real opt↔sim inconsistency and a latent correctness issue for
high-TP sweeps. Worth confirming whether any swept `(model, tp)` actually hits
`tp > num_kv_heads`.

---

## F5 — [reported] Analytic vs simulator KV-write overlap basis differs (Low)

Optimizer subtracts *total per-layer compute* (attention + FFN) when computing unhidden KV
write (`parallelism_optimizer.cpp:430-431`), whereas the simulator subtracts only per-layer
**attention** compute (`attention_gen_impl.cpp:864`). Optimizer therefore under-counts the
flash write penalty (optimistic) — the "safe" search direction, but a divergence that should
be reconciled so the drift harness (`test.cpp:585-617`) compares like with like.

---

## F6 — [reported] Disaggregated path lost its KV-write critical-path term (Info)

The removed block in `cluster.cpp::runIterationSumGenSplit` was the only KV-write critical-
path term in the `disagg_system=on` path (which is `off` in all current sweeps). The removed
code was itself wrong (charged NVLink `transfer_latency`, not `flash_write` time), so removal
is defensible — but the disagg path now has **no** KV-write penalty. Flag before enabling
disaggregated runs.

> **[CORRECTION, independently re-verified]** This claim is **REFUTED** as stated. Both
> `runIterationMixed` and `runIterationSumGenSplit` call the identical `run()` → executor →
> module-graph chain, which unconditionally registers `AttentionGenExecutionGPU`/MLA kernels
> (`executor.cpp:59,88` — not gated on `disagg_system`). Those kernels contain the correct
> per-decode-step KV-write overlap logic (`attention_gen_impl.cpp:201-204` and MLA variants),
> which fires identically on both paths — **the disagg path still has its per-step KV-write
> penalty.** What the deleted block actually was: a disagg-specific *one-time prefill→decode
> KV-transfer* term (interconnect-bound, sized on full `input_len`, once per admitted
> sequence) — a physically distinct quantity from the steady-state incremental write penalty,
> and (per this file's own note) never modeled a flash *write* even before deletion. Also
> separately investigated per a follow-up question: `runIterationMixed`'s unconditional
> `total_time += time` (no sum/gen distinction) is real but **inert** under every sweep in
> this repo, because `decode_mode: on` (a separate flag from `disagg_system`) guarantees
> `hasSumSeq()` is always false — see the comment added at `cluster.cpp`'s
> `runIterationMixed` for the mechanism and the one latent fragility this surfaced (no
> defensive check if `decode_mode` were ever disabled).

---

## F7 — [verified] `chunk_size` unit footgun (Info / latent)

`layer_impl.h:61-64` and `parallelism_optimizer.cpp:397-401` interpret `system.chunk_size`
in **bytes**, clamped to SRAM capacity, with no unit guard. A user setting `chunk_size` in
**tokens** (a natural unit for "chunked attention") — e.g. 512 — would make
`num_chunks = kv_read_size/512`, charging a page-read latency per ~512 bytes and
catastrophically penalizing flash. The spec's "make chunk size configurable" invites exactly
this. Default `0` (auto = full SRAM capacity) is correct; recommend a sanity clamp/warning
(reject `0 < chunk_size < page_size_bytes`).

Related spec-compliance note: `use_chunked_attention` defaults false and is never set, so the
"chunked attention enabled globally" requirement is only met because the non-chunked branch
also chunks by SRAM. Behaviorally equivalent today, but brittle.

> **[CORRECTION, independently re-verified]** The "never set" claim is **factually incorrect**.
> `src/module/attention.cpp` sets `layer_info.use_chunked_attention = true` unconditionally at
> 5 call sites (lines 62, 113, 156, 363, 490), each explicitly commented "chunked attention
> should always be used regardless of configuration." This predates the current session; it
> is not new/recent code. The main finding's practical conclusion ("behaviorally equivalent
> today") is still correct, just for the wrong reason — chunked attention *is* forced on, not
> merely coincidentally equivalent via the non-chunked branch. The chunk_size unit-guard
> recommendation itself stands and has been implemented (`eval/test.cpp`'s chunk_size
> parsing now rejects `0 < chunk_size < page_size_bytes`).

---

## F8 — [reported] Optimizer vs simulator capacity *scope* mismatch (Info)

Optimizer flash check tests `weight + kv > flash_cap` (activation excluded, kept on scarce
tier — `footprint.h:67` **[verified]**); simulator total gate includes activation in the
same pool (`cluster.cpp:314`, per subagent). Because activation ≤ 320 MB (HBF+) / 36 GB (HBF)
vs a 3620–4096 GB pool, the practical effect is negligible, but it is a genuine divergence
(slightly over-strict on the simulator side for flash).

---

## Considered and dismissed (evidence these are NOT bugs)

- **SRAM infinite-bandwidth / 0-ns activations on HBF+/CONV+** — `layer_impl.h:34-37, 82-85`
  (`act_time = 0` when `num_hbm_stacks == 0`) and `activation_impl.cpp`. **Intentional
  modeling assumption, confirmed by user.** Not a bug; do not "fix."
- **Presets** — `hbf_memory_config.h:54-137` match `INSTRUCTIONS.md` §1 exactly (capacities
  288/3620/4096/3620/4096 GB; read 12.8/11.2/12.8/2.45/2.80 TB/s; write
  12.8/0.112/0.128/0.073/0.084 TB/s; page-read 0/1/1/3/3 µs; program 100 µs; SRAM
  3.13 MB/stack; logic SRAM 320 MB on "+"). **[verified]**
- **HBM4 exemption** — all flash/SRAM logic gated on `use_hbf && num_flash_stacks > 0`;
  `hbm4_preset` has `num_flash_stacks = 0`, so HBM4 pays no KV-write penalty
  (`getKVWriteDuration` returns 0), is not subjected to the SRAM activation limit
  (`footprint.h:37, 50`), and uses the lumped 288 GB capacity check. **[verified]**
- **Per-preset capacity & SRAM limit** — `footprint.h:36-96` applies the right limit to the
  right config (36 GB reserved-HBM for HBF/CONV, 320 MB logic SRAM for HBF+/CONV+); no
  cross-application of HBM4's 288 GB to flash presets. **[verified]**
- **KV-write not double-counted** — sized on `input_len`, amortized as
  `num_seq/output_len` (steady-state admission), only `max(0, kv_write − compute)` added to
  the critical path, then subtracted back out of the attention bucket in `setTimeBreakDown`
  so it is counted exactly once. **[verified formula; bucket subtraction reported]**
- **PEC / write traffic** — `test.cpp:566-576` emits per-token KV geometry + flash capacity;
  Python ratio is per-GPU-write ÷ per-GPU-capacity = system ÷ system (invariant); GiB used
  uniformly across flash presets. **[reported]**
- **Baseline normalization** — main metrics normalize to 8-GPU HBM4 at the matching SLO;
  SLO-sensitivity normalizes to HBM4 at the *same* SLO. **[reported]**
- **Batch sweep uncapped + OOM detection** — no 128 cap (analytic bound to 2^30,
  Python exponential+binary); OOM via `EXIT_FAILURE` and stdout markers. **[verified: test.cpp:308; reported: python]**
- **Steady-state batching** — `injection_rate=0 → fillSequenceQueue` refills to full batch
  every step, config-independent. **[reported]**

---

## Recommended vetting order for the next reviewer

1. **F1** — reproduce the CONV+/LONG/0.05 s zeroing and confirm the simulator would accept
   B=1 (highest impact; directly the "simulator must have final say" concern).
2. **F3** — quantify analytic-vs-measured latency gap at B=1 for CONV/CONV+ to bound how
   often F1 fires.
3. **F2** — decide with the author whether weights stage through the double-buffer (per-chunk
   latency) or not (per-op), then make weight-read and KV-read latency models consistent.
4. **F4/F5/F8** — reconcile opt↔sim divergences; confirm none affect a *reported* metric.
5. **F7** — add a `chunk_size` unit guard.

No code was modified in producing this report.
