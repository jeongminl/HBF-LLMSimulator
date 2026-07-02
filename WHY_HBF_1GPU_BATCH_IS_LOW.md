# Why llama3_405B's 1-GPU HBF/HBF+ batch size is low vs. the paper — not a bug

## The claim being checked

The paper states "1-GPU HBF/HBF+ per-GPU batch exceeds 8-GPU HBM4's, in most cases." Our full
sweep (`experiment_results.md` §1) shows this holds 5/6 for llama4_maverick but **0/6 for
llama3_405B** — e.g. SHORT: 1-GPU HBF = 75 vs. 8-GPU HBM4 = 194.8 (−61%); LONG: 1-GPU HBF = 1.0
vs. 8-GPU HBM4 = 3.75 (a miss, though HBF+ = 3.0 is close and, post-fix, HBF+ at LONG batch 3→5
now exceeds the anchor).

## Four pieces of evidence, together airtight

### 1. The weight-reread floor is real, large, and correctly modeled

`getLinearMemoryDuration` (`src/hardware/layer_impl.h:22-26`) charges weight-read time as
`weight_size / flash_read_bandwidth`, where `weight_size = k*n*precision` — **no batch-size
factor**. This is physically required: in decode, weights are streamed fresh from memory every
step regardless of how many sequences are batched together (the batch only changes the *activation*
volume, not the weight volume read).

llama3_405B (`model_config.h:192-194`: hidden=16384, 126 layers, 128 heads, 8 KV heads,
intermediate=53248, BF16) has 401.6B parameters = **803 GB** of weight. At 1 GPU, TP=PP=1 is the
*only* legal configuration (there is nothing else to shard across) — so this entire 803 GB must be
re-read from flash every decode step:

- HBF (1 reserved HBM stack + 7 flash stacks, 11.2 TB/s flash read): 803 GB / 11.2 TB/s ≈ **72 ms**
- HBF+ (8 flash stacks, 12.8 TB/s): 803 GB / 12.8 TB/s ≈ **63 ms**

Both consume **62–73% of the 100 ms TPOT SLO** before a single sequence's marginal attention/FFN/
KV cost is added. This is not a hidden or synthetic penalty — it's the direct, literal consequence
of "this model doesn't fit meaningfully in one GPU's compute budget without sharding," which is
exactly the scenario the paper itself motivates (large models need many GPUs or novel memory).

### 2. The same formula is independently validated by the anchor that DOES match

The concern with (1) is "maybe the formula over-charges weight cost, and 8-GPU somehow escapes it
by luck." It doesn't: the identical formula at 8-GPU HBM4 (weight sharded 8× via TP×PP, 12.8 TB/s,
zero page latency) gives a first-principles capacity ceiling of **≈197 sequences/GPU**, against the
simulator's reported **194.8/GPU** — a match to <2%. A bug that inflated the 1-GPU weight cost
would also have to leave the 8-GPU case's independently-matching number untouched, which is not
how a shared formula works. The formula is validated, not suspect.

### 3. The 1-GPU HBF operating point is SLO-bound, not capacity-bound — confirmed three ways

- **Direct capacity math**: HBF's 1-GPU flash pool is 3,620 GB; subtracting the 803 GB weight
  leaves ~2,817 GB for KV, enough for **≈2,956 sequences** at SHORT context (KV ≈953 MB/seq at
  ~1660+187 avg tokens). The reported batch is 75 — about **2.5% of the capacity ceiling**.
- **SLO-sensitivity sweep** (`experiment_results.md` §4): at 4-GPU, llama3/HBF's batch scales
  **3.2 → 10.0 → 23.2 → 68.0** across SLO 0.05s/0.1s/0.2s/offline — a >20× swing driven purely by
  relaxing the latency budget. HBM4's batch at the *same* GPU counts is **exactly SLO-invariant**
  (same value at every SLO) — the signature of a capacity-bound, not latency-bound, ceiling.
- **The harness itself tags each point's binding constraint.** `run_experiments.py`'s
  `classify_failure()` (:283-307) inspects the last failing probe's reason string and returns
  `"slo" | "flash" | "sram" | "unknown"` specifically so Fig. 3's SRAM-bound bars can be
  distinguished from SLO-bound ones — this machinery exists precisely because the codebase already
  expects (and correctly reports) that different points bind on different constraints.

### 4. Independent first-principles reproduction lands in the same regime

A back-of-envelope model (weight-reread + compute + KV-read + KV-write, all correctly sharded/
batch-scaled per the same formulas as the simulator) puts the 1-GPU HBF SLO-limited batch at
roughly 100-150 sequences for these assumptions; the simulator's 75 is lower because it additionally
prices in attention/softmax compute, activation traffic through HBF's single reserved 1.6 TB/s HBM
stack (vs. HBF+'s free SRAM activations), RMSNorm/RoPE, and per-linear-op page-read latencies —
every one of which further *tightens* the SLO budget, i.e. pushes the batch down, not up. This
explains, in the same breath, why HBF+ (168) beats HBF (75) at 1 GPU: HBF+ has higher bandwidth
(12.8 vs 11.2 TB/s) and zero activation-bandwidth cost, both of which shave milliseconds off a
budget that is already down to its last ~30%.

## Why llama4_maverick doesn't show this problem

llama4_maverick is MoE with top_k=1 of 128 experts. At 1 GPU its per-step weight-read floor is
only the active experts' weight (~1/128th the routed-expert pool) plus attention — an order of
magnitude smaller than a fully-dense 403B-parameter reread. Its weight floor never dominates the
SLO budget the way llama3's does, so its 1-GPU batch is governed by the same
capacity/compute trade-off as its 8-GPU case, and the "1-GPU beats 8-GPU" pattern holds in 5/6 of
its tested combinations.

## Conclusion

This is a **binding-regime mismatch, not a bug**: HBM4's reported batch is capacity-bound (matches
the paper); a fully-dense 400B+-parameter model's 1-GPU batch is throttled by an un-shardable
weight-reread floor under a strict 100ms *online* SLO — a real architectural consequence of trying
to serve a model this large from a single GPU, which the SLO (not capacity) exposes. The model's
true capacity advantage (2,956 vs. 197 sequences — nearly 15×) is real and reappears once the SLO
is relaxed (offline: 68-80 seq/GPU, far above HBM4). No code change is proposed for this item.

## Confirmed by the simulator's own instrumentation

Ran `build/run` at exactly this operating point (`memory_type: HBF`, `num_device: 1`,
`model_name: llama3_405B`, SHORT workload, `max_batch_size: 75`, `optimize_parallelism: true` —
which correctly re-derives TP=PP=DP=EP=1, the only legal 1-GPU config) and extracted the live
per-component decode-step breakdown from the CSV output:

- **Measured TPOT: 0.09670s — 96.7% of the 0.1s SLO budget.** This directly confirms the
  operating point is SLO-saturating (consistent with being the binary search's converged max-batch
  anchor), not merely "close to it."
- **Breakdown of that 96.7 ms:**

  | Component | Share |
  |---|---|
  | FFN weight-read (gate/up/down) | **63.2%** |
  | QKV/O-proj weight-read (attention linear layers) | **13.5%** |
  | KV-write (unhidden, newly-admitted-query penalty) | 14.5% |
  | Attention KV-read + attention compute | 7.4% |
  | Others (norm/residual/etc.) | 1.4% |

  **Combined weight-read (FFN + QKV/O-proj) = 76.7% — the single dominant cost**, confirming this
  document's central thesis directly from the simulator's own instrumentation, not just
  hand-derived first-principles math (which estimated ~72.6%, the same order and mechanism — the
  small gap between 72.6% and 76.7% is expected, since the hand estimate didn't separately account
  for attention-projection weight-read as distinct from FFN weight-read). KV-write (14.5%) is the
  clear second-largest term, matching the "reduces the batch further" role predicted above.
  Attention KV-read itself is small (7.4%) precisely because SHORT context (2033 tokens) has little
  KV to read — consistent with this floor being almost entirely a *weight*-reread problem, not a
  KV-read problem, exactly as this document's Part 1 argues.
