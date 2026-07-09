# Finding: `peakIntermediateBytes` over-counts concurrently-live attention intermediates

**For:** the LLMSimulator intermediate-data / SRAM-gate maintainers (the team working
BUGS.md items 9–18 and PAPER_INCONSISTENCIES.md's "uncounted-intermediate inventory").
**From:** the paper2-extension branch (Kyung et al. CAL 2026 replication), whose Fig-5
"required SRAM per device" output reuses `peakIntermediateBytes` and surfaced this.
**Type:** definitional over-count in a *shared* function — affects both the scarce-tier
capacity gate (`checkMemorySize` / `parallelism_optimizer`) and any metric that reports
`peakIntermediateBytes` as a required-SRAM figure. **Output-only** for paper2; for
paper1 it makes the capacity gate *conservative* (over-provisions), not unsafe.

This is the **over-count** counterpart to your existing **under-count** inventory
(missing LM-head logits, score tile, KV-write staging). They live in the same function
and partly offset — which is very likely why the paper's SRAM figure lands *between* a
naive estimate and a rigorous one.

---

## 1. Divergence

`peakIntermediateBytes` (`src/model/footprint.h:175`) returns `max(attn_total, ffn_total)`.
The `max()` across phases is correct (attention and FFN don't co-execute). **But within
the attention phase it SUMS every intermediate sub-tensor as if all are simultaneously
resident** (`footprint.h:186-200`, the `use_absorb` branch; the `compressed_kv` and
GQA-base branches sum analogously). In reality those tensors form a dataflow chain and
are freed as they're consumed, so the true peak concurrent bytes is the *max over the
chain*, not the *sum*. The sum is a safe upper bound for an OOM **gate**, but wrong as a
**required-SRAM** quantity.

The over-count is **architecture-dependent**: MLA (many attention intermediates) is
badly over-counted; GQA (few) barely. That is why, when this function is used as a
required-SRAM output, DeepSeek-R1 (MLA) *overshoots* the paper (+9…+21%) while
Llama-4-Maverick (GQA, and FFN-bound anyway) *undershoots* (−23…−41%). A single
"scatter" it is not — it is a systematic, per-architecture bias.

## 2. Evidence (exact, verified against the binary)

DeepSeek-R1 preset (H=7168, q_lora=1536, kv_lora=512, qk_rope=64, num_heads=128, tp=1),
`use_absorb` branch. The coded `attn_total` (pre-precision, element-units):

```
common_prefix = H(7168) + q_lora(1536) + kv_lora(512) + qk_rope(64)
              + (3*qk_rope + qk_nope)*num_heads = (192+128)*128 = 40960
attn_total    = common_prefix(50240)
              + tr_k_up(num_heads*kv_lora = 65536)
              + attn_context(num_heads*kv_lora = 65536)
              + v_up(num_heads*head_dim = 16384)
              + out_proj(H = 7168)
              = 204,864     <-- matches the binary's P2_SRAM_ACT_BYTES exactly
```

True **peak-of-chain liveness** (the max set that must coexist, at the score
computation): `H + kv_lora + qk_rope + query_proj(40960) + tr_k_up(65536) = 114,240`.

- `q_lora` (c_q) dies once `q_proj` is produced.
- `query_proj` and `tr_k_up` die once **scores** are computed — they never coexist with
  `attn_context`, `v_up`, or `out_proj` (which are produced *after* scores).
- So the sum double-counts the post-score tensors on top of the pre-score peak.

**Over-count = 204,864 / 114,240 = 1.79× (i.e. 45.8% high), and it is batch- and
context-INDEPENDENT** (ρ = 114,240/204,864 = 0.558 for every R1 cell).

## 3. How to trace/verify it yourself

1. Pick any MLA (`use_absorb`) preset. Read `footprint.h:186-200`; list each summed
   term and its producing op and its last consumer (the `// comments` name them).
2. Build the def→last-use interval per tensor along the chain
   `input → q_proj → (scores) → attn_context → v_up → out_proj`, with `input/residual`,
   `kv_lora`, `qk_rope` persistent across the phase.
3. Take the **max concurrent bytes** over the chain instead of the sum. For R1 it is the
   score-time set above (114,240) — 45.8% below the coded sum.
4. Cross-check: emit/print `P2_SRAM_ACT_BYTES` (or your gate's `activation_size`) for an
   R1 decode step; it equals the sum, not the peak.

## 4. Why the naive fix does NOT fully reconcile (important)

Replacing the sum with the rigorous peak-liveness (the correct computation) **flips R1
from +21% to −20%** against the paper — it does not converge. The paper's number sits
*between* the sum and the peak, because the same function ALSO **under-counts** the
tensors your BUGS 9–18 / inventory tracks (LM-head logits, the O(chunk) score/softmax
tile, and — for a required-SRAM figure — the KV-write staging burst). Over-count of
attention transients and under-count of omitted step-level tensors partly cancel.

So the only path to the paper's figure is **both** corrections together:
- **peak-liveness** for the attention intermediates (this finding), **and**
- the **missing-tensor inventory** (your existing work),
implemented as a **graph-based per-tensor liveness pass** (the module graph already
carries producer→consumer edges via `set_dependency_tensor` /
`dependency_tensor_list` in `module_graph.cpp`; it needs def/last-use intervals + a
max-concurrency sweep added). Anything less helps one model and hurts another.

## 5. Scope / non-goals

- **FFN is NOT over-counted.** `ffn_dense`/`ffn_moe` sum gate+up, which *genuinely*
  coexist (both feed the SiLU multiply), plus the down-proj and residual carries. So the
  fix is attention-specific. Maverick is FFN-bound (`ffn ≫ attn_total`), so this fix is
  inert for it — Maverick's undershoot is on the *missing-tensor* side, not here.
- For the **capacity gate** the current sum is conservative-safe (never under-provisions);
  this finding matters only if you want the gate *tight* or want to *report* required
  SRAM. For paper2 it is output-only (no effect on TPOT / throughput / lifetime).
- Refuted alternatives (don't bother): a flat I/O double-buffer `2·batch·hidden`
  (≈15.6 MB for R1, catastrophic undershoot); a separate weight-staging buffer (weights
  already stage through the same read double-buffer as KV — `layer_impl.h:39-51` — so a
  second one double-counts).

## 6. One-line summary

`peakIntermediateBytes` sums attention intermediates as concurrently-live (correct as an
upper-bound OOM gate, a 45.8% over-count for MLA as a required-SRAM figure); the true
fix is a graph-liveness pass reconciled with the missing-tensor inventory, since the
paper's SRAM sits between the two errors.
