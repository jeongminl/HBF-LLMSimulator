# MoE DP/EP Conflation Fix — Design Spec

> **RESOLVED 2026-07-08 — WON'T IMPLEMENT (not a bug).** The §7 topology decision was settled by
> checking the authors' reference tool **CB1 (`LLMSimulator_HBF`)**: CB1 implements **Model B
> (EP-across-DP)** — routed-expert weights are sharded across the whole `ne_tp·dp` pipeline-stage
> device set (`num_expert_per_device = num_routed·e_tp/(ne_tp·dp)`), the all-to-all pools tokens
> across dp replicas onto disjoint expert owners, and there is **no orthogonal `ep` knob** (`dp` is
> derived, and expert parallelism = `(ne_tp·dp)/e_tp` by construction; CB1 is single-config, no
> sweep). **This fork already matches CB1 exactly** (verified: `expert.cpp:42,50-51`,
> `decoder.cpp:137-139`, `parallelism_optimizer.cpp:174` all unmodified EP-across-DP). The proposed
> Model-A change ("dp = pure replication", `e_tp ≤ ne_tp`) would DIVERGE from the reference, so it is
> **not applied**. Two asymmetries CB1 encodes and this fork also has, kept for the record: **routed**
> experts absorb dp (Model B), but **shared** experts and the `e_tp` tensor-shard stay within a
> replica (`non_moe_device_list`, `use_dp=true`). This doc is retained as historical design analysis;
> archive candidate for `ledgers/`. Everything below is the pre-resolution spec.

Author: fix DESIGNER (this doc is the SPEC; a Sonnet implementer applies it, ON TOP of PP_FIX + MOE_TAG + PP_FLAGS, only after the fixer gate report is signed off).
Read-only reminder for this doc's author: no `src/` edits, no build. The implementer builds.
Context: CAMPAIGN_NOTES.md "Hunt H mechanism closure" + Refuter H Phase-2; source-verified anchors below.

> **HEADLINE FINDING (read first).** The current MoE parallelization is a *self-consistent* **EP-across-DP** scheme: BOTH the runtime (`expert.cpp:50-51`) AND the optimizer (`parallelism_optimizer.cpp:174`, comment `:171-173`) shard the routed-expert **weight per device by `devices_per_stage = ne_tp·dp`**, independent of `e_tp`. The a2a genuinely pools tokens cross-dp onto owners (`communication.cpp:247/413`, `running_queue[src_dp_idx]->local_num_token_in_expert`), so the GLOBAL `m` that Route uses is *correct for that scheme*. **Hunt H's two listed sites (route-local `m` + `weightReadOps`) are NECESSARY but INSUFFICIENT**: neither touches the routed-expert **weight bytes/device**, which is the `65.66×16/128 = 8.2 ms` collapse driver. Route-local fixes only the compute/act **high-mode** (removing the bimodality artifact) and `weightReadOps` fixes only a µs-scale page-latency term — the device would still stream **16** experts, not the physical **128**, at dp8/ep1. Restoring the physical floor requires changing **`num_expert_per_device` (module construction) AND the optimizer's routed-weight/e_active/e_tp-sweep terms** — ~7 coordinated sites, detailed in §3. §7 states the one topology decision the team-lead must confirm before the implementer proceeds.

---

## 1. Confirmed code facts (file:line)

**Layout.** `llm.cpp:27-31` builds `stage_device_list` = every rank `r` with `(r/ne_tp)%pp == pp_stage` → size `= num_total_device/pp = ne_tp·dp` (ALL dp replicas' devices for the stage). `decoder.cpp:138-139` passes this `device_list` to `ExpertFFN::Create`. (Attention/dense/shared-expert use `non_moe_device_list` = the `ne_tp` tp-peers, `decoder.cpp:20-21/105-106`, `expert.cpp:20-22` — so the SHARED expert is already per-replica-`ne_tp`, correct; only ROUTED experts have the issue.)

**Routed-expert partition (the EP-across-dp mechanism).** `expert.cpp:42` `parallel_num = device_list.size() = ne_tp·dp`; `:50` `num_expert_tp_gr_rank = parallel_num/e_tp_dg`; `:51` `num_expert_per_device = num_routed_expert / num_expert_tp_gr_rank = num_routed·e_tp/(ne_tp·dp)`; `:61-62` `expert_offset = num_expert_per_device·(local_rank/e_tp)` (distinct expert range per group, indexed across ALL `ne_tp·dp` devices). The expert FFN linears are column/row-parallel over `expert_device_list` (size `e_tp`, `:63-66`), so each tensor is `1/e_tp` wide. **Weight bytes/device** `= num_expert_per_device × full_expert/e_tp = num_routed/(ne_tp·dp) × full_expert` — dp reduces it. At `ne_tp=1,dp=8,e_tp=1`: `128/8 = 16` experts' bytes → the 8.2 ms floor.

**Site 1 (token m).** `route.cpp:152-193` `GateUpdate::aggregate_expert` (rank 0): `:172` saves `local_num_token_in_expert[j] = num_token_in_expert[j]` (per-replica), then `:169-184` sums across `running_queue` and writes the GLOBAL sum back into every replica's `num_token_in_expert[j]`. `Route::forward:229-234` sizes each hosted expert's activation tensor with the GLOBAL `num_token_in_expert[expert_idx]`; `linear_impl.cpp` uses that `m` directly for compute (`2mkn`) and activation-memory (`mk+mn`), **no `/dp` anywhere**. `getIdxHigh` hotness (`:240`) is also fed from the GLOBAL vector via `expert_token_list` (`:221-227`).

**a2a reads local (verified).** `MoEScatter`/`MoEGather` (`communication.cpp:188/194/247/253/348/354/413/419`) cost the crossing volume from `local_num_token_in_expert` (and `running_queue[src/dst_dp_idx]->local_num_token_in_expert` for cross-dp shards). Device group = the ExpertFFN `device_list` (stage = `ne_tp·dp`) for scatter/gather; `e_tp` group for the e-tp all-reduce.

**Site 2 (page amortization only).** `model_config.h:189` `expert_groups = devices_per_stage/e_tp` ⇒ `experts_per_device = num_routed·e_tp/devices_per_stage` — feeds `weightReadOpsPerIteration`, which ONLY sets the flash page-read fill-amortization divisor (`layer_impl.h:55-60`); a µs-scale additive term, **not** the weight bytes.

**Optimizer already matches the runtime scheme.** `parallelism_optimizer.cpp:174` `routed = num_routed_expert/devices_per_stage × full_expert` (comment `:171-173`: "the product is `num_routed/devices_per_stage × full_expert_weight`, independent of EP"). `:259` `experts_per_device = num_routed/devices_per_stage` for `e_active`; `:293-296` routed-active latency weight uses `e_active`. `EnumerateCandidates:56-57` `max_e_tp = min(devices_per_stage, num_routed)`. **So there is NO current optimizer/runtime mismatch** — both implement EP-across-dp. The "bug" is a *semantics* choice: the team-lead rules that `dp` must be pure **replication** and `e_tp` the only expert-distribution degree.

---

## 2. The (ep, dp) m-semantics table — heart of the fix

Notation: `B` = total batch, `R = B/dp` = per-replica tokens, top-`k` picks/token. `ne_tp = tp` (attention TP). A device *hosts* some routed experts and computes `m_e` token-rows for each hosted expert `e`.

### 2.1 CURRENT (buggy) semantics — EP-across-DP
| config | hosted experts/device | per-hosted-expert `m_e` | weight bytes/device |
|---|---|---|---|
| ep1, dp1 (`ne_tp=1`) | `num_routed` (128) | `R`-tokens count to `e` (= B, dp1) | 128·full |
| ep1, **dp8** (`ne_tp=1`) | `num_routed/(ne_tp·dp)` = **16** | **GLOBAL**: Σ over ALL dp of tokens→`e` (a2a-pooled) | **16·full** |
| ep8, dp1 (`ne_tp=8`) | `num_routed/(ne_tp/e_tp)` shard, tensor `/e_tp` | global(=local, dp1) | 16·full |

Self-consistent (a2a pools ⇒ global `m` correct) but **`dp` silently buys the EP weight discount** — this is the ruling's target defect.

### 2.2 FIXED semantics — `dp` = pure replication, `e_tp` the ONLY expert-distribution degree
Each dp replica is an independent full model on its `ne_tp` per-stage devices; routed experts are distributed **only across the replica's `ne_tp` devices** (distinct-expert groups `= ne_tp/e_tp`, tensor-split `/e_tp`); **no cross-dp expert movement**; each replica processes only its own `R = B/dp` tokens.

| config | hosted experts/device | per-hosted-expert `m_e` | weight bytes/device |
|---|---|---|---|
| ep1, dp1 (`ne_tp=1`) | 128 | local `R`-count to `e` (= B) | 128·full |
| ep1, **dp8** (`ne_tp=1`) | **128** (all resident/replica) | **LOCAL**: this replica's `R=B/8`-count to `e` | **128·full** ← physical floor restored |
| ep1, dp1, **tp8** (`ne_tp=8`) | `128/8 = 16` (sharded across the 8 tp-peers) | local (= B, dp1) | 16·full |
| **ep8**, tp8, dp1 | `128·8/8 = 128`, tensor `/8` | local | `128·full/8 = 16·full` (honest EP) |
| ep2, tp8, dp2 (16g) | `128·2/8 = 32`, tensor `/2` | local `R=B/2` | `32·full/2 = 16·full` |

**Invariant:** `m_e = local_num_token_in_expert[e]` (replica-local, restricted to hosted experts) for EVERY (ep, dp) — the GLOBAL vector is never the correct `m` under this model. Weight bytes/device `= num_routed·e_tp/ne_tp × full/e_tp = num_routed/ne_tp × full` — depends on `ne_tp` and `e_tp`, **never on `dp`**. **Hard constraint: `e_tp ≤ ne_tp`** (can't tensor-split an expert across more devices than the replica's stage has) — the current sweep's `e_tp ≤ ne_tp·dp` is what let `dp` masquerade as EP.

---

## 3. The complete, self-consistent site list (recommended: §2.2 "Interp-B")

> Hunt H listed only **(S3)** and **(S6)**. The weight-floor collapse needs **(S1)+(S2)** and the optimizer mirror needs **(S4a-d)**. Applying only Hunt H's two would leave the 16-expert weight floor intact (see §6-vii) — smooth+monotone but at the wrong cheap floor.

**Runtime**
- **(S1) Per-replica expert device list.** In `decoder.cpp` (the `MoEDecoder` ctor, `:137-139`) pass a **per-replica** routed-expert device list — the `ne_tp` tp-peers of this rank within its dp replica (i.e. `non_moe_device_list`, already computed at `decoder.cpp:105-106`) — to `ExpertFFN::Create` instead of `stage_device_list`. This single lever cascades correctly: `expert.cpp:42` `parallel_num = ne_tp`; `:51` `num_expert_per_device = num_routed·e_tp/ne_tp`; `:61-62` `expert_offset` within-replica; the MoEScatter/MoEGather/`all_reduce_for_gather` device groups (`expert.cpp:38/110/113`, built from `device_list`) shrink to `ne_tp` (⇒ **no cross-dp a2a**; at `ne_tp=1`, scatter/gather are single-device no-ops). Add an assert `e_tp_dg ≤ ne_tp_dg` in the `ExpertFFN` ctor (mirrors the existing `:45-48` group asserts) so an invalid `e_tp>ne_tp` aborts rather than mis-partitions.
- **(S2) Weight-stream count follows.** No extra edit — weight bytes/device now `= num_routed·e_tp/ne_tp × full/e_tp = num_routed/ne_tp × full` purely from (S1)'s `num_expert_per_device`. (This is the actual 65.66 ms-floor fix.)
- **(S3) Route sizes `m` from LOCAL.** `route.cpp Route::forward:229-234`: size each hosted expert's tensor from `sequences_metadata->local_num_token_in_expert[expert_idx]` (preserved at `:172`), not `num_token_in_expert`. Feed `getIdxHigh`'s `expert_token_list` (`:221-227`) from `local_num_token_in_expert` too (hotness is per-device/per-replica; note `parallel_execution` is dead-hardcoded-false in every preset so this is low runtime impact, but keep it consistent). With (S1) making the a2a local-only, local `m` is now the physically correct owner count. The global-broadcast writes in `aggregate_expert:169-184` are no longer needed for sizing — **keep the local vector; the global vector may be dropped OR retained only if a genuine cross-shard consumer remains (verify: after this fix none does — a2a reads `local_*`, sizing reads `local_*`, hotness reads `local_*`).** If retained, do NOT let it feed Route/hotness.
- **(S6) `weightReadOpsPerIteration` page count.** `model_config.h:189`: `expert_groups = (devices_per_stage/dp)/e_tp = ne_tp/e_tp` ⇒ `experts_per_device = num_routed·e_tp/ne_tp`. Since this fn receives `total_num_devices` (whence `devices_per_stage=ne_tp·dp`), thread `dp` (or `ne_tp`) in: compute `ne_tp = devices_per_stage/dp` and use `expert_groups = ne_tp/e_tp` (clamp ≥1). µs-scale, but keep it in lock-step so page-fill amortization matches the new expert count.

**Optimizer (BUGS-14 "both sides or neither")**
- **(S4a) Routed weight.** `parallelism_optimizer.cpp:174`: `routed = num_routed_expert/ne_tp × full` (replace `devices_per_stage` with `tp` — the optimizer's `tp` IS `ne_tp`). Now weight/device depends on `tp,e_tp` not `dp`. Update the `:171-173` comment (it currently *documents* the buggy invariant).
- **(S4b) `e_active` basis.** `:259` `experts_per_device = num_routed·e_tp/ne_tp` (= `num_routed·e_tp/tp`); and `:260` `assignments` should be **per-replica** (`R = batch_size/dp`, i.e. reuse `batch_size_per_gpu`, NOT the global `batch_size`) since each device only sees its replica's tokens. Recompute `device_share = experts_per_device/num_routed`. Compose with MOE_TAG_FIX_SPEC §7's `/pp` (the `e_active` used for latency is per-replica-microbatch: `R/pp` when `pp_pipelined_timing`).
- **(S4c) Routed-active latency weight.** `:293-296` already multiplies `e_active × full_expert` — with (S4a/b) fixed it now correctly reflects `num_routed/ne_tp`-resident × activated-fraction. Verify the `e_tp` handling (weight-for-latency should be `activated_experts × full/e_tp` streamed).
- **(S4d) `e_tp` sweep bound.** `EnumerateCandidates:56-57`: `max_e_tp = min(ne_tp /*=tp*/, num_routed)` instead of `min(devices_per_stage, num_routed)` — forbid `e_tp>ne_tp` (matches the (S1) assert). This is what makes the optimizer stop enumerating "dp-as-EP" configs and instead surface honest `tp≥ep` ones.

**Capacity note.** With (S4a) the optimizer's `weight_per_gpu` for routed experts rises `dp×` at fixed `tp` — capacity-tight cells may now OOM at dp>1/ep1 where they previously "fit" via the phantom EP discount. That is the corrected physics (pure DP replicates all experts). Confirm no *paper* cell relied on the phantom discount to fit; if one does, the honest winner is an `ep>1` config (verification iv).

---

## 4. Interaction with MOE_TAG_FIX_SPEC (my own fix) — MUST compose

MOE_TAG's cold-at-micro test compares `num_token_in_expert[e]` (full-B) vs `num_token_in_expert_micro[e]`. Under the DP fix, **both bases become REPLICA-LOCAL**:
- Full-B basis → `local_num_token_in_expert[e]` (the per-replica full count, `route.cpp:172`) — NOT the global vector.
- Micro basis → `num_token_in_expert_micro[e]` computed **per-replica** in `update_expert` (MOE_TAG §4.4 already computes it per-BatchedSequence over the first `ceil(R/pp)` tokens) and **NOT globally aggregated**. **DELETE the global-micro aggregation block that MOE_TAG_FIX_SPEC §4.5 added to `GateUpdate::aggregate_expert`** — under the DP fix it would (wrongly) globalize the micro vector while the full basis is local, breaking the comparison. Keep the local per-replica micro fill in `update_expert`.
- `Route::forward` cold-at-micro (MOE_TAG §4.5, the `output->cold_at_micro = …` block): compare `local_num_token_in_expert[expert_idx] > 0 && num_token_in_expert_micro[expert_idx] == 0`.

Net: cold-at-micro becomes "activated by this replica's `R` tokens but not by its first `R/pp`" — the physically correct microbatch statement once experts are replica-local. `pp==1` and dp1 remain no-ops (§5). The MOE_TAG per-op `total_duration·pp/(pp−1)` override and `pp_pipelined_timing` gating are unchanged.

---

## 5. dp1 / llama3 no-op proofs

- **dp1 byte-identity (both semantics coincide).** At `dp=1`: `devices_per_stage = ne_tp·1 = ne_tp`, so (S1) per-replica list == `stage_device_list` (identical device set) ⇒ `num_expert_per_device`, `expert_offset`, a2a groups all unchanged. `aggregate_expert` global sum over a single-replica `running_queue` == the local vector ⇒ (S3) local == global ⇒ Route sizes identically. `weightReadOps` `ne_tp/e_tp == devices_per_stage/e_tp` ⇒ (S6) identical. Optimizer `num_routed/tp == num_routed/devices_per_stage` and `assignments` per-replica == global ⇒ (S4) identical; `max_e_tp = min(tp,·) == min(devices_per_stage,·)` since `devices_per_stage=tp` ⇒ (S4d) identical. **Every dp1 cell byte-identical.** (This is why every prior dp1 audit — Hunt E T9 — saw "no cell moved": the bug is dp>1-only.)
- **llama3 (dense) byte-identity.** `num_routed_expert==0` ⇒ no `ExpertFFN`/`Route` path; (S1)(S3) unreached; `weightReadOps` MoE branch (`:188`) skipped; optimizer routed branch (`:169`) skipped (`mlp_weights` uses the dense `else` at `:196`). **Every llama3 cell byte-identical**, all dp.

---

## 6. Verification battery

- **(i) dp1 byte-identity** — §5 proof; spot-check a maverick dp1 cell `Total:` unchanged pre/post.
- **(ii) llama3 byte-identity** — §5 proof; spot-check one 16g llama3 dp2 cell unchanged.
- **(iii) dp8 expert_ffn smooth + monotone** — re-run ~8 points of the huntD batch ladder at maverick TP1/DP8/EP1: `expert_ffn` must be single-valued (no 7-8 vs 38-44 bimodality) AND monotone-nondecreasing in batch, sitting at the **128-expert weight floor** (≈`dp1`'s per-device 65.66 ms class, NOT 8.2 ms). `moe_dram` bytes unchanged-smooth (they were never the artifact).
- **(iv) PREDICTION TEST** — post-fix auto search maverick/HBF/SHORT/8g: expect an **honest `ep=ne_tp` config (e.g. tp8/ep8)** to win with per-GPU figures near the current ones (the 580→1679.7/GPU growth is physical only with EP-sharded weights, now reachable only via explicit `e_tp`); dp8/ep1 should now be EXPENSIVE (128 experts/device) and lose. **Record whatever actually wins** — if a pp/dp config wins instead, that is data about the true envelope, not a failure.
- **(v) EP>1 regression** — a forced `ep8` (tp8) cell pre/post must change ONLY per §2.2 (owner `m` global→local; weight/device `num_routed/devices_per_stage×full` → `num_routed/ne_tp×full/e_tp`, which at `tp8=devices_per_stage/dp` equals the same `16·full` when `dp=1`, so a dp1 ep8 cell is byte-identical per §5; a dp2 ep8 cell shifts by the defined `m`-locality only).
- **(vi) Bisection-monotonicity** — the per-config max-batch binary search assumes monotone latency-in-batch; confirm the fixed `expert_ffn` curve preserves it (the bimodality previously violated it — a search-correctness bonus of the fix).
- **(vii) Bimodality-driver diagnostic (for the record)** — one probe isolating Site-1 (global-`m` compute/act high-mode) vs the weight floor: run dp8/ep1 with ONLY (S3) route-local applied (module still 16-expert) → expect the bimodality to VANISH but the floor to stay at the **wrong 8.2 ms** (proving S3 fixes the artifact, not the semantic); then add (S1) → floor rises to 65.66 ms class (proving the weight-floor fix lives in the module partition, not in Hunt H's two sites). Archive both CSVs.

---

## 7. Where I disagree with Hunt H's fix direction (escalation — resolve before implementing)

1. **The 2-site fix is insufficient and, applied alone, internally inconsistent.** Route-local `m` under the *unchanged* EP-across-dp module (which still pools tokens cross-dp via the `ne_tp·dp`-wide a2a) would UNDER-count the owner (it processes a2a'd-in tokens from all replicas but would be sized with only one replica's) — a different wrong. Local `m` is correct ONLY together with (S1) making the a2a replica-local. And neither route-local nor `weightReadOps` moves the routed **weight bytes/device**, which is the actual `8.2 ms` collapse floor. So the honest, self-consistent fix is the ~7-site (S1–S6) set in §3, not 2.
2. **One topology decision is required from the team-lead before coding.** The fix hard-constrains **`e_tp ≤ ne_tp`** (experts can only shard across a replica's own stage devices). This forbids configs like `tp1/ep8/dp8` that the current sweep enumerates. If the intended model must allow EP to span the dp devices (a real Megatron "expert parallelism = DP-group" scheme), then **"dp = pure replication" is self-contradictory** for the expert layer, and we instead need an explicit separate `ep_world` degree orthogonal to `dp` — a bigger redesign. **Confirm: is `e_tp ≤ ne_tp` acceptable (recommended, matches the "honest ep8 = tp8" prediction), or must EP span dp?** My spec assumes the former; if the latter, the whole (ep,dp) table (§2.2) and site list (§3) change.
3. **Capacity may tighten at dp>1/ep1** (S4a): pure-DP replicates all experts, so cells that "fit" via the phantom EP discount can now OOM. This is correct physics but will move some auto winners toward `ep>1` (verification iv) — flag any *paper* cell that depended on the phantom fit.
4. **Site-2 (`weightReadOps`) is mis-attributed in the notes.** CAMPAIGN_NOTES says "dp8 low mode 7-8ms ≈ 65.66×16/128 matches site-2 amortization." The `16/128` is the **module weight-partition** (`num_expert_per_device`), NOT `weightReadOps` (which only sets the page-fill divisor, a µs additive). The verification-vii probe pins this on the record.

---

## 8. Do-not-touch / sequencing
- Apply ON TOP of PP_FIX + MOE_TAG + PP_FLAGS, only after the pending fixer gate report sign-off.
- Shared (non-routed) expert is already per-replica (`non_moe_device_list`) — do NOT change it.
- Keep the dp1 and llama3 paths byte-identical (§5); every new read is replica-local and coincides with the old global at dp1.
- BUGS-14 discipline: land (S1-S3,S6) runtime and (S4a-d) optimizer together, never one side alone.
- Record the LATENT stranding bug (CAMPAIGN_NOTES: `batch % dp != 0` strands `dp−1` seqs) in BUGS.md separately — out of scope here, unreachable via `run_experiments`.
- At the end `git diff` should touch only: `decoder.cpp`, `expert.cpp` (assert), `route.cpp`, `model_config.h`, `parallelism_optimizer.cpp` (`EnumerateCandidates` + `EvaluateConfig`), and the MOE_TAG micro-aggregation deletion — plus none of the dense/attention paths.
