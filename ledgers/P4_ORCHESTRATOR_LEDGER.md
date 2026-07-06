# Pass-4 findings ledger (orchestrator working notes, 2026-07-04)

Raw working ledger of the fourth-pass bug hunt, kept as-written during the hunt (finder reports
→ refuter verdicts → verification runs → convergence calls, in arrival order). The polished
records live in CHANGES.md items 51-57, FINDINGS_REGISTER.md "Fourth-pass register", and
PAPER_INCONSISTENCIES.md's fourth-pass notes; this file preserves the orchestrator's
adjudication trail (including caveats raised against refuter arithmetic and the Fig-6 PDF
adjudication).

## PHASE 4 — CONVERGENCE LEDGER (blind findings vs prior docs)

| Item | Blind verdict (pass 4) | Prior docs | Convergence |
|---|---|---|---|
| U5 | F8a: no bug; weight-stream floor 72.1ms measured = 72.3ms first-principles; 10/11 cells exceed anchor, the 1 miss = the paper's "in most cases" hedge; sim matches paper's OWN 1-GPU bars ~1% | Same mechanism (weight-reread SLO floor, physically correct); U5 reclassified matching 2nd pass | CONVERGED + sharpened (claim-vs-own-figure ledger is new) |
| U7 | F8b: paper-internal inconsistency; 855-flat-✕ = score-inclusive SHORT ceiling; "327" = SAME accounting at MID ctx; breaks paper's own unmarked 745 MID bar; sim gate correct; no fix | Third-pass FINAL: identical disposition incl. 755.6 diag and 327.7=score-inclusive-MID | CONVERGED + sharpened (855 & 327 unified as ONE accounting at TWO contexts) |
| Residual-1 | F8c: capacity-bound; DP-pure = 460.00/GPU EXACT; TP=2 argmax legit per §III; no accounting defect; GiB corroborated via 460 (decimal GB → ~428) | Third-pass: same (3680=460/GPU, TP=2 KV-headroom win) | CONVERGED + NEW GiB corroboration |
| Deferred pair (score+MFU) | F5/F4 mechanisms confirmed exact; R1: paper never mentions FlashAttention, GQA symmetric non-support (flag structurally MLA-only) → keeping score bytes arguably paper-conformant; R3: no structural cancellation (score inert on HBF/HBF+, live only on HBM4 sum path); V-MFU: MFU bites SHORT +33.6%@0.5, MID +8% (m = per-DP-replica tokens) | Docs: "both-or-neither pair, cancels, deferred" | PARTIAL DISAGREEMENT: pair framing WEAKENED — R3+R1 show they are largely independent; docs' "cancellation" holds only loosely at HBM4 SHORT/MID cells; keep-deferred outcome UNCHANGED but rationale corrected: score-charging is paper-conformant (not just 'cancels'), MFU unanchored |
| C5/I5 pruning | C5-B empirical: 0-pruned run = identical winner at worst-case LONG cell | Docs: C5 partial, prune-risk unverified | RESOLVED empirically (clean) |
| GiB (F3) | Undecidable from Table I (units cancel); binary physically right; no formula mixes bases; F8c's 460-exact corroboration | Docs: "decimal-GB ruled out (wrong direction)" | CONVERGED, stronger evidence |
| Energy/ramulator (F1/F2) | Decorative end-to-end; use_ramulator OFF in all sweeps; counts from run_ideal; nothing reported depends on it | Not previously audited (fresh ground) | NEW (doc-notes) |
| C8 latencies | F7: invented constants confirmed, ~0.3% effect | Docs: C8 same | CONVERGED |
| CONV write-BW | F7: plane-count derivation (16/25) reconstructs Table I exactly | Anchor sheet: ambiguity #5 unresolved | RESOLVED (closes ambiguity) |
| I2 (1-token KV append) | F5 low-conf observation, matches paper's definition | Docs: I2 same | CONVERGED |
| I3 (Rubin FLOPS) | F4-6 same | Docs: I3 | CONVERGED |
| L1 (iRoPE cap variants) | F4/F5 note mixed_impl lacks cap — same subject | Docs: L1 | CONVERGED (latent) |
| L4 (mixed memory terms) | F5-3 same subject | Docs: L4 | CONVERGED (latent) |

NEW findings with no prior-doc match (pass-4 yield):
- F7-1 MoE decode a2a link composition (FIXED — run-verified 19.1% vs ~4% comm share @16-GPU)
- F6-1 CSV filename EP/mem clobber (FIXED)
- F4-3 optimizer shared-expert FLOPs omission (FIXED, prune-safe hygiene)
- F4-1 MFU m-basis miskeying (dormant; DOC-NOTE — fix requires modeling decision, zero live effect)
- F6-2 Fig-6 normalization/scale anomaly (DOC-NOTE; paper's own baseline plots at 0.84≠1.0, unexplainable from caption; sim convention numerically equivalent to caption given HBM4 SLO-invariance)
- F1-3/F2-2 device.cpp pCH ==1 typo (FIXED, decorative path)
- F3-1 OOM GiB mislabel (FIXED, cosmetic)
- F7-2 node.cpp dead misleading init (COMMENTED)
- F6-3 HBM-capacity→"flash" bucket label (DOC-NOTE)
- F1-1/F1-2 power.h ×8-vs-X4 comment + uncited mac_energy (DOC-NOTES)
- F2-1 ramulator preset-blind yaml landmine if ever enabled (DOC-NOTE)
- F4-4 optimizer KV-hide basis drift (DOC-NOTE; C5-B shows no empirical prune flip)
- F5-4 GQA prefill sum kernel also non-flash (latent, DOC-NOTE)
- F1-5 StatusBoard missing initializers (latent, DOC-NOTE)
- Dead code: data_object.cpp, getDramEnergyForLoad (DOC-NOTES)

## Status (final)
- All phases complete: 10 finder tracks reported, 6 refuters reported, verification waves run,
  fixes 51-57 applied, post-fix 13-cell regression GREEN (≤8-GPU cells bit-identical;
  l4_HBFp_16_LONG 174→175/GPU; l4_HBFp_8_SHORT completed at 2374/GPU slo — the U7 cell).
- Committed as ef02997 on main. Full-sweep regeneration of experiment_results.md pending
  (user-triggered phase).

## F1 — energy/power path (REPORTED)
VERDICT: entire DRAM/MAC energy path decorative — no paper-replication output depends on it
(Fig5 = time columns; Fig7 PEC = pec_kv_bytes/pec_capacity markers; paper has no energy figure).
CORRECTION vs exploration: FC/Attn/MoE energy columns ARE assigned (cluster.cpp:1003-1008,
1263-1268) — written to CSV, never read back. NOT dead zeros.
- F1-1 (M/L): power.h:37-44 all_* multiplier ×8/×32 vs comments "X4"/"X16" — 2× self-inconsistency, uncited. Doc-note.
- F1-2 (L): power.h:35 kMAC=0.46/2/1000 nJ, uncited /2; depends on flops convention (2·MNK vs MNK). Doc-note.
- F1-3 (L): device.cpp:225 rw_cmd_to_pCH_1 uses %num_channel==1 where line 224 uses ==0 — copy-paste typo, ±1 access, feeds energy only. Trivial fix candidate (decorative path).
- F1-4 (L): cluster.cpp:916-1008 FC/Attn/comm energy = device0 × num_total_device — wrong under PP. Decorative.
- F1-5 (L): status.h:86-102 StatusBoard energy fields lack initializers — latent UB smell, currently safe.
- Checked-correct: timeboard deltas, opb guard, per-iteration scope, count×constant math, gpuEnergy provenance, initializeDRAM scaling, getTotalEnergy.
- Dead code: getDramEnergyForLoad + *_energy_load fields — deletion candidate.
DISPOSITION (prelim): doc-notes; F1-3 trivial typo fix OK (grounded, no paper impact); no refuter needed beyond F2 cross-check of decorative claim.

## F2 — ramulator2 integration (REPORTED)
VERDICT: ramulator NEVER RUNS in paper sweeps — config.yaml use_ramulator: off + all presets
default false. Timing = analytic max(compute, bytes/BW) everywhere; energy counts come from
run_ideal (hand-computed), and no reported figure reads them. STRONGER than F1's claim; the two
independently agree the path is inert. Cross-validation: F1's "decorative" CONFIRMED by F2.
- F2-1 (H, latent): device.cpp:40-90 ramulator yaml chosen by gpu_gen only — always HBM3/HBM3E;
  HBF/CONV presets indistinguishable to ramulator. If use_ramulator ever flipped on for flash,
  tpot collapses to HBM3E behavior. Doc-note landmine, unreached.
- F2-2 (H bug, immaterial): device.cpp:224-225 same typo as F1-3 (CONVERGENT, independent).
  ≤1 command off, energy-only, unreported. Trivial fix candidate.
- F2-3 (info): DRAMInterface constructed even when off — one-time cost, negligible.
- data_object.cpp: DEAD (confirmed). mmap addresses wasted-but-harmless in sweeps.
DISPOSITION (prelim): no fixes required for paper numbers; F2-2/F1-3 typo fix trivial+grounded;
F2-1 doc-note. No refuter needed (two independent traces agree).

## F3 — GB vs GiB units (REPORTED)
VERDICT: undecidable from the paper — Table I totals are linear combos of 36 and 512 so the
unit cancels; exact in BOTH conventions. Binary physically defensible (DRAM power-of-2 dies).
No formula mixes 1e9 and 2^30. Gates are bytes-vs-bytes everywhere (4 consumers traced).
- F3-1 (H): footprint.h:77-78,98-99 + cluster.cpp:340-341 OOM display divides binary bytes by
  1e9, labels "GB" — contradicts code's own comment. COSMETIC fix candidate (no figure impact;
  classify_failure matches substrings not numbers).
- F3-2/F3-3 (M): if paper meant decimal, PEC −7.37%, SRAM-bound HBF+ bars ~−7%. NOT tunable —
  would be calibration. Documented ambiguity.
- F3-4 (H): 3.13MiB staging inert (diagnostic-only consumer).
- Checked-correct: all 5 bandwidth pairs match Table I exactly (decimal); per-stack derivations
  consistent; SRAM 320MiB mapping matches paper "320 MB per GPU".
DISPOSITION (prelim): F3-1 cosmetic fix OK; rest = documented ambiguity, NO constant changes.
Refuter not needed (finding is a non-finding; the load-bearing claim is "no bug", conservative).

## F5 — FlashAttention score-traffic (REPORTED)
- F5-1 (H exists/M magnitude): AttentionGenExecutionGPU (attention_gen_impl.cpp:75,159) has NO
  use_flash_attention branch — charges S=QK^T write + P read-back (num_heads·n each way) that
  tiled FlashAttention keeps in SRAM. MLA path DOES honor the flag (:594) — internal kernel
  inconsistency; paper's own models silently run non-flash byte accounting. Overcharge:
  llama3 ~11.1% of attention memory (~6-8% tpot at Fig-5 MID/LONG 8-GPU), llama4 ~3.76%
  (~2.5% tpot). Fixing RAISES HBM4 TPS reference → shrinks HBF/HBF+ advantage in Figs 4/5.
  = memory half of the deferred pair. NOT to be applied alone (pairing = my call).
- F5-2 (M): on HBF preset score bytes priced on 1.6TB/s reserved HBM tier (7× slower/byte than
  flash KV) → amplified overcharge on HBF Fig-3/4 bars; HBF+ act_time=0 → unaffected (correct).
- F5-3 (H/latent): mixed_impl omits score-write AND output-write — unreconciled physics vs Gen
  (subject L4 territory).
- F5-4 (H/latent-low): GQA prefill sum kernel also non-flash (score write+read charged).
- Obs (L): per-step 1-token KV append omitted — matches paper's "KV Write = newly admitted
  queries" definition (prior subject I2). Flag only.
- Checked-correct: softmax zero-memory ✓, no Q re-read ✓, KV-read volume/tier ✓, iRoPE caps ✓,
  page-latency double-buffering ✓, KV-write overlap = paper footnote 2 ✓, KV-write volume ✓.
DISPOSITION (prelim): F5-1+F5-2 = the deferred-pair memory half, now with kernel-inconsistency
evidence (MLA honors flag, GQA doesn't → arguably a plain bug, not a modeling choice!). Needs
F4's compute-half report + refuters (3, per plan) + pairing decision. F5-3/F5-4 latent doc-notes.

## F4 — compute/roofline + flops (REPORTED)
- F4-1 (H, NEW structural): hardware_config.h:224-227 effectiveMFU keyed on per-op
  m=num_process_token — =1 for every decode-attention op vs =batch for FFN GEMMs. Any nonzero
  mfu_m_half derates attention ~(batch/m_half)× harder than FFN → the MFU knob CANNOT represent
  uniform GEMM efficiency as-is. Blocks any future resolution of the deferred pair until fixed.
- F4-2 (H): MFU≡1.0 default = 100%-peak GEMMs; deferred-pair compute half; paper prints no
  efficiency ⇒ any value would be fitted. EVIDENCE only.
- F4-3 (M, NEW): parallelism_optimizer.cpp:159-162,388-390 omits shared-expert FLOPs from MoE
  compute — ~50% per-token under-count for maverick (top_k=1 + 1 shared) → optimizer may pick
  memory-bound when live is compute-bound → prune risk at high-batch HBF+ cells. VERIFY:
  OptValidation divergence / HBF_DISABLE_CONFIG_PRUNING A/B on llama4 HBF+ SHORT.
- F4-4 (L, NEW): optimizer KV-write hiding budget basis = QKVO-projection flops vs sim's
  score+context compute — comment claims exact match, actually differs; LONG cells conservative.
- F4-5 (L, latent): flash-MLA decode compute lacks effectiveMFU (DeepSeek only).
- F4-6 (evidence): Rubin 8750 TF unanchored (= prior I3); memory_bandwidth=22TB/s overwritten by
  preset (confirmed benign).
- Checked-correct: GEMM 2mkn ✓, GQA scoring/context per-head aggregation ✓, gen≡sum(m=1) ✓,
  iRoPE cap applied to FLOPS ✓ (mixed_impl missing cap = L1 territory), softmax 7mn ✓, roofline
  composition no double-count ✓, optimizer dense-FFN ✓, MFU no-op at defaults ✓.
DISPOSITION (prelim): F4-1 fix candidate on correctness grounds (knob semantics broken)
independent of pair decision — but zero effect at current defaults; F4-3 needs verification run;
F4-4 doc-note or small fix; pair itself stays deferred unless paired physical fix approved.

## F7 — LmHead/node/CONV presets (REPORTED)
- F7-1 (M, NEW): communication.cpp:177-187,370-380,441-449 (+optimizer mirror :572-576) MoE
  scatter/gather DECODE branch at num_node>1 charges (intra+inter)/IB — intra-node bytes on the
  100 GB/s link, serialized; prefill branch correctly does max(intra/NVLink, inter/IB).
  Up to ~2× comm overcharge, 16-GPU llama4 only. Contradicts paper Assumption 4.
  VERIFY: 16-GPU llama4 cell, decode comm stat, A/B max-composition. llama3/≤8-GPU unchanged.
- F7-2 (H): node.cpp:7-8 node_ict_* init'd to intra-node values, never read — dead/misleading.
- F7-3 (L): optimizer PP-hop link selection coarse for pp>2 straddling nodes — ranking-only.
- Checked-correct: ALL 5 presets match Table I exactly; CONV write-BW plane-count derivation
  (16/25 planes) CONFIRMED correct — closes anchor-sheet ambiguity #5; interconnect constants
  match paper (latencies invented = prior C8, flagged in code, ~0.3% tpot); 16-GPU node split ✓;
  AllReduce node-aware ✓ (2⌈log2 N⌉ + 2(N-1)/N); lm_head vocab-parallel + negligible ✓;
  embedding zero-read correct ✓; PipelineStage sum-of-stages consistent both sides ✓; router not
  double-counted ✓; config plumbing ✓ (MLA flags auto-off via q_lora_rank guard).
- GiB observation consistent with F3 (ambiguity, not bug).
DISPOSITION (prelim): F7-1 → refuter + verification run (16-GPU llama4). F7-2 trivial cleanup.
F7-3 doc-note.

## F6 — harness mechanics (REPORTED)
- F6-1 (M, NEW real latent bug): eval/test.cpp:747-774 CSV filename omits EP (and mem_type) →
  MoE EP-variant probe at same (TP,PP,DP,batch) in same worker dir overwrites winner's CSV →
  Fig-5 breakdown parsed from WRONG config (parse deferred to end of main()). llama4 Fig-5 cells
  only; dense llama3 immune. FIX: add _EP{e_tp_dg} (+mem?) to filename. VERIFY: list data/w*/
  collisions under disabled pruning.
- F6-2 (L-M, adjudicate): run_experiments.py:1377-1396 Fig-6 normalizes each SLO to its own
  HBM4/8-GPU cell → HBM4 rows structurally 1.00× vs paper's 0.64-0.90 readings (paper likely
  normalizes to fixed 0.1s baseline). Systematic Fig-6 error contributor. Convention fix =
  paper-methodology alignment, NOT calibration. Check paper Fig-6 caption text first.
- F6-3 (H, cosmetic): run_experiments.py:471 "HBM capacity exceeded"→"flash" label; harmless
  (hatching only consults sram for HBF+/CONV+).
- Checked-correct: classify_failure covers ALL C++ emission sites + stream capture ✓; compute_pec
  end-to-end matches paper §IV ✓ (amortized identity exact); breakdown mapping complete, sums to
  1, kv_write unhidden-only no double-count ✓; bisection monotone-valid, winner re-verified real
  sim ✓; pruning direction sound given est≤measured invariant ✓; compare_error_rates joins ✓;
  templating ✓.
DISPOSITION (prelim): F6-1 fix candidate (grounded, mechanical); F6-2 needs paper-caption
adjudication then likely fix; F6-3 one-liner. → R6 refuter launched.

## F8a — re-blind U5 (REPORTED)
VERDICT: NO BUG, NO PAPER SELF-CONTRADICTION. Blind decomposition: batch-invariant
weight-streaming floor 72.1ms measured ≈ 810GB/11.2TB/s = 72.3ms first-principles (72% of SLO
budget at 1-GPU HBF; weights live on flash only per paper data placement, HBM stack =
intermediates only). Batch-scaled KV/attention fills the rest → max batch exactly 177 (boundary
probed 177 pass / 178 fail). Sim matches paper's OWN Fig-3 bars to ~1% (177 vs 176.2; 197.9 vs
195.5). Paper claim ledger: 10/11 comparable 1-GPU-vs-8-GPU-HBM4 cells exceed; the single miss
IS llama3/SHORT/HBF = the paper's "in most cases" hedge. CONVERGES with prior U5 disposition
(weight-reread SLO floor, physically correct), sharper: claim fully consistent with paper's own
figure. No fix.

## F8b — re-blind U7 (REPORTED)
VERDICT: paper-internal inconsistency, NOT a sim bug — CONVERGES with prior disposition.
NEW SHARPENING: paper's "854.7 flat ✕ SHORT bar" and its text "327 seq/GPU" are ONE
score-inclusive accounting at TWO contexts (SHORT ctx≈2033 → 755.6/GPU diag ceiling ≈ 855 bar;
MID ctx≈6399 → ~294-327 ≈ the text quote) — but the paper's own MID bar is 745 UNMARKED, which
that accounting caps at ~327 → no single accounting fits the paper's own bars. Sim gate is
context-independent (score routed through 3.13MB staging per FlashAttention semantics), measured
SRAM ceiling 2824/GPU (tp1/ep1/dp8), SHORT/MID identical caps (context-independence confirmed by
run). Score-inclusive ceiling ~flat across tp (755→651) → paper's flat ✕ not a tp artifact.
RECOMMENDATION: keep current gate, no code change. (Disclosed: footprint.h's own comments
reference the U7 diagnostic — partial context leak via allowed code comments; evidence is
independent measurement, treat convergence with that caveat.)

## F8c — re-blind Residual-1 (REPORTED)
VERDICT: legitimate divergence, paper-tool restriction — CONVERGES with prior disposition.
TP scan (all capacity-bound, tpot 24-33ms ≪ SLO): tp1/dp8=460.00/GPU EXACTLY (=paper anchor,
19046 TPS/GPU vs paper 19000 = +0.24%); tp2/dp4=488.5 & 19280 TPS = argmax (winner). Mechanism:
routed-expert weights TP-independent; TP shards only non-expert weights (116.4→106.0 GiB)
freeing KV headroom; TP=2 batch +6.2% beats tpot +4.9%. TP comm correctly signed (matched-batch
deficit −0.7%). NO capacity-accounting defect (recorded==analytic weights; KV sharding correct;
both configs hit 288-GiB gate cleanly).
NEW CORROBORATION (settles F3 too): with 288 DECIMAL GB, pure-DP → ~428 ≠ 460; only GiB + pure-DP
reproduces 460 exactly ⇒ paper's tool used binary GiB — code convention VINDICATED by the
paper's own anchor. Cosmetics noted: "flash" bucket label for HBM cap; OOM msg "weight+kv"
includes activation term.

## Verification wave A (RESULTS)
- V-MFU discriminator (forced llama3/HBM4/8 tp8/dp1, winner batches): mfu_max=0.5 →
  SHORT tpot 54.16→72.33ms (+33.6%); MID 33.14→35.79ms (+8.0%). CONFIRMS my caveat: GEMM m =
  per-DP-replica tokens; MFU bites hard at SHORT, modestly at MID. R3's "inert at MID/LONG"
  overstated for MID; its structural claims (attention untouched by MFU; score-bytes zero on
  HBF+) unaffected. Pair adjudication: MFU value remains unanchored → still no-calibration.
- V-comm16 (F7-1): llama4/HBM4/16-GPU MID batch 2432, optimizer winner TP2/DP8: communication
  4.80ms of 25.20ms step = 19.1% comm share vs paper Fig-5 llama4 16-GPU ~3.7-3.9% → ~5×
  overcharge CONFIRMED by run. F7-1 fix will materially improve 16-GPU llama4 cells.
- V-classify: forced 60000-batch llama4/HBF+/8 SHORT → OOM/Crash classified "sram" ✓ correct.
- V-C5 (pruning A/B on l4/HBF+/8/LONG): baseline (pruned) 169/GPU tps 1692 dp4; no-pruning
  (30 configs, 0 pruned) → IDENTICAL winner tp2/dp4 batch 1352, tps 1692.07. Pruning invariant
  empirically clean at the worst-case LONG cell → F4-4/C5 prune-risk does not materialize;
  F4-4 stays doc-note.

## POST-FIX VERIFICATION (fixed binary, rebuilt)
- MFU cells (8-GPU llama3): tpot BIT-FOR-BIT identical pre/post → ≤8-GPU no-op guarantee of the
  comm fix holds; optimizer-side fixes (2,3) did not shift 8-GPU forced-run behavior.
- comm16 (16-GPU llama4 MID, winner TP2/EP4/DP8): tpot 25.20→22.32ms (−11.4%); communication
  4.80→1.95ms; comm share 19.1%→8.7% (paper ~3.9%). Fix direction+magnitude verified. atten_gen
  unchanged (14.93ms) ✓.
- FIX 4 verified: new CSV name carries _EP4_MEMHBM4 → clobber class eliminated.
- sram_classify still correctly classified.
- Post-fix 13-cell regression: RUNNING (expect ≤8-GPU cells identical; l4_HBFp_16_LONG will
  legitimately RISE — SLO-bound cell, comm cheaper).

## Refuter verdicts
- R4 on F7-1: SURVIVES all 4 angles. 4 sites (adds MoEScatter receive :247-256). Decode branch
  gates on GLOBAL num_node (not per-communication like AllReduce :63 / PipelineStage :538);
  serializes intra+inter on IB with no NIC-gateway model anywhere; reachable in every 16-GPU
  llama4 cell (num_node=2 hard-coded, winners have tp<16); magnitude ~2-4pp tpot at 16-GPU
  llama4 (paper Fig-5 comm 6.8%/3.7% MID/LONG). → CONFIRMED pending verification run.
  FIX SHAPE: per-communication node gating + max(intra/NVLink, inter/IB) composition, decode
  branch, all 4 sites + optimizer mirror :572-576 together.
- R1 on F5-1/F5-2: arithmetic+physics SURVIVE (11.11%=G/(G+hd) re-derived exact; decode
  memory-bound so bytes exposed; FlashAttention O(tile) claim correct — repo's own MLA tiling
  proves in-repo understanding). "Bug/inconsistency" framing REFUTED: use_flash_attention is
  STRUCTURALLY MLA-only (GQA SelfAttentionGen/Sum classes lack the member entirely; MLA-Gen uses
  use_flash_mla, MLA-Sum uses use_flash_attention — symmetric GQA non-support, not selective
  breakage). Paper NEVER mentions FlashAttention; paper's "chunked attention" extension = HBF
  SRAM staging (different concept). Paper's tool (LLMSimulator lineage) plausibly charges score
  bytes too → keeping them arguably paper-CONFORMANT. → F5-1 = documented scope gap; leans
  KEEP-DEFERRED unless R2/R3 flip the picture.
- R2 on F5-1/F5-2 (Claim A): SURVIVES mechanically (dispatch, byte flow, tier routing all
  verified exact). MAGNITUDE CAVEAT: on HBF act_time/kv_read_time = (G/hd)×(flash/hbm BW) =
  0.875 (llama3) / 0.27 (llama4) → score bytes usually HIDDEN under max(kv,act) in steady-state
  decode → F5-2's "amplified on HBF" largely neutralized (llama3 near crossover though). On the
  plain HBM4 path bytes are additive (sum, not max) → F5-1's 11.1%/3.76% of attention memory
  stands THERE. Net: conservative vs naive kernel (softmax r/w not charged).
- R2 on F4-1 (Claim B): SURVIVES — attention call sites contradict hardware_config.h's own doc
  comment ("M = batch*tokens for that specific op"; per-seq loop passes m=1, batch never folded).
  But DORMANT: mfu_m_half=0 default, config.yaml sets neither key, no driver passes non-None →
  zero effect on any current output. Latent fix candidate (align call sites w/ documented spec)
  OR doc-note; whether M=batch is "right physics" is modeling-philosophy (FlashDecoding-style
  batched kernels support M=batch keying).
- R3 on the pair (cancellation): REFUTED as structural cancellation. Key proofs: (i) decode
  attention memory-bound 48×(l3)/142×(l4) → MFU cannot move attention timing; (ii) score bytes
  IDENTICALLY ZERO on HBF+/CONV+ (num_hbm_stacks=0 guard → act_time≡0) and INERT on HBF
  (act/kv = 7G/hd = 0.875/0.273 < 1, loses the max) → score-fix is a no-op on all flash presets;
  only the plain-HBM4 sum path is live; (iii) MFU affects only projection/FFN GEMMs past
  intensity ~684·MFU flops/byte — Fig-5 (MID/LONG) cells stay memory-bound → B inert at every
  paper-verifiable cell. Direction map: fix-A-alone → HBM4 cells strictly faster (worsens the
  known 4-6%-fast drift); fix-B-alone → ~nothing at MID/LONG. Recommendation: decouple A and B.
  MY CAVEAT (orchestrator): R3's threshold arithmetic used PER-GPU batch as GEMM m; correct m =
  per-DP-replica tokens (TP shards see the full replica batch). At llama3 SHORT TP=8/DP=1
  m≈1565 (>684 ⇒ compute-bound at MFU=1); llama4 SHORT DP=4 m≈977. So MFU DOES bite at SHORT
  cells — consistent with docs' "SHORT/MID reads fast" being MFU-exposed at SHORT. MID m≈496 and
  LONG memory-bound → R3's Fig-5 conclusion stands. Weigh in Phase 4.
- R6 on F6-1: SURVIVES, strengthened — CSV write is unconditional truncate (cluster.cpp:538-573)
  before SLO check; probe batches EP-invariant (k_cap probe :570-577); search continues past
  winner (:632-654); deferred read at :1331. BONUS: mem_type also absent from filename + PID
  reuse across cells (worker_tag=w{pid}, :689) → sequential cross-cell collisions possible too.
  Only Fig-5 breakdown columns at risk (tpot/batch from stdout, safe); communication column most
  EP-sensitive. FIX: add EP (and mem preset) to CSV filename. CONFIRMED fix candidate.
- R6 on F6-2: SURVIVES — paper caption verbatim: "All values are normalized to the 8-GPU HBM4
  for each workload" (single fixed baseline; same phrasing for Figs 3/4). Sim renormalizes per
  SLO → HBM4 ≡ 1.00 structurally; paper HBM4 rows read ~0.84 TPS / 0.64 batch.
  ORCHESTRATOR ADJUDICATION (viewed actual Fig 6 in PDF, p.4): pixel readings are INTERNALLY
  consistent (HBF+/HBM4 ratio 1.090/0.840=1.298 at l3 8-GPU 0.1s; l4 16-GPU offline
  1.131/0.990=+14.2% ≈ paper text "+14.8%") but the paper's own 8-GPU HBM4 baseline plots at
  ~0.84 ≠ 1.0 under EVERY reading of its own caption → paper's absolute scale unexplained
  (possible plot-scale artifact). Since HBM4 is SLO-invariant, our per-SLO convention ≡ the
  caption's fixed-per-workload baseline numerically → changing the writer alters nothing real.
  DISPOSITION: doc-note, NO code fix; Fig-6 comparison error rates carry a structural ~16-19%
  (TPS) / larger (batch, offset-dominated near-zero bars) component from the paper's scale —
  flag when presenting Phase-7 error figures.
- R5 on F4-3: omission REAL (2× per-token MoE compute undercount, maverick; no shared-expert
  compute term anywhere in optimizer — grep-verified; live side confirmed charging full batch)
  but IMPACT REFUTED: prune-safe (underestimating latency preserves the seed_tps upper bound;
  prune condition seed_tps<=best.real_tps), Optimize() bypassed in the sweep (forced
  distribution; run_analytic_sweep dead code; batch=1 fallback negligible). → hygiene fix:
  add shared-expert flops at :161-162 to mirror expert.cpp:202-206. No result changes expected.
