import sys
sys.path.insert(0, "/home/arcuser/jeongmin/HBF-LLMSimulator")
import run_experiments as R

SRAM = 320 * 1024 * 1024  # per-GPU logic SRAM bytes (HBF+)

# ---- maverick config (from model_config.h) ----
hidden = 5120
head_dim = 128
num_heads = 40
num_kv_heads = 8
intermediate_dim = 16384
expert_intermediate_dim = 8192
num_routed = 128
num_shared = 1
top_k = 1
expert_freq = 2
prec = 2

def peak_per_seq(tp, e_tp, ep_devices_per_stage, total_batch, batch_per_dp):
    # GQA base attention (per batch_per_dp; report per-seq by /batch_per_dp)
    attn = ((hidden) +
            ((3.0*0 + head_dim)*num_heads/tp) +      # q proj
            (2.0*head_dim*num_kv_heads/tp) +          # cur k,v
            (num_heads*head_dim/tp) +                 # ctx out
            (hidden)) * prec
    # ffn dense
    ffn_dense = (2.0*intermediate_dim/tp + hidden) * prec
    # ffn moe
    expert_batch = total_batch * top_k / num_routed
    nrpd = num_routed * e_tp / ep_devices_per_stage
    routed = nrpd * (2.0*(expert_batch*expert_intermediate_dim/e_tp) +
                     expert_batch*expert_intermediate_dim/e_tp +
                     expert_batch*hidden)
    shared = num_shared * (2.0*(batch_per_dp*expert_intermediate_dim/tp) +
                           batch_per_dp*expert_intermediate_dim/tp +
                           batch_per_dp*hidden)
    ffn_moe = (routed + shared) * prec
    # NB routed term scales with total_batch (not batch_per_dp) -> per-seq depends on batch
    ffn = max(ffn_moe, ffn_dense)
    peak = max(attn, ffn)
    return attn, ffn_dense, ffn_moe, peak

print("=== per-seq footprint (batch_per_dp=1 for attn/dense; moe at given batch) ===")
for tp in [1,2,4,8]:
    a,fd,fm,pk = peak_per_seq(tp, e_tp=1, ep_devices_per_stage=8, total_batch=1, batch_per_dp=1)
    print(f"tp={tp} e_tp=1 ep=8dev: attn={a:.0f}B dense={fd:.0f}B moe(b=1)={fm:.0f}B peak={pk:.0f}B -> SRAM_ceil={SRAM/pk:.0f} seq")

print()
print("paper implied: SHORT 854.7 seq/gpu -> per-seq =", SRAM/854.7, "B")
print("paper 327 quote -> per-seq =", SRAM/327, "B")
print("paper MID 745 -> per-seq =", SRAM/745.1, "B")

# ---- analytic configs from the binary ----
for wl, il, ol in [("SHORT",1660,373),("MID",5900,499),("LONG",103500,1100)]:
    print(f"\n=== run_analytic_configs maverick HBF+ 8-GPU {wl} ({il},{ol}) ===")
    res = R.run_analytic_configs("llama4_maverick", "HBF+", 8, il, ol,
                                 tpot_slo=0.1, temp_cfg_name="config_p4_f8b_ac_%s.yaml" % wl.lower())
    print("cap_feasible_at_1:", res["cap_feasible_at_1"], " n_configs:", len(res["configs"]))
    # show configs with best cap_batch
    cfgs = sorted(res["configs"], key=lambda c: -c.get("cap_batch",0))
    for c in cfgs[:12]:
        dp = c.get("dp",1)
        capb = c.get("cap_batch",0)
        per_gpu = capb/8.0
        print(f"  tp={c.get('tp')} pp={c.get('pp')} ep={c.get('ep')} dp={dp} cap_batch={capb} -> cap_per_gpu={per_gpu:.1f} slo_hint={c.get('slo_hint_batch')}")
