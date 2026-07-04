import sys
sys.path.insert(0, ".")
import run_experiments as R

SHORT = (1660, 373)

def show(tag, model, mem, gpu, batch, dist):
    res = R.run_simulation(model, mem, gpu, batch, SHORT[0], SHORT[1],
                           optimize_parallelism=False, tpot_slo=0.1,
                           distribution=dist,
                           temp_cfg_name=f"config_p4_f8a_{tag}.yaml",
                           worker_tag=f"p4_f8a_{tag}")
    print(f"=== {tag}: {model} {mem} {gpu}gpu batch={batch} dist={dist} ===")
    print("success:", res.get("success"), "reason:", res.get("reason"))
    print("tpot:", res.get("tpot"))
    print("csv:", res.get("csv_file"))
    bd = R.parse_csv_breakdown(res.get("csv_file"))
    if bd:
        keys = ["qkvgen","atten_sum","atten_gen","o_proj","ffn","expert_ffn",
                "communication","kv_write","layernorm","residual","rope","lm_head"]
        tot = 0.0
        for k in keys:
            v = bd.get(k, 0.0)
            tot += v
            print(f"  {k:16s} {v*1e-3:10.3f} us")
        print(f"  {'SUM':16s} {tot*1e-3:10.3f} us")
    print()

if __name__ == "__main__":
    # 1-GPU HBF, near-SLO batch
    show("hbf1_177", "llama3_405B", "HBF", 1, 177, {"tp":1,"pp":1,"ep":1})
