import sys
sys.path.insert(0, ".")
import run_experiments as R

SHORT = (1660, 373)
KEYS = ["qkvgen","atten_gen","o_proj","ffn","kv_write","layernorm","residual","lm_head"]

def row(tag, model, mem, gpu, batch, dist):
    res = R.run_simulation(model, mem, gpu, batch, SHORT[0], SHORT[1],
                           optimize_parallelism=False, tpot_slo=1e9,
                           distribution=dist,
                           temp_cfg_name=f"config_p4_f8a_{tag}.yaml",
                           worker_tag=f"p4_f8a_{tag}")
    bd = R.parse_csv_breakdown(res.get("csv_file"))
    tpot = res.get("tpot")
    vals = {k: (bd.get(k,0.0)*1e-3 if bd else 0.0) for k in KEYS}
    print(f"{tag:14s} b={batch:4d} tpot={tpot*1e3:8.3f}ms " +
          " ".join(f"{k}={vals[k]:7.2f}" for k in KEYS))
    return vals

if __name__ == "__main__":
    dist = {"tp":1,"pp":1,"ep":1}
    for b in [1, 20, 90, 177, 178, 250]:
        row(f"hbf1_b{b}", "llama3_405B", "HBF", 1, b, dist)
