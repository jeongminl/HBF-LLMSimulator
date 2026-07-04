import sys
sys.path.insert(0, "/home/arcuser/jeongmin/HBF-LLMSimulator")
import run_experiments as R

SRAM = 320 * 1024 * 1024

for wl, il, ol, bs in [("SHORT",1660,373,800),("MID",5900,499,400)]:
    res = R.run_simulation("llama4_maverick", "HBF+", 8, bs, il, ol,
                           tpot_slo=0.1, distribution={"tp":1,"pp":1,"ep":1},
                           temp_cfg_name="config_p4_f8b_fr_%s.yaml" % wl.lower(),
                           worker_tag="p4_f8b_fr_%s" % wl.lower())
    diag = res.get("sram_diag_ceiling")
    print(f"[{wl}] batch={bs} ctx={il+ol} score-inclusive SRAM ceiling/gpu = {diag}")
    if diag:
        print(f"    -> implied per-seq(score-incl) = {SRAM/ (diag*8/ (bs)) if False else SRAM/diag*8/bs:.0f} (rough)")
    print("    keys:", {k:res.get(k) for k in ("max_batch","tps","bound_reason") if k in res})
