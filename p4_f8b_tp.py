import sys
sys.path.insert(0, "/home/arcuser/jeongmin/HBF-LLMSimulator")
import run_experiments as R

# SHORT, vary tp to see whether the score-inclusive ceiling escapes (score /tp)
for tp in [1,2,4,8]:
    dp = 8//tp
    bs = 400*dp if dp>0 else 400
    res = R.run_simulation("llama4_maverick", "HBF+", 8, bs, 1660, 373,
                           tpot_slo=0.1, distribution={"tp":tp,"pp":1,"ep":1},
                           temp_cfg_name="config_p4_f8b_tp%d.yaml" % tp,
                           worker_tag="p4_f8b_tp%d" % tp)
    print(f"SHORT tp={tp} dp={dp} score-incl SRAM ceiling/gpu = {res.get('sram_diag_ceiling')}")
