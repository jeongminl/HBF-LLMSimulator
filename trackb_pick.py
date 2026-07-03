import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as R

# Single optimizer run at a representative batch: which config does it pick,
# and what score-inclusive SRAM ceiling does cluster.cpp's diagnostic print?
# batch chosen near the paper's per-gpu*gpu operating point.
cases = [
    ("llama4_maverick", "HBF+", 8, 1660,  373,  6080, "l4_SHORT_8g_b760x8"),
    ("llama4_maverick", "HBF+", 8, 5900,  499,  5936, "l4_MID_8g_b742x8"),
    ("llama3_405B",     "HBF+", 8, 103500, 1100, 152, "l3_LONG_8g_b19x8"),
]
for model, mem, g, lin, lout, b, label in cases:
    tag = "config_tbp_%s.yaml" % label
    res = R.run_simulation(model, mem, g, b, lin, lout, True, 0.1,
                           temp_cfg_name=tag, worker_tag="trackbp")
    so = res.get("stdout","")
    tp=pp=ep=dp=None
    ceil=None; diagact=None
    for line in so.split("\n"):
        s=line.strip()
        if s.startswith("SRAM_DIAG_CEILING_BATCH_PER_GPU:"): ceil=s.split(":",1)[1].strip()
        if s.startswith("SRAM_DIAG_SCORE_INCLUSIVE_ACT_BYTES:"): diagact=s.split(":",1)[1].strip()
        if "Selected" in line or "selected" in line or "ne_tp" in line or "TP=" in line.upper():
            pass
    print("CASE %s: success=%s dp=%s tpot=%s ceil_per_gpu=%s score_act=%s" % (
        label, res.get("success"), res.get("dp"), res.get("tpot"), ceil, diagact))
    # dump any config-selection line
    for line in so.split("\n"):
        if any(k in line for k in ["ne_tp","e_tp","pp_dg","Config","config chosen","Parallel"]):
            print("   >", line.strip()[:120])
