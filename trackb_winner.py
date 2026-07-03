import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as R

points = [
    ("llama4_maverick", "HBF+", 8, 1660,  373,  "l4_SHORT_8g"),
    ("llama4_maverick", "HBF+", 8, 5900,  499,  "l4_MID_8g"),
    ("llama3_405B",     "HBF+", 8, 103500, 1100, "l3_LONG_8g"),
]
for model, mem, g, lin, lout, label in points:
    tag = "config_tbw_%s.yaml" % label
    out = R.find_max_batch_size(model, mem, g, lin, lout, tpot_slo=0.1,
                                temp_cfg_name=tag, worker_tag="trackbw")
    max_b, tpot, csv, pec_kv, pec_cap, dp, bound = out
    print("WIN %s: total_batch=%d per_gpu=%.1f dp=%d tp*pp=%d tpot=%.4f bound=%s" % (
        label, max_b, max_b/float(g), dp, g//max(dp,1), tpot, bound))
