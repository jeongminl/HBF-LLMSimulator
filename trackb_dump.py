import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as R

# (model, mem, gpus, Lin, Lout, label)
points = [
    ("llama4_maverick", "HBF+", 1,  1660,  373, "l4_SHORT_1g"),
    ("llama4_maverick", "HBF+", 8,  1660,  373, "l4_SHORT_8g"),
    ("llama4_maverick", "HBF+", 16, 1660,  373, "l4_SHORT_16g"),
    ("llama4_maverick", "HBF+", 8,  5900,  499, "l4_MID_8g"),
    ("llama4_maverick", "HBF+", 16, 5900,  499, "l4_MID_16g"),
    ("llama4_maverick", "HBF+", 8,  103500,1100, "l4_LONG_8g"),
    ("llama3_405B",     "HBF+", 8,  103500,1100, "l3_LONG_8g"),
    ("llama3_405B",     "HBF+", 8,  1660,  373, "l3_SHORT_8g"),
]

for model, mem, g, lin, lout, label in points:
    tag = "config_trackb_%s.yaml" % label
    res = R.run_analytic_configs(model, mem, g, lin, lout, tpot_slo=0.1, temp_cfg_name=tag)
    print("==== %s (gpus=%d ctx=%d) feasible@1=%s" % (label, g, lin+lout, res["cap_feasible_at_1"]))
    cfgs = res["configs"]
    # sort by cap_batch descending
    cfgs = sorted(cfgs, key=lambda c: c.get("cap_batch", 0), reverse=True)
    for c in cfgs:
        print("  tp=%d pp=%d ep=%d dp=%d cap_batch=%d cap_per_gpu=%.1f slo_hint=%s" % (
            c.get("tp",-1), c.get("pp",-1), c.get("ep",-1), c.get("dp",-1),
            c.get("cap_batch",-1),
            c.get("cap_batch",0)/max(1,c.get("dp",1)),
            c.get("slo_hint_batch","?")))
