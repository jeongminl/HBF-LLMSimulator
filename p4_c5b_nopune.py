#!/usr/bin/env python3
"""Pass-4 C5 pruning A/B, side B: llama4/HBF+/8/LONG with config pruning DISABLED.
Baseline (pruned) from the 13-cell regression: batch/GPU=169.0, TPS/GPU=1692.07,
tpot=0.099878, dp=4. Run with HBF_DISABLE_CONFIG_PRUNING=1 in the environment."""
import json
import os
import sys

import run_experiments as rx


def main():
    assert os.environ.get("HBF_DISABLE_CONFIG_PRUNING") == "1", "env var not set"
    max_b, tpot, csvf, pec_kv, pec_cap, dp, bound = rx.find_max_batch_size(
        "llama4_maverick", "HBF+", 8, 103500, 1100, tpot_slo=0.1,
        temp_cfg_name="config_p4_c5b.yaml", worker_tag="p4_c5b")
    print(json.dumps({
        "label": "c5b_noprune_l4_HBFp_8_LONG",
        "batch_per_gpu": max_b / 8 if max_b else 0,
        "tps_per_gpu": (max_b / (tpot * 8)) if (max_b and tpot) else 0,
        "tpot": tpot, "dp": dp, "bound": bound,
        "baseline_pruned": {"batch_per_gpu": 169.0, "tps_per_gpu": 1692.07, "dp": 4},
    }))


if __name__ == "__main__":
    sys.exit(main())
