#!/usr/bin/env python3
"""Parse the comm16 CSV with the harness's own breakdown parser."""
import json

import run_experiments as rx

CSV = ("/home/arcuser/jeongmin/HBF-LLMSimulator/data/p4_v_comm16_l4_mid/"
       "llama4_maverick_synthesis_5900_499_GPU_N2_D8_TP2_DP8_EP4_MEMHBM4_maxbatch2432_"
       "maxprocess524288_iter10_skew8_precision_byte2_parallel_execution0_decode.csv")

print(json.dumps(rx.parse_csv_breakdown(CSV), indent=1))
