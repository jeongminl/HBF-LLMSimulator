import csv

f1 = "data/e1_tp1_b3680/llama4_maverick_synthesis_1660_373_GPU_N1_D8_TP1_DP8_maxbatch3680_maxprocess524288_iter10_skew8_precision_byte2_parallel_execution0_decode.csv"
f2 = "data/e1_tp2_b3680/llama4_maverick_synthesis_1660_373_GPU_N1_D8_TP2_DP4_maxbatch3680_maxprocess524288_iter10_skew8_precision_byte2_parallel_execution0_decode.csv"

cols = ["latency","qkvgen","q_down_proj","kv_down_proj","kr_proj","q_up_proj","qr_proj",
        "kv_up_proj","tr_k_up_proj","v_up_proj","atten_sum","atten_gen","o_proj","ffn",
        "expert_ffn","communication","kv_write","rope","layernorm","residual","lm_head"]

def load(path):
    rows = []
    with open(path, newline='') as fh:
        r = csv.DictReader(fh)
        for row in r:
            rows.append(row)
    return rows

r1 = load(f1)
r2 = load(f2)

print("TP1 rows:", len(r1))
print("TP2 rows:", len(r2))
for row in r1:
    print("TP1", row.get("iter_info"), row.get("type"), {c: row.get(c) for c in cols})
print()
for row in r2:
    print("TP2", row.get("iter_info"), row.get("type"), {c: row.get(c) for c in cols})
