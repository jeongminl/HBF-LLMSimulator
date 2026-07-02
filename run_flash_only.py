import os
import sys
import yaml
import subprocess
import glob
import time

# NOTE: this is a fast 8-GPU SMOKE CHECK, not the paper-comparison harness. It
# uses the paper's canonical workload token lengths (matching run_experiments.py's
# WORKLOADS, the single source of truth for any Son et al. IEEE CAL 2026
# comparison), but only sweeps 8 GPUs / one SLO -- it cannot reproduce anchors
# that need other GPU counts or the full SLO sweep (see PAPER_INCONSISTENCIES.md).
# Use run_experiments.py for anything compared against the paper.
models = ["llama3_405B", "llama4_maverick"]
workloads = {
    "SHORT": (1660, 373),
    "MID": (5900, 499),
    "LONG": (103500, 1100),
}
mem_types = ["HBF", "HBF+", "CONV", "CONV+"]
gpus = [8]

def run_simulation(model, mem_type, num_device, batch_size, input_len, output_len, optimize_parallelism=True, tpot_slo=0.1):
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    cfg["model"]["model_name"] = model
    cfg["system"]["memory_type"] = mem_type
    
    if num_device == 16:
        cfg["system"]["num_node"] = 2
        cfg["system"]["num_device"] = 8
    else:
        cfg["system"]["num_node"] = 1
        cfg["system"]["num_device"] = num_device
        
    cfg["serving"]["max_batch_size"] = batch_size
    cfg["simulation"]["input_len"] = input_len
    cfg["simulation"]["output_len"] = output_len
    cfg["simulation"]["iter"] = 10
    cfg["simulation"]["injection_rate"] = 0
    cfg["system"]["optimize_parallelism"] = optimize_parallelism
    cfg["system"]["tpot_slo"] = tpot_slo
    cfg["simulation"]["exit_out_of_memory"] = True

    temp_cfg_path = os.path.abspath("build/config_temp.yaml")
    with open(temp_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    cmd = ["./run", "config_temp.yaml"]
    try:
        res = subprocess.run(cmd, cwd="build", capture_output=True, text=True, timeout=1800)
        stdout = res.stdout + "\n" + res.stderr

        fail_markers = ["Out of Memory", "Activations exceed",
                        "Flash capacity exceeded", "capacity exceeded",
                        "resulted in OOM"]
        if res.returncode != 0 or any(m in stdout for m in fail_markers):
            return {"success": False, "reason": "OOM/Crash", "stdout": stdout}
            
        # cluster.cpp prints multiple "Total: " lines: two memory-report lines shaped
        # "Total: <float>GB" (cluster.cpp:186,343) and the real total-time line shaped
        # "Total: <int>" with no suffix (cluster.cpp:484, scheduler->total_time in ns).
        # Explicitly exclude the "GB"-suffixed lines instead of relying on float()
        # to throw on them, so a future format change can't silently mis-parse.
        total_time_ns = None
        for line in stdout.split("\n"):
            if line.startswith("Total: ") and not line.rstrip().endswith("GB"):
                try:
                    total_time_ns = float(line[len("Total: "):].strip())
                except ValueError:
                    pass
        if total_time_ns is None:
            return {"success": False, "reason": "No total time printed", "stdout": stdout}
            
        tpot = total_time_ns / (10.0 * 1e9)
        if tpot > tpot_slo:
            return {"success": False, "reason": f"SLO Violated ({tpot:.3f}s > {tpot_slo:.3f}s)", "tpot": tpot, "stdout": stdout}
            
        return {"success": True, "tpot": tpot, "stdout": stdout}
    except Exception as e:
        return {"success": False, "reason": str(e), "stdout": ""}

def find_max_batch_size(model, mem_type, num_device, input_len, output_len, tpot_slo=0.1):
    # Phase 1: Exponential Search
    res = run_simulation(model, mem_type, num_device, 1, input_len, output_len, True, tpot_slo)
    if not res["success"]:
        return 0, 0.0
        
    b_success = 1
    tpot_success = res["tpot"]
    b = 2
    while True:
        res = run_simulation(model, mem_type, num_device, b, input_len, output_len, True, tpot_slo)
        if res["success"]:
            b_success = b
            tpot_success = res["tpot"]
            b *= 2
        else:
            b_fail = b
            break
            
    # Phase 2: Binary Search
    low = b_success
    high = b_fail - 1
    max_b = b_success
    max_tpot = tpot_success
    
    while low <= high:
        mid = (low + high) // 2
        res = run_simulation(model, mem_type, num_device, mid, input_len, output_len, True, tpot_slo)
        if res["success"]:
            max_b = mid
            max_tpot = res["tpot"]
            low = mid + 1
        else:
            high = mid - 1
            
    return max_b, max_tpot

print("--- FLASH ONLY FAST CHECK SWEEPS ---")
for model in models:
    for wl_name, (in_len, out_len) in workloads.items():
        for mem in mem_types:
            for gpu in gpus:
                print(f"Running {model} | {wl_name} | {mem}...", flush=True)
                max_b, tpot = find_max_batch_size(model, mem, gpu, in_len, out_len, 0.1)
                tps = max_b / (tpot * gpu) if tpot > 0 else 0.0
                print(f"  Result: Max Batch = {max_b}, TPOT = {tpot:.3f}s, TPS/GPU = {tps:.2f}\n", flush=True)
