import os
import sys
import yaml
import subprocess
import glob
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

def run_simulation(model, mem_type, num_device, batch_size, input_len, output_len, optimize_parallelism=True, tpot_slo=0.1):
    # Load default config
    with open(os.path.join(SCRIPT_DIR, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    # Update config values
    cfg["model"]["model_name"] = model
    cfg["system"]["memory_type"] = mem_type

    # Handle num_device and num_node partitioning
    if num_device == 16:
        cfg["system"]["num_node"] = 2
        cfg["system"]["num_device"] = 8
    else:
        cfg["system"]["num_node"] = 1
        cfg["system"]["num_device"] = num_device

    cfg["serving"]["max_batch_size"] = batch_size
    cfg["simulation"]["input_len"] = input_len
    cfg["simulation"]["output_len"] = output_len
    n_iter = 10
    cfg["simulation"]["iter"] = n_iter
    cfg["simulation"]["injection_rate"] = 0  # Continuous batching

    cfg["system"]["optimize_parallelism"] = optimize_parallelism
    cfg["system"]["tpot_slo"] = tpot_slo
    # Make capacity / SRAM violations a hard failure so the batch-size sweep's
    # stop condition enforces all three constraints (capacity, SRAM, SLO), not
    # just the SLO.  The optimizer already exits non-zero when no parallelism
    # config fits; this also catches any over-capacity the simulation itself sees.
    cfg["simulation"]["exit_out_of_memory"] = True

    # Save temp config inside build
    temp_cfg_path = os.path.join(BUILD_DIR, "config_temp.yaml")
    with open(temp_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    cmd = ["./run", "config_temp.yaml"]
    try:
        res = subprocess.run(cmd, cwd=BUILD_DIR, capture_output=True, text=True, timeout=1800)
        stdout = res.stdout + "\n" + res.stderr

        # returncode != 0 catches optimizer OOM (exits with EXIT_FAILURE).
        # String markers catch the simulation OOM path which does "return 0".
        fail_markers = ["Out of Memory", "Activations exceed",
                        "Flash capacity exceeded", "capacity exceeded"]
        if res.returncode != 0 or any(m in stdout for m in fail_markers):
            return {"success": False, "reason": "OOM/Crash", "stdout": stdout}

        # Parse total simulation time.  cluster.cpp also emits "Total: X.XXXeGB"
        # memory-size lines; skip those by rejecting values containing letters.
        total_time_ns = None
        for line in stdout.split("\n"):
            if line.startswith("Total: "):
                raw = line[len("Total: "):].strip()
                if raw and not any(c.isalpha() for c in raw):
                    try:
                        total_time_ns = float(raw)
                    except ValueError:
                        pass
        if total_time_ns is None:
            return {"success": False, "reason": "No total time printed", "stdout": stdout}

        tpot = total_time_ns / (n_iter * 1e9)
        if tpot > tpot_slo:
            return {"success": False, "reason": f"SLO Violated ({tpot:.3f}s > {tpot_slo:.3f}s)", "tpot": tpot, "stdout": stdout}

        # Parse PEC geometry emitted by the C++ binary (single source of truth for
        # model dimensions, precision, and flash capacity).
        pec_kv_bytes = None
        pec_capacity = None
        for line in stdout.split("\n"):
            if line.startswith("PEC_KV_BYTES_PER_TOKEN: "):
                try:
                    pec_kv_bytes = float(line.split(": ", 1)[1])
                except Exception:
                    pass
            elif line.startswith("PEC_FLASH_CAPACITY_BYTES: "):
                try:
                    pec_capacity = float(line.split(": ", 1)[1])
                except Exception:
                    pass

        # Find the newly generated or modified CSV file
        csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        csv_file = None
        newest_time = 0
        current_time = time.time()
        for f in csv_files:
            try:
                mtime = os.path.getmtime(f)
                if mtime > newest_time and (current_time - mtime) < 15:
                    newest_time = mtime
                    csv_file = f
            except:
                pass

        return {"success": True, "tpot": tpot, "csv_file": csv_file, "stdout": stdout,
                "pec_kv_bytes": pec_kv_bytes, "pec_capacity": pec_capacity}
    except Exception as e:
        return {"success": False, "reason": str(e), "stdout": ""}

def find_max_batch_size(model, mem_type, num_device, input_len, output_len, tpot_slo=0.1):
    # Phase 1: Exponential Search — double until failure (no arbitrary cap;
    # termination is guaranteed by the capacity / SRAM / SLO constraint).
    res = run_simulation(model, mem_type, num_device, 1, input_len, output_len, True, tpot_slo)
    if not res["success"]:
        return 0, 0.0, None, None, None

    b_success = 1
    tpot_success = res["tpot"]
    last_csv = res.get("csv_file")
    last_pec_kv = res.get("pec_kv_bytes")
    last_pec_cap = res.get("pec_capacity")

    b = 2
    while True:
        res = run_simulation(model, mem_type, num_device, b, input_len, output_len, True, tpot_slo)
        if res["success"]:
            b_success = b
            tpot_success = res["tpot"]
            last_csv = res.get("csv_file")
            last_pec_kv = res.get("pec_kv_bytes")
            last_pec_cap = res.get("pec_capacity")
            b *= 2
        else:
            b_fail = b
            break

    # Phase 2: Binary Search — b_success is already confirmed good, so search
    # starts one above it; max_b is pre-initialised to avoid re-testing b_success.
    low = b_success + 1
    high = b_fail - 1
    max_b = b_success
    max_tpot = tpot_success

    while low <= high:
        mid = (low + high) // 2
        res = run_simulation(model, mem_type, num_device, mid, input_len, output_len, True, tpot_slo)
        if res["success"]:
            max_b = mid
            max_tpot = res["tpot"]
            last_csv = res.get("csv_file")
            last_pec_kv = res.get("pec_kv_bytes")
            last_pec_cap = res.get("pec_capacity")
            low = mid + 1
        else:
            high = mid - 1

    return max_b, max_tpot, last_csv, last_pec_kv, last_pec_cap

def parse_csv_breakdown(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        import csv
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # We find rows with type = t2t or rows with actual duration info
        t2t_rows = [r for r in rows if r.get("type") == "t2t"]
        if not t2t_rows:
            return None

        # Parse average values
        avg_row = {}
        for col in t2t_rows[0].keys():
            try:
                avg_row[col] = sum(float(r[col]) for r in t2t_rows) / len(t2t_rows)
            except:
                avg_row[col] = 0.0
        return avg_row
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return None

def main():
    # Make sure output data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # We will record results for:
    # Models: llama3_405B, llama4_maverick
    # GPU counts: 1, 2, 4, 8, 16
    # Memory types: HBM4, HBF, HBF+, CONV, CONV+
    # Workloads: SHORT, MID, LONG

    models = ["llama3_405B", "llama4_maverick"]
    workloads = {
        "SHORT": (1660, 373),
        "MID": (5900, 499),
        "LONG": (103500, 1100)
    }
    mem_types = ["HBM4", "HBF", "HBF+", "CONV", "CONV+"]
    gpus = [1, 2, 4, 8, 16]

    # Gather baselines first (8 GPU, HBM4, slo=0.1); cached and reused in the
    # main sweep to avoid re-running the same simulation.
    baselines = {}
    print("--- GATHERING BASELINES (8 GPU HBM4) ---")
    for model in models:
        baselines[model] = {}
        for wl_name, (in_len, out_len) in workloads.items():
            max_b, tpot, csv_file, pec_kv, pec_cap = find_max_batch_size(
                model, "HBM4", 8, in_len, out_len, 0.1)
            tps = max_b / (tpot * 8) if tpot > 0 else 0.0
            baselines[model][wl_name] = {
                "max_batch": max_b,
                "tps": tps,
                "tpot": tpot,
                "csv_file": csv_file,
                "pec_kv_bytes": pec_kv,
                "pec_capacity": pec_cap,
            }
            print(f"Baseline {model} {wl_name}: Max Batch = {max_b}, TPS/GPU = {tps:.2f}")

    results = []

    print("\n--- RUNNING ALL SWEEPS ---")
    for model in models:
        for wl_name, (in_len, out_len) in workloads.items():
            for mem in mem_types:
                for gpu in gpus:
                    print(f"Running {model} | {wl_name} | {mem} | {gpu} GPUs...")

                    # HBM4/8-GPU result was already gathered as the baseline;
                    # pull from cache to avoid a duplicate simulation.
                    if mem == "HBM4" and gpu == 8:
                        bl = baselines[model][wl_name]
                        max_b      = bl["max_batch"]
                        tpot       = bl["tpot"]
                        csv_path   = bl["csv_file"]
                        last_pec_kv = bl["pec_kv_bytes"]
                        last_pec_cap = bl["pec_capacity"]
                    else:
                        max_b, tpot, csv_path, last_pec_kv, last_pec_cap = find_max_batch_size(
                            model, mem, gpu, in_len, out_len, 0.1)

                    tps = max_b / (tpot * gpu) if tpot > 0 else 0.0

                    # Normalization
                    base_max_batch = baselines[model][wl_name]["max_batch"]
                    base_tps = baselines[model][wl_name]["tps"]

                    norm_batch = max_b / base_max_batch if base_max_batch > 0 else 0.0
                    norm_tps = tps / base_tps if base_tps > 0 else 0.0

                    # Breakdown parse
                    breakdown = parse_csv_breakdown(csv_path) if csv_path else None

                    results.append({
                        "model": model,
                        "workload": wl_name,
                        "memory": mem,
                        "gpus": gpu,
                        "max_batch": max_b,
                        "tps": tps,
                        "tpot": tpot,
                        "norm_batch": norm_batch,
                        "norm_tps": norm_tps,
                        "breakdown": breakdown,
                        "csv_path": csv_path,
                        "pec_kv_bytes": last_pec_kv,
                        "pec_capacity": last_pec_cap,
                    })
                    print(f"  -> Max Batch: {max_b} (Norm: {norm_batch:.2f}), TPS/GPU: {tps:.2f} (Norm: {norm_tps:.2f})")

    # SLO Sensitivity Sweeps (LONG Workload only)
    print("\n--- RUNNING SLO SENSITIVITY SWEEPS (LONG Workload) ---")
    slo_results = []
    long_in, long_out = workloads["LONG"]
    slos = [0.05, 0.1, 0.2, 86400.0]  # SLO values (Offline represents 24 hours)

    # Gather sensitivity baselines (HBM4, 8 GPU) per SLO.
    # slo=0.1 / LONG workload was already gathered as the main baseline.
    sens_baselines = {}
    for model in models:
        sens_baselines[model] = {}
        for slo in slos:
            if slo == 0.1:
                bl = baselines[model]["LONG"]
                max_b = bl["max_batch"]
                tpot  = bl["tpot"]
            else:
                max_b, tpot, _, _, _ = find_max_batch_size(
                    model, "HBM4", 8, long_in, long_out, slo)
            tps = max_b / (tpot * 8) if tpot > 0 else 0.0
            sens_baselines[model][slo] = {
                "max_batch": max_b,
                "tps": tps,
                "tpot": tpot
            }
            print(f"SLO Sensitivity Baseline {model} SLO {slo}: Max Batch = {max_b}, TPS/GPU = {tps:.2f}")

    for model in models:
        for mem in mem_types:
            for gpu in gpus:
                for slo in slos:
                    # HBM4/8-GPU/slo=0.1 is the main LONG baseline — reuse it.
                    if mem == "HBM4" and gpu == 8 and slo == 0.1:
                        bl = baselines[model]["LONG"]
                        max_b = bl["max_batch"]
                        tpot  = bl["tpot"]
                    else:
                        print(f"Sensitivity Sweep: {model} | {mem} | {gpu} GPUs | SLO {slo}s...")
                        max_b, tpot, _, _, _ = find_max_batch_size(
                            model, mem, gpu, long_in, long_out, slo)
                    tps = max_b / (tpot * gpu) if tpot > 0 else 0.0

                    base_max_batch = sens_baselines[model][slo]["max_batch"]
                    base_tps = sens_baselines[model][slo]["tps"]

                    norm_batch = max_b / base_max_batch if base_max_batch > 0 else 0.0
                    norm_tps = tps / base_tps if base_tps > 0 else 0.0

                    slo_results.append({
                        "model": model,
                        "memory": mem,
                        "gpus": gpu,
                        "slo": slo,
                        "max_batch": max_b,
                        "tps": tps,
                        "norm_batch": norm_batch,
                        "norm_tps": norm_tps
                    })
                    if not (mem == "HBM4" and gpu == 8 and slo == 0.1):
                        print(f"  -> Max Batch: {max_b} (Norm: {norm_batch:.2f}), TPS/GPU: {tps:.2f} (Norm: {norm_tps:.2f})")

    # Generate Markdown Report — written to the project root alongside this script.
    report_path = os.path.join(SCRIPT_DIR, "experiment_results.md")
    print(f"\nWriting final report to {report_path}...")

    with open(report_path, "w") as f:
        f.write("# HBF memory model simulation results\n\n")
        f.write("This report presents the findings from the automated sweeps evaluating HBF memory models.\n\n")

        # 1. Maximum Per-GPU Batch Size (Figure 3)
        f.write("## 1. Maximum Per-GPU Batch Size (Figure 3 Replication)\n\n")
        f.write("| Model | Workload | Memory Config | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 16 GPU |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for model in models:
            for wl in workloads.keys():
                for mem in mem_types:
                    gpu_vals = []
                    for gpu in gpus:
                        val = next(r for r in results if r["model"] == model and r["workload"] == wl and r["memory"] == mem and r["gpus"] == gpu)
                        gpu_vals.append(f"{val['max_batch']} ({val['norm_batch']:.2f}x)")
                    f.write(f"| {model} | {wl} | {mem} | " + " | ".join(gpu_vals) + " |\n")

        f.write("\n")

        # 2. System Throughput (Figure 4)
        f.write("## 2. System Throughput (Figure 4 Replication)\n\n")
        f.write("| Model | Workload | Memory Config | 1 GPU | 2 GPU | 4 GPU | 8 GPU | 16 GPU |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for model in models:
            for wl in workloads.keys():
                for mem in mem_types:
                    gpu_vals = []
                    for gpu in gpus:
                        val = next(r for r in results if r["model"] == model and r["workload"] == wl and r["memory"] == mem and r["gpus"] == gpu)
                        gpu_vals.append(f"{val['tps']:.2f} ({val['norm_tps']:.2f}x)")
                    f.write(f"| {model} | {wl} | {mem} | " + " | ".join(gpu_vals) + " |\n")

        f.write("\n")

        # 3. Performance Breakdown (Figure 5)
        f.write("## 3. Runtime Performance Breakdown (Figure 5 Replication)\n\n")
        f.write("Fractions of decode execution time spent on key components under a 0.1s TPOT SLO (8-GPU configuration):\n\n")
        f.write("| Model | Workload | Memory | Attention | FFN | KV Write | Communication | Others |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for model in models:
            for wl in workloads.keys():
                for mem in mem_types:
                    val = next(r for r in results if r["model"] == model and r["workload"] == wl and r["memory"] == mem and r["gpus"] == 8)
                    bd = val["breakdown"]

                    # If OOM/failed, write N/A
                    if val["max_batch"] == 0 or not bd:
                        f.write(f"| {model} | {wl} | {mem} | N/A | N/A | N/A | N/A | N/A |\n")
                        continue

                    # Attention duration sums GQA or MLA components.
                    # MLA_MODELS must match model names with compressed_kv=true in model_config.h.
                    MLA_MODELS = {"deepseekV3", "deepseekR1"}
                    if model in MLA_MODELS:
                        # MLA
                        attn = (bd.get("q_down_proj", 0.0) + bd.get("kv_down_proj", 0.0) + bd.get("kr_proj", 0.0) +
                                bd.get("q_up_proj", 0.0) + bd.get("qr_proj", 0.0) + bd.get("kv_up_proj", 0.0) +
                                bd.get("tr_k_up_proj", 0.0) + bd.get("v_up_proj", 0.0) + bd.get("atten_sum", 0.0) +
                                bd.get("atten_gen", 0.0) + bd.get("o_proj", 0.0))
                    else:
                        # Standard GQA
                        attn = (bd.get("atten_sum", 0.0) + bd.get("atten_gen", 0.0) +
                                bd.get("o_proj", 0.0) + bd.get("qkvgen", 0.0))
                    ffn = bd.get("ffn", 0.0) + bd.get("expert_ffn", 0.0)

                    comm = bd.get("communication", 0.0)
                    others = bd.get("layernorm", 0.0) + bd.get("residual", 0.0) + bd.get("rope", 0.0) + bd.get("lm_head", 0.0)

                    kv_write_per_step = bd.get("kv_write", 0.0)

                    # Normalize breakdown
                    tot = attn + ffn + comm + kv_write_per_step + others
                    if tot == 0:
                        tot = 1.0
                    f_attn = attn / tot
                    f_ffn = ffn / tot
                    f_kv = kv_write_per_step / tot
                    f_comm = comm / tot
                    f_others = others / tot

                    f.write(f"| {model} | {wl} | {mem} | {f_attn:.1%} | {f_ffn:.1%} | {f_kv:.1%} | {f_comm:.1%} | {f_others:.1%} |\n")

        f.write("\n")

        # 4. SLO Sensitivity Analysis (Figure 6) — all GPU counts
        f.write("## 4. SLO Sensitivity Analysis (Figure 6 Replication)\n\n")
        gpu_headers = " | ".join(f"{g} GPU" for g in gpus)
        f.write(f"| Model | Memory | SLO | Metric | {gpu_headers} |\n")
        f.write("|" + "---|" * (5 + len(gpus)) + "\n")
        for model in models:
            for mem in mem_types:
                for slo in slos:
                    slo_label = f"{slo}s" if slo < 1000 else "Offline (24h)"
                    # Batch size row
                    b_vals = []
                    for gpu in gpus:
                        val = next(r for r in slo_results if r["model"] == model and r["memory"] == mem and r["gpus"] == gpu and r["slo"] == slo)
                        b_vals.append(f"{val['max_batch']} ({val['norm_batch']:.2f}x)")
                    f.write(f"| {model} | {mem} | {slo_label} | Batch Size | " + " | ".join(b_vals) + " |\n")

                    # TPS row
                    tps_vals = []
                    for gpu in gpus:
                        val = next(r for r in slo_results if r["model"] == model and r["memory"] == mem and r["gpus"] == gpu and r["slo"] == slo)
                        tps_vals.append(f"{val['tps']:.2f} ({val['norm_tps']:.2f}x)")
                    f.write(f"| {model} | {mem} | {slo_label} | TPS/GPU | " + " | ".join(tps_vals) + " |\n")

        f.write("\n")

        # 5. Write Traffic and Endurance Assessment (Figure 7)
        f.write("## 5. Write Traffic and Endurance Assessment (Figure 7 Replication)\n\n")
        f.write("Assuming continuous batching in steady state over a 3-year warranty lifespan:\n\n")
        f.write("| Model | Workload | Memory | Batch Size | TPS/GPU | Write rate/GPU (MB/s) | 3-Year PEC | Lifetime Status |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for model in models:
            for wl in workloads.keys():
                for mem in ["HBF", "HBF+", "CONV", "CONV+"]:
                    val = next(r for r in results if r["model"] == model and r["workload"] == wl and r["memory"] == mem and r["gpus"] == 8)
                    if val["max_batch"] == 0:
                        f.write(f"| {model} | {wl} | {mem} | 0 | 0.00 | 0.0 | N/A | N/A |\n")
                        continue

                    in_len, out_len = workloads[wl]

                    # KV geometry and flash capacity sourced from C++ binary output,
                    # so precision and model dimensions are always consistent with the
                    # model preset (no duplication / desync risk).
                    kv_bytes_per_token = val.get("pec_kv_bytes")
                    capacity = val.get("pec_capacity")
                    if kv_bytes_per_token is None or capacity is None or capacity == 0:
                        f.write(f"| {model} | {wl} | {mem} | {val['max_batch']} | {val['tps']:.2f} | N/A | N/A | N/A |\n")
                        continue

                    # kv_bytes_per_token is the system-total KV bytes per token (all
                    # GPUs, all layers).  Dividing by val["gpus"] gives per-GPU write
                    # rate, which is correct for any TP/PP split (uniform sharding).
                    kv_size = kv_bytes_per_token * (in_len + out_len)
                    num_gpus = val["gpus"]

                    # Completion rate ≈ max_batch / out_len per decode step.
                    write_rate_bps = (val["max_batch"] / out_len) * kv_size / (val["tpot"] * num_gpus)
                    write_rate_mbps = write_rate_bps / 1e6

                    # 3-year PEC: total bytes written / flash capacity per GPU.
                    pec = (write_rate_bps * 3600 * 24 * 365 * 3) / capacity

                    status = "PASS (Safe)" if pec < 100000 else "FAIL (Wear-out)"
                    f.write(f"| {model} | {wl} | {mem} | {val['max_batch']} | {val['tps']:.2f} | {write_rate_mbps:.1f} | {pec:.1f} | {status} |\n")

if __name__ == "__main__":
    main()
