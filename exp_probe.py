import sys, os, json, re
sys.path.insert(0, "/home/arcuser/jeongmin/HBF-LLMSimulator/.claude/worktrees/bughunt-third-pass")
from concurrent.futures import ProcessPoolExecutor, as_completed
import run_experiments as R

MODEL="llama4_maverick"; MEM="HBM4"; GPU=8; IN_LEN,OUT_LEN=1660,373; SLO=0.1

# (tag, tp, total_batch)
CASES = [
    ("tp1_3680",1,3680), ("tp1_3720",1,3720), ("tp1_3760",1,3760), ("tp1_3800",1,3800),
    ("tp2_3868",2,3868), ("tp2_3900",2,3900), ("tp2_3960",2,3960), ("tp2_4000",2,4000),
]

def grab(stdout, key):
    m = re.search(rf"{key}: ([\-0-9.eE+]+)GB", stdout or "")
    return float(m.group(1)) if m else None

def actwgt(stdout):
    m = re.search(r"ACT: ([\-0-9.eE+]+)GB, Weight: ([\-0-9.eE+]+)GB, Cache: ([\-0-9.eE+]+)GB", stdout or "")
    if m: return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None,None,None

def cap_reason(stdout):
    for pat in ["HBM capacity exceeded[^\n]*", "Flash capacity exceeded[^\n]*",
                "activation capacity exceeded[^\n]*", "Activations exceed[^\n]*"]:
        m = re.search(pat, stdout or "")
        if m: return m.group(0)
    return None

def work(case):
    tag,tp,batch=case
    dist={"tp":tp,"pp":1,"ep":1}
    res=R.run_simulation(MODEL,MEM,GPU,batch,IN_LEN,OUT_LEN,optimize_parallelism=False,
        tpot_slo=SLO,distribution=dist,temp_cfg_name=f"cfg_{tag}.yaml",worker_tag=f"pr_{tag}")
    so=res.get("stdout","")
    act,wgt,cache=actwgt(so)
    return {"tag":tag,"tp":tp,"batch":batch,"success":res.get("success"),
            "reason":res.get("reason"),"tpot":res.get("tpot"),
            "act":act,"weight":wgt,"cache":cache,"total_gb":grab(so,"Total"),
            "cap_reason":cap_reason(so)}

if __name__=="__main__":
    results=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs={ex.submit(work,c):c for c in CASES}
        for f in as_completed(futs): results.append(f.result())
    results.sort(key=lambda r:r["tag"])
    for r in results:
        st="OK  " if r["success"] else "FAIL"
        tp=f"{r['tpot']*1000:.2f}ms" if r["tpot"] else "-"
        print(f"{r['tag']} {st} tpot={tp} ACT={r['act']} W={r['weight']} KV={r['cache']} Tot={r['total_gb']} | {r['reason']} | {r['cap_reason']}")
    json.dump(results,open("/home/arcuser/jeongmin/HBF-LLMSimulator/.claude/worktrees/bughunt-third-pass/exp_probe_out.json","w"),indent=2)
    print("DONE")
