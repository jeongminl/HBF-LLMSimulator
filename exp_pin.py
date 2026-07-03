import sys, json, re
sys.path.insert(0, "/home/arcuser/jeongmin/HBF-LLMSimulator/.claude/worktrees/bughunt-third-pass")
from concurrent.futures import ProcessPoolExecutor, as_completed
import run_experiments as R
MODEL="llama4_maverick"; MEM="HBM4"; GPU=8; IN,OUT=1660,373; SLO=0.1
CASES=[("tp2_3908",2,3908),("tp2_3920",2,3920),("tp2_3940",2,3940),("tp2_3956",2,3956)]
def work(c):
    tag,tp,b=c
    r=R.run_simulation(MODEL,MEM,GPU,b,IN,OUT,optimize_parallelism=False,tpot_slo=SLO,
        distribution={"tp":tp,"pp":1,"ep":1},temp_cfg_name=f"cfg_{tag}.yaml",worker_tag=f"pin_{tag}")
    m=re.search(r"HBM capacity exceeded[^\n]*", r.get("stdout","") or "")
    tps=b/(r["tpot"]*GPU) if r.get("success") else None
    return {"tag":tag,"batch":b,"ok":r.get("success"),"tpot":r.get("tpot"),
            "tps_gpu":tps,"reason":r.get("reason"),"oom":m.group(0) if m else None}
if __name__=="__main__":
    res=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        fs={ex.submit(work,c):c for c in CASES}
        for f in as_completed(fs): res.append(f.result())
    res.sort(key=lambda r:r["batch"])
    for r in res:
        t=f"{r['tpot']*1000:.2f}ms" if r["tpot"] else "-"
        tg=f"{r['tps_gpu']:.0f}" if r["tps_gpu"] else "-"
        print(f"{r['tag']} ok={r['ok']} tpot={t} tps/gpu={tg} | {r['reason']} | {r['oom']}")
    print("DONE")
