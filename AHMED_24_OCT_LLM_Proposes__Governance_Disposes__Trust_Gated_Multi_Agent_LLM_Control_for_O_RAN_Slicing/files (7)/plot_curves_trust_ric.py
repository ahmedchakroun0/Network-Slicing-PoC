"""TRUST-RIC - courbes (style conference IEEE, 1 colonne).
Les courbes sont obtenues par une VRAIE implementation du gate deterministe
(gS, gP, gL, gR, memes bornes que Sec. III-C) attaquee par injection de fautes
(Monte Carlo). Le reviewer gV est MODELISE (pas de LLM appele) : voir README."""
import time, numpy as np
from math import erf, sqrt
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True,
    "grid.linestyle": ":", "grid.alpha": .6, "legend.fontsize": 7, "lines.linewidth": 1.4,
    "lines.markersize": 4, "savefig.dpi": 300, "savefig.bbox": "tight"})
OUT = "/mnt/user-data/outputs/"
Phi = np.vectorize(lambda x: .5*(1+erf(x/sqrt(2))))

# ---------------- Gate deterministe (Sec. III-C) ----------------
TOOLS = {  # outil -> (agent, schema, bornes)
 "set_prb_share": ("NEAR", {"cell","slice","share","rationale","effect"}),
 "set_scheduler": ("NEAR", {"cell","slice","sched","rationale","effect"}),
 "set_qos":       ("NEAR", {"cell","slice","prio","pdb","gbr","rationale","effect"}),
 "set_link":      ("NEAR", {"cell","slice","mcs","bler","rationale","effect"}),
 "set_handover":  ("NEAR", {"cell","slice","margin","rationale","effect"}),
 "set_power":     ("NEAR", {"cell","slice","off","rationale","effect"}),
 "set_policy":    ("NON",  {"cell","slice","pmin","pmax","rationale","effect"}),
}
RNG_ = {"share":(5,90),"prio":(1,10),"pdb":(5,300),"gbr":(0,10),"mcs":(-5,5),
        "bler":(.001,.2),"margin":(0,10),"off":(-6,6),"pmin":(5,80),"pmax":(20,100)}
SCHED = {"rr","wf","pf"}

def gS(p, c): 
    t = TOOLS.get(p["tool"]);  a = p["args"]
    return t is not None and set(a) == t[1] and a.get("rationale") and a.get("effect")
def gP(p, c):
    t = TOOLS[p["tool"]]
    return p["agent"] == t[0] and p["args"]["cell"] == c["cell"] and p["args"]["slice"] == c["slice"]
def gL(p, c):
    if not c["V"]: return False
    return not (p["tool"] == "set_policy" and c["rho"] < 3)
def gR(p, c):
    a = p["args"]
    for k, (lo, hi) in RNG_.items():
        if k in a and not (lo <= a[k] <= hi): return False
    if "sched" in a and a["sched"] not in SCHED: return False
    if p["tool"] == "set_policy":
        if a["pmin"] > a["pmax"] or not (a["pmin"] <= c["prb_in_force"] <= a["pmax"]): return False
    if p["tool"] == "set_prb_share" and c["policy"] is not None:
        if not (c["policy"][0] <= a["share"] <= c["policy"][1]): return False
    return True
STAGES = [gS, gP, gL, gR]
def det(p, c):
    for g in STAGES:
        if not g(p, c): return False
    return True

# ---------------- Generateur d'appels ----------------
def valid_call():
    tool = rng.choice(list(TOOLS)); agent, sch = TOOLS[tool]
    a = {k: 0 for k in sch}; a.update(cell="bs2", slice="eMBB", rationale="r", effect="e")
    pol = (int(rng.integers(5, 40)), int(rng.integers(60, 100)))
    for k, (lo, hi) in RNG_.items():
        if k in a: a[k] = rng.uniform(lo, hi)
    if "sched" in a: a["sched"] = rng.choice(list(SCHED))
    c = dict(cell="bs2", slice="eMBB", V={"thr"}, rho=int(rng.integers(1, 6)), policy=pol,
             prb_in_force=float(rng.uniform(*pol)))
    if tool == "set_prb_share": a["share"] = rng.uniform(max(5, pol[0]), min(90, pol[1]))
    if tool == "set_policy":
        a["pmin"], a["pmax"] = pol[0] if pol[0]>=5 else 5, max(pol[1], 20)
        c["rho"] = int(rng.integers(3, 8)); a["pmin"]=min(a["pmin"],c["prb_in_force"]); a["pmax"]=max(a["pmax"],c["prb_in_force"])
    return dict(tool=tool, agent=agent, args=a), c

def corrupt(p, c, cls):
    a = p["args"]
    if cls == 1:   # structurel
        m = rng.integers(3)
        if m == 0: a["extra"] = 1
        elif m == 1: a.pop("rationale")
        else: p["tool"] = "set_prb_share_v2"
    elif cls == 2: # hors plage / inverse
        ks = [k for k in RNG_ if k in a]
        if ks: k = rng.choice(ks); lo, hi = RNG_[k]; a[k] = hi*1.5+1 if rng.random()<.5 else lo-abs(lo)-1
        elif "sched" in a: a["sched"] = "random"
        else: a["share"] = 99 if p["tool"]=="set_prb_share" else a.get("share",0)
        if p["tool"] not in ("set_prb_share","set_policy") and not ks and "sched" not in a:
            a["x"]=1
        if not any(k in a for k in list(RNG_)+["sched"]): a["share"]=140
    elif cls == 3: # injustifie
        m = rng.integers(3)
        if m == 0: c["V"] = set()
        elif m == 1: a["slice"] = "URLLC"
        else:
            p["agent"] = "NON" if p["agent"] == "NEAR" else "NEAR"
    elif cls == 4: # incoherent avec controles en vigueur
        if p["tool"] == "set_policy":
            a["pmin"] = int(c["prb_in_force"]) + 3; a["pmax"] = max(a["pmin"]+5, 20)
            a["pmin"] = min(a["pmin"], 80)
            if a["pmin"] <= c["prb_in_force"]: a["pmin"]=80; c["prb_in_force"]=10
        else:
            p["tool"] = "set_prb_share"; p["agent"] = "NEAR"
            a.clear(); a.update(cell="bs2", slice="eMBB", share=min(90, c["policy"][1]+5) if c["policy"][1] < 85 else 4, rationale="r", effect="e")
    return p, c   # cls 5 : bien forme mais semantiquement faux -> inchange

def build_pool(N=20000):
    X = []
    for i in range(N):
        cls = int(rng.integers(0, 6))       # 0 = valide, 1..5 = classes (i)..(v)
        p, c = valid_call(); p, c = corrupt(p, c, cls)
        X.append((cls, det(p, c)))
    return np.array(X)

pool = build_pool()
cls, ok = pool[:,0], pool[:,1].astype(bool)
print("Verification gate : valides rejetes =", int((~ok[cls==0]).sum()), "/", int((cls==0).sum()))
for k, n in zip(range(1,6), ["(i) structurel","(ii) plage","(iii) injustifie","(iv) incoherent","(v) semantique"]):
    print(f"detection {n:16s}: {100*(~ok[cls==k]).mean():5.1f}%")
det_rate = {k: (~ok[cls==k]).mean() for k in range(1,6)}

# ---------- Courbe 1 : appels fautifs executes vs taux de fausse-acceptation du reviewer ----------
q = np.linspace(0, 1, 21)
def exec_rate(mix5, q, n=200000):
    """mix5 : part de la classe (v) parmi les appels fautifs. Les classes (i)-(iv) se partagent le reste."""
    w = np.array([(1-mix5)/4]*4 + [mix5]); k = rng.choice(5, size=n, p=w)+1
    pass_det = np.array([ (1-det_rate[j]) for j in range(1,6)])[k-1]
    pd = rng.random(n) < pass_det
    return np.array([(pd & (rng.random(n) < qq)).mean() for qq in q])

fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.plot(q, np.ones_like(q), "k-", label="No gate")
ax.plot(q, q, "s-", color="tab:red", markevery=2, label="LLM-only verifier")
for m, mk, col in [(0.10, "o", "tab:blue"), (0.25, "^", "tab:green"), (0.50, "v", "tab:purple")]:
    ax.plot(q, exec_rate(m, q), mk+"-", markevery=2, color=col, label=rf"TRUST-RIC ($\pi_v$={m:.2f})")
ax.set_xlabel("Reviewer false-accept rate $q$"); ax.set_ylabel("Faulty calls executed (fraction)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.legend(loc="upper left")
fig.savefig(OUT+"curve1_faulty_executed_vs_q.png"); plt.close()

# ---------- Courbe 2 : ROC / compromis (valides bloques vs fautifs executes) ----------
t = np.linspace(-4, 7, 300)
fig, ax = plt.subplots(figsize=(3.5, 2.7))
pi_v = 0.25; pass_det = np.mean([1-det_rate[j] for j in (1,2,3,4)])*(1-pi_v) + pi_v*(1-det_rate[5])
for d, ls in [(1.0, "-"), (2.0, "--")]:
    fr = 1 - Phi(t); miss = Phi(t - d)
    ax.plot(fr, miss, "r"+ls, label=f"LLM-only, $d'$={d:g}")
    ax.plot(fr, miss*pass_det, "b"+ls, label=f"TRUST-RIC, $d'$={d:g}")
ax.set_xlabel("Valid calls blocked (false-reject rate)"); ax.set_ylabel("Faulty calls executed (miss rate)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(loc="upper right")
fig.savefig(OUT+"curve2_tradeoff_roc.png"); plt.close()

# ---------- Courbe 3 : fautes executees par 1000 appels vs taux de fautes de l'agent ----------
f = np.linspace(0, .8, 17); q0 = 0.30  # reviewer qui laisse passer 30 % des fautifs
def mixed(archi, ff, n=400000):
    bad = rng.random(n) < ff
    k = rng.choice(5, size=n, p=[.2,.2,.2,.2,.2])+1
    pd = rng.random(n) < np.array([1-det_rate[j] for j in range(1,6)])[k-1]
    rev = rng.random(n) < q0
    if archi == "none": ex = bad
    elif archi == "llm": ex = bad & rev
    else: ex = bad & pd & rev
    return 1000*ex.mean()
fig, ax = plt.subplots(figsize=(3.5, 2.7))
for a, st, col, lab in [("none","k-","k","No gate"),("llm","s-r","r","LLM-only verifier"),("trust","o-b","b","TRUST-RIC")]:
    ax.plot(100*f, [mixed(a, x) for x in f], st[0]+"-" if False else st, markevery=2, label=lab)
ax.set_xlabel("Faulty proposals from the agent (%)"); ax.set_ylabel("Faulty calls executed per 1000 calls")
ax.legend(loc="upper left"); fig.savefig(OUT+"curve3_executed_vs_fault_rate.png"); plt.close()

# ---------- Courbe 4 : CDF de la latence de decision (mesuree) ----------
lat = {g.__name__: [] for g in STAGES}; tot = []
samples = []
for _ in range(20000):
    p, c = valid_call(); samples.append((p, c))
for g in STAGES:
    L = []
    for p, c in samples:
        t0 = time.perf_counter_ns(); g(p, c); L.append(time.perf_counter_ns()-t0)
    lat[g.__name__] = np.array(L)/1e6    # ms
L = []
for p, c in samples:
    t0 = time.perf_counter_ns(); det(p, c); L.append(time.perf_counter_ns()-t0)
L = np.sort(np.array(L)/1e6)
fig, ax = plt.subplots(figsize=(3.5, 2.7))
for (n, v), ls in zip(lat.items(), ["-","--","-.",":"]):
    v = np.sort(v); ax.plot(v, np.arange(1, len(v)+1)/len(v), ls, label=n)
ax.plot(L, np.arange(1, len(L)+1)/len(L), "k-", lw=2, label=r"$g_Sg_Pg_Lg_R$ (total)")
gv = 273/11*1000     # ms : temps agent R2 / appels modele (Table II) ~ ordre de grandeur de gV
ax.axvline(10, color="tab:red", ls="--"); ax.text(11, .5, "Near-RT\n10 ms", color="tab:red", fontsize=6)
ax.axvline(1000, color="tab:orange", ls="--"); ax.text(1100, .25, "1 s", color="tab:orange", fontsize=6)
ax.axvline(gv, color="gray", ls="-."); ax.text(gv*1.15, .75, r"$g_V$ ($\approx$25 s,"+"\nfrom Table II)", fontsize=6)
ax.set_xscale("log"); ax.set_xlim(1e-4, 1e5); ax.set_xlabel("Decision latency (ms)"); ax.set_ylabel("CDF")
ax.legend(loc="center", bbox_to_anchor=(0.42,0.3), fontsize=6); fig.savefig(OUT+"curve4_latency_cdf.png"); plt.close()
print("median det. gate latency (ms):", np.median(L), " p99:", np.percentile(L, 99))
