"""
Fast Verification Experiments for ToN Paper
============================================
Reduced scale: 1000 time slots, 3 seeds, 5x5 grid
Still uses paper defaults: M_v=4, T_coh=20, lambda=0.5, F^th=0.65
"""


import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO_ROOT = _os.path.dirname(_HERE)
EXPERIMENTS_DIR = _os.path.join(REPO_ROOT, "experiments")
FIGURES_DIR = _os.path.join(REPO_ROOT, "figures")
_os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
_os.makedirs(FIGURES_DIR, exist_ok=True)

import sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import json
import time
from simulation_engine import QuantumNetworkSimulator, SimulationConfig
from network_model import (
    QuantumNetwork, fidelity_decay, swap_fidelity, chain_fidelity
)
from protocols import AoEARS, ShortestPathRouting, FidelityAwareRouting, GreedyScheduling

# Paper defaults
NUM_SLOTS = 1000
NUM_SEEDS = 3
ROWS, COLS = 5, 5
M_V = 4
T_COH = 20.0
P_GEN = 0.5
F0 = 0.95
LAMBDA = 0.5
FTH = 0.65


def make_net(memory_slots=M_V, coherence_time=T_COH):
    return QuantumNetwork.create_grid(ROWS, COLS,
        memory_slots=memory_slots, coherence_time=coherence_time,
        link_success_prob=P_GEN, link_fidelity=F0)


def run_one(proto_name, fth=FTH, seed=42, proto_kwargs=None):
    net = make_net()
    cfg = SimulationConfig(
        num_time_slots=NUM_SLOTS,
        request_arrival_rate=LAMBDA,
        fidelity_threshold=fth,
        seed=seed,
        min_path_length=2,
        max_concurrent_requests=50,
    )
    pk = proto_kwargs or {}
    if proto_name == 'aoe_ars':
        proto = AoEARS(net, seed=seed, **pk)
    elif proto_name == 'shortest_path':
        proto = ShortestPathRouting(net, seed=seed)
    elif proto_name == 'fidelity_aware':
        proto = FidelityAwareRouting(net, seed=seed, fidelity_threshold=fth)
    elif proto_name == 'greedy':
        proto = GreedyScheduling(net, seed=seed)
    else:
        raise ValueError(proto_name)
    
    sim = QuantumNetworkSimulator(net, proto, cfg)
    m = sim.run()
    s = m.summary()
    s['eta_eff'] = s['throughput'] * (1.0 - s['violation_rate'])
    return s


def run_avg(proto_name, fth=FTH, proto_kwargs=None, n_seeds=NUM_SEEDS):
    results = []
    for i in range(n_seeds):
        s = run_one(proto_name, fth=fth, seed=i*17+42, proto_kwargs=proto_kwargs)
        results.append(s)
    avg = {}
    for k in results[0]:
        vals = [r[k] for r in results]
        avg[k] = float(np.mean(vals))
    return avg


# ============================================================
print("="*70)
print("ToN Verification Experiments (fast mode: 1000 slots, 3 seeds)")
print("="*70)
t0 = time.time()
all_results = {}

# --- C8: Werner validation (instant, no sim needed) ---
print("\n[C8] Werner Approximation Validation...")
rng = np.random.default_rng(42)
c8 = {}
for path_len in range(2, 9):
    werner_preds = []
    mc_actuals = []
    for _ in range(10000):
        ages = rng.uniform(0, T_COH * 0.8, size=path_len)
        aged = [fidelity_decay(F0, a, T_COH) for a in ages]
        wp = chain_fidelity(aged)
        # sequential swap
        cur = aged[0]
        for i in range(1, path_len):
            cur = swap_fidelity(cur, aged[i])
        werner_preds.append(wp)
        mc_actuals.append(cur)
    errs = np.abs(np.array(werner_preds) - np.array(mc_actuals))
    c8[path_len] = {
        'werner_mean': float(np.mean(werner_preds)),
        'mc_mean': float(np.mean(mc_actuals)),
        'mean_error': float(np.mean(errs)),
        'max_error': float(np.max(errs)),
    }
    print(f"  L={path_len}: Werner={np.mean(werner_preds):.5f}, MC={np.mean(mc_actuals):.5f}, "
          f"|err|={np.mean(errs):.6f}, max={np.max(errs):.6f}")
all_results['C8'] = c8
print(f"  Done in {time.time()-t0:.1f}s")

# --- A3: F^th sweep ---
print("\n[A3] Fidelity Threshold Sweep...")
t1 = time.time()
a3 = {}
fth_vals = [0.65, 0.75, 0.85, 0.90, 0.95]
protos = ['aoe_ars', 'shortest_path', 'fidelity_aware', 'greedy']
for fth in fth_vals:
    a3[str(fth)] = {}
    for p in protos:
        pk = {'base_discard_threshold': 0.7} if p == 'aoe_ars' else None
        avg = run_avg(p, fth=fth, proto_kwargs=pk)
        a3[str(fth)][p] = avg
        print(f"  F^th={fth}, {p:20s}: del={avg['throughput']:.4f} viol={avg['violation_rate']:.4f} eta={avg['eta_eff']:.4f}")
all_results['A3'] = a3
print(f"  Done in {time.time()-t1:.1f}s")

# --- C2: Beta sweep ---
print("\n[C2] Beta Parameter Sweep...")
t1 = time.time()
c2 = {}
for beta in [0.1, 0.5, 1.0, 2.0, 5.0]:
    pk = {'alpha': 1.0, 'beta': beta, 'base_discard_threshold': 0.7, 'k_paths': 5}
    avg = run_avg('aoe_ars', proto_kwargs=pk)
    c2[str(beta)] = avg
    print(f"  beta={beta}: del={avg['throughput']:.4f} fid={avg['avg_fidelity']:.4f} "
          f"viol={avg['violation_rate']:.4f} eta={avg['eta_eff']:.4f} lat={avg['avg_latency']:.1f}")
all_results['C2'] = c2
print(f"  Done in {time.time()-t1:.1f}s")

# --- B7: Ablation ---
print("\n[B7] Component Ablation...")
t1 = time.time()
b7_configs = [
    ('SP only',           {'alpha': 0.0, 'beta': 0.0, 'base_discard_threshold': 0.0, 'k_paths': 1}),
    ('SP+Refresh',        {'alpha': 0.0, 'beta': 0.0, 'base_discard_threshold': 0.7, 'k_paths': 1}),
    ('SP+AoE',            {'alpha': 1.0, 'beta': 0.5, 'base_discard_threshold': 0.0, 'k_paths': 5}),
    ('Refresh+AoE',       {'alpha': 1.0, 'beta': 0.5, 'base_discard_threshold': 0.7, 'k_paths': 5}),
    ('Full(+Admission)',  {'alpha': 1.0, 'beta': 0.5, 'base_discard_threshold': 0.7, 'k_paths': 5}),
]
b7 = {}
for name, pk in b7_configs:
    avg = run_avg('aoe_ars', proto_kwargs=pk)
    b7[name] = avg
    print(f"  {name:20s}: del={avg['throughput']:.4f} fid={avg['avg_fidelity']:.4f} "
          f"viol={avg['violation_rate']:.4f} eta={avg['eta_eff']:.4f}")
all_results['B7'] = b7
print(f"  Done in {time.time()-t1:.1f}s")

# --- B6: QoS breakdown ---
print("\n[B6] Per-QoS Class Breakdown...")
t1 = time.time()
b6 = {}
qos = {'Premium': 0.90, 'Standard': 0.75, 'Best-effort': 0.65}
for qname, qfth in qos.items():
    b6[qname] = {}
    for p in ['aoe_ars', 'shortest_path']:
        pk = {'base_discard_threshold': 0.7} if p == 'aoe_ars' else None
        avg = run_avg(p, fth=qfth, proto_kwargs=pk)
        b6[qname][p] = avg
        print(f"  {qname:12s} {p:15s}: del={avg['throughput']:.4f} fid={avg['avg_fidelity']:.4f} "
              f"viol={avg['violation_rate']:.4f} eta={avg['eta_eff']:.4f}")
all_results['B6'] = b6
print(f"  Done in {time.time()-t1:.1f}s")

# --- C1: tau x alpha heatmap (3x3 for speed) ---
print("\n[C1] tau x alpha Heatmap...")
t1 = time.time()
c1 = {}
tau_vals = [0.5, 0.7, 0.9]
alpha_vals = [0.5, 1.0, 5.0]
for tau in tau_vals:
    c1[str(tau)] = {}
    for alpha in alpha_vals:
        pk = {'alpha': alpha, 'beta': 0.5, 'base_discard_threshold': tau, 'k_paths': 5}
        avg = run_avg('aoe_ars', proto_kwargs=pk, n_seeds=2)
        c1[str(tau)][str(alpha)] = avg['eta_eff']
    print(f"  tau={tau}: " + " ".join(f"a={a}:{c1[str(tau)][str(a)]:.4f}" for a in alpha_vals))
all_results['C1'] = c1
print(f"  Done in {time.time()-t1:.1f}s")

# --- C3: Mixed deployment ---
print("\n[C3] Mixed Deployment...")
t1 = time.time()
c3 = {}
levels = {
    '0%_SP': {'alpha': 0.0, 'beta': 0.0, 'base_discard_threshold': 0.0, 'k_paths': 1},
    '30%': {'alpha': 0.3, 'beta': 0.15, 'base_discard_threshold': 0.3, 'k_paths': 2},
    '50%': {'alpha': 0.5, 'beta': 0.25, 'base_discard_threshold': 0.5, 'k_paths': 3},
    '70%': {'alpha': 0.7, 'beta': 0.35, 'base_discard_threshold': 0.6, 'k_paths': 4},
    '100%_Full': {'alpha': 1.0, 'beta': 0.5, 'base_discard_threshold': 0.7, 'k_paths': 5},
}
for name, pk in levels.items():
    avg = run_avg('aoe_ars', proto_kwargs=pk)
    c3[name] = avg
    print(f"  {name:12s}: del={avg['throughput']:.4f} fid={avg['avg_fidelity']:.4f} "
          f"viol={avg['violation_rate']:.4f} eta={avg['eta_eff']:.4f}")
all_results['C3'] = c3
print(f"  Done in {time.time()-t1:.1f}s")

# --- SAVE ---
elapsed = time.time() - t0
print(f"\n{'='*70}")
print(f"ALL DONE in {elapsed:.1f}s")
print(f"{'='*70}")

class NE(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

with open(_os.path.join(EXPERIMENTS_DIR, "verification_results.json"), 'w') as f:
    json.dump(all_results, f, indent=2, cls=NE)
print("Results saved to verification_results.json")
