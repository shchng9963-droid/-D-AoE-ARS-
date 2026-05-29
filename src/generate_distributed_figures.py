"""Generate figures for distributed experiments."""


import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO_ROOT = _os.path.dirname(_HERE)
EXPERIMENTS_DIR = _os.path.join(REPO_ROOT, "experiments")
FIGURES_DIR = _os.path.join(REPO_ROOT, "figures")
_os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
_os.makedirs(FIGURES_DIR, exist_ok=True)

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Load results
with open(_os.path.join(EXPERIMENTS_DIR, "distributed_results.json")) as f:
    results = json.load(f)

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150
})

colors = {
    'D-AoE-ARS': '#d62728',
    'D-Greedy': '#ff7f0e', 
    'SP': '#1f77b4',
    'Q-CAST': '#2ca02c'
}
markers = {
    'D-AoE-ARS': 'o',
    'D-Greedy': 'D',
    'SP': 's',
    'Q-CAST': '^'
}

# ============================================================
# Figure 1: Communication Delay Sweep (KEY FIGURE)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

delays = [0, 1, 2, 5, 10, 20]
protocols = ['D-AoE-ARS', 'D-Greedy', 'SP', 'Q-CAST']

# (a) Delivery Ratio vs Delay
ax = axes[0]
for proto in protocols:
    means = [results['comm_delay'][str(d)][proto]['delivery_mean'] for d in delays]
    stds = [results['comm_delay'][str(d)][proto]['delivery_std'] for d in delays]
    ax.errorbar(delays, means, yerr=stds, marker=markers[proto], 
                color=colors[proto], label=proto, capsize=3, linewidth=1.5)
ax.set_xlabel('Communication Delay (time slots)')
ax.set_ylabel('Delivery Ratio')
ax.set_title('(a) Throughput vs Delay')
ax.legend(loc='lower left', fontsize=8)
ax.set_ylim(0, 0.8)
ax.grid(True, alpha=0.3)

# (b) Fidelity vs Delay
ax = axes[1]
for proto in protocols:
    means = [results['comm_delay'][str(d)][proto]['fidelity_mean'] for d in delays]
    stds = [results['comm_delay'][str(d)][proto]['fidelity_std'] for d in delays]
    ax.errorbar(delays, means, yerr=stds, marker=markers[proto],
                color=colors[proto], label=proto, capsize=3, linewidth=1.5)
ax.axhline(y=0.65, color='gray', linestyle='--', alpha=0.5, label='Threshold')
ax.set_xlabel('Communication Delay (time slots)')
ax.set_ylabel('Average Delivery Fidelity')
ax.set_title('(b) Fidelity vs Delay')
ax.legend(loc='lower left', fontsize=8)
ax.set_ylim(0.4, 0.85)
ax.grid(True, alpha=0.3)

# (c) Violation Rate vs Delay
ax = axes[2]
for proto in protocols:
    means = [results['comm_delay'][str(d)][proto]['violation_mean'] for d in delays]
    stds = [results['comm_delay'][str(d)][proto]['violation_std'] for d in delays]
    ax.errorbar(delays, means, yerr=stds, marker=markers[proto],
                color=colors[proto], label=proto, capsize=3, linewidth=1.5)
ax.set_xlabel('Communication Delay (time slots)')
ax.set_ylabel('Fidelity Violation Rate')
ax.set_title('(c) Violations vs Delay')
ax.legend(loc='upper left', fontsize=8)
ax.set_ylim(-0.05, 1.0)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(_os.path.join(FIGURES_DIR, "exp_comm_delay.pdf"), bbox_inches='tight')
plt.savefig(_os.path.join(FIGURES_DIR, "exp_comm_delay.png"), bbox_inches='tight')
print("Saved: exp_comm_delay.pdf/png")

# ============================================================
# Figure 2: Real Topologies Comparison (Bar chart)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

topos = ['NSFNET', 'COST-239', 'SURFnet']
protocols_bar = ['D-AoE-ARS', 'D-Greedy', 'SP', 'Q-CAST']
x = np.arange(len(topos))
width = 0.2

# (a) Delivery Ratio
ax = axes[0]
for i, proto in enumerate(protocols_bar):
    means = [results['topologies'][t][proto]['delivery_mean'] for t in topos]
    stds = [results['topologies'][t][proto]['delivery_std'] for t in topos]
    ax.bar(x + i*width - 1.5*width, means, width, yerr=stds,
           label=proto, color=colors[proto], capsize=3, alpha=0.85)
ax.set_xlabel('Topology')
ax.set_ylabel('Delivery Ratio')
ax.set_title('(a) Throughput on Real Topologies')
ax.set_xticks(x)
ax.set_xticklabels(topos)
ax.legend(fontsize=8)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis='y')

# (b) Fidelity
ax = axes[1]
for i, proto in enumerate(protocols_bar):
    means = [results['topologies'][t][proto]['fidelity_mean'] for t in topos]
    stds = [results['topologies'][t][proto]['fidelity_std'] for t in topos]
    ax.bar(x + i*width - 1.5*width, means, width, yerr=stds,
           label=proto, color=colors[proto], capsize=3, alpha=0.85)
ax.axhline(y=0.65, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Topology')
ax.set_ylabel('Average Fidelity')
ax.set_title('(b) Fidelity on Real Topologies')
ax.set_xticks(x)
ax.set_xticklabels(topos)
ax.legend(fontsize=8)
ax.set_ylim(0.3, 0.9)
ax.grid(True, alpha=0.3, axis='y')

# (c) Violation Rate
ax = axes[2]
for i, proto in enumerate(protocols_bar):
    means = [results['topologies'][t][proto]['violation_mean'] for t in topos]
    stds = [results['topologies'][t][proto]['violation_std'] for t in topos]
    ax.bar(x + i*width - 1.5*width, means, width, yerr=stds,
           label=proto, color=colors[proto], capsize=3, alpha=0.85)
ax.set_xlabel('Topology')
ax.set_ylabel('Violation Rate')
ax.set_title('(c) Violations on Real Topologies')
ax.set_xticks(x)
ax.set_xticklabels(topos)
ax.legend(fontsize=8)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(_os.path.join(FIGURES_DIR, "exp_topologies.pdf"), bbox_inches='tight')
plt.savefig(_os.path.join(FIGURES_DIR, "exp_topologies.png"), bbox_inches='tight')
print("Saved: exp_topologies.pdf/png")

# ============================================================
# Figure 3: QoS Comparison
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

protocols_qos = ['D-AoE-ARS', 'D-Greedy', 'SP']
metrics = ['delivery_mean', 'fidelity_mean', 'violation_mean']
metric_labels = ['Delivery Ratio', 'Avg Fidelity', 'Violation Rate']

x = np.arange(len(metrics))
width = 0.25

for i, proto in enumerate(protocols_qos):
    vals = [results['qos'][proto][m] for m in metrics]
    ax.bar(x + i*width - width, vals, width, label=proto, 
           color=colors[proto], alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylabel('Value')
ax.set_title('Multi-class QoS Performance (d_cc=5)')
ax.legend()
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(_os.path.join(FIGURES_DIR, "exp_qos.pdf"), bbox_inches='tight')
plt.savefig(_os.path.join(FIGURES_DIR, "exp_qos.png"), bbox_inches='tight')
print("Saved: exp_qos.pdf/png")

# ============================================================
# Figure 4: Computational Overhead
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

overhead = results['overhead']
sizes = sorted([int(k) for k in overhead.keys()])
protocols_oh = ['D-AoE-ARS', 'D-Greedy', 'SP']

for proto in protocols_oh:
    times = [overhead[str(s)][proto] for s in sizes]
    ax.plot(sizes, times, marker=markers[proto], color=colors[proto],
            label=proto, linewidth=1.5)

ax.set_xlabel('Network Size (nodes)')
ax.set_ylabel('Computation Time (seconds)')
ax.set_title('Computational Overhead vs Network Size')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(_os.path.join(FIGURES_DIR, "exp_overhead.pdf"), bbox_inches='tight')
plt.savefig(_os.path.join(FIGURES_DIR, "exp_overhead.png"), bbox_inches='tight')
print("Saved: exp_overhead.pdf/png")

print("\nAll distributed figures generated successfully!")
