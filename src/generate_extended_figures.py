"""Generate figures for extended experiments (ToN quality)."""

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

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'legend.fontsize': 9,
    'figure.dpi': 150, 'font.family': 'serif'
})

colors = {'D-AoE-ARS': '#d62728', 'D-Greedy': '#ff7f0e', 'SP': '#1f77b4', 'Q-CAST': '#2ca02c'}
markers = {'D-AoE-ARS': 'o', 'D-Greedy': 'D', 'SP': 's', 'Q-CAST': '^'}

# Load all results
with open(_os.path.join(EXPERIMENTS_DIR, "short_coherence_results.json")) as f:
    sc_results = json.load(f)
with open(_os.path.join(EXPERIMENTS_DIR, "dynamic_failure_results.json")) as f:
    df_results = json.load(f)
with open(_os.path.join(EXPERIMENTS_DIR, "scalability_results.json")) as f:
    scale_results = json.load(f)

# ============================================================
# Figure 1: Short Coherence + Delay (KEY FIGURE for ToN)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

delays = [0, 2, 5, 10, 15, 20, 30]
protocols = ['D-AoE-ARS', 'D-Greedy', 'SP']

# (a) Delivery Ratio
ax = axes[0]
for proto in protocols:
    means = [sc_results[str(d)][proto]['del_m'] for d in delays]
    stds = [sc_results[str(d)][proto]['del_s'] for d in delays]
    ax.errorbar(delays, means, yerr=stds, marker=markers[proto],
                color=colors[proto], label=proto, capsize=3, linewidth=1.5, markersize=6)
ax.set_xlabel('Classical Communication Delay $d_{cc}$ (time slots)')
ax.set_ylabel('Delivery Ratio')
ax.set_title('(a) Throughput ($T_{coh}=20$)')
ax.legend(loc='lower right', fontsize=9)
ax.set_ylim(0.1, 0.7)
ax.grid(True, alpha=0.3)

# (b) Fidelity
ax = axes[1]
for proto in protocols:
    means = [sc_results[str(d)][proto]['fid_m'] for d in delays]
    stds = [sc_results[str(d)][proto]['fid_s'] for d in delays]
    ax.errorbar(delays, means, yerr=stds, marker=markers[proto],
                color=colors[proto], label=proto, capsize=3, linewidth=1.5, markersize=6)
ax.axhline(y=0.65, color='gray', linestyle='--', alpha=0.7, label='$F_{th}=0.65$')
ax.set_xlabel('Classical Communication Delay $d_{cc}$ (time slots)')
ax.set_ylabel('Average Delivery Fidelity')
ax.set_title('(b) Fidelity ($T_{coh}=20$)')
ax.legend(loc='lower left', fontsize=9)
ax.set_ylim(0.35, 0.85)
ax.grid(True, alpha=0.3)

# (c) Violation Rate
ax = axes[2]
for proto in protocols:
    means = [sc_results[str(d)][proto]['viol_m'] for d in delays]
    stds = [sc_results[str(d)][proto]['viol_s'] for d in delays]
    ax.errorbar(delays, means, yerr=stds, marker=markers[proto],
                color=colors[proto], label=proto, capsize=3, linewidth=1.5, markersize=6)
ax.set_xlabel('Classical Communication Delay $d_{cc}$ (time slots)')
ax.set_ylabel('Fidelity Violation Rate')
ax.set_title('(c) QoS Violations ($T_{coh}=20$)')
ax.legend(loc='center right', fontsize=9)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(_os.path.join(FIGURES_DIR, "fig_short_coherence.pdf"), bbox_inches='tight')
plt.savefig(_os.path.join(FIGURES_DIR, "fig_short_coherence.png"), bbox_inches='tight')
print("Saved: fig_short_coherence")

# ============================================================
# Figure 2: Dynamic Failures
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

pfails = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15]

# (a) Delivery
ax = axes[0]
for proto in protocols:
    means = [df_results[str(p)][proto]['del_m'] for p in pfails]
    stds = [df_results[str(p)][proto]['del_s'] for p in pfails]
    ax.errorbar(pfails, means, yerr=stds, marker=markers[proto],
                color=colors[proto], label=proto, capsize=3, linewidth=1.5, markersize=6)
ax.set_xlabel('Link Failure Probability $p_{fail}$')
ax.set_ylabel('Delivery Ratio')
ax.set_title('(a) Throughput under Dynamic Failures')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (b) Fidelity
ax = axes[1]
for proto in protocols:
    means = [df_results[str(p)][proto]['fid_m'] for p in pfails]
    stds = [df_results[str(p)][proto]['fid_s'] for p in pfails]
    ax.errorbar(pfails, means, yerr=stds, marker=markers[proto],
                color=colors[proto], label=proto, capsize=3, linewidth=1.5, markersize=6)
ax.axhline(y=0.65, color='gray', linestyle='--', alpha=0.7, label='$F_{th}=0.65$')
ax.set_xlabel('Link Failure Probability $p_{fail}$')
ax.set_ylabel('Average Delivery Fidelity')
ax.set_title('(b) Fidelity under Dynamic Failures')
ax.legend(fontsize=9)
ax.set_ylim(0.35, 0.85)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(_os.path.join(FIGURES_DIR, "fig_dynamic_failures.pdf"), bbox_inches='tight')
plt.savefig(_os.path.join(FIGURES_DIR, "fig_dynamic_failures.png"), bbox_inches='tight')
print("Saved: fig_dynamic_failures")

# ============================================================
# Figure 3: Scalability
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

sizes = [20, 50, 75, 100]

# (a) Delivery vs size
ax = axes[0]
for proto in protocols:
    means = [scale_results[str(n)][proto]['del_m'] for n in sizes]
    stds = [scale_results[str(n)][proto]['del_s'] for n in sizes]
    ax.errorbar(sizes, means, yerr=stds, marker=markers[proto],
                color=colors[proto], label=proto, capsize=3, linewidth=1.5, markersize=6)
ax.set_xlabel('Network Size (nodes)')
ax.set_ylabel('Delivery Ratio')
ax.set_title('(a) Throughput Scalability')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.5)

# (b) Fidelity vs size
ax = axes[1]
for proto in protocols:
    means = [scale_results[str(n)][proto]['fid_m'] for n in sizes]
    stds = [scale_results[str(n)][proto]['fid_s'] for n in sizes]
    ax.errorbar(sizes, means, yerr=stds, marker=markers[proto],
                color=colors[proto], label=proto, capsize=3, linewidth=1.5, markersize=6)
ax.axhline(y=0.65, color='gray', linestyle='--', alpha=0.7, label='$F_{th}=0.65$')
ax.set_xlabel('Network Size (nodes)')
ax.set_ylabel('Average Delivery Fidelity')
ax.set_title('(b) Fidelity Scalability')
ax.legend(fontsize=9)
ax.set_ylim(0.3, 0.85)
ax.grid(True, alpha=0.3)

# (c) Computation time vs size
ax = axes[2]
for proto in protocols:
    means = [scale_results[str(n)][proto]['time_m'] for n in sizes]
    ax.plot(sizes, means, marker=markers[proto], color=colors[proto],
            label=proto, linewidth=1.5, markersize=6)
ax.set_xlabel('Network Size (nodes)')
ax.set_ylabel('Computation Time (s)')
ax.set_title('(c) Computational Overhead')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(_os.path.join(FIGURES_DIR, "fig_scalability.pdf"), bbox_inches='tight')
plt.savefig(_os.path.join(FIGURES_DIR, "fig_scalability.png"), bbox_inches='tight')
print("Saved: fig_scalability")

# ============================================================
# Figure 4: Summary comparison table (for paper)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('off')

# Create summary table
col_labels = ['Metric', 'D-AoE-ARS', 'D-Greedy', 'SP']
table_data = [
    ['Delivery (T_coh=20, d=5)', '0.526', '0.358', '0.418'],
    ['Fidelity (T_coh=20, d=5)', '0.737', '0.470', '0.444'],
    ['Violation Rate', '0.000', '0.873', '0.886'],
    ['Delivery (N=100, d=5)', '0.361', '0.039', '0.152'],
    ['Fidelity (N=100, d=5)', '0.737', '0.603', '0.483'],
    ['Scalability (time, N=100)', '0.885s', '0.979s', '0.369s'],
]

table = ax.table(cellText=table_data, colLabels=col_labels, loc='center',
                 cellLoc='center', colColours=['#f0f0f0']*4)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Color the best values
for i in range(len(table_data)):
    table[i+1, 1].set_facecolor('#ffe0e0')  # AoE-ARS column highlighted

plt.title('Summary: D-AoE-ARS vs Baselines (Distributed Setting)', fontsize=12, pad=20)
plt.tight_layout()
plt.savefig(_os.path.join(FIGURES_DIR, "fig_summary_table.pdf"), bbox_inches='tight')
plt.savefig(_os.path.join(FIGURES_DIR, "fig_summary_table.png"), bbox_inches='tight')
print("Saved: fig_summary_table")

print("\nAll extended figures generated!")
