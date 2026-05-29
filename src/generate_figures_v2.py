"""
Generate additional figures for heterogeneous and hotspot experiments.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'lines.markersize': 7,
})

PROTO_STYLES = {
    'aoe_ars': {'label': 'AoE-ARS (Ours)', 'color': '#d62728', 'marker': 'o', 'linestyle': '-'},
    'shortest_path': {'label': 'Shortest Path', 'color': '#1f77b4', 'marker': 's', 'linestyle': '--'},
    'fidelity_aware': {'label': 'Fidelity-Aware', 'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},
    'greedy': {'label': 'Greedy', 'color': '#ff7f0e', 'marker': 'D', 'linestyle': ':'},
}
PROTOCOLS = ['aoe_ars', 'shortest_path', 'fidelity_aware', 'greedy']

results_path = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'results_additional.json')
with open(results_path, 'r') as f:
    all_results = json.load(f)

fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(fig_dir, exist_ok=True)


def plot_exp6():
    """Heterogeneous network."""
    data = all_results['exp6_heterogeneous']
    x = data['arrival_rates']
    results = data['results']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        ax.plot(x, results[proto]['throughput'], label=style['label'],
                color=style['color'], marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Request Arrival Rate')
    ax.set_ylabel('Delivery Ratio')
    ax.set_title('(a) Throughput — Heterogeneous Network')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    ax = axes[0, 1]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        ax.plot(x, results[proto]['fidelity'], label=style['label'],
                color=style['color'], marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Request Arrival Rate')
    ax.set_ylabel('Average Delivery Fidelity')
    ax.set_title('(b) Fidelity — Heterogeneous Network')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.65, color='gray', linestyle='--', alpha=0.5)

    ax = axes[1, 0]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        ax.plot(x, results[proto]['aoe'], label=style['label'],
                color=style['color'], marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Request Arrival Rate')
    ax.set_ylabel('Average AoE (time slots)')
    ax.set_title('(c) Age-of-Entanglement — Heterogeneous Network')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    ax = axes[1, 1]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        ax.plot(x, results[proto]['violation'], label=style['label'],
                color=style['color'], marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Request Arrival Rate')
    ax.set_ylabel('Fidelity Violation Rate')
    ax.set_title('(d) Violations — Heterogeneous Network')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'exp6_heterogeneous.pdf'))
    plt.savefig(os.path.join(fig_dir, 'exp6_heterogeneous.png'))
    plt.close()
    print("  Figure: exp6_heterogeneous.pdf")


def plot_exp7():
    """Hotspot traffic."""
    data = all_results['exp7_hotspot']
    x = data['arrival_rates']
    results = data['results']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        ax.plot(x, results[proto]['throughput'], label=style['label'],
                color=style['color'], marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Request Arrival Rate')
    ax.set_ylabel('Delivery Ratio')
    ax.set_title('(a) Throughput — Hotspot Traffic')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    ax = axes[0, 1]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        ax.plot(x, results[proto]['fidelity'], label=style['label'],
                color=style['color'], marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Request Arrival Rate')
    ax.set_ylabel('Average Delivery Fidelity')
    ax.set_title('(b) Fidelity — Hotspot Traffic')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.65, color='gray', linestyle='--', alpha=0.5)

    ax = axes[1, 0]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        ax.plot(x, results[proto]['aoe'], label=style['label'],
                color=style['color'], marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Request Arrival Rate')
    ax.set_ylabel('Average AoE (time slots)')
    ax.set_title('(c) Age-of-Entanglement — Hotspot Traffic')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    ax = axes[1, 1]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        ax.plot(x, results[proto]['violation'], label=style['label'],
                color=style['color'], marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Request Arrival Rate')
    ax.set_ylabel('Fidelity Violation Rate')
    ax.set_title('(d) Violations — Hotspot Traffic')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'exp7_hotspot.pdf'))
    plt.savefig(os.path.join(fig_dir, 'exp7_hotspot.png'))
    plt.close()
    print("  Figure: exp7_hotspot.pdf")


def plot_combined_summary():
    """Combined summary: bar chart at key operating points."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Exp7 hotspot at rate=0.5
    throughput = [963/964, 385/570, 447/620, 963/964]
    fidelity = [0.804, 0.773, 0.745, 0.809]
    aoe = [2.7, 31.9, 42.3, 2.7]

    x_pos = np.arange(4)
    colors = [PROTO_STYLES[p]['color'] for p in PROTOCOLS]
    labels = [PROTO_STYLES[p]['label'] for p in PROTOCOLS]

    ax = axes[0]
    ax.bar(x_pos, throughput, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Delivery Ratio')
    ax.set_title('(a) Throughput (Hotspot, $\\lambda$=0.5)')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x_pos, fidelity, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Average Fidelity')
    ax.set_title('(b) Fidelity (Hotspot, $\\lambda$=0.5)')
    ax.axhline(y=0.65, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylim([0.6, 0.85])
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2]
    ax.bar(x_pos, aoe, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Average AoE (time slots)')
    ax.set_title('(c) AoE (Hotspot, $\\lambda$=0.5)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'summary_hotspot.pdf'))
    plt.savefig(os.path.join(fig_dir, 'summary_hotspot.png'))
    plt.close()
    print("  Figure: summary_hotspot.pdf")


if __name__ == "__main__":
    print("Generating additional figures...")
    plot_exp6()
    plot_exp7()
    plot_combined_summary()
    print(f"All figures saved to {fig_dir}/")
