"""
Figure Generation for AoE-ARS Paper
=====================================
Publication-quality plots using matplotlib.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

# Style settings for publication
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

# Protocol display names and styles
PROTO_STYLES = {
    'aoe_ars': {'label': 'AoE-ARS (Ours)', 'color': '#d62728', 'marker': 'o', 'linestyle': '-'},
    'shortest_path': {'label': 'Shortest Path', 'color': '#1f77b4', 'marker': 's', 'linestyle': '--'},
    'fidelity_aware': {'label': 'Fidelity-Aware', 'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},
    'greedy': {'label': 'Greedy', 'color': '#ff7f0e', 'marker': 'D', 'linestyle': ':'},
}

PROTOCOLS = ['aoe_ars', 'shortest_path', 'fidelity_aware', 'greedy']

# Load results
results_path = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'results.json')
with open(results_path, 'r') as f:
    all_results = json.load(f)

fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(fig_dir, exist_ok=True)


def plot_experiment_1():
    """Congestion stress test — 4 subplots."""
    data = all_results['exp1_congestion']
    x = data['arrival_rates']
    results = data['results']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) Throughput (delivery ratio)
    ax = axes[0, 0]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        y = results[proto]['throughput']
        ax.plot(x, y, label=style['label'], color=style['color'],
                marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Request Arrival Rate')
    ax.set_ylabel('Delivery Ratio')
    ax.set_title('(a) Throughput')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # (b) Average Fidelity
    ax = axes[0, 1]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        y = results[proto]['fidelity']
        ax.plot(x, y, label=style['label'], color=style['color'],
                marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Request Arrival Rate')
    ax.set_ylabel('Average Delivery Fidelity')
    ax.set_title('(b) Fidelity')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.65, color='gray', linestyle='--', alpha=0.5, label='Threshold')

    # (c) Average AoE
    ax = axes[1, 0]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        y = results[proto]['aoe']
        ax.plot(x, y, label=style['label'], color=style['color'],
                marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Request Arrival Rate')
    ax.set_ylabel('Average AoE (time slots)')
    ax.set_title('(c) Age-of-Entanglement')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # (d) Fidelity Violation Rate
    ax = axes[1, 1]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        y = results[proto]['violation']
        ax.plot(x, y, label=style['label'], color=style['color'],
                marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Request Arrival Rate')
    ax.set_ylabel('Fidelity Violation Rate')
    ax.set_title('(d) Violation Rate')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'exp1_congestion.pdf'))
    plt.savefig(os.path.join(fig_dir, 'exp1_congestion.png'))
    plt.close()
    print("  Figure 1: exp1_congestion.pdf")


def plot_experiment_2():
    """Coherence time sensitivity."""
    data = all_results['exp2_coherence']
    x = data['coherence_times']
    results = data['results']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) Fidelity vs T_coh
    ax = axes[0]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        y = results[proto]['fidelity']
        ax.plot(x, y, label=style['label'], color=style['color'],
                marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Coherence Time $T_{coh}$ (time slots)')
    ax.set_ylabel('Average Delivery Fidelity')
    ax.set_title('(a) Fidelity vs. Coherence Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # (b) Violation rate vs T_coh
    ax = axes[1]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        y = results[proto]['violation']
        ax.plot(x, y, label=style['label'], color=style['color'],
                marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Coherence Time $T_{coh}$ (time slots)')
    ax.set_ylabel('Fidelity Violation Rate')
    ax.set_title('(b) Violations vs. Coherence Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'exp2_coherence.pdf'))
    plt.savefig(os.path.join(fig_dir, 'exp2_coherence.png'))
    plt.close()
    print("  Figure 2: exp2_coherence.pdf")


def plot_experiment_3():
    """Scalability."""
    data = all_results['exp3_scalability']
    x_labels = data['grid_sizes']
    x = [9, 16, 25, 36, 49]  # number of nodes
    results = data['results']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) Fidelity vs network size
    ax = axes[0]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        y = results[proto]['fidelity']
        ax.plot(x, y, label=style['label'], color=style['color'],
                marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Average Delivery Fidelity')
    ax.set_title('(a) Fidelity vs. Network Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)

    # (b) Latency vs network size
    ax = axes[1]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        y = results[proto]['latency']
        ax.plot(x, y, label=style['label'], color=style['color'],
                marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Average Latency (time slots)')
    ax.set_title('(b) Latency vs. Network Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'exp3_scalability.pdf'))
    plt.savefig(os.path.join(fig_dir, 'exp3_scalability.png'))
    plt.close()
    print("  Figure 3: exp3_scalability.pdf")


def plot_experiment_4():
    """Memory constraints."""
    data = all_results['exp4_memory']
    x = data['memory_sizes']
    results = data['results']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) Throughput vs memory
    ax = axes[0]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        y = results[proto]['throughput']
        ax.plot(x, y, label=style['label'], color=style['color'],
                marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Memory Slots per Node')
    ax.set_ylabel('Delivery Ratio')
    ax.set_title('(a) Throughput vs. Memory')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # (b) Violation rate vs memory
    ax = axes[1]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        y = results[proto]['violation']
        ax.plot(x, y, label=style['label'], color=style['color'],
                marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Memory Slots per Node')
    ax.set_ylabel('Fidelity Violation Rate')
    ax.set_title('(b) Violations vs. Memory')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'exp4_memory.pdf'))
    plt.savefig(os.path.join(fig_dir, 'exp4_memory.png'))
    plt.close()
    print("  Figure 4: exp4_memory.pdf")


def plot_experiment_5():
    """Link quality."""
    data = all_results['exp5_link_quality']
    x = data['success_probs']
    results = data['results']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) Throughput vs link quality
    ax = axes[0]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        y = results[proto]['throughput']
        ax.plot(x, y, label=style['label'], color=style['color'],
                marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Link Success Probability')
    ax.set_ylabel('Delivery Ratio')
    ax.set_title('(a) Throughput vs. Link Quality')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # (b) AoE vs link quality
    ax = axes[1]
    for proto in PROTOCOLS:
        style = PROTO_STYLES[proto]
        y = results[proto]['aoe']
        ax.plot(x, y, label=style['label'], color=style['color'],
                marker=style['marker'], linestyle=style['linestyle'])
    ax.set_xlabel('Link Success Probability')
    ax.set_ylabel('Average AoE (time slots)')
    ax.set_title('(b) AoE vs. Link Quality')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'exp5_link_quality.pdf'))
    plt.savefig(os.path.join(fig_dir, 'exp5_link_quality.png'))
    plt.close()
    print("  Figure 5: exp5_link_quality.pdf")


def plot_summary_bar():
    """Summary bar chart comparing protocols at key operating points."""
    # Key operating point: rate=0.6, memory=3, link_prob=0.1
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Data from experiments at stress points
    # Exp1 rate=0.6
    throughput_06 = [1203/1212, 113/469, 166/504, 1162/1164]
    fidelity_06 = [0.814, 0.729, 0.642, 0.828]
    aoe_06 = [4.5, 72.8, 132.8, 3.1]

    x_pos = np.arange(4)
    colors = [PROTO_STYLES[p]['color'] for p in PROTOCOLS]
    labels = [PROTO_STYLES[p]['label'] for p in PROTOCOLS]

    # (a) Throughput at rate=0.6
    ax = axes[0]
    bars = ax.bar(x_pos, throughput_06, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Delivery Ratio')
    ax.set_title('(a) Throughput (rate=0.6)')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    # (b) Fidelity at rate=0.6
    ax = axes[1]
    bars = ax.bar(x_pos, fidelity_06, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Average Fidelity')
    ax.set_title('(b) Fidelity (rate=0.6)')
    ax.axhline(y=0.65, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylim([0.5, 0.9])
    ax.grid(True, alpha=0.3, axis='y')

    # (c) AoE at rate=0.6
    ax = axes[2]
    bars = ax.bar(x_pos, aoe_06, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Average AoE (time slots)')
    ax.set_title('(c) AoE (rate=0.6)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'summary_bar.pdf'))
    plt.savefig(os.path.join(fig_dir, 'summary_bar.png'))
    plt.close()
    print("  Figure 6: summary_bar.pdf")


if __name__ == "__main__":
    print("Generating publication figures...")
    plot_experiment_1()
    plot_experiment_2()
    plot_experiment_3()
    plot_experiment_4()
    plot_experiment_5()
    plot_summary_bar()
    print(f"\nAll figures saved to {fig_dir}/")
