import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO_ROOT = _os.path.dirname(_HERE)
EXPERIMENTS_DIR = _os.path.join(REPO_ROOT, "experiments")
FIGURES_DIR = _os.path.join(REPO_ROOT, "figures")
_os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
_os.makedirs(FIGURES_DIR, exist_ok=True)

#!/usr/bin/env python3
"""
Hardware Platform Comparison Experiment.

Maps real quantum hardware parameters to simulation time slots and compares
D-AoE-ARS performance across three platforms:
  1. NV Centers in Diamond: T_coh~10ms, low generation rate
  2. Trapped Ions: T_coh~1s, moderate generation rate  
  3. Neutral Atoms (Rydberg): T_coh~1s, high generation rate

Physical-to-simulation mapping:
  - One time slot = one entanglement generation attempt cycle
  - NV centers: cycle~1ms, T_coh=10ms -> 10 slots
  - Trapped ions: cycle~10ms, T_coh=1s -> 100 slots
  - Neutral atoms: cycle~1ms, T_coh=1s -> 1000 slots
"""
import sys
import os
import json
import numpy as np
from collections import defaultdict

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from network_model import (QuantumNetwork, QuantumNode, QuantumLink,
                           fidelity_decay, chain_fidelity)
from topologies import create_nsfnet


# ─── Hardware Platform Definitions ───────────────────────────────────────────

PLATFORMS = {
    'NV Centers': {
        'T_coh': 10,          # 10ms / 1ms per slot = 10 slots
        'p_e': 0.05,          # Low heralded entanglement probability
        'F0': 0.87,           # Initial fidelity (limited by photon loss)
        'M_v': 2,             # Few memory qubits per node
    },
    'Trapped Ions': {
        'T_coh': 100,         # 1s / 10ms per slot = 100 slots
        'p_e': 0.30,          # Moderate (photonic interconnect)
        'F0': 0.94,           # High gate fidelity
        'M_v': 4,             # Multiple ion qubits
    },
    'Neutral Atoms': {
        'T_coh': 1000,        # 1s / 1ms per slot = 1000 slots
        'p_e': 0.60,          # High (cavity-enhanced)
        'F0': 0.92,           # Good initial fidelity
        'M_v': 8,             # Many atomic qubits in array
    },
}

NUM_SEEDS = 15
NUM_SLOTS = 300
F_TH = 0.70
NUM_FLOWS = 5


# ─── Simulation Functions (same pattern as exp_convergence.py) ────────────────

def simulate_daoe_ars(network, T_coh, M_v, num_slots, flows, rng, tau=0.7):
    """D-AoE-ARS: refresh + admission control + freshest-first."""
    link_pairs = defaultdict(list)
    total_delivered = 0
    total_valid = 0
    total_violations = 0
    total_attempts = 0
    all_fidelities = []

    for t in range(num_slots):
        active_flows = min(len(flows), 1 + t // 10)

        # Phase 1: Refresh - discard old pairs
        for link_id in list(link_pairs.keys()):
            link_pairs[link_id] = [
                (gt, f0) for gt, f0 in link_pairs[link_id]
                if (t - gt) / T_coh <= tau
            ]

        # Phase 2: Generate new pairs
        for link in network.links.values():
            if rng.random() < link.success_prob:
                if len(link_pairs[link.link_id]) < M_v:
                    link_pairs[link.link_id].append((t, link.initial_fidelity))

        # Phase 3: Route with admission control
        for fi in range(active_flows):
            src, dst = flows[fi]
            paths = network.k_shortest_paths(src, dst, k=1)
            if not paths:
                continue
            path = paths[0]
            total_attempts += 1

            # Check link availability
            path_links = []
            all_available = True
            for i in range(len(path) - 1):
                link = network.get_link(path[i], path[i+1])
                if link is None or not link_pairs[link.link_id]:
                    all_available = False
                    break
                path_links.append(link)

            if not all_available:
                continue

            # Select freshest pairs, compute e2e fidelity
            link_fidelities = []
            pairs_to_consume = []
            for link in path_links:
                pairs = link_pairs[link.link_id]
                best_idx = min(range(len(pairs)), key=lambda i: t - pairs[i][0])
                gt, f0 = pairs[best_idx]
                F_cur = fidelity_decay(f0, t - gt, T_coh)
                link_fidelities.append(F_cur)
                pairs_to_consume.append((link.link_id, best_idx))

            F_e2e = chain_fidelity(link_fidelities)

            if F_e2e >= F_TH:
                for link_id, idx in sorted(pairs_to_consume, key=lambda x: -x[1]):
                    link_pairs[link_id].pop(idx)
                total_delivered += 1
                total_valid += 1
                all_fidelities.append(F_e2e)

    return {
        'valid_throughput': total_valid / max(num_slots, 1),
        'fidelity': np.mean(all_fidelities) if all_fidelities else 0.5,
        'violation_rate': 0.0,  # By design
    }


def simulate_greedy(network, T_coh, M_v, num_slots, flows, rng):
    """Greedy: no refresh, FIFO pair selection, no admission control."""
    link_pairs = defaultdict(list)
    total_delivered = 0
    total_violations = 0
    all_fidelities = []

    for t in range(num_slots):
        active_flows = min(len(flows), 1 + t // 10)

        # Generate
        for link in network.links.values():
            if rng.random() < link.success_prob:
                if len(link_pairs[link.link_id]) < M_v:
                    link_pairs[link.link_id].append((t, link.initial_fidelity))

        # Route: FIFO (oldest first), no admission control
        for fi in range(active_flows):
            src, dst = flows[fi]
            paths = network.k_shortest_paths(src, dst, k=1)
            if not paths:
                continue
            path = paths[0]

            path_links = []
            all_available = True
            for i in range(len(path) - 1):
                link = network.get_link(path[i], path[i+1])
                if link is None or not link_pairs[link.link_id]:
                    all_available = False
                    break
                path_links.append(link)

            if not all_available:
                continue

            # Use oldest pair (FIFO)
            link_fidelities = []
            pairs_to_consume = []
            for link in path_links:
                pairs = link_pairs[link.link_id]
                gt, f0 = pairs[0]  # oldest
                F_cur = fidelity_decay(f0, t - gt, T_coh)
                link_fidelities.append(F_cur)
                pairs_to_consume.append((link.link_id, 0))

            F_e2e = chain_fidelity(link_fidelities)

            # Deliver regardless of fidelity
            for link_id, idx in sorted(pairs_to_consume, key=lambda x: -x[1]):
                link_pairs[link_id].pop(idx)
            total_delivered += 1
            all_fidelities.append(F_e2e)
            if F_e2e < F_TH:
                total_violations += 1

    valid = total_delivered - total_violations
    return {
        'valid_throughput': max(valid, 0) / max(num_slots, 1),
        'fidelity': np.mean(all_fidelities) if all_fidelities else 0.5,
        'violation_rate': total_violations / max(total_delivered, 1),
    }


def simulate_qcast(network, T_coh, M_v, num_slots, flows, rng, K=3):
    """Q-CAST: K shortest paths, pick best available, no admission control."""
    link_pairs = defaultdict(list)
    total_delivered = 0
    total_violations = 0
    all_fidelities = []

    for t in range(num_slots):
        active_flows = min(len(flows), 1 + t // 10)

        # Generate
        for link in network.links.values():
            if rng.random() < link.success_prob:
                if len(link_pairs[link.link_id]) < M_v:
                    link_pairs[link.link_id].append((t, link.initial_fidelity))

        # Route: K shortest paths, pick one with most available links
        for fi in range(active_flows):
            src, dst = flows[fi]
            paths = network.k_shortest_paths(src, dst, k=K)
            if not paths:
                continue

            # Pick path with most available links
            best_path = None
            best_score = -1
            for path in paths:
                score = 0
                for i in range(len(path) - 1):
                    link = network.get_link(path[i], path[i+1])
                    if link and link_pairs[link.link_id]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_path = path

            if best_path is None:
                continue
            path = best_path

            path_links = []
            all_available = True
            for i in range(len(path) - 1):
                link = network.get_link(path[i], path[i+1])
                if link is None or not link_pairs[link.link_id]:
                    all_available = False
                    break
                path_links.append(link)

            if not all_available:
                continue

            # Use freshest pair per link (Q-CAST is quality-aware)
            link_fidelities = []
            pairs_to_consume = []
            for link in path_links:
                pairs = link_pairs[link.link_id]
                best_idx = min(range(len(pairs)), key=lambda i: t - pairs[i][0])
                gt, f0 = pairs[best_idx]
                F_cur = fidelity_decay(f0, t - gt, T_coh)
                link_fidelities.append(F_cur)
                pairs_to_consume.append((link.link_id, best_idx))

            F_e2e = chain_fidelity(link_fidelities)

            # Deliver regardless
            for link_id, idx in sorted(pairs_to_consume, key=lambda x: -x[1]):
                link_pairs[link_id].pop(idx)
            total_delivered += 1
            all_fidelities.append(F_e2e)
            if F_e2e < F_TH:
                total_violations += 1

    valid = total_delivered - total_violations
    return {
        'valid_throughput': max(valid, 0) / max(num_slots, 1),
        'fidelity': np.mean(all_fidelities) if all_fidelities else 0.5,
        'violation_rate': total_violations / max(total_delivered, 1),
    }


def simulate_sp(network, T_coh, M_v, num_slots, flows, rng):
    """Shortest Path: single shortest path, FIFO, no admission control."""
    link_pairs = defaultdict(list)
    total_delivered = 0
    total_violations = 0
    all_fidelities = []

    for t in range(num_slots):
        active_flows = min(len(flows), 1 + t // 10)

        # Generate
        for link in network.links.values():
            if rng.random() < link.success_prob:
                if len(link_pairs[link.link_id]) < M_v:
                    link_pairs[link.link_id].append((t, link.initial_fidelity))

        # Route: single shortest path, FIFO
        for fi in range(active_flows):
            src, dst = flows[fi]
            paths = network.k_shortest_paths(src, dst, k=1)
            if not paths:
                continue
            path = paths[0]

            path_links = []
            all_available = True
            for i in range(len(path) - 1):
                link = network.get_link(path[i], path[i+1])
                if link is None or not link_pairs[link.link_id]:
                    all_available = False
                    break
                path_links.append(link)

            if not all_available:
                continue

            # FIFO
            link_fidelities = []
            pairs_to_consume = []
            for link in path_links:
                pairs = link_pairs[link.link_id]
                gt, f0 = pairs[0]
                F_cur = fidelity_decay(f0, t - gt, T_coh)
                link_fidelities.append(F_cur)
                pairs_to_consume.append((link.link_id, 0))

            F_e2e = chain_fidelity(link_fidelities)

            for link_id, idx in sorted(pairs_to_consume, key=lambda x: -x[1]):
                link_pairs[link_id].pop(idx)
            total_delivered += 1
            all_fidelities.append(F_e2e)
            if F_e2e < F_TH:
                total_violations += 1

    valid = total_delivered - total_violations
    return {
        'valid_throughput': max(valid, 0) / max(num_slots, 1),
        'fidelity': np.mean(all_fidelities) if all_fidelities else 0.5,
        'violation_rate': total_violations / max(total_delivered, 1),
    }


# ─── Main Experiment ─────────────────────────────────────────────────────────

def run_experiment():
    print("=" * 60)
    print("  Hardware Platform Comparison Experiment")
    print("=" * 60)

    all_results = {}

    for platform_name, params in PLATFORMS.items():
        clean_name = platform_name.replace('\n', ' ')
        print(f"\n{'─' * 60}")
        print(f"  Platform: {clean_name}")
        print(f"  T_coh={params['T_coh']} slots, p_e={params['p_e']}, "
              f"F0={params['F0']}, M_v={params['M_v']}")
        print(f"{'─' * 60}")

        seed_results = {p: [] for p in ['D-AoE-ARS', 'D-Greedy', 'Q-CAST', 'SP']}

        for seed_idx in range(NUM_SEEDS):
            seed = seed_idx * 42 + 7
            rng = np.random.default_rng(seed)

            # Create network with platform-specific parameters
            network = create_nsfnet(num_memory_slots=params['M_v'],
                                    coherence_time=params['T_coh'])

            # Override link parameters for this platform
            for link in network.links.values():
                link.success_prob = params['p_e'] * (0.8 + 0.4 * rng.random())
                link.initial_fidelity = min(params['F0'] * (0.95 + 0.05 * rng.random()), 0.99)

            # Generate flows (same for all protocols in this seed)
            node_ids = list(network.nodes.keys())
            flows = []
            for _ in range(NUM_FLOWS):
                s, d = rng.choice(node_ids, size=2, replace=False)
                flows.append((s, d))

            # Run each protocol
            r = simulate_daoe_ars(network, params['T_coh'], params['M_v'],
                                  NUM_SLOTS, flows, np.random.default_rng(seed))
            seed_results['D-AoE-ARS'].append(r)

            r = simulate_greedy(network, params['T_coh'], params['M_v'],
                               NUM_SLOTS, flows, np.random.default_rng(seed))
            seed_results['D-Greedy'].append(r)

            r = simulate_qcast(network, params['T_coh'], params['M_v'],
                              NUM_SLOTS, flows, np.random.default_rng(seed))
            seed_results['Q-CAST'].append(r)

            r = simulate_sp(network, params['T_coh'], params['M_v'],
                           NUM_SLOTS, flows, np.random.default_rng(seed))
            seed_results['SP'].append(r)

            if (seed_idx + 1) % 5 == 0:
                print(f"  Completed {seed_idx+1}/{NUM_SEEDS} seeds")

        # Aggregate
        agg = {}
        for proto, results_list in seed_results.items():
            metrics = {}
            for metric in ['valid_throughput', 'fidelity', 'violation_rate']:
                values = [r[metric] for r in results_list]
                metrics[metric] = {'mean': float(np.mean(values)),
                                   'std': float(np.std(values))}
            agg[proto] = metrics

        all_results[clean_name] = agg

        # Print summary
        print(f"\n  {'Protocol':<12} {'Valid Tput':>12} {'Fidelity':>10} {'Violation':>10}")
        for proto, m in agg.items():
            print(f"  {proto:<12} "
                  f"{m['valid_throughput']['mean']:>8.3f}±{m['valid_throughput']['std']:.3f} "
                  f"{m['fidelity']['mean']:>8.3f} "
                  f"{m['violation_rate']['mean']:>8.3f}")

    # Save
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    with open(_os.path.join(EXPERIMENTS_DIR, "hardware_platform_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved.")

    # Generate figure
    generate_figure(all_results)
    return all_results


def generate_figure(results):
    """Generate grouped bar chart comparing platforms."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'serif',
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 150,
    })

    platforms = list(results.keys())
    protocols = ['D-AoE-ARS', 'D-Greedy', 'Q-CAST', 'SP']
    colors = {
        'D-AoE-ARS': '#2E86C1',
        'D-Greedy': '#E74C3C',
        'Q-CAST': '#27AE60',
        'SP': '#8E44AD',
    }
    hatches = {
        'D-AoE-ARS': '',
        'D-Greedy': '//',
        'Q-CAST': '\\\\',
        'SP': 'xx',
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.2))

    metrics_info = [
        ('valid_throughput', 'Valid Throughput\n(pairs/slot)', '(a)'),
        ('fidelity', 'Avg. Delivered Fidelity', '(b)'),
        ('violation_rate', 'Fidelity Violation Rate', '(c)'),
    ]

    bar_width = 0.18
    x = np.arange(len(platforms))

    for ax_idx, (metric, ylabel, panel_label) in enumerate(metrics_info):
        ax = axes[ax_idx]

        for i, proto in enumerate(protocols):
            means = [results[p][proto][metric]['mean'] for p in platforms]
            stds = [results[p][proto][metric]['std'] for p in platforms]

            ax.bar(x + (i - 1.5) * bar_width, means, bar_width,
                   yerr=stds, capsize=2,
                   color=colors[proto], alpha=0.85,
                   hatch=hatches[proto], edgecolor='white', linewidth=0.5,
                   error_kw={'linewidth': 0.8})

        # Fidelity threshold line
        if metric == 'fidelity':
            ax.axhline(y=F_TH, color='red', linestyle='--', linewidth=0.8,
                      alpha=0.7)
            ax.set_ylim(0.4, 1.0)

        if metric == 'violation_rate':
            ax.set_ylim(0, 1.05)

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        # Use abbreviated labels to avoid overlap
        xlabels = ['NV\n($T_c$=10)', 'Ion\n($T_c$=100)', 'Atom\n($T_c$=1k)']
        ax.set_xticklabels(xlabels, fontsize=8, ha='center', linespacing=0.9)
        ax.set_title(panel_label, fontsize=10, fontweight='bold', loc='left')
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend
    legend_elements = [Patch(facecolor=colors[p], alpha=0.85, hatch=hatches[p],
                            edgecolor='white', label=p) for p in protocols]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4,
              bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.91], w_pad=2.5)

    fig.savefig(_os.path.join(FIGURES_DIR, "fig_hardware_platforms.pdf"),
                bbox_inches='tight', dpi=300)
    fig.savefig(_os.path.join(FIGURES_DIR, "fig_hardware_platforms.png"),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("Figure saved to figures/fig_hardware_platforms.pdf")


if __name__ == '__main__':
    run_experiment()
