"""
Convergence Experiment: Time-series plot showing how D-AoE-ARS converges
to stable performance while baselines oscillate or degrade.

Tracks per-window metrics over 500 time slots:
- Effective throughput (delivered pairs with F >= F_th)
- Average delivered fidelity
- Cumulative violation rate

Uses NSFNET topology with 5 concurrent flows, 10 seeds.
Generates fig_convergence.pdf for the paper.
"""

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO_ROOT = _os.path.dirname(_HERE)
EXPERIMENTS_DIR = _os.path.join(REPO_ROOT, "experiments")
FIGURES_DIR = _os.path.join(REPO_ROOT, "figures")
_os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
_os.makedirs(FIGURES_DIR, exist_ok=True)

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from network_model import (QuantumNetwork, QuantumNode, QuantumLink, 
                           MemorySlot, SlotStatus, Request, RequestStatus,
                           fidelity_decay, chain_fidelity)
from topologies import create_nsfnet


class ConvergenceSimulator:
    """
    Simplified time-slot simulator for convergence analysis.
    Tracks per-slot metrics for time-series plotting.
    """
    
    def __init__(self, network, coherence_time=50.0, fidelity_threshold=0.7,
                 num_flows=5, seed=42):
        self.network = network
        self.T_coh = coherence_time
        self.F_th = fidelity_threshold
        self.num_flows = num_flows
        self.rng = np.random.default_rng(seed)
        
        # Set coherence time for all nodes
        for node in self.network.nodes.values():
            node.coherence_time = coherence_time
        
        # Generate random source-destination pairs
        node_ids = list(self.network.nodes.keys())
        self.flows = []
        for _ in range(num_flows):
            s, d = self.rng.choice(node_ids, size=2, replace=False)
            self.flows.append((s, d))
    
    def simulate_daoe_ars(self, num_slots=500, tau=0.7, alpha=5.0):
        """Simulate D-AoE-ARS with per-slot tracking."""
        metrics = {'throughput': [], 'fidelity': [], 'violations': []}
        
        # State: per-link stored pairs with generation times
        # Cold start: all memory empty
        link_pairs = defaultdict(list)  # link_id -> [(gen_time, F0)]
        
        total_delivered = 0
        total_violations = 0
        total_attempts = 0
        
        # Arrival rate ramp-up: requests increase over time (network warming up)
        for t in range(num_slots):
            slot_delivered = 0
            slot_fidelities = []
            slot_violations = 0
            
            # Dynamic load: flows arrive gradually (cold start effect)
            active_flows = min(self.num_flows, 1 + t // 15)
            
            # Phase 1: Refresh - discard old pairs
            for link_id in list(link_pairs.keys()):
                link_pairs[link_id] = [
                    (gt, f0) for gt, f0 in link_pairs[link_id]
                    if (t - gt) / self.T_coh <= tau
                ]
            
            # Phase 2: Generate new pairs on each link
            for link in self.network.links.values():
                # Each link attempts generation
                if self.rng.random() < link.success_prob:
                    if len(link_pairs[link.link_id]) < 4:  # memory limit
                        link_pairs[link.link_id].append((t, link.initial_fidelity))
            
            # Phase 3: Route & Swap for each active flow
            for fi in range(active_flows):
                src, dst = self.flows[fi]
                # Find path
                paths = self.network.k_shortest_paths(src, dst, k=1)
                if not paths:
                    continue
                path = paths[0]
                total_attempts += 1
                
                # Check if all links on path have pairs
                path_links = []
                all_available = True
                for i in range(len(path) - 1):
                    link = self.network.get_link(path[i], path[i+1])
                    if link is None or not link_pairs[link.link_id]:
                        all_available = False
                        break
                    path_links.append(link)
                
                if not all_available:
                    continue
                
                # Compute end-to-end fidelity using freshest pairs (AoE-aware)
                link_fidelities = []
                pairs_to_consume = []
                for link in path_links:
                    # Select freshest pair (lowest AoE)
                    pairs = link_pairs[link.link_id]
                    best_idx = min(range(len(pairs)), key=lambda i: t - pairs[i][0])
                    gt, f0 = pairs[best_idx]
                    age = t - gt
                    F_current = fidelity_decay(f0, age, self.T_coh)
                    link_fidelities.append(F_current)
                    pairs_to_consume.append((link.link_id, best_idx))
                
                # Admission control: predict e2e fidelity
                F_e2e = chain_fidelity(link_fidelities)
                
                if F_e2e >= self.F_th:
                    # Deliver: consume pairs
                    for link_id, idx in sorted(pairs_to_consume, key=lambda x: -x[1]):
                        link_pairs[link_id].pop(idx)
                    slot_delivered += 1
                    slot_fidelities.append(F_e2e)
                    total_delivered += 1
                # else: hold request (admission control rejects)
            
            # Record metrics
            metrics['throughput'].append(slot_delivered)
            metrics['fidelity'].append(np.mean(slot_fidelities) if slot_fidelities else np.nan)
            metrics['violations'].append(0)  # D-AoE-ARS: always 0 by design
        
        return metrics
    
    def simulate_greedy(self, num_slots=500):
        """Simulate Greedy baseline (no AoE awareness, no admission control)."""
        metrics = {'throughput': [], 'fidelity': [], 'violations': []}
        
        link_pairs = defaultdict(list)
        
        for t in range(num_slots):
            slot_delivered = 0
            slot_fidelities = []
            slot_violations = 0
            
            # Dynamic load (same ramp-up as D-AoE-ARS)
            active_flows = min(self.num_flows, 1 + t // 15)
            
            # Generate (same as D-AoE-ARS)
            for link in self.network.links.values():
                if self.rng.random() < link.success_prob:
                    if len(link_pairs[link.link_id]) < 4:
                        link_pairs[link.link_id].append((t, link.initial_fidelity))
            
            # NO refresh - greedy keeps all pairs
            # Route: use oldest pair first (FIFO, no AoE awareness)
            for fi in range(active_flows):
                src, dst = self.flows[fi]
                paths = self.network.k_shortest_paths(src, dst, k=1)
                if not paths:
                    continue
                path = paths[0]
                
                path_links = []
                all_available = True
                for i in range(len(path) - 1):
                    link = self.network.get_link(path[i], path[i+1])
                    if link is None or not link_pairs[link.link_id]:
                        all_available = False
                        break
                    path_links.append(link)
                
                if not all_available:
                    continue
                
                # Use oldest pair (FIFO - no AoE awareness)
                link_fidelities = []
                pairs_to_consume = []
                for link in path_links:
                    pairs = link_pairs[link.link_id]
                    gt, f0 = pairs[0]  # oldest
                    age = t - gt
                    F_current = fidelity_decay(f0, age, self.T_coh)
                    link_fidelities.append(F_current)
                    pairs_to_consume.append(link.link_id)
                
                F_e2e = chain_fidelity(link_fidelities)
                
                # No admission control - always deliver
                for link_id in pairs_to_consume:
                    link_pairs[link_id].pop(0)
                slot_delivered += 1
                slot_fidelities.append(F_e2e)
                if F_e2e < self.F_th:
                    slot_violations += 1
            
            metrics['throughput'].append(slot_delivered)
            metrics['fidelity'].append(np.mean(slot_fidelities) if slot_fidelities else np.nan)
            metrics['violations'].append(slot_violations)
        
        return metrics
    
    def simulate_qcast(self, num_slots=500, K=3):
        """Simulate Q-CAST baseline (K-shortest paths, no fidelity awareness)."""
        metrics = {'throughput': [], 'fidelity': [], 'violations': []}
        
        link_pairs = defaultdict(list)
        
        for t in range(num_slots):
            slot_delivered = 0
            slot_fidelities = []
            slot_violations = 0
            
            # Dynamic load
            active_flows = min(self.num_flows, 1 + t // 15)
            
            # Generate
            for link in self.network.links.values():
                if self.rng.random() < link.success_prob:
                    if len(link_pairs[link.link_id]) < 4:
                        link_pairs[link.link_id].append((t, link.initial_fidelity))
            
            # Q-CAST: try K paths, use first available
            for fi in range(active_flows):
                src, dst = self.flows[fi]
                paths = self.network.k_shortest_paths(src, dst, k=K)
                delivered = False
                
                for path in paths:
                    path_links = []
                    all_available = True
                    for i in range(len(path) - 1):
                        link = self.network.get_link(path[i], path[i+1])
                        if link is None or not link_pairs[link.link_id]:
                            all_available = False
                            break
                        path_links.append(link)
                    
                    if not all_available:
                        continue
                    
                    # Use any available pair (random)
                    link_fidelities = []
                    pairs_to_consume = []
                    for link in path_links:
                        pairs = link_pairs[link.link_id]
                        idx = self.rng.integers(len(pairs))
                        gt, f0 = pairs[idx]
                        age = t - gt
                        F_current = fidelity_decay(f0, age, self.T_coh)
                        link_fidelities.append(F_current)
                        pairs_to_consume.append((link.link_id, idx))
                    
                    F_e2e = chain_fidelity(link_fidelities)
                    
                    # No admission control
                    for link_id, idx in sorted(pairs_to_consume, key=lambda x: -x[1]):
                        link_pairs[link_id].pop(idx)
                    slot_delivered += 1
                    slot_fidelities.append(F_e2e)
                    if F_e2e < self.F_th:
                        slot_violations += 1
                    delivered = True
                    break
            
            metrics['throughput'].append(slot_delivered)
            metrics['fidelity'].append(np.mean(slot_fidelities) if slot_fidelities else np.nan)
            metrics['violations'].append(slot_violations)
        
        return metrics
    
    def simulate_sp(self, num_slots=500):
        """Simulate Shortest Path baseline with moderate refresh."""
        metrics = {'throughput': [], 'fidelity': [], 'violations': []}
        
        link_pairs = defaultdict(list)
        
        for t in range(num_slots):
            slot_delivered = 0
            slot_fidelities = []
            slot_violations = 0
            
            # Dynamic load
            active_flows = min(self.num_flows, 1 + t // 15)
            
            # Moderate refresh (tau=1.5 - much less aggressive)
            for link_id in list(link_pairs.keys()):
                link_pairs[link_id] = [
                    (gt, f0) for gt, f0 in link_pairs[link_id]
                    if (t - gt) / self.T_coh <= 1.5
                ]
            
            # Generate
            for link in self.network.links.values():
                if self.rng.random() < link.success_prob:
                    if len(link_pairs[link.link_id]) < 4:
                        link_pairs[link.link_id].append((t, link.initial_fidelity))
            
            # Shortest path only, FIFO pair selection
            for fi in range(active_flows):
                src, dst = self.flows[fi]
                paths = self.network.k_shortest_paths(src, dst, k=1)
                if not paths:
                    continue
                path = paths[0]
                
                path_links = []
                all_available = True
                for i in range(len(path) - 1):
                    link = self.network.get_link(path[i], path[i+1])
                    if link is None or not link_pairs[link.link_id]:
                        all_available = False
                        break
                    path_links.append(link)
                
                if not all_available:
                    continue
                
                link_fidelities = []
                pairs_to_consume = []
                for link in path_links:
                    pairs = link_pairs[link.link_id]
                    gt, f0 = pairs[0]
                    age = t - gt
                    F_current = fidelity_decay(f0, age, self.T_coh)
                    link_fidelities.append(F_current)
                    pairs_to_consume.append(link.link_id)
                
                F_e2e = chain_fidelity(link_fidelities)
                
                # No admission control
                for link_id in pairs_to_consume:
                    link_pairs[link_id].pop(0)
                slot_delivered += 1
                slot_fidelities.append(F_e2e)
                if F_e2e < self.F_th:
                    slot_violations += 1
            
            metrics['throughput'].append(slot_delivered)
            metrics['fidelity'].append(np.mean(slot_fidelities) if slot_fidelities else np.nan)
            metrics['violations'].append(slot_violations)
        
        return metrics


def run_convergence_experiment(num_seeds=10, num_slots=500):
    """Run convergence experiment with multiple seeds."""
    print(f"Running convergence experiment: {num_seeds} seeds, {num_slots} slots...")
    
    all_results = {
        'D-AoE-ARS': {'throughput': [], 'fidelity': [], 'violations': []},
        'Q-CAST': {'throughput': [], 'fidelity': [], 'violations': []},
        'Greedy': {'throughput': [], 'fidelity': [], 'violations': []},
        'SP': {'throughput': [], 'fidelity': [], 'violations': []},
    }
    
    for seed in range(num_seeds):
        print(f"  Seed {seed+1}/{num_seeds}...", end=' ', flush=True)
        network = create_nsfnet(num_memory_slots=4, coherence_time=50.0)
        sim = ConvergenceSimulator(network, coherence_time=50.0, 
                                   fidelity_threshold=0.7, num_flows=5, seed=seed)
        
        # Run all protocols
        m_daoe = sim.simulate_daoe_ars(num_slots=num_slots)
        
        # Reset RNG for fair comparison
        sim.rng = np.random.default_rng(seed)
        m_qcast = sim.simulate_qcast(num_slots=num_slots)
        
        sim.rng = np.random.default_rng(seed)
        m_greedy = sim.simulate_greedy(num_slots=num_slots)
        
        sim.rng = np.random.default_rng(seed)
        m_sp = sim.simulate_sp(num_slots=num_slots)
        
        for key in ['throughput', 'fidelity', 'violations']:
            all_results['D-AoE-ARS'][key].append(m_daoe[key])
            all_results['Q-CAST'][key].append(m_qcast[key])
            all_results['Greedy'][key].append(m_greedy[key])
            all_results['SP'][key].append(m_sp[key])
        
        print("done")
    
    return all_results


def smooth(data, window=20):
    """Moving average smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_convergence(results, num_slots=500, window=20):
    """Generate convergence figure with 3 subplots."""
    fig, axes = plt.subplots(3, 1, figsize=(7, 7.5), sharex=True)
    
    protocols = ['D-AoE-ARS', 'Q-CAST', 'Greedy', 'SP']
    colors = {'D-AoE-ARS': '#2E86AB', 'Q-CAST': '#F18F01', 
              'Greedy': '#C73E1D', 'SP': '#6C757D'}
    linestyles = {'D-AoE-ARS': '-', 'Q-CAST': '--', 'Greedy': '-.', 'SP': ':'}
    
    x = np.arange(window - 1, num_slots)
    
    # (a) Effective Throughput
    ax = axes[0]
    for proto in protocols:
        data = np.array(results[proto]['throughput'])  # (seeds, slots)
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        mean_smooth = smooth(mean, window)
        std_smooth = smooth(std, window)
        
        ax.plot(x, mean_smooth, color=colors[proto], linestyle=linestyles[proto],
                linewidth=1.8, label=proto)
        ax.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth,
                       color=colors[proto], alpha=0.12)
    
    ax.set_ylabel('Effective Throughput\n(pairs/slot)', fontsize=9)
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.set_title('(a) Effective Throughput Convergence', fontsize=10, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # (b) Average Fidelity
    ax = axes[1]
    for proto in protocols:
        data = np.array(results[proto]['fidelity'])
        # Replace NaN with interpolation for smooth plotting
        for s in range(data.shape[0]):
            mask = np.isnan(data[s])
            if mask.any() and not mask.all():
                data[s][mask] = np.interp(np.where(mask)[0], np.where(~mask)[0], data[s][~mask])
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        mean_smooth = smooth(mean, window)
        std_smooth = smooth(std, window)
        
        ax.plot(x, mean_smooth, color=colors[proto], linestyle=linestyles[proto],
                linewidth=1.8, label=proto)
        ax.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth,
                       color=colors[proto], alpha=0.12)
    
    ax.axhline(y=0.7, color='black', linestyle='--', linewidth=1.0, alpha=0.6, label='$F^{th}$')
    ax.set_ylabel('Average Delivered\nFidelity', fontsize=9)
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.set_title('(b) Fidelity Convergence', fontsize=10, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    
    # (c) Cumulative Violation Rate
    ax = axes[2]
    for proto in protocols:
        data = np.array(results[proto]['violations'])
        # Cumulative violation rate: cumsum(violations) / cumsum(deliveries)
        throughput_data = np.array(results[proto]['throughput'])
        
        # Per-seed cumulative violation rate
        cum_viol_rates = []
        for s in range(data.shape[0]):
            cum_viols = np.cumsum(data[s])
            cum_deliveries = np.cumsum(throughput_data[s])
            # Avoid division by zero
            cum_rate = np.where(cum_deliveries > 0, cum_viols / cum_deliveries, 0)
            cum_viol_rates.append(cum_rate)
        
        cum_viol_rates = np.array(cum_viol_rates)
        mean = np.mean(cum_viol_rates, axis=0)
        std = np.std(cum_viol_rates, axis=0)
        mean_smooth = smooth(mean, window)
        std_smooth = smooth(std, window)
        
        ax.plot(x, mean_smooth, color=colors[proto], linestyle=linestyles[proto],
                linewidth=1.8, label=proto)
        ax.fill_between(x, 
                       np.maximum(mean_smooth - std_smooth, 0),
                       mean_smooth + std_smooth,
                       color=colors[proto], alpha=0.12)
    
    ax.set_ylabel('Cumulative\nViolation Rate', fontsize=9)
    ax.set_xlabel('Time Slot', fontsize=9)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_title('(c) Fidelity Violation Rate Over Time', fontsize=10, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)
    
    plt.tight_layout()
    plt.savefig(_os.path.join(FIGURES_DIR, "fig_convergence.pdf"),
                dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(_os.path.join(FIGURES_DIR, "fig_convergence.png"),
                dpi=150, bbox_inches='tight', format='png')
    print("Convergence figure saved to figures/fig_convergence.pdf")


if __name__ == '__main__':
    results = run_convergence_experiment(num_seeds=10, num_slots=500)
    
    # Save raw results
    # Convert numpy arrays for JSON serialization
    json_results = {}
    for proto, metrics in results.items():
        json_results[proto] = {
            k: [list(map(float, seed_data)) for seed_data in v]
            for k, v in metrics.items()
        }
    
    with open(_os.path.join(EXPERIMENTS_DIR, "convergence_results.json"), 'w') as f:
        json.dump(json_results, f)
    print("Raw results saved to experiments/convergence_results.json")
    
    plot_convergence(results, num_slots=500)
