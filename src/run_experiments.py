"""
Experimental Evaluation — AoE-ARS vs Baselines
================================================
Phase 6: Comprehensive experiments under stress conditions.
"""

import numpy as np
import sys
import os
import json
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network_model import QuantumNetwork
from protocols import AoEARS, ShortestPathRouting, FidelityAwareRouting, GreedyScheduling
from simulation_engine import SimulationConfig, QuantumNetworkSimulator, run_experiment


def experiment_1_congestion():
    """Experiment 1: Varying request arrival rate (congestion stress test)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Congestion Stress Test (Grid 4x4)")
    print("  Varying request arrival rate with limited memory")
    print("=" * 70)

    arrival_rates = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    protocols = ['aoe_ars', 'shortest_path', 'fidelity_aware', 'greedy']
    results = {p: {'fidelity': [], 'aoe': [], 'throughput': [], 'violation': [], 'latency': []}
               for p in protocols}

    for rate in arrival_rates:
        print(f"\n  Arrival rate = {rate:.2f}")
        for proto in protocols:
            config = SimulationConfig(
                num_time_slots=2000,
                seed=42,
                request_arrival_rate=rate,
                fidelity_threshold=0.65,
                min_path_length=2,
                max_concurrent_requests=100
            )
            metrics = run_experiment(
                'grid', proto, config,
                rows=4, cols=4,
                memory_slots=4,          # Limited memory!
                coherence_time=100.0,    # Short coherence (stress aging)
                link_success_prob=0.4,   # Moderate success rate
                link_fidelity=0.95
            )
            s = metrics.summary()
            results[proto]['fidelity'].append(s['avg_fidelity'])
            results[proto]['aoe'].append(s['avg_aoe'])
            results[proto]['throughput'].append(s['throughput'])
            results[proto]['violation'].append(s['violation_rate'])
            results[proto]['latency'].append(s['avg_latency'])
            print(f"    {proto:20s}: del={s['delivered']:3d}/{s['total']:3d} "
                  f"fid={s['avg_fidelity']:.3f} aoe={s['avg_aoe']:.1f} "
                  f"lat={s['avg_latency']:.1f} viol={s['violation_rate']:.2f}")

    return {'arrival_rates': arrival_rates, 'results': results}


def experiment_2_coherence():
    """Experiment 2: Varying coherence time (aging sensitivity)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Coherence Time Sensitivity (Grid 4x4)")
    print("  Varying T_coh with moderate load")
    print("=" * 70)

    coherence_times = [20, 50, 100, 200, 500, 1000, 2000]
    protocols = ['aoe_ars', 'shortest_path', 'fidelity_aware', 'greedy']
    results = {p: {'fidelity': [], 'aoe': [], 'throughput': [], 'violation': []}
               for p in protocols}

    for t_coh in coherence_times:
        print(f"\n  T_coh = {t_coh}")
        for proto in protocols:
            config = SimulationConfig(
                num_time_slots=2000,
                seed=42,
                request_arrival_rate=0.3,
                fidelity_threshold=0.65,
                min_path_length=2,
                max_concurrent_requests=80
            )
            metrics = run_experiment(
                'grid', proto, config,
                rows=4, cols=4,
                memory_slots=5,
                coherence_time=float(t_coh),
                link_success_prob=0.4,
                link_fidelity=0.95
            )
            s = metrics.summary()
            results[proto]['fidelity'].append(s['avg_fidelity'])
            results[proto]['aoe'].append(s['avg_aoe'])
            results[proto]['throughput'].append(s['throughput'])
            results[proto]['violation'].append(s['violation_rate'])
            print(f"    {proto:20s}: del={s['delivered']:3d}/{s['total']:3d} "
                  f"fid={s['avg_fidelity']:.3f} aoe={s['avg_aoe']:.1f} "
                  f"viol={s['violation_rate']:.2f}")

    return {'coherence_times': coherence_times, 'results': results}


def experiment_3_scalability():
    """Experiment 3: Network size scalability."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Scalability (Grid NxN)")
    print("  Varying network size")
    print("=" * 70)

    grid_sizes = [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]
    protocols = ['aoe_ars', 'shortest_path', 'fidelity_aware', 'greedy']
    results = {p: {'fidelity': [], 'aoe': [], 'throughput': [], 'latency': []}
               for p in protocols}

    for (rows, cols) in grid_sizes:
        n_nodes = rows * cols
        print(f"\n  Grid {rows}x{cols} ({n_nodes} nodes)")
        for proto in protocols:
            config = SimulationConfig(
                num_time_slots=2000,
                seed=42,
                request_arrival_rate=0.2,
                fidelity_threshold=0.6,
                min_path_length=2,
                max_concurrent_requests=80
            )
            metrics = run_experiment(
                'grid', proto, config,
                rows=rows, cols=cols,
                memory_slots=5,
                coherence_time=150.0,
                link_success_prob=0.5,
                link_fidelity=0.95
            )
            s = metrics.summary()
            results[proto]['fidelity'].append(s['avg_fidelity'])
            results[proto]['aoe'].append(s['avg_aoe'])
            results[proto]['throughput'].append(s['throughput'])
            results[proto]['latency'].append(s['avg_latency'])
            print(f"    {proto:20s}: del={s['delivered']:3d}/{s['total']:3d} "
                  f"fid={s['avg_fidelity']:.3f} aoe={s['avg_aoe']:.1f} "
                  f"lat={s['avg_latency']:.1f}")

    return {'grid_sizes': [f"{r}x{c}" for r, c in grid_sizes], 'results': results}


def experiment_4_memory():
    """Experiment 4: Memory constraint sensitivity."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Memory Constraints (Grid 4x4)")
    print("  Varying memory slots per node")
    print("=" * 70)

    memory_sizes = [2, 3, 4, 6, 8, 12, 16]
    protocols = ['aoe_ars', 'shortest_path', 'fidelity_aware', 'greedy']
    results = {p: {'fidelity': [], 'aoe': [], 'throughput': [], 'violation': []}
               for p in protocols}

    for mem in memory_sizes:
        print(f"\n  Memory slots = {mem}")
        for proto in protocols:
            config = SimulationConfig(
                num_time_slots=2000,
                seed=42,
                request_arrival_rate=0.3,
                fidelity_threshold=0.65,
                min_path_length=2,
                max_concurrent_requests=80
            )
            metrics = run_experiment(
                'grid', proto, config,
                rows=4, cols=4,
                memory_slots=mem,
                coherence_time=100.0,
                link_success_prob=0.4,
                link_fidelity=0.95
            )
            s = metrics.summary()
            results[proto]['fidelity'].append(s['avg_fidelity'])
            results[proto]['aoe'].append(s['avg_aoe'])
            results[proto]['throughput'].append(s['throughput'])
            results[proto]['violation'].append(s['violation_rate'])
            print(f"    {proto:20s}: del={s['delivered']:3d}/{s['total']:3d} "
                  f"fid={s['avg_fidelity']:.3f} aoe={s['avg_aoe']:.1f} "
                  f"viol={s['violation_rate']:.2f}")

    return {'memory_sizes': memory_sizes, 'results': results}


def experiment_5_link_quality():
    """Experiment 5: Heterogeneous link quality."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Link Success Probability (Grid 4x4)")
    print("  Varying link generation success rate")
    print("=" * 70)

    success_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    protocols = ['aoe_ars', 'shortest_path', 'fidelity_aware', 'greedy']
    results = {p: {'fidelity': [], 'aoe': [], 'throughput': [], 'latency': []}
               for p in protocols}

    for p_succ in success_probs:
        print(f"\n  Link success prob = {p_succ:.2f}")
        for proto in protocols:
            config = SimulationConfig(
                num_time_slots=2000,
                seed=42,
                request_arrival_rate=0.2,
                fidelity_threshold=0.6,
                min_path_length=2,
                max_concurrent_requests=80
            )
            metrics = run_experiment(
                'grid', proto, config,
                rows=4, cols=4,
                memory_slots=5,
                coherence_time=100.0,
                link_success_prob=p_succ,
                link_fidelity=0.95
            )
            s = metrics.summary()
            results[proto]['fidelity'].append(s['avg_fidelity'])
            results[proto]['aoe'].append(s['avg_aoe'])
            results[proto]['throughput'].append(s['throughput'])
            results[proto]['latency'].append(s['avg_latency'])
            print(f"    {proto:20s}: del={s['delivered']:3d}/{s['total']:3d} "
                  f"fid={s['avg_fidelity']:.3f} aoe={s['avg_aoe']:.1f} "
                  f"lat={s['avg_latency']:.1f}")

    return {'success_probs': success_probs, 'results': results}


if __name__ == "__main__":
    print("AoE-ARS: Full Experimental Evaluation")
    print("=" * 70)

    all_results = {}

    all_results['exp1_congestion'] = experiment_1_congestion()
    all_results['exp2_coherence'] = experiment_2_coherence()
    all_results['exp3_scalability'] = experiment_3_scalability()
    all_results['exp4_memory'] = experiment_4_memory()
    all_results['exp5_link_quality'] = experiment_5_link_quality()

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\n\nResults saved to {output_path}")
    print("=" * 70)
    print("DONE")
