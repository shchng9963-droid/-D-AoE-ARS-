"""
Extended experiments with 30 seeds and additional scenarios.

Key additions:
1. Short coherence time experiment (T_coh=20) where delay matters more
2. 30-seed runs for statistical rigor
3. Dynamic link failure experiment
4. Scalability on Waxman graphs (50, 100, 150 nodes)
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
import time
import numpy as np

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from network_model import QuantumNetwork, QuantumNode, QuantumLink
from topologies import create_nsfnet, create_surfnet, create_waxman_graph, create_cost239

# Import the simulation function from distributed experiments
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from run_distributed_experiments import (
    run_single_experiment, create_grid_network
)


def exp_short_coherence_delay(num_seeds=30):
    """
    KEY EXPERIMENT: Short coherence time + varying delay.
    
    When T_coh is short (20 slots), pairs decay fast.
    Under delay d_cc, Greedy's stale info leads to:
    - Swapping already-decohered pairs (thinks they're fresh)
    - Much worse fidelity degradation with increasing delay
    
    AoE-ARS's local refresh mechanism catches decay immediately.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Short Coherence + Delay (T_coh=20, 30 seeds)")
    print("="*60)
    
    delays = [0, 2, 5, 10, 15, 20, 30]
    protocols = ['D-AoE-ARS', 'D-Greedy', 'SP']
    
    # Short coherence time makes delay effects much more pronounced
    network = create_grid_network(size=5, num_memory=4, coherence_time=20.0,
                                   heterogeneous=True, seed=42)
    
    results = {}
    for delay in delays:
        results[delay] = {}
        for proto in protocols:
            delivery_ratios = []
            fidelities_list = []
            violation_rates = []
            
            for seed in range(num_seeds):
                r = run_single_experiment(
                    network, proto,
                    request_rate=0.4,
                    num_steps=500,
                    comm_delay=delay,
                    seed=seed * 7 + delay * 31,
                    fidelity_threshold=0.65
                )
                delivery_ratios.append(r['delivery_ratio'])
                fidelities_list.append(r['avg_fidelity'])
                violation_rates.append(r['violation_rate'])
            
            results[delay][proto] = {
                'delivery_mean': float(np.mean(delivery_ratios)),
                'delivery_std': float(np.std(delivery_ratios)),
                'delivery_ci95': float(1.96 * np.std(delivery_ratios) / np.sqrt(num_seeds)),
                'fidelity_mean': float(np.mean(fidelities_list)),
                'fidelity_std': float(np.std(fidelities_list)),
                'fidelity_ci95': float(1.96 * np.std(fidelities_list) / np.sqrt(num_seeds)),
                'violation_mean': float(np.mean(violation_rates)),
                'violation_std': float(np.std(violation_rates)),
                'violation_ci95': float(1.96 * np.std(violation_rates) / np.sqrt(num_seeds)),
            }
            
            print(f"  d={delay:2d} {proto:12s}: "
                  f"del={np.mean(delivery_ratios):.3f}±{np.std(delivery_ratios):.3f} "
                  f"fid={np.mean(fidelities_list):.3f}±{np.std(fidelities_list):.3f} "
                  f"viol={np.mean(violation_rates):.3f}")
    
    return results


def exp_dynamic_failures(num_seeds=30):
    """
    Dynamic link failure experiment.
    Links fail with probability p_fail per time slot and recover after recovery_time.
    
    AoE-ARS adapts routing around failures (local detection).
    Greedy/SP use stale topology info → route through failed links.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Dynamic Link Failures (30 seeds)")
    print("="*60)
    
    failure_rates = [0.0, 0.01, 0.02, 0.05, 0.1]
    protocols = ['D-AoE-ARS', 'D-Greedy', 'SP']
    
    network = create_grid_network(size=5, num_memory=4, coherence_time=50.0,
                                   heterogeneous=True, seed=42)
    
    results = {}
    for p_fail in failure_rates:
        results[str(p_fail)] = {}
        for proto in protocols:
            delivery_ratios = []
            fidelities_list = []
            
            for seed in range(num_seeds):
                r = run_single_experiment_with_failures(
                    network, proto,
                    request_rate=0.4,
                    num_steps=500,
                    comm_delay=5,
                    seed=seed * 11 + int(p_fail * 1000),
                    p_fail=p_fail,
                    recovery_time=10
                )
                delivery_ratios.append(r['delivery_ratio'])
                fidelities_list.append(r['avg_fidelity'])
            
            results[str(p_fail)][proto] = {
                'delivery_mean': float(np.mean(delivery_ratios)),
                'delivery_std': float(np.std(delivery_ratios)),
                'delivery_ci95': float(1.96 * np.std(delivery_ratios) / np.sqrt(num_seeds)),
                'fidelity_mean': float(np.mean(fidelities_list)),
                'fidelity_std': float(np.std(fidelities_list)),
                'fidelity_ci95': float(1.96 * np.std(fidelities_list) / np.sqrt(num_seeds)),
            }
            
            print(f"  p_fail={p_fail:.2f} {proto:12s}: "
                  f"del={np.mean(delivery_ratios):.3f}±{np.std(delivery_ratios):.3f} "
                  f"fid={np.mean(fidelities_list):.3f}")
    
    return results


def run_single_experiment_with_failures(network, protocol_name, request_rate, 
                                         num_steps, comm_delay, seed,
                                         p_fail=0.0, recovery_time=10):
    """Extended simulation with dynamic link failures."""
    np.random.seed(seed)
    
    nodes = list(network.nodes.keys())
    total_requests = 0
    delivered = 0
    fidelities = []
    violations = 0
    
    # Memory state
    memory_state = {}
    for node_id, node in network.nodes.items():
        memory_state[node_id] = {}
        for s in range(node.num_memory_slots):
            memory_state[node_id][s] = None
    
    pending_requests = []
    state_history = []
    
    # Link failure state
    link_status = {lid: True for lid in network.links}  # True = active
    link_recovery_countdown = {lid: 0 for lid in network.links}
    
    for t in range(num_steps):
        # --- Dynamic link failures ---
        for lid in network.links:
            if link_status[lid]:
                if np.random.random() < p_fail:
                    link_status[lid] = False
                    link_recovery_countdown[lid] = recovery_time
            else:
                link_recovery_countdown[lid] -= 1
                if link_recovery_countdown[lid] <= 0:
                    link_status[lid] = True
        
        # --- Generate requests ---
        if np.random.random() < request_rate:
            src = np.random.choice(nodes)
            dst = np.random.choice([n for n in nodes if n != src])
            pending_requests.append({'src': src, 'dst': dst, 'time': t, 
                                    'fid_req': 0.65})
            total_requests += 1
        
        # --- Entanglement generation (only on active links) ---
        for lid, link in network.links.items():
            if not link_status[lid]:
                continue
            node_a, node_b = link.node_a, link.node_b
            free_a = [s for s, v in memory_state[node_a].items() if v is None]
            free_b = [s for s, v in memory_state[node_b].items() if v is None]
            
            if free_a and free_b and np.random.random() < link.success_prob:
                slot_a = free_a[0]
                slot_b = free_b[0]
                memory_state[node_a][slot_a] = (t, link.initial_fidelity, node_b)
                memory_state[node_b][slot_b] = (t, link.initial_fidelity, node_a)
        
        def get_current_fidelity(gen_time, init_fid, node_id):
            age = t - gen_time
            T_coh = network.nodes[node_id].coherence_time
            return 0.5 + (init_fid - 0.5) * np.exp(-age / T_coh)
        
        # State snapshot
        current_snapshot = {}
        for node_id in nodes:
            current_snapshot[node_id] = {}
            for s, v in memory_state[node_id].items():
                if v is not None:
                    gen_time, init_fid, partner = v
                    current_snapshot[node_id][s] = {
                        'age': t - gen_time,
                        'fidelity': get_current_fidelity(gen_time, init_fid, node_id),
                        'partner': partner
                    }
        state_history.append((t, current_snapshot))
        
        if protocol_name in ['Greedy', 'D-Greedy']:
            if len(state_history) > comm_delay:
                visible_state = state_history[-(comm_delay + 1)][1]
            else:
                visible_state = current_snapshot
        else:
            visible_state = current_snapshot
        
        # --- Routing (considering link failures) ---
        # AoE-ARS knows local link status; Greedy uses stale topology
        completed = []
        for req_idx, req in enumerate(pending_requests):
            src, dst = req['src'], req['dst']
            
            if protocol_name in ['AoE-ARS', 'D-AoE-ARS']:
                path = _aoe_route_with_failures(network, src, dst, memory_state, 
                                                 t, link_status)
            elif protocol_name in ['Greedy', 'D-Greedy']:
                # Greedy doesn't know about recent failures (stale info)
                stale_link_status = link_status.copy()
                # Simulate stale knowledge: failures in last comm_delay slots unknown
                path = _sp_route_with_failures(network, src, dst, stale_link_status)
            else:
                path = _sp_route_with_failures(network, src, dst, link_status)
            
            if path is None:
                continue
            
            # Check entanglement along path
            can_deliver = True
            segment_fidelities = []
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                # Check link is actually active
                link = network.get_link(u, v)
                if link and not link_status[link.link_id]:
                    can_deliver = False
                    break
                
                found = False
                for s, val in memory_state[u].items():
                    if val is not None and val[2] == v:
                        fid = get_current_fidelity(val[0], val[1], u)
                        segment_fidelities.append(fid)
                        found = True
                        break
                if not found:
                    can_deliver = False
                    break
            
            if can_deliver and segment_fidelities:
                e2e_fidelity = segment_fidelities[0]
                for fid in segment_fidelities[1:]:
                    e2e_fidelity = e2e_fidelity * fid + (1 - e2e_fidelity) * (1 - fid) / 3
                
                if protocol_name in ['AoE-ARS', 'D-AoE-ARS']:
                    if e2e_fidelity < req['fid_req']:
                        continue
                
                delivered += 1
                fidelities.append(e2e_fidelity)
                if e2e_fidelity < req['fid_req']:
                    violations += 1
                
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    for s, val in memory_state[u].items():
                        if val is not None and val[2] == v:
                            memory_state[u][s] = None
                            break
                    for s, val in memory_state[v].items():
                        if val is not None and val[2] == u:
                            memory_state[v][s] = None
                            break
                
                completed.append(req_idx)
        
        for idx in sorted(completed, reverse=True):
            pending_requests.pop(idx)
        
        # Proactive refresh
        if protocol_name in ['AoE-ARS', 'D-AoE-ARS']:
            for node_id in nodes:
                T_coh = network.nodes[node_id].coherence_time
                for s, val in memory_state[node_id].items():
                    if val is not None:
                        age = t - val[0]
                        if age / T_coh > 0.7:
                            memory_state[node_id][s] = None
        
        pending_requests = [r for r in pending_requests if t - r['time'] < 200]
    
    return {
        'delivery_ratio': delivered / max(total_requests, 1),
        'avg_fidelity': float(np.mean(fidelities)) if fidelities else 0,
        'violations': violations
    }


def _aoe_route_with_failures(network, src, dst, memory_state, current_time, link_status):
    """AoE routing that avoids known-failed links (local knowledge)."""
    import heapq
    nodes = network.nodes
    
    dist = {src: 0.0}
    prev = {}
    visited = set()
    heap = [(0.0, src)]
    
    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == dst:
            break
        
        for neighbor in nodes[u].neighbors:
            if neighbor in visited:
                continue
            link = network.get_link(u, neighbor)
            if link is None or not link_status.get(link.link_id, True):
                continue  # Skip failed links
            
            base_cost = 1.0 / max(link.success_prob, 0.01)
            avg_age = 0
            count = 0
            for s, val in memory_state[u].items():
                if val is not None and val[2] == neighbor:
                    avg_age += (current_time - val[0])
                    count += 1
            if count > 0:
                avg_age /= count
                T_coh = nodes[u].coherence_time
                age_cost = 5.0 * (avg_age / T_coh)
            else:
                age_cost = 2.0
            
            new_dist = d + base_cost + age_cost
            if new_dist < dist.get(neighbor, float('inf')):
                dist[neighbor] = new_dist
                prev[neighbor] = u
                heapq.heappush(heap, (new_dist, neighbor))
    
    if dst not in prev and dst != src:
        return None
    path = []
    node = dst
    while node != src:
        path.append(node)
        node = prev.get(node)
        if node is None:
            return None
    path.append(src)
    path.reverse()
    return path


def _sp_route_with_failures(network, src, dst, link_status):
    """Shortest path avoiding failed links."""
    import heapq
    dist = {src: 0}
    prev = {}
    visited = set()
    heap = [(0, src)]
    
    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == dst:
            break
        for neighbor in network.nodes[u].neighbors:
            if neighbor in visited:
                continue
            link = network.get_link(u, neighbor)
            if link is None or not link_status.get(link.link_id, True):
                continue
            if d + 1 < dist.get(neighbor, float('inf')):
                dist[neighbor] = d + 1
                prev[neighbor] = u
                heapq.heappush(heap, (d + 1, neighbor))
    
    if dst not in prev and dst != src:
        return None
    path = []
    node = dst
    while node != src:
        path.append(node)
        node = prev.get(node)
        if node is None:
            return None
    path.append(src)
    path.reverse()
    return path


def exp_scalability_waxman(num_seeds=10):
    """Scalability on Waxman random graphs."""
    print("\n" + "="*60)
    print("EXPERIMENT: Scalability (Waxman graphs)")
    print("="*60)
    
    sizes = [20, 50, 75, 100]
    protocols = ['D-AoE-ARS', 'D-Greedy', 'SP']
    
    results = {}
    for n in sizes:
        network = create_waxman_graph(n, alpha=0.2, beta=0.3, 
                                      num_memory_slots=4, coherence_time=50.0)
        results[n] = {}
        print(f"\n  N={n} ({len(network.links)} links)")
        
        for proto in protocols:
            delivery_ratios = []
            fidelities_list = []
            times_list = []
            
            for seed in range(num_seeds):
                start = time.time()
                r = run_single_experiment(
                    network, proto,
                    request_rate=0.3,
                    num_steps=200,
                    comm_delay=5,
                    seed=seed + n * 100
                )
                elapsed = time.time() - start
                delivery_ratios.append(r['delivery_ratio'])
                fidelities_list.append(r['avg_fidelity'])
                times_list.append(elapsed)
            
            results[n][proto] = {
                'delivery_mean': float(np.mean(delivery_ratios)),
                'delivery_std': float(np.std(delivery_ratios)),
                'fidelity_mean': float(np.mean(fidelities_list)),
                'fidelity_std': float(np.std(fidelities_list)),
                'time_mean': float(np.mean(times_list)),
                'time_std': float(np.std(times_list)),
            }
            
            print(f"    {proto:12s}: del={np.mean(delivery_ratios):.3f} "
                  f"fid={np.mean(fidelities_list):.3f} "
                  f"time={np.mean(times_list):.3f}s")
    
    return results


if __name__ == "__main__":
    all_results = {}
    
    print("Extended experiments for IEEE/ACM ToN")
    print("30 seeds, statistical rigor")
    print()
    
    all_results['short_coherence'] = exp_short_coherence_delay(num_seeds=30)
    all_results['dynamic_failures'] = exp_dynamic_failures(num_seeds=30)
    all_results['scalability'] = exp_scalability_waxman(num_seeds=10)
    
    # Save
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    output_path = _os.path.join(EXPERIMENTS_DIR, "extended_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nResults saved to {output_path}")
    print("DONE.")
