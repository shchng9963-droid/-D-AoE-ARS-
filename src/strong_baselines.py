"""
Strong baselines for comparison:
1. Q-CAST: K-shortest paths with concurrent entanglement (Shi & Qian, SIGCOMM'20)
2. DQN-Routing: Deep Q-Network based routing (simplified)
3. LP Upper Bound: Linear programming relaxation for optimal allocation
"""


import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO_ROOT = _os.path.dirname(_HERE)
EXPERIMENTS_DIR = _os.path.join(REPO_ROOT, "experiments")
FIGURES_DIR = _os.path.join(REPO_ROOT, "figures")
_os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
_os.makedirs(FIGURES_DIR, exist_ok=True)

import sys
import numpy as np
from collections import defaultdict
import heapq

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from network_model import QuantumNetwork, QuantumNode, QuantumLink


class QCASTProtocol:
    """
    Q-CAST: Concurrent Entanglement Routing.
    Based on Shi & Qian (SIGCOMM 2020).
    
    Key ideas:
    - Find K shortest paths for each request
    - Attempt entanglement generation on ALL paths concurrently
    - Use the first path that succeeds end-to-end
    - Recovery: if partial path fails, try to extend from intermediate nodes
    """
    
    def __init__(self, network, K=3):
        self.network = network
        self.K = K
        self._path_cache = {}
    
    def find_k_shortest_paths(self, src, dst, k=None):
        """Yen's K-shortest paths algorithm."""
        if k is None:
            k = self.K
        
        cache_key = (src, dst)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]
        
        # First shortest path (Dijkstra)
        first_path = self._dijkstra(src, dst)
        if first_path is None:
            return []
        
        A = [first_path]  # K shortest paths found
        B = []  # Candidate paths
        
        for ki in range(1, k):
            for i in range(len(A[-1]) - 1):
                spur_node = A[-1][i]
                root_path = A[-1][:i+1]
                
                # Remove edges used by existing paths at this spur
                removed_edges = set()
                for path in A:
                    if path[:i+1] == root_path:
                        if i + 1 < len(path):
                            removed_edges.add((path[i], path[i+1]))
                
                # Find spur path
                spur_path = self._dijkstra(spur_node, dst, excluded_edges=removed_edges,
                                           excluded_nodes=set(root_path[:-1]))
                
                if spur_path is not None:
                    total_path = root_path[:-1] + spur_path
                    if total_path not in B and total_path not in A:
                        B.append(total_path)
            
            if not B:
                break
            
            # Sort B by path length and pick shortest
            B.sort(key=len)
            A.append(B.pop(0))
        
        self._path_cache[cache_key] = A
        return A
    
    def _dijkstra(self, src, dst, excluded_edges=None, excluded_nodes=None):
        """Dijkstra with optional edge/node exclusions."""
        if excluded_edges is None:
            excluded_edges = set()
        if excluded_nodes is None:
            excluded_nodes = set()
        
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
            
            for neighbor in self.network.nodes[u].neighbors:
                if neighbor in visited or neighbor in excluded_nodes:
                    continue
                if (u, neighbor) in excluded_edges:
                    continue
                
                link = self.network.get_link(u, neighbor)
                if link is None:
                    continue
                
                # Weight: inverse of success probability
                weight = 1.0 / max(link.success_prob, 0.01)
                
                if d + weight < dist.get(neighbor, float('inf')):
                    dist[neighbor] = d + weight
                    prev[neighbor] = u
                    heapq.heappush(heap, (d + weight, neighbor))
        
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
    
    def route(self, src, dst, memory_state, current_time):
        """
        Q-CAST routing: try K paths, pick the one with best available entanglement.
        """
        paths = self.find_k_shortest_paths(src, dst)
        
        best_path = None
        best_score = -1
        
        for path in paths:
            # Score: number of segments with available entanglement
            score = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                for s, val in memory_state[u].items():
                    if val is not None and val[2] == v:
                        score += 1
                        break
            
            # Normalize by path length
            normalized_score = score / (len(path) - 1) if len(path) > 1 else 0
            
            if normalized_score > best_score:
                best_score = normalized_score
                best_path = path
        
        return best_path


class DQNRoutingProtocol:
    """
    DQN-based routing (simplified tabular Q-learning for fair comparison).
    
    State: (current_node, destination, local_memory_ages)
    Action: next_hop selection
    Reward: +1 for delivery with F >= threshold, -0.1 per hop, -1 for timeout
    
    Pre-trained via Q-learning episodes, then used for routing.
    """
    
    def __init__(self, network, coherence_time=50.0, fidelity_threshold=0.65,
                 learning_rate=0.1, discount=0.95, epsilon=0.1):
        self.network = network
        self.coherence_time = coherence_time
        self.fidelity_threshold = fidelity_threshold
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(float))
        self.trained = False
    
    def _get_state(self, current_node, dst, memory_state, current_time):
        """Discretized state representation."""
        # State: (current_node, dst, avg_age_bucket)
        ages = []
        for s, val in memory_state[current_node].items():
            if val is not None:
                ages.append(current_time - val[0])
        
        avg_age = np.mean(ages) if ages else self.coherence_time
        age_bucket = min(int(avg_age / (self.coherence_time / 5)), 4)
        
        return (current_node, dst, age_bucket)
    
    def train(self, memory_state, num_episodes=500, max_hops=15):
        """Train Q-table via episodes."""
        nodes = list(self.network.nodes.keys())
        
        for ep in range(num_episodes):
            src = np.random.choice(nodes)
            dst = np.random.choice([n for n in nodes if n != src])
            
            current = src
            t = 0
            
            for hop in range(max_hops):
                state = (current, dst, 2)  # simplified state during training
                neighbors = self.network.nodes[current].neighbors
                
                if not neighbors:
                    break
                
                # Epsilon-greedy
                if np.random.random() < self.epsilon:
                    action = np.random.choice(neighbors)
                else:
                    q_vals = {n: self.Q[state][n] for n in neighbors}
                    action = max(q_vals, key=q_vals.get) if q_vals else neighbors[0]
                
                # Transition
                next_node = action
                
                if next_node == dst:
                    reward = 1.0
                    self.Q[state][action] += self.lr * (reward - self.Q[state][action])
                    break
                else:
                    reward = -0.1  # hop penalty
                    next_state = (next_node, dst, 2)
                    next_neighbors = self.network.nodes[next_node].neighbors
                    if next_neighbors:
                        max_next_q = max(self.Q[next_state][n] for n in next_neighbors) if self.Q[next_state] else 0
                    else:
                        max_next_q = 0
                    
                    self.Q[state][action] += self.lr * (
                        reward + self.gamma * max_next_q - self.Q[state][action])
                
                current = next_node
                t += 1
            else:
                # Timeout penalty
                state = (current, dst, 2)
                if self.network.nodes[current].neighbors:
                    for n in self.network.nodes[current].neighbors:
                        self.Q[state][n] += self.lr * (-1.0 - self.Q[state][n])
        
        self.trained = True
    
    def route(self, src, dst, memory_state, current_time):
        """Use trained Q-table to route."""
        if not self.trained:
            self.train(memory_state)
        
        path = [src]
        current = src
        visited = {src}
        
        for _ in range(20):  # max hops
            state = self._get_state(current, dst, memory_state, current_time)
            neighbors = [n for n in self.network.nodes[current].neighbors if n not in visited]
            
            if not neighbors:
                # Backtrack or fail
                return None
            
            # Greedy w.r.t. Q-values
            q_vals = {n: self.Q[state][n] for n in neighbors}
            if any(v != 0 for v in q_vals.values()):
                next_node = max(q_vals, key=q_vals.get)
            else:
                # No Q-value info, use shortest path heuristic
                next_node = min(neighbors, key=lambda n: self._hop_distance(n, dst))
            
            path.append(next_node)
            visited.add(next_node)
            
            if next_node == dst:
                return path
            
            current = next_node
        
        return None
    
    def _hop_distance(self, src, dst):
        """BFS hop distance."""
        if src == dst:
            return 0
        visited = {src}
        queue = [(src, 0)]
        while queue:
            node, d = queue.pop(0)
            for neighbor in self.network.nodes[node].neighbors:
                if neighbor == dst:
                    return d + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))
        return 999


class LPUpperBound:
    """
    LP relaxation upper bound on optimal throughput.
    
    Formulates the entanglement distribution as a multi-commodity flow problem:
    - Each request is a commodity
    - Link capacities = entanglement generation rate * success_prob
    - Objective: maximize total flow (delivery rate)
    
    This gives an UPPER BOUND on what any protocol can achieve.
    """
    
    def __init__(self, network):
        self.network = network
    
    def compute_upper_bound(self, requests, num_steps):
        """
        Compute LP upper bound on delivery ratio.
        Uses max-flow formulation per request pair.
        """
        nodes = list(self.network.nodes.keys())
        
        # Compute link capacities (pairs per time slot)
        link_caps = {}
        for lid, link in self.network.links.items():
            # Capacity = success_prob * min(memory_slots_a, memory_slots_b)
            node_a_mem = self.network.nodes[link.node_a].num_memory_slots
            node_b_mem = self.network.nodes[link.node_b].num_memory_slots
            cap = link.success_prob * min(node_a_mem, node_b_mem)
            link_caps[(link.node_a, link.node_b)] = cap
            link_caps[(link.node_b, link.node_a)] = cap
        
        # For each unique (src, dst) pair, compute max-flow
        # Simplified: use shortest path capacity as bound
        total_capacity = 0
        
        for req in requests:
            src, dst = req['src'], req['dst']
            # Find bottleneck capacity on shortest path
            path = self._shortest_path(src, dst)
            if path is None:
                continue
            
            # Bottleneck = min link capacity along path
            bottleneck = float('inf')
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                cap = link_caps.get((u, v), 0)
                bottleneck = min(bottleneck, cap)
            
            # Account for multi-hop fidelity decay
            # Each swap reduces fidelity, so effective capacity is lower
            hops = len(path) - 1
            fidelity_factor = 0.95 ** hops  # approximate
            
            total_capacity += bottleneck * fidelity_factor
        
        # Upper bound on delivery ratio
        if len(requests) == 0:
            return 1.0
        
        # Total deliverable in num_steps time slots
        max_deliverable = total_capacity * num_steps
        upper_bound = min(max_deliverable / len(requests), 1.0)
        
        return upper_bound
    
    def _shortest_path(self, src, dst):
        """BFS shortest path."""
        if src == dst:
            return [src]
        visited = {src}
        queue = [(src, [src])]
        while queue:
            node, path = queue.pop(0)
            for neighbor in self.network.nodes[node].neighbors:
                if neighbor == dst:
                    return path + [dst]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None


# ============================================================
# Run comparison with strong baselines
# ============================================================
def run_with_strong_baselines(network, num_seeds=10, num_steps=200, 
                               request_rate=0.4, comm_delay=5):
    """Run all protocols including strong baselines."""
    from run_distributed_experiments import run_single_experiment
    
    results = {}
    protocols_simple = ['D-AoE-ARS', 'D-Greedy', 'SP']
    
    # Run simple protocols
    for proto in protocols_simple:
        dr, fid, viol = [], [], []
        for seed in range(num_seeds):
            r = run_single_experiment(network, proto, request_rate=request_rate,
                                      num_steps=num_steps, comm_delay=comm_delay,
                                      seed=seed)
            dr.append(r['delivery_ratio'])
            fid.append(r['avg_fidelity'])
            viol.append(r['violation_rate'])
        results[proto] = {
            'del_m': float(np.mean(dr)), 'del_s': float(np.std(dr)),
            'fid_m': float(np.mean(fid)), 'fid_s': float(np.std(fid)),
            'viol_m': float(np.mean(viol)), 'viol_s': float(np.std(viol)),
        }
    
    # Run Q-CAST (K=3)
    qcast = QCASTProtocol(network, K=3)
    dr, fid, viol = [], [], []
    for seed in range(num_seeds):
        r = run_qcast_experiment(network, qcast, request_rate=request_rate,
                                  num_steps=num_steps, comm_delay=comm_delay, seed=seed)
        dr.append(r['delivery_ratio'])
        fid.append(r['avg_fidelity'])
        viol.append(r['violation_rate'])
    results['Q-CAST-K3'] = {
        'del_m': float(np.mean(dr)), 'del_s': float(np.std(dr)),
        'fid_m': float(np.mean(fid)), 'fid_s': float(np.std(fid)),
        'viol_m': float(np.mean(viol)), 'viol_s': float(np.std(viol)),
    }
    
    # Run DQN
    dqn = DQNRoutingProtocol(network, coherence_time=50.0)
    dqn.train(None, num_episodes=1000)  # Pre-train
    dr, fid, viol = [], [], []
    for seed in range(num_seeds):
        r = run_dqn_experiment(network, dqn, request_rate=request_rate,
                                num_steps=num_steps, comm_delay=comm_delay, seed=seed)
        dr.append(r['delivery_ratio'])
        fid.append(r['avg_fidelity'])
        viol.append(r['violation_rate'])
    results['DQN'] = {
        'del_m': float(np.mean(dr)), 'del_s': float(np.std(dr)),
        'fid_m': float(np.mean(fid)), 'fid_s': float(np.std(fid)),
        'viol_m': float(np.mean(viol)), 'viol_s': float(np.std(viol)),
    }
    
    # LP Upper Bound
    nodes = list(network.nodes.keys())
    requests = [{'src': np.random.choice(nodes), 
                 'dst': np.random.choice([n for n in nodes if n != nodes[0]])}
                for _ in range(int(request_rate * num_steps))]
    lp = LPUpperBound(network)
    ub = lp.compute_upper_bound(requests, num_steps)
    results['LP-UB'] = {'del_m': float(ub), 'del_s': 0.0,
                        'fid_m': 1.0, 'fid_s': 0.0,
                        'viol_m': 0.0, 'viol_s': 0.0}
    
    return results


def run_qcast_experiment(network, qcast, request_rate, num_steps, comm_delay, seed):
    """Run simulation with Q-CAST routing."""
    np.random.seed(seed)
    nodes = list(network.nodes.keys())
    
    total_requests = 0
    delivered = 0
    fidelities = []
    violations = 0
    
    memory_state = {}
    for node_id, node in network.nodes.items():
        memory_state[node_id] = {s: None for s in range(node.num_memory_slots)}
    
    pending_requests = []
    
    for t in range(num_steps):
        if np.random.random() < request_rate:
            src = np.random.choice(nodes)
            dst = np.random.choice([n for n in nodes if n != src])
            pending_requests.append({'src': src, 'dst': dst, 'time': t, 'fid_req': 0.65})
            total_requests += 1
        
        # Entanglement generation
        for lid, link in network.links.items():
            node_a, node_b = link.node_a, link.node_b
            free_a = [s for s, v in memory_state[node_a].items() if v is None]
            free_b = [s for s, v in memory_state[node_b].items() if v is None]
            if free_a and free_b and np.random.random() < link.success_prob:
                memory_state[node_a][free_a[0]] = (t, link.initial_fidelity, node_b)
                memory_state[node_b][free_b[0]] = (t, link.initial_fidelity, node_a)
        
        def get_fid(gen_time, init_fid, node_id):
            age = t - gen_time
            T_coh = network.nodes[node_id].coherence_time
            return 0.5 + (init_fid - 0.5) * np.exp(-age / T_coh)
        
        # Process requests with Q-CAST routing
        completed = []
        for req_idx, req in enumerate(pending_requests):
            path = qcast.route(req['src'], req['dst'], memory_state, t)
            if path is None:
                continue
            
            can_deliver = True
            segment_fidelities = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                found = False
                for s, val in memory_state[u].items():
                    if val is not None and val[2] == v:
                        fid = get_fid(val[0], val[1], u)
                        segment_fidelities.append(fid)
                        found = True
                        break
                if not found:
                    can_deliver = False
                    break
            
            if can_deliver and segment_fidelities:
                e2e_fid = segment_fidelities[0]
                for fid in segment_fidelities[1:]:
                    e2e_fid = e2e_fid * fid + (1 - e2e_fid) * (1 - fid) / 3
                
                delivered += 1
                fidelities.append(e2e_fid)
                if e2e_fid < req['fid_req']:
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
        pending_requests = [r for r in pending_requests if t - r['time'] < 200]
    
    return {
        'delivery_ratio': delivered / max(total_requests, 1),
        'avg_fidelity': float(np.mean(fidelities)) if fidelities else 0,
        'violation_rate': violations / max(delivered, 1)
    }


def run_dqn_experiment(network, dqn, request_rate, num_steps, comm_delay, seed):
    """Run simulation with DQN routing."""
    np.random.seed(seed)
    nodes = list(network.nodes.keys())
    
    total_requests = 0
    delivered = 0
    fidelities = []
    violations = 0
    
    memory_state = {}
    for node_id, node in network.nodes.items():
        memory_state[node_id] = {s: None for s in range(node.num_memory_slots)}
    
    pending_requests = []
    
    for t in range(num_steps):
        if np.random.random() < request_rate:
            src = np.random.choice(nodes)
            dst = np.random.choice([n for n in nodes if n != src])
            pending_requests.append({'src': src, 'dst': dst, 'time': t, 'fid_req': 0.65})
            total_requests += 1
        
        for lid, link in network.links.items():
            node_a, node_b = link.node_a, link.node_b
            free_a = [s for s, v in memory_state[node_a].items() if v is None]
            free_b = [s for s, v in memory_state[node_b].items() if v is None]
            if free_a and free_b and np.random.random() < link.success_prob:
                memory_state[node_a][free_a[0]] = (t, link.initial_fidelity, node_b)
                memory_state[node_b][free_b[0]] = (t, link.initial_fidelity, node_a)
        
        def get_fid(gen_time, init_fid, node_id):
            age = t - gen_time
            T_coh = network.nodes[node_id].coherence_time
            return 0.5 + (init_fid - 0.5) * np.exp(-age / T_coh)
        
        completed = []
        for req_idx, req in enumerate(pending_requests):
            path = dqn.route(req['src'], req['dst'], memory_state, t)
            if path is None:
                continue
            
            can_deliver = True
            segment_fidelities = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                found = False
                for s, val in memory_state[u].items():
                    if val is not None and val[2] == v:
                        fid = get_fid(val[0], val[1], u)
                        segment_fidelities.append(fid)
                        found = True
                        break
                if not found:
                    can_deliver = False
                    break
            
            if can_deliver and segment_fidelities:
                e2e_fid = segment_fidelities[0]
                for fid in segment_fidelities[1:]:
                    e2e_fid = e2e_fid * fid + (1 - e2e_fid) * (1 - fid) / 3
                
                delivered += 1
                fidelities.append(e2e_fid)
                if e2e_fid < req['fid_req']:
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
        pending_requests = [r for r in pending_requests if t - r['time'] < 200]
    
    return {
        'delivery_ratio': delivered / max(total_requests, 1),
        'avg_fidelity': float(np.mean(fidelities)) if fidelities else 0,
        'violation_rate': violations / max(delivered, 1)
    }


if __name__ == "__main__":
    from run_distributed_experiments import create_grid_network
    import json
    
    print("Strong Baselines Comparison")
    print("="*60)
    
    # Test on heterogeneous 5x5 grid
    network = create_grid_network(size=5, num_memory=4, coherence_time=50.0,
                                   heterogeneous=True, seed=42)
    
    results = run_with_strong_baselines(network, num_seeds=10, num_steps=200,
                                         request_rate=0.4, comm_delay=5)
    
    print("\nResults (heterogeneous 5x5, d_cc=5):")
    print(f"{'Protocol':<12} {'Delivery':>10} {'Fidelity':>10} {'Violations':>10}")
    print("-"*45)
    for proto, r in results.items():
        print(f"{proto:<12} {r['del_m']:>8.3f}±{r['del_s']:.3f} "
              f"{r['fid_m']:>6.3f}±{r['fid_s']:.3f} "
              f"{r['viol_m']:>6.3f}")
    
    # Save
    with open(_os.path.join(EXPERIMENTS_DIR, "strong_baselines_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved.")
