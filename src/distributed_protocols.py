"""
Distributed AoE-ARS (D-AoE-ARS) and Distributed Greedy protocols.

Key insight: In real quantum networks, classical communication has non-zero latency.
- D-AoE-ARS uses ONLY local information (own memory slots + 1-hop neighbor states)
- Distributed Greedy requires global state, which arrives with delay d_cc
- Under delay, Greedy's decisions are based on stale information → performance degrades

This module implements:
1. D-AoE-ARS: Distributed Age-of-Entanglement Aware Routing & Scheduling
2. D-Greedy: Greedy with realistic communication delay
3. D-SP: Distributed Shortest Path (baseline)
4. Oracle-Greedy: Greedy with perfect global info (upper bound)
"""

import numpy as np
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set


class DistributedNodeState:
    """Local state maintained by each node in distributed setting."""
    
    def __init__(self, node_id: str, neighbors: List[str], coherence_time: float):
        self.node_id = node_id
        self.neighbors = neighbors
        self.coherence_time = coherence_time
        
        # Local information (always fresh)
        self.local_memory_ages = {}  # slot_id -> age (time since generation)
        self.local_memory_fidelities = {}  # slot_id -> estimated fidelity
        self.local_link_states = {}  # neighbor_id -> last_success_time
        self.pending_requests = []  # requests this node is source for
        
        # Neighbor information (received with delay)
        self.neighbor_states = {}  # neighbor_id -> {ages, fidelities, timestamp}
        self.neighbor_state_age = {}  # neighbor_id -> how old the info is
        
        # Global information (for Greedy, received with larger delay)
        self.global_state = {}  # node_id -> state (stale)
        self.global_state_age = 0  # how old global info is


class DistributedAoEARS:
    """
    Distributed AoE-ARS Protocol.
    
    Each node makes decisions using ONLY:
    1. Its own memory slot states (age, fidelity) — always fresh
    2. 1-hop neighbor states — delayed by 1 classical hop (d_cc)
    3. Static topology knowledge — no delay
    
    Key mechanisms:
    - Local AoE-weighted routing: compute path costs using local + neighbor AoE
    - Distributed scheduling: Lyapunov drift with local queue info
    - Proactive refresh: purely local decision based on own slot ages
    - Predictive swap: use local fidelity estimates
    
    Communication overhead: O(degree) messages per time slot
    """
    
    def __init__(self, network, V=10.0, refresh_threshold=0.7, 
                 comm_delay=0, k_hop=1):
        """
        Args:
            network: QuantumNetwork instance
            V: Lyapunov tradeoff parameter
            refresh_threshold: AoE/T_coh ratio to trigger refresh
            comm_delay: classical communication delay (time slots)
            k_hop: how many hops of neighbor info available (1 or 2)
        """
        self.network = network
        self.V = V
        self.refresh_threshold = refresh_threshold
        self.comm_delay = comm_delay
        self.k_hop = k_hop
        self.name = f"D-AoE-ARS(k={k_hop},d={comm_delay})"
        
        # Each node maintains local state
        self.node_states = {}
        for node_id, node in network.nodes.items():
            neighbors = [link.node_b if link.node_a == node_id else link.node_a 
                        for link in network.get_node_links(node_id)]
            self.node_states[node_id] = DistributedNodeState(
                node_id, neighbors, node.coherence_time
            )
    
    def get_name(self):
        return self.name
    
    def compute_local_aoe_cost(self, node_id: str, neighbor_id: str, 
                                current_time: int) -> float:
        """
        Compute edge cost using locally available information.
        Cost = base_hop_cost + alpha * estimated_AoE_at_neighbor
        
        If neighbor state is available (within k_hop), use it.
        Otherwise, use pessimistic estimate based on coherence time.
        """
        state = self.node_states[node_id]
        link = self.network.get_link(node_id, neighbor_id)
        if link is None:
            return float('inf')
        
        # Base cost: inverse of link success probability
        base_cost = 1.0 / max(link.success_prob, 0.01)
        
        # AoE component: estimate freshness at neighbor
        if neighbor_id in state.neighbor_states:
            neighbor_info = state.neighbor_states[neighbor_id]
            info_age = state.neighbor_state_age.get(neighbor_id, 0)
            
            # Average AoE of neighbor's slots, adjusted for staleness
            if neighbor_info.get('avg_aoe') is not None:
                estimated_aoe = neighbor_info['avg_aoe'] + info_age
            else:
                estimated_aoe = state.coherence_time * 0.5  # pessimistic
        else:
            # No info available — use pessimistic estimate
            estimated_aoe = state.coherence_time * 0.5
        
        # Normalize AoE by coherence time
        aoe_cost = estimated_aoe / state.coherence_time
        
        return base_cost + self.V * aoe_cost
    
    def route_request(self, source: str, destination: str, 
                      current_time: int) -> Optional[List[str]]:
        """
        Distributed AoE-weighted Dijkstra using local + k-hop info.
        
        For edges within k-hop of source: use actual AoE info
        For edges beyond k-hop: use static cost (topology-based estimate)
        """
        # Dijkstra with AoE-weighted costs
        dist = {source: 0.0}
        prev = {}
        visited = set()
        heap = [(0.0, source)]
        
        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            
            if u == destination:
                break
            
            # Get neighbors
            for link in self.network.get_node_links(u):
                v = link.node_b if link.node_a == u else link.node_a
                if v in visited:
                    continue
                
                # Compute cost based on available information
                hops_from_source = self._hop_distance(source, u)
                if hops_from_source <= self.k_hop:
                    # Within k-hop: use AoE-aware cost
                    cost = self.compute_local_aoe_cost(u, v, current_time)
                else:
                    # Beyond k-hop: use static cost
                    cost = 1.0 / max(link.success_prob, 0.01)
                
                new_dist = d + cost
                if new_dist < dist.get(v, float('inf')):
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(heap, (new_dist, v))
        
        # Reconstruct path
        if destination not in prev and destination != source:
            return None
        
        path = []
        node = destination
        while node != source:
            path.append(node)
            node = prev.get(node)
            if node is None:
                return None
        path.append(source)
        path.reverse()
        return path
    
    def _hop_distance(self, source: str, node: str) -> int:
        """BFS hop distance (cached in practice)."""
        if source == node:
            return 0
        visited = {source}
        queue = [(source, 0)]
        while queue:
            current, dist = queue.pop(0)
            for link in self.network.get_node_links(current):
                neighbor = link.node_b if link.node_a == current else link.node_a
                if neighbor == node:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float('inf')
    
    def schedule_links(self, current_time: int, active_requests: list) -> List[str]:
        """
        Distributed Lyapunov-drift scheduling.
        Each node independently decides which of its links to activate
        based on local queue backlog and AoE.
        """
        scheduled_links = []
        
        for node_id, state in self.node_states.items():
            # Local decision: which link to prioritize
            best_link = None
            best_weight = -float('inf')
            
            for link in self.network.get_node_links(node_id):
                neighbor = link.node_b if link.node_a == node_id else link.node_a
                
                # Lyapunov weight: queue_backlog * link_rate - V * AoE_cost
                # Queue backlog estimated from local pending requests
                queue_len = sum(1 for r in active_requests 
                              if self._node_on_path(node_id, neighbor, r))
                
                # AoE penalty for this link
                avg_aoe = self._get_link_avg_aoe(node_id, neighbor, current_time)
                
                weight = queue_len * link.success_prob - self.V * avg_aoe / state.coherence_time
                
                if weight > best_weight:
                    best_weight = weight
                    best_link = link.link_id
            
            if best_link and best_weight > 0:
                scheduled_links.append(best_link)
        
        return scheduled_links
    
    def should_refresh(self, node_id: str, slot_id: str, current_time: int) -> bool:
        """
        Proactive refresh: purely local decision.
        Refresh if AoE/T_coh > threshold.
        """
        state = self.node_states[node_id]
        if slot_id in state.local_memory_ages:
            age = state.local_memory_ages[slot_id]
            return age / state.coherence_time > self.refresh_threshold
        return False
    
    def should_swap(self, node_id: str, slot_a: str, slot_b: str,
                    current_time: int) -> bool:
        """
        Swap decision based on predicted end-to-end fidelity.
        Uses local fidelity estimates.
        """
        state = self.node_states[node_id]
        f_a = state.local_memory_fidelities.get(slot_a, 0.5)
        f_b = state.local_memory_fidelities.get(slot_b, 0.5)
        
        # Predicted swap fidelity
        f_swap = f_a * f_b + (1 - f_a) * (1 - f_b) / 3
        
        # Only swap if result exceeds threshold
        return f_swap > 0.65  # minimum useful fidelity
    
    def _node_on_path(self, node_id, neighbor, request):
        """Check if edge (node_id, neighbor) is on request's path."""
        if hasattr(request, 'path') and request.path:
            for i in range(len(request.path) - 1):
                if (request.path[i] == node_id and request.path[i+1] == neighbor) or \
                   (request.path[i] == neighbor and request.path[i+1] == node_id):
                    return True
        return False
    
    def _get_link_avg_aoe(self, node_a, node_b, current_time):
        """Get average AoE of entangled pairs on a link."""
        link = self.network.get_link(node_a, node_b)
        if link is None:
            return 0
        # Use local memory state
        state = self.node_states[node_a]
        ages = [v for k, v in state.local_memory_ages.items() 
                if k.startswith(f"{node_a}_{node_b}")]
        return np.mean(ages) if ages else 0
    
    def update_neighbor_state(self, node_id: str, neighbor_id: str, 
                              state_info: dict, delay: int):
        """Receive neighbor state update (arrives with delay)."""
        self.node_states[node_id].neighbor_states[neighbor_id] = state_info
        self.node_states[node_id].neighbor_state_age[neighbor_id] = delay


class DistributedGreedy:
    """
    Greedy protocol under distributed constraints.
    
    Greedy needs global state to make optimal swap decisions:
    - Which pairs to swap first (globally optimal ordering)
    - Which paths have the freshest entanglement end-to-end
    
    Under communication delay d_cc:
    - Global state arrives d_cc slots late
    - Decisions based on stale information
    - Entanglement may have decohered since state was reported
    
    Communication overhead: O(N) messages per time slot (broadcast)
    """
    
    def __init__(self, network, comm_delay=0):
        """
        Args:
            network: QuantumNetwork instance
            comm_delay: classical communication delay for global state
        """
        self.network = network
        self.comm_delay = comm_delay
        self.name = f"D-Greedy(d={comm_delay})"
        
        # Stale global state (delayed by comm_delay)
        self.stale_memory_states = {}  # node_id -> {slot_id: (age, fidelity)}
        self.state_timestamp = 0
    
    def get_name(self):
        return self.name
    
    def route_request(self, source: str, destination: str,
                      current_time: int) -> Optional[List[str]]:
        """
        Greedy routing: pick path with best estimated fidelity.
        But fidelity estimates are STALE by comm_delay slots.
        """
        # Use stale state to estimate link fidelities
        # The actual fidelity has decayed further since the report
        
        dist = {source: 1.0}  # fidelity (maximize)
        prev = {}
        visited = set()
        heap = [(-1.0, source)]  # max-heap via negation
        
        while heap:
            neg_f, u = heapq.heappop(heap)
            f_u = -neg_f
            if u in visited:
                continue
            visited.add(u)
            
            if u == destination:
                break
            
            for link in self.network.get_node_links(u):
                v = link.node_b if link.node_a == u else link.node_a
                if v in visited:
                    continue
                
                # Estimate link fidelity from stale state
                stale_fidelity = self._get_stale_link_fidelity(u, v, current_time)
                
                new_f = f_u * stale_fidelity
                if new_f > dist.get(v, 0):
                    dist[v] = new_f
                    prev[v] = u
                    heapq.heappush(heap, (-new_f, v))
        
        if destination not in prev and destination != source:
            return None
        
        path = []
        node = destination
        while node != source:
            path.append(node)
            node = prev.get(node)
            if node is None:
                return None
        path.append(source)
        path.reverse()
        return path
    
    def _get_stale_link_fidelity(self, node_a: str, node_b: str, 
                                  current_time: int) -> float:
        """
        Get fidelity estimate from stale global state.
        The reported fidelity was accurate at (current_time - comm_delay),
        but has decayed further since then.
        """
        link = self.network.get_link(node_a, node_b)
        if link is None:
            return 0.0
        
        if node_a in self.stale_memory_states:
            # Use stale info — this is what Greedy "thinks" the fidelity is
            stale_info = self.stale_memory_states[node_a]
            # But the ACTUAL fidelity is lower because of additional decay
            # Greedy doesn't know this — it makes decisions on stale data
            reported_fidelity = stale_info.get(f"{node_a}_{node_b}_fidelity", 
                                               link.initial_fidelity)
            return reported_fidelity
        
        return link.initial_fidelity
    
    def update_global_state(self, all_states: dict, current_time: int):
        """
        Receive global state broadcast (arrives with delay).
        In real network: this requires O(N) messages flooding.
        """
        self.stale_memory_states = all_states
        self.state_timestamp = current_time - self.comm_delay
    
    def schedule_links(self, current_time: int, active_requests: list) -> List[str]:
        """Greedy: activate all links with pending requests."""
        scheduled = []
        for link_id, link in self.network.links.items():
            scheduled.append(link_id)
        return scheduled
    
    def should_swap(self, node_id, slot_a, slot_b, current_time):
        """Greedy: always swap immediately."""
        return True


class QCASTProtocol:
    """
    Q-CAST: Quantum Concurrent Entanglement Routing.
    Based on Shi & Qian (SIGCOMM 2020).
    
    Key ideas:
    - Find multiple edge-disjoint paths for each request
    - Attempt entanglement generation on all paths concurrently
    - Use the first path that succeeds end-to-end
    - Recovery mechanism for partial failures
    
    Simplified implementation focusing on:
    - K-shortest edge-disjoint paths
    - Concurrent generation
    - Path-level success/failure
    """
    
    def __init__(self, network, K=3, recovery=True):
        """
        Args:
            network: QuantumNetwork instance
            K: number of concurrent paths to try
            recovery: whether to attempt recovery on partial success
        """
        self.network = network
        self.K = K
        self.recovery = recovery
        self.name = f"Q-CAST(K={K})"
    
    def get_name(self):
        return self.name
    
    def route_request(self, source: str, destination: str,
                      current_time: int) -> Optional[List[List[str]]]:
        """
        Find K edge-disjoint shortest paths.
        Returns list of paths (concurrent attempts).
        """
        paths = []
        used_edges = set()
        
        for _ in range(self.K):
            path = self._find_path_avoiding_edges(source, destination, used_edges)
            if path is None:
                break
            paths.append(path)
            # Mark edges as used
            for i in range(len(path) - 1):
                edge = tuple(sorted([path[i], path[i+1]]))
                used_edges.add(edge)
        
        return paths if paths else None
    
    def _find_path_avoiding_edges(self, source: str, destination: str,
                                   used_edges: Set[Tuple]) -> Optional[List[str]]:
        """BFS/Dijkstra avoiding used edges."""
        dist = {source: 0}
        prev = {}
        visited = set()
        heap = [(0, source)]
        
        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            
            if u == destination:
                break
            
            for link in self.network.get_node_links(u):
                v = link.node_b if link.node_a == u else link.node_a
                if v in visited:
                    continue
                
                edge = tuple(sorted([u, v]))
                if edge in used_edges:
                    continue
                
                cost = 1.0 / max(link.success_prob, 0.01)
                new_dist = d + cost
                if new_dist < dist.get(v, float('inf')):
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(heap, (new_dist, v))
        
        if destination not in prev and destination != source:
            return None
        
        path = []
        node = destination
        while node != source:
            path.append(node)
            node = prev.get(node)
            if node is None:
                return None
        path.append(source)
        path.reverse()
        return path
    
    def schedule_links(self, current_time: int, active_requests: list) -> List[str]:
        """Q-CAST: activate all links on any active path."""
        scheduled = set()
        for link_id in self.network.links:
            scheduled.add(link_id)
        return list(scheduled)
    
    def should_swap(self, node_id, slot_a, slot_b, current_time):
        """Q-CAST: swap when both segments ready."""
        return True


class RLRoutingProtocol:
    """
    DQN-based Routing Protocol (simplified).
    
    Uses a trained Q-table (tabular for tractability) that maps
    (current_node, destination, local_state_features) -> next_hop.
    
    Training: offline using simulated episodes.
    Features: node degree, avg memory age, queue length, hop distance to dest.
    
    This is a simplified version — real RL routing would use neural networks,
    but tabular Q-learning captures the key idea for comparison.
    """
    
    def __init__(self, network, epsilon=0.1, learning_rate=0.1, 
                 discount=0.95, pretrained=False):
        self.network = network
        self.epsilon = epsilon
        self.lr = learning_rate
        self.discount = discount
        self.name = "RL-Routing(DQN)"
        
        # Q-table: (node, destination, state_bucket) -> {next_hop: Q-value}
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Pre-train with shortest path as initialization
        if pretrained:
            self._pretrain_from_shortest_path()
    
    def get_name(self):
        return self.name
    
    def _pretrain_from_shortest_path(self):
        """Initialize Q-values from shortest path distances."""
        nodes = list(self.network.nodes.keys())
        for dest in nodes:
            # BFS from dest to get distances
            dist = {dest: 0}
            queue = [dest]
            while queue:
                u = queue.pop(0)
                for link in self.network.get_node_links(u):
                    v = link.node_b if link.node_a == u else link.node_a
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            
            # Set Q-values: prefer neighbors closer to destination
            for node in nodes:
                if node == dest:
                    continue
                for link in self.network.get_node_links(node):
                    neighbor = link.node_b if link.node_a == node else link.node_a
                    # Reward = negative distance (closer is better)
                    reward = -(dist.get(neighbor, 100))
                    self.q_table[(node, dest)][neighbor] = reward
    
    def route_request(self, source: str, destination: str,
                      current_time: int) -> Optional[List[str]]:
        """
        Route using Q-table with epsilon-greedy exploration.
        """
        path = [source]
        current = source
        visited = {source}
        max_hops = len(self.network.nodes) * 2
        
        for _ in range(max_hops):
            if current == destination:
                return path
            
            # Get valid next hops
            neighbors = []
            for link in self.network.get_node_links(current):
                v = link.node_b if link.node_a == current else link.node_a
                if v not in visited:
                    neighbors.append(v)
            
            if not neighbors:
                return None  # Dead end
            
            # Epsilon-greedy action selection
            if np.random.random() < self.epsilon:
                next_hop = np.random.choice(neighbors)
            else:
                # Pick neighbor with highest Q-value
                q_values = {n: self.q_table[(current, destination)].get(n, 0) 
                           for n in neighbors}
                next_hop = max(q_values, key=q_values.get)
            
            path.append(next_hop)
            visited.add(next_hop)
            current = next_hop
        
        return None  # Failed to reach destination
    
    def update_q_value(self, node, destination, next_hop, reward, next_node):
        """TD update for Q-learning."""
        old_q = self.q_table[(node, destination)].get(next_hop, 0)
        
        # Max Q-value from next state
        next_q_values = self.q_table[(next_node, destination)]
        max_next_q = max(next_q_values.values()) if next_q_values else 0
        
        # TD update
        new_q = old_q + self.lr * (reward + self.discount * max_next_q - old_q)
        self.q_table[(node, destination)][next_hop] = new_q
    
    def schedule_links(self, current_time, active_requests):
        """RL routing: activate all links."""
        return list(self.network.links.keys())
    
    def should_swap(self, node_id, slot_a, slot_b, current_time):
        """RL: always swap when possible."""
        return True
    
    def train_episode(self, source, destination, success, path, fidelity):
        """Update Q-values based on episode outcome."""
        if path is None:
            return
        
        # Reward based on outcome
        if success:
            reward = fidelity  # Higher fidelity = better reward
        else:
            reward = -1.0  # Penalty for failure
        
        # Backward update along path
        for i in range(len(path) - 1):
            node = path[i]
            next_hop = path[i + 1]
            # Discounted reward
            step_reward = reward * (self.discount ** (len(path) - 1 - i))
            next_node = path[i + 1]
            self.update_q_value(node, destination, next_hop, step_reward, next_node)
