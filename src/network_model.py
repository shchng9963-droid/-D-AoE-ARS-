"""
Quantum Network Model for AoE-ARS Simulator
============================================
Core data structures and physics models for quantum network simulation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import heapq


class SlotStatus(Enum):
    EMPTY = 0
    ENTANGLED = 1
    RESERVED = 2


class RequestStatus(Enum):
    PENDING = 0
    ROUTING = 1
    ASSEMBLING = 2
    DELIVERED = 3
    FAILED = 4


@dataclass
class MemorySlot:
    slot_id: int
    status: SlotStatus = SlotStatus.EMPTY
    partner_node: int = -1
    partner_slot: int = -1
    generation_time: float = -1.0
    initial_fidelity: float = 0.0
    assigned_request: int = -1
    link_id: int = -1

    def age(self, current_time: float) -> float:
        if self.status != SlotStatus.ENTANGLED:
            return 0.0
        return current_time - self.generation_time

    def current_fidelity(self, current_time: float, coherence_time: float) -> float:
        if self.status != SlotStatus.ENTANGLED:
            return 0.0
        tau = self.age(current_time)
        return fidelity_decay(self.initial_fidelity, tau, coherence_time)


@dataclass
class QuantumLink:
    link_id: int
    node_a: int
    node_b: int
    success_prob: float  # per-attempt success probability
    attempt_rate: float  # attempts per time slot
    initial_fidelity: float  # F_0 upon successful generation
    distance_km: float  # physical distance

    @property
    def expected_gen_time(self) -> float:
        """Expected time slots to generate one pair."""
        return 1.0 / (self.success_prob * self.attempt_rate)


@dataclass
class QuantumNode:
    node_id: int
    num_memory_slots: int
    coherence_time: float  # T_coh in time slots
    position: Tuple[float, float] = (0.0, 0.0)
    memory_slots: List[MemorySlot] = field(default_factory=list)
    neighbors: List[int] = field(default_factory=list)
    max_gen_per_slot: int = 2  # max parallel generation attempts

    def __post_init__(self):
        if not self.memory_slots:
            self.memory_slots = [
                MemorySlot(slot_id=i) for i in range(self.num_memory_slots)
            ]

    def free_slots(self) -> int:
        return sum(1 for s in self.memory_slots if s.status == SlotStatus.EMPTY)

    def occupied_slots(self) -> int:
        return sum(1 for s in self.memory_slots if s.status == SlotStatus.ENTANGLED)

    def get_free_slot(self) -> Optional[MemorySlot]:
        for s in self.memory_slots:
            if s.status == SlotStatus.EMPTY:
                return s
        return None

    def average_aoe(self, current_time: float) -> float:
        ages = [s.age(current_time) for s in self.memory_slots
                if s.status == SlotStatus.ENTANGLED]
        return np.mean(ages) if ages else 0.0


@dataclass
class Request:
    request_id: int
    source: int
    destination: int
    fidelity_threshold: float
    arrival_time: float
    deadline: float = float('inf')
    assigned_path: List[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.PENDING
    delivery_time: float = -1.0
    delivery_fidelity: float = 0.0
    delivery_aoe: float = 0.0


# ============================================================
# Physics Functions
# ============================================================

def fidelity_decay(F0: float, tau: float, T_coh: float) -> float:
    """Werner state fidelity decay under dephasing.
    F(tau) = 0.5 * (1 + (2*F0 - 1) * exp(-tau / T_coh))
    """
    if tau <= 0:
        return F0
    return 0.5 * (1.0 + (2.0 * F0 - 1.0) * np.exp(-tau / T_coh))


def swap_fidelity(F1: float, F2: float) -> float:
    """Fidelity after entanglement swapping of two Werner-state pairs.
    F_swap = F1*F2 + (1-F1)*(1-F2)/3
    """
    return F1 * F2 + (1.0 - F1) * (1.0 - F2) / 3.0


def chain_fidelity(fidelities: List[float]) -> float:
    """End-to-end fidelity for a chain of links after sequential swaps.
    Uses Werner parameter multiplication.
    """
    if not fidelities:
        return 0.0
    # Werner parameter: w = (4F - 1) / 3
    w_product = 1.0
    for F in fidelities:
        w = (4.0 * F - 1.0) / 3.0
        w_product *= w
    # Back to fidelity: F = (1 + 3w) / 4
    return (1.0 + 3.0 * w_product) / 4.0


def purification_output_fidelity(F1: float, F2: float) -> float:
    """Output fidelity of BBPSSW purification protocol."""
    num = F1 * F2 + (1 - F1) * (1 - F2) / 9.0
    denom = (F1 * F2 + F1 * (1 - F2) / 3.0 +
             (1 - F1) * F2 / 3.0 + 5 * (1 - F1) * (1 - F2) / 9.0)
    if denom == 0:
        return 0.5
    return num / denom


def purification_success_prob(F1: float, F2: float) -> float:
    """Success probability of BBPSSW purification."""
    return (F1 * F2 + F1 * (1 - F2) / 3.0 +
            (1 - F1) * F2 / 3.0 + 5 * (1 - F1) * (1 - F2) / 9.0)


# ============================================================
# Network Topology
# ============================================================

class QuantumNetwork:
    """Quantum network topology and state."""

    def __init__(self):
        self.nodes: Dict[int, QuantumNode] = {}
        self.links: Dict[int, QuantumLink] = {}
        self.link_map: Dict[Tuple[int, int], int] = {}  # (a,b) -> link_id
        self.time: float = 0.0

    def add_node(self, node: QuantumNode):
        self.nodes[node.node_id] = node

    def add_link(self, link: QuantumLink):
        self.links[link.link_id] = link
        self.link_map[(link.node_a, link.node_b)] = link.link_id
        self.link_map[(link.node_b, link.node_a)] = link.link_id
        # Update neighbor lists
        if link.node_b not in self.nodes[link.node_a].neighbors:
            self.nodes[link.node_a].neighbors.append(link.node_b)
        if link.node_a not in self.nodes[link.node_b].neighbors:
            self.nodes[link.node_b].neighbors.append(link.node_a)

    def get_link(self, node_a: int, node_b: int) -> Optional[QuantumLink]:
        lid = self.link_map.get((node_a, node_b))
        if lid is not None:
            return self.links[lid]
        return None

    def get_node_links(self, node_id) -> List[QuantumLink]:
        """Get all links connected to a node."""
        result = []
        for neighbor in self.nodes[node_id].neighbors:
            link = self.get_link(node_id, neighbor)
            if link is not None:
                result.append(link)
        return result

    def k_shortest_paths(self, source: int, dest: int, k: int = 5) -> List[List[int]]:
        """Yen's K-shortest paths algorithm."""
        # Dijkstra for shortest path
        def dijkstra(src, dst, excluded_nodes=set(), excluded_edges=set()):
            dist = {src: 0.0}
            prev = {src: None}
            visited = set()
            heap = [(0.0, src)]

            while heap:
                d, u = heapq.heappop(heap)
                if u in visited:
                    continue
                visited.add(u)
                if u == dst:
                    # Reconstruct path
                    path = []
                    node = dst
                    while node is not None:
                        path.append(node)
                        node = prev[node]
                    return path[::-1], d
                for v in self.nodes[u].neighbors:
                    if v in excluded_nodes or (u, v) in excluded_edges:
                        continue
                    link = self.get_link(u, v)
                    if link is None:
                        continue
                    # Weight: expected generation time (proxy for AoE contribution)
                    w = link.expected_gen_time
                    if d + w < dist.get(v, float('inf')):
                        dist[v] = d + w
                        prev[v] = u
                        heapq.heappush(heap, (d + w, v))
            return None, float('inf')

        # Yen's algorithm
        A = []  # K shortest paths
        B = []  # Candidate paths (heap)

        shortest, cost = dijkstra(source, dest)
        if shortest is None:
            return []
        A.append(shortest)

        for k_i in range(1, k):
            for i in range(len(A[-1]) - 1):
                spur_node = A[-1][i]
                root_path = A[-1][:i + 1]

                excluded_edges = set()
                for path in A:
                    if path[:i + 1] == root_path:
                        excluded_edges.add((path[i], path[i + 1]))

                excluded_nodes = set(root_path[:-1])

                spur_path, spur_cost = dijkstra(
                    spur_node, dest, excluded_nodes, excluded_edges
                )
                if spur_path is not None:
                    total_path = root_path[:-1] + spur_path
                    if total_path not in A:
                        heapq.heappush(B, (spur_cost + i, total_path))

            if not B:
                break
            _, next_path = heapq.heappop(B)
            A.append(next_path)

        return A

    # ============================================================
    # Topology Generators
    # ============================================================

    @classmethod
    def create_grid(cls, rows: int, cols: int,
                    memory_slots: int = 10,
                    coherence_time: float = 1000.0,
                    link_success_prob: float = 0.5,
                    link_fidelity: float = 0.95,
                    attempt_rate: float = 1.0) -> 'QuantumNetwork':
        """Create a grid topology."""
        net = cls()
        link_id = 0

        # Create nodes
        for r in range(rows):
            for c in range(cols):
                nid = r * cols + c
                node = QuantumNode(
                    node_id=nid,
                    num_memory_slots=memory_slots,
                    coherence_time=coherence_time,
                    position=(float(c), float(r))
                )
                net.add_node(node)

        # Create links (grid edges)
        for r in range(rows):
            for c in range(cols):
                nid = r * cols + c
                # Right neighbor
                if c + 1 < cols:
                    rid = r * cols + (c + 1)
                    link = QuantumLink(
                        link_id=link_id, node_a=nid, node_b=rid,
                        success_prob=link_success_prob,
                        attempt_rate=attempt_rate,
                        initial_fidelity=link_fidelity,
                        distance_km=10.0
                    )
                    net.add_link(link)
                    link_id += 1
                # Down neighbor
                if r + 1 < rows:
                    did = (r + 1) * cols + c
                    link = QuantumLink(
                        link_id=link_id, node_a=nid, node_b=did,
                        success_prob=link_success_prob,
                        attempt_rate=attempt_rate,
                        initial_fidelity=link_fidelity,
                        distance_km=10.0
                    )
                    net.add_link(link)
                    link_id += 1

        return net

    @classmethod
    def create_random(cls, num_nodes: int, avg_degree: float = 3.0,
                      memory_slots: int = 10,
                      coherence_time: float = 1000.0,
                      link_success_prob: float = 0.5,
                      link_fidelity: float = 0.95,
                      seed: int = 42) -> 'QuantumNetwork':
        """Create a random geometric graph topology."""
        rng = np.random.default_rng(seed)
        net = cls()

        # Place nodes randomly in unit square
        positions = rng.uniform(0, 1, size=(num_nodes, 2))

        for i in range(num_nodes):
            node = QuantumNode(
                node_id=i,
                num_memory_slots=memory_slots,
                coherence_time=coherence_time,
                position=(positions[i, 0], positions[i, 1])
            )
            net.add_node(node)

        # Connect nodes within radius (adjusted for desired avg degree)
        radius = np.sqrt(avg_degree / (num_nodes * np.pi))
        link_id = 0

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < radius:
                    # Distance-dependent success probability
                    p = link_success_prob * np.exp(-dist / radius)
                    link = QuantumLink(
                        link_id=link_id, node_a=i, node_b=j,
                        success_prob=max(0.1, p),
                        attempt_rate=1.0,
                        initial_fidelity=link_fidelity,
                        distance_km=dist * 100
                    )
                    net.add_link(link)
                    link_id += 1

        # Ensure connectivity (add edges if needed)
        # Simple check: BFS from node 0
        visited = set()
        queue = [0]
        visited.add(0)
        while queue:
            u = queue.pop(0)
            for v in net.nodes[u].neighbors:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        # Connect disconnected components
        for i in range(num_nodes):
            if i not in visited:
                # Connect to nearest visited node
                min_dist = float('inf')
                nearest = 0
                for j in visited:
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < min_dist:
                        min_dist = d
                        nearest = j
                link = QuantumLink(
                    link_id=link_id, node_a=i, node_b=nearest,
                    success_prob=link_success_prob * 0.5,
                    attempt_rate=1.0,
                    initial_fidelity=link_fidelity,
                    distance_km=min_dist * 100
                )
                net.add_link(link)
                link_id += 1
                visited.add(i)

        return net

    @classmethod
    def create_linear(cls, num_nodes: int,
                      memory_slots: int = 10,
                      coherence_time: float = 1000.0,
                      link_success_prob: float = 0.5,
                      link_fidelity: float = 0.95) -> 'QuantumNetwork':
        """Create a linear chain topology."""
        net = cls()
        for i in range(num_nodes):
            node = QuantumNode(
                node_id=i,
                num_memory_slots=memory_slots,
                coherence_time=coherence_time,
                position=(float(i), 0.0)
            )
            net.add_node(node)

        for i in range(num_nodes - 1):
            link = QuantumLink(
                link_id=i, node_a=i, node_b=i + 1,
                success_prob=link_success_prob,
                attempt_rate=1.0,
                initial_fidelity=link_fidelity,
                distance_km=10.0
            )
            net.add_link(link)

        return net


if __name__ == "__main__":
    # Quick test
    net = QuantumNetwork.create_grid(3, 3)
    print(f"Grid 3x3: {len(net.nodes)} nodes, {len(net.links)} links")
    paths = net.k_shortest_paths(0, 8, k=3)
    print(f"Paths from 0 to 8: {paths}")

    net2 = QuantumNetwork.create_linear(5)
    print(f"\nLinear 5: {len(net2.nodes)} nodes, {len(net2.links)} links")
    paths2 = net2.k_shortest_paths(0, 4, k=2)
    print(f"Paths from 0 to 4: {paths2}")

    # Test fidelity functions
    print(f"\nFidelity decay: F0=0.95, tau=100, T_coh=1000 -> {fidelity_decay(0.95, 100, 1000):.4f}")
    print(f"Swap fidelity: F1=0.9, F2=0.9 -> {swap_fidelity(0.9, 0.9):.4f}")
    print(f"Chain fidelity [0.95, 0.95, 0.95]: {chain_fidelity([0.95, 0.95, 0.95]):.4f}")
