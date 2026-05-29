"""
Routing and Scheduling Protocols for Quantum Network Simulation
================================================================
Implements:
1. AoE-ARS (our protocol)
2. Shortest Path Routing (baseline)
3. Fidelity-Aware Routing (baseline)
4. Greedy Scheduling (baseline)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from network_model import (
    QuantumNetwork, QuantumNode, QuantumLink, MemorySlot,
    Request, SlotStatus, RequestStatus,
    fidelity_decay, swap_fidelity, chain_fidelity
)


class BaseProtocol:
    """Base class for routing/scheduling protocols."""

    def __init__(self, network: QuantumNetwork, seed: int = 42):
        self.network = network
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self.delivered = []  # completed requests
        self.failed = []  # failed requests

    def select_path(self, request: Request) -> Optional[List[int]]:
        raise NotImplementedError

    def schedule_links(self, node: QuantumNode) -> List[Tuple[int, int]]:
        """Return list of (node_a, node_b) links to attempt generation."""
        raise NotImplementedError

    def swap_decision(self, node: QuantumNode, request: Request,
                      left_pair: Optional[MemorySlot],
                      right_pair: Optional[MemorySlot]) -> str:
        """Return: 'SWAP_NOW', 'WAIT', 'DISCARD_LEFT', 'DISCARD_RIGHT'"""
        raise NotImplementedError

    def discard_policy(self, node: QuantumNode) -> List[int]:
        """Return list of slot_ids to discard."""
        raise NotImplementedError


# ============================================================
# PROTOCOL 1: AoE-ARS (Our Method)
# ============================================================

class AoEARS(BaseProtocol):
    """Age-of-Entanglement Aware Adaptive Routing and Scheduling."""

    def __init__(self, network: QuantumNetwork, seed: int = 42,
                 alpha: float = 1.0, beta: float = 0.5,
                 base_discard_threshold: float = 0.7,
                 congestion_boost: float = 0.15,
                 state_exchange_period: int = 10,
                 k_paths: int = 5):
        super().__init__(network, seed)
        self.alpha = alpha
        self.beta = beta
        self.base_discard_threshold = base_discard_threshold
        self.congestion_boost = congestion_boost
        self.state_exchange_period = state_exchange_period
        self.k_paths = k_paths

    def predict_aoe(self, path: List[int]) -> Tuple[float, float]:
        """Predict expected AoE and fidelity at delivery for a path."""
        k = len(path) - 1  # number of hops
        if k == 0:
            return 0.0, 1.0

        ready_times = []
        initial_ages = []
        link_fidelities = []

        for i in range(k):
            u, v = path[i], path[i + 1]
            link = self.network.get_link(u, v)
            if link is None:
                return float('inf'), 0.0

            # Check if there's already a stored pair for this link
            stored_pair = self._find_stored_pair(u, v)
            if stored_pair is not None:
                ready_times.append(0.0)
                initial_ages.append(stored_pair.age(self.time))
            else:
                # Expected generation time, adjusted for free memory
                node_u = self.network.nodes[u]
                node_v = self.network.nodes[v]
                free_factor = min(
                    node_u.free_slots() / max(1, node_u.num_memory_slots),
                    node_v.free_slots() / max(1, node_v.num_memory_slots)
                )
                if free_factor == 0:
                    ready_times.append(float('inf'))
                else:
                    ready_times.append(link.expected_gen_time / free_factor)
                initial_ages.append(0.0)

            link_fidelities.append(link.initial_fidelity)

        # Assembly time: max ready time + swap levels
        max_ready = max(ready_times)
        if max_ready == float('inf'):
            return float('inf'), 0.0

        # Binary tree swap: ceil(log2(k)) levels, each takes ~1 time slot
        swap_levels = int(np.ceil(np.log2(max(k, 1))))
        assembly_time = max_ready + swap_levels

        # Predicted AoE: oldest link ages for the full assembly time
        min_ready_idx = int(np.argmin(ready_times))
        oldest_age = assembly_time - ready_times[min_ready_idx] + initial_ages[min_ready_idx]
        predicted_aoe = oldest_age

        # Predicted fidelity: account for aging of each link
        aged_fidelities = []
        for i in range(k):
            age_at_swap = assembly_time - ready_times[i] + initial_ages[i]
            node = self.network.nodes[path[i]]
            F_aged = fidelity_decay(link_fidelities[i], age_at_swap, node.coherence_time)
            aged_fidelities.append(F_aged)

        predicted_fidelity = chain_fidelity(aged_fidelities)

        return predicted_aoe, predicted_fidelity

    def select_path(self, request: Request) -> Optional[List[int]]:
        """Select path minimizing predicted AoE subject to fidelity constraint."""
        candidates = self.network.k_shortest_paths(
            request.source, request.destination, k=self.k_paths
        )

        if not candidates:
            return None

        viable = []
        for path in candidates:
            pred_aoe, pred_fid = self.predict_aoe(path)
            if pred_fid >= request.fidelity_threshold:
                # Score: minimize AoE with fidelity bonus
                score = self.alpha * pred_aoe - self.beta * (pred_fid - request.fidelity_threshold)
                viable.append((path, score, pred_aoe, pred_fid))

        if not viable:
            # Relax: pick path with best fidelity even if below threshold
            best_fid = -1
            best_path = None
            for path in candidates:
                _, pred_fid = self.predict_aoe(path)
                if pred_fid > best_fid:
                    best_fid = pred_fid
                    best_path = path
            return best_path

        # Sort by score (lower is better)
        viable.sort(key=lambda x: x[1])
        return viable[0][0]

    def schedule_links(self, node: QuantumNode) -> List[Tuple[int, int]]:
        """AoE-weighted link scheduling."""
        pending = []

        for neighbor_id in node.neighbors:
            link = self.network.get_link(node.node_id, neighbor_id)
            if link is None:
                continue

            # Check if we need a pair on this link
            existing = self._find_stored_pair(node.node_id, neighbor_id)
            if existing is not None:
                continue  # already have a pair

            if node.free_slots() == 0:
                continue  # no memory available

            neighbor = self.network.nodes[neighbor_id]
            if neighbor.free_slots() == 0:
                continue

            # Priority: based on AoE urgency of requests using this link
            priority = self._link_priority(node, neighbor_id)
            pending.append((priority, node.node_id, neighbor_id))

        # Sort by priority (descending)
        pending.sort(key=lambda x: -x[0])

        # Return top max_gen_per_slot links
        result = []
        for i in range(min(node.max_gen_per_slot, len(pending))):
            result.append((pending[i][1], pending[i][2]))

        return result

    def swap_decision(self, node: QuantumNode, request: Request,
                      left_pair: Optional[MemorySlot],
                      right_pair: Optional[MemorySlot]) -> str:
        """Freshness-aware swap decision."""
        if left_pair is not None and right_pair is not None:
            return 'SWAP_NOW'

        if left_pair is None and right_pair is None:
            return 'WAIT'

        # One pair available, other pending
        available = left_pair if left_pair is not None else right_pair
        tau = available.age(self.time)
        T_coh = node.coherence_time

        # Compute threshold: should we discard and regenerate?
        current_fid = fidelity_decay(available.initial_fidelity, tau, T_coh)

        # Estimate time for the other pair
        if left_pair is None:
            # Need left pair
            path_idx = request.assigned_path.index(node.node_id)
            if path_idx > 0:
                other_node = request.assigned_path[path_idx - 1]
                link = self.network.get_link(node.node_id, other_node)
                expected_wait = link.expected_gen_time if link else 10.0
            else:
                expected_wait = 5.0
        else:
            # Need right pair
            path_idx = request.assigned_path.index(node.node_id)
            if path_idx < len(request.assigned_path) - 1:
                other_node = request.assigned_path[path_idx + 1]
                link = self.network.get_link(node.node_id, other_node)
                expected_wait = link.expected_gen_time if link else 10.0
            else:
                expected_wait = 5.0

        # Predicted fidelity at swap time if we wait
        predicted_fid_wait = fidelity_decay(
            available.initial_fidelity, tau + expected_wait, T_coh
        )

        # If waiting would drop below threshold, discard
        if predicted_fid_wait < self.base_discard_threshold:
            return 'DISCARD_LEFT' if left_pair is not None else 'DISCARD_RIGHT'

        return 'WAIT'

    def discard_policy(self, node: QuantumNode) -> List[int]:
        """Adaptive discard based on predicted fidelity at consumption."""
        to_discard = []
        congestion = 1.0 - (node.free_slots() / max(1, node.num_memory_slots))
        threshold = self.base_discard_threshold + congestion * self.congestion_boost

        for slot in node.memory_slots:
            if slot.status != SlotStatus.ENTANGLED:
                continue

            current_fid = slot.current_fidelity(self.time, node.coherence_time)

            if current_fid < threshold:
                to_discard.append(slot.slot_id)

        return to_discard

    def _find_stored_pair(self, node_a: int, node_b: int) -> Optional[MemorySlot]:
        """Find a stored entangled pair between two nodes."""
        node = self.network.nodes[node_a]
        for slot in node.memory_slots:
            if slot.status == SlotStatus.ENTANGLED and slot.partner_node == node_b:
                return slot
        return None

    def _link_priority(self, node: QuantumNode, neighbor_id: int) -> float:
        """Compute priority for generating a link based on AoE urgency."""
        # Base priority: proactive generation
        priority = 0.1

        # Check if any active request needs this link
        # (In full implementation, would check request assignments)
        # For now, use average AoE as proxy
        avg_aoe = node.average_aoe(self.time)
        priority += avg_aoe / node.coherence_time

        return priority


# ============================================================
# PROTOCOL 2: Shortest Path Routing (Baseline)
# ============================================================

class ShortestPathRouting(BaseProtocol):
    """Simple shortest-path routing with FIFO scheduling."""

    def select_path(self, request: Request) -> Optional[List[int]]:
        """Select shortest path (minimum hops)."""
        paths = self.network.k_shortest_paths(
            request.source, request.destination, k=1
        )
        return paths[0] if paths else None

    def schedule_links(self, node: QuantumNode) -> List[Tuple[int, int]]:
        """Round-robin link scheduling."""
        result = []
        for neighbor_id in node.neighbors:
            if node.free_slots() == 0:
                break
            existing = self._find_stored_pair(node.node_id, neighbor_id)
            if existing is None:
                neighbor = self.network.nodes[neighbor_id]
                if neighbor.free_slots() > 0:
                    result.append((node.node_id, neighbor_id))
                    if len(result) >= node.max_gen_per_slot:
                        break
        return result

    def swap_decision(self, node: QuantumNode, request: Request,
                      left_pair: Optional[MemorySlot],
                      right_pair: Optional[MemorySlot]) -> str:
        """Swap immediately when both pairs available."""
        if left_pair is not None and right_pair is not None:
            return 'SWAP_NOW'
        return 'WAIT'

    def discard_policy(self, node: QuantumNode) -> List[int]:
        """Discard only when memory is full and pair is very old."""
        to_discard = []
        if node.free_slots() == 0:
            # Find oldest pair
            oldest_slot = None
            oldest_age = 0
            for slot in node.memory_slots:
                if slot.status == SlotStatus.ENTANGLED:
                    age = slot.age(self.time)
                    if age > oldest_age:
                        oldest_age = age
                        oldest_slot = slot
            if oldest_slot and oldest_age > 3 * node.coherence_time:
                to_discard.append(oldest_slot.slot_id)
        return to_discard

    def _find_stored_pair(self, node_a: int, node_b: int) -> Optional[MemorySlot]:
        node = self.network.nodes[node_a]
        for slot in node.memory_slots:
            if slot.status == SlotStatus.ENTANGLED and slot.partner_node == node_b:
                return slot
        return None


# ============================================================
# PROTOCOL 3: Fidelity-Aware Routing (Baseline)
# ============================================================

class FidelityAwareRouting(BaseProtocol):
    """Routes based on static fidelity estimates (no AoE awareness)."""

    def __init__(self, network: QuantumNetwork, seed: int = 42,
                 fidelity_threshold: float = 0.7):
        super().__init__(network, seed)
        self.fidelity_threshold = fidelity_threshold

    def select_path(self, request: Request) -> Optional[List[int]]:
        """Select path maximizing static end-to-end fidelity."""
        candidates = self.network.k_shortest_paths(
            request.source, request.destination, k=5
        )

        if not candidates:
            return None

        best_path = None
        best_fidelity = -1

        for path in candidates:
            # Static fidelity estimate (no aging)
            link_fids = []
            for i in range(len(path) - 1):
                link = self.network.get_link(path[i], path[i + 1])
                if link:
                    link_fids.append(link.initial_fidelity)
                else:
                    link_fids.append(0.5)

            e2e_fid = chain_fidelity(link_fids)
            if e2e_fid > best_fidelity:
                best_fidelity = e2e_fid
                best_path = path

        return best_path

    def schedule_links(self, node: QuantumNode) -> List[Tuple[int, int]]:
        """Same as shortest path — round robin."""
        result = []
        for neighbor_id in node.neighbors:
            if node.free_slots() == 0:
                break
            existing = self._find_stored_pair(node.node_id, neighbor_id)
            if existing is None:
                neighbor = self.network.nodes[neighbor_id]
                if neighbor.free_slots() > 0:
                    result.append((node.node_id, neighbor_id))
                    if len(result) >= node.max_gen_per_slot:
                        break
        return result

    def swap_decision(self, node: QuantumNode, request: Request,
                      left_pair: Optional[MemorySlot],
                      right_pair: Optional[MemorySlot]) -> str:
        """Swap when both available and fidelity above threshold."""
        if left_pair is not None and right_pair is not None:
            F1 = left_pair.current_fidelity(self.time, node.coherence_time)
            F2 = right_pair.current_fidelity(self.time, node.coherence_time)
            if swap_fidelity(F1, F2) >= self.fidelity_threshold:
                return 'SWAP_NOW'
            else:
                # Discard the worse one
                if F1 < F2:
                    return 'DISCARD_LEFT'
                else:
                    return 'DISCARD_RIGHT'
        return 'WAIT'

    def discard_policy(self, node: QuantumNode) -> List[int]:
        """Discard pairs below fidelity threshold."""
        to_discard = []
        for slot in node.memory_slots:
            if slot.status == SlotStatus.ENTANGLED:
                fid = slot.current_fidelity(self.time, node.coherence_time)
                if fid < self.fidelity_threshold:
                    to_discard.append(slot.slot_id)
        return to_discard

    def _find_stored_pair(self, node_a: int, node_b: int) -> Optional[MemorySlot]:
        node = self.network.nodes[node_a]
        for slot in node.memory_slots:
            if slot.status == SlotStatus.ENTANGLED and slot.partner_node == node_b:
                return slot
        return None


# ============================================================
# PROTOCOL 4: Greedy Scheduling (Baseline)
# ============================================================

class GreedyScheduling(BaseProtocol):
    """Greedy: always generate, swap immediately, no discard until forced."""

    def select_path(self, request: Request) -> Optional[List[int]]:
        """Shortest path."""
        paths = self.network.k_shortest_paths(
            request.source, request.destination, k=1
        )
        return paths[0] if paths else None

    def schedule_links(self, node: QuantumNode) -> List[Tuple[int, int]]:
        """Generate on ALL links with free memory."""
        result = []
        for neighbor_id in node.neighbors:
            if node.free_slots() == 0:
                break
            existing = self._find_stored_pair(node.node_id, neighbor_id)
            if existing is None:
                neighbor = self.network.nodes[neighbor_id]
                if neighbor.free_slots() > 0:
                    result.append((node.node_id, neighbor_id))
        # No limit on parallel generation (greedy)
        return result[:node.max_gen_per_slot]

    def swap_decision(self, node: QuantumNode, request: Request,
                      left_pair: Optional[MemorySlot],
                      right_pair: Optional[MemorySlot]) -> str:
        """Always swap immediately."""
        if left_pair is not None and right_pair is not None:
            return 'SWAP_NOW'
        return 'WAIT'

    def discard_policy(self, node: QuantumNode) -> List[int]:
        """Never discard unless memory completely full."""
        if node.free_slots() > 0:
            return []
        # Discard oldest
        oldest_slot = None
        oldest_age = 0
        for slot in node.memory_slots:
            if slot.status == SlotStatus.ENTANGLED:
                age = slot.age(self.time)
                if age > oldest_age:
                    oldest_age = age
                    oldest_slot = slot
        if oldest_slot:
            return [oldest_slot.slot_id]
        return []

    def _find_stored_pair(self, node_a: int, node_b: int) -> Optional[MemorySlot]:
        node = self.network.nodes[node_a]
        for slot in node.memory_slots:
            if slot.status == SlotStatus.ENTANGLED and slot.partner_node == node_b:
                return slot
        return None
