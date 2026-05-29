"""
Discrete-Event Simulation Engine for Quantum Networks (v2)
===========================================================
Fixed: proper pair-request matching, segment-based swap tracking.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from network_model import (
    QuantumNetwork, QuantumNode, QuantumLink, MemorySlot, Request,
    SlotStatus, RequestStatus, fidelity_decay, swap_fidelity, chain_fidelity
)
from protocols import BaseProtocol, AoEARS, ShortestPathRouting, FidelityAwareRouting, GreedyScheduling


@dataclass
class SimulationConfig:
    num_time_slots: int = 5000
    request_arrival_rate: float = 0.1
    fidelity_threshold: float = 0.7
    seed: int = 42
    min_path_length: int = 2
    max_concurrent_requests: int = 50


@dataclass
class SimulationMetrics:
    delivery_fidelities: List[float] = field(default_factory=list)
    delivery_aoes: List[float] = field(default_factory=list)
    delivery_latencies: List[float] = field(default_factory=list)
    throughput_over_time: List[float] = field(default_factory=list)
    memory_utilization: List[float] = field(default_factory=list)
    fidelity_violations: int = 0
    total_requests: int = 0
    delivered_requests: int = 0
    failed_requests: int = 0

    @property
    def avg_fidelity(self):
        return float(np.mean(self.delivery_fidelities)) if self.delivery_fidelities else 0.0

    @property
    def avg_aoe(self):
        return float(np.mean(self.delivery_aoes)) if self.delivery_aoes else 0.0

    @property
    def avg_latency(self):
        return float(np.mean(self.delivery_latencies)) if self.delivery_latencies else 0.0

    @property
    def throughput(self):
        return self.delivered_requests / max(1, self.total_requests)

    @property
    def violation_rate(self):
        return self.fidelity_violations / max(1, self.delivered_requests)

    def summary(self) -> Dict:
        return {
            'avg_fidelity': self.avg_fidelity,
            'avg_aoe': self.avg_aoe,
            'avg_latency': self.avg_latency,
            'throughput': self.throughput,
            'violation_rate': self.violation_rate,
            'delivered': self.delivered_requests,
            'failed': self.failed_requests,
            'total': self.total_requests,
            'avg_memory_util': float(np.mean(self.memory_utilization)) if self.memory_utilization else 0.0
        }


@dataclass
class Segment:
    """Represents a contiguous entangled segment along a path.
    Initially each elementary link is a segment of length 1.
    Swaps merge adjacent segments.
    """
    left_node: int      # leftmost node of segment
    right_node: int     # rightmost node of segment
    fidelity: float     # current fidelity (at generation/swap time)
    generation_time: float  # time of oldest component
    request_id: int


class QuantumNetworkSimulator:
    """Discrete-event simulator v2 with segment-based tracking."""

    def __init__(self, network: QuantumNetwork, protocol: BaseProtocol,
                 config: SimulationConfig):
        self.network = network
        self.protocol = protocol
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.metrics = SimulationMetrics()
        self.time = 0.0
        self.active_requests: List[Request] = []
        self.request_counter = 0

        # Segment tracking: request_id -> list of Segments
        self.segments: Dict[int, List[Segment]] = {}

    def run(self) -> SimulationMetrics:
        for t in range(self.config.num_time_slots):
            self.time = float(t)
            self.protocol.time = self.time
            self.network.time = self.time

            self._generate_requests()
            self._route_requests()
            self._apply_discard()
            self._attempt_link_generation()
            self._attempt_swaps()
            self._check_deliveries()
            self._check_timeouts()
            self._record_metrics()

        return self.metrics

    def _generate_requests(self):
        if len(self.active_requests) >= self.config.max_concurrent_requests:
            return

        num_new = self.rng.poisson(self.config.request_arrival_rate)
        nodes = list(self.network.nodes.keys())

        for _ in range(num_new):
            if len(self.active_requests) >= self.config.max_concurrent_requests:
                break

            attempts = 0
            src, dst = None, None
            while attempts < 20:
                src = int(self.rng.choice(nodes))
                dst = int(self.rng.choice(nodes))
                if src != dst:
                    paths = self.network.k_shortest_paths(src, dst, k=1)
                    if paths and len(paths[0]) - 1 >= self.config.min_path_length:
                        break
                attempts += 1

            if attempts >= 20:
                continue

            request = Request(
                request_id=self.request_counter,
                source=src,
                destination=dst,
                fidelity_threshold=self.config.fidelity_threshold,
                arrival_time=self.time,
                deadline=self.time + 500
            )
            self.active_requests.append(request)
            self.metrics.total_requests += 1
            self.request_counter += 1

    def _route_requests(self):
        for request in self.active_requests:
            if request.status != RequestStatus.PENDING:
                continue

            path = self.protocol.select_path(request)
            if path is not None:
                request.assigned_path = [int(x) for x in path]
                request.status = RequestStatus.ASSEMBLING
                self.segments[request.request_id] = []
            else:
                request.status = RequestStatus.FAILED
                self.metrics.failed_requests += 1

    def _apply_discard(self):
        for node in self.network.nodes.values():
            slots_to_discard = self.protocol.discard_policy(node)
            for slot_id in slots_to_discard:
                self._free_slot(node, slot_id)

    def _attempt_link_generation(self):
        """Generate elementary links for active requests."""
        # For each assembling request, try to generate needed links
        for request in self.active_requests:
            if request.status != RequestStatus.ASSEMBLING:
                continue

            path = request.assigned_path
            existing_segments = self.segments.get(request.request_id, [])

            # Determine which elementary links still need generation
            covered = set()
            for seg in existing_segments:
                li = path.index(seg.left_node)
                ri = path.index(seg.right_node)
                for idx in range(li, ri):
                    covered.add(idx)

            # Try to generate uncovered links
            for i in range(len(path) - 1):
                if i in covered:
                    continue

                u, v = path[i], path[i + 1]
                link = self.network.get_link(u, v)
                if link is None:
                    continue

                node_u = self.network.nodes[u]
                node_v = self.network.nodes[v]

                # Need free memory at both ends
                if node_u.free_slots() == 0 or node_v.free_slots() == 0:
                    continue

                # Attempt generation (probabilistic)
                if self.rng.random() < link.success_prob * link.attempt_rate:
                    # Success — allocate memory slots
                    slot_u = node_u.get_free_slot()
                    slot_v = node_v.get_free_slot()

                    if slot_u and slot_v:
                        slot_u.status = SlotStatus.ENTANGLED
                        slot_u.partner_node = v
                        slot_u.partner_slot = slot_v.slot_id
                        slot_u.generation_time = self.time
                        slot_u.initial_fidelity = link.initial_fidelity
                        slot_u.assigned_request = request.request_id

                        slot_v.status = SlotStatus.ENTANGLED
                        slot_v.partner_node = u
                        slot_v.partner_slot = slot_u.slot_id
                        slot_v.generation_time = self.time
                        slot_v.initial_fidelity = link.initial_fidelity
                        slot_v.assigned_request = request.request_id

                        # Create segment
                        seg = Segment(
                            left_node=u, right_node=v,
                            fidelity=link.initial_fidelity,
                            generation_time=self.time,
                            request_id=request.request_id
                        )
                        existing_segments.append(seg)

    def _attempt_swaps(self):
        """Merge adjacent segments via entanglement swapping."""
        for request in self.active_requests:
            if request.status != RequestStatus.ASSEMBLING:
                continue

            path = request.assigned_path
            segments = self.segments.get(request.request_id, [])

            if len(segments) < 2:
                continue

            # Sort segments by position in path
            def seg_left_idx(seg):
                return path.index(seg.left_node)

            segments.sort(key=seg_left_idx)

            # Try to merge adjacent segments
            merged = True
            while merged:
                merged = False
                for i in range(len(segments) - 1):
                    seg_a = segments[i]
                    seg_b = segments[i + 1]

                    # Check if adjacent: seg_a.right_node == seg_b.left_node
                    if seg_a.right_node == seg_b.left_node:
                        swap_node = self.network.nodes[seg_a.right_node]

                        # Compute fidelities with aging
                        age_a = self.time - seg_a.generation_time
                        age_b = self.time - seg_b.generation_time
                        F_a = fidelity_decay(seg_a.fidelity, age_a, swap_node.coherence_time)
                        F_b = fidelity_decay(seg_b.fidelity, age_b, swap_node.coherence_time)

                        # Swap decision from protocol
                        decision = self.protocol.swap_decision(
                            swap_node, request,
                            self._seg_to_slot(seg_a, swap_node),
                            self._seg_to_slot(seg_b, swap_node)
                        )

                        if decision == 'SWAP_NOW':
                            # Perform swap
                            F_result = swap_fidelity(F_a, F_b)
                            oldest_time = min(seg_a.generation_time, seg_b.generation_time)

                            # Free memory at swap node
                            self._free_request_slots_at_node(
                                swap_node, request.request_id
                            )

                            # Create merged segment
                            new_seg = Segment(
                                left_node=seg_a.left_node,
                                right_node=seg_b.right_node,
                                fidelity=F_result,
                                generation_time=oldest_time,
                                request_id=request.request_id
                            )

                            segments.remove(seg_a)
                            segments.remove(seg_b)
                            segments.append(new_seg)
                            merged = True
                            break

                        elif decision == 'DISCARD_LEFT':
                            self._free_request_slots_at_node(swap_node, request.request_id)
                            segments.remove(seg_a)
                            merged = True
                            break

                        elif decision == 'DISCARD_RIGHT':
                            self._free_request_slots_at_node(swap_node, request.request_id)
                            segments.remove(seg_b)
                            merged = True
                            break

            self.segments[request.request_id] = segments

    def _check_deliveries(self):
        completed = []

        for request in self.active_requests:
            if request.status != RequestStatus.ASSEMBLING:
                continue

            path = request.assigned_path
            segments = self.segments.get(request.request_id, [])

            # Check if we have a single segment spanning the full path
            for seg in segments:
                if seg.left_node == path[0] and seg.right_node == path[-1]:
                    # End-to-end entanglement achieved!
                    age = self.time - seg.generation_time
                    src_node = self.network.nodes[path[0]]
                    fid = fidelity_decay(seg.fidelity, age, src_node.coherence_time)
                    aoe = age

                    self._deliver_request(request, fid, aoe)
                    completed.append(request)
                    break

        for req in completed:
            self.active_requests.remove(req)

    def _deliver_request(self, request: Request, fidelity: float, aoe: float):
        request.status = RequestStatus.DELIVERED
        request.delivery_time = self.time
        request.delivery_fidelity = fidelity
        request.delivery_aoe = aoe

        latency = self.time - request.arrival_time

        self.metrics.delivery_fidelities.append(fidelity)
        self.metrics.delivery_aoes.append(aoe)
        self.metrics.delivery_latencies.append(latency)
        self.metrics.delivered_requests += 1

        if fidelity < request.fidelity_threshold:
            self.metrics.fidelity_violations += 1

        # Cleanup
        self._cleanup_request(request)

    def _check_timeouts(self):
        timed_out = []
        for request in self.active_requests:
            if self.time > request.deadline:
                request.status = RequestStatus.FAILED
                self.metrics.failed_requests += 1
                self._cleanup_request(request)
                timed_out.append(request)

        for req in timed_out:
            self.active_requests.remove(req)

    def _record_metrics(self):
        total_slots = sum(n.num_memory_slots for n in self.network.nodes.values())
        used_slots = sum(n.occupied_slots() for n in self.network.nodes.values())
        self.metrics.memory_utilization.append(used_slots / max(1, total_slots))

    def _cleanup_request(self, request: Request):
        """Free all resources for a request."""
        rid = request.request_id
        # Free memory slots
        for node in self.network.nodes.values():
            for slot in node.memory_slots:
                if slot.assigned_request == rid:
                    slot.status = SlotStatus.EMPTY
                    slot.partner_node = -1
                    slot.assigned_request = -1
                    slot.generation_time = -1.0
        # Remove segments
        if rid in self.segments:
            del self.segments[rid]

    def _free_slot(self, node: QuantumNode, slot_id: int):
        slot = node.memory_slots[slot_id]
        slot.status = SlotStatus.EMPTY
        slot.partner_node = -1
        slot.assigned_request = -1
        slot.generation_time = -1.0

    def _free_request_slots_at_node(self, node: QuantumNode, request_id: int):
        """Free slots at a specific node for a request (after swap)."""
        for slot in node.memory_slots:
            if slot.assigned_request == request_id:
                slot.status = SlotStatus.EMPTY
                slot.partner_node = -1
                slot.assigned_request = -1
                slot.generation_time = -1.0

    def _seg_to_slot(self, seg: Segment, node: QuantumNode) -> Optional[MemorySlot]:
        """Create a virtual MemorySlot from a segment for protocol decisions."""
        # Find actual slot at this node for this request
        for slot in node.memory_slots:
            if slot.assigned_request == seg.request_id and slot.status == SlotStatus.ENTANGLED:
                return slot
        # Return a virtual slot with segment info
        virtual = MemorySlot(
            slot_id=-1,
            status=SlotStatus.ENTANGLED,
            generation_time=seg.generation_time,
            initial_fidelity=seg.fidelity,
            assigned_request=seg.request_id
        )
        return virtual


def run_experiment(topology: str, protocol_name: str,
                   config: SimulationConfig,
                   **topology_kwargs) -> SimulationMetrics:
    """Run a single experiment."""
    if topology == 'grid':
        rows = topology_kwargs.pop('rows', 4)
        cols = topology_kwargs.pop('cols', 4)
        network = QuantumNetwork.create_grid(rows, cols, **topology_kwargs)
    elif topology == 'linear':
        num_nodes = topology_kwargs.pop('num_nodes', 5)
        network = QuantumNetwork.create_linear(num_nodes, **topology_kwargs)
    elif topology == 'random':
        num_nodes = topology_kwargs.pop('num_nodes', 20)
        network = QuantumNetwork.create_random(num_nodes, **topology_kwargs)
    else:
        raise ValueError(f"Unknown topology: {topology}")

    if protocol_name == 'aoe_ars':
        protocol = AoEARS(network, seed=config.seed)
    elif protocol_name == 'shortest_path':
        protocol = ShortestPathRouting(network, seed=config.seed)
    elif protocol_name == 'fidelity_aware':
        protocol = FidelityAwareRouting(network, seed=config.seed,
                                        fidelity_threshold=config.fidelity_threshold)
    elif protocol_name == 'greedy':
        protocol = GreedyScheduling(network, seed=config.seed)
    else:
        raise ValueError(f"Unknown protocol: {protocol_name}")

    simulator = QuantumNetworkSimulator(network, protocol, config)
    return simulator.run()


if __name__ == "__main__":
    print("=" * 70)
    print("AoE-ARS Quantum Network Simulator - Validation Run")
    print("=" * 70)

    # Test on linear topology (simple, predictable)
    config = SimulationConfig(
        num_time_slots=1000,
        seed=42,
        request_arrival_rate=0.05,
        fidelity_threshold=0.6,
        min_path_length=2
    )

    print("\n--- Linear Topology (5 nodes) ---")
    for proto in ['aoe_ars', 'shortest_path', 'fidelity_aware', 'greedy']:
        metrics = run_experiment('linear', proto, config, num_nodes=5,
                                link_success_prob=0.6, coherence_time=500.0)
        s = metrics.summary()
        print(f"  {proto:20s}: delivered={s['delivered']:3d}/{s['total']:3d}, "
              f"fid={s['avg_fidelity']:.3f}, "
              f"aoe={s['avg_aoe']:.1f}, "
              f"lat={s['avg_latency']:.1f}, "
              f"viol={s['violation_rate']:.2f}")

    print("\n--- Grid Topology (3x3) ---")
    config2 = SimulationConfig(
        num_time_slots=1000,
        seed=42,
        request_arrival_rate=0.08,
        fidelity_threshold=0.6,
        min_path_length=2
    )
    for proto in ['aoe_ars', 'shortest_path', 'fidelity_aware', 'greedy']:
        metrics = run_experiment('grid', proto, config2, rows=3, cols=3,
                                link_success_prob=0.6, coherence_time=500.0)
        s = metrics.summary()
        print(f"  {proto:20s}: delivered={s['delivered']:3d}/{s['total']:3d}, "
              f"fid={s['avg_fidelity']:.3f}, "
              f"aoe={s['avg_aoe']:.1f}, "
              f"lat={s['avg_latency']:.1f}, "
              f"viol={s['violation_rate']:.2f}")
