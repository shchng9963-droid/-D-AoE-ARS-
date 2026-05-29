"""
Additional Experiments — Heterogeneous Networks and Hotspot Traffic
===================================================================
These scenarios stress the ROUTING dimension where AoE-ARS should shine.
"""

import numpy as np
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network_model import QuantumNetwork, QuantumLink
from protocols import AoEARS, ShortestPathRouting, FidelityAwareRouting, GreedyScheduling
from simulation_engine import SimulationConfig, QuantumNetworkSimulator


def create_heterogeneous_grid(rows=4, cols=4, seed=42):
    """Create a grid with highly variable link qualities.
    Some links are 'fiber' (high quality), others are 'free-space' (poor).
    """
    from network_model import QuantumNode, QuantumLink
    rng = np.random.default_rng(seed)
    network = QuantumNetwork()

    # Create nodes
    for i in range(rows * cols):
        node = QuantumNode(node_id=i, num_memory_slots=4, coherence_time=100.0)
        network.add_node(node)

    # Create links with heterogeneous quality
    link_id = 0
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            # Right neighbor
            if c < cols - 1:
                neighbor = r * cols + (c + 1)
                # 30% chance of being a "bad" link
                if rng.random() < 0.3:
                    p_succ, fid = 0.1, 0.85  # bad link
                else:
                    p_succ, fid = 0.6, 0.95  # good link
                link = QuantumLink(link_id=link_id, node_a=node, node_b=neighbor,
                                   success_prob=p_succ, attempt_rate=1.0,
                                   initial_fidelity=fid, distance_km=10.0)
                network.add_link(link)
                link_id += 1
            # Down neighbor
            if r < rows - 1:
                neighbor = (r + 1) * cols + c
                if rng.random() < 0.3:
                    p_succ, fid = 0.1, 0.85
                else:
                    p_succ, fid = 0.6, 0.95
                link = QuantumLink(link_id=link_id, node_a=node, node_b=neighbor,
                                   success_prob=p_succ, attempt_rate=1.0,
                                   initial_fidelity=fid, distance_km=10.0)
                network.add_link(link)
                link_id += 1

    return network


def create_hotspot_traffic(network, config, rng):
    """Generate traffic concentrated on a few source-destination pairs."""
    from network_model import Request, RequestStatus
    nodes = list(network.nodes.keys())
    n = len(nodes)

    # Hotspot: 70% of traffic goes to/from corner nodes
    corners = [0, 3, 12, 15]  # corners of 4x4 grid
    requests = []
    req_id = 0

    for t in range(config.num_time_slots):
        num_new = rng.poisson(config.request_arrival_rate)
        for _ in range(num_new):
            if rng.random() < 0.7:
                # Hotspot traffic: between corners
                src = int(rng.choice(corners))
                dst = int(rng.choice([c for c in corners if c != src]))
            else:
                # Background traffic: random
                src = int(rng.choice(nodes))
                dst = int(rng.choice([n for n in nodes if n != src]))

            requests.append((t, src, dst, req_id))
            req_id += 1

    return requests


class HotspotSimulator(QuantumNetworkSimulator):
    """Simulator with pre-generated hotspot traffic pattern."""

    def __init__(self, network, protocol, config, traffic):
        super().__init__(network, protocol, config)
        # Group traffic by time slot
        self.traffic_schedule = {}
        for (t, src, dst, rid) in traffic:
            if t not in self.traffic_schedule:
                self.traffic_schedule[t] = []
            self.traffic_schedule[t].append((src, dst, rid))

    def _generate_requests(self):
        """Override: use pre-generated traffic."""
        from network_model import Request, RequestStatus
        t = int(self.time)
        if t not in self.traffic_schedule:
            return

        for (src, dst, rid) in self.traffic_schedule[t]:
            if len(self.active_requests) >= self.config.max_concurrent_requests:
                break
            request = Request(
                request_id=rid,
                source=src,
                destination=dst,
                fidelity_threshold=self.config.fidelity_threshold,
                arrival_time=self.time,
                deadline=self.time + 500
            )
            self.active_requests.append(request)
            self.metrics.total_requests += 1


def experiment_6_heterogeneous():
    """Experiment 6: Heterogeneous link quality network."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Heterogeneous Network (4x4 grid, mixed link quality)")
    print("  30% of links are 'bad' (p=0.1, F=0.85), rest 'good' (p=0.6, F=0.95)")
    print("=" * 70)

    arrival_rates = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    protocols_list = ['aoe_ars', 'shortest_path', 'fidelity_aware', 'greedy']
    results = {p: {'fidelity': [], 'aoe': [], 'throughput': [], 'violation': [], 'latency': []}
               for p in protocols_list}

    for rate in arrival_rates:
        print(f"\n  Arrival rate = {rate:.2f}")
        for proto_name in protocols_list:
            network = create_heterogeneous_grid(4, 4, seed=42)

            if proto_name == 'aoe_ars':
                protocol = AoEARS(network, seed=42)
            elif proto_name == 'shortest_path':
                protocol = ShortestPathRouting(network, seed=42)
            elif proto_name == 'fidelity_aware':
                protocol = FidelityAwareRouting(network, seed=42, fidelity_threshold=0.65)
            elif proto_name == 'greedy':
                protocol = GreedyScheduling(network, seed=42)

            config = SimulationConfig(
                num_time_slots=2000,
                seed=42,
                request_arrival_rate=rate,
                fidelity_threshold=0.65,
                min_path_length=2,
                max_concurrent_requests=80
            )

            sim = QuantumNetworkSimulator(network, protocol, config)
            metrics = sim.run()
            s = metrics.summary()

            results[proto_name]['fidelity'].append(s['avg_fidelity'])
            results[proto_name]['aoe'].append(s['avg_aoe'])
            results[proto_name]['throughput'].append(s['throughput'])
            results[proto_name]['violation'].append(s['violation_rate'])
            results[proto_name]['latency'].append(s['avg_latency'])

            print(f"    {proto_name:20s}: del={s['delivered']:3d}/{s['total']:3d} "
                  f"fid={s['avg_fidelity']:.3f} aoe={s['avg_aoe']:.1f} "
                  f"lat={s['avg_latency']:.1f} viol={s['violation_rate']:.2f}")

    return {'arrival_rates': arrival_rates, 'results': results}


def experiment_7_hotspot():
    """Experiment 7: Hotspot traffic pattern."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Hotspot Traffic (4x4 grid)")
    print("  70% of traffic between corner nodes, creating bottlenecks")
    print("=" * 70)

    arrival_rates = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    protocols_list = ['aoe_ars', 'shortest_path', 'fidelity_aware', 'greedy']
    results = {p: {'fidelity': [], 'aoe': [], 'throughput': [], 'violation': [], 'latency': []}
               for p in protocols_list}

    for rate in arrival_rates:
        print(f"\n  Arrival rate = {rate:.2f}")

        # Generate traffic once, use for all protocols
        rng = np.random.default_rng(42)
        config = SimulationConfig(
            num_time_slots=2000,
            seed=42,
            request_arrival_rate=rate,
            fidelity_threshold=0.65,
            min_path_length=2,
            max_concurrent_requests=80
        )

        # Create a dummy network to generate traffic
        dummy_net = QuantumNetwork.create_grid(4, 4, memory_slots=4,
                                               coherence_time=100.0,
                                               link_success_prob=0.5,
                                               link_fidelity=0.95)
        traffic = create_hotspot_traffic(dummy_net, config, rng)

        for proto_name in protocols_list:
            network = QuantumNetwork.create_grid(4, 4, memory_slots=4,
                                                 coherence_time=100.0,
                                                 link_success_prob=0.5,
                                                 link_fidelity=0.95)

            if proto_name == 'aoe_ars':
                protocol = AoEARS(network, seed=42)
            elif proto_name == 'shortest_path':
                protocol = ShortestPathRouting(network, seed=42)
            elif proto_name == 'fidelity_aware':
                protocol = FidelityAwareRouting(network, seed=42, fidelity_threshold=0.65)
            elif proto_name == 'greedy':
                protocol = GreedyScheduling(network, seed=42)

            sim = HotspotSimulator(network, protocol, config, traffic)
            metrics = sim.run()
            s = metrics.summary()

            results[proto_name]['fidelity'].append(s['avg_fidelity'])
            results[proto_name]['aoe'].append(s['avg_aoe'])
            results[proto_name]['throughput'].append(s['throughput'])
            results[proto_name]['violation'].append(s['violation_rate'])
            results[proto_name]['latency'].append(s['avg_latency'])

            print(f"    {proto_name:20s}: del={s['delivered']:3d}/{s['total']:3d} "
                  f"fid={s['avg_fidelity']:.3f} aoe={s['avg_aoe']:.1f} "
                  f"lat={s['avg_latency']:.1f} viol={s['violation_rate']:.2f}")

    return {'arrival_rates': arrival_rates, 'results': results}


if __name__ == "__main__":
    print("AoE-ARS: Additional Experiments (Heterogeneous + Hotspot)")
    print("=" * 70)

    all_results = {}
    all_results['exp6_heterogeneous'] = experiment_6_heterogeneous()
    all_results['exp7_hotspot'] = experiment_7_hotspot()

    # Save
    output_path = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'results_additional.json')

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

    print(f"\nResults saved to {output_path}")
