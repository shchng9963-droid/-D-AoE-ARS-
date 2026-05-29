"""
Real-world and synthetic quantum network topologies.

Includes:
1. NSFNET (14 nodes) - US backbone
2. SURFnet (50 nodes) - European research network (approximated)
3. Waxman random graphs (scalable)
4. COST-239 (11 nodes) - European optical network
5. IBM Q-Network topology (approximated)
"""


import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO_ROOT = _os.path.dirname(_HERE)
EXPERIMENTS_DIR = _os.path.join(REPO_ROOT, "experiments")
FIGURES_DIR = _os.path.join(REPO_ROOT, "figures")
_os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
_os.makedirs(FIGURES_DIR, exist_ok=True)

import numpy as np
import sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from network_model import QuantumNetwork, QuantumNode, QuantumLink


def create_nsfnet(num_memory_slots=4, coherence_time=100.0):
    """
    NSFNET topology (14 nodes, 21 links).
    US backbone network widely used in networking research.
    
    Node positions approximate real geographic locations.
    Link distances derived from geographic separation.
    """
    network = QuantumNetwork()
    
    # 14 nodes (major US cities)
    node_names = [
        "WA", "CA1", "CA2", "UT", "CO", "NE", "TX",
        "IL", "MI", "PA", "NY", "GA", "DC", "NJ"
    ]
    
    # Approximate distances (km) determine link quality
    # Shorter links → higher success probability
    for name in node_names:
        network.add_node(QuantumNode(name, num_memory_slots, coherence_time))
    
    # NSFNET edges with approximate distances (km)
    edges = [
        ("WA", "CA1", 1100), ("WA", "UT", 1200), ("WA", "NE", 2800),
        ("CA1", "CA2", 600), ("CA1", "UT", 1000),
        ("CA2", "TX", 2000), ("CA2", "CO", 1400),
        ("UT", "CO", 700), ("CO", "NE", 800),
        ("NE", "IL", 750), ("NE", "TX", 1200),
        ("TX", "GA", 1300), ("TX", "IL", 1500),
        ("IL", "MI", 400), ("IL", "PA", 900),
        ("MI", "NY", 800), ("MI", "NJ", 900),
        ("PA", "NY", 200), ("PA", "NJ", 150),
        ("NY", "DC", 350), ("GA", "DC", 900)
    ]
    
    for node_a, node_b, dist_km in edges:
        # Success probability decreases with distance
        # p = exp(-distance / L_att), L_att ~ 20km for fiber
        # But with repeaters every ~50km, effective p is higher
        p_success = max(0.1, np.exp(-dist_km / 3000))  # effective with repeaters
        initial_fidelity = 0.95 - 0.0001 * dist_km  # slight degradation
        initial_fidelity = max(0.85, initial_fidelity)
        
        link_id = f"{node_a}-{node_b}"
        network.add_link(QuantumLink(
            link_id, node_a, node_b,
            success_prob=p_success,
            attempt_rate=1.0,
            initial_fidelity=initial_fidelity,
            distance_km=dist_km
        ))
    
    return network


def create_surfnet(num_memory_slots=4, coherence_time=100.0):
    """
    SURFnet-inspired topology (50 nodes).
    Based on European research network structure.
    Uses hierarchical design: core (8) + metro (16) + access (26).
    """
    network = QuantumNetwork()
    np.random.seed(42)  # reproducible
    
    # Core nodes (8) - major hubs
    core_nodes = [f"C{i}" for i in range(8)]
    # Metro nodes (16) - regional hubs
    metro_nodes = [f"M{i}" for i in range(16)]
    # Access nodes (26) - end points
    access_nodes = [f"A{i}" for i in range(26)]
    
    all_nodes = core_nodes + metro_nodes + access_nodes
    
    for name in all_nodes:
        # Core nodes have more memory
        if name.startswith("C"):
            slots = num_memory_slots * 2
        elif name.startswith("M"):
            slots = num_memory_slots
        else:
            slots = max(2, num_memory_slots - 1)
        network.add_node(QuantumNode(name, slots, coherence_time))
    
    link_count = 0
    
    # Core ring + cross connections (high quality, short distance)
    for i in range(len(core_nodes)):
        j = (i + 1) % len(core_nodes)
        dist = np.random.uniform(100, 400)
        link_id = f"L{link_count}"
        network.add_link(QuantumLink(
            link_id, core_nodes[i], core_nodes[j],
            success_prob=np.random.uniform(0.7, 0.95),
            attempt_rate=1.0,
            initial_fidelity=np.random.uniform(0.90, 0.97),
            distance_km=dist
        ))
        link_count += 1
    
    # Core cross-links (4 additional)
    core_cross = [(0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in core_cross:
        dist = np.random.uniform(200, 600)
        link_id = f"L{link_count}"
        network.add_link(QuantumLink(
            link_id, core_nodes[i], core_nodes[j],
            success_prob=np.random.uniform(0.6, 0.85),
            attempt_rate=1.0,
            initial_fidelity=np.random.uniform(0.88, 0.95),
            distance_km=dist
        ))
        link_count += 1
    
    # Metro nodes connect to 2 core nodes each
    for i, metro in enumerate(metro_nodes):
        core1 = core_nodes[i % len(core_nodes)]
        core2 = core_nodes[(i + 1) % len(core_nodes)]
        
        for core in [core1, core2]:
            dist = np.random.uniform(50, 200)
            link_id = f"L{link_count}"
            network.add_link(QuantumLink(
                link_id, metro, core,
                success_prob=np.random.uniform(0.5, 0.8),
                attempt_rate=1.0,
                initial_fidelity=np.random.uniform(0.87, 0.94),
                distance_km=dist
            ))
            link_count += 1
    
    # Metro-metro links (some)
    metro_links = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)]
    for i, j in metro_links:
        dist = np.random.uniform(30, 150)
        link_id = f"L{link_count}"
        network.add_link(QuantumLink(
            link_id, metro_nodes[i], metro_nodes[j],
            success_prob=np.random.uniform(0.5, 0.75),
            attempt_rate=1.0,
            initial_fidelity=np.random.uniform(0.86, 0.93),
            distance_km=dist
        ))
        link_count += 1
    
    # Access nodes connect to 1-2 metro nodes
    for i, access in enumerate(access_nodes):
        metro1 = metro_nodes[i % len(metro_nodes)]
        dist = np.random.uniform(10, 80)
        link_id = f"L{link_count}"
        network.add_link(QuantumLink(
            link_id, access, metro1,
            success_prob=np.random.uniform(0.4, 0.7),
            attempt_rate=1.0,
            initial_fidelity=np.random.uniform(0.85, 0.92),
            distance_km=dist
        ))
        link_count += 1
        
        # Some access nodes have redundant connection
        if i % 3 == 0:
            metro2 = metro_nodes[(i + 1) % len(metro_nodes)]
            dist = np.random.uniform(20, 100)
            link_id = f"L{link_count}"
            network.add_link(QuantumLink(
                link_id, access, metro2,
                success_prob=np.random.uniform(0.3, 0.6),
                attempt_rate=1.0,
                initial_fidelity=np.random.uniform(0.83, 0.90),
                distance_km=dist
            ))
            link_count += 1
    
    return network


def create_waxman_graph(num_nodes=100, alpha=0.4, beta=0.4,
                        num_memory_slots=4, coherence_time=100.0,
                        seed=42):
    """
    Waxman random graph model for quantum networks.
    
    P(edge u,v) = alpha * exp(-d(u,v) / (beta * L))
    where L is maximum distance between any two nodes.
    
    Commonly used in networking research for realistic topology generation.
    """
    np.random.seed(seed)
    network = QuantumNetwork()
    
    # Place nodes randomly in 1000km x 1000km area
    positions = np.random.uniform(0, 1000, size=(num_nodes, 2))
    
    # Create nodes
    node_names = [f"N{i}" for i in range(num_nodes)]
    for name in node_names:
        network.add_node(QuantumNode(name, num_memory_slots, coherence_time))
    
    # Compute pairwise distances
    max_dist = 0
    distances = {}
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            d = np.sqrt(np.sum((positions[i] - positions[j])**2))
            distances[(i, j)] = d
            max_dist = max(max_dist, d)
    
    # Create edges according to Waxman model
    link_count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            d = distances[(i, j)]
            prob = alpha * np.exp(-d / (beta * max_dist))
            
            if np.random.random() < prob:
                # Link quality depends on distance
                p_success = max(0.1, 0.9 * np.exp(-d / 500))
                initial_fidelity = max(0.80, 0.97 - 0.0002 * d)
                
                link_id = f"L{link_count}"
                network.add_link(QuantumLink(
                    link_id, node_names[i], node_names[j],
                    success_prob=p_success,
                    attempt_rate=1.0,
                    initial_fidelity=initial_fidelity,
                    distance_km=d
                ))
                link_count += 1
    
    # Ensure connectivity: add edges for isolated nodes
    connected = _get_connected_component(network, node_names[0])
    while len(connected) < num_nodes:
        # Find an unconnected node
        unconnected = [n for n in node_names if n not in connected]
        # Connect it to nearest connected node
        best_dist = float('inf')
        best_pair = None
        for uc in unconnected[:1]:  # just fix one at a time
            uc_idx = int(uc[1:])
            for c in connected:
                c_idx = int(c[1:])
                key = (min(uc_idx, c_idx), max(uc_idx, c_idx))
                d = distances.get(key, float('inf'))
                if d < best_dist:
                    best_dist = d
                    best_pair = (uc, c, d)
        
        if best_pair:
            uc, c, d = best_pair
            p_success = max(0.1, 0.9 * np.exp(-d / 500))
            initial_fidelity = max(0.80, 0.97 - 0.0002 * d)
            link_id = f"L{link_count}"
            network.add_link(QuantumLink(
                link_id, uc, c,
                success_prob=p_success,
                attempt_rate=1.0,
                initial_fidelity=initial_fidelity,
                distance_km=d
            ))
            link_count += 1
        
        connected = _get_connected_component(network, node_names[0])
    
    return network


def create_cost239(num_memory_slots=4, coherence_time=100.0):
    """
    COST-239 European optical network (11 nodes, 26 links).
    Standard benchmark in optical networking research.
    """
    network = QuantumNetwork()
    
    node_names = [
        "London", "Paris", "Brussels", "Amsterdam", "Luxembourg",
        "Zurich", "Milan", "Prague", "Berlin", "Vienna", "Copenhagen"
    ]
    
    for name in node_names:
        network.add_node(QuantumNode(name, num_memory_slots, coherence_time))
    
    # COST-239 edges with approximate distances
    edges = [
        ("London", "Paris", 340), ("London", "Brussels", 370),
        ("London", "Amsterdam", 360), ("Paris", "Brussels", 310),
        ("Paris", "Luxembourg", 380), ("Paris", "Zurich", 490),
        ("Brussels", "Amsterdam", 210), ("Brussels", "Luxembourg", 220),
        ("Amsterdam", "Berlin", 650), ("Amsterdam", "Copenhagen", 620),
        ("Luxembourg", "Zurich", 380), ("Luxembourg", "Prague", 700),
        ("Zurich", "Milan", 280), ("Zurich", "Vienna", 600),
        ("Milan", "Vienna", 600), ("Milan", "Zurich", 280),
        ("Prague", "Berlin", 350), ("Prague", "Vienna", 330),
        ("Berlin", "Copenhagen", 360), ("Berlin", "Vienna", 640),
        ("Vienna", "Milan", 600), ("Copenhagen", "Berlin", 360),
        ("London", "Copenhagen", 960), ("Paris", "Milan", 640),
        ("Brussels", "Berlin", 780), ("Amsterdam", "Luxembourg", 320)
    ]
    
    link_count = 0
    added_edges = set()
    for node_a, node_b, dist_km in edges:
        edge_key = tuple(sorted([node_a, node_b]))
        if edge_key in added_edges:
            continue
        added_edges.add(edge_key)
        
        p_success = max(0.15, np.exp(-dist_km / 2500))
        initial_fidelity = max(0.85, 0.96 - 0.0001 * dist_km)
        
        link_id = f"L{link_count}"
        network.add_link(QuantumLink(
            link_id, node_a, node_b,
            success_prob=p_success,
            attempt_rate=1.0,
            initial_fidelity=initial_fidelity,
            distance_km=dist_km
        ))
        link_count += 1
    
    return network


def _get_connected_component(network, start_node):
    """BFS to find connected component."""
    visited = {start_node}
    queue = [start_node]
    while queue:
        node = queue.pop(0)
        for link in network.get_node_links(node):
            neighbor = link.node_b if link.node_a == node else link.node_a
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def get_topology_stats(network):
    """Print topology statistics."""
    num_nodes = len(network.nodes)
    num_links = len(network.links)
    
    # Degree distribution
    degrees = []
    for node_id in network.nodes:
        degree = len(network.get_node_links(node_id))
        degrees.append(degree)
    
    # Link quality stats
    success_probs = [l.success_prob for l in network.links.values()]
    fidelities = [l.initial_fidelity for l in network.links.values()]
    distances = [l.distance_km for l in network.links.values()]
    
    stats = {
        'num_nodes': num_nodes,
        'num_links': num_links,
        'avg_degree': np.mean(degrees),
        'min_degree': min(degrees),
        'max_degree': max(degrees),
        'avg_success_prob': np.mean(success_probs),
        'avg_fidelity': np.mean(fidelities),
        'avg_distance_km': np.mean(distances),
        'diameter': _compute_diameter(network)
    }
    return stats


def _compute_diameter(network):
    """Compute network diameter (max shortest path length)."""
    nodes = list(network.nodes.keys())
    max_dist = 0
    
    for source in nodes:
        # BFS from source
        dist = {source: 0}
        queue = [source]
        while queue:
            u = queue.pop(0)
            for link in network.get_node_links(u):
                v = link.node_b if link.node_a == u else link.node_a
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        
        if dist:
            max_dist = max(max_dist, max(dist.values()))
    
    return max_dist


if __name__ == "__main__":
    print("=" * 60)
    print("TOPOLOGY STATISTICS")
    print("=" * 60)
    
    topologies = {
        "NSFNET (14 nodes)": create_nsfnet(),
        "COST-239 (11 nodes)": create_cost239(),
        "SURFnet (50 nodes)": create_surfnet(),
        "Waxman-100": create_waxman_graph(100, alpha=0.15, beta=0.25),
    }
    
    for name, net in topologies.items():
        stats = get_topology_stats(net)
        print(f"\n{name}:")
        print(f"  Nodes: {stats['num_nodes']}, Links: {stats['num_links']}")
        print(f"  Degree: avg={stats['avg_degree']:.1f}, "
              f"min={stats['min_degree']}, max={stats['max_degree']}")
        print(f"  Avg success prob: {stats['avg_success_prob']:.3f}")
        print(f"  Avg fidelity: {stats['avg_fidelity']:.3f}")
        print(f"  Avg distance: {stats['avg_distance_km']:.0f} km")
        print(f"  Diameter: {stats['diameter']} hops")
