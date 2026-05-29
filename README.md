# AoE-ARS: Age-of-Entanglement Aware Adaptive Routing and Scheduling

A discrete-event simulator and reference implementation of **AoE-ARS**, a
routing and scheduling protocol for entanglement distribution in quantum
networks. The protocol introduces the *Age-of-Entanglement* (AoE) metric,
bridging classical Age-of-Information theory with quantum decoherence physics.

The simulator also ships with several baselines (Shortest-Path,
Fidelity-Aware, Greedy, Q-CAST, DQN routing, LP upper bound) and standard
topologies (NSFNET, SURFnet, COST239, Waxman) for reproducible comparisons.

## Repository layout

```
quantum-net-research/
├── src/
│   ├── network_model.py             # Core data structures + physics models
│   ├── protocols.py                 # AoE-ARS + 3 centralized baselines
│   ├── distributed_protocols.py     # Distributed AoE-ARS, D-Greedy, Q-CAST, RL
│   ├── strong_baselines.py          # Q-CAST, DQN routing, LP upper bound
│   ├── simulation_engine.py         # Discrete-event simulator
│   ├── topologies.py                # NSFNET / SURFnet / COST239 / Waxman
│   ├── run_experiments.py           # 5 main scenarios
│   ├── run_experiments_v2.py        # Heterogeneous + hotspot
│   ├── run_extended_experiments.py  # 30-seed runs, dynamic failures
│   ├── run_distributed_experiments.py  # Comm-delay sweep, real topologies
│   ├── run_verification_*.py        # Verification / ablation experiments
│   ├── exp_convergence.py           # Convergence study
│   ├── exp_hardware_platforms.py    # Hardware-platform comparison
│   ├── generate_figures*.py         # Plot scripts (PDF + PNG)
│   ├── generate_distributed_figures.py
│   ├── generate_extended_figures.py
│   └── gen_architecture_fig.py
├── requirements.txt
└── README.md
```

Running any experiment script writes its results JSON to `experiments/`
and figures to `figures/`. Both directories are created on demand and are
git-ignored.

## Installation

```bash
git clone <your-fork-url> quantum-net-research
cd quantum-net-research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.10+. The only runtime dependencies are `numpy` and
`matplotlib`.

## Quick start

A minimal smoke test that builds NSFNET and runs AoE-ARS for a few hundred
slots:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from topologies import create_nsfnet
from protocols import AoEARS
from simulation_engine import SimulationConfig, QuantumNetworkSimulator

net = create_nsfnet(num_memory_slots=4, coherence_time=100.0)
cfg = SimulationConfig(num_time_slots=500, request_arrival_rate=0.3, seed=42)
sim = QuantumNetworkSimulator(net, AoEARS(net, seed=42), cfg)
metrics = sim.run()
print("delivered:", len(metrics.delivery_fidelities))
print("avg fidelity:", sum(metrics.delivery_fidelities) / max(1, len(metrics.delivery_fidelities)))
```

## Reproducing the experiments

Each script under `src/` is self-contained. Run from the repo root:

```bash
python src/run_experiments.py              # main 5 scenarios
python src/run_experiments_v2.py           # heterogeneous + hotspot
python src/run_extended_experiments.py     # 30-seed extended scenarios
python src/run_distributed_experiments.py  # comm-delay & real topologies
python src/run_all_verification_experiments.py
python src/exp_convergence.py
python src/exp_hardware_platforms.py
```

Then regenerate figures:

```bash
python src/generate_figures.py
python src/generate_figures_v2.py
python src/generate_distributed_figures.py
python src/generate_extended_figures.py
python src/gen_architecture_fig.py
```

Outputs land in `experiments/*.json` and `figures/*.{pdf,png}`.

## Protocol overview

`AoEARS` combines four mechanisms:

1. **AoE-weighted Dijkstra routing** — each link weight encodes both
   classical hop cost and the expected age penalty.
2. **Lyapunov-drift scheduling** — admits requests by minimizing a
   drift-plus-penalty bound, giving provable throughput-AoE trade-offs.
3. **Proactive memory refresh** — refreshes near-decohered memory slots
   before they fall below the fidelity threshold.
4. **Fidelity-predictive swap decisions** — defers entanglement swaps
   until the predicted post-swap fidelity passes a threshold.

Baselines (`ShortestPathRouting`, `FidelityAwareRouting`,
`GreedyScheduling`) live next to it in `protocols.py`. Stronger baselines
(`QCASTProtocol`, `DQNRoutingProtocol`, `LPUpperBound`) live in
`strong_baselines.py`.

## License

No license file is shipped; add one (e.g. MIT, Apache-2.0) before making
the repository public.
