"""
Generate D-AoE-ARS System Architecture Diagram for ToN paper (v2 - fixed layout).
Two-panel figure: (a) per-node protocol stack, (b) distributed network operation.
"""

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
REPO_ROOT = _os.path.dirname(_HERE)
EXPERIMENTS_DIR = _os.path.join(REPO_ROOT, "experiments")
FIGURES_DIR = _os.path.join(REPO_ROOT, "figures")
_os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
_os.makedirs(FIGURES_DIR, exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig = plt.figure(figsize=(7.2, 5.0))

# Two subfigures side by side
ax_left = fig.add_axes([0.01, 0.02, 0.52, 0.92])   # protocol stack
ax_right = fig.add_axes([0.54, 0.02, 0.45, 0.92])  # network view

for ax in [ax_left, ax_right]:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

# Color scheme
C_QUANTUM = '#2E86AB'
C_AOE = '#A23B72'
C_ROUTING = '#F18F01'
C_REFRESH = '#C73E1D'
C_ADMISSION = '#3B7A57'
C_SCHED = '#5C4D7D'
C_CLASSICAL = '#6C757D'
C_BG = '#F5F6F7'

def draw_box(ax, x, y, w, h, color, lines, fontsize=7.5, alpha=0.85, tc='white'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor='black', linewidth=0.9, alpha=alpha)
    ax.add_patch(box)
    text = '\n'.join(lines)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=tc, linespacing=1.3)

# ============================================================
# LEFT PANEL: Protocol Stack
# ============================================================
ax_left.text(5, 9.7, '(a) Node $v$ — Protocol Stack', ha='center', va='bottom',
             fontsize=9.5, fontweight='bold')

# Outer node boundary
node_box = FancyBboxPatch((0.2, 0.3), 9.6, 9.2, boxstyle="round,pad=0.1",
                           facecolor=C_BG, edgecolor='black', linewidth=1.5, alpha=0.25)
ax_left.add_patch(node_box)

# Layer 1 (bottom): Quantum Hardware
draw_box(ax_left, 0.5, 0.6, 9.0, 1.3, C_QUANTUM,
         ['Quantum Hardware Layer', 'Memory Slots  |  Entanglement Gen.  |  BSM'],
         fontsize=7)

# Layer 2: AoE Monitor
draw_box(ax_left, 0.5, 2.2, 9.0, 1.3, C_AOE,
         ['AoE Monitor', 'AoE$(s)=(t-t_g^s)/T_{coh}$    '
          '$F(s)=\\frac{1}{2}+(F^0-\\frac{1}{2})e^{-\\mathrm{AoE}}$'],
         fontsize=6.8)

# Layer 3: Four core components (2x2)
cw = 4.2  # component width
ch = 1.8  # component height
gap = 0.6
x1 = 0.5
x2 = x1 + cw + gap
y1 = 3.8   # bottom row
y2 = 5.9   # top row

# Bottom-left: Proactive Refresh
draw_box(ax_left, x1, y1, cw, ch, C_REFRESH,
         ['Proactive Refresh', '───────────', 'Discard if AoE$(s) > \\tau$', 'Free slot for reuse'],
         fontsize=6.5)

# Bottom-right: Admission Control
draw_box(ax_left, x2, y1, cw, ch, C_ADMISSION,
         ['Fidelity Admission Control', '───────────', '$F^{e2e}_{pred} \\geq F^{th}$?', 'Hold if insufficient'],
         fontsize=6.5)

# Top-left: AoE-Weighted Routing
draw_box(ax_left, x1, y2, cw, ch, C_ROUTING,
         ['AoE-Weighted Routing', '───────────', '$\\min\\{1/p + \\alpha\\overline{\\mathrm{AoE}} + \\beta h\\}$', 'Local + 1-hop info only'],
         fontsize=6.5)

# Top-right: Distributed Scheduling
draw_box(ax_left, x2, y2, cw, ch, C_SCHED,
         ['Distributed Scheduling', '───────────', 'priority $= F^{th}/(\\mathrm{AoE}+\\epsilon)$', 'Lyapunov drift-based'],
         fontsize=6.5)

# Layer 4 (top): Classical Control Plane
draw_box(ax_left, 0.5, 8.0, 9.0, 1.2, C_CLASSICAL,
         ['Classical Control Plane', '1-hop State Exchange — $O(\\mathrm{degree})$ msgs/slot'],
         fontsize=7)

# Vertical arrows between layers
arrow_kw = dict(arrowstyle='->', color='black', lw=1.2)
# Quantum → AoE
ax_left.annotate('', xy=(5, 2.2), xytext=(5, 1.9), arrowprops=arrow_kw)
# AoE → Components (two arrows)
ax_left.annotate('', xy=(2.6, 3.8), xytext=(2.6, 3.5), arrowprops=arrow_kw)
ax_left.annotate('', xy=(7.4, 3.8), xytext=(7.4, 3.5), arrowprops=arrow_kw)
# Components bottom → top
ax_left.annotate('', xy=(2.6, 5.9), xytext=(2.6, 5.6), arrowprops=dict(arrowstyle='->', color='gray', lw=0.9, linestyle='dashed'))
ax_left.annotate('', xy=(7.4, 5.9), xytext=(7.4, 5.6), arrowprops=dict(arrowstyle='->', color='gray', lw=0.9, linestyle='dashed'))
# Components → Classical
ax_left.annotate('', xy=(2.6, 8.0), xytext=(2.6, 7.7), arrowprops=arrow_kw)
ax_left.annotate('', xy=(7.4, 8.0), xytext=(7.4, 7.7), arrowprops=arrow_kw)
# Horizontal: Routing ↔ Admission
ax_left.annotate('', xy=(x2, 4.7), xytext=(x1+cw, 4.7),
                 arrowprops=dict(arrowstyle='<->', color='gray', lw=0.9, linestyle='dashed'))
ax_left.annotate('', xy=(x2, 6.8), xytext=(x1+cw, 6.8),
                 arrowprops=dict(arrowstyle='<->', color='gray', lw=0.9, linestyle='dashed'))

# ============================================================
# RIGHT PANEL: Network View + Timeline
# ============================================================
ax_right.text(5, 9.7, '(b) Distributed Network Operation', ha='center', va='bottom',
              fontsize=9.5, fontweight='bold')

# Three nodes in triangle
nodes = {'A': (2.0, 7.5), 'B': (5.0, 4.5), 'C': (8.0, 7.5)}

# Quantum links first (behind nodes)
links = [('A', 'B'), ('B', 'C'), ('A', 'C')]
for n1, n2 in links:
    x1p, y1p = nodes[n1]
    x2p, y2p = nodes[n2]
    ax_right.plot([x1p, x2p], [y1p, y2p], color=C_QUANTUM, linewidth=3.0, alpha=0.6, zorder=1)
    mx, my = (x1p+x2p)/2, (y1p+y2p)/2
    # offset label slightly
    if n1 == 'A' and n2 == 'C':
        my += 0.35
    elif n1 == 'A':
        mx -= 0.4
    else:
        mx += 0.4
    ax_right.text(mx, my, 'EP', fontsize=7.5, color='#1A5276', ha='center',
                  fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.9))

# Draw nodes
for name, (nx, ny) in nodes.items():
    circle = plt.Circle((nx, ny), 0.7, facecolor='white', edgecolor='black',
                        linewidth=1.8, zorder=2)
    ax_right.add_patch(circle)
    ax_right.text(nx, ny + 0.15, name, ha='center', va='center', fontsize=12,
                  fontweight='bold', zorder=3)
    ax_right.text(nx, ny - 0.3, 'D-AoE-ARS', ha='center', va='center',
                  fontsize=6.5, color='#444444', zorder=3)

# Classical messages (dotted arrows, offset from quantum links)
ax_right.annotate('', xy=(4.5, 5.0), xytext=(2.5, 7.0),
                  arrowprops=dict(arrowstyle='->', color=C_CLASSICAL, lw=1.3, linestyle='dotted'))
ax_right.text(2.8, 5.7, '1-hop\nstate', fontsize=5.5, color=C_CLASSICAL, ha='center')

ax_right.annotate('', xy=(7.5, 7.0), xytext=(5.5, 5.0),
                  arrowprops=dict(arrowstyle='->', color=C_CLASSICAL, lw=1.3, linestyle='dotted'))
ax_right.text(7.2, 5.7, '1-hop\nstate', fontsize=5.5, color=C_CLASSICAL, ha='center')

# Legend
ax_right.plot([1.0, 2.2], [3.2, 3.2], color=C_QUANTUM, linewidth=3.0, alpha=0.6)
ax_right.text(2.5, 3.2, 'Quantum link (entangled pairs)', fontsize=6, va='center')
ax_right.plot([1.0, 2.2], [2.7, 2.7], color=C_CLASSICAL, linewidth=1.3, linestyle='dotted')
ax_right.text(2.5, 2.7, 'Classical control ($d_{cc}$ delay)', fontsize=6, va='center')

# Per-slot timeline
ax_right.text(5, 2.0, 'Per-Slot Timeline', ha='center', fontsize=7.5, fontweight='bold')

phases = [
    (1.0, 2.4, C_REFRESH, 'Phase 1:\nRefresh'),
    (3.5, 2.4, C_QUANTUM, 'Phase 2:\nGenerate'),
    (6.0, 2.4, C_ROUTING, 'Phase 3:\nRoute & Swap'),
]

ty = 1.2
for px, pw, color, label in phases:
    box = FancyBboxPatch((px, ty - 0.4), pw, 0.9, boxstyle="round,pad=0.04",
                         facecolor=color, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax_right.add_patch(box)
    ax_right.text(px + pw/2, ty + 0.05, label, ha='center', va='center',
                  fontsize=6, fontweight='bold', color='white')

# Arrows between phases
ax_right.annotate('', xy=(3.4, ty+0.05), xytext=(3.5, ty+0.05),
                  arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
ax_right.annotate('', xy=(5.9, ty+0.05), xytext=(6.0, ty+0.05),
                  arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# Time axis
ax_right.annotate('', xy=(9.2, ty - 0.65), xytext=(0.8, ty - 0.65),
                  arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
ax_right.text(5, ty - 0.95, 'time slot $t$', ha='center', fontsize=6.5)

plt.savefig(_os.path.join(FIGURES_DIR, "fig_architecture.pdf"),
            dpi=300, bbox_inches='tight', format='pdf')
plt.savefig(_os.path.join(FIGURES_DIR, "fig_architecture.png"),
            dpi=150, bbox_inches='tight', format='png')
print("Architecture figure v2 saved.")
