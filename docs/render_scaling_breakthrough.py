"""Render the 'Scaling Bottleneck: CPU vs GPU' breakthrough slide as SVG.

Minimal design, HK Grotesk typography. Pulls per-rank H2O wall times from the
thesis-backing scaling data on the CPU side and from the HPC GPU re-run on the
right side.

No external dependencies. Output: docs/scaling_breakthrough.svg
"""

import os
import json

# CPU side: thesis Table 4 (results/scaling/scaling_p8.log + cited values).
cpu_h2o = {1: 210.42, 2: 170.66, 4: 305.87, 8: 1038.50}

# GPU side: HPC re-run on RTX 6000 Ada.
GPU_FILES = {
    1: "results/rtx-6000-ada-generation/simulator/simulator_20260416_234018.json",
    2: "results/rtx-6000-ada-generation/simulator/simulator_20260416_234809.json",
    4: "results/rtx-6000-ada-generation/simulator/simulator_20260416_235706.json",
    8: "results/rtx-6000-ada-generation/simulator/simulator_20260417_000732.json",
}
gpu_h2o = {}
for P, path in GPU_FILES.items():
    with open(path) as f:
        d = json.load(f)
    gpu_h2o[P] = d["molecules"]["H2O"]["wall_time"]

ranks = [1, 2, 4, 8]
cpu_eff = {P: (cpu_h2o[1] / cpu_h2o[P]) / P * 100 for P in ranks}
gpu_eff = {P: (gpu_h2o[1] / gpu_h2o[P]) / P * 100 for P in ranks}

# ---------- SVG layout ----------
W, H = 1100, 480
PANEL_W = 460
GAP = 60
LEFT_X = (W - 2 * PANEL_W - GAP) / 2
RIGHT_X = LEFT_X + PANEL_W + GAP
TOP, BOTTOM = 110, 80
n = len(ranks)

FONT = ("'HK Grotesk', 'Hanken Grotesk', 'Inter', "
        "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif")
INK = "#111111"
MUTE = "#888888"
RED = "#c0392b"
GREEN = "#27ae60"


def panel(x0, title, values_pct, color):
    out = []
    plot_w = PANEL_W - 60
    plot_left = x0 + 50
    plot_top = TOP
    plot_bot = H - BOTTOM - 30
    bar_slot = plot_w / n
    bar_w = bar_slot * 0.42

    # Panel title - single short line
    out.append(f'<text x="{x0 + PANEL_W/2}" y="{TOP - 30}" text-anchor="middle" '
               f'font-size="15" font-weight="600" fill="{INK}" '
               f'style="font-family:{FONT}">{title}</text>')

    # Single 100% reference line at top
    out.append(f'<line x1="{plot_left}" y1="{plot_top}" '
               f'x2="{plot_left + plot_w}" y2="{plot_top}" '
               f'stroke="{MUTE}" stroke-width="0.75" stroke-dasharray="3,4"/>')
    out.append(f'<text x="{plot_left + plot_w + 6}" y="{plot_top + 4}" '
               f'font-size="10" fill="{MUTE}" style="font-family:{FONT}">100%</text>')

    # Bars
    for i, P in enumerate(ranks):
        v = values_pct[P]
        bx = plot_left + i * bar_slot + (bar_slot - bar_w) / 2
        bh = (v / 100.0) * (plot_bot - plot_top)
        by = plot_bot - bh
        out.append(f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bh}" '
                   f'fill="{color}" rx="2"/>')
        # Value label above each bar
        out.append(f'<text x="{bx + bar_w/2}" y="{by - 8}" text-anchor="middle" '
                   f'font-size="13" font-weight="600" fill="{INK}" '
                   f'style="font-family:{FONT}">{v:.1f}%</text>')
        # X axis label P=N
        out.append(f'<text x="{bx + bar_w/2}" y="{plot_bot + 22}" text-anchor="middle" '
                   f'font-size="12" fill="{INK}" '
                   f'style="font-family:{FONT}">P={P}</text>')

    return out


lines = []
lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
             f'font-family="{FONT}">')
lines.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# Top headline only - no subtitle
lines.append(f'<text x="{W/2}" y="50" text-anchor="middle" font-size="22" '
             f'font-weight="700" fill="{INK}" style="font-family:{FONT}">'
             f'Parallel efficiency: CPU vs. GPU</text>')

# Two panels
lines += panel(LEFT_X, "CPU only", cpu_eff, RED)
lines += panel(RIGHT_X, "GPU accelerated", gpu_eff, GREEN)

# Footnote: data source
lines.append(f'<text x="{W/2}" y="{H - 20}" text-anchor="middle" font-size="10" '
             f'fill="{MUTE}" style="font-family:{FONT}">'
             f'H₂O ground-state VQE  ·  parallel efficiency = (T₁ / Tₚ) / P</text>')

lines.append('</svg>')

out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "scaling_breakthrough.svg")
with open(out_path, "w") as f:
    f.write("\n".join(lines))

print(f"[plot] Wrote {out_path}")
print("CPU efficiency:")
for P in ranks: print(f"  P={P}: {cpu_eff[P]:5.1f}%")
print("GPU efficiency:")
for P in ranks: print(f"  P={P}: {gpu_eff[P]:5.1f}%")