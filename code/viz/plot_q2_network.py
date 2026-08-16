# -*- coding: utf-8 -*-
"""问题二航线网络图：MDS 投影 + 机场三色分块，展示出海/海返/穿梭联合运输形态。

数据源：distances.csv（MDS）+ q2-routes.csv（每架次按 stop_order 记录访问节点）。
"""
import os

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from viz_common import (CODE_DIR, FIGURES_DIR, AIRPORT_COLORS, BLOCKS, FACILITIES,
                        INK, MUTED, load_distance_df, classical_mds, setup_font)

OUTPUT = os.path.join(FIGURES_DIR, "问题二_航线网络图.png")


def main():
    setup_font()
    coords = classical_mds(load_distance_df())
    routes = pd.read_csv(os.path.join(CODE_DIR, "q2-routes.csv"))

    fig, ax = plt.subplots(figsize=(11, 8.5))

    for airport, fs in BLOCKS.items():
        pts = coords.loc[[f for f in fs if f in coords.index]]
        ax.scatter(pts.x, pts.y, s=20, alpha=0.5, color=AIRPORT_COLORS[airport],
                   zorder=3, label=f"{airport} 服务区")
    for airport, c in AIRPORT_COLORS.items():
        ax.scatter(coords.loc[airport, "x"], coords.loc[airport, "y"], s=220,
                   marker="*", color=c, edgecolor="black", linewidth=0.8, zorder=5)
        ax.text(coords.loc[airport, "x"], coords.loc[airport, "y"] + 6, f"{airport}",
                fontsize=11, weight="bold", ha="center", va="bottom", color=c)
    for f in FACILITIES:
        ax.text(coords.loc[f, "x"], coords.loc[f, "y"], f[1:], fontsize=5,
                color=MUTED, ha="center", va="bottom", zorder=4)

    # 每架次：按 (aircraft_type, flight_no) 分组，取 stop_order 排序后的节点序列
    n_flights = 0
    for (_, flight_no), grp in routes.groupby(["aircraft_type", "flight_no"]):
        nodes = grp.sort_values("stop_order")["facility_id"].tolist()
        airport = nodes[0]  # 机场在 stop_order 0
        pts = coords.loc[nodes]
        ax.plot(pts.x, pts.y, color=AIRPORT_COLORS.get(airport, "#888888"),
                linewidth=0.6, alpha=0.5, zorder=1)
        n_flights += 1

    ax.set_title(f"问题二航线网络（{n_flights} 架次，出海/海返/穿梭联合运输）",
                 fontsize=14, color=INK)
    ax.set_xlabel("MDS 第一维", color=MUTED)
    ax.set_ylabel("MDS 第二维", color=MUTED)
    ax.tick_params(labelcolor=MUTED)

    handles = [Patch(facecolor=AIRPORT_COLORS[a], alpha=0.5, label=f"{a} 服务区") for a in AIRPORT_COLORS]
    ax.legend(handles=handles, loc="upper left", framealpha=0.9, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    print("已导出：", OUTPUT, f"（{n_flights} 架次）")


if __name__ == "__main__":
    main()
