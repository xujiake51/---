# -*- coding: utf-8 -*-
"""问题一航线网络图：MDS 投影 + 机场三色分块 + 线型区分满座/尾数。

数据源：distances.csv（MDS 投影）+ q1_flight_summary.csv（source 字段 full/tail，
route 字段 "A01 -> F003 -> A01"）。
"""
import os

from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from viz_common import (OUTPUT_DIR, AIRPORT_COLORS, FACILITIES, BLOCKS,
                        INK, MUTED, load_distance_df, classical_mds, setup_font)

OUTPUT = os.path.join(OUTPUT_DIR, "问题一_航线网络图.png")

# 满座直飞 / 尾数环游 的线型
FULL_LW, FULL_ALPHA = 0.7, 0.45
TAIL_LW, TAIL_ALPHA = 1.9, 0.85


def main():
    setup_font()
    import pandas as pd

    coords = classical_mds(load_distance_df())

    summary = pd.read_csv(os.path.join(OUTPUT_DIR, "q1_flight_summary.csv"))
    # route 字符串 -> 节点序列
    summary["nodes"] = summary["route"].map(lambda s: [x.strip() for x in s.split("->")])

    fig, ax = plt.subplots(figsize=(11, 8.5))

    # 海上设施：按机场分块着色（浅色）
    for airport, fs in BLOCKS.items():
        pts = coords.loc[[f for f in fs if f in coords.index]]
        ax.scatter(pts.x, pts.y, s=20, alpha=0.5, color=AIRPORT_COLORS[airport],
                   zorder=3, label=f"{airport} 服务区")
    # 机场：星形
    for airport, c in AIRPORT_COLORS.items():
        ax.scatter(coords.loc[airport, "x"], coords.loc[airport, "y"], s=220,
                   marker="*", color=c, edgecolor="black", linewidth=0.8, zorder=5)
        ax.text(coords.loc[airport, "x"], coords.loc[airport, "y"] + 6, f"{airport}",
                fontsize=11, weight="bold", ha="center", va="bottom", color=c)

    # 设施编号标注
    for f in FACILITIES:
        ax.text(coords.loc[f, "x"], coords.loc[f, "y"], f[1:], fontsize=5,
                color=MUTED, ha="center", va="bottom", zorder=4)

    # 航线：满座=细线、尾数=粗线，颜色=所属机场
    for _, row in summary.iterrows():
        nodes = row["nodes"]
        pts = coords.loc[nodes]
        full = row["source"] == "full"
        lw, alpha = (FULL_LW, FULL_ALPHA) if full else (TAIL_LW, TAIL_ALPHA)
        ax.plot(pts.x, pts.y, color=AIRPORT_COLORS[row["airport"]],
                linewidth=lw, alpha=alpha, zorder=1)

    ax.set_title("问题一航线网络（88 架次，满座细线 / 尾数环游粗线）", fontsize=14, color=INK)
    ax.set_xlabel("MDS 第一维", color=MUTED)
    ax.set_ylabel("MDS 第二维", color=MUTED)
    ax.tick_params(labelcolor=MUTED)

    # 图例
    handles = [Patch(facecolor=AIRPORT_COLORS[a], alpha=0.5, label=f"{a} 服务区") for a in AIRPORT_COLORS]
    handles += [Line2D([0], [0], color="black", lw=FULL_LW, label="满座直飞（细线）"),
                Line2D([0], [0], color="black", lw=TAIL_LW, label="尾数环游（粗线）")]
    ax.legend(handles=handles, loc="upper left", framealpha=0.9, fontsize=9)

    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    print("已导出：", OUTPUT)


if __name__ == "__main__":
    main()
