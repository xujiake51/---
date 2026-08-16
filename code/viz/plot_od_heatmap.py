# -*- coding: utf-8 -*-
"""OD 需求矩阵热力图（图 2.1）：坐标轴中文化，顺序蓝，0 值白。"""
import os
import csv

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from viz_common import OUTPUT_DIR, FIGURES_DIR, SEQ_BLUE, INK, MUTED, GRID, setup_font

SRC = os.path.join(OUTPUT_DIR, "question1_OD矩阵.csv")
OUTPUT = os.path.join(FIGURES_DIR, "question1_OD矩阵热力图.png")


def main():
    setup_font()

    with open(SRC, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    header = rows[0]
    dests = header[1:-1]
    body = [r for r in rows[1:] if r[0] != "合计"]
    origins = [r[0] for r in body]
    matrix = np.array([[int(v) for v in r[1:-1]] for r in body])

    cmap = mcolors.LinearSegmentedColormap.from_list("blue_seq", SEQ_BLUE)
    cmap.set_under("#ffffff")
    norm = mcolors.Normalize(vmin=1, vmax=matrix.max())

    fig, ax = plt.subplots(figsize=(12, 2.4))
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(dests)))
    ax.set_xticklabels(dests, rotation=90, fontsize=9, color=MUTED)
    ax.set_yticks(range(len(origins)))
    ax.set_yticklabels(origins, fontsize=11, color=INK)
    ax.tick_params(length=0)

    ax.set_xticks(np.arange(-0.5, len(dests), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(origins), 1), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    ax.set_xlabel("目的地（海上设施）", fontsize=12, color=INK)
    ax.set_ylabel("出发地（机场 / 未指定）", fontsize=12, color=INK)
    ax.set_title("出发地 × 目的地 交通运输请求量 OD 矩阵", fontsize=14, color=INK, pad=10)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        orientation="horizontal", pad=0.38, aspect=50, shrink=0.8,
                        extend="min")
    cbar.set_label("请求数（人次）", fontsize=11, color=INK)
    cbar.ax.tick_params(labelsize=10, colors=MUTED)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    print("已导出：", OUTPUT)


if __name__ == "__main__":
    main()
