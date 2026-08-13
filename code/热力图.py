# -*- coding: utf-8 -*-
"""读取 output/请求统计_OD矩阵.csv，绘制 OD 矩阵热力图并输出 PNG。"""
import os

import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "output", "请求统计_OD矩阵.csv")
OUTPUT = os.path.join(os.path.dirname(HERE), "output", "OD矩阵热力图.png")

# ---- 读取 OD 矩阵 CSV（去掉“合计”行/列）----
with open(SRC, encoding="utf-8-sig", newline="") as f:
    all_rows = list(csv.reader(f))

header = all_rows[0]
dests = header[1:-1]                      # 排除首列标签和末列“合计”
body = [r for r in all_rows[1:] if r[0] != "合计"]   # 去掉“合计”行
origins = [r[0] for r in body]
matrix = np.array([[int(v) for v in r[1:-1]] for r in body])

# ---- 顺序色带（蓝色，浅→深），0 用白色表示 ----
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
cmap = mcolors.LinearSegmentedColormap.from_list("blue_seq", SEQ)
cmap.set_under("#ffffff")                 # 0 → 白色
norm = mcolors.Normalize(vmin=1, vmax=matrix.max())

# ---- 颜色墨水 ----
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#c3c2b7"

# ---- 绘制 ----
fig, ax = plt.subplots(figsize=(12, 2.4))
ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

# 坐标轴与刻度
ax.set_xticks(range(len(dests)))
ax.set_xticklabels(dests, rotation=90, fontsize=9, color=MUTED)
ax.set_yticks(range(len(origins)))
ax.set_yticklabels(origins, fontsize=11, color=INK)
ax.tick_params(length=0)

# 色块之间的网格线
ax.set_xticks(np.arange(-0.5, len(dests), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(origins), 1), minor=True)
ax.grid(which="minor", color=GRID, linewidth=0.5)
ax.tick_params(which="minor", length=0)

# 轴标签与标题
ax.set_xlabel("目的地 destination_id", fontsize=12, color=INK)
ax.set_ylabel("出发地 origin_id", fontsize=12, color=INK)
ax.set_title("出发地 × 目的地 交通运输请求量 OD 矩阵", fontsize=14, color=INK, pad=10)

# 颜色条
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                    orientation="horizontal", pad=0.38, aspect=50, shrink=0.8,
                    extend="min")
cbar.set_label("请求数（人次）", fontsize=11, color=INK)
cbar.ax.tick_params(labelsize=10, colors=MUTED)

# 去掉多余边框
for spine in ax.spines.values():
    spine.set_visible(False)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
plt.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
print("热力图已导出：", OUTPUT)
