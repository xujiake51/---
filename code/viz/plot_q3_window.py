# -*- coding: utf-8 -*-
"""问题三窗口时长箱线图：四类任务的「最晚到达 − 最早登机」窗口时长分布。

数据源：peopleQ3.csv。突出「紧核（应急/增储）单天 vs 松壳（倒班/临时）跨天」的量级差。
"""
import os

import pandas as pd
from matplotlib import pyplot as plt

from viz_common import ATTACH_DIR, CODE_DIR, INK, MUTED, GRID, setup_font

OUTPUT = os.path.join(CODE_DIR, "问题三_窗口时长箱线图.png")

# 任务类型显示名（按题面优先级顺序）
TASK_NAMES = [("emergency", "应急处置"), ("production", "增储上产"),
              ("shift", "常规倒班"), ("temporary", "临时任务")]
# 紧核暖色、松壳冷色
TASK_COLORS = ["#d64550", "#ef8a62", "#4e79a7", "#76b7b2"]


def main():
    setup_font()
    df = pd.read_csv(os.path.join(ATTACH_DIR, "peopleQ3.csv"))
    df["pickup"] = pd.to_datetime(df["earliest_pickup_time"])
    df["arrival"] = pd.to_datetime(df["latest_arrival_time"])
    df["window_h"] = (df["arrival"] - df["pickup"]).dt.total_seconds() / 3600.0

    data = []
    labels = []
    colors = []
    for key, name in TASK_NAMES:
        vals = df.loc[df["task_type"] == key, "window_h"].values
        data.append(vals)
        labels.append(name)
        colors.append(TASK_COLORS[[k for k, _ in TASK_NAMES].index(key)])

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                    widths=0.55, medianprops=dict(color=INK, linewidth=1.5))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    for wh in bp["whiskers"]:
        wh.set_color(MUTED)
    for cap in bp["caps"]:
        cap.set_color(MUTED)

    ax.set_yscale("log")
    ax.set_ylabel("窗口时长（小时，对数轴）", color=INK)
    ax.set_xlabel("任务类型", color=INK)
    ax.set_title("四类任务的运输时间窗时长分布（紧核单天 / 松壳跨天）",
                 fontsize=13, color=INK)
    ax.tick_params(labelcolor=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.6, zorder=0)

    # 紧核 / 松壳分隔与标注
    ax.axvline(2.5, color=GRID, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(1.6, ax.get_ylim()[1] * 0.55, "紧核（单天）", ha="center", fontsize=10,
            color="#c0392b", weight="bold")
    ax.text(3.6, ax.get_ylim()[1] * 0.55, "松壳（跨天）", ha="center", fontsize=10,
            color="#2e7d32", weight="bold")

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    print("已导出：", OUTPUT)


if __name__ == "__main__":
    main()
