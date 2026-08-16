# -*- coding: utf-8 -*-
"""问题二净差直方图：各设施「出海 − 海返」人数差，发散色（负蓝 / 正红 / 0 白）。

数据源：peopleQ2.csv（出海 = 出发地为机场/未指定、目的地为设施；
海返 = 出发地为设施、目的地为机场/未指定）。
"""
import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from viz_common import ATTACH_DIR, FIGURES_DIR, FACILITIES, DIVERGING, INK, MUTED, GRID, setup_font

OUTPUT = os.path.join(FIGURES_DIR, "问题二_净差直方图.png")
LAND = {"LAND", "A01", "A02", "A03"}


def main():
    setup_font()
    df = pd.read_csv(os.path.join(ATTACH_DIR, "peopleQ2.csv"))

    out = df[df["destination_id"].str.startswith("F") & df["origin_id"].isin(LAND)]
    ret = df[df["origin_id"].str.startswith("F") & df["destination_id"].isin(LAND)]
    out_cnt = out.groupby("destination_id").size().reindex(FACILITIES).fillna(0)
    ret_cnt = ret.groupby("origin_id").size().reindex(FACILITIES).fillna(0)
    net = (out_cnt - ret_cnt).astype(int)

    # 校验三类需求总量
    shuttle = df[df["origin_id"].str.startswith("F") & df["destination_id"].str.startswith("F")]
    assert int(out_cnt.sum()) + int(ret_cnt.sum()) + len(shuttle) == len(df)

    # 净差取值分布
    values = np.arange(int(net.min()), int(net.max()) + 1)
    counts = np.array([int((net == v).sum()) for v in values])

    cmap = mcolors.LinearSegmentedColormap.from_list("div", DIVERGING)
    norm = mcolors.TwoSlopeNorm(vmin=values.min(), vcenter=0, vmax=values.max())

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(values, counts, color=[cmap(norm(v)) for v in values],
                  edgecolor="white", width=0.8, zorder=3)
    for v, c in zip(values, counts):
        if c:
            ax.text(v, c + 0.3, str(int(c)), ha="center", va="bottom", fontsize=9, color=INK)

    ax.axvline(0, color=GRID, linewidth=1, zorder=2)
    ax.axhline(0, color=GRID, linewidth=1)

    # 标注 ±3 范围（48/52 设施落在内）
    within3 = int((net.abs() <= 3).sum())
    ax.set_title(f"各设施出海−海返净差分布（{within3}/52 设施落在 ±3 内）",
                 fontsize=13, color=INK)
    ax.set_xlabel("净差（出海 − 海返，人）", color=INK)
    ax.set_ylabel("设施数", color=INK)
    ax.tick_params(labelcolor=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # 发散色标
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cbar.set_label("净差符号", color=MUTED)
    cbar.set_ticks([])

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    print("已导出：", OUTPUT)


if __name__ == "__main__":
    main()
