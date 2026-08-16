# -*- coding: utf-8 -*-
"""问题三机队排班图：24 架飞机 × 7 日的架次甘特图，机场三色。

数据源：q3-routes.csv（每架次按 stop_order 记录 arrival/departure 时刻）。
"""
import os

import pandas as pd
import matplotlib.dates as mdates
from matplotlib import pyplot as plt

from viz_common import CODE_DIR, FIGURES_DIR, AIRPORT_COLORS, INK, MUTED, GRID, setup_font

OUTPUT = os.path.join(FIGURES_DIR, "问题三_机队排班图.png")

AIRPORT_ORDER = {"A01": 0, "A02": 1, "A03": 2}
TYPE_ORDER = {"T1": 0, "T2": 1, "T3": 2}


def _sort_key(aid):
    airport, typ, num = aid.split("-")[0], aid.split("-")[1], aid.split("-")[2]
    return (AIRPORT_ORDER[airport], TYPE_ORDER[typ], num)


def main():
    setup_font()
    routes = pd.read_csv(os.path.join(CODE_DIR, "q3-routes.csv"))
    routes["departure_time"] = pd.to_datetime(routes["departure_time"])
    routes["arrival_time"] = pd.to_datetime(routes["arrival_time"])

    # 每架次的起止时刻
    flights = []
    for (aid, fno), grp in routes.groupby(["aircraft_id", "flight_no"]):
        g = grp.sort_values("stop_order")
        flights.append((aid, g["departure_time"].iloc[0], g["arrival_time"].iloc[-1]))
    flights = pd.DataFrame(flights, columns=["aircraft_id", "start", "end"])

    # 只保留 24 架真实飞机（剔除“备用”）
    flights = flights[~flights["aircraft_id"].str.contains("备用")]
    aircraft = sorted(flights["aircraft_id"].unique(), key=_sort_key)

    fig, ax = plt.subplots(figsize=(13, 7.5))
    for i, aid in enumerate(aircraft):
        airport = aid.split("-")[0]
        c = AIRPORT_COLORS[airport]
        sub = flights[flights["aircraft_id"] == aid]
        for _, r in sub.iterrows():
            dur = (r["end"] - r["start"]).total_seconds() / 3600.0
            ax.barh(i, dur, left=r["start"], height=0.62, color=c, alpha=0.8,
                    edgecolor="white", linewidth=0.4)

    ax.set_yticks(range(len(aircraft)))
    ax.set_yticklabels(aircraft, fontsize=8.5)
    ax.invert_yaxis()  # A01 在最上
    ax.set_ylim(-0.6, len(aircraft) - 0.4)

    # 机场分组分隔线
    y = 0
    for airport in ("A01", "A02", "A03"):
        cnt = sum(1 for a in aircraft if a.startswith(airport))
        if cnt:
            ax.axhline(y - 0.5, color=GRID, linewidth=1)
            ax.text(-0.01, y + cnt / 2 - 0.5, airport, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=12, weight="bold",
                    color=AIRPORT_COLORS[airport])
        y += cnt

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.set_xlabel("日期（2026-08）", color=INK)
    ax.set_ylabel("飞机", color=INK)
    ax.set_title(f"问题三机队排班（{len(aircraft)} 架飞机 × 7 日，机场三色）", fontsize=14, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.6, zorder=0)
    ax.tick_params(labelcolor=MUTED)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    print("已导出：", OUTPUT, f"（{len(aircraft)} 架飞机，{len(flights)} 架次）")


if __name__ == "__main__":
    main()
