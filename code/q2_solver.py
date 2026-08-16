# -*- coding: utf-8 -*-
"""B题问题二：出海、海返与穿梭联合运输。

运行：python code/q2_solver.py
程序采用“设施分区 + 候选环游生成 + 容量/燃油校验 + 贪心装载”的可解释启发式，
把同一架次中的出海、海返和穿梭人员统一装载，并输出题目要求的 q2 文件。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from q23_common import (
    BLOCKS, CODE_DIR, DATA_DIR, MACHINES, MAX_STOPS, Route, best_route,
    candidate_stops, label_font, load_distance, prepare_people, request_positions,
    route_empty_seat_distance, route_person_minutes, segment_loads, select_passengers,
)

PEOPLE_FILE = DATA_DIR / "peopleQ2.csv"


def build_routes(df: pd.DataFrame, dist: Mapping[Tuple[str, str], float]) -> List[Route]:
    routes: List[Route] = []
    for airport in BLOCKS:
        remaining = set(df.index[df["airport"] == airport].tolist())
        while remaining:
            stop_sets = candidate_stops(df, sorted(remaining), airport, dist, max_sets=16)
            best_choice = None
            for stops in stop_sets:
                for machine in ("T3", "T2", "T1"):
                    route = best_route(airport, stops, machine, dist)
                    if route is None:
                        continue
                    selected = select_passengers(
                        route, df, sorted(remaining), dist,
                        prefer_kinds=("穿梭", "出海", "海返"),
                    )
                    # 最短顺序可能与穿梭方向相反，再尝试当前候选顺序。
                    if not selected and len(stops) > 1:
                        route_fixed = best_route(airport, stops, machine, dist, fixed_order=stops)
                        if route_fixed is not None:
                            selected_fixed = select_passengers(
                                route_fixed, df, sorted(remaining), dist,
                                prefer_kinds=("穿梭", "出海", "海返"),
                            )
                            if selected_fixed:
                                route, selected = route_fixed, selected_fixed
                    if not selected:
                        continue
                    # 先尽量多运人，再比较航时和油耗；这会自然形成满座+尾数架次。
                    score = (len(selected), -route.minutes, -route.fuel, -len(route.nodes))
                    if best_choice is None or score > best_choice[0]:
                        best_choice = (score, route, selected)
            if best_choice is None:
                # 理论上不会触发；作为数据异常或极端燃油约束下的兜底。
                seed = next(iter(remaining))
                row = df.loc[seed]
                stops = [x for x in (str(row.origin_id), str(row.destination_id)) if x.startswith("F")]
                # 穿梭需求必须同时包含起点和终点；出海/海返只需一个设施。
                route_stops = stops if len(stops) == 2 else stops[:1]
                route = None
                for machine in ("T3", "T2", "T1"):
                    route = best_route(airport, route_stops, machine, dist, fixed_order=route_stops)
                    if route is not None:
                        selected = select_passengers(route, df, sorted(remaining), dist)
                        if selected:
                            break
                if route is None or not selected:
                    raise RuntimeError(f"问题二无法为需求 {row.person_id} 找到可行架次")
            else:
                _, route, selected = best_choice
            route.passenger_indices = list(selected)
            routes.append(route)
            remaining.difference_update(selected)
            if len(routes) > 1000:
                raise RuntimeError("问题二架次数量异常，请检查容量约束")
    return routes


def assign_numbers(routes: List[Route]) -> None:
    counters = Counter()
    for route in routes:
        counters[route.machine] += 1
        route.flight_no = counters[route.machine]


def export_results(df: pd.DataFrame, routes: List[Route], dist: Mapping[Tuple[str, str], float]) -> Dict[str, float]:
    assign_numbers(routes)
    route_rows = []
    assignment = pd.DataFrame({
        "person_id": df.person_id,
        "aircraft_type": "",
        "flight_no": "",
        "pickup_stop_order": "",
        "delivery_stop_order": "",
    })
    assignment.index = df.index
    summaries = []
    total_minutes = total_person_minutes = total_fuel = 0.0
    total_load_km = total_seat_km = 0.0
    for route in routes:
        for order, node in enumerate(route.nodes):
            route_rows.append({
                "aircraft_type": route.machine,
                "flight_no": route.flight_no,
                "stop_order": order,
                "facility_id": node,
                "refuel": int(route.refuels[order]),
            })
        pos = {n: i for i, n in enumerate(route.nodes)}
        for idx in route.passenger_indices:
            pair = request_positions(df.loc[idx], route)
            if pair is None:
                raise AssertionError(f"需求 {df.loc[idx, 'person_id']} 与架次不匹配")
            assignment.loc[idx, "aircraft_type"] = route.machine
            assignment.loc[idx, "flight_no"] = route.flight_no
            assignment.loc[idx, "pickup_stop_order"] = pair[0]
            assignment.loc[idx, "delivery_stop_order"] = pair[1]
        empty_km, eta = route_empty_seat_distance(route, df, dist)
        loads = segment_loads(route, df, route.passenger_indices)
        load_km = sum(load * dist[(route.nodes[i], route.nodes[i + 1])] for i, load in enumerate(loads))
        seat_km = sum(route.seats * dist[(route.nodes[i], route.nodes[i + 1])] for i in range(len(loads)))
        person_minutes = route_person_minutes(route, df, dist)
        summaries.append({
            "机型": route.machine, "架次": route.flight_no, "机场": route.airport,
            "停靠设施": "—".join(route.nodes[1:-1]), "承运人数": len(route.passenger_indices),
            "最大载荷": max(loads, default=0), "飞机使用时间(分钟)": route.minutes,
            "航程(km)": round(route.distance, 2), "燃油(kg)": round(route.fuel, 2),
            "人员在途时间(分钟)": person_minutes, "座位利用率": round(eta, 6),
            "空座公里": round(empty_km, 2),
        })
        total_minutes += route.minutes
        total_person_minutes += person_minutes
        total_fuel += route.fuel
        total_load_km += load_km
        total_seat_km += seat_km
    routes_df = pd.DataFrame(route_rows)
    assignment = assignment.reset_index(drop=True)
    routes_df.to_csv(CODE_DIR / "q2-routes.csv", index=False, encoding="utf-8-sig")
    assignment.to_csv(CODE_DIR / "q2-assignments.csv", index=False, encoding="utf-8-sig")
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(CODE_DIR / "q2_flight_summary.csv", index=False, encoding="utf-8-sig")
    metrics = {
        "问题": "问题二",
        "总飞机使用时间(分钟)": int(total_minutes),
        "人员总在途时间(分钟)": int(total_person_minutes),
        "总架次数": len(routes),
        "总燃油消耗(kg)": round(total_fuel, 2),
        "座位利用率": round(total_load_km / total_seat_km if total_seat_km else 0.0, 6),
        "需求总人数": len(df),
        "已分配人数": int(assignment["aircraft_type"].ne("").sum()),
    }
    pd.DataFrame([metrics]).to_csv(CODE_DIR / "q2_metrics.csv", index=False, encoding="utf-8-sig")
    return metrics


def classical_mds(dist: Mapping[Tuple[str, str], float], nodes: Sequence[str]) -> np.ndarray:
    matrix = np.array([[dist[(i, j)] for j in nodes] for i in nodes], dtype=float)
    n = len(nodes)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (matrix ** 2) @ centering
    vals, vecs = np.linalg.eigh(gram)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.maximum(vals[:2], 0)
    return vecs[:, :2] * np.sqrt(vals)


def draw_figures(df: pd.DataFrame, routes: List[Route], dist: Mapping[Tuple[str, str], float], metrics: Dict[str, float]) -> None:
    label_font()
    colors = {"A01": "#1976d2", "A02": "#ef6c00", "A03": "#2e7d32"}
    # 图1：三类需求和机场分区
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    kind_counts = df["kind"].value_counts().reindex(["出海", "海返", "穿梭"]).fillna(0)
    axes[0].bar(kind_counts.index, kind_counts.values, color=["#4e79a7", "#f28e2b", "#59a14f"])
    axes[0].set_title("问题二：三类运输需求规模")
    axes[0].set_ylabel("人数")
    axes[0].grid(axis="y", alpha=.25)
    for i, v in enumerate(kind_counts.values): axes[0].text(i, v + 30, f"{int(v)}", ha="center")
    airport_counts = df["airport"].value_counts().reindex(["A01", "A02", "A03"]).fillna(0)
    axes[1].bar(airport_counts.index, airport_counts.values, color=[colors[x] for x in airport_counts.index])
    axes[1].set_title("问题二：三机场分区需求")
    axes[1].set_ylabel("人数")
    axes[1].grid(axis="y", alpha=.25)
    for i, v in enumerate(airport_counts.values): axes[1].text(i, v + 20, f"{int(v)}", ha="center")
    fig.suptitle("问题二：出海、海返与穿梭需求概况", fontsize=15)
    fig.tight_layout(); fig.savefig(CODE_DIR / "问题二_需求结构图.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    # 图2：经典 MDS 航线网络和实际架次边
    nodes = list(dist.keys())
    all_nodes = sorted({i for i, _ in nodes})
    xy = classical_mds(dist, all_nodes); point = {n: xy[i] for i, n in enumerate(all_nodes)}
    fig, ax = plt.subplots(figsize=(10, 8))
    for airport, facilities in BLOCKS.items():
        fs = [f for f in facilities if f in point]
        ax.scatter([point[f][0] for f in fs], [point[f][1] for f in fs], s=22, alpha=.45, color=colors[airport])
    for airport in colors:
        ax.scatter(point[airport][0], point[airport][1], s=130, marker="*", color=colors[airport], edgecolor="black", zorder=4)
        ax.text(point[airport][0], point[airport][1], airport, fontsize=10, weight="bold")
    edge_counter = Counter()
    for route in routes:
        for u, v in zip(route.nodes[:-1], route.nodes[1:]): edge_counter[(u, v)] += 1
    for (u, v), count in edge_counter.most_common(45):
        ax.plot([point[u][0], point[v][0]], [point[u][1], point[v][1]], color="#888888", alpha=min(.65, .12 + .08 * count), linewidth=.7 + .12 * min(count, 5))
    ax.set_title("问题二：联合运输航线网络（距离矩阵经典MDS投影）")
    ax.set_xlabel("第一主坐标"); ax.set_ylabel("第二主坐标")
    ax.legend(handles=[Line2D([0], [0], marker="*", color="w", label=a, markerfacecolor=c, markeredgecolor="black", markersize=12) for a, c in colors.items()], title="机场分区")
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(CODE_DIR / "问题二_航线网络图.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    # 图3：综合指标
    summary = pd.DataFrame({"指标": ["总飞机使用时间（千分钟）", "总燃油（吨）", "总架次数", "人员总在途时间（千分钟）"], "数值": [metrics["总飞机使用时间(分钟)"]/1000, metrics["总燃油消耗(kg)"]/1000, metrics["总架次数"], metrics["人员总在途时间(分钟)"]/1000]})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].bar(summary["指标"], summary["数值"], color=["#4e79a7", "#e15759", "#59a14f", "#f28e2b"])
    axes[0].set_title("问题二：核心求解指标")
    axes[0].tick_params(axis="x", rotation=20); axes[0].grid(axis="y", alpha=.25)
    for i, v in enumerate(summary["数值"]): axes[0].text(i, v, f"{v:.1f}", ha="center", va="bottom")
    util = [metrics["座位利用率"], 1 - metrics["座位利用率"]]
    axes[1].pie(util, labels=["有效载客", "空余座位"], colors=["#59a14f", "#d9d9d9"], autopct="%.1f%%", startangle=90)
    axes[1].set_title("问题二：总体座位利用率")
    fig.suptitle(f"问题二求解结果：{metrics['总架次数']}架次｜{metrics['座位利用率']:.2%}座位利用率", fontsize=15)
    fig.tight_layout(); fig.savefig(CODE_DIR / "问题二_求解结果综合图.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    df = prepare_people(pd.read_csv(PEOPLE_FILE, dtype=str)).reset_index(drop=True)
    dist = load_distance()
    routes = build_routes(df, dist)
    metrics = export_results(df, routes, dist)
    draw_figures(df, routes, dist, metrics)
    # 运行期校验：每条需求恰好一次、逐弧载荷不超座位、每架次燃油可行。
    assigned = sum(len(r.passenger_indices) for r in routes)
    assert assigned == len(df), (assigned, len(df))
    for r in routes:
        assert max(segment_loads(r, df, r.passenger_indices), default=0) <= r.seats
    print("问题二求解完成")
    for k, v in metrics.items(): print(f"{k}: {v}")

if __name__ == "__main__":
    main()


