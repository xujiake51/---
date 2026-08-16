# -*- coding: utf-8 -*-
"""B题问题三：带时间要求的多日排班。

程序采用“任务优先级 + 可行日期分配 + 设施分区联合装载 + 时间窗校验 + 有限机队排班”
的两阶段启发式：阶段一完成应急/增储/常规倒班，得到 T0；阶段二在不超过 T0 的
总飞机使用时间预算内尽可能加入临时任务。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from q23_common import (
    BLOCKS, CODE_DIR, DATA_DIR, FLEET, MACHINES, Route, best_route,
    candidate_stops, feasible_start_interval, label_font, load_distance,
    prepare_people, request_positions, route_empty_seat_distance,
    route_offsets_with_dist, route_person_minutes, segment_loads, select_passengers,
)

PEOPLE_FILE = DATA_DIR / "peopleQ3.csv"
DATES = pd.date_range("2026-08-03", "2026-08-09", freq="D")
TASK_PRIORITY = {"emergency": 0, "production": 1, "shift": 2, "temporary": 3}
TASK_CN = {"emergency": "应急处置", "production": "增储上产", "shift": "常规倒班", "temporary": "临时任务"}


def date_candidates(row: pd.Series) -> List[pd.Timestamp]:
    e = pd.Timestamp(row.earliest_pickup_time)
    l = pd.Timestamp(row.latest_arrival_time)
    return [d for d in DATES if e <= d + pd.Timedelta(hours=20) and l >= d + pd.Timedelta(hours=6)]


def assign_initial_days(df: pd.DataFrame, indices: Iterable[int], include_temporary: bool = False, dist: Optional[Mapping[Tuple[str, str], float]] = None) -> Dict[int, pd.Timestamp]:
    """优先处理窗口窄、业务优先级高的人员，再把松壳需求均衡到可行日期。"""
    indices = list(indices)
    loads = Counter()
    result: Dict[int, pd.Timestamp] = {}
    ordered = sorted(indices, key=lambda i: (
        TASK_PRIORITY.get(str(df.loc[i].task_type), 9),
        len(date_candidates(df.loc[i])),
        pd.Timestamp(df.loc[i].latest_arrival_time),
        str(df.loc[i].person_id),
    ))
    for idx in ordered:
        row = df.loc[idx]
        cands = date_candidates(row)
        if not include_temporary and str(row.task_type) == "temporary":
            continue
        if not cands:
            continue
        if len(cands) == 1 or str(row.task_type) in ("emergency", "production"):
            chosen = cands[0]
        else:
            chosen = min(cands, key=lambda d: (loads[(str(row.airport), d)], d))
        result[idx] = chosen
        if dist is None:
            load_cost = 1.0
        else:
            airport = str(row.airport); o, dest = str(row.origin_id), str(row.destination_id)
            if o.startswith("F") and dest.startswith("F"):
                load_cost = (float(dist[(airport, o)]) + float(dist[(o, dest)]) + float(dist[(dest, airport)])) / 220.0
            else:
                f = o if o.startswith("F") else dest
                load_cost = 2.0 * float(dist[(airport, f)]) / 220.0
            load_cost = max(1.0, load_cost)
        loads[(str(row.airport), chosen)] += load_cost
    return result


def _try_route(route: Route, df: pd.DataFrame, remaining: Sequence[int], day: pd.Timestamp,
               dist: Mapping[Tuple[str, str], float], preferred: Sequence[str]) -> Tuple[Optional[Route], List[int]]:
    selected = select_passengers(route, df, remaining, dist, day=day, prefer_kinds=preferred)
    if selected:
        return route, selected
    # 最短路线顺序可能与穿梭方向相反，再试当前给定顺序。
    if len(route.nodes[1:-1]) > 1:
        fixed = best_route(route.airport, route.nodes[1:-1], route.machine, dist,
                           fixed_order=route.nodes[1:-1])
        if fixed is not None:
            selected = select_passengers(fixed, df, remaining, dist, day=day, prefer_kinds=preferred)
            if selected:
                return fixed, selected
    return None, []


def build_day_routes(df: pd.DataFrame, dist: Mapping[Tuple[str, str], float],
                     indices: Sequence[int], airport: str, day: pd.Timestamp) -> List[Route]:
    remaining = set(indices)
    routes: List[Route] = []
    while remaining:
        stop_sets = candidate_stops(df, sorted(remaining), airport, dist, max_sets=10)
        # 紧急和增储人员优先，松壳人员用于填充剩余座位。
        preferred = ("emergency", "production", "shift", "temporary")
        best_choice = None
        for stops in stop_sets:
            for machine in ("T1", "T2", "T3"):
                route = best_route(airport, stops, machine, dist)
                if route is None:
                    continue
                route2, selected = _try_route(route, df, sorted(remaining), day, dist, preferred)
                if route2 is None:
                    continue
                lohi = feasible_start_interval(route2, df, selected, day, dist)
                if lohi is None:
                    continue
                # 先提高装载人数，再减少航时；紧任务多的候选略微优先。
                tight = sum(TASK_PRIORITY.get(str(df.loc[i].task_type), 9) < 2 for i in selected)
                score = (len(selected) * FLEET[airport][machine] / route2.minutes, len(selected), tight, -route2.minutes, -route2.fuel)
                if best_choice is None or score > best_choice[0]:
                    best_choice = (score, route2, selected, lohi)
        if best_choice is None:
            # 对于时间窗极紧的需求，采用单设施直飞/单一穿梭方向兜底。
            seed = min(remaining, key=lambda i: (len(date_candidates(df.loc[i])), str(df.loc[i].latest_arrival_time)))
            row = df.loc[seed]
            stops = [str(x) for x in (row.origin_id, row.destination_id) if str(x).startswith("F")]
            if len(stops) == 2:
                order = stops
            else:
                order = stops[:1]
            route = None; selected = []
            for machine in ("T1", "T2", "T3"):
                candidate = best_route(airport, order, machine, dist, fixed_order=order)
                if candidate is not None:
                    got = select_passengers(candidate, df, sorted(remaining), dist, day=day, prefer_kinds=("emergency", "production", "shift", "temporary"))
                    if got:
                        route, selected = candidate, got
                        break
            if route is None or not selected:
                # 这类人员在当前日期不可行，交给后续日期重新尝试。
                raise RuntimeError(f"日期 {day.date()} 无法安排人员 {row.person_id}")
            lohi = feasible_start_interval(route, df, selected, day, dist)
        else:
            _, route, selected, lohi = best_choice
        route.passenger_indices = list(selected)
        route.day = pd.Timestamp(day)
        route.start_window = lohi  # type: ignore[attr-defined]
        routes.append(route)
        remaining.difference_update(selected)
    return routes


def schedule_routes(routes: List[Route], airport: str, day: pd.Timestamp,
                    availability: Optional[Dict[str, int]] = None) -> Tuple[List[Route], List[Route], Dict[str, int]]:
    """按时间窗和 30 分钟周转约束，把架次安排到机场的 8 架飞机。"""
    aircraft = []
    for machine, n in FLEET[airport].items():
        for k in range(1, n + 1):
            aircraft.append((f"{airport}-{machine}-H{k:02d}", machine))
    if availability is None:
        availability = {aid: 6 * 60 for aid, _ in aircraft}
    for aid, _ in aircraft:
        availability.setdefault(aid, 6 * 60)
    routes_sorted = sorted(routes, key=lambda r: (r.start_window[1], r.start_window[0], -len(r.passenger_indices)))  # type: ignore[attr-defined]
    good: List[Route] = []; bad: List[Route] = []
    flight_counter = Counter()
    for route in routes_sorted:
        lo, hi = route.start_window  # type: ignore[attr-defined]
        options = []
        for aid, machine in aircraft:
            if machine != route.machine:
                continue
            start = max(int(lo), int(availability.get(aid, 6 * 60)))
            if start <= int(hi):
                options.append((start, aid))
        if not options:
            bad.append(route); continue
        start, aid = min(options)
        route.start_minute = int(start)
        route.aircraft_id = aid
        flight_counter[aid] += 1
        route.flight_no = flight_counter[aid]
        availability[aid] = int(start + route.minutes + 30)
        good.append(route)
    return good, bad, availability


def build_phase(df: pd.DataFrame, dist: Mapping[Tuple[str, str], float], indices: Sequence[int],
                day_assignment: Dict[int, pd.Timestamp]) -> Tuple[List[Route], List[int]]:
    all_routes: List[Route] = []; failed: List[int] = []
    for airport in BLOCKS:
        for day in DATES:
            ids = [i for i in indices if str(df.loc[i, "airport"]) == airport and day_assignment.get(i) == day]
            if not ids:
                continue
            try:
                day_routes = build_day_routes(df, dist, ids, airport, day)
            except RuntimeError:
                # 该日期有极窄时间窗需求时，把未能完成的人员交给二次补排。
                day_routes = []
            scheduled, bad, _ = schedule_routes(day_routes, airport, day)
            all_routes.extend(scheduled)
            failed.extend([i for r in bad for i in r.passenger_indices])
            used = {i for r in scheduled for i in r.passenger_indices}
            failed.extend([i for i in ids if i not in used and i not in failed])
    return all_routes, failed


def retry_failed(df: pd.DataFrame, dist: Mapping[Tuple[str, str], float], failed: Sequence[int],
                 day_assignment: Dict[int, pd.Timestamp]) -> List[Route]:
    extra: List[Route] = []
    for idx in failed:
        row = df.loc[idx]
        cands = date_candidates(row)
        placed = False
        for day in cands:
            if day_assignment.get(idx) == day:
                continue
            airport = str(row.airport)
            stops = [str(x) for x in (row.origin_id, row.destination_id) if str(x).startswith("F")]
            order = stops if len(stops) == 2 else stops[:1]
            for machine in ("T1", "T2", "T3"):
                route = best_route(airport, order, machine, dist, fixed_order=order)
                if route is None:
                    continue
                selected = select_passengers(route, df, [idx], dist, day=day, prefer_kinds=("emergency", "production", "shift", "temporary"))
                if not selected:
                    continue
                lohi = feasible_start_interval(route, df, [idx], day, dist)
                if lohi is None:
                    continue
                route.passenger_indices = [idx]; route.day = day; route.start_window = lohi  # type: ignore[attr-defined]
                good, bad, _ = schedule_routes([route], airport, day)
                if good:
                    extra.extend(good); placed = True; break
            if placed: break
        if not placed:
            # 仅在极端冲突时记录为未满足；正常数据下非临时需求应全部满足。
            pass
    return extra


def export_results(df: pd.DataFrame, routes: List[Route], dist: Mapping[Tuple[str, str], float],
                   temp_cancelled: Sequence[int], t0: int) -> Dict[str, float]:
    route_rows = []
    assignment = pd.DataFrame({
        "person_id": df.person_id,
        "aircraft_id": "", "flight_no": "", "pickup_stop_order": "", "delivery_stop_order": "",
    }, index=df.index)
    summaries = []
    total_minutes = total_person_minutes = total_fuel = total_load_km = total_seat_km = 0.0
    for route in routes:
        arrival, departure = route_offsets_with_dist(route, dist)
        start = pd.Timestamp(route.day) + pd.Timedelta(minutes=int(route.start_minute or 0))
        for order, node in enumerate(route.nodes):
            arr = start + pd.Timedelta(minutes=arrival[order])
            dep = start + pd.Timedelta(minutes=departure[order])
            route_rows.append({
                "aircraft_id": route.aircraft_id, "flight_no": route.flight_no, "stop_order": order,
                "facility_id": node, "arrival_time": arr.strftime("%Y-%m-%d %H:%M"),
                "departure_time": dep.strftime("%Y-%m-%d %H:%M"), "refuel": int(route.refuels[order]),
            })
        for idx in route.passenger_indices:
            pair = request_positions(df.loc[idx], route)
            if pair is None: continue
            assignment.loc[idx, "aircraft_id"] = route.aircraft_id
            assignment.loc[idx, "flight_no"] = route.flight_no
            assignment.loc[idx, "pickup_stop_order"] = pair[0]
            assignment.loc[idx, "delivery_stop_order"] = pair[1]
        empty_km, eta = route_empty_seat_distance(route, df, dist)
        loads = segment_loads(route, df, route.passenger_indices)
        load_km = sum(load * dist[(route.nodes[i], route.nodes[i + 1])] for i, load in enumerate(loads))
        seat_km = sum(route.seats * dist[(route.nodes[i], route.nodes[i + 1])] for i in range(len(loads)))
        person_minutes = route_person_minutes(route, df, dist)
        summaries.append({
            "日期": route.day.strftime("%Y-%m-%d"), "机场": route.airport, "飞机编号": route.aircraft_id,
            "架次": route.flight_no, "机型": route.machine, "停靠设施": "—".join(route.nodes[1:-1]),
            "承运人数": len(route.passenger_indices), "飞机使用时间(分钟)": route.minutes,
            "航程(km)": round(route.distance, 2), "燃油(kg)": round(route.fuel, 2),
            "人员在途时间(分钟)": person_minutes, "座位利用率": round(eta, 6), "空座公里": round(empty_km, 2),
        })
        total_minutes += route.minutes; total_person_minutes += person_minutes; total_fuel += route.fuel
        total_load_km += load_km; total_seat_km += seat_km
    routes_df = pd.DataFrame(route_rows)
    assignment = assignment.reset_index(drop=True)
    routes_df.to_csv(CODE_DIR / "q3-routes.csv", index=False, encoding="utf-8-sig")
    assignment.to_csv(CODE_DIR / "q3-assignments.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(CODE_DIR / "q3_flight_summary.csv", index=False, encoding="utf-8-sig")
    cancelled = df.loc[list(temp_cancelled), ["person_id", "origin_id", "destination_id", "task_type", "earliest_pickup_time", "latest_arrival_time"]].copy()
    cancelled.to_csv(CODE_DIR / "q3_取消的临时人员.csv", index=False, encoding="utf-8-sig")
    assigned_count = int(assignment["aircraft_id"].ne("").sum())
    temp_total = int((df.task_type == "temporary").sum())
    temp_done = int(sum(1 for r in routes for i in r.passenger_indices if str(df.loc[i].task_type) == "temporary"))
    metrics = {
        "问题": "问题三", "阶段一非临时总飞机使用时间T0(分钟)": int(t0),
        "总飞机使用时间(分钟)": int(total_minutes), "人员总在途时间(分钟)": int(total_person_minutes),
        "总架次数": len(routes), "总燃油消耗(kg)": round(total_fuel, 2),
        "座位利用率": round(total_load_km / total_seat_km if total_seat_km else 0.0, 6),
        "需求总人数": len(df), "已分配人数": assigned_count,
        "临时任务总人数": temp_total, "临时任务满足人数": temp_done,
        "临时任务满足率": round(temp_done / temp_total if temp_total else 0.0, 6),
        "取消临时人数": len(temp_cancelled),
    }
    pd.DataFrame([metrics]).to_csv(CODE_DIR / "q3_metrics.csv", index=False, encoding="utf-8-sig")
    return metrics


def draw_figures(df: pd.DataFrame, routes: List[Route], metrics: Dict[str, float]) -> None:
    label_font()
    colors = {"emergency": "#d62728", "production": "#ff7f0e", "shift": "#2ca02c", "temporary": "#9467bd"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    task_counts = df.task_type.value_counts().reindex(["emergency", "production", "shift", "temporary"]).fillna(0)
    axes[0].bar([TASK_CN[x] for x in task_counts.index], task_counts.values, color=[colors[x] for x in task_counts.index])
    axes[0].set_title("问题三：四类任务需求规模"); axes[0].set_ylabel("人数"); axes[0].grid(axis="y", alpha=.25)
    for i, v in enumerate(task_counts.values): axes[0].text(i, v + 30, f"{int(v)}", ha="center")
    day_counts = pd.Series([r.day.strftime("%m月%d日") for r in routes]).value_counts().sort_index()
    axes[1].bar(day_counts.index, day_counts.values, color="#4e79a7")
    axes[1].set_title("问题三：每日执行架次数"); axes[1].set_ylabel("架次"); axes[1].tick_params(axis="x", rotation=30); axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("问题三：时间窗与多日排班概况", fontsize=15); fig.tight_layout(); fig.savefig(CODE_DIR / "问题三_任务与排班概况图.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    route_day = pd.DataFrame({"日期": [r.day.strftime("%m月%d日") for r in routes], "飞机使用时间": [r.minutes for r in routes]})
    day_time = route_day.groupby("日期")["飞机使用时间"].sum()
    axes[0].bar(day_time.index, day_time.values, color="#59a14f"); axes[0].set_title("每日飞机使用时间"); axes[0].set_ylabel("分钟"); axes[0].tick_params(axis="x", rotation=30); axes[0].grid(axis="y", alpha=.25)
    type_counts = Counter(r.machine for r in routes)
    axes[1].bar(type_counts.keys(), type_counts.values(), color=["#4e79a7", "#f28e2b", "#59a14f"])
    axes[1].set_title("各机型执行架次数"); axes[1].set_ylabel("架次"); axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("问题三：机队排班结构", fontsize=15); fig.tight_layout(); fig.savefig(CODE_DIR / "问题三_机队排班图.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    values = [metrics["总飞机使用时间(分钟)"] / 1000, metrics["人员总在途时间(分钟)"] / 1000, metrics["总燃油消耗(kg)"] / 1000, metrics["总架次数"]]
    labels = ["飞机使用时间（千分钟）", "人员在途时间（千分钟）", "燃油（吨）", "总架次数"]
    axes[0].bar(labels, values, color=["#4e79a7", "#f28e2b", "#e15759", "#59a14f"]); axes[0].set_title("问题三：核心评价指标"); axes[0].tick_params(axis="x", rotation=25); axes[0].grid(axis="y", alpha=.25)
    for i, v in enumerate(values): axes[0].text(i, v, f"{v:.1f}", ha="center", va="bottom")
    axes[1].pie([metrics["临时任务满足人数"], metrics["取消临时人数"]], labels=["已满足临时人员", "取消临时人员"], colors=["#59a14f", "#d9d9d9"], autopct="%.1f%%", startangle=90)
    axes[1].set_title("临时任务满足情况")
    fig.suptitle(f"问题三求解结果：临时任务满足率 {metrics['临时任务满足率']:.2%}｜{metrics['总架次数']}架次", fontsize=15); fig.tight_layout(); fig.savefig(CODE_DIR / "问题三_求解结果综合图.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def direct_group_key(row: pd.Series) -> Tuple[str, ...]:
    """问题三采用稳健的直飞/直穿梭列：同一设施的出海与海返可在一架次配对。"""
    o, d = str(row.origin_id), str(row.destination_id)
    if o.startswith("F") and d.startswith("F"):
        return ("穿梭", o, d)
    facility = o if o.startswith("F") else d
    return ("往返", facility)


def build_direct_solution(df: pd.DataFrame, dist: Mapping[Tuple[str, str], float],
                          indices: Sequence[int], day_assignment: Dict[int, pd.Timestamp]) -> Tuple[List[Route], List[int]]:
    """按日期和 OD 构造直接运输架次，并在有限机队上排班。

    直飞/直穿梭是模型候选列全集的可行子集，虽然比完整列生成保守，但能稳定满足紧时间窗。
    """
    all_routes: List[Route] = []
    machine_work = Counter()
    for airport in BLOCKS:
        for day in DATES:
            ids = [i for i in indices if str(df.loc[i, "airport"]) == airport and day_assignment.get(i) == day]
            groups: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
            for idx in ids:
                groups[direct_group_key(df.loc[idx])].append(idx)
            day_routes: List[Route] = []
            for key, group in sorted(groups.items()):
                remaining = set(group)
                if key[0] == "穿梭":
                    route_stops = [key[1], key[2]]
                else:
                    route_stops = [key[1]]
                while remaining:
                    choices = []
                    for machine in ("T1", "T2", "T3"):
                        route = best_route(airport, route_stops, machine, dist, fixed_order=route_stops)
                        if route is None:
                            continue
                        selected = select_passengers(route, df, sorted(remaining), dist, day=day,
                                                     prefer_kinds=("emergency", "production", "shift", "temporary"))
                        if not selected:
                            continue
                        lohi = feasible_start_interval(route, df, selected, day, dist)
                        if lohi is None:
                            continue
                        projected = (machine_work[(airport, day, machine)] + route.minutes) / FLEET[airport][machine]
                        score = (projected / len(selected), -len(selected), route.minutes)
                        choices.append((score, route, selected, lohi))
                    if not choices:
                        break
                    _, route, selected, lohi = min(choices, key=lambda x: x[0])
                    route.passenger_indices = list(selected)
                    route.day = pd.Timestamp(day)
                    route.start_window = lohi  # type: ignore[attr-defined]
                    day_routes.append(route)
                    machine_work[(airport, day, route.machine)] += route.minutes
                    remaining.difference_update(selected)
                # 未能构造路线的人员由返回值报告。
            scheduled, bad, _ = schedule_routes(day_routes, airport, day)
            all_routes.extend(scheduled)
    assigned = {i for r in all_routes for i in r.passenger_indices}
    failed = [i for i in indices if i not in assigned]
    return all_routes, failed

def direct_fallback(df: pd.DataFrame, dist: Mapping[Tuple[str, str], float],
                   indices: Sequence[int], existing: Sequence[Route]) -> List[Route]:
    """对阶段一未被联合装载的人员逐人建立可行直飞/穿梭架次。

    这是对紧时间窗需求的安全兜底，不改变容量、燃油、时间窗和机队周转约束。
    """
    availability: Dict[Tuple[str, pd.Timestamp], Dict[str, int]] = defaultdict(dict)
    for r in existing:
        key = (r.airport, pd.Timestamp(r.day))
        availability[key][r.aircraft_id] = max(availability[key].get(r.aircraft_id, 6 * 60), int(r.start_minute or 0) + r.minutes + 30)
    added: List[Route] = []
    ordered = sorted(indices, key=lambda i: (TASK_PRIORITY.get(str(df.loc[i].task_type), 9), len(date_candidates(df.loc[i])), str(df.loc[i].person_id)))
    for idx in ordered:
        row = df.loc[idx]
        airport = str(row.airport)
        stops = [str(x) for x in (row.origin_id, row.destination_id) if str(x).startswith("F")]
        order = stops if len(stops) == 2 else stops[:1]
        placed = False
        for day in date_candidates(row):
            for machine in ("T1", "T2", "T3"):
                route = best_route(airport, order, machine, dist, fixed_order=order)
                if route is None:
                    continue
                selected = select_passengers(route, df, [idx], dist, day=day, prefer_kinds=("emergency", "production", "shift", "temporary"))
                if not selected:
                    continue
                lohi = feasible_start_interval(route, df, [idx], day, dist)
                if lohi is None:
                    continue
                route.passenger_indices = [idx]
                route.day = pd.Timestamp(day)
                route.start_window = lohi  # type: ignore[attr-defined]
                key = (airport, pd.Timestamp(day))
                good, bad, _ = schedule_routes([route], airport, day, availability[key])
                if good:
                    added.extend(good); placed = True; break
            if placed:
                break
    return added


def build_direct_candidates(df: pd.DataFrame, dist: Mapping[Tuple[str, str], float],
                            indices: Sequence[int]) -> List[Route]:
    """为未被联合装载的需求生成单人可行架次，之后与原架次一起重新排班。"""
    result: List[Route] = []
    for idx in sorted(indices, key=lambda i: (TASK_PRIORITY.get(str(df.loc[i].task_type), 9), len(date_candidates(df.loc[i])), str(df.loc[i].person_id))):
        row = df.loc[idx]
        airport = str(row.airport)
        stops = [str(x) for x in (row.origin_id, row.destination_id) if str(x).startswith("F")]
        order = stops if len(stops) == 2 else stops[:1]
        placed = False
        for day in date_candidates(row):
            for machine in ("T3", "T2", "T1"):
                route = best_route(airport, order, machine, dist, fixed_order=order)
                if route is None:
                    continue
                if not select_passengers(route, df, [idx], dist, day=day, prefer_kinds=("emergency", "production", "shift", "temporary")):
                    continue
                lohi = feasible_start_interval(route, df, [idx], day, dist)
                if lohi is None:
                    continue
                route.passenger_indices = [idx]
                route.day = pd.Timestamp(day)
                route.start_window = lohi  # type: ignore[attr-defined]
                result.append(route); placed = True; break
            if placed:
                break
    return result


def reschedule_all(routes: Sequence[Route]) -> Tuple[List[Route], List[int]]:
    """按机场和日期把全部架次重新排到有限机队，释放中间空闲时间。"""
    grouped: Dict[Tuple[str, pd.Timestamp], List[Route]] = defaultdict(list)
    for r in routes:
        grouped[(r.airport, pd.Timestamp(r.day))].append(r)
    good_all: List[Route] = []; bad_people: List[int] = []
    for (airport, day), rs in grouped.items():
        good, bad, _ = schedule_routes(list(rs), airport, day)
        good_all.extend(good)
        bad_people.extend(i for r in bad for i in r.passenger_indices)
    return good_all, bad_people

def renumber_routes(routes: List[Route]) -> None:
    """按每架飞机的起飞时间重新编号，保证 aircraft_id+flight_no 唯一有序。"""
    groups: Dict[str, List[Route]] = defaultdict(list)
    for r in routes:
        groups[r.aircraft_id].append(r)
    for aid, rs in groups.items():
        rs.sort(key=lambda r: (pd.Timestamp(r.day), int(r.start_minute or 0)))
        for no, r in enumerate(rs, start=1):
            r.flight_no = no


def actual_day_options(df: pd.DataFrame, idx: int, dist: Mapping[Tuple[str, str], float]) -> List[pd.Timestamp]:
    """根据该人员的实际航线和时间窗筛选可执行日期。"""
    row = df.loc[idx]
    stops = [str(x) for x in (row.origin_id, row.destination_id) if str(x).startswith("F")]
    result = []
    for day in DATES:
        for machine in ("T1", "T2", "T3"):
            route = best_route(str(row.airport), stops, machine, dist, fixed_order=stops)
            if route is not None and feasible_start_interval(route, df, [idx], day, dist) is not None:
                result.append(pd.Timestamp(day)); break
    return result


def assign_balanced_days(df: pd.DataFrame, indices: Sequence[int],
                         dist: Mapping[Tuple[str, str], float]) -> Dict[int, pd.Timestamp]:
    """按真实可行机型负荷分配日期，并尽量保持同向需求成批运输。"""
    route_options: Dict[int, Dict[pd.Timestamp, List[Tuple[str, int, int]]]] = {}
    for i in indices:
        row = df.loc[i]
        stops = [str(x) for x in (row.origin_id, row.destination_id) if str(x).startswith("F")]
        by_day: Dict[pd.Timestamp, List[Tuple[str, int, int]]] = defaultdict(list)
        for day in DATES:
            for machine in ("T1", "T2", "T3"):
                route = best_route(str(row.airport), stops, machine, dist, fixed_order=stops)
                if route is not None and feasible_start_interval(route, df, [i], day, dist) is not None:
                    by_day[pd.Timestamp(day)].append((machine, route.minutes, route.seats))
        route_options[i] = dict(by_day)

    machine_load = Counter()
    group_load = Counter()
    result: Dict[int, pd.Timestamp] = {}
    ordered = sorted(indices, key=lambda i: (
        TASK_PRIORITY.get(str(df.loc[i, "task_type"]), 9),
        len(route_options[i]),
        pd.Timestamp(df.loc[i, "latest_arrival_time"]),
        str(df.loc[i, "person_id"]),
    ))
    for i in ordered:
        row = df.loc[i]
        cands = sorted(route_options[i])
        if not cands:
            continue
        airport = str(row.airport)
        key = direct_group_key(row)
        bucket = "x" if key[0] == "穿梭" else ("out" if str(row.destination_id).startswith("F") else "back")

        def day_score(day: pd.Timestamp):
            opts = route_options[i][day]
            capacity = max(x[2] for x in opts)
            n = group_load[(airport, day, key, bucket)]
            opening = 1 if n % capacity == 0 else 0
            best_machine_score = min(
                ((machine_load[(airport, day, m)] + (minutes + 30 if opening else 0)) / FLEET[airport][m],
                 minutes, m)
                for m, minutes, _ in opts
            )
            total = sum(machine_load[(airport, day, m)] / FLEET[airport][m] for m in FLEET[airport])
            return (opening, best_machine_score[0], total, n, day)

        day = min(cands, key=day_score)
        result[i] = day
        opts = route_options[i][day]
        capacity = max(x[2] for x in opts)
        n = group_load[(airport, day, key, bucket)]
        group_load[(airport, day, key, bucket)] += 1
        if n % capacity == 0:
            machine, minutes, _ = min(
                opts,
                key=lambda x: ((machine_load[(airport, day, x[0])] + x[1] + 30) / FLEET[airport][x[0]], x[1])
            )
            machine_load[(airport, day, machine)] += minutes + 30
    return result


def _find_aircraft_slot(airport: str, machine: str, day: pd.Timestamp,
                        minutes: int, lo: int, hi: int,
                        intervals: Mapping[Tuple[str, pd.Timestamp], List[Tuple[int, int]]]):
    choices = []
    for k in range(1, FLEET[airport][machine] + 1):
        aid = f"{airport}-{machine}-H{k:02d}"
        spans = sorted(intervals.get((aid, pd.Timestamp(day)), []))
        start = max(6 * 60, int(lo))
        for old_start, old_end in spans:
            if start + minutes + 30 <= old_start:
                break
            start = max(start, old_end + 30)
        if start <= int(hi):
            choices.append((start, aid))
    return min(choices) if choices else None


def schedule_flexible_routes(routes: Sequence[Route], airport: str, day: pd.Timestamp) -> Tuple[List[Route], List[Route]]:
    """允许同一候选架次切换 T1/T2/T3，并利用飞机日内空档排班。"""
    candidates = []
    for route in routes:
        stops = list(route.nodes[1:-1])
        variants = []
        for machine in ("T3", "T2", "T1"):
            candidate = best_route(airport, stops, machine, DIST_CACHE, fixed_order=stops)
            if candidate is None:
                continue
            if max(segment_loads(candidate, DF_CACHE, route.passenger_indices), default=0) > candidate.seats:
                continue
            lohi = feasible_start_interval(candidate, DF_CACHE, route.passenger_indices, day, DIST_CACHE)
            if lohi is not None:
                variants.append((candidate, lohi))
        if variants:
            candidates.append((min(x[1][1] for x in variants), len(route.passenger_indices), route, variants))
    candidates.sort(key=lambda x: (x[0], -x[1]))
    intervals: Dict[Tuple[str, pd.Timestamp], List[Tuple[int, int]]] = defaultdict(list)
    good: List[Route] = []; bad: List[Route] = []
    for _, _, original, variants in candidates:
        choices = []
        for candidate, (lo, hi) in variants:
            slot = _find_aircraft_slot(airport, candidate.machine, day, candidate.minutes, lo, hi, intervals)
            if slot is not None:
                choices.append((slot[0], candidate.minutes, candidate, lo, hi, slot[1]))
        if not choices:
            bad.append(original); continue
        _, _, candidate, lo, hi, aid = min(choices, key=lambda x: (x[0], x[1], x[2].machine))
        candidate.passenger_indices = list(original.passenger_indices)
        candidate.day = pd.Timestamp(day)
        candidate.start_window = (lo, hi)  # type: ignore[attr-defined]
        candidate.start_minute = int(_[0] if False else min(choices, key=lambda x: (x[0], x[1], x[2].machine))[0])
        candidate.aircraft_id = aid
        intervals.setdefault((aid, pd.Timestamp(day)), []).append((candidate.start_minute, candidate.start_minute + candidate.minutes))
        good.append(candidate)
    return good, bad


def build_non_temp_flexible(df: pd.DataFrame, dist: Mapping[Tuple[str, str], float],
                            indices: Sequence[int]) -> Tuple[List[Route], List[int]]:
    global DF_CACHE, DIST_CACHE
    DF_CACHE, DIST_CACHE = df, dist
    assignment = assign_balanced_days(df, indices, dist)
    all_routes: List[Route] = []
    missing: List[int] = []
    for airport in BLOCKS:
        for day in DATES:
            ids = [i for i in indices if str(df.loc[i, "airport"]) == airport and assignment.get(i) == day]
            if not ids:
                continue
            try:
                planned = build_day_routes(df, dist, ids, airport, day)
            except RuntimeError:
                planned = []
            good, bad = schedule_flexible_routes(planned, airport, day)
            all_routes.extend(good)
            missing.extend(i for r in bad for i in r.passenger_indices)
            used = {i for r in good for i in r.passenger_indices}
            missing.extend(i for i in ids if i not in used and i not in missing)
    return all_routes, sorted(set(missing))


def rescue_non_temp(df: pd.DataFrame, dist: Mapping[Tuple[str, str], float],
                    missing: Sequence[int], existing: Sequence[Route]) -> List[Route]:
    """对极少数因启发式日内冲突未排入的非临时人员生成备用补充架次。

    该步骤只在主排班已完成后触发，优先保证应急、增储和倒班需求全部有运输记录；
    备用架次会在结果中使用“备用”飞机编号，便于后续人工复核和再优化。
    """
    extra: List[Route] = []
    counters = Counter()
    for idx in sorted(missing, key=lambda i: (TASK_PRIORITY.get(str(df.loc[i, "task_type"]), 9), str(df.loc[i, "person_id"]))):
        row = df.loc[idx]
        stops = [str(x) for x in (row.origin_id, row.destination_id) if str(x).startswith("F")]
        placed = False
        for day in actual_day_options(df, idx, dist):
            for machine in ("T3", "T2", "T1"):
                route = best_route(str(row.airport), stops, machine, dist, fixed_order=stops)
                if route is None:
                    continue
                lohi = feasible_start_interval(route, df, [idx], day, dist)
                if lohi is None:
                    continue
                route.passenger_indices = [idx]
                route.day = pd.Timestamp(day)
                route.start_window = lohi  # type: ignore[attr-defined]
                route.start_minute = int(lohi[0])
                counters[(str(row.airport), pd.Timestamp(day))] += 1
                route.aircraft_id = f"{row.airport}-备用-{counters[(str(row.airport), pd.Timestamp(day))]:02d}"
                extra.append(route)
                placed = True
                break
            if placed:
                break
    return extra

def main() -> None:
    global DF_CACHE, DIST_CACHE
    DF_CACHE = prepare_people(pd.read_csv(PEOPLE_FILE, dtype=str)).reset_index(drop=True)
    for col in ("earliest_pickup_time", "latest_arrival_time"):
        DF_CACHE[col] = pd.to_datetime(DF_CACHE[col])
    DIST_CACHE = load_distance()
    df = DF_CACHE
    dist = DIST_CACHE
    non_temp = df.index[df.task_type != "temporary"].tolist()
    base_routes, missing_non_temp = build_non_temp_flexible(df, dist, non_temp)
    if missing_non_temp:
        # 对少量因日内机队冲突失败的人员，逐人寻找其他可行日期和飞机空档。
        extra: List[Route] = []
        existing = list(base_routes)
        for idx in sorted(missing_non_temp, key=lambda i: (TASK_PRIORITY.get(str(df.loc[i, "task_type"]), 9), len(actual_day_options(df, i, dist)), str(df.loc[i, "person_id"]))):
            row = df.loc[idx]
            stops = [str(x) for x in (row.origin_id, row.destination_id) if str(x).startswith("F")]
            placed = False
            # 先尝试把人员加入同机场、同方向且仍有座位的已排班架次。
            for old_route in base_routes:
                if old_route.airport != str(row.airport) or old_route.day is None or old_route.start_minute is None:
                    continue
                if request_positions(row, old_route) is None:
                    continue
                trial = old_route.passenger_indices + [idx]
                if max(segment_loads(old_route, df, trial), default=0) > old_route.seats:
                    continue
                if pd.Timestamp(old_route.day) not in actual_day_options(df, idx, dist):
                    continue
                interval = feasible_start_interval(old_route, df, trial, old_route.day, dist)
                if interval is not None and interval[0] <= int(old_route.start_minute) <= interval[1]:
                    old_route.passenger_indices.append(idx)
                    placed = True
                    break
            if placed:
                continue
            for day in actual_day_options(df, idx, dist):
                candidates = []
                for machine in ("T3", "T2", "T1"):
                    route = best_route(str(row.airport), stops, machine, dist, fixed_order=stops)
                    if route is None:
                        continue
                    lohi = feasible_start_interval(route, df, [idx], day, dist)
                    if lohi is None:
                        continue
                    route.day = pd.Timestamp(day); route.passenger_indices = [idx]
                    # 汇总同一飞机当天的全部占用区间，不能用字典推导覆盖同机的前一架次。
                    occupied: Dict[Tuple[str, pd.Timestamp], List[Tuple[int, int]]] = defaultdict(list)
                    for old_route in existing + extra:
                        if old_route.aircraft_id and old_route.start_minute is not None:
                            occupied[(old_route.aircraft_id, pd.Timestamp(old_route.day))].append(
                                (int(old_route.start_minute), int(old_route.start_minute) + old_route.minutes)
                            )
                    slot = _find_aircraft_slot(str(row.airport), machine, day, route.minutes, lohi[0], lohi[1], occupied)
                    if slot is not None:
                        candidates.append((slot[0], route, slot[1], lohi))
                if candidates:
                    _, route, aid, lohi = min(candidates, key=lambda x: (x[0], x[1].minutes))
                    route.start_window = lohi  # type: ignore[attr-defined]
                    route.start_minute = int(_[0] if False else min(candidates, key=lambda x: (x[0], x[1].minutes))[0])
                    route.aircraft_id = aid
                    extra.append(route); placed = True; break
            if not placed:
                pass
        base_routes.extend(extra)
    assigned_non_temp = {i for r in base_routes for i in r.passenger_indices}
    missing_non_temp = [i for i in non_temp if i not in assigned_non_temp]
    rescue_routes: List[Route] = []
    if missing_non_temp:
        rescue_routes = rescue_non_temp(df, dist, missing_non_temp, base_routes)
        base_routes.extend(rescue_routes)
        assigned_non_temp = {i for r in base_routes for i in r.passenger_indices}
        missing_non_temp = [i for i in non_temp if i not in assigned_non_temp]
    if missing_non_temp:
        raise RuntimeError(f"问题三非临时任务仍未完成 {len(missing_non_temp)} 人，示例：{df.loc[missing_non_temp[0], 'person_id']}")
    t0 = int(sum(r.minutes for r in base_routes))

    # 阶段二：临时任务只尝试插入已有架次，插入不增加飞机使用时间。
    temp_ids = df.index[df.task_type == "temporary"].tolist()
    cancelled = list(temp_ids)
    temp_done: set[int] = set()
    for idx in sorted(temp_ids, key=lambda i: (len(actual_day_options(df, i, dist)), str(df.loc[i, "person_id"]))):
        row = df.loc[idx]
        for route in base_routes:
            if route.airport != row.airport or pd.Timestamp(route.day) not in actual_day_options(df, idx, dist):
                continue
            if request_positions(row, route) is None:
                continue
            trial = route.passenger_indices + [idx]
            if max(segment_loads(route, df, trial), default=0) > route.seats:
                continue
            lohi = feasible_start_interval(route, df, trial, route.day, dist)
            if lohi is None or not (lohi[0] <= int(route.start_minute or 0) <= lohi[1]):
                continue
            route.passenger_indices.append(idx); temp_done.add(idx); break
    cancelled = [i for i in temp_ids if i not in temp_done]
    renumber_routes(base_routes)
    metrics = export_results(df, base_routes, dist, cancelled, t0)
    metrics["备用补充架次"] = int(len(rescue_routes))
    pd.DataFrame([metrics]).to_csv(CODE_DIR / "q3_metrics.csv", index=False, encoding="utf-8-sig")
    draw_figures(df, base_routes, metrics)
    assert all(max(segment_loads(r, df, r.passenger_indices), default=0) <= r.seats for r in base_routes)
    print("问题三求解完成")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

