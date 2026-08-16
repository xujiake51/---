# -*- coding: utf-8 -*-
"""问题二、问题三共用的航线、容量、燃油和绘图工具。"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "2026年度“策联杯”数学建模精英联赛-B题-附件"
REFUEL_FACILITIES = {"F006", "F011", "F018", "F024", "F031", "F038", "F044", "F050"}
BLOCKS = {
    "A01": [f"F{i:03d}" for i in range(1, 18)],
    "A02": [f"F{i:03d}" for i in range(18, 36)],
    "A03": [f"F{i:03d}" for i in range(36, 53)],
}
FACILITY_TO_AIRPORT = {f: a for a, fs in BLOCKS.items() for f in fs}
MACHINES = {
    "T1": {"seats": 12, "speed": 250.0, "burn": 3.4, "tank": 1000.0, "reserve": 150.0},
    "T2": {"seats": 16, "speed": 220.0, "burn": 2.5, "tank": 1150.0, "reserve": 150.0},
    "T3": {"seats": 19, "speed": 190.0, "burn": 2.9, "tank": 1600.0, "reserve": 200.0},
}
MACHINE_ORDER = ("T3", "T2", "T1")
FLEET = {
    "A01": {"T1": 3, "T2": 3, "T3": 2},
    "A02": {"T1": 2, "T2": 4, "T3": 2},
    "A03": {"T1": 2, "T2": 3, "T3": 3},
}
MAX_STOPS = 5

@dataclass
class Route:
    airport: str
    machine: str
    nodes: Tuple[str, ...]
    refuels: Tuple[int, ...]
    minutes: int
    distance: float
    fuel: float
    passenger_indices: List[int] = field(default_factory=list)
    day: Optional[pd.Timestamp] = None
    start_minute: Optional[int] = None
    aircraft_id: str = ""
    flight_no: int = 0

    @property
    def seats(self) -> int:
        return MACHINES[self.machine]["seats"]


def load_distance() -> Dict[Tuple[str, str], float]:
    raw = pd.read_csv(DATA_DIR / "distances.csv", index_col=0)
    raw.index = raw.index.astype(str)
    raw.columns = raw.columns.astype(str)
    raw = raw.astype(float)
    return {(i, j): float(raw.loc[i, j]) for i in raw.index for j in raw.columns}


def facility_airport(node: str) -> Optional[str]:
    return FACILITY_TO_AIRPORT.get(node)


def assign_airport(origin: str, destination: str) -> str:
    """依据指定机场或设施分区，把一条需求归属到唯一机场。"""
    if origin in FLEET:
        return origin
    if destination in FLEET:
        return destination
    if origin.startswith("F"):
        return FACILITY_TO_AIRPORT[origin]
    if destination.startswith("F"):
        return FACILITY_TO_AIRPORT[destination]
    raise ValueError(f"无法判定需求机场：{origin}->{destination}")


def request_kind(origin: str, destination: str) -> str:
    if destination.startswith("F") and (origin == "LAND" or origin in FLEET):
        return "出海"
    if origin.startswith("F") and (destination == "LAND" or destination in FLEET):
        return "海返"
    return "穿梭"


def prepare_people(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["airport"] = [assign_airport(o, d) for o, d in zip(out.origin_id, out.destination_id)]
    out["kind"] = [request_kind(o, d) for o, d in zip(out.origin_id, out.destination_id)]
    return out


def flight_minutes(distance: float, machine: str) -> int:
    return int(math.ceil(60.0 * distance / MACHINES[machine]["speed"] - 1e-12))


def _best_route_basic(airport: str, stops: Sequence[str], machine: str,
                      dist: Mapping[Tuple[str, str], float],
                      fixed_order: Optional[Sequence[str]] = None) -> Optional[Route]:
    """不插入新设施时的路线枚举。"""
    stops = tuple(dict.fromkeys(stops))
    if not stops or len(stops) > MAX_STOPS:
        return None
    par = MACHINES[machine]
    best: Optional[Route] = None
    permutations_to_check = [tuple(fixed_order)] if fixed_order is not None else itertools.permutations(stops)
    for perm in permutations_to_check:
        nodes = (airport,) + tuple(perm) + (airport,)
        station_positions = [i for i, n in enumerate(perm, start=1) if n in REFUEL_FACILITIES]
        for mask in range(1 << len(station_positions)):
            fuel_left = par["tank"]
            feasible = True
            refs = [0] * len(nodes)
            total_distance = 0.0
            total_flight_minutes = 0
            ref_pos_set = {pos for b, pos in enumerate(station_positions) if mask & (1 << b)}
            for leg_i, (u, v) in enumerate(zip(nodes[:-1], nodes[1:])):
                d = float(dist[(u, v)])
                total_distance += d
                total_flight_minutes += flight_minutes(d, machine)
                fuel_left -= par["burn"] * d
                if fuel_left < par["reserve"] - 1e-9:
                    feasible = False
                    break
                if leg_i + 1 < len(nodes) - 1 and leg_i + 1 in ref_pos_set:
                    refs[leg_i + 1] = 1
                    fuel_left = par["tank"]
            if not feasible:
                continue
            minutes = total_flight_minutes + sum(20 if x else 10 for x in refs[1:-1])
            candidate = Route(airport, machine, nodes, tuple(refs), int(minutes), total_distance,
                              total_distance * par["burn"])
            score = (candidate.minutes, candidate.fuel, candidate.distance, candidate.nodes)
            if best is None or score < (best.minutes, best.fuel, best.distance, best.nodes):
                best = candidate
    return best


def best_route(airport: str, stops: Sequence[str], machine: str,
               dist: Mapping[Tuple[str, str], float],
               fixed_order: Optional[Sequence[str]] = None) -> Optional[Route]:
    """枚举燃油可行路线，并比较所有加油站插入位置。

    旧版本找到第一个可行加油站后立即返回，容易把加油站放在乘客目的地之前，
    从而使紧时间窗的到达任务被误判为不可行。这里保留所有候选路线，选择总
    飞行时间最短的方案；同样时长时优先少停靠、少燃油的方案。
    """
    stops = tuple(dict.fromkeys(stops))
    candidates: List[Route] = []
    base = _best_route_basic(airport, stops, machine, dist, fixed_order)
    if base is not None:
        candidates.append(base)
    if len(stops) < MAX_STOPS:
        stations = [s for s in REFUEL_FACILITIES
                    if facility_airport(s) == airport and s not in stops]
        if fixed_order is not None:
            order0 = list(fixed_order)
            for station in stations:
                for i in range(len(order0) + 1):
                    order = tuple(order0[:i] + [station] + order0[i:])
                    candidate = _best_route_basic(airport, order, machine, dist,
                                                  fixed_order=order)
                    if candidate is not None:
                        candidates.append(candidate)
        else:
            for station in stations:
                candidate = _best_route_basic(airport, tuple(stops) + (station,), machine, dist)
                if candidate is not None:
                    candidates.append(candidate)
    if not candidates:
        return None
    return min(candidates, key=lambda r: (r.minutes, len(r.nodes), r.fuel, r.distance, r.nodes))

def stop_positions(route: Route) -> Dict[str, int]:
    return {node: i for i, node in enumerate(route.nodes)}


def request_positions(row: pd.Series, route: Route) -> Optional[Tuple[int, int]]:
    pos = stop_positions(route)
    last = len(route.nodes) - 1
    def endpoint_position(node: str, is_origin: bool) -> Optional[int]:
        if node == "LAND" or node == route.airport:
            return 0 if is_origin else last
        if node in pos:
            return pos[node]
        return None
    p0 = endpoint_position(str(row.origin_id), True)
    p1 = endpoint_position(str(row.destination_id), False)
    if p0 is None or p1 is None or p0 >= p1:
        return None
    if str(row.origin_id) in FLEET and str(row.origin_id) != route.airport:
        return None
    if str(row.destination_id) in FLEET and str(row.destination_id) != route.airport:
        return None
    return p0, p1


def segment_loads(route: Route, df: pd.DataFrame, indices: Sequence[int]) -> List[int]:
    loads = [0] * (len(route.nodes) - 1)
    for idx in indices:
        pair = request_positions(df.loc[idx], route)
        if pair is None:
            continue
        p0, p1 = pair
        for s in range(p0, p1):
            loads[s] += 1
    return loads


def route_offsets(route: Route) -> Tuple[List[int], List[int]]:
    """返回各停靠点到达、离开架次起飞时刻的分钟偏移。"""
    arrival = [0] * len(route.nodes)
    departure = [0] * len(route.nodes)
    for i, (u, v) in enumerate(zip(route.nodes[:-1], route.nodes[1:])):
        arrival[i + 1] = arrival[i] + flight_minutes(
            # route distance is not enough, so this is filled by caller through _dist cache
            0.0, route.machine
        )
    return arrival, departure


def route_offsets_with_dist(route: Route, dist: Mapping[Tuple[str, str], float]) -> Tuple[List[int], List[int]]:
    arrival = [0] * len(route.nodes)
    departure = [0] * len(route.nodes)
    for i, (u, v) in enumerate(zip(route.nodes[:-1], route.nodes[1:])):
        arrival[i + 1] = arrival[i] + flight_minutes(float(dist[(u, v)]), route.machine)
        if i + 1 < len(route.nodes) - 1:
            departure[i + 1] = arrival[i + 1] + (20 if route.refuels[i + 1] else 10)
        else:
            departure[i + 1] = arrival[i + 1]
        # next leg begins after the stop
        if i + 1 < len(route.nodes) - 1:
            # arrival of next point will be corrected by using departure below
            pass
    # second pass because an internal stop adds dwell time to subsequent arrivals
    cur = 0
    for i, (u, v) in enumerate(zip(route.nodes[:-1], route.nodes[1:])):
        arrival[i] = cur
        cur += flight_minutes(float(dist[(u, v)]), route.machine)
        arrival[i + 1] = cur
        if i + 1 < len(route.nodes) - 1:
            departure[i + 1] = cur + (20 if route.refuels[i + 1] else 10)
            cur = departure[i + 1]
        else:
            departure[i + 1] = cur
    return arrival, departure


def passenger_times(row: pd.Series, route: Route,
                    dist: Mapping[Tuple[str, str], float]) -> Optional[Tuple[int, int]]:
    pair = request_positions(row, route)
    if pair is None:
        return None
    p0, p1 = pair
    arrival, departure = route_offsets_with_dist(route, dist)
    pickup_offset = 0 if p0 == 0 else departure[p0]
    arrival_offset = arrival[p1]
    return pickup_offset, arrival_offset


def feasible_start_interval(route: Route, df: pd.DataFrame,
                            indices: Sequence[int], day: pd.Timestamp,
                            dist: Mapping[Tuple[str, str], float]) -> Optional[Tuple[int, int]]:
    """根据人员时间窗求架次起飞时刻的整数分钟区间。"""
    day = pd.Timestamp(day).normalize()
    lo = 6 * 60
    hi = min(18 * 60, 20 * 60 - route.minutes)
    for idx in indices:
        row = df.loc[idx]
        times = passenger_times(row, route, dist)
        if times is None:
            return None
        pickup_offset, arrival_offset = times
        earliest = pd.Timestamp(row.earliest_pickup_time)
        latest = pd.Timestamp(row.latest_arrival_time)
        lo = max(lo, int(math.ceil((earliest - day).total_seconds() / 60.0 - pickup_offset)))
        hi = min(hi, int(math.floor((latest - day).total_seconds() / 60.0 - arrival_offset)))
    if lo > hi:
        return None
    return int(lo), int(hi)


def route_person_minutes(route: Route, df: pd.DataFrame, dist: Mapping[Tuple[str, str], float]) -> int:
    if route.day is None or route.start_minute is None:
        # 问题二没有绝对日期，人员在途时间按离开起点到到达终点计算
        total = 0
        for idx in route.passenger_indices:
            times = passenger_times(df.loc[idx], route, dist)
            if times:
                total += times[1] - times[0]
        return int(total)
    total = 0
    for idx in route.passenger_indices:
        times = passenger_times(df.loc[idx], route, dist)
        if times:
            total += times[1] - times[0]
    return int(total)


def route_empty_seat_distance(route: Route, df: pd.DataFrame,
                              dist: Mapping[Tuple[str, str], float]) -> Tuple[float, float]:
    loads = segment_loads(route, df, route.passenger_indices)
    seat_km = 0.0
    load_km = 0.0
    for s, load in enumerate(loads):
        d = float(dist[(route.nodes[s], route.nodes[s + 1])])
        seat_km += route.seats * d
        load_km += load * d
    return seat_km - load_km, (load_km / seat_km if seat_km else 0.0)


def candidate_stops(df: pd.DataFrame, indices: Sequence[int], airport: str,
                    dist: Mapping[Tuple[str, str], float], max_sets: int = 24) -> List[Tuple[str, ...]]:
    counts: Dict[str, int] = {}
    for idx in indices:
        row = df.loc[idx]
        for node in (str(row.origin_id), str(row.destination_id)):
            if node.startswith("F") and facility_airport(node) == airport:
                counts[node] = counts.get(node, 0) + 1
    if not counts:
        return []
    ranked = sorted(counts, key=lambda x: (-counts[x], x))
    seeds = ranked[: min(12, len(ranked))]
    result: List[Tuple[str, ...]] = []
    seen: Set[Tuple[str, ...]] = set()
    def add(x: Sequence[str]) -> None:
        y = tuple(dict.fromkeys(x))
        if y and len(y) <= MAX_STOPS and y not in seen:
            seen.add(y); result.append(y)
    # 先加入穿梭需求的有向端点对，避免末尾只剩一条穿梭需求时被截断。
    for idx in list(indices)[:80]:
        row = df.loc[idx]
        fs = [str(x) for x in (row.origin_id, row.destination_id) if str(x).startswith("F")]
        if len(fs) == 2 and all(facility_airport(x) == airport for x in fs):
            add(fs)
    for seed in seeds:
        add((seed,))
        chosen = [seed]
        pool = [x for x in ranked if x != seed]
        while pool and len(chosen) < MAX_STOPS:
            nxt = min(pool, key=lambda x: (dist[(chosen[-1], x)], -counts[x], x))
            chosen.append(nxt); pool.remove(nxt); add(chosen)
    # 一条按需求强度组织的全局路线
    add(ranked[:MAX_STOPS])
    return result[:max_sets]


def select_passengers(route: Route, df: pd.DataFrame, indices: Sequence[int],
                       dist: Mapping[Tuple[str, str], float],
                       day: Optional[pd.Timestamp] = None,
                       prefer_kinds: Sequence[str] = ()) -> List[int]:
    eligible = [idx for idx in indices if request_positions(df.loc[idx], route) is not None]
    if not eligible:
        return []
    priority = {k: i for i, k in enumerate(prefer_kinds)}
    def key(idx: int):
        row = df.loc[idx]
        p = request_positions(row, route)
        span = (p[1] - p[0]) if p else 99
        return (priority.get(str(row.get("task_type", "")), 99), span, str(row.person_id))
    eligible.sort(key=key)
    selected: List[int] = []
    for idx in eligible:
        trial = selected + [idx]
        loads = segment_loads(route, df, trial)
        if max(loads, default=0) > route.seats:
            continue
        if day is not None:
            if feasible_start_interval(route, df, trial, day, dist) is None:
                continue
        selected = trial
    return selected


def label_font():
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False





