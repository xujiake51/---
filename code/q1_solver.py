# -*- coding: utf-8 -*-
"""B题问题1：单向出海运输求解器。运行：python code/q1_solver.py"""
from __future__ import annotations
import heapq, itertools, math, time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "2026年度“策联杯”数学建模精英联赛-B题-附件"
PEOPLE_FILE = DATA_DIR / "peopleQ1.csv"
DIST_FILE = DATA_DIR / "distances.csv"
REFUEL_FACILITIES = {"F006", "F011", "F018", "F024", "F031", "F038", "F044", "F050"}
BLOCKS = {
    "A01": [f"F{i:03d}" for i in range(1, 18)],
    "A02": [f"F{i:03d}" for i in range(18, 36)],
    "A03": [f"F{i:03d}" for i in range(36, 53)],
}
MACHINES = {
    "T1": {"seats": 12, "speed": 250.0, "burn": 3.4, "tank": 1000.0, "reserve": 150.0},
    "T2": {"seats": 16, "speed": 220.0, "burn": 2.5, "tank": 1150.0, "reserve": 150.0},
    "T3": {"seats": 19, "speed": 190.0, "burn": 2.9, "tank": 1600.0, "reserve": 200.0},
}
CAPACITY_TO_MACHINE = {v["seats"]: k for k, v in MACHINES.items()}
OPTION_CAPACITIES = (12, 16, 19)
MAX_STOPS = 5
FUEL_EPS = 1e-8

@dataclass(frozen=True)
class RouteResult:
    aircraft_type: str
    nodes: Tuple[str, ...]
    refuels: Tuple[int, ...]
    minutes: int
    distance: float
    fuel: float

@dataclass(frozen=True)
class SplitOption:
    option_id: int
    facility: str
    capacity: int
    full_count: int
    remainder: int
    full_route: Optional[RouteResult]

@dataclass
class TailColumn:
    items: Tuple[int, ...]
    facilities: Tuple[str, ...]
    passengers: int
    nominal_route: RouteResult
    exact_route: Optional[RouteResult] = None
    disabled: bool = False
    @property
    def active_route(self) -> RouteResult:
        return self.exact_route or self.nominal_route

@dataclass
class Flight:
    airport: str
    aircraft_type: str
    route: RouteResult
    passengers_by_facility: Dict[str, List[str]]
    source: str
    flight_no: int = 0

class Q1Solver:
    def __init__(self) -> None:
        self.people = pd.read_csv(PEOPLE_FILE, dtype=str)
        raw = pd.read_csv(DIST_FILE, index_col=0)
        raw.index, raw.columns = raw.index.astype(str), raw.columns.astype(str)
        self.dist_df = raw.astype(float)
        self.dist = {(i, j): float(self.dist_df.loc[i, j]) for i in raw.index for j in raw.columns}
        self.demands = self.people.groupby("destination_id").size().astype(int).to_dict()
        self.route_cache = {}
        self.nominal_cache = {}
        self.airport_solutions = {}
        self.flights: List[Flight] = []

    def distance(self, a: str, b: str) -> float:
        return self.dist[(a, b)]

    @staticmethod
    def flight_minutes(distance: float, machine: str) -> int:
        return int(math.ceil(60.0 * distance / MACHINES[machine]["speed"] - 1e-12))

    @staticmethod
    def scalar_cost(route: RouteResult, multiplier: int = 1) -> float:
        return multiplier * (route.minutes + FUEL_EPS * route.fuel)

    def make_result(self, machine: str, nodes: Sequence[str], refs: Sequence[int]) -> RouteResult:
        distance = sum(self.distance(a, b) for a, b in zip(nodes[:-1], nodes[1:]))
        minutes = 0
        for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
            minutes += self.flight_minutes(self.distance(a, b), machine)
            if i + 1 < len(nodes) - 1:
                minutes += 20 if refs[i + 1] else 10
        return RouteResult(machine, tuple(nodes), tuple(map(int, refs)), int(minutes),
                           float(distance), float(distance * MACHINES[machine]["burn"]))

    def nominal_route(self, airport: str, facilities: Iterable[str], machine: str) -> RouteResult:
        required = tuple(sorted(set(facilities)))
        key = (airport, required, machine)
        if key in self.nominal_cache:
            return self.nominal_cache[key]
        best = None
        for perm in itertools.permutations(required):
            nodes = (airport,) + perm + (airport,)
            result = self.make_result(machine, nodes, (0,) * len(nodes))
            score = (result.minutes, result.fuel, result.distance, result.nodes)
            if best is None or score < best[0]:
                best = (score, result)
        self.nominal_cache[key] = best[1]
        return best[1]

    def best_feasible_route(self, airport: str, facilities: Iterable[str],
                            machine: str) -> Optional[RouteResult]:
        required = tuple(sorted(set(facilities)))
        key = (airport, required, machine)
        if key in self.route_cache:
            return self.route_cache[key]
        par = MACHINES[machine]
        max_range = (par["tank"] - par["reserve"]) / par["burn"]
        req_index = {f: i for i, f in enumerate(required)}
        full_mask = (1 << len(required)) - 1
        counter = itertools.count()
        heap = [(0, 0.0, next(counter), airport, 0, 0, 0.0, (airport,), (0,))]
        labels = defaultdict(list)
        labels[(airport, 0, 0)].append((0.0, 0, 0.0))
        best = None
        best_score = None
        while heap:
            minutes, total_dist, _, current, mask, stops, used, nodes, refs = heapq.heappop(heap)
            if best_score is not None and minutes > best_score[0]:
                continue
            if mask == full_mask:
                back = self.distance(current, airport)
                if used + back <= max_range + 1e-9:
                    result = self.make_result(machine, nodes + (airport,), refs + (0,))
                    score = (result.minutes, result.fuel, result.distance, result.nodes)
                    if best_score is None or score < best_score:
                        best_score, best = score, result
            if stops >= MAX_STOPS:
                continue
            unvisited = [f for f in required if not (mask & (1 << req_index[f]))]
            for nxt in sorted(set(unvisited) | REFUEL_FACILITIES):
                if nxt == current:
                    continue
                leg = self.distance(current, nxt)
                if used + leg > max_range + 1e-9:
                    continue
                bit = 1 << req_index[nxt] if nxt in req_index else 0
                is_new_required = bool(bit and not (mask & bit))
                new_mask = mask | bit
                if nxt in REFUEL_FACILITIES and is_new_required:
                    choices = (0, 1)
                elif nxt in REFUEL_FACILITIES:
                    choices = (1,)
                else:
                    choices = (0,)
                for do_refuel in choices:
                    new_used = 0.0 if do_refuel else used + leg
                    new_minutes = minutes + self.flight_minutes(leg, machine) + (20 if do_refuel else 10)
                    new_total = total_dist + leg
                    new_stops = stops + 1
                    state = (nxt, new_mask, new_stops)
                    old = labels[state]
                    if any(u <= new_used + 1e-9 and t <= new_minutes and d <= new_total + 1e-9
                           for u, t, d in old):
                        continue
                    labels[state] = [x for x in old if not (
                        new_used <= x[0] + 1e-9 and new_minutes <= x[1] and new_total <= x[2] + 1e-9)]
                    labels[state].append((new_used, new_minutes, new_total))
                    heapq.heappush(heap, (new_minutes, new_total, next(counter), nxt, new_mask,
                                          new_stops, new_used, nodes + (nxt,), refs + (do_refuel,)))
        self.route_cache[key] = best
        return best

    def best_exact_for_capacity(self, airport: str, facilities: Iterable[str],
                                passengers: int) -> Optional[RouteResult]:
        candidates = []
        for machine, par in MACHINES.items():
            if par["seats"] >= passengers:
                route = self.best_feasible_route(airport, facilities, machine)
                if route is not None:
                    candidates.append(route)
        return min(candidates, key=lambda r: (r.minutes, r.fuel, r.distance,
                    MACHINES[r.aircraft_type]["seats"])) if candidates else None

    def best_nominal_for_capacity(self, airport: str, facilities: Iterable[str],
                                  passengers: int) -> RouteResult:
        candidates = [self.nominal_route(airport, facilities, m) for m, p in MACHINES.items()
                      if p["seats"] >= passengers]
        return min(candidates, key=lambda r: (r.minutes, r.fuel, r.distance,
                   MACHINES[r.aircraft_type]["seats"]))

    def build_options(self, airport: str, facilities: Sequence[str]) -> List[SplitOption]:
        options = []
        for facility in facilities:
            q = int(self.demands[facility])
            for capacity in OPTION_CAPACITIES:
                full_count, remainder = divmod(q, capacity)
                route = self.best_feasible_route(airport, (facility,), CAPACITY_TO_MACHINE[capacity])
                options.append(SplitOption(len(options), facility, capacity, full_count, remainder, route))
        return options

    def enumerate_tail_columns(self, airport: str, facilities: Sequence[str],
                               options: Sequence[SplitOption]) -> List[TailColumn]:
        by_facility = defaultdict(list)
        for op in options:
            if op.remainder > 0 and op.full_route is not None:
                by_facility[op.facility].append(op)
        columns = []
        for k in range(1, MAX_STOPS + 1):
            for facility_group in itertools.combinations(facilities, k):
                groups = [by_facility[f] for f in facility_group]
                if any(not g for g in groups):
                    continue
                for picked in itertools.product(*groups):
                    passengers = sum(op.remainder for op in picked)
                    if passengers <= 19:
                        columns.append(TailColumn(
                            tuple(op.option_id for op in picked), tuple(facility_group), passengers,
                            self.best_nominal_for_capacity(airport, facility_group, passengers)))
        return columns

    def build_milp(self, facilities: Sequence[str], options: Sequence[SplitOption],
                   columns: Sequence[TailColumn]):
        n_y, n_x = len(options), len(columns)
        nonzero = [op for op in options if op.remainder > 0]
        tail_row = {op.option_id: len(facilities) + i for i, op in enumerate(nonzero)}
        n_rows = len(facilities) + len(nonzero)
        facility_row = {f: i for i, f in enumerate(facilities)}
        rr, cc, vv = [], [], []
        lower = np.ones(n_rows)
        upper = np.ones(n_rows)
        lower[len(facilities):] = 0.0
        upper[len(facilities):] = 0.0
        c = np.zeros(n_y + n_x)
        ub = np.ones(n_y + n_x)
        for j, op in enumerate(options):
            rr.append(facility_row[op.facility]); cc.append(j); vv.append(1.0)
            if op.full_route is None:
                ub[j], c[j] = 0.0, 1e9
            else:
                c[j] = self.scalar_cost(op.full_route, op.full_count)
            if op.remainder > 0:
                rr.append(tail_row[op.option_id]); cc.append(j); vv.append(-1.0)
        for j, col in enumerate(columns):
            var = n_y + j
            c[var] = self.scalar_cost(col.active_route)
            if col.disabled:
                ub[var] = 0.0
            for option_id in col.items:
                rr.append(tail_row[option_id]); cc.append(var); vv.append(1.0)
        A = coo_matrix((vv, (rr, cc)), shape=(n_rows, n_y + n_x)).tocsr()
        return c, ub, LinearConstraint(A, lower, upper)

    def solve_airport(self, airport: str, facilities: Sequence[str]) -> dict:
        print(f"\n[{airport}] demand={sum(self.demands[f] for f in facilities)}, facilities={len(facilities)}")
        t0 = time.time()
        options = self.build_options(airport, facilities)
        print(f"[{airport}] options={len(options)}, elapsed={time.time()-t0:.1f}s")
        columns = self.enumerate_tail_columns(airport, facilities, options)
        print(f"[{airport}] tail columns={len(columns)}, elapsed={time.time()-t0:.1f}s")
        result = None
        selected_y, selected_x = [], []
        for iteration in range(1, 50):
            c, ub, constraint = self.build_milp(facilities, options, columns)
            result = milp(c=c, integrality=np.ones_like(c), bounds=Bounds(np.zeros_like(c), ub),
                          constraints=constraint,
                          options={"presolve": True, "time_limit": 300.0, "mip_rel_gap": 0.0})
            if not result.success or result.x is None:
                raise RuntimeError(f"{airport} MILP failed: {result.message}")
            n_y = len(options)
            selected_y = [i for i, v in enumerate(result.x[:n_y]) if v > 0.5]
            selected_x = [i for i, v in enumerate(result.x[n_y:]) if v > 0.5]
            changed = 0
            for idx in selected_x:
                col = columns[idx]
                if col.exact_route is not None or col.disabled:
                    continue
                exact = self.best_exact_for_capacity(airport, col.facilities, col.passengers)
                if exact is None:
                    col.disabled = True
                else:
                    col.exact_route = exact
                changed += 1
            print(f"[{airport}] refine {iteration}: lower-model={result.fun:.3f}, "
                  f"selected={len(selected_x)}, newly-exact={changed}")
            if changed == 0:
                break
        else:
            raise RuntimeError(f"{airport} refinement did not converge")
        c, ub, constraint = self.build_milp(facilities, options, columns)
        lp = milp(c=c, integrality=np.zeros_like(c), bounds=Bounds(np.zeros_like(c), ub),
                  constraints=constraint, options={"presolve": True, "time_limit": 120.0})
        lp_bound = float(lp.fun) if lp.success and lp.fun is not None else float("nan")
        chosen_options = [options[i] for i in selected_y]
        chosen_columns = [columns[i] for i in selected_x]
        true_minutes = sum(op.full_count * op.full_route.minutes for op in chosen_options if op.full_route)
        true_minutes += sum(col.exact_route.minutes for col in chosen_columns if col.exact_route)
        true_fuel = sum(op.full_count * op.full_route.fuel for op in chosen_options if op.full_route)
        true_fuel += sum(col.exact_route.fuel for col in chosen_columns if col.exact_route)
        solution = {"airport": airport, "facilities": list(facilities), "options": options,
                    "columns": columns, "chosen_options": chosen_options,
                    "chosen_columns": chosen_columns, "minutes": int(true_minutes),
                    "fuel": float(true_fuel), "lp_bound": lp_bound,
                    "elapsed": time.time()-t0}
        print(f"[{airport}] solved: {true_minutes} min, {true_fuel:.1f} kg, "
              f"LP={lp_bound:.3f}, elapsed={solution['elapsed']:.1f}s")
        return solution

    def construct_flights_and_assignments(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        queues = {}
        for airport, facilities in BLOCKS.items():
            for facility in facilities:
                rows = self.people[self.people.destination_id == facility].copy()
                rows["origin_priority"] = (rows.origin_id == "LAND").astype(int)
                rows = rows.sort_values(["origin_priority", "person_id"])
                queues[(airport, facility)] = rows.person_id.tolist()
        flights = []
        for airport, solution in self.airport_solutions.items():
            for op in solution["chosen_options"]:
                for _ in range(op.full_count):
                    persons = queues[(airport, op.facility)][:op.capacity]
                    del queues[(airport, op.facility)][:op.capacity]
                    if len(persons) != op.capacity:
                        raise RuntimeError("full-flight allocation mismatch")
                    flights.append(Flight(airport, op.full_route.aircraft_type, op.full_route,
                                          {op.facility: persons}, "full"))
            lookup = {op.option_id: op for op in solution["options"]}
            for col in solution["chosen_columns"]:
                manifest = {}
                for option_id in col.items:
                    op = lookup[option_id]
                    persons = queues[(airport, op.facility)][:op.remainder]
                    del queues[(airport, op.facility)][:op.remainder]
                    if len(persons) != op.remainder:
                        raise RuntimeError("tail-flight allocation mismatch")
                    manifest[op.facility] = persons
                flights.append(Flight(airport, col.exact_route.aircraft_type, col.exact_route,
                                      manifest, "tail"))
        leftovers = {k: len(v) for k, v in queues.items() if v}
        if leftovers:
            raise RuntimeError(f"unassigned passengers: {leftovers}")
        counters = Counter()
        for flight in sorted(flights, key=lambda x: (x.aircraft_type, x.airport, x.route.nodes)):
            counters[flight.aircraft_type] += 1
            flight.flight_no = counters[flight.aircraft_type]
        self.flights = flights
        route_rows, assignment_rows = [], []
        for flight in flights:
            for stop_order, (node, refuel) in enumerate(zip(flight.route.nodes, flight.route.refuels)):
                route_rows.append({"aircraft_type": flight.aircraft_type,
                                   "flight_no": flight.flight_no, "stop_order": stop_order,
                                   "facility_id": node, "refuel": int(refuel)})
            first_stop = {}
            for i, node in enumerate(flight.route.nodes):
                first_stop.setdefault(node, i)
            for facility, persons in flight.passengers_by_facility.items():
                for person_id in persons:
                    assignment_rows.append({"person_id": person_id,
                        "aircraft_type": flight.aircraft_type, "flight_no": flight.flight_no,
                        "pickup_stop_order": 0, "delivery_stop_order": first_stop[facility]})
        routes = pd.DataFrame(route_rows).sort_values(["aircraft_type", "flight_no", "stop_order"])
        assignments = self.people[["person_id"]].merge(pd.DataFrame(assignment_rows),
                                                        on="person_id", how="left")
        return routes, assignments

    def calculate_metrics(self):
        total_minutes = total_person_minutes = 0
        total_fuel = passenger_km = seat_km = 0.0
        flight_rows = []
        facility_stats = defaultdict(lambda: {"passengers": 0, "deliveries": 0, "airport": ""})
        for flight in self.flights:
            route = flight.route
            first_stop = {}
            for i, node in enumerate(route.nodes):
                first_stop.setdefault(node, i)
            delivery_orders = []
            for facility, persons in flight.passengers_by_facility.items():
                delivery_orders.extend([first_stop[facility]] * len(persons))
                facility_stats[facility]["passengers"] += len(persons)
                facility_stats[facility]["deliveries"] += 1
                facility_stats[facility]["airport"] = flight.airport
            elapsed = 0
            arrivals = {0: 0}
            route_pk, route_sk = 0.0, 0.0
            for i, (a, b) in enumerate(zip(route.nodes[:-1], route.nodes[1:])):
                d = self.distance(a, b)
                onboard = sum(order > i for order in delivery_orders)
                passenger_km += onboard * d
                seat_km += MACHINES[flight.aircraft_type]["seats"] * d
                route_pk += onboard * d
                route_sk += MACHINES[flight.aircraft_type]["seats"] * d
                elapsed += self.flight_minutes(d, flight.aircraft_type)
                arrivals[i + 1] = elapsed
                if i + 1 < len(route.nodes) - 1:
                    elapsed += 20 if route.refuels[i + 1] else 10
            total_person_minutes += sum(arrivals[o] for o in delivery_orders)
            total_minutes += elapsed
            total_fuel += route.fuel
            flight_rows.append({"airport": flight.airport, "aircraft_type": flight.aircraft_type,
                "flight_no": flight.flight_no, "source": flight.source,
                "passengers": len(delivery_orders),
                "required_facilities": len(flight.passengers_by_facility),
                "offshore_landings": len(route.nodes)-2, "refuel_count": sum(route.refuels),
                "distance_km": route.distance, "flight_time_min": elapsed,
                "fuel_kg": route.fuel, "seat_utilization": route_pk/route_sk,
                "route": " -> ".join(route.nodes)})
        metrics = {"total_aircraft_time_min": int(total_minutes),
                   "total_aircraft_time_hour": total_minutes/60,
                   "total_person_time_min": int(total_person_minutes),
                   "total_person_time_hour": total_person_minutes/60,
                   "total_flights": len(self.flights), "total_fuel_kg": total_fuel,
                   "seat_utilization": passenger_km/seat_km,
                   "passenger_km": passenger_km, "available_seat_km": seat_km,
                   "total_refuels": int(sum(sum(f.route.refuels) for f in self.flights))}
        flight_df = pd.DataFrame(flight_rows).sort_values(["aircraft_type", "flight_no"])
        facility_df = pd.DataFrame([{"facility_id": f, "airport": facility_stats[f]["airport"],
            "passengers": facility_stats[f]["passengers"],
            "delivery_flights": facility_stats[f]["deliveries"]} for f in sorted(self.demands)])
        return metrics, flight_df, facility_df

    def validate(self, routes: pd.DataFrame, assignments: pd.DataFrame) -> None:
        assert len(assignments) == len(self.people) == 1600
        assert assignments.isna().sum().sum() == 0 and assignments.person_id.is_unique
        groups = routes.groupby(["aircraft_type", "flight_no"], sort=False)
        for (machine, number), group in groups:
            group = group.sort_values("stop_order")
            nodes, refs = group.facility_id.tolist(), group.refuel.astype(int).tolist()
            assert nodes[0] == nodes[-1] and nodes[0] in BLOCKS and len(nodes)-2 <= MAX_STOPS
            assert refs[0] == refs[-1] == 0
            assert all(r == 0 or n in REFUEL_FACILITIES for n, r in zip(nodes, refs))
            par = MACHINES[machine]
            max_range = (par["tank"]-par["reserve"])/par["burn"]
            used = 0.0
            for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
                used += self.distance(a, b)
                assert used <= max_range + 1e-7, (machine, number, nodes, used, max_range)
                if refs[i+1]: used = 0.0
        route_lookup = {key: g.sort_values("stop_order").facility_id.tolist() for key, g in groups}
        occupancy = Counter()
        for row in assignments.merge(self.people, on="person_id").itertuples(index=False):
            key = (row.aircraft_type, int(row.flight_no))
            nodes = route_lookup[key]
            pickup, delivery = int(row.pickup_stop_order), int(row.delivery_stop_order)
            assert pickup == 0 < delivery
            assert row.origin_id == "LAND" or row.origin_id == nodes[0]
            assert nodes[delivery] == row.destination_id
            assert nodes.index(row.destination_id, pickup+1) == delivery
            for seg in range(pickup, delivery): occupancy[(key, seg)] += 1
        for ((machine, number), seg), count in occupancy.items():
            assert count <= MACHINES[machine]["seats"], (machine, number, seg, count)
        for machine, group in routes.groupby("aircraft_type"):
            nums = sorted(group.flight_no.unique())
            assert nums == list(range(1, len(nums)+1))

    def classical_mds(self) -> pd.DataFrame:
        nodes = list(self.dist_df.index)
        D = self.dist_df.loc[nodes, nodes].to_numpy(float)
        n = len(nodes)
        J = np.eye(n) - np.ones((n, n))/n
        B = -0.5 * J @ (D**2) @ J
        values, vectors = np.linalg.eigh(B)
        idx = np.argsort(values)[::-1][:2]
        coords = vectors[:, idx] * np.sqrt(np.maximum(values[idx], 0))
        return pd.DataFrame(coords, index=nodes, columns=["x", "y"])

    def make_plots(self, metrics, flight_df, facility_df) -> None:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        colors = {"A01": "#1f77b4", "A02": "#ff7f0e", "A03": "#2ca02c"}
        mcolors = {"T1": "#4C78A8", "T2": "#F58518", "T3": "#54A24B"}
        fig, ax = plt.subplots(figsize=(15, 6))
        x = np.arange(len(facility_df))
        ax.bar(x, facility_df.passengers, color=[colors[a] for a in facility_df.airport])
        ax.set_xticks(x); ax.set_xticklabels(facility_df.facility_id, rotation=75, fontsize=8)
        ax.set_ylabel("需求人数（人）"); ax.set_xlabel("海上设施")
        ax.set_title("问题一：设施需求分布与机场服务分区")
        legend_handles = [Patch(facecolor=colors[a], label=f"{a}机场") for a in ("A01", "A02", "A03")]
        ax.legend(handles=legend_handles, ncol=3); fig.tight_layout()
        fig.savefig(CODE_DIR/"问题一_设施需求分布图.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        coords = self.classical_mds()
        fig, ax = plt.subplots(figsize=(12, 9))
        fs = [n for n in coords.index if n.startswith("F")]
        ax.scatter(coords.loc[fs, "x"], coords.loc[fs, "y"], s=22, c="#BBBBBB",
                   alpha=.85, label="海上设施", zorder=3)
        for airport in BLOCKS:
            ax.scatter(coords.loc[airport, "x"], coords.loc[airport, "y"], s=180, marker="*",
                       color=colors[airport], edgecolor="black", zorder=5)
            ax.text(coords.loc[airport, "x"], coords.loc[airport, "y"], f"{airport}机场",
                    fontsize=10, weight="bold", ha="left", va="bottom")
        for f in fs:
            ax.text(coords.loc[f, "x"], coords.loc[f, "y"], f[1:], fontsize=5.5,
                    color="#555555", ha="center", va="bottom")
        route_counts = Counter((fl.airport, fl.aircraft_type, fl.route.nodes) for fl in self.flights)
        for (airport, machine, nodes), count in route_counts.items():
            pts = coords.loc[list(nodes)]
            ax.plot(pts.x, pts.y, color=mcolors[machine], alpha=.18,
                    linewidth=.35+.18*math.log1p(count), zorder=1)
        for machine, color in mcolors.items(): ax.plot([], [], color=color, lw=2, label=machine)
        ax.set_title("问题一：航线网络（距离矩阵的经典MDS二维投影）")
        ax.set_xlabel("MDS第一维"); ax.set_ylabel("MDS第二维")
        ax.legend(ncol=4); ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout(); fig.savefig(CODE_DIR/"问题一_航线网络图.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        ms = flight_df.groupby("aircraft_type").agg(flights=("flight_no", "count"),
             passengers=("passengers", "sum"), fuel_kg=("fuel_kg", "sum"),
             time_min=("flight_time_min", "sum")).reindex(["T1", "T2", "T3"]).fillna(0)
        axes[0,0].bar(ms.index, ms.flights, color=[mcolors[x] for x in ms.index])
        axes[0,0].set_title("各机型执行架次数"); axes[0,0].set_ylabel("架次数")
        for i, value in enumerate(ms.flights): axes[0,0].text(i, value, str(int(value)), ha="center", va="bottom")
        axes[0,1].hist(flight_df.seat_utilization*100, bins=np.arange(0,105,5),
                       color="#72B7B2", edgecolor="white")
        axes[0,1].axvline(metrics["seat_utilization"]*100, color="#D62728", ls="--",
                          label=f"网络整体={metrics['seat_utilization']:.2%}")
        axes[0,1].set_title("单架次座位利用率分布")
        axes[0,1].set_xlabel("座位利用率（%）"); axes[0,1].set_ylabel("架次数"); axes[0,1].legend()
        ats = flight_df.groupby("airport").agg(time_min=("flight_time_min","sum"),
                                               fuel_kg=("fuel_kg","sum")).reindex(["A01","A02","A03"])
        ax1, ax2 = axes[1,0], axes[1,0].twinx(); xx=np.arange(3)
        ax1.bar(xx-.18, ats.time_min/60, width=.36, color="#4C78A8", label="飞机使用时间")
        ax2.bar(xx+.18, ats.fuel_kg/1000, width=.36, color="#E45756", label="燃油消耗量")
        ax1.set_xticks(xx); ax1.set_xticklabels(ats.index)
        ax1.set_ylabel("飞机使用时间（小时）"); ax2.set_ylabel("燃油消耗量（吨）")
        ax1.set_title("各机场工作量")
        h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc="upper left")
        axes[1,1].scatter(flight_df.distance_km, flight_df.flight_time_min,
                          c=[mcolors[m] for m in flight_df.aircraft_type],
                          s=24+4*flight_df.passengers, alpha=.65)
        axes[1,1].set_xlabel("航线距离（千米）"); axes[1,1].set_ylabel("飞机使用时间（分钟）")
        axes[1,1].set_title("航线距离与飞机使用时间")
        for machine, color in mcolors.items(): axes[1,1].scatter([], [], color=color, label=machine)
        axes[1,1].legend()
        fig.suptitle(f"问题一求解结果：{metrics['total_flights']}架次｜"
                     f"飞机使用{metrics['total_aircraft_time_hour']:.1f}小时｜"
                     f"燃油{metrics['total_fuel_kg']/1000:.1f}吨", fontsize=14)
        fig.tight_layout(rect=(0,0,1,.96))
        fig.savefig(CODE_DIR/"问题一_求解结果综合图.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    def run(self):
        total_start = time.time()
        for airport, facilities in BLOCKS.items():
            self.airport_solutions[airport] = self.solve_airport(airport, facilities)
        routes, assignments = self.construct_flights_and_assignments()
        metrics, flight_df, facility_df = self.calculate_metrics()
        self.validate(routes, assignments)
        routes.to_csv(CODE_DIR/"q1-routes.csv", index=False, encoding="utf-8-sig")
        assignments.to_csv(CODE_DIR/"q1-assignments.csv", index=False, encoding="utf-8-sig")
        flight_df.to_csv(CODE_DIR/"q1_flight_summary.csv", index=False, encoding="utf-8-sig")
        facility_df.to_csv(CODE_DIR/"q1_facility_summary.csv", index=False, encoding="utf-8-sig")
        lp_bound = sum(s["lp_bound"] for s in self.airport_solutions.values())
        metrics["lp_lower_bound_min"] = lp_bound
        metrics["relative_gap"] = ((metrics["total_aircraft_time_min"]-lp_bound)/lp_bound
                                   if lp_bound > 0 else float("nan"))
        metrics["runtime_sec"] = time.time()-total_start
        pd.DataFrame([metrics]).to_csv(CODE_DIR/"q1_metrics.csv", index=False, encoding="utf-8-sig")
        self.make_plots(metrics, flight_df, facility_df)
        print("\n=== Q1 final metrics ===")
        for key, value in metrics.items(): print(f"{key}: {value}")
        print(f"Outputs written to: {CODE_DIR}")
        return metrics

if __name__ == "__main__":
    Q1Solver().run()
