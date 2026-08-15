# -*- coding: utf-8 -*-
"""Q2 完整列探针（修正版）：成对尾数 + 穿梭尾数，枚举 (组合 × 机型 × 顺序 × 穿梭分配)。

修复上一版"超集重复计数"的 bug 后，成对尾数列数仅 ~2-4 万/区块。本脚本把穿梭尾数
也纳入，得到 B-软 真正需要枚举的"完整列"数量级。
"""
import csv
import random
from collections import Counter, defaultdict
from itertools import permutations

random.seed(0)

rows = list(csv.DictReader(open(
    "../2026年度“策联杯”数学建模精英联赛-B题-附件/peopleQ2.csv", encoding="utf-8-sig")))
q_out, q_ret, q_sh = Counter(), Counter(), Counter()
for r in rows:
    o, d = r["origin_id"], r["destination_id"]
    if o in ("A01", "A02", "A03", "LAND") and d.startswith("F"):
        q_out[d] += 1
    elif o.startswith("F") and d in ("A01", "A02", "A03", "LAND"):
        q_ret[o] += 1
    else:
        q_sh[(o, d)] += 1


def blk(f):
    n = int(f[1:])
    return 0 if n <= 17 else (1 if n <= 35 else 2)


fac_by_block = defaultdict(list)
for f in q_out:
    fac_by_block[blk(f)].append(f)


def coupled_tails(f):
    s = set()
    for c in (19, 16, 12):
        d, p = q_out[f] % c, q_ret[f] % c
        if d > 0 or p > 0:
            s.add((d, p))
    return sorted(s)


CT = {f: coupled_tails(f) for f in q_out}

# 穿梭尾数，按 OD 对
SH = {}
for od in q_sh:
    i, j = od
    if blk(i) == blk(j):
        SH[od] = sorted({q_sh[od] % c for c in (19, 16, 12) if q_sh[od] % c > 0})

CAPS = [(12, "T1"), (16, "T2"), (19, "T3")]


# 给定按访问顺序排列的成对尾数 + 穿梭 (a,b,h)，求峰值载荷
def peak(in_order, shuttles):
    k = len(in_order)
    D = [t[0] for t in in_order]
    P = [t[1] for t in in_order]
    total_d = sum(D)
    mx = total_d
    cumD = cumP = 0
    for t in range(k):
        cumD += D[t]
        cumP += P[t]
        L = (total_d - cumD) + cumP
        for (a, b, h) in shuttles:
            if a <= t < b:
                L += h
        mx = max(mx, L)
    mx = max(mx, sum(P) + sum(h for (a, b, h) in shuttles if a <= k - 1 < b))
    return mx


def full_count(combo_fac, combo_tails):
    """给定一个成对尾数组合（k 个设施），统计完整列数（含顺序与穿梭）。"""
    k = len(combo_fac)
    # 该组合内可能承载的穿梭 OD（两端都在组合内），按 (i,j) 方向
    ods = []
    for i in range(k):
        for j in range(k):
            if i != j and (combo_fac[i], combo_fac[j]) in SH:
                ods.append((i, j, SH[(combo_fac[i], combo_fac[j])]))  # (起, 终, 尾数列表)
    total = 0
    for perm in permutations(range(k)):
        order = [combo_tails[i] for i in perm]
        # 位置映射：原设施序号 -> 访问次序
        pos = [0] * k
        for t, i in enumerate(perm):
            pos[i] = t
        # 按顺序可承载的穿梭（起在终前）
        cand = []
        for (i, j, tails) in ods:
            if pos[i] < pos[j]:
                cand.append((pos[i], pos[j], tails))
        for cap, name in CAPS:
            if peak(order, []) > cap:
                continue
            # 枚举穿梭分配
            def sh_dfs(idx, cur):
                nonlocal total
                if idx == len(cand):
                    total += 1
                    return
                a, b, tails = cand[idx]
                sh_dfs(idx + 1, cur)               # 不取该穿梭
                for h in tails:
                    if peak(order, cur + [(a, b, h)]) <= cap:
                        sh_dfs(idx + 1, cur + [(a, b, h)])
            sh_dfs(0, [])
    return total


print("=" * 74)
print("Q2 完整列探针（成对尾数 + 穿梭，枚举组合×机型×顺序×穿梭）")
print("=" * 74)

for b in range(3):
    facs = sorted(fac_by_block[b])
    items = []
    for f in facs:
        for (d, p) in CT[f]:
            items.append((f, d, p))

    n_combo = 0
    n_bal = 0
    n_full = 0

    def dfs(i, chosen_f, chosen_t):
        global n_combo, n_bal, n_full
        if i == len(items):
            if chosen_f:
                n_combo += 1
                peak_ = min(peak([chosen_t[x] for x in perm], [])
                            for perm in permutations(range(len(chosen_f))))
                for cap, name in CAPS:
                    if peak_ <= cap:
                        n_bal += 1
                n_full += full_count(chosen_f, chosen_t)
            return
        f, d, p = items[i]
        dfs(i + 1, chosen_f, chosen_t)
        if (f not in chosen_f and len(chosen_f) < 5
                and sum(x[1] for x in chosen_t) + d <= 19):
            dfs(i + 1, chosen_f + [f], chosen_t + [(d, p)])

    dfs(0, [], [])
    print(f"\n[Block {b+1}] {len(facs)} 设施")
    print(f"  成对尾数组合数           : {n_combo:>12,}")
    print(f"  平衡列（组合×机型）      : {n_bal:>12,}")
    print(f"  完整列（+穿梭×顺序×机型）: {n_full:>12,}")
