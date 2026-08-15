# -*- coding: utf-8 -*-
"""Q2 探针（修正版）：复现算法 1 的 DFS——直接枚举尾数份，而非"设施子集×取/跳过"。
修复上一版"超集重复计数"的 bug。

Q2 的尾数份是成对 (d,p)=(出海尾数,海返尾数)，由同一机型 c 的拆分产生，绑在一起。
一条列 = 尾数组合 + 机型 + 访问顺序（顺序影响载荷剖面，须枚举/求最优）。
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

# 每设施成对尾数 (d,p)，由机型 c∈{19,16,12} 产生，去重
def coupled_tails(f):
    s = set()
    for c in (19, 16, 12):
        d, p = q_out[f] % c, q_ret[f] % c
        if d > 0 or p > 0:
            s.add((d, p))
    return sorted(s)

CT = {f: coupled_tails(f) for f in q_out}

# 给定一组成对尾数 (d_i,p_i)，求"最小可达峰值载荷"（遍历 k! 种顺序）
def min_peak(tails):
    k = len(tails)
    if k == 0:
        return 0
    total_d = sum(t[0] for t in tails)
    best = float("inf")
    for perm in permutations(range(k)):
        order = [tails[i] for i in perm]
        D = [t[0] for t in order]
        P = [t[1] for t in order]
        cumD = cumP = 0
        mx = total_d                      # 机场→首站
        for t in range(k):
            cumD += D[t]
            cumP += P[t]
            L = (total_d - cumD) + cumP   # 第 t→t+1 弧
            mx = max(mx, L)
        mx = max(mx, sum(P))              # 末站→机场
        best = min(best, mx)
    return best

CAPS = [(12, "T1"), (16, "T2"), (19, "T3")]

print("=" * 74)
print("Q2 修正版探针（成对尾数 + 机型 + 顺序，容量按最小峰值载荷）")
print("=" * 74)

for b in range(3):
    facs = sorted(fac_by_block[b])
    items = []
    for f in facs:
        for (d, p) in CT[f]:
            items.append((f, d, p))

    n_combo = 0        # 尾数组合数（非空）
    n_col = 0          # (组合 × 机型) 容量可行的列数
    col_by_type = Counter()

    def dfs(i, chosen):
        global n_combo, n_col
        if i == len(items):
            if chosen:
                n_combo += 1
                peak = min_peak(chosen)
                for cap, name in CAPS:
                    if peak <= cap:
                        n_col += 1
                        col_by_type[name] += 1
            return
        f, d, p = items[i]
        dfs(i + 1, chosen)                          # 不取
        if (all(f != x[0] for x in chosen) and len(chosen) < 5
                and sum(x[1] for x in chosen) + d <= 19):
            dfs(i + 1, chosen + [(d, p)])           # 取

    dfs(0, [])
    print(f"\n[Block {b+1}] {len(facs)} 设施, |P|={len(items)}")
    print(f"  成对尾数组合数（非空,≤5,≤1份/设施）: {n_combo:>12,}")
    print(f"  列数（组合×机型, 容量可行）        : {n_col:>12,}")
    print(f"    ├ T1: {col_by_type['T1']:,}  T2: {col_by_type['T2']:,}  T3: {col_by_type['T3']:,}")
