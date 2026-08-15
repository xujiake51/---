# -*- coding: utf-8 -*-
"""Q1 探针：精确复现《第一问建模》§7.3 算法 1（尾数合并架次枚举），
统计真实的组合数与列数，判断"显式枚举 + 一次性集合划分 IP"是否可行。
"""
import csv
import time
from collections import Counter, defaultdict

# ---------- 读数据：出海需求逐设施统计 ----------
rows = list(csv.DictReader(open(
    "../2026年度“策联杯”数学建模精英联赛-B题-附件/peopleQ1.csv", encoding="utf-8-sig")))
q = Counter()
for r in rows:
    d = r["destination_id"]
    if d.startswith("F"):
        q[d] += 1


def blk(f):
    n = int(f[1:])
    return 0 if n <= 17 else (1 if n <= 35 else 2)


fac_by_block = defaultdict(list)
for f in q:
    fac_by_block[blk(f)].append(f)

print("=" * 74)
print("Q1 尾数合并架次枚举探针（精确复现算法 1）")
print("=" * 74)

for b in range(3):
    facs = sorted(fac_by_block[b])

    # 尾数份集合 P：每设施每机型 c∈{19,16,12}，p=q%c，p>0，去重
    P = []
    for f in facs:
        seen = set()
        for c in (19, 16, 12):
            p = q[f] % c
            if p > 0 and p not in seen:
                seen.add(p)
                P.append((f, p))

    # 精确 DFS（算法 1）：逐份决策取/不取，≤1 份/设施，≤5 份，Σp≤19
    n_combo = 0          # 组合数（非空）
    n_col = 0            # 列数（组合 × 机型 c_m ≥ Σp）
    dist = Counter()     # Σp 分段：<=12 / (12,16] / (16,19]
    t0 = time.time()

    def dfs(i, chosen, total):
        global n_combo, n_col
        if i == len(P):
            if chosen:                       # 非空组合
                n_combo += 1
                if total <= 12:
                    n_col += 3; dist["<=12"] += 1
                elif total <= 16:
                    n_col += 2; dist["(12,16]"] += 1
                else:
                    n_col += 1; dist["(16,19]"] += 1
            return
        d, p = P[i]
        # 不取
        dfs(i + 1, chosen, total)
        # 取：不同设施 且 <5 且 容量 ≤19
        if all(d != x for x in chosen) and len(chosen) < 5 and total + p <= 19:
            dfs(i + 1, chosen + [d], total + p)

    dfs(0, [], 0)
    dt = time.time() - t0

    print(f"\n[Block {b+1}] {len(facs)} 设施, |P|={len(P)}, 枚举耗时 {dt:.2f}s")
    print(f"  组合数（非空, ≤5 份, Σp≤19, ≤1 份/设施）: {n_combo:>12,}")
    print(f"  列数（组合 × 可行机型）               : {n_col:>12,}")
    print(f"  Σp 分段: <=12 有 {dist['<=12']:,} | (12,16] 有 {dist['(12,16]']:,} | (16,19] 有 {dist['(16,19]']:,}")
