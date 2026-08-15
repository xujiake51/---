# -*- coding: utf-8 -*-
"""列数探针：估计 Q2 各单机场子问题在 B-软（显式枚举 + 一次性集合划分）下的列数量级。

关键模型（与 Q1 的"拆分 + 尾数"一致，但出海/海返须成对）：
  - 每个设施 f 的"尾数份"是 (d,p) 成对出现的——d = 出海尾数、p = 海返尾数，
    由同一机型 c 的拆分 q_out%c / q_ret%c 产生，二者绑在一起，不能拆开。
  - 一条尾数合并环游 = 有向访问顺序 S + 每个设施取一对 (d,p) 尾数 + 若干穿梭尾数，
    受逐弧载荷 ≤ 座位数的约束。
"""
import csv
import random
from itertools import combinations
from collections import Counter, defaultdict
from math import perm

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

# 每设施的成对尾数 (d,p)，由机型 c∈{12,16,19} 的拆分产生，去重
def scheme_tails(f):
    s = set()
    for c in (12, 16, 19):
        d, p = q_out[f] % c, q_ret[f] % c
        if d > 0 or p > 0:
            s.add((d, p))
    return sorted(s)

ST = {f: scheme_tails(f) for f in q_out}
# 穿梭尾数（按 OD 对），由机型拆分产生，去重
def sh_tails(od):
    return sorted({q_sh[od] % c for c in (12, 16, 19) if q_sh[od] % c > 0})

SH = {od: sh_tails(od) for od in q_sh}


def count_scheme_assign(S, cap=19):
    """无序：每个设施选一对 (d,p)，要求 Σd ≤ cap（出海容量，主导项）。"""
    tot = 0
    def dfs(i, s):
        nonlocal tot
        if i == len(S):
            tot += 1
            return
        dfs(i + 1, s)                      # 不取该设施尾数
        for (d, p) in ST[S[i]]:
            if s + d <= cap:
                dfs(i + 1, s + d)
    dfs(0, 0)
    return tot


def count_ordered_assign(S, cap=19):
    """有向环游 S 的完整列计数：成对尾数 + 穿梭尾数，逐弧载荷 ≤ cap。"""
    k = len(S)
    st = [ST[f] for f in S]
    sh_od = {}
    for i in range(k):
        for j in range(i + 1, k):
            od = (S[i], S[j])
            if od in SH:
                sh_od[(i, j)] = SH[od]
    tot = 0

    def dfs(t, load):
        nonlocal tot
        if t == k:
            tot += 1
            return
        for (d, p) in [(0, 0)] + st[t]:
            nl = load[:]
            for j in range(t):
                nl[j] += d        # 出海到 s_t 占用弧 0..t-1
            for j in range(t, k + 1):
                nl[j] += p        # 海返自 s_t 占用弧 t..k
            if max(nl) > cap:
                continue
            # 以 s_t 为终点的穿梭 (s_i -> s_t)，i < t
            ends = [(i, sh_od[(i, t)]) for i in range(t) if (i, t) in sh_od]
            def sh_dfs(idx, cur):
                if idx == len(ends):
                    dfs(t + 1, cur)
                    return
                i, opts = ends[idx]
                for h in [0] + opts:
                    cl = cur[:]
                    for j in range(i, t):
                        cl[j] += h
                    if max(cl) <= cap:
                        sh_dfs(idx + 1, cl)
            sh_dfs(0, nl)
    dfs(0, [0] * (k + 1))
    return tot


print("=" * 70)
print("各区块列数量级（B-软 显式枚举）")
print("=" * 70)
for b in range(3):
    facs = fac_by_block[b]
    n = len(facs)
    n_tours = sum(perm(n, k) for k in range(1, 6))

    # 1) 成对尾数环游（无序近似，Σd≤19）
    n_bal = 0
    for k in range(1, 6):
        for S in combinations(facs, k):
            n_bal += count_scheme_assign(S)

    # 2) 穿梭端点
    eps = set()
    for (i, j) in q_sh:
        if blk(i) == b:
            eps.add(i)
            eps.add(j)
    eps = sorted(eps)
    n_sh = sum(perm(len(eps), k) for k in range(1, 6))

    print(f"\n[Block {b+1}] {n} 设施, 穿梭端点 {len(eps)} 个")
    print(f"  有向环游(≤5停靠)            : {n_tours:>12,}")
    print(f"  成对尾数环游(Σd≤19 近似)    : {n_bal:>12,}")
    print(f"  穿梭端点的有向环游上限       : {n_sh:>12,}")

    # 3) 完整列（成对尾数 + 穿梭）抽样估计
    sample = 1500
    avg = 0.0
    # 按停靠数分层抽样，避免只抽到短环游
    for k in range(1, 6):
        cnt = min(sample // 5, perm(n, k))
        seen = set()
        got = 0
        while got < cnt:
            S = tuple(random.sample(facs, k))
            if S in seen:
                continue
            seen.add(S)
            avg += count_ordered_assign(list(S))
            got += 1
    avg /= (sample // 5) * 5
    print(f"  完整列(成对尾数+穿梭) 平均/环游: {avg:>12.2f}")
    print(f"  完整列 总量估计               : {int(avg * n_tours):>12,}")
