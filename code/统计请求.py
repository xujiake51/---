# -*- coding: utf-8 -*-
"""统计 peopleQ1.csv 中出发地 × 目的地的 OD 矩阵，输出到 CSV。"""
"""由deepseek-V4生成"""
import csv
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "2026年度“策联杯”数学建模精英联赛-B题-附件", "peopleQ1.csv")
OUTPUT = os.path.join(os.path.dirname(HERE), "output", "请求统计_OD矩阵.csv")

with open(DATA, encoding="utf-8-sig", newline="") as f:
    rows = [(r["origin_id"].strip(), r["destination_id"].strip()) for r in csv.DictReader(f)]

origins = sorted({o for o, _ in rows})
dests = sorted({d for _, d in rows})
od = Counter(rows)

header = ["出发地\\目的地"] + dests + ["合计"]
matrix = []
for o in origins:
    matrix.append([o] + [od[(o, d)] for d in dests] + [sum(od[(o, d)] for d in dests)])
matrix.append(["合计"] + [sum(od[(o, d)] for o in origins) for d in dests] + [len(rows)])

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8-sig", newline="") as f:
    csv.writer(f).writerows([header] + matrix)

print("OD 矩阵已导出：请求统计_OD矩阵.csv")
