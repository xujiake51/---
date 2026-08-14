# -*- coding: utf-8 -*-
"""统计 peopleQ1.csv / peopleQ2.csv 的出发地 × 目的地 OD 矩阵，分别输出到 CSV。"""
import csv
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "2026年度“策联杯”数学建模精英联赛-B题-附件")
OUT_DIR = os.path.join(os.path.dirname(HERE), "output")

# 输入数据文件名 → 输出 CSV 文件名
JOBS = [
    ("peopleQ1.csv", "question1_OD矩阵.csv"),
    ("peopleQ2.csv", "question2_OD矩阵.csv"),
]


def build_od_matrix(data_file, out_file):
    with open(os.path.join(DATA_DIR, data_file), encoding="utf-8-sig", newline="") as f:
        rows = [(r["origin_id"].strip(), r["destination_id"].strip()) for r in csv.DictReader(f)]

    origins = sorted({o for o, _ in rows})
    dests = sorted({d for _, d in rows})
    od = Counter(rows)

    header = ["出发地\\目的地"] + dests + ["合计"]
    matrix = []
    for o in origins:
        matrix.append([o] + [od[(o, d)] for d in dests] + [sum(od[(o, d)] for d in dests)])
    matrix.append(["合计"] + [sum(od[(o, d)] for o in origins) for d in dests] + [len(rows)])

    out_path = os.path.join(OUT_DIR, out_file)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows([header] + matrix)

    print(f"{data_file} → {out_file}：{len(origins)} 出发地 × {len(dests)} 目的地，共 {len(rows)} 条")
    return out_path


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for data_file, out_file in JOBS:
        build_od_matrix(data_file, out_file)
