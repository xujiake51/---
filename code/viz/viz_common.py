# -*- coding: utf-8 -*-
"""可视化公共模块：数据路径、字体、配色、MDS 投影、机场分块。

所有绘图脚本均从「赛题原始 CSV + 求解器已输出的结果 CSV」读取数据，
不依赖、不修改任何求解器代码。
"""
import os

import numpy as np
import pandas as pd

VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(VIZ_DIR)
ROOT = os.path.dirname(CODE_DIR)
OUTPUT_DIR = os.path.join(ROOT, "output")
FIGURES_DIR = os.path.join(ROOT, "paper", "figures")


def find_attach_dir():
    """定位赛题附件目录（目录名含全角引号，按后缀「附件」匹配以免硬编码）。"""
    for name in os.listdir(ROOT):
        if name.endswith("附件"):
            return os.path.join(ROOT, name)
    raise FileNotFoundError("未找到「附件」目录")


ATTACH_DIR = find_attach_dir()

# ---------------- 统一配色 ----------------
# 机场三色（蓝 / 橙 / 绿），沿用原有图并统一
AIRPORT_COLORS = {"A01": "#1976d2", "A02": "#ef6c00", "A03": "#2e7d32"}
# 机型三色
AIRCRAFT_COLORS = {"T1": "#4C78A8", "T2": "#F58518", "T3": "#54A24B"}
# 顺序蓝（数量级：人数/架次/分钟），0 值用白
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
# 发散色（净差：负蓝、正红、0 白），取自 RdBu
DIVERGING = ["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7", "#ef8a62", "#b2182b"]

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#d8d8d3"

# 设施编号与机场分块（沿海岸线自北向南等分三段）
BLOCKS = {
    "A01": [f"F{i:03d}" for i in range(1, 18)],    # F001--F017
    "A02": [f"F{i:03d}" for i in range(18, 36)],   # F018--F035
    "A03": [f"F{i:03d}" for i in range(36, 53)],   # F036--F052
}
FACILITIES = [f"F{i:03d}" for i in range(1, 53)]


def setup_font():
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def load_distance_df():
    return pd.read_csv(os.path.join(ATTACH_DIR, "distances.csv"), index_col=0)


def classical_mds(dist_df):
    """对距离矩阵做经典多维尺度（MDS）二维投影，保留节点间相对航程。"""
    nodes = list(dist_df.index)
    D = dist_df.loc[nodes, nodes].to_numpy(float)
    n = len(nodes)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    values, vectors = np.linalg.eigh(B)
    idx = np.argsort(values)[::-1][:2]
    coords = vectors[:, idx] * np.sqrt(np.maximum(values[idx], 0))
    return pd.DataFrame(coords, index=nodes, columns=["x", "y"])


def facility_block(fid):
    for airport, fs in BLOCKS.items():
        if fid in fs:
            return airport
    return None
