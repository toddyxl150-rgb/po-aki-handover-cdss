#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_feature_engineering.py — 三层特征工程引擎（630 维）
=====================================================

输入：模块 2 产出的窗口张量 X，形状 (N, 10, 12)
      —— 已按训练集参数归一化，无需在此再做任何缩放。

输出：630 维可解释特征，分三层

    第 1 层  原始归一化序列        10 × 12            = 120 维   {Var}_t1 ... {Var}_t12
    第 2 层  多分辨率子窗统计      10 × 6 × 8         = 480 维   {Var}_{Win}_{Stat}
    第 3 层  基尼波动与有向耦合    10 × 3             =  30 维   {Var}_1-12h_{GI|GV|DGI}
                                                      ────────
                                                        630 维

子窗口（按任务书，1-based，t1 = 窗口最早 1 小时）
    4h 分辨率：1-4h, 5-8h, 9-12h
    6h 分辨率：1-6h, 7-12h
    12h 分辨率：1-12h
统计量（8 个）：Mean, Median, Min, Max, Std, Skew, IQR, Slope

基尼三层定义
------------
    G   = Σ_{i<j} |x_i - x_j|                 （12 步内所有点对绝对差之和）
    GI  = 2G / (12 × 11)                      归一化基尼波动率
    GV  = 1 - GI                              原文稳态形式
    DGI = GI × sign(Slope_{1-12h})            有向基尼（波动 + 方向）

为什么 GI 用常数分母 12×11 而不是按均值归一
------------------------------------------
因为输入已归一化到 [0,1]，此时 GI = G/66 恰为「平均成对绝对差」，天然落在 [0,1]；
若按均值归一（RMAD 形式），分母会随均值漂移，跨库可比性反而变差。
**注意**：若你把模块 2 的 clip_scaled 关掉，测试集超出训练范围的值会让 GI > 1，
这是预期行为，不是 bug——此时 GI 表达的是"相对训练分布的越界程度"。

为什么所有统计量都在归一化后的值上算
----------------------------------
跨洲际验证时，三个库的原始单位/量程不同（如 SpO2 百分比 vs 小数）。
在训练集归一化后的尺度上统计，才能让"斜率=0.05/h"在三库间含义一致。
若要做单位敏感的分析，把 `on_scaled=False` 传进来即可（需自行传入原始张量）。

用法
----
    python 03_feature_engineering.py --self-test
    python 03_feature_engineering.py --demo --out outputs
    python 03_feature_engineering.py --npz outputs/windows_track1_X.npy \
        --meta outputs/windows_track1_meta.csv --out outputs
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdss_common import CHANNELS, ensure_dir, save_json  # noqa: E402

LOG = logging.getLogger("cdss.features")

# (名称, 起始步, 结束步) —— 1-based 闭区间，t1 = 窗口最早 1 小时
SUBWINDOWS: Tuple[Tuple[str, int, int], ...] = (
    ("1-4h", 0, 4),
    ("5-8h", 4, 8),
    ("9-12h", 8, 12),
    ("1-6h", 0, 6),
    ("7-12h", 6, 12),
    ("1-12h", 0, 12),
)
STATS: Tuple[str, ...] = ("Mean", "Median", "Min", "Max", "Std", "Skew", "IQR", "Slope")

WINDOW_H = 12
RAW_FEATURES = ["MAP_sub65_hours", "SBP_sub90_hours", "PulsePressure_Ratio"]
N_FEATURES = len(CHANNELS) * WINDOW_H + len(CHANNELS) * len(SUBWINDOWS) * len(STATS) \
    + len(CHANNELS) * 3          # 120 + 480 + 30 = 630（+3 见 RAW_FEATURES）


# =============================================================================
# 1. 统计量（全部沿时间轴，向量化）
# =============================================================================
def _slope(Xw: np.ndarray) -> np.ndarray:
    """最小二乘斜率（单位：通道单位/小时）。Xw 形状 (..., L)。"""
    L = Xw.shape[-1]
    t = np.arange(1, L + 1, dtype=np.float64)
    t_c = t - t.mean()
    denom = float(np.dot(t_c, t_c))
    if denom <= 0:
        return np.full(Xw.shape[:-1], np.nan)
    y_c = Xw - Xw.mean(axis=-1, keepdims=True)
    return np.einsum("...l,l->...", y_c, t_c) / denom


def _skew(Xw: np.ndarray) -> np.ndarray:
    """总体偏度 m3 / m2^1.5。

    零方差（Std < 1e-8）时偏度在数学上无定义。经方法学拍板，此处**返回 0.0**
    而不是 NaN：一方面 TabNet 无法处理 NaN，另一方面"窗口完全平直"本身已由
    Std=0 表达，偏度取中性值 0 不会引入虚假的方向信息。
    """
    mu = Xw.mean(axis=-1, keepdims=True)
    d = Xw - mu
    m2 = (d ** 2).mean(axis=-1)
    m3 = (d ** 3).mean(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = m3 / np.power(m2, 1.5)
    degenerate = (m2 < 1e-16) | ~np.isfinite(out)     # Std < 1e-8
    out = np.where(degenerate, 0.0, out)
    return out


def _iqr(Xw: np.ndarray) -> np.ndarray:
    q75, q25 = np.percentile(Xw, [75, 25], axis=-1)
    return q75 - q25


STAT_FUNCS = {
    "Mean": lambda x: x.mean(axis=-1),
    "Median": lambda x: np.median(x, axis=-1),
    "Min": lambda x: x.min(axis=-1),
    "Max": lambda x: x.max(axis=-1),
    "Std": lambda x: x.std(axis=-1, ddof=1),
    "Skew": _skew,
    "IQR": _iqr,
    "Slope": _slope,
}


# =============================================================================
# 2. 基尼三层
# =============================================================================
def gini_terms(Xw: np.ndarray,
               clip: Optional[Tuple[float, float]] = (0.0, 1.0)) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (G, GI)。Xw 形状 (..., L)。

    G 用排序恒等式避免 O(L²) 循环：
        升序排列后 x_k 作为"较大者"出现 k 次（正号）、作为"较小者"出现 L-1-k 次（负号），
        故 G = Σ_k (2k - L + 1) · x_(k)
    """
    if clip is not None:
        Xw = np.clip(Xw, *clip)          # 只在基尼分支截断，L1/L2 统计量保持原值
    L = Xw.shape[-1]
    xs = np.sort(Xw, axis=-1)
    w = (2 * np.arange(L, dtype=np.float64) - L + 1)
    G = np.einsum("...l,l->...", xs, w)
    GI = 2.0 * G / (L * (L - 1))
    return G, GI


def directed_gini(GI: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """DGI = GI × sign(Slope)。斜率恰为 0 时符号取 0（不是 ±1），避免制造虚假方向。"""
    s = np.sign(slope)
    s[np.abs(slope) < 1e-12] = 0.0
    return GI * s


# =============================================================================
# 3. 三层特征构建器
# =============================================================================
@dataclass
class ThreeLayerFeatureBuilder:
    channels: Tuple[str, ...] = CHANNELS
    window_h: int = WINDOW_H
    subwindows: Tuple[Tuple[str, int, int], ...] = SUBWINDOWS
    stats: Tuple[str, ...] = STATS
    # 基尼分支专用截断：保证外部测试集上 GI, GV ∈ [0,1]
    gini_clip: Optional[Tuple[float, float]] = (0.0, 1.0)
    # 反归一化参数（center/scale）：提供后额外输出 3 个"未截断"的肾脏低灌注特征
    raw_params: Optional[Dict[str, Tuple[float, float]]] = None

    # ---- 名称表 ----
    def names(self) -> List[str]:
        out: List[str] = []
        for ch in self.channels:                       # 第 1 层
            out += [f"{ch}_t{t}" for t in range(1, self.window_h + 1)]
        for ch in self.channels:                       # 第 2 层
            for win, _, _ in self.subwindows:
                out += [f"{ch}_{win}_{st}" for st in self.stats]
        for ch in self.channels:                       # 第 3 层
            out += [f"{ch}_1-{self.window_h}h_GI", f"{ch}_1-{self.window_h}h_GV",
                    f"{ch}_1-{self.window_h}h_DGI"]
        if self.raw_params:                            # 第 4 层（可选）：低灌注时长
            out += RAW_FEATURES
        return out

    # ---- 主变换 ----
    def transform(self, X: np.ndarray) -> np.ndarray:
        """(N, C, T) -> (N, 630) float32。"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(f"输入应为 (N, C, T) 三维数组，实得 {X.shape}")
        N, C, T = X.shape
        if C != len(self.channels):
            raise ValueError(f"通道数应为 {len(self.channels)}，实得 {C}")
        if T != self.window_h:
            raise ValueError(f"窗口长度应为 {self.window_h}，实得 {T}")

        blocks: List[np.ndarray] = []

        # 第 1 层：原始归一化序列 (N, C, T) -> (N, C*T)
        blocks.append(X.reshape(N, C * T))

        # 第 2 层：6 个子窗 × 8 统计量，按 [通道][子窗][统计量] 排布
        for c in range(C):
            for win, a, b in self.subwindows:
                sub = X[:, c, a:b]                     # (N, L)
                for st in self.stats:
                    blocks.append(STAT_FUNCS[st](sub).reshape(N, 1))

        # 第 3 层：GI / GV / DGI（固定 1-12h 全窗，与 Slope_1-12h 同源）
        for c in range(C):
            # GI/GV 用截断后的序列；DGI 的符号取自**未截断**序列的斜率，
            # 这样极端低灌注造成的陡降不会被截断抹平方向。
            _, GI = gini_terms(X[:, c, :], clip=self.gini_clip)
            GV = 1.0 - GI
            DGI = directed_gini(GI, _slope(X[:, c, :]))
            blocks.append(GI.reshape(N, 1))
            blocks.append(GV.reshape(N, 1))
            blocks.append(DGI.reshape(N, 1))

        if self.raw_params:
            blocks.extend(self._raw_perfusion(X, N))

        out = np.concatenate(blocks, axis=1).astype(np.float32)
        if out.shape[1] != len(self.names()):
            raise AssertionError(f"特征数 {out.shape[1]} 与名称数 {len(self.names())} 不一致")
        return out

    def _raw_perfusion(self, X: np.ndarray, N: int) -> List[np.ndarray]:
        """3 个未截断的肾脏低灌注特征（需反归一化回 mmHg）。

        为什么必须用原始值：归一化后的 MAP=0.3 在不同患者身上对应不同的 mmHg，
        而"MAP<65 持续几小时"是固定临床阈值，只能在原始尺度上数。
        """
        def inv(ch: str) -> np.ndarray:
            j = self.channels.index(ch)
            c, sc = self.raw_params.get(ch, (0.0, 1.0))
            return X[:, j, :] * float(sc) + float(c)

        map_raw, sbp_raw, pp_raw = inv("MAP"), inv("SBP"), inv("PP")
        map_sub65 = (map_raw < 65.0).sum(axis=1).astype(np.float32)      # 12h 内的小时数
        sbp_sub90 = (sbp_raw < 90.0).sum(axis=1).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            ppr = np.where(np.abs(sbp_raw) > 1e-6, pp_raw / sbp_raw, np.nan).mean(axis=1)
        ppr = np.nan_to_num(ppr, nan=0.0).astype(np.float32)
        return [map_sub65.reshape(N, 1), sbp_sub90.reshape(N, 1), ppr.reshape(N, 1)]

    # ---- 分组（供模块 4 消融）----
    def groups(self) -> Dict[str, List[str]]:
        n = self.names()
        g: Dict[str, List[str]] = {
            "L1_raw": [x for x in n if "_t" in x and x.split("_t")[-1].isdigit()],
            "L2_stats": [x for x in n if any(x.endswith("_" + s) for s in self.stats)],
            "L3_gini": [x for x in n if x.endswith(("_GI", "_GV", "_DGI"))],
        }
        for ch in self.channels:
            g[f"var_{ch}"] = [x for x in n if x.startswith(ch + "_")]
        return g

    # ---- 输出 ----
    def to_frame(self, F: np.ndarray, meta: Optional[pd.DataFrame] = None,
                 chunk: int = 200_000, nan_policy: str = "keep") -> pd.DataFrame:
        """转成带列名的表。

        nan_policy
            keep : 保留 NaN。Skew 在零方差子窗上无定义，这是真实信息（窗口是平的）
            zero : 用 0 填。**TabNet 不能吃 NaN**，训练前必须用这个
            （或自行用训练集统计量填），LightGBM 则可以保留 NaN。
        """
        if nan_policy not in ("keep", "zero"):
            raise ValueError("nan_policy 只能是 keep | zero")
        names = self.names()
        if meta is not None and len(meta) != F.shape[0]:
            raise ValueError(f"meta 行数 {len(meta)} 与特征行数 {F.shape[0]} 不一致")
        parts: List[pd.DataFrame] = []
        for i in range(0, F.shape[0], chunk):
            sl = slice(i, i + chunk)
            part = pd.DataFrame(F[sl], columns=names)
            if nan_policy == "zero":
                part = part.fillna(0.0)
            if meta is not None:
                m = meta.iloc[sl].reset_index(drop=True)
                part = pd.concat([m, part], axis=1)
            parts.append(part)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=names)


def parse_feature_name(name: str, channels: Sequence[str] = CHANNELS
                       ) -> Tuple[str, str, str]:
    """把特征名拆成 (变量, 窗口, 统计量)。

    因为通道名可能含连字符（如 MEWS-4）、窗口名本身也含连字符（如 1-12h），
    不能用 split("_") 硬切。这里先用已知通道前缀匹配，再 rsplit。
    """
    for ch in sorted(channels, key=len, reverse=True):
        if name.startswith(ch + "_"):
            rest = name[len(ch) + 1:]
            if rest.startswith("t") and rest[1:].isdigit():
                return ch, "t", rest[1:]
            win, stat = rest.rsplit("_", 1)
            return ch, win, stat
    raise ValueError(f"无法解析特征名：{name}")


def feature_names(df: pd.DataFrame) -> List[str]:
    """从带 meta 的表里挑出 630 个特征列。"""
    meta = {"patient_id", "track", "t_end_h", "y", "t_aki_onset", "followup_h",
            "observed_frac", "or_frac", "label_status", "age", "sex", "icu_type", "split"}
    return [c for c in df.columns if c not in meta and not c.startswith("miss_")]


# =============================================================================
# 4. 自检
# =============================================================================
def _toy_X() -> Tuple[np.ndarray, List[str]]:
    """构造 3 个可控窗口，用于手算校验。"""
    C, T = len(CHANNELS), WINDOW_H
    X = np.zeros((3, C, T), dtype=np.float64)
    X[0, :, :] = 0.5                                   # 常数窗
    X[1, :, :6] = 0.0
    X[1, :, 6:] = 1.0                                  # 前 6 步 0，后 6 步 1
    X[2, :, :6] = 1.0
    X[2, :, 6:] = 0.0                                  # 反向
    return X, list(CHANNELS)


def run_self_test() -> int:
    fails: List[str] = []
    fb = ThreeLayerFeatureBuilder()
    X, chans = _toy_X()
    F = fb.transform(X)
    names = fb.names()
    idx = {n: i for i, n in enumerate(names)}

    # ---- ① 维度与命名 ----
    if F.shape != (3, 630):
        fails.append(f"形状应为 (3,630)，实得 {F.shape}")
    if len(set(names)) != 630:
        fails.append(f"特征名有重复：{630 - len(set(names))} 个")
    import re
    # 第 1 层是 {Var}_t{n}，第 2/3 层是 {Var}_{Window}_{Stat}
    pat_l1 = re.compile(r"^[A-Za-z0-9\-]+_t\d+$")
    pat_l23 = re.compile(r"^[A-Za-z0-9\-]+_(1-4h|5-8h|9-12h|1-6h|7-12h|1-12h)_"
                         r"(Mean|Median|Min|Max|Std|Skew|IQR|Slope|GI|GV|DGI)$")
    bad = [n for n in names if not (pat_l1.match(n) or pat_l23.match(n))]
    if bad:
        fails.append(f"命名不符合 {{Var}}_{{Window}}_{{Stat}}：{bad[:5]}")

    # ---- ② 常数窗：Min=Max=Mean=0.5，Std=IQR=0，Skew 无定义 ----
    for st, exp in (("Mean", 0.5), ("Min", 0.5), ("Max", 0.5), ("Std", 0.0), ("IQR", 0.0)):
        v = F[0, idx[f"HR_1-12h_{st}"]]
        if not np.isclose(v, exp, atol=1e-6):
            fails.append(f"常数窗 HR_1-12h_{st} 应为 {exp}，实得 {v}")
    if not np.isclose(F[0, idx["HR_1-12h_Skew"]], 0.0):
        fails.append(f"常数窗 Skew 应为 0.0（零方差拍板值），实得 {F[0, idx['HR_1-12h_Skew']]}")

    # ---- ③ 阶跃窗：Min=0, Max=1, Median=0.5, GI=36/66, GV=1-GI ----
    v_min, v_max = F[1, idx["HR_1-12h_Min"]], F[1, idx["HR_1-12h_Max"]]
    if not (np.isclose(v_min, 0.0) and np.isclose(v_max, 1.0)):
        fails.append(f"阶跃窗 Min/Max 应为 0/1，实得 {v_min}/{v_max}")
    gi = F[1, idx["HR_1-12h_GI"]]
    if not np.isclose(gi, 36 / 66, atol=1e-6):
        fails.append(f"GI 应为 36/66={36/66:.6f}，实得 {gi:.6f}")
    gv = F[1, idx["HR_1-12h_GV"]]
    if not np.isclose(gv, 1 - 36 / 66, atol=1e-6):
        fails.append(f"GV 应为 1-GI，实得 {gv:.6f}")

    # ---- ④ 斜率：6 个 0 后 6 个 1 -> 18/143 ≈ 0.12587 ----
    sl = F[1, idx["HR_1-12h_Slope"]]
    if not np.isclose(sl, 18 / 143, atol=1e-6):
        fails.append(f"阶跃窗斜率应为 18/143={18/143:.6f}，实得 {sl:.6f}")

    # ---- ⑤ DGI 方向 ----
    dgi_up = F[1, idx["HR_1-12h_DGI"]]
    dgi_dn = F[2, idx["HR_1-12h_DGI"]]
    if not (dgi_up > 0 and dgi_dn < 0):
        fails.append(f"DGI 应随斜率变号，实得 up={dgi_up:.4f} dn={dgi_dn:.4f}")
    if not np.isclose(abs(dgi_up), abs(dgi_dn), atol=1e-6):
        fails.append("DGI 绝对值应等于 GI")

    # ---- ⑥ 线性斜坡：斜率应为 1.0/h，偏度应为 0 ----
    ramp = np.tile(np.arange(1, 13, dtype=float), (1, len(CHANNELS), 1))
    F_ramp = fb.transform(ramp)
    if not np.isclose(F_ramp[0, idx["HR_1-12h_Slope"]], 1.0, atol=1e-6):
        fails.append(f"线性斜坡斜率应为 1.0，实得 {F_ramp[0, idx['HR_1-12h_Slope']]}")
    if abs(F_ramp[0, idx["HR_1-12h_Skew"]]) > 1e-6:
        fails.append(f"对称斜坡偏度应为 0，实得 {F_ramp[0, idx['HR_1-12h_Skew']]}")

    # ---- ⑦ 子窗口确实只取自己的区间 ----
    # HR_9-12h_Mean 在阶跃窗上应为 1.0（后 4 步全 1），HR_1-4h_Mean 应为 0.0
    if not np.isclose(F[1, idx["HR_9-12h_Mean"]], 1.0):
        fails.append(f"HR_9-12h_Mean 应为 1.0，实得 {F[1, idx['HR_9-12h_Mean']]}")
    if not np.isclose(F[1, idx["HR_1-4h_Mean"]], 0.0):
        fails.append(f"HR_1-4h_Mean 应为 0.0，实得 {F[1, idx['HR_1-4h_Mean']]}")

    # ---- ⑧ 分组完整性 ----
    g = fb.groups()
    if len(g["L1_raw"]) != 120 or len(g["L2_stats"]) != 480 or len(g["L3_gini"]) != 30:
        fails.append(f"三层分组应为 120/480/30，实得 "
                     f"{len(g['L1_raw'])}/{len(g['L2_stats'])}/{len(g['L3_gini'])}")

    # ---- ⑩ 基尼截断保护：越界输入下 GI/GV 仍在 [0,1]，L2 统计量保留原值 ----
    Xo = np.full((1, len(CHANNELS), WINDOW_H), 1.8)
    Xo[0, :, :6] = -0.5
    Fo = fb.transform(Xo)
    gi_o = float(Fo[0, idx["HR_1-12h_GI"]]); gv_o = float(Fo[0, idx["HR_1-12h_GV"]])
    if not (0.0 <= gi_o <= 1.0 and 0.0 <= gv_o <= 1.0):
        fails.append(f"基尼截断失效：GI={gi_o:.4f}, GV={gv_o:.4f} 越界")
    if not np.isclose(Fo[0, idx["HR_1-12h_Max"]], 1.8):
        fails.append("L2 的 Max 不应被截断（应保留 1.8，以保住极端低灌注峰值）")

    # ---- ⑪ 特征名解析（含连字符通道） ----
    for nm, exp in (("MEWS-4_1-12h_DGI", ("MEWS-4", "1-12h", "DGI")),
                    ("MAP_5-8h_Min", ("MAP", "5-8h", "Min")),
                    ("HR_t12", ("HR", "t", "12"))):
        got = parse_feature_name(nm)
        if got != exp:
            fails.append(f"parse_feature_name({nm}) = {got}，应为 {exp}")

    # ---- ⑫ 低灌注特征（反归一化后按固定 mmHg 阈值计数）----
    params = {ch: (0.0, 100.0) for ch in CHANNELS}      # 原始 = scaled * 100
    fb_raw = ThreeLayerFeatureBuilder(raw_params=params)
    Xr = np.full((1, len(CHANNELS), WINDOW_H), 0.5)      # 全部通道 = 50
    Xr[0, list(CHANNELS).index("MAP"), :6] = 0.7         # MAP 前 6 步 = 70
    Xr[0, list(CHANNELS).index("SBP"), :] = 0.85         # SBP 全程 = 85 (<90)
    Fr = fb_raw.transform(Xr)
    nr = fb_raw.names()
    if len(nr) != 633:
        fails.append(f"含低灌注特征时列数应为 633，实得 {len(nr)}")
    v65 = float(Fr[0, nr.index("MAP_sub65_hours")])
    if not np.isclose(v65, 6.0):
        fails.append(f"MAP_sub65_hours 应为 6（后 6 步 MAP=50<65），实得 {v65}")
    v90 = float(Fr[0, nr.index("SBP_sub90_hours")])
    if not np.isclose(v90, 12.0):
        fails.append(f"SBP_sub90_hours 应为 12，实得 {v90}")
    ppr = float(Fr[0, nr.index("PulsePressure_Ratio")])
    # PP 通道保持 0.5 -> 原始 50；SBP 0.85 -> 85；故比值 = 50/85
    if not np.isclose(ppr, 50.0 / 85.0, atol=1e-5):
        fails.append(f"PulsePressure_Ratio 应为 50/85={50/85:.4f}，实得 {ppr:.4f}")

    # ---- ⑨ 窗口间互不影响 ----
    X2 = X.copy()
    X2[0, :, :] = 0.9
    F2 = fb.transform(X2)
    # equal_nan：常数子窗的 Skew 无定义（NaN），两侧同为 NaN 不算差异
    if not np.allclose(F[1], F2[1], atol=1e-7, equal_nan=True):
        fails.append("修改第 1 个窗口影响了第 2 个窗口的特征")

    print("=" * 70)
    print("模块3 自检：630 维 / 命名 / 统计量手算 / 基尼 / 方向")
    print("=" * 70)
    for f in fails:
        print("  [FAIL]", f)
    print(f"  特征数 {F.shape[1]}（L1=120, L2=480, L3=30）")
    print(f"  示例列：{names[120:124]} ... {names[-3:]}")
    print(f"  失败 {len(fails)} 项")
    return 1 if fails else 0


# =============================================================================
# 5. CLI
# =============================================================================
def _load_sibling(mod_name: str, filename: str):
    p = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(mod_name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="三层特征工程（630 维）")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--demo", action="store_true", help="合成数据 -> 窗口 -> 特征 全链路")
    ap.add_argument("--npz", help="模块2 产出的 *_X.npy")
    ap.add_argument("--meta", help="模块2 产出的 *_meta.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--tag", default="track1")
    ap.add_argument("--params", help="preprocess_params.json，提供后额外输出 3 个低灌注特征")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    if args.self_test:
        return run_self_test()

    out = ensure_dir(args.out)
    raw_params = None
    if args.params:
        d = json.loads(Path(args.params).read_text(encoding="utf-8"))
        raw_params = {ch: (float(d["center"][ch]), float(d["scale"][ch]))
                      for ch in d.get("center", {})}
        LOG.info("已加载反归一化参数，附加 3 个低灌注特征")
    fb = ThreeLayerFeatureBuilder(raw_params=raw_params)

    if args.demo:
        m2 = _load_sibling("m2_window", "02_window_preprocessor.py")
        from cdss_common import synthesize_demo
        LOG.info("生成合成演示数据（非真实患者）")
        vitals, cohort = synthesize_demo(n_patients=60, seed=2026)
        pp = m2.TemporalPreprocessor(min_observed_frac=0.3)
        pids = cohort["patient_id"].tolist()
        pp.fit(vitals, pids[: int(0.7 * len(pids))], cohort)
        ws = pp.run(vitals, cohort, "track1")
        LOG.info("窗口张量 %s", ws.X.shape)
        X, meta = ws.X, ws.meta
    else:
        if not args.npz or not args.meta:
            ap.error("需要 --npz 与 --meta（或使用 --demo / --self-test）")
        X = np.load(args.npz)
        meta = pd.read_csv(args.meta)

    F = fb.transform(X)
    LOG.info("特征矩阵 %s（%.1f MB）", F.shape, F.nbytes / 1e6)
    df = fb.to_frame(F, meta)

    npy = out / f"features_{args.tag}.npy"
    np.save(npy, F)
    try:
        tab = out / f"features_{args.tag}.parquet"
        df.to_parquet(tab, index=False)
    except Exception as exc:  # noqa: BLE001
        tab = out / f"features_{args.tag}.csv.gz"
        LOG.warning("parquet 写入失败（%s），改存 %s", exc, tab)
        df.to_csv(tab, index=False, compression="gzip")

    save_json({"n_features": int(F.shape[1]), "n_rows": int(F.shape[0]),
               "columns": fb.names(),
               "groups": {k: len(v) for k, v in fb.groups().items()}}, out / "feature_spec.json")

    print(f"特征矩阵: {F.shape}")
    print(f"  L1 原始序列 120 | L2 子窗统计 480 | L3 基尼 30")
    print(f"  前 5 列: {fb.names()[:5]}")
    print(f"  抽样列: {[fb.names()[120], fb.names()[480], fb.names()[-1]]}")
    print(f"  -> {npy}")
    print(f"  -> {tab}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
