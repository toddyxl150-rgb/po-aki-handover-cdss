#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cdss_common.py — PO_AKI_CDSS 公共层
===================================

放三样东西，避免模块 2/3/4 各写一份而产生口径分歧：

1. **10 个通道的定义**（6 基础 + 4 衍生），全工程唯一来源
2. **生理极端值范围**（专家定义）与 MEWS 打分表
3. **演示数据生成器**（合成数据，用于跑通流水线，不含任何真实患者信息）

通道口径（务必与论文方法学一致）
--------------------------------
基础 6 项：HR, SBP, DBP, RR, SpO2, Temp
衍生 4 项：
    MAP  = (SBP + 2*DBP) / 3        ← 按任务书公式，**不用**监护仪直接测的 MAP
    PP   = SBP - DBP
    SI   = HR / SBP                 （休克指数）
    MEWS-4 = 4 项分量各自 0-3 分后求和（SBP/HR/RR/Temp），总分 0-12

MEWS 说明：完整版 MEWS 含 AVPU 意识项（总分 0-14）。eICU / INSPIRE / VitalDB
均无逐小时 AVPU，强行补 0 分会系统性低估，故本工程固定用 4 项版，
通道标识符与输出列名统一为 `MEWS-4`（带连字符，`rsplit("_",2)` 仍可正确解析）。若后续数据源提供 AVPU，把 `mews_score(..., avpu=...)` 打开即可。

时间轴约定
----------
所有 hour 均为「相对 ICU 入科的小时数」，可以是负数：
    hour >= 0  ICU 内
    hour <  0  入 ICU 前（手术室 / PACU / 病房），Track 2 用它拼交接期窗口
小时桶定义为 [h, h+1)，即 floor(t) 落在哪个桶。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# 1. 通道定义
# =============================================================================
CHANNELS_BASE: Tuple[str, ...] = ("HR", "SBP", "DBP", "RR", "SpO2", "Temp")
CHANNELS_DERIVED: Tuple[str, ...] = ("MAP", "PP", "SI", "MEWS-4")
CHANNELS: Tuple[str, ...] = CHANNELS_BASE + CHANNELS_DERIVED   # 10 个

# 生理极端值（专家定义）：超出即置 NaN，按缺失处理，不参与插补前的统计
# 依据：成人 ICU 监护常见量程 + 明显不可能的录入错误（如 HR=0 或 999）
PHYSIO_RANGES: Dict[str, Tuple[float, float]] = {
    "HR":   (20.0, 250.0),     # 次/分
    "SBP":  (40.0, 300.0),     # mmHg
    "DBP":  (20.0, 200.0),     # mmHg
    "RR":   (2.0, 80.0),       # 次/分
    "SpO2": (50.0, 100.0),     # %
    "Temp": (30.0, 43.0),      # 摄氏度
}

# 跨库统一固定标尺（Fixed Physiological Bounds Scaling）
# 用固定生理极值而非训练集经验 min-max：INSPIRE 与 eICU 的同一数值映射到同一区间，
# 彻底消除"源库 HR 只有 50-120，目标库 160 被压到 >1"的标尺错位。
FIXED_BOUNDS: Dict[str, Tuple[float, float]] = {
    "HR": (30.0, 220.0), "SBP": (40.0, 260.0), "DBP": (20.0, 160.0),
    "MAP": (30.0, 180.0), "PP": (10.0, 160.0), "RR": (4.0, 60.0),
    "SpO2": (50.0, 100.0), "Temp": (32.0, 42.0),
    "SI": (0.2, 3.0), "MEWS-4": (0.0, 12.0),
}

# 衍生通道的合理范围（衍生后再过滤一次，防止 SBP/DBP 倒挂等异常）
DERIVED_RANGES: Dict[str, Tuple[float, float]] = {
    "MAP":  (30.0, 200.0),
    "PP":   (5.0, 150.0),
    "SI":   (0.2, 3.0),
    "MEWS-4": (0.0, 12.0),
}

# 各通道在原始数据中的常见别名（小写匹配），供各库适配器复用
CHANNEL_ALIASES: Dict[str, List[str]] = {
    "HR":   ["heart rate", "hr", "heartrate", "pulse", "heart_rate", "心率"],
    "SBP":  ["systolic blood pressure", "sbp", "systolic", "bpsys", "收缩压",
             "non invasive blood pressure systolic", "arterial bp systolic"],
    "DBP":  ["diastolic blood pressure", "dbp", "diastolic", "bpdia", "舒张压",
             "non invasive blood pressure diastolic", "arterial bp diastolic"],
    "RR":   ["respiratory rate", "rr", "resp_rate", "breath rate", "呼吸频率"],
    "SpO2": ["oxygen saturation", "spo2", "sao2", "o2 saturation", "血氧饱和度"],
    "Temp": ["temperature", "temp", "body temperature", "temperature (c)", "体温"],
}


# =============================================================================
# 2. MEWS（4 项版）
# =============================================================================
def _mews_sbp(x: np.ndarray) -> np.ndarray:
    return np.select(
        [x <= 70, x <= 80, x <= 100, x < 200, x >= 200],
        [3.0, 2.0, 1.0, 0.0, 2.0], default=np.nan)


def _mews_hr(x: np.ndarray) -> np.ndarray:
    return np.select(
        [x < 40, x < 50, x <= 100, x <= 110, x <= 129, x >= 130],
        [2.0, 1.0, 0.0, 1.0, 2.0, 3.0], default=np.nan)


def _mews_rr(x: np.ndarray) -> np.ndarray:
    return np.select(
        [x < 9, x < 15, x < 21, x < 30, x >= 30],
        [2.0, 0.0, 1.0, 2.0, 3.0], default=np.nan)


def _mews_temp(x: np.ndarray) -> np.ndarray:
    return np.select([x < 35.0, x < 38.5, x >= 38.5], [2.0, 0.0, 2.0], default=np.nan)


def mews_score(sbp: np.ndarray, hr: np.ndarray, rr: np.ndarray, temp: np.ndarray,
               avpu: Optional[np.ndarray] = None) -> np.ndarray:
    """逐小时 MEWS。四项缺一即 NaN（不做「缺项按 0 分」的乐观填补）。"""
    parts = [_mews_sbp(np.asarray(sbp, float)), _mews_hr(np.asarray(hr, float)),
             _mews_rr(np.asarray(rr, float)), _mews_temp(np.asarray(temp, float))]
    if avpu is not None:
        parts.append(np.asarray(avpu, float))
    stack = np.vstack(parts)
    return np.where(np.isnan(stack).any(axis=0), np.nan, stack.sum(axis=0))


# =============================================================================
# 3. 衍生通道
# =============================================================================
def filter_physio(df: pd.DataFrame, channels: Sequence[str] = CHANNELS_BASE,
                  inplace: bool = False) -> pd.DataFrame:
    """把超出专家定义范围的观测置为 NaN（不删除行，保留时间轴完整性）。"""
    out = df if inplace else df.copy()
    for ch in channels:
        if ch not in out.columns:
            continue
        lo, hi = PHYSIO_RANGES[ch]
        col = pd.to_numeric(out[ch], errors="coerce")
        out[ch] = col.mask((col < lo) | (col > hi))
    return out


def derive_channels(df: pd.DataFrame, drop_invalid: bool = True,
                    bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> pd.DataFrame:
    """在宽表上生成 4 个衍生通道。

    必须在**插补之前**调用：用真实观测算衍生量，而不是用中位数填出来的假值。
    缺分量的地方保持 NaN，交给后面的 LOCF / 中位数兜底。
    """
    out = df.copy()
    sbp = pd.to_numeric(out.get("SBP"), errors="coerce")
    dbp = pd.to_numeric(out.get("DBP"), errors="coerce")
    hr = pd.to_numeric(out.get("HR"), errors="coerce")

    if sbp is not None and dbp is not None:
        out["MAP"] = (sbp + 2.0 * dbp) / 3.0
        out["PP"] = sbp - dbp
    if sbp is not None and hr is not None:
        out["SI"] = hr / sbp.replace(0, np.nan)

    if all(c in out.columns for c in ("SBP", "HR", "RR", "Temp")):
        out["MEWS-4"] = mews_score(out["SBP"].to_numpy(float), out["HR"].to_numpy(float),
                                 out["RR"].to_numpy(float), out["Temp"].to_numpy(float))

    if drop_invalid:
        rng = bounds if bounds is not None else DERIVED_RANGES
        for ch, (lo, hi) in rng.items():
            if ch in out.columns:
                col = out[ch]
                out[ch] = col.mask((col < lo) | (col > hi))
    return out


# =============================================================================
# 4. 演示数据（合成，仅供跑通流水线）
# =============================================================================
def synthesize_demo(n_patients: int = 40, seed: int = 2026,
                    include_or: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成符合本项目输入契约的合成数据。

    返回
    ----
    vitals_long : patient_id, hour, variable, value
    cohort      : patient_id, t_aki_onset, followup_h, age, sex, icu_type

    说明：这里的数据是**随机合成**的，仅用于验证代码可运行、维度与命名正确，
    任何数值都不代表真实临床规律，不得用于任何结论。
    """
    rng = np.random.default_rng(seed)
    vrows: List[Dict] = []
    crows: List[Dict] = []

    for pid in range(1, n_patients + 1):
        followup = int(rng.integers(36, 97))          # 36-96h
        # 约 1/3 患者发生 PO-AKI，发病时间必须严格早于随访结束
        has_aki = rng.random() < 0.35
        t_aki = float(rng.integers(6, max(7, followup - 1))) if has_aki else np.nan

        base_hr = 70 + rng.normal(0, 10)
        base_sbp = 118 + rng.normal(0, 12)
        base_dbp = 68 + rng.normal(0, 8)
        base_rr = 16 + rng.normal(0, 2)
        base_spo2 = 97 + rng.normal(0, 1.5)
        base_temp = 36.9 + rng.normal(0, 0.4)

        # 负小时段：术室/麻醉恢复（Track 2 用）
        hours = list(range(-12, 0)) if include_or else []
        hours += list(range(0, followup))

        for h in hours:
            # 每格 12% 概率缺失，模拟护理记录断档
            def maybe(v, miss=0.12):
                return v if rng.random() > miss else np.nan

            drift = 0.25 * h if has_aki and h > t_aki else 0.0   # 发病后血流动力学恶化
            vrows.append({"patient_id": pid, "hour": h, "variable": "HR",
                          "value": maybe(base_hr + 0.05 * h + drift + rng.normal(0, 4))})
            vrows.append({"patient_id": pid, "hour": h, "variable": "SBP",
                          "value": maybe(base_sbp - 0.1 * h - 2 * drift + rng.normal(0, 6))})
            vrows.append({"patient_id": pid, "hour": h, "variable": "DBP",
                          "value": maybe(base_dbp - 0.05 * h + rng.normal(0, 4))})
            vrows.append({"patient_id": pid, "hour": h, "variable": "RR",
                          "value": maybe(base_rr + 0.02 * h + rng.normal(0, 2))})
            vrows.append({"patient_id": pid, "hour": h, "variable": "SpO2",
                          "value": maybe(min(100.0, base_spo2 - 0.02 * h + rng.normal(0, 1.2)))})
            vrows.append({"patient_id": pid, "hour": h, "variable": "Temp",
                          "value": maybe(base_temp + rng.normal(0, 0.3))})

        crows.append({"patient_id": pid, "t_aki_onset": t_aki, "followup_h": float(followup),
                      "age": float(rng.integers(40, 85)),
                      "sex": "F" if rng.random() < 0.45 else "M",
                      "icu_type": ["SICU", "MICU", "CSRU", "CTICU"][int(rng.integers(0, 4))]})

    return pd.DataFrame(vrows), pd.DataFrame(crows)


# =============================================================================
# 5. 小工具
# =============================================================================
def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str),
                          encoding="utf-8")


def channel_column_order() -> List[str]:
    return list(CHANNELS)
