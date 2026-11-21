#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_window_preprocessor.py — 防泄露时序分桶与双轨滑动窗口生成器
=============================================================

本模块负责「把原始长表变成模型能吃的窗口张量」，且全程不泄露未来信息。

流水线
------
    长表 (patient_id, hour, variable, value)
      │
      ├─ ① 变量名归一化 → 10 个通道
      ├─ ② 生理极端值剔除（专家定义范围，超限置 NaN）
      ├─ ③ 1 小时分桶取均值（桶 = [h, h+1)）
      ├─ ④ 衍生 4 通道（MAP/PP/SI/MEWS），**用真实观测算，不用填补值**
      ├─ ⑤ 缺失掩码记录（填补前）
      ├─ ⑥ LOCF 前向结转（逐患者，只向过去取数）
      ├─ ⑦ 训练集中位数兜底（起始端缺失）
      ├─ ⑧ 训练集 Min-Max / Robust 归一化
      └─ ⑨ 滑窗切片 + 滚动标签
            Track 1：ICU 内，t_end ∈ [12, followup]，窗口 [t_end-12, t_end)
            Track 2：交接期，窗口 [-4, +8)，术中末 4h + 入 ICU 后 8h，t_end = +8

三条硬性约束（违反即报错，不静默放过）
------------------------------------
1. 归一化器与中位数只在 `train_patients` 上估计；`fit()` 会检查传入集合非空。
2. 插补只允许 LOCF：本模块用逐小时 `np.where(isnan, prev, cur)` 实现，
   结构上不存在向后填充的可能（自检里有反证用例）。
3. 发病后窗口一律截断：`t_end >= t_aki_onset` 的窗口全部丢弃，不参与训练与评估。

为什么用 3D 数组而不是逐患者循环
------------------------------
按 (患者 × 小时 × 通道) 组织成三维数组后，LOCF 只需沿时间轴做 H 次向量化赋值，
比逐患者 groupby.ffill 快一到两个数量级；切片时再按每位患者的真实随访长度取视图，
不会把出院后的空白当成"继续观测"。

用法
----
    python 02_window_preprocessor.py --self-test
    python 02_window_preprocessor.py --demo --out outputs
    python 02_window_preprocessor.py --vitals vitals.csv --cohort cohort.csv \
        --out outputs --track track1 --scaler minmax

输入契约
--------
vitals : patient_id, hour, variable, value        （hour 可负，负值代表入 ICU 前）
cohort : patient_id, t_aki_onset, followup_h      （t_aki_onset 无发病则留空）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdss_common import (  # noqa: E402
    CHANNELS, CHANNELS_BASE, CHANNEL_ALIASES, DERIVED_RANGES, FIXED_BOUNDS,
    PHYSIO_RANGES, derive_channels, ensure_dir, save_json, synthesize_demo,
)

LOG = logging.getLogger("cdss.window")

PID, HOUR, VAR, VAL = "patient_id", "hour", "variable", "value"


# =============================================================================
# 1. 数据结构
# =============================================================================
@dataclass
class WindowSet:
    """一批窗口张量 + 元数据。X 形状 (N, C, T)，C=10 通道，T=12 时间步。"""
    X: np.ndarray
    meta: pd.DataFrame
    channels: Tuple[str, ...]
    window_h: int
    track: str
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    def save(self, out_dir: str | Path, name: Optional[str] = None) -> Dict[str, str]:
        out_dir = ensure_dir(out_dir)
        name = name or f"windows_{self.track}"
        x_path = out_dir / f"{name}_X.npy"
        m_path = out_dir / f"{name}_meta.csv"
        j_path = out_dir / f"{name}_manifest.json"
        np.save(x_path, self.X.astype(np.float32))
        self.meta.to_csv(m_path, index=False, encoding="utf-8-sig")
        save_json({"track": self.track, "channels": list(self.channels),
                   "window_h": self.window_h, "n_windows": self.n,
                   "X_shape": list(self.X.shape), "stats": self.stats}, j_path)
        return {"X": str(x_path), "meta": str(m_path), "manifest": str(j_path)}

    @classmethod
    def load(cls, out_dir: str | Path, name: str) -> "WindowSet":
        out_dir = Path(out_dir)
        man = json.loads((out_dir / f"{name}_manifest.json").read_text(encoding="utf-8"))
        X = np.load(out_dir / f"{name}_X.npy")
        meta = pd.read_csv(out_dir / f"{name}_meta.csv")
        return cls(X=X, meta=meta, channels=tuple(man["channels"]),
                   window_h=int(man["window_h"]), track=man["track"],
                   stats=man.get("stats", {}))


@dataclass
class ScalerParams:
    """训练集上估计出的归一化参数。transform 只做算术，不再看数据分布。"""
    method: str
    center: np.ndarray      # (C,)
    scale: np.ndarray       # (C,)
    n_fit: int

    def apply(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.center) / self.scale


# =============================================================================
# 2. 预处理器
# =============================================================================
class TemporalPreprocessor:
    """1h 分桶 + 因果插补 + 训练集归一化 + 双轨滑窗。

    典型用法（单次切分）::

        pp = TemporalPreprocessor()
        pp.fit(vitals, train_patients=train_ids)
        ws = pp.run(vitals, cohort, track="track1")

    10 折交叉验证时（模块 4），每折重新 ``fit`` 一次即可——绝不能在折外 fit。
    """

    def __init__(self, channels: Sequence[str] = CHANNELS, window_h: int = 12,
                 step_h: int = 1, horizon_h: int = 24, scaler: str = "fixed",
                 min_observed_frac: float = 0.5, max_hours: float = 168.0,
                 clip_scaled: Optional[Tuple[float, float]] = None,
                 unscaled_channels: Sequence[str] = (),
                 stage_target: int = 1):
        if scaler not in ("minmax", "robust", "zscore", "fixed"):
            raise ValueError("scaler 只能是 minmax | robust | zscore | fixed")
        self.channels = tuple(channels)
        self.window_h = int(window_h)
        self.step_h = int(step_h)
        self.horizon_h = int(horizon_h)
        self.scaler = scaler
        self.min_observed_frac = float(min_observed_frac)
        self.max_hours = float(max_hours)
        # 默认 None：保留极端低灌注峰值，供 L1/L2 统计量使用。
        # Gini 分支所需的 [0,1] 截断在模块 3 内部单独施加（见 03_feature_engineering.py）。
        self.clip_scaled = clip_scaled
        # 不做归一化的通道：MEWS 这类有固定临床量纲（0-12 分）的指标，
        # 归一化后 MEWS_1-12h_Max 就不再等于"最高几分"，可解释性受损。
        self.unscaled_channels = tuple(unscaled_channels)
        # 1 = Stage>=1（全分级 AKI）；2 = Stage>=2（中重度 AKI，主结局）
        if stage_target not in (1, 2):
            raise ValueError('stage_target 只能是 1 或 2')
        self.stage_target = int(stage_target)
        # fixed：用跨库统一的生理极值标尺，过滤范围与标尺保持一致，
        # 保证任何库的任何合法值都落在 [0,1]，不出现量程越界。
        self._bounds = dict(FIXED_BOUNDS) if scaler == "fixed" else dict(PHYSIO_RANGES)
        self.onset_col = 't_aki_onset' if stage_target == 1 else 't_stage2'

        self.medians_: Optional[np.ndarray] = None
        self.scaler_: Optional[ScalerParams] = None
        self.train_patients_: Optional[set] = None

    # ------------------------------------------------------------------ 基础
    def _normalize_long(self, long_df: pd.DataFrame) -> pd.DataFrame:
        """列名归一化 + 变量名映射到 10 通道 + 生理极值剔除。"""
        need = {PID, HOUR, VAR, VAL}
        missing = need - set(long_df.columns)
        if missing:
            raise KeyError(f"vitals 缺少列 {sorted(missing)}；实际列 {list(long_df.columns)}")

        df = long_df[[PID, HOUR, VAR, VAL]].copy()
        df[HOUR] = pd.to_numeric(df[HOUR], errors="coerce")
        df[VAL] = pd.to_numeric(df[VAL], errors="coerce")
        df = df.dropna(subset=[HOUR, VAL])
        df["hour_bin"] = np.floor(df[HOUR]).astype(int)

        # 变量别名 -> 标准通道
        alias = {a.lower(): ch for ch, names in CHANNEL_ALIASES.items() for a in names}
        v = df[VAR].astype(str).str.strip().str.lower()
        df["channel"] = v.map(alias)
        unmapped = df.loc[df["channel"].isna(), VAR].astype(str).unique()
        if len(unmapped):
            LOG.warning("以下变量名无法映射到通道，已丢弃：%s", list(unmapped)[:8])
        df = df.dropna(subset=["channel"])

        # 生理极值剔除（向量化：按通道查上下限）
        lo = df["channel"].map(lambda c: self._bounds.get(c, PHYSIO_RANGES.get(c, (0, 1e9)))[0])
        hi = df["channel"].map(lambda c: self._bounds.get(c, PHYSIO_RANGES.get(c, (0, 1e9)))[1])
        bad = (df[VAL] < lo) | (df[VAL] > hi)
        if bad.any():
            LOG.info("剔除生理极端值 %d 条（%.2f%%）", int(bad.sum()),
                     100.0 * bad.mean())
        df = df.loc[~bad]
        # 7.7M 行的长表里 object 字符串极占内存；转 category 后内存降一个量级
        df[PID] = df[PID].astype("category")
        df["channel"] = df["channel"].astype("category")
        return df[[PID, "hour_bin", "channel", VAL]]

    def _build_cube(self, long_df: pd.DataFrame, hour_lo: int, hour_hi_map: pd.Series,
                    patients: Sequence) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (cube, obs_mask, hours)。

        cube     : (P, H+1, C) float32，已插补+归一化
        obs_mask : (P, H+1, C) bool，**填补前**是否有真实观测
        hours    : (H+1,) 小时轴
        """
        df = self._normalize_long(long_df)
        df = df[df[PID].isin(patients)]
        if df.empty:
            raise ValueError("过滤后没有任何观测，检查 patient_id 类型是否一致")

        h_max = int(np.nanmax(hour_hi_map.reindex(patients).to_numpy()))
        h_max = max(h_max, hour_lo + 1)
        hours = np.arange(hour_lo, h_max + 1)

        wide = df.pivot_table(index=[PID, "hour_bin"], columns="channel",
                              values=VAL, aggfunc="mean")
        wide = wide.reindex(columns=[c for c in CHANNELS_BASE])

        pids = list(patients)
        idx = pd.MultiIndex.from_product([pids, hours], names=[PID, "hour_bin"])
        wide = wide.reindex(idx)

        P, H, Cb = len(pids), len(hours), len(CHANNELS_BASE)
        arr = wide.to_numpy(dtype=np.float64).reshape(P, H, Cb).copy()

        # 随访窗之外的行置 NaN：避免把出院后的空白当作"继续观测"参与统计
        hi = hour_hi_map.reindex(pids).to_numpy(dtype=float)
        beyond = hours[None, :] > hi[:, None]
        arr[beyond] = np.nan

        cube = np.full((P, H, len(self.channels)), np.nan, dtype=np.float64)
        for j, ch in enumerate(self.channels):
            if ch in CHANNELS_BASE:
                cube[:, :, j] = arr[:, :, list(CHANNELS_BASE).index(ch)]

        # 衍生通道：用真实观测计算（缺失保持 NaN）
        if any(ch not in CHANNELS_BASE for ch in self.channels):
            tmp = pd.DataFrame(arr.reshape(P * H, Cb), columns=list(CHANNELS_BASE))
            tmp = derive_channels(tmp, bounds=(FIXED_BOUNDS if self.scaler == "fixed"
                                               else DERIVED_RANGES))
            for j, ch in enumerate(self.channels):
                if ch not in CHANNELS_BASE and ch in tmp.columns:
                    cube[:, :, j] = tmp[ch].to_numpy().reshape(P, H)

        obs_mask = ~np.isnan(cube)
        return cube, obs_mask, hours

    # ------------------------------------------------------------------ 插补
    @staticmethod
    def _locf(cube: np.ndarray) -> np.ndarray:
        """沿时间轴前向结转。逐小时 np.where，结构上不可能出现向后填充。"""
        out = cube.copy()
        for h in range(1, out.shape[1]):
            miss = np.isnan(out[:, h, :])
            out[:, h, :] = np.where(miss, out[:, h - 1, :], out[:, h, :])
        return out

    def _fill(self, cube: np.ndarray) -> np.ndarray:
        out = self._locf(cube)
        if self.medians_ is None:
            raise RuntimeError("请先 fit()：中位数必须来自训练集")
        miss = np.isnan(out)
        if miss.any():
            out[miss] = np.take(self.medians_, np.where(miss)[2])
        return out

    # ------------------------------------------------------------------ 拟合
    def fit(self, long_df: pd.DataFrame, train_patients: Sequence,
            cohort: Optional[pd.DataFrame] = None) -> "TemporalPreprocessor":
        """在训练集患者上估计：各通道中位数 + 归一化参数。"""
        train_patients = list(dict.fromkeys(train_patients))
        if not train_patients:
            raise ValueError("train_patients 为空：拒绝在无训练集的情况下 fit")

        if cohort is not None:
            hi = cohort.set_index(PID)["followup_h"].astype(float)
        else:
            hi = (long_df.groupby(PID)[HOUR].max().astype(float) + 1.0)
        hi = hi.clip(upper=self.max_hours)

        cube, obs_mask, _ = self._build_cube(long_df, 0, hi, train_patients)

        # 中位数取自**真实观测**（不是 LOCF 后的序列，避免长缺口把分布拉偏）
        obs_vals = np.where(obs_mask, cube, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # 整列无观测时 nanmedian 会告警
            med = np.nanmedian(obs_vals.reshape(-1, len(self.channels)), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        self.medians_ = med.astype(np.float64)

        filled = self._fill(cube)
        flat = filled.reshape(-1, len(self.channels))
        flat = flat[np.isfinite(flat).all(axis=1)]
        if flat.shape[0] < 2:
            raise ValueError("训练集有效行不足 2 行，无法估计归一化参数")
        if flat.shape[0] < 50:
            LOG.warning("训练集仅 %d 行有效，归一化参数可能不稳定（真实数据不应出现）",
                        flat.shape[0])

        if self.scaler == "fixed":
            center = np.array([FIXED_BOUNDS[c][0] for c in self.channels], dtype=float)
            scale = np.array([FIXED_BOUNDS[c][1] - FIXED_BOUNDS[c][0]
                              for c in self.channels], dtype=float)
            LOG.info("使用跨库固定生理标尺（非训练集经验 min-max）")
        elif self.scaler == "minmax":
            center = np.nanmin(flat, axis=0)
            scale = np.nanmax(flat, axis=0) - center
        elif self.scaler == "robust":
            center = np.nanmedian(flat, axis=0)
            q75, q25 = np.nanpercentile(flat, [75, 25], axis=0)
            scale = q75 - q25
        else:
            center = np.nanmean(flat, axis=0)
            scale = np.nanstd(flat, axis=0)

        scale = np.where(np.isfinite(scale) & (np.abs(scale) > 1e-8), scale, 1.0)

        # 保留原生量纲的通道：center=0, scale=1，即恒等变换
        for ch in self.unscaled_channels:
            if ch in self.channels:
                j = self.channels.index(ch)
                center[j], scale[j] = 0.0, 1.0
                LOG.info("通道 %s 保持原生量纲（不归一化）", ch)
        self.scaler_ = ScalerParams(self.scaler, center.astype(np.float64),
                                    scale.astype(np.float64), int(flat.shape[0]))
        self.train_patients_ = set(train_patients)
        LOG.info("fit 完成：%d 名训练患者，%d 行有效；scaler=%s",
                 len(train_patients), flat.shape[0], self.scaler)
        return self

    def load_params(self, path: str | Path) -> "TemporalPreprocessor":
        """从源库的 preprocess_params.json 载入中位数与归一化参数，跳过 fit。

        跨库外部验证必须这样做：若在目标库重新 fit 归一化，就等于用目标库的分布
        做了无监督域适配，不再是"纯外部验证"。
        """
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        med = np.array([float(d["medians"][c]) for c in self.channels])
        if self.scaler == "fixed":
            # 固定标尺是跨库常量，不能被源库的经验 min-max 覆盖；
            # 只沿用源库中位数（填补用），保证填补与缩放都不借用目标库信息。
            center = np.array([FIXED_BOUNDS[c][0] for c in self.channels], dtype=float)
            scale = np.array([FIXED_BOUNDS[c][1] - FIXED_BOUNDS[c][0]
                              for c in self.channels], dtype=float)
            LOG.info("固定标尺：忽略源库 center/scale，仅沿用其 medians")
        else:
            center = np.array([float(d["center"][c]) for c in self.channels])
            scale = np.array([float(d["scale"][c]) for c in self.channels])
            scale = np.where(np.abs(scale) > 1e-8, scale, 1.0)
        self.medians_ = med
        self.scaler_ = ScalerParams(d.get("scaler", "minmax"), center, scale,
                                    int(d.get("n_train_patients", 0)))
        LOG.info("已载入源库参数（%s）：中位数/归一化沿用源库，不做目标库适配",
                 Path(path).name)
        return self

    # ------------------------------------------------------------------ 变换
    def transform(self, long_df: pd.DataFrame, cohort: pd.DataFrame, track: str) -> Dict:
        """产出归一化后的三维立方体与观测掩码（尚未切窗）。"""
        if self.medians_ is None or self.scaler_ is None:
            raise RuntimeError("请先 fit()")

        pids = cohort[PID].tolist()
        if track == "track2b":
            # VitalDB Track 2B：纯术中 12h，t ∈ [-12, 0)，t_end = 0（术毕=入 ICU）
            hour_lo = -self.window_h
            hi = pd.Series(-1.0, index=cohort[PID])
        elif track == "track2":
            # 交接期窗口 t ∈ [-4, +8)：术中末 4h + 入 ICU 后 8h，t_end = +8h
            hour_lo = -4
            hi = pd.Series(float(self.window_h - 4 - 1), index=cohort[PID])   # 网格 -4..7
        else:
            hour_lo = 0
            hi = cohort.set_index(PID)["followup_h"].astype(float).clip(upper=self.max_hours)

        cube, obs_mask, hours = self._build_cube(long_df, hour_lo, hi, pids)
        filled = self._fill(cube)
        scaled = self.scaler_.apply(filled)
        if self.clip_scaled is not None:
            scaled = np.clip(scaled, *self.clip_scaled)
        scaled = np.where(obs_mask | np.isfinite(scaled), scaled, np.nan)
        return {"values": scaled.astype(np.float32), "obs_mask": obs_mask,
                "hours": hours, "pids": pids, "hi": hi.reindex(pids).to_numpy(float)}

    # ------------------------------------------------------------------ 切窗
    def build_windows(self, cube_dict: Dict, cohort: pd.DataFrame, track: str) -> WindowSet:
        values, obs_mask = cube_dict["values"], cube_dict["obs_mask"]
        hours, pids = cube_dict["hours"], cube_dict["pids"]

        meta_cohort = cohort.set_index(PID)
        xs: List[np.ndarray] = []
        rows: List[pd.DataFrame] = []

        for i, pid in enumerate(pids):
            if track == "track2b":
                if values.shape[1] < self.window_h:
                    continue
                win = values[i, :self.window_h, :].T            # (C, T) 全部为术中
                msk = obs_mask[i, :self.window_h, :].T
                obs_frac = float(msk.mean())
                if obs_frac < self.min_observed_frac:
                    continue
                xs.append(win[None, :, :])
                rows.append(self._meta_row(pid, meta_cohort, track, 0.0, obs_frac, obs_frac, msk))
                continue

            if track == "track2":
                if values.shape[1] < self.window_h:
                    continue
                # 网格恰为 12 行：t = -4..7（-4..-1 术中，0..7 ICU），t_end = +8
                win = values[i, :self.window_h, :].T            # (C, T)
                msk = obs_mask[i, :self.window_h, :].T          # (C, T)
                obs_frac = float(msk.mean())
                or_frac = float(msk[:, :4].mean())              # 前 4 步 = 术中末 4h
                if obs_frac < self.min_observed_frac:
                    continue
                xs.append(win[None, :, :])
                rows.append(self._meta_row(pid, meta_cohort, track, 8.0, obs_frac, or_frac, msk))
                continue

            h_i = int(min(cube_dict["hi"][i], hours[-1]))
            if h_i < self.window_h:
                continue
            v = values[i, : h_i + 1, :]
            m = obs_mask[i, : h_i + 1, :]
            # 注意：sliding_window_view 把窗口维度追加在末尾 -> (n, C, T)
            sw = np.lib.stride_tricks.sliding_window_view(v, self.window_h, axis=0)
            swm = np.lib.stride_tricks.sliding_window_view(m, self.window_h, axis=0)
            t_ends = np.arange(self.window_h, h_i + 1, self.step_h)
            idx = t_ends - self.window_h
            if idx.size == 0:
                continue
            wins = sw[idx]                                     # (n, C, T)
            masks = swm[idx]                                   # (n, C, T)
            obs_frac = masks.mean(axis=(1, 2))
            keep = obs_frac >= self.min_observed_frac
            if not keep.any():
                continue
            xs.append(wins[keep].astype(np.float32))           # (n, C, T)
            for k, te in enumerate(t_ends[keep]):
                rows.append(self._meta_row(pid, meta_cohort, track, float(te),
                                           float(obs_frac[keep][k]), np.nan,
                                           masks[keep][k]))

        if not xs:
            raise RuntimeError(
                f"{track} 没有产出任何窗口：检查 followup_h 是否 >= {self.window_h}h、"
                f"观测覆盖率是否 >= {self.min_observed_frac}")

        X = np.concatenate(xs, axis=0).astype(np.float32)
        meta = pd.concat(rows, ignore_index=True)

        # ---- 滚动标签 ----
        meta = self._assign_labels(meta)
        dropped = meta["label_status"].ne("kept")
        stats = {
            "n_raw_windows": int(len(meta)),
            "n_kept": int((~dropped).sum()),
            "dropped_prevalent": int((meta["label_status"] == "prevalent").sum()),
            "dropped_censored": int((meta["label_status"] == "censored").sum()),
            "n_patients": int(meta.loc[~dropped, PID].nunique()),
            "pos_rate": float(meta.loc[~dropped, "y"].mean()) if (~dropped).any() else 0.0,
        }
        meta = meta.loc[~dropped].reset_index(drop=True)
        X = X[~dropped.to_numpy()]

        return WindowSet(X=X, meta=meta, channels=self.channels,
                         window_h=self.window_h, track=track, stats=stats)

    def _meta_row(self, pid, meta_cohort: pd.DataFrame, track: str, t_end: float,
                  obs_frac: float, or_frac: float, mask: np.ndarray) -> pd.DataFrame:
        rec = {PID: pid, "track": track, "t_end_h": t_end,
               "observed_frac": obs_frac, "or_frac": or_frac}
        if pid in meta_cohort.index:
            rec["t_aki_onset"] = meta_cohort.at[pid, "t_aki_onset"]
            for c in ("t_stage2", "t_stage3"):
                if c in meta_cohort.columns:
                    rec[c] = meta_cohort.at[pid, c]
            rec["followup_h"] = meta_cohort.at[pid, "followup_h"]
            for c in ("age", "sex", "icu_type", "baseline_source", "baseline_scr",
                      "in_vitaldb_overlap"):
                if c in meta_cohort.columns:
                    rec[c] = meta_cohort.at[pid, c]
        # 每通道缺失率（填补前）：不占 630 维，单独放 meta。mask 布局为 (C, T)
        for j, ch in enumerate(self.channels):
            rec[f"miss_{ch}"] = float(1.0 - mask[j, :].mean())
        return pd.DataFrame([rec])

    def _assign_labels(self, meta: pd.DataFrame) -> pd.DataFrame:
        """滚动标签：y=1 当且仅当 AKI 发生在 (t_end, t_end+24h]。

        三种丢弃情形：
          prevalent —— 发病已发生（t_end >= onset），按任务书要求截断
          censored  —— 随访不足以覆盖整个视界，无法判定
        """
        t_end = meta["t_end_h"].to_numpy(float)
        col = self.onset_col if self.onset_col in meta.columns else "t_aki_onset"
        onset = pd.to_numeric(meta[col], errors="coerce").to_numpy(float)
        fu = pd.to_numeric(meta["followup_h"], errors="coerce").to_numpy(float)

        positive = (onset > t_end) & (onset <= t_end + self.horizon_h)
        prevalent = onset <= t_end
        # 阴性有两种：全程无 AKI 且随访足够；或 AKI 发生在视界之后（确定阴性）
        negative = ((np.isnan(onset)) & (fu >= t_end + self.horizon_h)) | \
                   ((onset > t_end + self.horizon_h))

        status = np.select([positive, negative, prevalent],
                           ["kept", "kept", "prevalent"], default="censored")
        meta = meta.copy()
        meta["label_status"] = status
        meta["y"] = np.where(positive, 1, np.where(negative, 0, np.nan))
        meta.loc[status != "kept", "y"] = np.nan
        meta["y"] = meta["y"].astype("float32")
        return meta

    # ------------------------------------------------------------------ 便捷
    def run(self, long_df: pd.DataFrame, cohort: pd.DataFrame,
            track: str = "track1", train_patients: Optional[Sequence] = None,
            batch_patients: int = 1200) -> WindowSet:
        """分批构建窗口。全量 eICU/INSPIRE 的长表有数百万行，一次性建三维立方体会 OOM，
        按患者分批后峰值内存与批大小成正比。"""
        if self.medians_ is None:
            train_patients = train_patients or cohort[PID].tolist()
            self.fit(long_df, train_patients, cohort)

        pids = cohort[PID].drop_duplicates().tolist()
        if len(pids) <= batch_patients:
            return self.build_windows(self.transform(long_df, cohort, track), cohort, track)

        xs, metas, agg = [], [], {}
        for i in range(0, len(pids), batch_patients):
            sub = cohort[cohort[PID].isin(pids[i:i + batch_patients])]
            LOG.info("  批次 %d/%d（%d 人）", i // batch_patients + 1,
                     int(np.ceil(len(pids) / batch_patients)), len(sub))
            try:
                ws = self.build_windows(self.transform(long_df, sub, track), sub, track)
            except RuntimeError:
                continue
            xs.append(ws.X)
            metas.append(ws.meta)
            for k, v in ws.stats.items():
                agg[k] = agg.get(k, 0) + int(v)
        if not xs:
            raise RuntimeError(f"{track} 没有产出任何窗口")
        meta = pd.concat(metas, ignore_index=True)
        agg["n_patients"] = int(meta[PID].nunique())
        agg["pos_rate"] = float(meta["y"].mean()) if len(meta) else 0.0
        return WindowSet(X=np.concatenate(xs, axis=0), meta=meta, channels=self.channels,
                         window_h=self.window_h, track=track, stats=agg)


# =============================================================================
# 3. 自检
# =============================================================================
def run_self_test() -> int:
    fails: List[str] = []
    ch = list(CHANNELS)

    # ---- ① 分桶：同小时两次观测取均值 ----
    long = pd.DataFrame({
        PID: [1, 1, 1, 1], HOUR: [0.2, 0.7, 1.1, 1.4],
        VAR: ["HR", "HR", "HR", "HR"], VAL: [80.0, 100.0, 60.0, 70.0]})
    coh = pd.DataFrame({PID: [1], "t_aki_onset": [np.nan], "followup_h": [3.0]})
    pp = TemporalPreprocessor()
    pp.fit(long, [1], coh)
    cube = pp.transform(long, coh, "track1")
    # 用未归一化的立方体校验分桶：反解回原始值
    raw = cube["values"][0, :2, ch.index("HR")] * pp.scaler_.scale[ch.index("HR")] \
        + pp.scaler_.center[ch.index("HR")]
    if abs(raw[0] - 90.0) > 1e-4:
        fails.append(f"分桶: hour0 应为 (80+100)/2=90，实得 {raw[0]:.3f}")

    # ---- ② 生理极值剔除 ----
    long2 = pd.DataFrame({
        PID: [1] * 4, HOUR: [0, 1, 2, 3], VAR: ["HR"] * 4,
        VAL: [80.0, 999.0, 85.0, 88.0]})
    pp2 = TemporalPreprocessor()
    pp2.fit(long2, [1], coh)
    c2 = pp2.transform(long2, coh, "track1")
    hr_idx = ch.index("HR")
    if c2["obs_mask"][0, 1, hr_idx]:
        fails.append("极值剔除: HR=999 应被置为缺失")

    # ---- ③ 因果性：LOCF 不能把未来值搬到过去 ----
    long3 = pd.DataFrame({
        PID: [1] * 6, HOUR: list(range(6)), VAR: ["HR"] * 6,
        VAL: [np.nan, np.nan, np.nan, 120.0, np.nan, np.nan]})
    coh3 = pd.DataFrame({PID: [1], "t_aki_onset": [np.nan], "followup_h": [6.0]})
    pp3 = TemporalPreprocessor()
    pp3.fit(long3, [1], coh3)
    c3 = pp3.transform(long3, coh3, "track1")
    hr = c3["values"][0, :, hr_idx] * pp3.scaler_.scale[hr_idx] + pp3.scaler_.center[hr_idx]
    if not np.allclose(hr[:3], pp3.medians_[hr_idx], atol=1e-4):
        fails.append(f"LOCB 泄露: t<3 应等于训练中位数，实得 {hr[:3]}")
    if abs(hr[5] - 120.0) > 1e-4:
        fails.append(f"LOCF 失效: t=5 应为 120，实得 {hr[5]:.3f}")

    # ---- ④ 训练集中位数不得被测试集污染 ----
    train = pd.DataFrame({PID: [1, 1], HOUR: [0, 1], VAR: ["HR", "HR"], VAL: [80.0, 82.0]})
    test = pd.DataFrame({PID: [2, 2], HOUR: [0, 1], VAR: ["HR", "HR"], VAL: [200.0, 205.0]})
    coh4 = pd.DataFrame({PID: [1, 2], "t_aki_onset": [np.nan, np.nan],
                         "followup_h": [2.0, 2.0]})
    pp4 = TemporalPreprocessor(min_observed_frac=0.0)
    pp4.fit(pd.concat([train, test]), [1], coh4)
    if abs(pp4.medians_[hr_idx] - 81.0) > 1e-6:
        fails.append(f"中位数污染: 应仅用训练集(=81)，实得 {pp4.medians_[hr_idx]}")

    # ---- ⑤ 窗口与标签 ----
    vit, coh5 = synthesize_demo(n_patients=12, seed=7)
    pp5 = TemporalPreprocessor(min_observed_frac=0.3)
    train_ids = coh5[PID].tolist()[:8]
    ws = pp5.run(vit, coh5, "track1", train_patients=train_ids)
    if ws.X.shape[1:] != (10, 12):
        fails.append(f"窗口形状应为 (N,10,12)，实得 {ws.X.shape[1:]}")
    if not np.isfinite(ws.X).all():
        fails.append("窗口张量存在 NaN：插补或归一化有漏洞")
    # 发病后窗口必须全部丢弃
    bad = ws.meta[(ws.meta["t_aki_onset"].notna()) &
                  (ws.meta["t_end_h"] >= ws.meta["t_aki_onset"])]
    if len(bad):
        fails.append(f"截断失效: {len(bad)} 个发病后窗口未被丢弃")
    # 阳性标签必须落在 (t_end, t_end+24]
    pos = ws.meta[ws.meta["y"] == 1]
    if len(pos):
        ok = ((pos["t_aki_onset"] > pos["t_end_h"]) &
              (pos["t_aki_onset"] <= pos["t_end_h"] + 24)).all()
        if not ok:
            fails.append("阳性标签越界：存在不在 (t_end, t_end+24] 内的正样本")

    # ---- ⑥ Track 2 ----
    ws2 = pp5.run(vit, coh5, "track2", train_patients=train_ids)
    if ws2.X.shape[1:] != (10, 12):
        fails.append(f"Track2 形状应为 (N,10,12)，实得 {ws2.X.shape[1:]}")
    if (ws2.meta["t_end_h"] != 8).any():
        fails.append("Track2 的 t_end 应恒为 +8h（窗口 t∈[-4,+8)）")
    # 无术中记录时，or_frac 必须为 0，obs_frac 约等于 8/12（只有 ICU 段）
    vit_no_or = vit[vit[HOUR] >= 0]
    ws2b = pp5.run(vit_no_or, coh5, "track2", train_patients=train_ids)
    if not np.allclose(ws2b.meta["or_frac"], 0.0):
        fails.append("Track2 无术中数据时 or_frac 应为 0")
    if not (0.5 < ws2b.meta["observed_frac"].median() < 0.8):
        fails.append(f"Track2 仅有 ICU 段时 obs_frac 中位应约 8/12，"
                     f"实得 {ws2b.meta['observed_frac'].median():.3f}")

    # ---- ⑦ 固定生理标尺：端点精确映射到 0/1，越界值不参与缩放 ----
    pfx = TemporalPreprocessor(scaler="fixed", min_observed_frac=0.2)
    pfx.fit(vit, train_ids, coh)
    j = pfx.channels.index("HR")
    lo, hi = FIXED_BOUNDS["HR"]
    z = (np.array([lo, hi]) - pfx.scaler_.center[j]) / pfx.scaler_.scale[j]
    if not (abs(z[0]) < 1e-9 and abs(z[1] - 1.0) < 1e-9):
        fails.append(f"固定标尺端点应为 0/1，实得 {z[0]:.4f}/{z[1]:.4f}")

    # ---- ⑦ 未来篡改不影响过去窗口 ----
    vit2 = vit.copy()
    m = (vit2[PID] == 1) & (vit2[HOUR] >= 30)
    vit2.loc[m, VAL] = vit2.loc[m, VAL] * 1.5 + 20
    ws_b = pp5.run(vit2, coh5, "track1", train_patients=train_ids)
    a = ws.meta[(ws.meta[PID] == 1) & (ws.meta["t_end_h"] == 20)].index
    b = ws_b.meta[(ws_b.meta[PID] == 1) & (ws_b.meta["t_end_h"] == 20)].index
    if len(a) and len(b):
        d = np.abs(ws.X[a[0]] - ws_b.X[b[0]]).max()
        if d > 1e-5:
            fails.append(f"未来泄露: 篡改 t>=30 后 t_end=20 的窗口变化 {d:.4f}")

    print("=" * 70)
    print("模块2 自检：分桶 / 极值 / 因果插补 / 训练集隔离 / 滑窗标签")
    print("=" * 70)
    for f in fails:
        print("  [FAIL]", f)
    print(f"  Track1: {ws.stats}")
    print(f"  Track2: {ws2.stats}")
    print(f"  失败 {len(fails)} 项")
    return 1 if fails else 0


# =============================================================================
# 4. CLI
# =============================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="双轨滑窗预处理器")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--demo", action="store_true", help="用合成数据跑通并落盘")
    ap.add_argument("--vitals")
    ap.add_argument("--cohort")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--track", choices=["track1", "track2", "track2b", "both"], default="both")
    ap.add_argument("--load-params", default=None,
                    help="载入源库 preprocess_params.json（跨库外部验证用）")
    ap.add_argument("--stage-target", type=int, choices=[1, 2], default=1,
                    help="1=Stage>=1 全分级；2=Stage>=2 中重度（主结局）")
    ap.add_argument("--scaler", choices=["minmax", "robust", "zscore", "fixed"],
                    default="fixed",
                    help="默认 fixed：跨库统一生理极值标尺；minmax 为训练集经验标尺（仅库内分析）")
    ap.add_argument("--window-h", type=int, default=12)
    ap.add_argument("--horizon-h", type=int, default=24)
    ap.add_argument("--min-observed-frac", type=float, default=0.5)
    ap.add_argument("--unscaled", default="",
                    help="不归一化的通道，逗号分隔（如 MEWS,SI 以保留临床量纲）")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    if args.self_test:
        return run_self_test()

    if args.demo:
        LOG.info("生成合成演示数据（非真实患者，仅用于跑通流水线）")
        vitals, cohort = synthesize_demo(n_patients=60, seed=2026)
    else:
        if not args.vitals or not args.cohort:
            ap.error("需要 --vitals 与 --cohort（或使用 --demo / --self-test）")
        vitals = pd.read_csv(args.vitals)
        cohort = pd.read_csv(args.cohort)

    out = ensure_dir(args.out)
    pp = TemporalPreprocessor(window_h=args.window_h, horizon_h=args.horizon_h,
                              scaler=args.scaler,
                              min_observed_frac=args.min_observed_frac,
                              unscaled_channels=[s.strip() for s in args.unscaled.split(",")
                                                 if s.strip()],
                              stage_target=args.stage_target)
    pids = cohort[PID].drop_duplicates().tolist()
    train_ids = []
    if args.load_params:
        pp.load_params(args.load_params)          # 外部验证：沿用源库参数，不 fit
    else:
        rng = np.random.default_rng(42)
        rng.shuffle(pids)
        train_ids = pids[: max(1, int(0.7 * len(pids)))]
        pp.fit(vitals, train_ids, cohort)
    save_json({"scaler": pp.scaler, "medians": dict(zip(pp.channels, pp.medians_.tolist())),
               "center": dict(zip(pp.channels, pp.scaler_.center.tolist())),
               "scale": dict(zip(pp.channels, pp.scaler_.scale.tolist())),
               "n_train_patients": len(train_ids)}, out / "preprocess_params.json")

    tracks = ["track1", "track2"] if args.track == "both" else [args.track]
    for tr in tracks:
        try:
            ws = pp.run(vitals, cohort, tr)
        except RuntimeError as exc:
            LOG.warning("%s 跳过：%s", tr, exc)
            continue
        paths = ws.save(out)
        print(f"[{tr}] X={ws.X.shape}  {ws.stats}")
        print(f"       -> {paths['X']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
