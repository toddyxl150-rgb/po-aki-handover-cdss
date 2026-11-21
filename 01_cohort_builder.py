#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_cohort_builder.py — 三库真实队列提取、KDIGO 动态打标与契约校验
================================================================

数据源
------
eICU-CRD  : CSV 目录（PhysioNet 2.0 解压后的表）
INSPIRE   : CSV 目录（operations / vitals / ward_vitals / labs）
VitalDB   : CSV 目录（clinical_data / lab_data；术中波形需 vitaldb 包）

输出
----
outputs/cohort_<db>.csv           患者级队列（含 t_aki_onset、baseline_source）
outputs/vitals_<db>.csv.gz        逐小时生命体征长表（Track 1/2 输入）
outputs/cohort_attrition_real.csv 三库纳排瀑布表
outputs/contract_check_report.txt 数据契约校验日志

baseline_source 编码（用户拍板）
-------------------------------
1 = 真实术前/入ICU前基线 SCr，取 [-180d, 0h)
2 = 入ICU早期首次 SCr，取 [0h, +6h]
3 = 估算/最低值基线（术前 7d 最低值，或 MDRD 反推）

时间轴
------
所有 hour 相对 ICU 入科（eICU: unit admit；INSPIRE: icuin_time；VitalDB: caseend）。
负值 = 入 ICU 前（手术室 / 麻醉恢复）。

用法
----
    python 01_cohort_builder.py --db all --out outputs
    python 01_cohort_builder.py --discover
    python 01_cohort_builder.py --check --vitals outputs/vitals_eicu.csv.gz \
        --cohort outputs/cohort_eicu.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdss_common import ensure_dir  # noqa: E402

LOG = logging.getLogger("cdss.cohort")

PID, HOUR, VAR, VAL = "patient_id", "hour", "variable", "value"

COHORT_REQUIRED = ["patient_id", "t_aki_onset", "followup_h", "baseline_source"]
VITALS_REQUIRED = ["patient_id", "hour", "variable", "value"]

BSRC_PREOP, BSRC_ICU_FIRST, BSRC_ESTIMATED = 1, 2, 3
BSRC_LABEL = {1: "pre_icu[-180d,0h)", 2: "icu_first[0h,6h]", 3: "estimated_lowest"}


# =============================================================================
# 0. 配置
# =============================================================================
@dataclass
class CohortConfig:
    age_min: int = 18
    age_max: int = 99
    los_floor_h: float = 24.0
    pre_icu_guard_h: float = 12.0
    baseline_lookback_d: int = 180
    icu_first_window_h: float = 6.0
    max_followup_h: float = 168.0
    uo_min_coverage: float = 0.75
    mdrd_egfr: float = 75.0
    vital_chunk: int = 2_000_000
    limit: Optional[int] = None
    vitaldb_horizon_h: float = 168.0
    vitaldb_root: str = ""
    inspire_root: str = ""
    vitals_sample: Optional[int] = None
    vital_workers: int = 4


# =============================================================================
# 1. KDIGO 引擎（纯函数）
# =============================================================================
def _scr_events(scr_h: np.ndarray, scr_v: np.ndarray, baseline: float
                ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if scr_h.size == 0 or not np.isfinite(baseline) or baseline <= 0:
        return None, None, None
    order = np.argsort(scr_h, kind="mergesort")
    scr_h, scr_v = scr_h[order], scr_v[order]
    t1 = t2 = t3 = None
    for i in range(scr_h.size):
        h, v = float(scr_h[i]), float(scr_v[i])
        if not np.isfinite(v) or v <= 0:
            continue
        prior7 = scr_v[(scr_h >= h - 168) & (scr_h < h)]
        ref7 = min([baseline] + prior7.tolist())
        prior48 = scr_v[(scr_h >= h - 48) & (scr_h < h)]
        delta48 = (v - float(prior48.min())) if prior48.size else np.nan
        ratio = v / ref7 if ref7 > 0 else np.nan

        stage = 0
        if np.isfinite(ratio) and ratio >= 3.0:
            stage = 3
        elif v >= 4.0 and ((np.isfinite(delta48) and delta48 >= 0.3) or
                           (np.isfinite(ratio) and ratio >= 1.5)):
            stage = 3
        if stage < 2 and np.isfinite(ratio) and ratio >= 2.0:
            stage = 2
        if stage < 1 and ((np.isfinite(ratio) and ratio >= 1.5) or
                          (np.isfinite(delta48) and delta48 >= 0.3)):
            stage = 1
        if stage >= 1 and t1 is None:
            t1 = h
        if stage >= 2 and t2 is None:
            t2 = h
        if stage >= 3 and t3 is None:
            t3 = h
    return t1, t2, t3


def _uo_events(uo_h: np.ndarray, uo_ml: np.ndarray, weight: float, n_hours: int,
               min_cov: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if weight is None or not np.isfinite(weight) or weight <= 0 or uo_h.size == 0 or n_hours < 6:
        return None, None, None
    ml = np.zeros(n_hours); cnt = np.zeros(n_hours)
    idx = np.clip(np.floor(uo_h).astype(int), 0, n_hours - 1)
    np.add.at(ml, idx, np.nan_to_num(uo_ml, nan=0.0))
    np.add.at(cnt, idx, 1.0)
    cs = np.concatenate([[0.0], np.cumsum(ml)])
    cc = np.concatenate([[0.0], np.cumsum(cnt)])
    hits = {1: np.zeros(n_hours, bool), 2: np.zeros(n_hours, bool), 3: np.zeros(n_hours, bool)}
    for w, thr, st in ((6, 0.5, 1), (12, 0.5, 2), (24, 0.3, 3)):
        if n_hours < w:
            continue
        for end in range(w - 1, n_hours):
            s = cs[end + 1] - cs[end + 1 - w]
            cov = (cc[end + 1] - cc[end + 1 - w]) / w
            if cov < min_cov:
                continue
            if s / (weight * w) < thr:
                hits[st][end] = True
    if n_hours >= 12:
        for end in range(11, n_hours):
            s = cs[end + 1] - cs[end + 1 - 12]
            cov = (cc[end + 1] - cc[end + 1 - 12]) / 12
            if cov >= min_cov and s <= 0:
                hits[3][end] = True

    def first(mask):
        ix = np.flatnonzero(mask)
        return float(ix[0]) if ix.size else None
    return first(hits[1]), first(hits[2]), first(hits[3])


def mdrd_scr(age: float, sex: str, egfr: float = 75.0) -> float:
    if not np.isfinite(age) or age <= 0:
        return np.nan
    f = 0.742 if str(sex).upper().startswith(("F", "女")) else 1.0
    return float((175.0 * age ** -0.203 * f / egfr) ** (1 / 1.154))


def label_patient(scr: pd.DataFrame, uo: pd.DataFrame, rrt: pd.DataFrame,
                  baseline: float, weight: float, los_h: float,
                  cfg: CohortConfig) -> Dict[str, float]:
    # 有效随访上限：LOS 与 7d 上限取小。所有判据（SCr / 尿量 / RRT）必须共用它，
    # 否则任一判据在窗后触发都会让 t_aki_onset > followup_h，下游会把整段窗口误判为已发病。
    eff_los = min(float(los_h), float(cfg.max_followup_h))
    n_h = int(max(math.floor(eff_los), 1))
    scr = scr[(scr["hour"] >= 0) & (scr["hour"] <= eff_los)]
    rrt = rrt[(rrt["hour"] >= 0) & (rrt["hour"] <= eff_los)]
    s1, s2, s3 = _scr_events(scr["hour"].to_numpy(float), scr["value"].to_numpy(float), baseline)
    u1, u2, u3 = _uo_events(uo["hour"].to_numpy(float), uo["ml"].to_numpy(float),
                            weight, n_h, cfg.uo_min_coverage)
    r = float(rrt["hour"].min()) if len(rrt) else None
    c1 = [t for t in (s1, u1, r) if t is not None]
    c2 = [t for t in (s2, u2, r) if t is not None]
    c3 = [t for t in (s3, u3, r) if t is not None]
    t1 = min(c1) if c1 else None
    t2 = min(c2) if c2 else None
    t3 = min(c3) if c3 else None
    stage = 3 if t3 is not None else 2 if t2 is not None else 1 if t1 is not None else 0
    return {"t_aki_onset": (t1 if t1 is not None else np.nan), "max_stage": stage,
            "t_stage1": t1 if t1 is not None else np.nan,
            "t_stage2": t2 if t2 is not None else np.nan,
            "t_stage3": t3 if t3 is not None else np.nan,
            "crit_scr": int(s1 is not None), "crit_uo": int(u1 is not None),
            "crit_rrt": int(r is not None)}


def pick_baseline(pre: pd.Series, icu_first: float, age: float, sex: str,
                  cfg: CohortConfig) -> Tuple[float, int]:
    if len(pre):
        return float(pre.iloc[-1]), BSRC_PREOP
    if np.isfinite(icu_first) and icu_first > 0:
        return float(icu_first), BSRC_ICU_FIRST
    est = mdrd_scr(age, sex, cfg.mdrd_egfr)
    if np.isfinite(est) and est > 0:
        return float(est), BSRC_ESTIMATED
    return np.nan, 0


# =============================================================================
# 2. 通用工具
# =============================================================================
def _iter_chunks(path: Path, usecols: Optional[Sequence[str]], chunk: int
                 ) -> Iterator[pd.DataFrame]:
    for ch in pd.read_csv(path, usecols=usecols, chunksize=chunk,
                          low_memory=False, compression="infer"):
        yield ch


def _gather(path: Path, usecols: Sequence[str], chunk: int,
            fn: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    out: List[pd.DataFrame] = []
    for ch in _iter_chunks(path, usecols, chunk):
        r = fn(ch)
        if len(r):
            out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=list(usecols))


_VITALDB_ID_CACHE: Dict[str, set] = {}


def _vitaldb_caseids(cfg: CohortConfig) -> set:
    """VitalDB 全部 caseid（用于 INSPIRE 的重叠标记）。"""
    root = cfg.vitaldb_root
    if not root:
        return set()
    if root in _VITALDB_ID_CACHE:
        return _VITALDB_ID_CACHE[root]
    p = Path(root) / "clinical_data.csv"
    ids = set()
    if p.exists():
        ids = set(pd.read_csv(p, usecols=["caseid"])["caseid"].astype("int64").tolist())
    _VITALDB_ID_CACHE[root] = ids
    return ids


_INSPIRE_ID_CACHE: Dict[str, set] = {}


def _inspire_caseids(cfg: CohortConfig) -> set:
    root = cfg.inspire_root
    if not root or root in _INSPIRE_ID_CACHE:
        return _INSPIRE_ID_CACHE.get(root, set())
    p = Path(root) / "operations.csv.gz"
    ids = set()
    if p.exists():
        d = pd.read_csv(p, usecols=["case_id"])
        ids = set(pd.to_numeric(d["case_id"], errors="coerce").dropna().astype("int64").tolist())
    _INSPIRE_ID_CACHE[root] = ids
    return ids


# VitalDB 术中波形轨道 -> 本项目通道
VITAL_TRACKS = {
    "Solar8000/HR": "HR", "Solar8000/ART_SBP": "SBP", "Solar8000/ART_DBP": "DBP",
    "Solar8000/PLETH_SPO2": "SpO2", "Solar8000/RR_CO2": "RR", "Solar8000/BT": "Temp",
}


def _parse_one_vital(task):
    """解析单个 .vital 文件，返回 (caseid, long_df)。进程池要求函数在模块级。"""
    path, caseid, caseend_sec = task
    try:
        import vitaldb
        names = list(VITAL_TRACKS)
        vf = vitaldb.read_vital(str(path), names)
        df = vf.to_pandas(names, 60)          # 60s 重采样
    except Exception:  # noqa: BLE001
        return caseid, None
    if df is None or len(df) == 0:
        return caseid, None
    hour = (df.index.to_numpy(float) * 60.0 - float(caseend_sec)) / 3600.0
    keep = (hour >= -24) & (hour < 0)         # 只保留术中段
    if not keep.any():
        return caseid, None
    sub = df.loc[keep]
    recs = []
    for track, ch in VITAL_TRACKS.items():
        if track not in sub.columns:
            continue
        v = pd.to_numeric(sub[track], errors="coerce").to_numpy(dtype=float, copy=True)
        v[v < 0] = np.nan                     # VitalDB 用负值表示"未测量"
        recs.append(pd.DataFrame({PID: caseid, HOUR: hour[keep], VAR: ch, VAL: v}))
    return caseid, (pd.concat(recs, ignore_index=True) if recs else None)


@dataclass
class BuildResult:
    db: str
    cohort: pd.DataFrame
    vitals: pd.DataFrame
    attrition: List[Dict]
    notes: List[str] = field(default_factory=list)


class _Tracker:
    def __init__(self, db: str):
        self.db = db
        self.rows: List[Dict] = []
        self.initial = 0

    def start(self, n: int, note: str = ""):
        self.initial = int(n)
        self.rows.append({"db": self.db, "step_order": 0, "step": "初始人群",
                          "n_remaining": int(n), "n_excluded": 0, "pct_of_initial": 100.0,
                          "note": note})

    def step(self, name: str, before: int, after: int, note: str = ""):
        self.rows.append({"db": self.db, "step_order": len(self.rows), "step": name,
                          "n_remaining": int(after), "n_excluded": int(before - after),
                          "pct_of_initial": round(100 * after / max(self.initial, 1), 2),
                          "note": note})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# =============================================================================
# 3. eICU-CRD
# =============================================================================
class EicuAdapter:
    name = "eicu"

    def __init__(self, root: str, cfg: CohortConfig):
        self.root = Path(root)
        self.cfg = cfg

    def _f(self, name: str) -> Path:
        for ext in (".csv.gz", ".csv"):
            p = self.root / f"{name}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"{self.root} 缺少 {name}.csv(.gz)")

    def discover(self) -> str:
        lines = [f"[eICU] root={self.root}"]
        for t in ("patient", "lab", "intakeOutput", "treatment", "diagnosis",
                  "pasthistory", "apachePredVar", "vitalPeriodic"):
            p = self.root / f"{t}.csv.gz"
            if p.exists():
                lines.append(f"  {t:<15} cols={list(pd.read_csv(p, nrows=2).columns)}")
            else:
                lines.append(f"  {t:<15} !! 缺失")
        return "\n".join(lines)

    def build(self) -> BuildResult:
        cfg, tr = self.cfg, _Tracker(self.name)
        pat = pd.read_csv(self._f("patient"), low_memory=False)
        tr.start(len(pat), "eICU patient 表全部 ICU stay")

        units = pd.DataFrame({
            PID: pat["patientunitstayid"].astype(str),
            "subject_id": pat["uniquepid"].astype(str),
            "age": pd.to_numeric(pat["age"].astype(str).str.replace("> 89", "90", regex=False),
                                 errors="coerce"),
            "sex": pat["gender"].astype(str),
            "weight_kg": pd.to_numeric(pat.get("admissionweight"), errors="coerce"),
            "icu_type": pat["unittype"].astype(str),
            "admitsource": pat["unitadmitsource"].astype(str),
            "los_hours": pd.to_numeric(pat["unitdischargeoffset"], errors="coerce") / 60.0,
            "visit_no": pd.to_numeric(pat.get("unitvisitnumber"), errors="coerce"),
            "died_icu": pat["unitdischargestatus"].astype(str).str.lower()
                          .str.contains("death|expired", na=False),
            "died_hosp": pat["hospitaldischargestatus"].astype(str).str.lower()
                           .str.contains("death|expired", na=False),
        })
        n0 = len(units)
        units = units[units["age"].between(cfg.age_min, cfg.age_max)]
        tr.step(f"年龄 {cfg.age_min}-{cfg.age_max}", n0, len(units), "'>89' 按 90 计")

        surg_src = units["admitsource"].str.contains(
            r"operating room|pacu|post-anesthesia|recovery", case=False, na=False)
        surg = surg_src.copy()
        rule = pd.Series(np.where(surg_src, "admitsource", "none"), index=units.index)

        try:
            ap = pd.read_csv(self._f("apachePredVar"), usecols=["patientunitstayid", "postop"],
                             low_memory=False)
            ap["patientunitstayid"] = ap["patientunitstayid"].astype(str)
            postop = ap.drop_duplicates("patientunitstayid").set_index("patientunitstayid")["postop"]
            m = units[PID].map(postop).fillna(0).astype(float).eq(1)
            surg |= m
            rule = rule.mask(m & (rule == "none"), "apache_postop")
        except Exception as exc:  # noqa: BLE001
            LOG.warning("eICU apachePredVar 不可用：%s", exc)

        trt = _gather(self._f("treatment"),
                      ["patientunitstayid", "treatmentoffset", "treatmentstring"], cfg.vital_chunk,
                      lambda d: d[d["treatmentstring"].astype(str).str.contains(
                          r"surg|operat|resect|anastomo|graft|repair|fixation|excision",
                          case=False, na=False)])
        trt["patientunitstayid"] = trt["patientunitstayid"].astype(str)
        surg_trt = trt[pd.to_numeric(trt["treatmentoffset"], errors="coerce") <= 24 * 60]
        m_trt = units[PID].isin(set(surg_trt["patientunitstayid"]))
        surg |= m_trt
        rule = rule.mask(m_trt & (rule == "none"), "treatment_record")

        n1 = len(units)
        units = units[surg.to_numpy()]
        tr.step("外科手术后入 ICU", n1, len(units), "OR/PACU ∪ postop ∪ 术后24h手术记录")
        units["surgical_rule"] = rule[surg.to_numpy()].to_numpy()

        n2 = len(units)
        units = units[units["los_hours"] >= cfg.los_floor_h]
        tr.step(f"ICU LOS >= {cfg.los_floor_h:g}h", n2, len(units),
                "仅最低随访门槛，不按 145h 剔除阴性")

        n3 = len(units)
        units = units[units["visit_no"].fillna(1) <= 1]
        tr.step("仅首次 ICU 入住", n3, len(units), "unitvisitnumber <= 1")

        if cfg.limit:
            n_before = len(units)
            units = units.head(cfg.limit)
            # 运行上限必须显式记入瀑布表，否则 40,037 -> 3,603 会出现无法解释的数字断层
            tr.step(f"运行上限抽样（--limit {cfg.limit}，可复现调试用）",
                    n_before, len(units),
                    "按 unitvisitnumber 排序后截断；论文全量分析须去掉 --limit")
        keep = set(units[PID])
        LOG.info("[eICU] 候选 stay %d", len(keep))

        ckd_ids = set()
        try:
            ph = _gather(self._f("pasthistory"),
                         ["patientunitstayid", "pasthistorypath", "pasthistoryvalue"], cfg.vital_chunk,
                         lambda d: d[(d["pasthistorypath"].astype(str) + " " +
                                      d["pasthistoryvalue"].astype(str)).str.contains(
                             r"chronic kidney|ckd|end stage renal|esrd|dialysis", case=False, na=False)])
            ckd_ids |= set(ph["patientunitstayid"].astype(str)) & keep
        except Exception as exc:  # noqa: BLE001
            LOG.warning("eICU pasthistory 不可用：%s", exc)
        try:
            dx = _gather(self._f("diagnosis"),
                         ["patientunitstayid", "diagnosisstring", "icd9code"], cfg.vital_chunk,
                         lambda d: d[(d["diagnosisstring"].astype(str).str.contains(
                             r"chronic kidney|end stage renal|esrd", case=False, na=False)) |
                             (d["icd9code"].astype(str).str.startswith(("5855", "5856", "V4511")))])
            ckd_ids |= set(dx["patientunitstayid"].astype(str)) & keep
        except Exception as exc:  # noqa: BLE001
            LOG.warning("eICU diagnosis 不可用：%s", exc)

        n4 = len(units)
        units = units[~units[PID].isin(ckd_ids)]
        tr.step("排除术前 CKD5/ESRD/透析", n4, len(units), f"命中 {len(ckd_ids)}")

        lab = _gather(self._f("lab"),
                      ["patientunitstayid", "labresultoffset", "labname", "labresult"], cfg.vital_chunk,
                      lambda d: d[d["labname"].astype(str).str.contains("creatinine", case=False, na=False)])
        lab = lab[lab["patientunitstayid"].astype(str).isin(keep)]
        scr = pd.DataFrame({
            PID: lab["patientunitstayid"].astype(str),
            HOUR: pd.to_numeric(lab["labresultoffset"], errors="coerce") / 60.0,
            "value": pd.to_numeric(lab["labresult"], errors="coerce"),
        }).dropna()
        scr = scr[(scr["value"] > 0) & (scr["value"] < 30)]
        LOG.info("[eICU] 肌酐记录 %d 条", len(scr))

        io = _gather(self._f("intakeOutput"),
                     ["patientunitstayid", "intakeoutputoffset", "cellpath", "celllabel",
                      "cellvaluenumeric"], cfg.vital_chunk,
                     lambda d: d[(d["cellpath"].astype(str) + " " + d["celllabel"].astype(str))
                                 .str.contains("urine", case=False, na=False)])
        io = io[io["patientunitstayid"].astype(str).isin(keep)]
        uo = pd.DataFrame({
            PID: io["patientunitstayid"].astype(str),
            HOUR: pd.to_numeric(io["intakeoutputoffset"], errors="coerce") / 60.0,
            "ml": pd.to_numeric(io["cellvaluenumeric"], errors="coerce"),
        }).dropna()
        uo = uo[(uo["ml"] >= 0) & (uo["ml"] < 5000) & (uo[HOUR] >= 0)]
        LOG.info("[eICU] 尿量记录 %d 条", len(uo))

        rrt_raw = _gather(self._f("treatment"),
                          ["patientunitstayid", "treatmentoffset", "treatmentstring"], cfg.vital_chunk,
                          lambda d: d[d["treatmentstring"].astype(str).str.contains(
                              r"dialysis|crrt|renal replacement|ultrafiltration", case=False, na=False)])
        rrt = pd.DataFrame({
            PID: rrt_raw["patientunitstayid"].astype(str),
            HOUR: pd.to_numeric(rrt_raw["treatmentoffset"], errors="coerce") / 60.0,
        }).dropna()
        rrt = rrt[(rrt[HOUR] >= 0) & rrt[PID].isin(keep)]

        scr_by = {k: v.sort_values(HOUR) for k, v in scr.groupby(PID)}
        uo_by = {k: v for k, v in uo.groupby(PID)}
        rrt_by = {k: v for k, v in rrt.groupby(PID)}

        base_rows = []
        for _, u in units.iterrows():
            s = scr_by.get(u[PID], scr.iloc[0:0])
            pre_all = s[s[HOUR] < 0]
            pre = pre_all[pre_all[HOUR] >= -cfg.baseline_lookback_d * 24]
            first = s[(s[HOUR] >= 0) & (s[HOUR] <= cfg.icu_first_window_h)]["value"]
            b, src = pick_baseline(pre["value"],
                                   float(first.iloc[0]) if len(first) else np.nan,
                                   u["age"], u["sex"], cfg)
            pre_aki = 0
            if len(pre_all) and np.isfinite(b) and b > 0:
                near = pre_all[pre_all[HOUR] >= -cfg.pre_icu_guard_h]["value"]
                if len(near) and near.max() / b >= 1.5:
                    pre_aki = 1
            base_rows.append({PID: u[PID], "baseline_scr": b, "baseline_source": src,
                              "pre_icu_aki": pre_aki})
        units = units.merge(pd.DataFrame(base_rows), on=PID, how="left")

        n5 = len(units)
        units = units[units["baseline_scr"].notna() & (units["baseline_scr"] > 0)]
        tr.step("基线 SCr 可得", n5, len(units), "1=术前 2=入科首次 3=估算")

        n6 = len(units)
        units = units[(units["pre_icu_aki"] == 0) &
                      ~(units["died_icu"] & (units["los_hours"] <= cfg.pre_icu_guard_h))]
        tr.step(f"排除入 ICU 前 {cfg.pre_icu_guard_h:g}h 内 AKI/死亡", n6, len(units), "")

        LOG.info("[eICU] KDIGO 打标 %d 人 ...", len(units))
        rows = []
        for _, u in units.iterrows():
            s = scr_by.get(u[PID], scr.iloc[0:0])
            rows.append({**u.to_dict(),
                         **label_patient(s[s[HOUR] >= 0], uo_by.get(u[PID], uo.iloc[0:0]),
                                         rrt_by.get(u[PID], rrt.iloc[0:0]),
                                         float(u["baseline_scr"]), float(u["weight_kg"]),
                                         float(u["los_hours"]), cfg)})
        cohort = pd.DataFrame(rows)
        tr.step("最终纳入（PO-AKI 可判定）", len(cohort), len(cohort), "")

        # vitalPeriodic 全量约 1.4 亿行，按"全部阳性 + 随机阴性"分层抽样，
        # 并在瀑布表里显式记一步，避免出现隐式数字断层。
        vit_ids = set(cohort[PID])
        if cfg.vitals_sample and len(vit_ids) > cfg.vitals_sample:
            pos = set(cohort.loc[cohort["t_aki_onset"].notna(), PID])
            neg = sorted(vit_ids - pos)
            rng = np.random.default_rng(42)
            keep_neg = set(rng.choice(neg, max(cfg.vitals_sample - len(pos), 1), replace=False))
            vit_ids = pos | keep_neg
            tr.step("生命体征抽样（保留全部阳性 + 随机阴性，seed=42）",
                    len(cohort), len(vit_ids),
                    f"vitalPeriodic 全量过大；窗口仅覆盖抽样患者，结局统计仍为全量")
        LOG.info("[eICU] 抽取 vitalPeriodic（%d 人）...", len(vit_ids))
        vp = _gather(self._f("vitalPeriodic"),
                     ["patientunitstayid", "observationoffset", "temperature", "sao2", "heartrate",
                      "respiration", "systemicsystolic", "systemicdiastolic"], cfg.vital_chunk,
                     lambda d: d[d["patientunitstayid"].astype(str).isin(vit_ids)])
        vp[PID] = vp["patientunitstayid"].astype(str)
        vp["hour"] = pd.to_numeric(vp["observationoffset"], errors="coerce") / 60.0
        vmap = {"heartrate": "HR", "systemicsystolic": "SBP", "systemicdiastolic": "DBP",
                "respiration": "RR", "sao2": "SpO2", "temperature": "Temp"}
        long = vp.melt(id_vars=[PID, "hour"], value_vars=list(vmap),
                       var_name="variable", value_name=VAL)
        long["variable"] = long["variable"].map(vmap)
        long["src"] = "arterial"

        # ---- 无创袖带血压（vitalAperiodic）----
        # 术后病房多数患者只有 NIBP，没有动脉置管。只读 vitalPeriodic 会让
        # SBP/DBP 大面积缺失，进而使 SHAP 前三的血压类特征变成中位数填出来的假值。
        try:
            va = _gather(self._f("vitalAperiodic"),
                         ["patientunitstayid", "observationoffset",
                          "noninvasivesystolic", "noninvasivediastolic"], cfg.vital_chunk,
                         lambda d: d[d["patientunitstayid"].astype(str).isin(vit_ids)])
            va[PID] = va["patientunitstayid"].astype(str)
            va["hour"] = pd.to_numeric(va["observationoffset"], errors="coerce") / 60.0
            vmap_n = {"noninvasivesystolic": "SBP", "noninvasivediastolic": "DBP"}
            ln = va.melt(id_vars=[PID, "hour"], value_vars=list(vmap_n),
                         var_name="variable", value_name=VAL)
            ln["variable"] = ln["variable"].map(vmap_n)
            ln["src"] = "nibp"
            LOG.info("[eICU] 无创血压记录 %d 条", int(ln[VAL].notna().sum()))
            long = pd.concat([long, ln], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("[eICU] vitalAperiodic 不可用：%s", exc)

        long = long.dropna(subset=["hour", VAL])
        long = long[long[HOUR] >= 0]

        # 同一小时优先有创动脉压，缺失时才用无创（与 INSPIRE._coalesce_bp 同口径）
        bp = long[long[VAR].isin(["SBP", "DBP"])].copy()
        other = long[~long[VAR].isin(["SBP", "DBP"])]
        if len(bp):
            bp["hb"] = np.floor(bp[HOUR]).astype(int)
            has_art = (bp[bp["src"] == "arterial"].groupby([PID, VAR, "hb"]).size()
                       .rename("n_art").reset_index())
            bp = bp.merge(has_art, on=[PID, VAR, "hb"], how="left")
            bp = bp[~((bp["src"] == "nibp") & (bp["n_art"].fillna(0) > 0))]
            LOG.info("[eICU] 血压合并后：有创 %d 条，无创保留 %d 条",
                     int((bp["src"] == "arterial").sum()), int((bp["src"] == "nibp").sum()))
        long = pd.concat([bp[[PID, HOUR, VAR, VAL]], other[[PID, HOUR, VAR, VAL]]],
                         ignore_index=True)

        return BuildResult(self.name, cohort, long[[PID, HOUR, VAR, VAL]], tr.rows)


# =============================================================================
# 4. INSPIRE
# =============================================================================
INSPIRE_VITAL_MAP = {
    "art_sbp": ("SBP", "art"), "art_dbp": ("DBP", "art"),
    "nibp_sbp": ("SBP", "nibp"), "nibp_dbp": ("DBP", "nibp"),
    "hr": ("HR", None), "rr": ("RR", None), "spo2": ("SpO2", None), "bt": ("Temp", None),
}


class InspireAdapter:
    name = "inspire"

    def __init__(self, root: str, cfg: CohortConfig):
        self.root = Path(root)
        self.cfg = cfg

    def _f(self, name: str) -> Path:
        for ext in (".csv.gz", ".csv"):
            p = self.root / f"{name}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"{self.root} 缺少 {name}.csv(.gz)")

    def discover(self) -> str:
        lines = [f"[INSPIRE] root={self.root}"]
        for t in ("operations", "vitals", "ward_vitals", "labs"):
            p = self.root / f"{t}.csv.gz"
            lines.append(f"  {t:<13} "
                         f"{'cols=' + str(list(pd.read_csv(p, nrows=2).columns)) if p.exists() else '!! 缺失'}")
        return "\n".join(lines)

    def build(self) -> BuildResult:
        cfg, tr = self.cfg, _Tracker(self.name)
        ops = pd.read_csv(self._f("operations"), low_memory=False)
        # 与 VitalDB 的精确重叠标记（case_id 为官方外键）
        vital_ids = _vitaldb_caseids(cfg)
        if "case_id" in ops.columns and vital_ids:
            cid = pd.to_numeric(ops["case_id"], errors="coerce")
            ops["in_vitaldb_overlap"] = cid.isin(vital_ids).to_numpy()
            LOG.info("[INSPIRE] 与 VitalDB 精确重叠 %d 例", int(ops["in_vitaldb_overlap"].sum()))
        else:
            ops["in_vitaldb_overlap"] = False
        tr.start(len(ops), "operations 表全部手术")

        n_icu = len(ops)
        ops = ops[ops["icuin_time"].notna()].copy()
        tr.step("术后转入 ICU（icuin_time 非空）", n_icu, len(ops), "")

        n0 = len(ops)
        ops = ops[pd.to_numeric(ops["age"], errors="coerce").between(cfg.age_min, cfg.age_max)]
        tr.step(f"年龄 {cfg.age_min}-{cfg.age_max}", n0, len(ops), "")

        ops["los_hours"] = (pd.to_numeric(ops["icuout_time"], errors="coerce")
                            - pd.to_numeric(ops["icuin_time"], errors="coerce")) / 60.0
        n1 = len(ops)
        ops = ops[ops["los_hours"] >= cfg.los_floor_h]
        tr.step(f"ICU LOS >= {cfg.los_floor_h:g}h", n1, len(ops), "")

        if cfg.limit:
            n_before = len(ops)
            ops = ops.head(cfg.limit)
            tr.step(f"运行上限抽样（--limit {cfg.limit}，可复现调试用）",
                    n_before, len(ops), "论文全量分析须去掉 --limit")
        ops[PID] = ops["op_id"].astype(str)
        keep_subj = set(ops["subject_id"].astype(str))
        op_time = ops.set_index(PID)["icuin_time"].astype(float)
        subj_ops = ops[[PID, "subject_id", "icuin_time", "icuout_time", "los_hours"]].copy()
        subj_ops["subject_id"] = subj_ops["subject_id"].astype(str)
        LOG.info("[INSPIRE] 候选 op %d", len(ops))

        labs = _gather(self._f("labs"), ["subject_id", "chart_time", "item_name", "value"],
                       cfg.vital_chunk,
                       lambda d: d[d["item_name"].astype(str).str.lower().eq("creatinine")])
        labs["subject_id"] = labs["subject_id"].astype(str)
        labs = labs[labs["subject_id"].isin(keep_subj)]
        m = labs.merge(subj_ops, on="subject_id", how="inner")
        m = m[(m["chart_time"] >= m["icuin_time"] - cfg.baseline_lookback_d * 24 * 60) &
              (m["chart_time"] <= m["icuin_time"] + cfg.max_followup_h * 60)]
        m["hour"] = (m["chart_time"] - m["icuin_time"]) / 60.0
        scr = pd.DataFrame({PID: m[PID], HOUR: m["hour"],
                            "value": pd.to_numeric(m["value"], errors="coerce")}).dropna()
        scr = scr[(scr["value"] > 0) & (scr["value"] < 30)]
        LOG.info("[INSPIRE] 肌酐记录 %d 条", len(scr))

        wv = _gather(self._f("ward_vitals"), ["subject_id", "chart_time", "item_name", "value"],
                     cfg.vital_chunk,
                     lambda d: d[d["item_name"].astype(str).str.lower().isin(
                         ["uo", "crrt", "hr", "rr", "spo2", "bt", "nibp_sbp", "nibp_dbp"])])
        wv["subject_id"] = wv["subject_id"].astype(str)
        wv = wv[wv["subject_id"].isin(keep_subj)]
        mw = wv.merge(subj_ops, on="subject_id", how="inner")
        mw = mw[(mw["chart_time"] >= mw["icuin_time"]) &
                (mw["chart_time"] <= mw["icuin_time"] + cfg.max_followup_h * 60)]
        mw["hour"] = (mw["chart_time"] - mw["icuin_time"]) / 60.0

        uo_raw = mw[mw["item_name"].str.lower() == "uo"]
        uo = pd.DataFrame({PID: uo_raw[PID], HOUR: uo_raw["hour"],
                           "ml": pd.to_numeric(uo_raw["value"], errors="coerce")}).dropna()
        uo = uo[(uo["ml"] >= 0) & (uo["ml"] < 5000)]
        crrt_raw = mw[mw["item_name"].str.lower() == "crrt"]
        crrt_raw = crrt_raw[pd.to_numeric(crrt_raw["value"], errors="coerce").fillna(0) >= 1]
        rrt = pd.DataFrame({PID: crrt_raw[PID], HOUR: crrt_raw["hour"]}).dropna()
        LOG.info("[INSPIRE] 尿量 %d 条，CRRT %d 条", len(uo), len(rrt))

        iv = _gather(self._f("vitals"), ["op_id", "chart_time", "item_name", "value"],
                     cfg.vital_chunk,
                     lambda d: d[d["item_name"].astype(str).str.lower().isin(list(INSPIRE_VITAL_MAP))])
        iv["op_id"] = iv["op_id"].astype(str)
        iv = iv[iv["op_id"].isin(set(ops[PID]))]
        iv[PID] = iv["op_id"]          # op_id 即本工程的 patient_id
        iv["hour"] = (pd.to_numeric(iv["chart_time"], errors="coerce")
                      - iv["op_id"].map(op_time).astype(float)) / 60.0
        iv["variable"] = iv["item_name"].str.lower().map(lambda k: INSPIRE_VITAL_MAP[k][0])
        iv["src"] = iv["item_name"].str.lower().map(lambda k: INSPIRE_VITAL_MAP[k][1])
        iv = iv.dropna(subset=["hour"])
        iv = iv[(iv["hour"] >= -24) & (iv["hour"] < 0)]

        pw = mw[mw["item_name"].str.lower().isin(
            ["hr", "rr", "spo2", "bt", "nibp_sbp", "nibp_dbp"])].copy()
        pw["variable"] = pw["item_name"].str.lower().map(lambda k: INSPIRE_VITAL_MAP[k][0])
        pw["src"] = pw["item_name"].str.lower().map(lambda k: INSPIRE_VITAL_MAP[k][1])

        vit = pd.concat([iv[[PID, HOUR, VAR, VAL, "src"]],
                         pw[[PID, HOUR, VAR, VAL, "src"]]], ignore_index=True)
        vit[VAL] = pd.to_numeric(vit[VAL], errors="coerce")
        vit = vit.dropna(subset=[VAL])
        vit = self._coalesce_bp(vit)

        # 先按患者分组一次，循环内只做字典查询，避免 9,560 次全表扫描
        scr_by = {k: v.sort_values(HOUR) for k, v in scr.groupby(PID)}
        empty_scr = scr.iloc[0:0]
        rows = []
        for _, u in ops.iterrows():
            s = scr_by.get(u[PID], empty_scr)
            pre = s[(s[HOUR] < 0) & (s[HOUR] >= -cfg.baseline_lookback_d * 24)]
            first = s[(s[HOUR] >= 0) & (s[HOUR] <= cfg.icu_first_window_h)]["value"]
            b, src = pick_baseline(pre["value"],
                                   float(first.iloc[0]) if len(first) else np.nan,
                                   float(u["age"]), str(u["sex"]), cfg)
            pre_aki = 0
            if len(pre) and np.isfinite(b) and b > 0:
                near = pre[pre[HOUR] >= -cfg.pre_icu_guard_h]["value"]
                if len(near) and near.max() / b >= 1.5:
                    pre_aki = 1
            rows.append({**u.to_dict(), "baseline_scr": b, "baseline_source": src,
                         "pre_icu_aki": pre_aki})
        units = pd.DataFrame(rows)

        n2 = len(units)
        units = units[units["baseline_scr"].notna() & (units["baseline_scr"] > 0)]
        tr.step("基线 SCr 可得", n2, len(units), "1=术前 2=入科首次 3=估算")

        n3 = len(units)
        units = units[units["pre_icu_aki"] == 0]
        tr.step(f"排除入 ICU 前 {cfg.pre_icu_guard_h:g}h 内 AKI", n3, len(units), "")

        LOG.info("[INSPIRE] KDIGO 打标 %d 人 ...", len(units))
        uo_by = {k: v for k, v in uo.groupby(PID)}
        rrt_by = {k: v for k, v in rrt.groupby(PID)}
        out = []
        for _, u in units.iterrows():
            s = scr_by.get(u[PID], scr.iloc[0:0])
            out.append({**u.to_dict(),
                        **label_patient(s[s[HOUR] >= 0], uo_by.get(u[PID], uo.iloc[0:0]),
                                        rrt_by.get(u[PID], rrt.iloc[0:0]),
                                        float(u["baseline_scr"]), float(u["weight"]),
                                        float(u["los_hours"]), cfg)})
        cohort = pd.DataFrame(out)
        tr.step("最终纳入（PO-AKI 可判定）", len(cohort), len(cohort), "")

        cohort["icu_type"] = cohort["department"].astype(str)
        cohort["sex"] = cohort["sex"].astype(str)
        return BuildResult(self.name, cohort,
                           vit[vit[PID].isin(set(cohort[PID]))][[PID, HOUR, VAR, VAL]], tr.rows)

    @staticmethod
    def _coalesce_bp(vit: pd.DataFrame) -> pd.DataFrame:
        """同一小时优先有创动脉压；有创缺失时才用无创。"""
        bp = vit[vit[VAR].isin(["SBP", "DBP"])].copy()
        other = vit[~vit[VAR].isin(["SBP", "DBP"])]
        if bp.empty:
            return other
        bp["hb"] = np.floor(bp[HOUR]).astype(int)
        has_art = (bp[bp["src"] == "art"].groupby([PID, VAR, "hb"]).size()
                   .rename("n_art").reset_index())
        bp = bp.merge(has_art, on=[PID, VAR, "hb"], how="left")
        bp = bp[~((bp["src"] == "nibp") & (bp["n_art"].fillna(0) > 0))]
        out = pd.concat([bp[[PID, HOUR, VAR, VAL]], other[[PID, HOUR, VAR, VAL]]],
                        ignore_index=True)
        return out.drop_duplicates(subset=[PID, HOUR, VAR])


# =============================================================================
# 5. VitalDB（Track 2B）
# =============================================================================
class VitaldbAdapter:
    """VitalDB：术后 SCr 完整（lab_data.name='cr'），术中波形需 vitaldb 包解析。"""
    name = "vitaldb"

    def __init__(self, root: str, cfg: CohortConfig):
        self.root = Path(root)
        self.cfg = cfg

    def _f(self, name: str) -> Path:
        p = self.root / name
        if not p.exists():
            raise FileNotFoundError(f"{self.root} 缺少 {name}")
        return p

    def discover(self) -> str:
        lines = [f"[VitalDB] root={self.root}"]
        for t in ("clinical_data.csv", "lab_data.csv", "tracks.csv"):
            p = self.root / t
            lines.append(f"  {t:<20} "
                         f"{'cols=' + str(list(pd.read_csv(p, nrows=2).columns)) if p.exists() else '!! 缺失'}")
        vf = self.root / "vital_files"
        n = len(list(vf.glob("*.vital"))) if vf.exists() else 0
        lines.append(f"  vital_files/*.vital    {n} 个（需 vitaldb 包解析）")
        return "\n".join(lines)

    def build(self) -> BuildResult:
        cfg, tr = self.cfg, _Tracker(self.name)
        cl = pd.read_csv(self._f("clinical_data.csv"), low_memory=False)
        tr.start(len(cl), "clinical_data 全部手术")

        n0 = len(cl)
        cl = cl[pd.to_numeric(cl["age"], errors="coerce").between(cfg.age_min, cfg.age_max)]
        tr.step(f"年龄 {cfg.age_min}-{cfg.age_max}", n0, len(cl), "")

        cl = cl.copy()
        cl["los_hours"] = pd.to_numeric(cl.get("icu_days"), errors="coerce").fillna(0) * 24.0
        tr.step("有术后随访窗（icu_days 可解析）", len(cl), len(cl),
                "VitalDB 日间手术多，不施加 LOS>=24h 门槛")

        if cfg.limit:
            n_before = len(cl)
            cl = cl.head(cfg.limit)
            tr.step(f"运行上限抽样（--limit {cfg.limit}，可复现调试用）",
                    n_before, len(cl), "论文全量分析须去掉 --limit")
        cl[PID] = cl["caseid"].astype(str)
        keep = set(cl[PID])

        lab = pd.read_csv(self._f("lab_data.csv"), low_memory=False)
        cr = lab[lab["name"].astype(str).str.lower().eq("cr") &
                 lab["caseid"].astype(str).isin(keep)].copy()
        cr[PID] = cr["caseid"].astype(str)
        cl["t0_sec"] = pd.to_numeric(cl["caseend"], errors="coerce")
        t0 = cl.set_index(PID)["t0_sec"]
        cr["hour"] = (pd.to_numeric(cr["dt"], errors="coerce")
                      - cr[PID].map(t0).astype(float)) / 3600.0
        scr = pd.DataFrame({PID: cr[PID], HOUR: cr["hour"],
                            "value": pd.to_numeric(cr["result"], errors="coerce")}).dropna()
        scr = scr[(scr["value"] > 0) & (scr["value"] < 30)]
        LOG.info("[VitalDB] 肌酐记录 %d 条", len(scr))

        rows = []
        for _, u in cl.iterrows():
            s = scr[scr[PID] == u[PID]].sort_values(HOUR)
            pre = s[(s[HOUR] < 0) & (s[HOUR] >= -cfg.baseline_lookback_d * 24)]
            # 真实随访长度 = 最后一次肌酐观测（上限 7 天），不能拿 icu_days 代替
            post = s[(s[HOUR] >= 0) & (s[HOUR] <= cfg.vitaldb_horizon_h)]
            fu = float(post[HOUR].max()) if len(post) else 0.0
            preop = pd.to_numeric(pd.Series([u.get("preop_cr")]), errors="coerce").iloc[0]
            b, src = pick_baseline(pre["value"],
                                   float(preop) if np.isfinite(preop) else np.nan,
                                   float(u["age"]), str(u["sex"]), cfg)
            rows.append({**u.to_dict(), "baseline_scr": b, "baseline_source": src,
                         "followup_h": fu})
        units = pd.DataFrame(rows)

        n2 = len(units)
        units = units[units["baseline_scr"].notna() & (units["baseline_scr"] > 0)]
        tr.step("基线 SCr 可得", n2, len(units), "VitalDB 的 preop_cr 属编码 1")

        LOG.info("[VitalDB] KDIGO 打标 %d 人（7 天视界）...", len(units))
        scr_by = {k: v.sort_values(HOUR) for k, v in scr.groupby(PID)}
        empty_uo = pd.DataFrame(columns=[PID, HOUR, "ml"])
        empty_rrt = pd.DataFrame(columns=[PID, HOUR])
        out = []
        for _, u in units.iterrows():
            s = scr_by.get(u[PID], scr.iloc[0:0])
            out.append({**u.to_dict(),
                        **label_patient(s[s[HOUR] >= 0], empty_uo, empty_rrt,
                                        float(u["baseline_scr"]), np.nan,
                                        float(u["followup_h"]), cfg)})
        cohort = pd.DataFrame(out)
        cohort["icu_type"] = cohort["department"].astype(str)
        cohort["sex"] = cohort["sex"].astype(str)
        tr.step("最终纳入（7 天 KDIGO 可判定）", len(cohort), len(cohort), "Track 2B")

        # 与 INSPIRE 的重叠标记（caseid 命中 INSPIRE.operations.case_id）
        insp = _inspire_caseids(cfg)
        if insp:
            cohort["in_vitaldb_overlap"] = cohort[PID].astype("int64").isin(insp).to_numpy()
            LOG.info("[VitalDB] 与 INSPIRE 精确重叠 %d 例",
                     int(cohort["in_vitaldb_overlap"].sum()))
        else:
            cohort["in_vitaldb_overlap"] = False

        vitals = self._extract_vitals(cohort, cfg)
        notes = [] if len(vitals) else ["vitals 为空：本地缺 .vital 或 vitaldb 包不可用"]
        return BuildResult(self.name, cohort, vitals, tr.rows, notes=notes)

    def _extract_vitals(self, cohort: pd.DataFrame, cfg: CohortConfig) -> pd.DataFrame:
        """并行解析本地 .vital 波形，抽取术中 6 通道（60s 重采样）。"""
        vdir = self.root / "vital_files"
        if not vdir.exists():
            LOG.warning("[VitalDB] 无 vital_files 目录，跳过波形抽取")
            return pd.DataFrame(columns=[PID, HOUR, VAR, VAL])
        try:
            import vitaldb  # noqa: F401
        except ImportError:
            LOG.warning("[VitalDB] 未安装 vitaldb 包，跳过波形抽取")
            return pd.DataFrame(columns=[PID, HOUR, VAR, VAL])

        end = cohort.set_index(PID)["caseend"].astype(float)
        tasks = []
        for cid in cohort[PID].astype("int64"):
            f = vdir / f"{cid}.vital"
            if f.exists():
                tasks.append((str(f), str(cid), float(end.get(str(cid), np.nan))))
        if not tasks:
            LOG.warning("[VitalDB] 队列中的 caseid 均无本地 .vital 文件")
            return pd.DataFrame(columns=[PID, HOUR, VAR, VAL])

        LOG.info("[VitalDB] 解析 %d 个 .vital（%d 进程）...", len(tasks), cfg.vital_workers)
        from concurrent.futures import ProcessPoolExecutor
        frames, t0 = [], __import__("time").time()
        with ProcessPoolExecutor(max_workers=cfg.vital_workers) as ex:
            for k, (cid, df) in enumerate(ex.map(_parse_one_vital, tasks, chunksize=20), 1):
                if df is not None:
                    frames.append(df)
                if k % 500 == 0:
                    LOG.info("  已解析 %d/%d（%.0fs）", k, len(tasks),
                             __import__("time").time() - t0)
        if not frames:
            return pd.DataFrame(columns=[PID, HOUR, VAR, VAL])
        out = pd.concat(frames, ignore_index=True)
        out[VAL] = pd.to_numeric(out[VAL], errors="coerce")
        out = out.dropna(subset=[VAL])
        LOG.info("[VitalDB] 波形长表 %d 行，覆盖 %d 例", len(out), out[PID].nunique())
        return out


# =============================================================================
# 6. 指纹去重
# =============================================================================
def fingerprint_dedup(inspire_ops: pd.DataFrame, vital_cl: pd.DataFrame) -> pd.DataFrame:
    """比对 INSPIRE 与 VitalDB 的潜在重叠患者。

    优先用显式外键：INSPIRE.operations.case_id 官方标注为「linker to VitalDB」。
    外键缺失时回退指纹：年龄/性别/身高/体重/手术时长 五项取整后完全一致。
    """
    rep = {"vitaldb_total": len(vital_cl)}
    if "case_id" in inspire_ops.columns:
        cid = pd.to_numeric(inspire_ops["case_id"], errors="coerce").dropna().astype("int64")
        rep["inspire_case_id_present"] = int(len(cid))
        rep["exact_case_id"] = int(cid.isin(set(vital_cl["caseid"])).sum())
    else:
        rep["inspire_case_id_present"] = 0
        rep["exact_case_id"] = 0

    def fp(age, sex, h, w, dur):
        return (pd.to_numeric(age, errors="coerce").round(0).astype("Int64").astype(str) + "|" +
                sex.astype(str).str.upper().str[:1] + "|" +
                pd.to_numeric(h, errors="coerce").round(0).astype("Int64").astype(str) + "|" +
                pd.to_numeric(w, errors="coerce").round(0).astype("Int64").astype(str) + "|" +
                pd.to_numeric(dur, errors="coerce").round(-1).astype("Int64").astype(str))

    fi = fp(inspire_ops["age"], inspire_ops["sex"], inspire_ops["height"], inspire_ops["weight"],
            inspire_ops["opend_time"] - inspire_ops["opstart_time"])
    fv = fp(vital_cl["age"], vital_cl["sex"], vital_cl["height"], vital_cl["weight"],
            (vital_cl["opend"] - vital_cl["opstart"]) / 60.0)
    rep["fingerprint"] = int(fv.isin(set(fi)).sum())
    return pd.DataFrame([rep])


# =============================================================================
# 7. 契约校验
# =============================================================================
def validate_cohort(df: pd.DataFrame) -> List[str]:
    issues = []
    missing = [c for c in COHORT_REQUIRED if c not in df.columns]
    if missing:
        return [f"cohort 缺少必需列：{missing}"]
    if df["patient_id"].duplicated().any():
        issues.append(f"patient_id 重复 {int(df['patient_id'].duplicated().sum())} 个")
    fu = pd.to_numeric(df["followup_h"], errors="coerce")
    if fu.isna().any():
        issues.append(f"followup_h 缺失 {int(fu.isna().sum())} 个")
    onset = pd.to_numeric(df["t_aki_onset"], errors="coerce")
    v = onset.notna()
    if v.any():
        bad = (onset[v] < 0) | (onset[v] > fu[v] + 1e-6)
        if bad.any():
            issues.append(f"t_aki_onset 越界 {int(bad.sum())} 个")
    src = pd.to_numeric(df["baseline_source"], errors="coerce")
    if not src.isin([1, 2, 3]).all():
        issues.append(f"baseline_source 取值异常：{sorted(set(src.dropna().unique()))}")
    return issues


def validate_vitals(df: pd.DataFrame) -> List[str]:
    issues = []
    missing = [c for c in VITALS_REQUIRED if c not in df.columns]
    if missing:
        return [f"vitals 缺少必需列：{missing}"]
    return issues


# =============================================================================
# 8. 主流程
# =============================================================================
def run_db(adapter, out: Path, cfg: CohortConfig) -> Tuple[BuildResult, pd.DataFrame]:
    res = adapter.build()
    out.mkdir(parents=True, exist_ok=True)

    cohort = res.cohort.copy()
    if "followup_h" not in cohort.columns:
        cohort["followup_h"] = pd.to_numeric(cohort.get("los_hours"), errors="coerce")
    cohort["db"] = res.db
    cols = ["db", PID, "t_aki_onset", "followup_h", "baseline_source", "max_stage",
            "t_stage1", "t_stage2", "t_stage3", "crit_scr", "crit_uo", "crit_rrt",
            "age", "sex", "icu_type", "baseline_scr", "in_vitaldb_overlap"]
    cohort[[c for c in cols if c in cohort.columns]].to_csv(
        out / f"cohort_{res.db}.csv", index=False, encoding="utf-8-sig")

    if len(res.vitals):
        res.vitals.to_csv(out / f"vitals_{res.db}.csv.gz", index=False, compression="gzip")
    attr = pd.DataFrame(res.attrition)
    attr.to_csv(out / f"attrition_{res.db}.csv", index=False, encoding="utf-8-sig")
    return res, attr


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="三库队列提取与 KDIGO 打标")
    ap.add_argument("--db", choices=["eicu", "inspire", "vitaldb", "all"], default="all")
    # 数据库根目录一律由命令行参数或环境变量提供，仓库内不留任何本机绝对路径。
    ap.add_argument("--eicu-root", default=os.environ.get("PO_AKI_EICU_ROOT", "data/eicu"))
    ap.add_argument("--inspire-root", default=os.environ.get("PO_AKI_INSPIRE_ROOT", "data/inspire"))
    ap.add_argument("--vitaldb-root", default=os.environ.get("PO_AKI_VITALDB_ROOT", "data/vitaldb"))
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--limit", type=int, default=None, help="每库最多处理 N 个 stay（调试）")
    ap.add_argument("--vitals-sample", type=int, default=None,
                    help="eICU 生命体征抽样人数（保留全部阳性+随机阴性，瀑布表记录）")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--vitals")
    ap.add_argument("--cohort")
    ap.add_argument("--discover", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    out = ensure_dir(args.out)

    if args.check:
        if not args.vitals or not args.cohort:
            ap.error("--check 需要 --vitals 与 --cohort")
        c = pd.read_csv(args.cohort)
        if not Path(args.vitals).exists():
            print(f"[vitals] 文件不存在：{args.vitals}（VitalDB 无波形时属预期）")
            v = pd.DataFrame(columns=VITALS_REQUIRED)
        else:
            v = pd.read_csv(args.vitals)
        issues = validate_vitals(v) + validate_cohort(c)
        n_aki = int(pd.to_numeric(c["t_aki_onset"], errors="coerce").notna().sum())
        lines = [f"[vitals] {len(v):,} 行 × {len(v.columns)} 列",
                 f"[cohort] {len(c):,} 行 × {len(c.columns)} 列",
                 f"[队列] PO-AKI {n_aki} / {len(c)}（{100*n_aki/max(len(c),1):.2f}%）"]
        if len(v):
            n_neg = int((pd.to_numeric(v['hour'], errors='coerce') < 0).sum())
            lines.append(f"[提示] 负小时记录 {n_neg:,} 条"
                         + ("（无术中段，Track 2 不适用）" if n_neg == 0 else "（可构建 Track 2）"))
            lines += [f"[质量] value 缺失率 "
                      f"{100*pd.to_numeric(v['value'], errors='coerce').isna().mean():.2f}%",
                      ]
        lines += (["发现问题："] + [f"  - {i}" for i in issues]) if issues else ["契约校验通过。"]
        print("\n".join(lines))
        return 1 if issues else 0

    cfg = CohortConfig(limit=args.limit, vitaldb_root=args.vitaldb_root,
                       inspire_root=args.inspire_root, vitals_sample=args.vitals_sample)
    adapters = []
    if args.db in ("eicu", "all"):
        adapters.append(EicuAdapter(args.eicu_root, cfg))
    if args.db in ("inspire", "all"):
        adapters.append(InspireAdapter(args.inspire_root, cfg))
    if args.db in ("vitaldb", "all"):
        adapters.append(VitaldbAdapter(args.vitaldb_root, cfg))

    if args.discover:
        for a in adapters:
            print(a.discover())
        return 0

    all_attr, report = [], []
    for a in adapters:
        LOG.info("=== 构建 %s ===", a.name)
        try:
            res, attr = run_db(a, out, cfg)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("%s 构建失败", a.name)
            report.append(f"[{a.name}] 失败：{exc}")
            continue
        all_attr.append(attr)
        c = res.cohort
        n = len(c)
        k = int(pd.to_numeric(c["t_aki_onset"], errors="coerce").notna().sum())
        report.append(f"[{a.name}] 最终纳入 {n}，PO-AKI {k}（{100*k/max(n,1):.2f}%），"
                      f"vitals {len(res.vitals):,} 行")
        if res.notes:
            report += [f"  note: {x}" for x in res.notes]

    if all_attr:
        attr = pd.concat(all_attr, ignore_index=True)
        attr.to_csv(out / "cohort_attrition_real.csv", index=False, encoding="utf-8-sig")
        print(attr.to_string(index=False))

    try:
        ops = pd.read_csv(Path(args.inspire_root) / "operations.csv.gz", low_memory=False)
        cl = pd.read_csv(Path(args.vitaldb_root) / "clinical_data.csv", low_memory=False)
        ded = fingerprint_dedup(ops, cl)
        ded.to_csv(out / "dedup_inspire_vitaldb.csv", index=False)
        report.append(f"[去重] INSPIRE 有 case_id {int(ded.iloc[0]['inspire_case_id_present'])} 条，"
                      f"与 VitalDB 精确重叠 {int(ded.iloc[0]['exact_case_id'])} 条；"
                      f"指纹重叠 {int(ded.iloc[0]['fingerprint'])} 条")
    except Exception as exc:  # noqa: BLE001
        report.append(f"[去重] 跳过：{exc}")

    (out / "contract_check_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
