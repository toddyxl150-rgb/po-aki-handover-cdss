#!/usr/bin/env python
"""Generate a tiny *synthetic* eICU-CRD-shaped directory tree.

Purpose
-------
Let a reviewer who has **no PhysioNet credential** run `01_cohort_builder.py`
end to end.  Every value produced here is simulated from a random number
generator; not a single row comes from a real patient, and the schema is the
only thing borrowed from eICU-CRD.

The generator reproduces just enough structure for the adapter's contract:

    patient.csv.gz          patientunitstayid, uniquepid, age, gender, ...
    lab.csv.gz              creatinine series (baseline + optional rise)
    intakeOutput.csv.gz     hourly urine output
    treatment.csv.gz        surgical / dialysis strings
    pasthistory.csv.gz      CKD history (drives an exclusion step)
    diagnosis.csv.gz        ICD-9 based CKD exclusion
    apachePredVar.csv.gz    postop flag
    vitalPeriodic.csv.gz    arterial + monitored vitals, hourly
    vitalAperiodic.csv.gz   non-invasive cuff blood pressure

Usage
-----
    python demo/make_synthetic_eicu.py --out demo/data/eicu --n 160 --seed 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

VITALS = ("heartrate", "systemicsystolic", "systemicdiastolic",
          "respiration", "sao2", "temperature")


def _stay(rng: np.random.Generator, sid: int) -> Dict[str, object]:
    """Draw one synthetic ICU stay and its event streams."""
    age = int(rng.integers(19, 90))
    sex = str(rng.choice(["Male", "Female"]))
    weight = float(rng.uniform(55, 105))
    los_h = float(rng.uniform(30, 110))
    baseline = float(rng.uniform(0.65, 1.30))

    # ---- surgical admission: three independent routes, mirroring eICU ----
    route = rng.choice(["or", "postop", "treatment", "or", "or"], p=None)
    surg_src = "Operating Room" if route == "or" else str(
        rng.choice(["Floor", "Emergency Department", "Transfer"]))
    postop = 1 if route == "postop" else 0
    surg_treatment = route == "treatment"

    # ---- AKI phenotype: ~45% of stays develop KDIGO stage >= 1 ----
    aki = rng.random() < 0.24
    onset_h = float(rng.uniform(10, min(los_h - 4, 70))) if aki else np.nan
    ratio = float(rng.uniform(1.55, 2.60)) if aki else 1.0
    oliguric = (not aki) and rng.random() < 0.05      # UO-only AKI
    rrt = rng.random() < 0.015
    ckd = rng.random() < 0.05                          # excluded by the cohort rule

    # ---- creatinine: sampled every 6 h, baseline at t = 0 ----
    lab_rows: List[Dict[str, object]] = []
    if rng.random() < 0.5:                             # pre-ICU value -> baseline tier 1
        lab_rows.append({"patientunitstayid": sid,
                         "labresultoffset": int(-rng.integers(12, 40) * 60),
                         "labname": "creatinine", "labresult": round(baseline * 0.95, 2)})
    for h in range(0, int(los_h) + 1, 6):
        v = baseline
        if aki and h >= onset_h:
            v = baseline * ratio
        elif aki and h >= onset_h - 6:
            v = baseline * (1.0 + 0.35 * (ratio - 1.0))
        lab_rows.append({"patientunitstayid": sid, "labresultoffset": int(h * 60),
                         "labname": "creatinine", "labresult": round(v, 2)})

    # ---- urine output: hourly, ml ----
    io_rows: List[Dict[str, object]] = []
    for h in range(int(los_h)):
        if oliguric and h >= 6:
            rate = rng.uniform(0.15, 0.35)
        elif aki and h >= onset_h:
            rate = rng.uniform(0.30, 0.60)
        else:
            rate = rng.uniform(0.80, 1.70)
        io_rows.append({"patientunitstayid": sid, "intakeoutputoffset": int(h * 60),
                        "cellpath": "flowsheet|intakeoutput|output|urine",
                        "celllabel": "Urine", "cellvaluenumeric": round(rate * weight, 1)})

    # ---- treatments ----
    trt_rows: List[Dict[str, object]] = []
    if surg_treatment:
        trt_rows.append({"patientunitstayid": sid, "treatmentoffset": 6 * 60,
                         "treatmentstring": "surgery|operative care|postoperative"})
    if rrt:
        trt_rows.append({"patientunitstayid": sid,
                         "treatmentoffset": int(min(onset_h if aki else 24, los_h - 2) * 60),
                         "treatmentstring": "renal|dialysis|renal replacement therapy"})

    # ---- vitals: hourly, drifting before AKI onset ----
    arterial = rng.random() > 0.30                     # 30% cuff-only
    vp_rows: List[Dict[str, object]] = []
    va_rows: List[Dict[str, object]] = []
    for h in range(int(los_h)):
        lead = 0.0
        if aki and h < onset_h:
            lead = float(np.clip((h - (onset_h - 12)) / 12.0, 0.0, 1.0))
        hr = 78 + 22 * lead + rng.normal(0, 5)
        sbp = 124 - 16 * lead + rng.normal(0, 8)
        dbp = 74 - 9 * lead + rng.normal(0, 5)
        rr = 16 + 5 * lead + rng.normal(0, 1.8)
        spo2 = 97 - 2.5 * lead + rng.normal(0, 1.0)
        temp = 36.8 + 0.5 * lead + rng.normal(0, 0.25)
        row = {"patientunitstayid": sid, "observationoffset": int(h * 60),
               "heartrate": float(np.clip(hr, 35, 190)),
               "respiration": float(np.clip(rr, 6, 45)),
               "sao2": float(np.clip(spo2, 70, 100)),
               "temperature": round(float(np.clip(temp, 34, 41)), 2)}
        if arterial:
            row["systemicsystolic"] = float(np.clip(sbp, 60, 220))
            row["systemicdiastolic"] = float(np.clip(dbp, 30, 130))
        else:
            row["systemicsystolic"] = np.nan
            row["systemicdiastolic"] = np.nan
            if h % 2 == 0:
                va_rows.append({"patientunitstayid": sid, "observationoffset": int(h * 60),
                                "noninvasivesystolic": float(np.clip(sbp, 60, 220)),
                                "noninvasivediastolic": float(np.clip(dbp, 30, 130))})
        vp_rows.append(row)

    return {
        "patient": {"patientunitstayid": sid, "uniquepid": f"SYN-{sid:06d}",
                    "age": str(age), "gender": sex, "admissionweight": round(weight, 1),
                    "unittype": str(rng.choice(["MICU", "SICU", "TSICU"])),
                    "unitadmitsource": surg_src,
                    "unitdischargeoffset": int(los_h * 60),
                    "unitvisitnumber": 1,
                    "unitdischargestatus": "Expired" if rng.random() < 0.06 else "Alive",
                    "hospitaldischargestatus": "Alive"},
        "apache": {"patientunitstayid": sid, "postop": postop},
        "lab": lab_rows, "io": io_rows, "treatment": trt_rows,
        "pasthistory": ([] if not ckd else
                        [{"patientunitstayid": sid,
                          "pasthistorypath": "renal|chronic kidney disease",
                          "pasthistoryvalue": "chronic kidney disease"}]),
        "diagnosis": ([] if not ckd else
                      [{"patientunitstayid": sid,
                        "diagnosisstring": "renal|chronic kidney disease|stage 5",
                        "icd9code": "5856"}]),
        "vp": vp_rows, "va": va_rows,
    }


def build(out: Path, n: int = 160, seed: int = 2026) -> Path:
    rng = np.random.default_rng(seed)
    parts = {k: [] for k in
             ("patient", "apache", "lab", "io", "treatment",
              "pasthistory", "diagnosis", "vp", "va")}
    for sid in range(1, n + 1):
        stay = _stay(rng, sid)
        for k in parts:
            v = stay[k]
            parts[k].extend(v if isinstance(v, list) else [v])

    out.mkdir(parents=True, exist_ok=True)
    name = {"patient": "patient", "apache": "apachePredVar", "lab": "lab",
            "io": "intakeOutput", "treatment": "treatment",
            "pasthistory": "pasthistory", "diagnosis": "diagnosis",
            "vp": "vitalPeriodic", "va": "vitalAperiodic"}
    for k, rows in parts.items():
        df = pd.DataFrame(rows)
        df.to_csv(out / f"{name[k]}.csv.gz", index=False, compression="gzip")
        print(f"  {name[k]:<16} {len(df):>7,} rows x {len(df.columns)} cols")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="生成合成 eICU 形状数据（无真实患者）")
    ap.add_argument("--out", default="demo/data/eicu")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    print(f"[synthetic] 生成 {args.n} 个模拟 ICU stay -> {args.out}")
    build(Path(args.out), args.n, args.seed)
    print("[synthetic] 完成。全部数值由 RNG 生成，不含任何真实患者记录。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
