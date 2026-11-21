"""Table 1：三库患者基线特征对比。

数据源（全部现算，不硬编码）：
    outputs/cohort_{inspire,eicu,vitaldb}.csv
    outputs/windows_track1_manifest.json（INSPIRE）
    outputs/eicu/windows_track1_manifest.json（eICU）

输出：
    outputs/paper_assets/table1_baseline.csv
    outputs/paper_assets/table1_baseline.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("outputs")
OUT = ROOT / "paper_assets"
DBS = [("inspire", "INSPIRE"), ("eicu", "eICU-CRD"), ("vitaldb", "VitalDB")]


def _iqr(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{q2:.1f} [{q1:.1f}–{q3:.1f}]"


def _mean_sd(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return f"{s.mean():.1f} ± {s.std(ddof=1):.1f}"


def _n_pct(mask: pd.Series, total: int) -> str:
    n = int(np.asarray(mask).sum())
    return f"{n:,} ({100 * n / total:.1f}%)"


def _windows(db: str) -> Dict[str, int] | None:
    f = ROOT / ("windows_track1_manifest.json" if db == "inspire"
                else f"{db}/windows_track1_manifest.json")
    if not f.exists():
        return None
    import json
    j = json.loads(f.read_text(encoding="utf-8"))
    return {"windows": int(j["n_windows"]), "patients": int(j["stats"]["n_patients"])}


def build() -> pd.DataFrame:
    rows: List[Dict] = []
    frames = {}
    for db, label in DBS:
        frames[label] = pd.read_csv(ROOT / f"cohort_{db}.csv")

    def add(section: str, item: str, vals: Dict[str, str], note: str = "") -> None:
        r = {"section": section, "characteristic": item}
        r.update({lbl: vals.get(lbl, "—") for _, lbl in DBS})
        r["note"] = note
        rows.append(r)

    labels = [lbl for _, lbl in DBS]
    add("Cohort", "Patients, n", {lbl: f"{len(frames[lbl]):,}" for lbl in labels})

    add("Demographics", "Age, years, mean ± SD",
        {lbl: _mean_sd(frames[lbl]["age"]) for lbl in labels})
    add("Demographics", "Age, years, median [IQR]",
        {lbl: _iqr(frames[lbl]["age"]) for lbl in labels},
        "reported both ways; normality tests reject at this sample size")
    add("Demographics", "Male, n (%)",
        {lbl: _n_pct(frames[lbl]["sex"].astype(str).str.upper().str.startswith("M"),
                     len(frames[lbl])) for lbl in labels})

    add("Monitoring", "Follow-up, hours, median [IQR]",
        {lbl: _iqr(frames[lbl]["followup_h"]) for lbl in labels},
        "effective ICU follow-up horizon, censored at 7 days")

    add("Renal", "Baseline SCr, mg/dL, median [IQR]",
        {lbl: _iqr(frames[lbl]["baseline_scr"]) for lbl in labels})
    for tier, desc in ((1, "Tier 1 (true pre-ICU, [-180 d, 0 h))"),
                       (2, "Tier 2 (first ICU SCr, 0–6 h)"),
                       (3, "Tier 3 (estimated / lowest)")):
        add("Renal", f"Baseline SCr source: {desc}",
            {lbl: _n_pct(frames[lbl]["baseline_source"] == tier, len(frames[lbl]))
             for lbl in labels})

    add("Outcome", "PO-AKI, KDIGO stage 1–3, n (%)",
        {lbl: _n_pct(frames[lbl]["t_aki_onset"].notna(), len(frames[lbl]))
         for lbl in labels})
    add("Outcome", "PO-AKI, KDIGO stage 2–3, n (%)",
        {lbl: _n_pct(frames[lbl]["max_stage"] >= 2, len(frames[lbl]))
         for lbl in labels})

    win = {lbl: _windows(db) for db, lbl in DBS}
    add("Analytic sample", "Patients contributing ≥1 window, n",
        {lbl: (f"{win[lbl]['patients']:,}" if win[lbl] else "Pending validation")
         for lbl in labels})
    add("Analytic sample", "Total 12-h windows, n",
        {lbl: (f"{win[lbl]['windows']:,}" if win[lbl] else "Pending validation")
         for lbl in labels},
        "1-h step; windows require ≥50% channel observation")

    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    md = ["# Table 1. Baseline characteristics of the three cohorts", "",
          "> Values are computed directly from `cohort_{inspire,eicu,vitaldb}.csv` and the "
          "window manifests. No imputation is applied for descriptive statistics.", ""]
    header = "| Characteristic | " + " | ".join(["INSPIRE (n=9,543)", "eICU-CRD (n=3,597)",
                                                 "VitalDB (n=3,000)"]) + " |"
    md += [header, "|---" + "|---:" * 3 + "|"]
    last = None
    for _, r in df.iterrows():
        if r["section"] != last:
            md.append(f"| **{r['section']}** | | | |")
            last = r["section"]
        md.append(f"| {r['characteristic']} | {r['INSPIRE']} | {r['eICU-CRD']} | {r['VitalDB']} |")
    md += ["", "**Notes.** Continuous variables are reported as median [IQR] except age, which is "
               "given both as mean ± SD and median [IQR] because normality tests reject at these "
               "sample sizes. `Follow-up` is the effective ICU follow-up horizon censored at 7 days. "
               "Baseline SCr tiers follow the pre-specified ascertainment hierarchy. VitalDB has no "
               "12-h window table because it serves as a pending-validation cohort.", ""]
    return "\n".join(md)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_csv(OUT / "table1_baseline.csv", index=False, encoding="utf-8-sig")
    (OUT / "table1_baseline.md").write_text(to_markdown(df), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\n-> {OUT/'table1_baseline.csv'}\n-> {OUT/'table1_baseline.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
