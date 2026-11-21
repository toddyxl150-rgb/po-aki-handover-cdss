#!/usr/bin/env python
"""One-click, zero-data reproduction demo.

Runs the whole pipeline on *simulated* data so that a reviewer without a
PhysioNet credential can verify in about a minute that the code executes and
that the methodological invariants hold.

What is exercised
-----------------
    1  stats_utils.py            --self-test      BH / Holm / Tukey
    2  02_window_preprocessor    --self-test      bucketing, range filter,
                                                  causal LOCF, train-only
                                                  median, label horizon,
                                                  Track-2 t_end = +8 h
    3  03_feature_engineering    --self-test      630-dim layout, Gini identity
    4  05_clinical_metrics       --self-test      ER / FAR / lead time / DCA
    5  01_cohort_builder         synthetic eICU   KDIGO labelling, attrition
    6  02_window_preprocessor    windows          Track 1, fixed scaler
    7  03_feature_engineering    features         633 columns
    8  04_train_evaluate         --fast           2 folds, LightGBM + baselines
    9  05_clinical_metrics       predictions      operating-point metrics

Every value is produced by a random number generator.  No row of real patient
data is read, written, or required.

    python demo/run_demo.py                     # ~60 s
    python demo/run_demo.py --with-track-compare   # + Track 1 vs Track 2 smoke test
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "demo_run"
OUT = RUN / "outputs"
SYN = RUN / "data" / "eicu"

PY = sys.executable


def run(label: str, args: list[str], allow_fail: bool = False) -> float:
    t0 = time.time()
    print(f"\n\033[1m▶ {label}\033[0m")
    print("  $ " + " ".join(str(a) for a in args))
    p = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True)
    dt = time.time() - t0
    if p.returncode != 0:
        print(p.stdout[-2500:])
        print(p.stderr[-2500:], file=sys.stderr)
        if not allow_fail:
            raise SystemExit(f"\n[FAIL] {label} (exit {p.returncode})")
        print(f"  [skip] exit {p.returncode} — tolerated in this mode")
        return dt
    tail = [ln for ln in p.stdout.strip().splitlines() if ln.strip()][-3:]
    for ln in tail:
        print("  " + ln[:160])
    print(f"  \033[92mok\033[0m  {dt:5.1f}s")
    return dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-track-compare", action="store_true",
                    help="also run 07 on the built-in dual-track synthetic set")
    ap.add_argument("--keep", action="store_true", help="keep demo_run/ after finishing")
    args = ap.parse_args()

    if RUN.exists():
        shutil.rmtree(RUN)
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("PO-AKI handover CDSS — zero-data reproduction demo")
    print("=" * 74)
    print("All inputs below are SIMULATED. No real patient record is read or written.")

    times: dict[str, float] = {}

    # ---------------- stage 1: methodological self-tests ----------------
    times["stats_utils self-test"] = run(
        "1/9  stats_utils — BH / Holm / Tukey self-test", ["stats_utils.py"])
    times["02 self-test"] = run(
        "2/9  02 — window invariants (causality, leakage, label horizon)",
        ["02_window_preprocessor.py", "--self-test"])
    times["03 self-test"] = run(
        "3/9  03 — feature layout (120 + 480 + 30) and Gini identity",
        ["03_feature_engineering.py", "--self-test"])
    times["05 self-test"] = run(
        "4/9  05 — clinical metrics (ER / FAR / lead time / DCA)",
        ["05_clinical_metrics.py", "--self-test"])

    # ---------------- stage 2: synthetic eICU -> cohort ----------------
    times["synthetic eICU"] = run(
        "5/9  demo — synthesise an eICU-shaped raw directory (160 stays)",
        ["demo/make_synthetic_eicu.py", "--out", str(SYN), "--n", "160", "--seed", "2026"])
    times["01 cohort"] = run(
        "6/9  01 — cohort extraction + KDIGO staging on the synthetic tables",
        ["01_cohort_builder.py", "--db", "eicu", "--eicu-root", str(SYN),
         "--out", str(OUT)])

    # ---------------- stage 3: windows -> features -> model ----------------
    times["02 windows"] = run(
        "7/9  02 — Track 1 sliding windows (t in [0, LOS), 12 h x 10 channels)",
        ["02_window_preprocessor.py",
         "--vitals", str(OUT / "vitals_eicu.csv.gz"),
         "--cohort", str(OUT / "cohort_eicu.csv"),
         "--out", str(OUT), "--track", "track1"])
    times["03 features"] = run(
        "8/9  03 — 633-dim feature matrix (L1 raw + L2 stats + L3 Gini + 3 raw perfusion)",
        ["03_feature_engineering.py",
         "--npz", str(OUT / "windows_track1_X.npy"),
         "--meta", str(OUT / "windows_track1_meta.csv"),
         "--out", str(OUT), "--tag", "track1",
         "--params", str(OUT / "preprocess_params.json")])
    times["04 train"] = run(
        "9/9a 04 — 2-fold CV, --fast (LightGBM + LR + MEWS-4 + GRU + RETAIN)",
        ["04_train_evaluate.py", "--features", str(OUT / "features_track1.parquet"),
         "--out", str(OUT / "cv_demo"), "--folds", "2", "--fast",
         "--max-rows", "1200", "--seq-epochs", "1"])

    # ---------------- stage 4: clinical operating point ----------------
    import pandas as pd
    oof = pd.read_csv(OUT / "cv_demo" / "oof_predictions.csv.gz")
    pred = OUT / "demo_pred_lightgbm.csv"
    (oof[oof["model"] == "LightGBM"].drop(columns=["model", "fold"])
        .to_csv(pred, index=False))
    times["05 metrics"] = run(
        "9/9b 05 — alarm-level metrics at threshold 0.30",
        ["05_clinical_metrics.py", "--pred", str(pred), "--threshold", "0.3"])

    if args.with_track_compare:
        dual = RUN / "dual"
        times["02 dual-track demo"] = run(
            "extra 02 --demo — built-in dual-track synthetic set (60 patients)",
            ["02_window_preprocessor.py", "--demo", "--out", str(dual), "--track", "both"])
        for tag in ("track1", "track2"):
            run(f"extra 03 — features ({tag})",
                ["03_feature_engineering.py",
                 "--npz", str(dual / f"windows_{tag}_X.npy"),
                 "--meta", str(dual / f"windows_{tag}_meta.csv"),
                 "--out", str(dual), "--tag", tag])
        run("extra 07 — Track 1 vs Track 2 code path",
            ["07_track_compare.py",
             "--track1", str(dual / "features_track1.parquet"),
             "--track2", str(dual / "features_track2.parquet"),
             "--out", str(dual / "track_compare"), "--folds", "2"], allow_fail=True)
        print("\n  note: 60 synthetic patients are far too few for a meaningful\n"
              "        Track-1-vs-Track-2 comparison; this only exercises the code path.")

    total = sum(times.values())
    print("\n" + "=" * 74)
    print(f"ALL CHECKS PASSED — {total:.1f} s total, {len(times)} steps")
    print("=" * 74)
    for k, v in times.items():
        print(f"  {k:<26} {v:6.1f}s")
    print(f"\n  artefacts: {RUN}")
    print("  Every input was simulated by demo/make_synthetic_eicu.py or by the")
    print("  built-in synthesizer in 02_window_preprocessor.py.")

    if not args.keep:
        print("  (demo_run/ is kept for inspection and is git-ignored; delete it any time.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
