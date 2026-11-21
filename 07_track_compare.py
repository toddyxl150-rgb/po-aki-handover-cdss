"""Track 1（纯 ICU 前 12h）vs Track 2（术中末 4h + 入 ICU 后 8h）对比。

研究问题
--------
INSPIRE 同时具备术中（负小时）与术后 ICU（非负小时）生命体征。若把「术中末 4h」
拼进窗口，模型能否在**更早**的时间点（t_end=+8 而非 +12）就给出同等甚至更好的判别力？

- Track 1：t ∈ [0, +12)，最早可用窗口 t_end = +12h
- Track 2：t ∈ [-4, +8)，-4..-1 为术中、0..7 为入 ICU 后，t_end = +8h

两条轨道都在预测「窗口结束后 24h 内是否发生 PO-AKI」，因此 Track 2 的预警天然早 4 小时。
本脚本在**同一批患者**上分别训练、交叉验证，比较 AUROC / AUPRC / 事件召回 / 中位提前量。

用法
----
python 07_track_compare.py \
    --track1 outputs/features_inspire.parquet \
    --track2 outputs/features_inspire_track2.parquet \
    --out outputs/track_compare --folds 5 --threshold 0.3
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("track_compare")


def _load_module(fname: str, name: str):
    p = Path(__file__).resolve().parent / fname
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, m)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def oof_predict(df: pd.DataFrame, cols: Sequence[str], folds: int,
                seed: int = 42) -> pd.DataFrame:
    """患者级 StratifiedGroupKFold 折外预测（同一患者绝不跨折）。"""
    from sklearn.model_selection import StratifiedGroupKFold

    m4 = _load_module("04_train_evaluate.py", "m04")
    X = m4.to_matrix(df, list(cols))
    y = df["y"].to_numpy(int)
    groups = df["patient_id"].astype(str).to_numpy()
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))

    out = np.full(len(df), np.nan, dtype=float)
    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(sgkf.split(X, y, groups)):
        p, _ = m4._fit_predict("LightGBM", X[tr], y[tr], X[va], seed=seed,
                               weight_pos=pos_weight)
        out[va] = p
        LOG.info("  fold %d/%d 完成（验证 %d 行，阳性 %d）", k + 1, folds, len(va),
                 int(y[va].sum()))
    res = df[["patient_id", "t_end_h", "y", "t_aki_onset", "followup_h"]].copy()
    res["pred_prob"] = out
    return res.dropna(subset=["pred_prob"]).reset_index(drop=True)


def summarise(name: str, pred: pd.DataFrame, m05, threshold: float) -> Dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = pred["y"].to_numpy(int)
    p = pred["pred_prob"].to_numpy(float)
    row: Dict = {"track": name, "n_windows": int(len(pred)),
                 "n_patients": int(pred["patient_id"].nunique()),
                 "pos_rate": float(y.mean()),
                 "t_end_median": float(pred["t_end_h"].median())}
    if 0 < y.sum() < y.size:
        row["auroc"] = float(roc_auc_score(y, p))
        row["auprc"] = float(average_precision_score(y, p))
    else:
        row["auroc"] = row["auprc"] = np.nan
    clin = m05.evaluate(pred.assign(alarm=pred["pred_prob"] >= threshold), threshold)
    for k in ("event_recall", "far", "median_lead_h", "sensitivity",
              "specificity", "ppv", "n_captured", "n_aki_patients"):
        if k in clin:
            row[k] = clin[k]
    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Track1 vs Track2 交接期对比")
    ap.add_argument("--track1", default="outputs/features_inspire.parquet")
    ap.add_argument("--track2", default="outputs/features_inspire_track2.parquet")
    ap.add_argument("--out", default="outputs/track_compare")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-patients", type=int, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    m4 = _load_module("04_train_evaluate.py", "m04")
    m05 = _load_module("05_clinical_metrics.py", "m05")

    t1 = pd.read_parquet(args.track1)
    t2 = pd.read_parquet(args.track2)
    t1["patient_id"] = t1["patient_id"].astype(str)
    t2["patient_id"] = t2["patient_id"].astype(str)

    # 只在**两条轨道都有窗口**的患者上比较，否则差异会混入人群构成
    common = sorted(set(t1["patient_id"]) & set(t2["patient_id"]))
    if args.max_patients:
        common = common[: args.max_patients]
    LOG.info("共同患者 %d（Track1 %d 人 / Track2 %d 人）", len(common),
             t1["patient_id"].nunique(), t2["patient_id"].nunique())
    t1 = t1[t1["patient_id"].isin(common)].reset_index(drop=True)
    t2 = t2[t2["patient_id"].isin(common)].reset_index(drop=True)

    cols = [c for c in m4.feature_columns(t1) if c in set(t2.columns)]
    LOG.info("Track1 %d 窗口 / Track2 %d 窗口 / 共同特征 %d",
             len(t1), len(t2), len(cols))

    rows: List[Dict] = []
    preds = {}
    for name, df in (("Track1_ICU_0-12h", t1), ("Track2_handover_-4-8h", t2)):
        LOG.info("---- %s ----", name)
        pred = oof_predict(df, cols, args.folds, args.seed)
        rows.append(summarise(name, pred, m05, args.threshold))
        preds[name] = pred
        pred.to_csv(out / f"oof_{name}.csv.gz", index=False, compression="gzip")

    res = pd.DataFrame(rows)
    res.to_csv(out / "track_comparison.csv", index=False, encoding="utf-8-sig")
    print("\n=== Track1 vs Track2（同一批患者，24h 视界）===")
    show = ["track", "n_windows", "n_patients", "pos_rate", "t_end_median",
            "auroc", "auprc", "event_recall", "median_lead_h", "sensitivity"]
    print(res[[c for c in show if c in res.columns]].round(4).to_string(index=False))
    print(f"\n-> {out/'track_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
