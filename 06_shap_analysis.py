#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_shap_analysis.py — LightGBM 的 SHAP 全局归因
==============================================

主模型已定为 LightGBM，TabNet 的 `explain()` 不再承担解释职责。本脚本输出：

1. `lgbm_shap_importance.csv`  按 mean(|SHAP|) 降序的 Top 30 特征
2. `lgbm_shap_values.parquet`  测试集逐样本 SHAP 值（供 Summary / Beeswarm 图）
3. `shap_summary.png`          SHAP Summary Plot（matplotlib 可用时）

方法学要点
----------
* **按患者切分**（8:2），不是按窗口随机切分。同一患者的窗口高度相关，
  随机切分会让"解释集"与"训练集"共享患者，SHAP 值偏乐观。
* `TreeExplainer` 对 LightGBM 是精确解（不是采样近似），无需 KernelSHAP。
* 二分类时 `shap_values` 返回的是"阳性类"的贡献，符号为正表示推高风险。
* SHAP 解释的是**模型行为**，不是因果效应。论文里必须写清这一点。

用法
----
    python 06_shap_analysis.py --features outputs/features_inspire.parquet \
        --out outputs/shap --max-rows 40000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdss_common import ensure_dir  # noqa: E402
from stats_utils import self_test as _  # noqa: F401,E402  (确保同目录可导入)

LOG = logging.getLogger("cdss.shap")

# 与 04_train_evaluate.py 中 LightGBM 基线一致的"最优参数"
LGBM_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
                   subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbosity=-1)

# 剔除采样伪影与制度伪影：
#   t_end_h        —— 右删失采样偏差（AKI 患者窗口 t_end 中位 32h vs 非 AKI 40h）
#   observed_frac  —— 护理记录密度，属制度差异而非生理信号
#   miss_*         —— 缺失掩码，同样反映记录习惯
NON_FEATURE = {"patient_id", "track", "y", "t_aki_onset", "followup_h", "label_status",
               "icu_type", "split", "fold", "db", "t_stage2", "t_stage3", "or_frac",
               "t_end_h", "observed_frac",
               "baseline_source", "baseline_scr", "in_vitaldb_overlap", "max_stage"}


def feature_columns(df: pd.DataFrame) -> List[str]:
    """630 维纯生理 + 3 个低灌注负荷 + age/sex（共 633 维）。

    明确排除 t_end_h / observed_frac / or_frac / miss_* / baseline_source /
    in_vitaldb_overlap / baseline_scr，避免模型学习采样、记录制度与分层伪影。
    """
    return [c for c in df.columns
            if c not in NON_FEATURE and not c.startswith("miss_")]


def to_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    X = df[list(cols)].copy()
    for c in cols:
        if pd.api.types.is_numeric_dtype(X[c]):
            continue
        if c == "sex":
            X[c] = X[c].astype(str).str.upper().str.startswith("M").astype(np.float32)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.to_numpy(np.float32)


def patient_split(df: pd.DataFrame, test_frac: float = 0.2,
                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """按 patient_id 切分，返回 (train_idx, test_idx)。"""
    pids = df["patient_id"].astype(str).unique()
    rng = np.random.default_rng(seed)
    pids = np.asarray(pids, dtype=object)
    rng.shuffle(pids)
    n_test = max(int(len(pids) * test_frac), 1)
    test_pids = set(pids[:n_test])
    is_test = df["patient_id"].astype(str).isin(test_pids).to_numpy()
    return np.where(~is_test)[0], np.where(is_test)[0]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="LightGBM SHAP 归因")
    ap.add_argument("--features", default="outputs/features_inspire.parquet")
    ap.add_argument("--out", default="outputs/shap")
    ap.add_argument("--max-rows", type=int, default=40000)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suffix", default="", help="输出文件名后缀，如 _clean")
    ap.add_argument("--cohort-csv", default=None,
                    help="队列表，并入 baseline_scr 作为临床锚点特征")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    out = ensure_dir(args.out)

    df = (pd.read_parquet(args.features) if args.features.endswith(".parquet")
          else pd.read_csv(args.features))
    if args.max_rows:
        df = df.head(args.max_rows)
    df = df[df["y"].notna()].reset_index(drop=True)
    if args.cohort_csv:
        c = pd.read_csv(args.cohort_csv)
        if "baseline_scr" in c.columns and "baseline_scr" not in df.columns:
            c = c[["patient_id", "baseline_scr"]].drop_duplicates("patient_id")
            c["patient_id"] = c["patient_id"].astype(str)
            df = df.copy()
            df["patient_id"] = df["patient_id"].astype(str)
            df = df.merge(c, on="patient_id", how="left")
            LOG.info("已并入 baseline_scr 作为锚点特征")
    cols = feature_columns(df)
    LOG.info("样本 %d 窗口 / %d 患者 / %d 特征", len(df),
             df["patient_id"].nunique(), len(cols))

    tr_idx, te_idx = patient_split(df, args.test_frac, args.seed)
    Xtr = to_matrix(df.iloc[tr_idx], cols)
    Xte = to_matrix(df.iloc[te_idx], cols)
    ytr = df.iloc[tr_idx]["y"].to_numpy(int)
    yte = df.iloc[te_idx]["y"].to_numpy(int)
    LOG.info("训练 %d / 测试 %d（按患者切分，阳性率 %.3f / %.3f）",
             len(tr_idx), len(te_idx), ytr.mean(), yte.mean())

    import lightgbm as lgb
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    model = lgb.LGBMClassifier(scale_pos_weight=pos_weight, random_state=args.seed,
                               **LGBM_PARAMS)
    model.fit(Xtr, ytr)

    from sklearn.metrics import average_precision_score, roc_auc_score
    p = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p) if 0 < yte.sum() < yte.size else float("nan")
    ap_score = average_precision_score(yte, p) if 0 < yte.sum() < yte.size else float("nan")
    LOG.info("测试集 AUROC %.4f / AUPRC %.4f", auc, ap_score)

    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xte)
    # 二分类在不同版本可能返回 list([neg, pos]) 或 ndarray
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:            # (N, F, 2)
        sv = sv[:, :, 1]
    LOG.info("SHAP 矩阵 %s", sv.shape)

    mean_abs = np.abs(sv).mean(axis=0)
    imp = (pd.DataFrame({"feature": cols, "mean_abs_shap": mean_abs,
                         "mean_shap": sv.mean(axis=0)})
           .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    imp["rank"] = np.arange(1, len(imp) + 1)
    sfx = args.suffix
    imp.head(args.top_k).to_csv(out / f"lgbm_shap_importance{sfx}.csv",
                                index=False, encoding="utf-8-sig")
    imp.to_csv(out / f"lgbm_shap_importance_full{sfx}.csv", index=False, encoding="utf-8-sig")

    # 逐样本 SHAP（供作图）
    top = imp.head(args.top_k)["feature"].tolist()
    pos = [cols.index(c) for c in top]
    pd.DataFrame(sv[:, pos], columns=top).assign(y=yte).to_parquet(
        out / "lgbm_shap_values.parquet", index=False)
    df.iloc[te_idx][["patient_id", "t_end_h"]].reset_index(drop=True).to_csv(
        out / "shap_test_index.csv", index=False, encoding="utf-8-sig")

    # Summary plot（matplotlib 可用时）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 9))
        shap.summary_plot(sv[:, pos], Xte[:, pos], feature_names=top,
                          show=False, max_display=args.top_k)
        plt.tight_layout()
        plt.savefig(out / f"shap_summary{sfx}.png", dpi=300, bbox_inches="tight")
        plt.close()
        LOG.info("已保存 %s", out / f"shap_summary{sfx}.png")
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Summary plot 跳过（%s）；SHAP 值已存 parquet，可另作图", exc)

    print("\n=== Top 15 特征（mean |SHAP|）===")
    print(imp.head(15)[["rank", "feature", "mean_abs_shap", "mean_shap"]]
          .round(5).to_string(index=False))
    print(f"\n测试集 AUROC {auc:.4f} / AUPRC {ap_score:.4f}")
    print(f"-> {out/f'lgbm_shap_importance{sfx}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
