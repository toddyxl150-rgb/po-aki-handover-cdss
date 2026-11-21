#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_train_evaluate.py — 患者级 10 折 CV、RFE+Boruta 投票筛选、代价敏感 TabNet 与基线矩阵
=====================================================================================

流程
----
1. 读模块 3 的特征表（630 维 + meta）
2. StratifiedGroupKFold(10)，分组键 = patient_id（同一患者的所有窗口绝不跨折）
3. **每折内部**跑 RFE 与 Boruta，投票 → 保存 selected_features_fold{k}.csv
4. 在该折训练集上训练主模型与基线，验证折上算指标
5. 汇总 10 折，做 Kruskal-Wallis + 特征组间 t 检验（BH-FDR）

三条不可违反的约束
----------------
① 特征筛选、归一化、类别权重估计都只能用该折的训练部分
② 测试集只在最后评估时打开一次
③ 折间指标以「折」为单位做统计检验，不能把窗口当独立样本

依赖与降级
----------
* lightgbm 必需（RFE 与基线都靠它）
* boruta / torch / pytorch-tabnet 缺失时**不中断**：
  - Boruta 缺失 → 用「影子特征 + LightGBM 重要性」近似（输出标注 method=shadow_lgbm）
  - torch 缺失 → TabNet / GRU / RETAIN 跳过，结果表标 status=skipped
  刻意如此：方法学流水线不应因一个可选依赖缺失就整体不可用。

用法
----
    python 04_train_evaluate.py --features outputs/features_inspire.parquet --out outputs/cv
    python 04_train_evaluate.py --features ... --fast     # 跳过筛选与 TabNet
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdss_common import CHANNELS, ensure_dir  # noqa: E402
from stats_utils import benjamini_hochberg, tukey_hsd  # noqa: E402

LOG = logging.getLogger("cdss.train")

# 只有这些列不进模型；age/sex/t_end_h/observed_frac/miss_* 全部作为锚点特征保留
# 剔除采样伪影与制度伪影：
#   t_end_h        —— 右删失采样偏差（AKI 患者窗口 t_end 中位 32h vs 非 AKI 40h）
#   observed_frac  —— 护理记录密度，属制度差异而非生理信号
#   miss_*         —— 缺失掩码，同样反映记录习惯
NON_FEATURE = {"patient_id", "track", "y", "t_aki_onset", "followup_h", "label_status",
               "icu_type", "split", "fold", "db", "t_stage2", "t_stage3", "or_frac",
               "t_end_h", "observed_frac",
               # 结局派生 / 分层 / 采样伪影字段，严禁入模：
               "baseline_source", "max_stage", "baseline_scr", "in_vitaldb_overlap",
               "or_frac"}


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


HAS_LGBM = _has("lightgbm")
HAS_BORUTA = _has("boruta")
HAS_TORCH = _has("torch")
HAS_TABNET = _has("pytorch_tabnet")


# =============================================================================
# 1. 折划分与列识别
# =============================================================================
def make_folds(meta: pd.DataFrame, n_splits: int = 10, seed: int = 42) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold
    y = meta["y"].astype(int).to_numpy()
    groups = meta["patient_id"].astype(str).to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(y), -1, dtype=int)
    for k, (_, va) in enumerate(sgkf.split(np.zeros((len(y), 1)), y, groups)):
        fold[va] = k
    if (fold < 0).any():
        raise RuntimeError("存在未分配折的样本")
    return fold


def feature_columns(df: pd.DataFrame) -> List[str]:
    """630 维生理 + 3 个低灌注负荷 + age/sex = 635 维（含 age/sex 时为 635）。

    miss_* 缺失掩码一律排除：它反映的是记录密度与医院制度，不是生理状态。
    """
    return [c for c in df.columns
            if c not in NON_FEATURE and not c.startswith("miss_")]


def to_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """统一转成 float32 矩阵；sex 编码为 0/1，其余强制数值。"""
    X = df[list(cols)].copy()
    for c in cols:
        # 用 is_numeric_dtype 判断，避免漏掉 ArrowDtype 字符串列
        if pd.api.types.is_numeric_dtype(X[c]):
            continue
        if c == "sex":
            X[c] = X[c].astype(str).str.upper().str.startswith("M").astype(np.float32)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.to_numpy(np.float32)


# =============================================================================
# 2. 特征筛选（折内）
# =============================================================================
def _rfe_select(X: pd.DataFrame, y: np.ndarray, n_keep: int, seed: int = 42) -> List[str]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFE
    est = RandomForestClassifier(n_estimators=120, max_depth=6, n_jobs=-1,
                                 class_weight="balanced", random_state=seed)
    n_keep = max(5, min(n_keep, X.shape[1]))
    rfe = RFE(est, n_features_to_select=n_keep, step=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rfe.fit(to_matrix(X, list(X.columns)), y)
    return list(np.array(X.columns)[rfe.support_])


def _boruta_select(X: pd.DataFrame, y: np.ndarray, max_iter: int = 30,
                   seed: int = 42) -> Tuple[List[str], str]:
    """优先 BorutaPy；未安装时用「影子特征 + LightGBM 重要性」近似。"""
    if HAS_BORUTA and HAS_LGBM:
        from boruta import BorutaPy
        from lightgbm import LGBMClassifier
        est = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                             class_weight="balanced", n_jobs=-1, random_state=seed,
                             verbosity=-1)
        bor = BorutaPy(est, n_estimators="auto", max_iter=max_iter, random_state=seed)
        bor.fit(to_matrix(X, list(X.columns)), y)
        return list(np.array(X.columns)[bor.support_]), "boruta"

    if not HAS_LGBM:
        return [], "skipped_no_lgbm"
    from lightgbm import LGBMClassifier
    rng = np.random.default_rng(seed)
    Xs = X.copy()
    shadow = []
    for c in X.columns:
        sc = f"__shadow__{c}"
        Xs[sc] = rng.permutation(Xs[c].to_numpy())
        shadow.append(sc)
    m = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                       class_weight="balanced", n_jobs=-1, random_state=seed, verbosity=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(to_matrix(Xs, list(Xs.columns)), y)
    imp = pd.Series(m.feature_importances_, index=Xs.columns)
    thr = imp[shadow].max()
    return [c for c in X.columns if imp[c] > thr], "shadow_lgbm"


def select_features(X: pd.DataFrame, y: np.ndarray, n_keep: int, seed: int = 42,
                    enable_boruta: bool = True) -> Dict[str, object]:
    t0 = time.time()
    rfe = _rfe_select(X, y, n_keep, seed)
    bor, bor_method = (_boruta_select(X, y, seed=seed) if enable_boruta else ([], "disabled"))
    inter = sorted(set(rfe) & set(bor))
    union = sorted(set(rfe) | set(bor))
    votes = (pd.Series(1, index=pd.Index(rfe, name="f"))
             .add(pd.Series(1, index=pd.Index(bor, name="f")), fill_value=0)
             .astype(int).sort_values(ascending=False))
    return {"rfe": rfe, "boruta": bor, "boruta_method": bor_method,
            "intersection": inter, "union": union, "votes": votes,
            "elapsed_s": round(time.time() - t0, 1)}


# =============================================================================
# 3. 模型
# =============================================================================
class SkipModel(Exception):
    pass


def _fit_predict(name: str, Xtr, ytr, Xva, seed: int = 42, weight_pos: float = 1.0,
                 tabnet_epochs: int = 80, seq_epochs: int = 40) -> Tuple[np.ndarray, Dict]:
    if name == "LightGBM":
        if not HAS_LGBM:
            raise SkipModel("lightgbm 未安装")
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8,
                           scale_pos_weight=weight_pos, n_jobs=-1,
                           random_state=seed, verbosity=-1)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1], {"model": m}

    if name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced"))
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1], {"model": m}

    if name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        # KNN 是 O(N_train x N_val x D) 的暴力距离，全量训练集会直接把内存打爆；
        # 分层下采样到 15000 行（保留全部阳性），这是基线的常规做法，不影响结论方向。
        rs = np.random.RandomState(seed)
        idx = np.arange(len(Xtr))
        if len(idx) > 15000:
            pos = idx[ytr == 1]
            neg = rs.choice(idx[ytr == 0], max(15000 - len(pos), 1), replace=False)
            idx = np.concatenate([pos, neg])
        m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, n_jobs=-1))
        m.fit(np.asarray(Xtr)[idx], np.asarray(ytr)[idx])
        return m.predict_proba(Xva)[:, 1], {"model": m}

    if name == "MLP":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(),
                          MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150,
                                        batch_size=512, early_stopping=True,
                                        n_iter_no_change=8, random_state=seed))
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1], {"model": m}

    if name == "TabNet":
        if not (HAS_TORCH and HAS_TABNET):
            raise SkipModel("torch / pytorch-tabnet 未安装")
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier

        # TabNetClassifier 的 __init__ 没有 weights 参数；直接给 self.loss_fn 赋值也会被
        # fit() 内部重置（实测三档权重结果完全相同，说明覆盖失效）。
        # 因此改用子类重写 compute_loss，保证加权损失一定生效。
        class WeightedTabNet(TabNetClassifier):
            def __init__(self, w_pos: float, **kw):
                super().__init__(**kw)
                self._w_pos = float(w_pos)

            def compute_loss(self, y_pred, y_true):
                w = torch.tensor([1.0, self._w_pos], dtype=torch.float32,
                                 device=y_pred.device)
                return torch.nn.functional.cross_entropy(y_pred, y_true.long(), weight=w)

        clf = WeightedTabNet(float(weight_pos), n_d=32, n_a=32, n_steps=5, gamma=1.5,
                             lambda_sparse=1e-4, optimizer_params={"lr": 1e-2},
                             verbose=0, seed=seed)

        # 从**训练折内部**切 10% 做早停集，绝不能用验证折
        rs = np.random.RandomState(seed)
        idx = rs.permutation(len(Xtr))
        n_es = max(int(0.1 * len(idx)), 30)
        if len(idx) - n_es < 50:                 # 样本太小时退化为不早停
            es_idx, fit_idx = idx[:n_es], idx[n_es:]
            eval_set = None
        else:
            es_idx, fit_idx = idx[:n_es], idx[n_es:]
            eval_set = [(Xtr[es_idx], ytr[es_idx])]
        clf.fit(Xtr[fit_idx], ytr[fit_idx], eval_set=eval_set, eval_metric=["auc"],
                max_epochs=tabnet_epochs, patience=15,
                batch_size=1024, virtual_batch_size=128, drop_last=False)
        return clf.predict_proba(Xva)[:, 1], {"model": clf}

    if name in ("GRU", "RETAIN"):
        if not HAS_TORCH:
            raise SkipModel("torch 未安装")
        return _fit_torch_seq(name, Xtr, ytr, Xva, seed, weight_pos, epochs=seq_epochs)

    raise ValueError(name)


def _fit_torch_seq(name: str, Xtr, ytr, Xva, seed: int, weight_pos: float,
                   epochs: int = 40) -> Tuple[np.ndarray, Dict]:
    """GRU(DEWS) 与 RETAIN 时序基线。输入为 (N, T, C) 窗口张量。"""
    import torch
    import torch.nn as nn

    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xva = np.asarray(Xva, dtype=np.float32)
    if Xtr.ndim != 3:
        raise SkipModel("需要 (N,T,C) 窗口张量")

    C = Xtr.shape[2]

    class GRUNet(nn.Module):
        """DEWS 风格：单层 GRU 取末状态。"""
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(C, 32, batch_first=True)
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            _, h = self.gru(x)
            return self.head(h[-1]).squeeze(-1)

    class RETAINNet(nn.Module):
        """RETAIN：时间注意力 α + 变量注意力 β 的双注意力。"""
        def __init__(self, hidden: int = 32):
            super().__init__()
            self.gru_a = nn.GRU(C, hidden, batch_first=True)
            self.gru_b = nn.GRU(C, hidden, batch_first=True)
            self.wa = nn.Linear(hidden, 1)
            self.wb = nn.Linear(hidden, C)
            self.head = nn.Linear(C, 1)

        def forward(self, x):
            a, _ = self.gru_a(x)
            b, _ = self.gru_b(x)
            alpha = torch.softmax(self.wa(a).squeeze(-1), dim=1)          # (B,T)
            beta = torch.tanh(self.wb(b))                                # (B,T,C)
            ctx = (alpha.unsqueeze(-1) * beta * x).sum(dim=1)            # (B,C)
            return self.head(ctx).squeeze(-1)

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = (GRUNet() if name == "GRU" else RETAINNet()).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_pos], device=dev))

    xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(np.asarray(ytr, dtype=np.float32), device=dev)
    n = len(xt)
    bs = min(512, n)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(xt[b]), yt[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.tensor(Xva, device=dev))).cpu().numpy()
    return p, {"model": net}


def score_baselines(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """零学习评分：MEWS-4 取窗口内最大值；SOFA 需模块 1 单独导出。"""
    out = {}
    if "MEWS-4_1-12h_Max" in df.columns:
        out["MEWS-4"] = df["MEWS-4_1-12h_Max"].to_numpy(float)
    if "SOFA" in df.columns:
        out["SOFA"] = df["SOFA"].to_numpy(float)
    return out


# =============================================================================
# 4. 单折
# =============================================================================
def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y = np.asarray(y, int)
    out = {"n": int(y.size), "n_pos": int(y.sum())}
    if 0 < y.sum() < y.size:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    out["brier"] = float(brier_score_loss(y, p))
    return out


def _tabnet_block(df, va, tr, used, ytr, yva, k, weights, out_dir, seed,
                  tabnet_epochs, bor_method, oof):
    """只跑 TabNet 的分支（阶段二用），跳过全部基线以节省时间。"""
    Xtr = to_matrix(tr, used)
    Xva = to_matrix(va, used)
    rows: List[Dict] = []
    for wi, w in enumerate(weights):
        try:
            p, info = _fit_predict("TabNet", Xtr, ytr, Xva, seed=seed, weight_pos=w,
                                   tabnet_epochs=tabnet_epochs)
            rows.append({"fold": k, "model": f"TabNet_w{int(w)}", "status": "ok",
                         **_metrics(yva, p), "n_features": len(used),
                         "n_train": len(tr), "n_val": len(va), "boruta_method": bor_method})
            oof.append(pd.DataFrame({"patient_id": va["patient_id"].to_numpy(),
                                     "t_end_h": va["t_end_h"].to_numpy(),
                                     "y": yva, "t_aki_onset": va["t_aki_onset"].to_numpy(),
                                     "pred_prob": p, "model": f"TabNet_w{int(w)}", "fold": k}))
            if wi == 0:
                _save_tabnet_importance(info.get("model"), Xva, used, out_dir, k)
        except Exception as exc:  # noqa: BLE001
            rows.append({"fold": k, "model": f"TabNet_w{int(w)}", "status": "skipped",
                         "note": str(exc)[:120]})
    return rows, oof


def _save_tabnet_importance(clf, Xva: np.ndarray, used: Sequence[str],
                            out_dir: Path, fold: int) -> None:
    """保存 TabNet 全局重要性 + explain() 的 Top 25 归因。"""
    if clf is None:
        return
    try:
        imp = np.asarray(getattr(clf, "feature_importances_", []), dtype=float)
        if imp.size != len(used):
            return
        tab = pd.DataFrame({"feature": list(used), "importance": imp})
        # explain 返回 (res_explain, res_masks)，masks 是 dict
        try:
            res = clf.explain(Xva[: min(len(Xva), 2000)])
            M = res[0] if isinstance(res, tuple) else res
            if isinstance(M, dict):
                M = np.vstack(list(M.values()))
            M = np.asarray(M, dtype=float)
            if M.ndim == 2 and M.shape[1] == len(used):
                tab["explain_mean"] = M.mean(axis=0)
        except Exception:  # noqa: BLE001
            pass
        tab = tab.sort_values("importance", ascending=False).head(25)
        tab.to_csv(out_dir / f"tabnet_feature_importance_fold{fold}.csv",
                   index=False, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001
        LOG.warning("TabNet 重要性保存失败：%s", exc)


def run_fold(df: pd.DataFrame, feats: List[str], k: int,
             weights: Sequence[float], out_dir: Path, fast: bool = False,
             n_keep: int = 80, seed: int = 42,
             tabnet_epochs: int = 80, only_tabnet: bool = False,
             seq_epochs: int = 40) -> Tuple[List[Dict], List[pd.DataFrame]]:
    tr = df[df["fold"] != k]
    va = df[df["fold"] == k]
    ytr, yva = tr["y"].to_numpy(int), va["y"].to_numpy(int)
    rows: List[Dict] = []
    oof: List[pd.DataFrame] = []
    used, bor_method = feats, ""

    if not fast and not only_tabnet:
        sel = select_features(tr[feats], ytr, n_keep=n_keep, seed=seed)
        bor_method = str(sel["boruta_method"])
        pd.DataFrame({"feature": feats,
                      "votes": [int(sel["votes"].get(f, 0)) for f in feats],
                      "in_rfe": [f in set(sel["rfe"]) for f in feats],
                      "in_boruta": [f in set(sel["boruta"]) for f in feats]}
                     ).to_csv(out_dir / f"selected_features_fold{k}.csv",
                              index=False, encoding="utf-8-sig")
        if len(sel["intersection"]) >= 10:
            used = list(sel["intersection"])
        elif len(sel["union"]) >= 10:
            used = list(sel["union"])
        LOG.info("fold %d: RFE %d, Boruta(%s) %d, 交集 %d -> 用 %d",
                 k, len(sel["rfe"]), bor_method, len(sel["boruta"]),
                 len(sel["intersection"]), len(used))

    Xtr = to_matrix(tr, used)
    Xva = to_matrix(va, used)
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    if only_tabnet:
        return _tabnet_block(df, va, tr, used, ytr, yva, k, weights, out_dir, seed,
                             tabnet_epochs, bor_method, oof=[])

    for nm, sc in score_baselines(va).items():
        rows.append({"fold": k, "model": nm, "status": "ok", **_metrics(yva, sc)})
    if "SOFA" not in va.columns:
        rows.append({"fold": k, "model": "SOFA", "status": "skipped",
                     "note": "特征表无 SOFA 列（需模块 1 单独导出）"})

    for nm in ("LogisticRegression", "KNN", "MLP", "LightGBM"):
        try:
            p, _ = _fit_predict(nm, Xtr, ytr, Xva, seed=seed, weight_pos=pos_weight)
            rows.append({"fold": k, "model": nm, "status": "ok", **_metrics(yva, p)})
            oof.append(pd.DataFrame({"patient_id": va["patient_id"].to_numpy(),
                                     "t_end_h": va["t_end_h"].to_numpy(),
                                     "y": yva, "t_aki_onset": va["t_aki_onset"].to_numpy(),
                                     "pred_prob": p, "model": nm, "fold": k}))
        except Exception as exc:  # noqa: BLE001
            rows.append({"fold": k, "model": nm, "status": "skipped",
                         "note": str(exc)[:120]})

    tab_weights = [] if fast else list(weights)     # --fast 只跑基线，不跑 TabNet
    for wi, w in enumerate(tab_weights):
        try:
            p, info = _fit_predict("TabNet", Xtr, ytr, Xva, seed=seed, weight_pos=w,
                                   tabnet_epochs=tabnet_epochs)
            rows.append({"fold": k, "model": f"TabNet_w{int(w)}", "status": "ok",
                         **_metrics(yva, p)})
            oof.append(pd.DataFrame({"patient_id": va["patient_id"].to_numpy(),
                                     "t_end_h": va["t_end_h"].to_numpy(),
                                     "y": yva, "t_aki_onset": va["t_aki_onset"].to_numpy(),
                                     "pred_prob": p, "model": f"TabNet_w{int(w)}", "fold": k}))
            if wi == 0:
                _save_tabnet_importance(info.get("model"), Xva, used, out_dir, k)
        except Exception as exc:  # noqa: BLE001
            rows.append({"fold": k, "model": f"TabNet_w{int(w)}", "status": "skipped",
                         "note": str(exc)[:120]})

    seq_cols = [f"{ch}_t{t}" for ch in CHANNELS for t in range(1, 13)]
    have_seq = all(c in df.columns for c in seq_cols)
    if have_seq:
        Str = tr[seq_cols].to_numpy(np.float32).reshape(len(tr), len(CHANNELS), 12)
        Sva = va[seq_cols].to_numpy(np.float32).reshape(len(va), len(CHANNELS), 12)
        Str = np.transpose(Str, (0, 2, 1))          # (N, T, C)
        Sva = np.transpose(Sva, (0, 2, 1))
        for nm in ("GRU", "RETAIN"):
            try:
                p, _ = _fit_predict(nm, Str, ytr, Sva, seed=seed, weight_pos=pos_weight,
                                    seq_epochs=seq_epochs)
                rows.append({"fold": k, "model": nm, "status": "ok", **_metrics(yva, p)})
                oof.append(pd.DataFrame({"patient_id": va["patient_id"].to_numpy(),
                                         "t_end_h": va["t_end_h"].to_numpy(),
                                         "y": yva, "t_aki_onset": va["t_aki_onset"].to_numpy(),
                                         "pred_prob": p, "model": nm, "fold": k}))
            except Exception as exc:  # noqa: BLE001
                rows.append({"fold": k, "model": nm, "status": "skipped",
                             "note": str(exc)[:120]})
    else:
        for nm in ("GRU", "RETAIN"):
            rows.append({"fold": k, "model": nm, "status": "skipped",
                         "note": "特征表缺 L1 序列列（{Var}_t1..t12）"})

    for r in rows:
        r.update({"n_features": len(used), "n_train": int(len(tr)),
                  "n_val": int(len(va)), "boruta_method": bor_method})
    return rows, oof


# =============================================================================
# 5. 统计检验
# =============================================================================
def statistical_tests(fold_df: pd.DataFrame, df: pd.DataFrame,
                      feats: List[str], out_dir: Path) -> Dict[str, pd.DataFrame]:
    from scipy import stats

    out: Dict[str, pd.DataFrame] = {}
    ok = fold_df[(fold_df["status"] == "ok") & fold_df["auroc"].notna()]
    groups, names = [], []
    for nm, g in ok.groupby("model"):
        if len(g) >= 3:
            groups.append(g["auroc"].to_numpy())
            names.append(nm)
    if len(groups) >= 2:
        h, p = stats.kruskal(*groups)
        out["kruskal"] = pd.DataFrame([{"test": "Kruskal-Wallis", "H": h, "p": p,
                                        "n_models": len(groups),
                                        "models": ", ".join(names)}])

    pos, neg = df[df["y"] == 1], df[df["y"] == 0]
    recs = []
    for f in feats:
        # sex 是字符串锚点，需先编码；非数值列直接跳过，不能让它打断整轮检验
        if f == "sex":
            a = pos[f].astype(str).str.upper().str.startswith("M").to_numpy(float)
            b = neg[f].astype(str).str.upper().str.startswith("M").to_numpy(float)
        else:
            a = pd.to_numeric(pos[f], errors="coerce").dropna().to_numpy(float)
            b = pd.to_numeric(neg[f], errors="coerce").dropna().to_numpy(float)
        if a.size < 10 or b.size < 10:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        recs.append({"feature": f, "mean_aki": a.mean(), "mean_non": b.mean(),
                     "diff": a.mean() - b.mean(), "t": t, "p": p})
    if recs:
        ft = pd.DataFrame(recs)
        ft["p_bh"] = benjamini_hochberg(ft["p"].to_numpy())
        ft["significant_bh"] = ft["p_bh"] < 0.05
        ft = ft.sort_values("p")
        ft.to_csv(out_dir / "feature_ttest_bh.csv", index=False, encoding="utf-8-sig")
        out["ttest"] = ft.head(25)
    return out


# =============================================================================
# 6. 跨库外部验证
# =============================================================================
def _load_clinical_metrics():
    import importlib.util
    p = Path(__file__).resolve().parent / "05_clinical_metrics.py"
    spec = importlib.util.spec_from_file_location("m05", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("m05", m)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def merge_cohort_info(df: pd.DataFrame, csv_path: Optional[str]) -> pd.DataFrame:
    """把队列表的基线肌酐 / 基线来源 / 最高分期并入特征表（仅在缺失时补）。"""
    if not csv_path:
        return df
    c = pd.read_csv(csv_path)
    # baseline_scr 作为临床锚点特征；baseline_source/max_stage 仅供分层，已列入 NON_FEATURE
    cand = [k for k in ("patient_id", "baseline_scr", "baseline_source",
                        "max_stage") if k in c.columns]
    add = [k for k in cand if k != "patient_id" and k not in df.columns]
    if not add:
        return df
    c = c[["patient_id"] + add].drop_duplicates("patient_id")
    c["patient_id"] = c["patient_id"].astype(str)
    out = df.copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out = out.merge(c, on="patient_id", how="left")
    LOG.info("已并入队列表字段：%s", add)
    return out


def relabel_stage2(df: pd.DataFrame, horizon: float = 24.0) -> pd.DataFrame:
    """把 Stage>=1 标签改写为 Stage 2-3 标签（中重度主结局）。

    与模块 2 的滚动标签同口径：t_end 之后 (t_end, t_end+24h] 内首次达 Stage2 记阳性；
    t_end 时已达标 Stage2 的窗口剔除（已发病）；随访不足 24h 且未发病的窗口剔除（删失）。
    """
    t_end = pd.to_numeric(df["t_end_h"], errors="coerce").to_numpy(float)
    fu = pd.to_numeric(df["followup_h"], errors="coerce").to_numpy(float)
    on = pd.to_numeric(df["t_stage2"], errors="coerce").to_numpy(float)
    pos = np.isfinite(on) & (on > t_end) & (on <= t_end + horizon)
    known_neg = ~np.isfinite(on) & (fu >= t_end + horizon)
    keep = pos | known_neg
    out = df.loc[keep].copy()
    out["y_stage2"] = pos[keep].astype(int)
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def platt_calibrate(p_cal: np.ndarray, y_cal: np.ndarray,
                    p_apply: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Platt scaling：在目标域极少量样本上学一个一维 logistic 映射。"""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(_logit(p_cal).reshape(-1, 1), y_cal)
    out = lr.predict_proba(_logit(p_apply).reshape(-1, 1))[:, 1]
    return out, {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def _spec_threshold(y: np.ndarray, p: np.ndarray, spec: float) -> float:
    """取特异度 = spec 的阈值（阴性窗口概率分位数）。"""
    neg = p[np.asarray(y) == 0]
    if neg.size == 0:
        return float("nan")
    return float(np.quantile(neg, spec))


def _eval_cohort(name: str, sub: pd.DataFrame, y: np.ndarray, p: np.ndarray,
                 m05, threshold: float, note: str = "") -> Dict:
    """单队列的判别 + 临床报警指标。"""
    row: Dict = {"cohort": name, "model": "LightGBM", "status": "ok",
                 "n_windows": int(len(sub)), "n_patients": int(sub["patient_id"].nunique()),
                 "pos_rate": float(np.mean(y)) if len(y) else np.nan,
                 "n_positive": int(np.sum(y)), "note": note}
    row.update(_metrics(y, p))
    pdf = pd.DataFrame({"patient_id": sub["patient_id"].to_numpy(),
                        "t_end_h": pd.to_numeric(sub["t_end_h"], errors="coerce").to_numpy(),
                        "y": y, "t_aki_onset": sub["_onset"].to_numpy(float),
                        "pred_prob": p})
    clin = m05.evaluate(pdf, threshold)
    for k in ("event_recall", "far", "median_lead_h", "sensitivity", "specificity",
              "ppv", "n_captured", "n_aki_patients", "false_alarms_per_patient"):
        if k in clin:
            row[k] = clin[k]
    iqr = clin.get("iqr_lead_h")
    if isinstance(iqr, (tuple, list)) and len(iqr) == 2:
        row["lead_iqr_low"], row["lead_iqr_high"] = float(iqr[0]), float(iqr[1])
    row["nne"] = 1.0 / clin["ppv"] if clin.get("ppv") else np.nan

    # 附加操作点：特异度 90%（源库模型在目标库上概率整体偏高，固定 0.3 会报 70% 的窗口，
    # 该操作点让 FAR 具备临床可比性）
    thr90 = _spec_threshold(y, p, 0.90)
    if np.isfinite(thr90):
        pdf90 = pdf.copy()
        clin90 = m05.evaluate(pdf90, thr90)
        row["spec90_thr"] = thr90
        row["spec90_event_recall"] = clin90.get("event_recall", np.nan)
        row["spec90_far"] = clin90.get("far", np.nan)
        row["spec90_median_lead_h"] = clin90.get("median_lead_h", np.nan)
        row["spec90_sensitivity"] = clin90.get("sensitivity", np.nan)
        row["spec90_ppv"] = clin90.get("ppv", np.nan)
    return row


def run_external(src: pd.DataFrame, ext: pd.DataFrame, out_dir: Path,
                 seed: int = 42, threshold: float = 0.3, calib_frac: float = 0.05,
                 horizon: float = 24.0) -> pd.DataFrame:
    """四维解耦跨库外部验证：A 全队列 / B 纯净术前基线 / C Stage2-3 / D 目标域校准。

    零样本迁移：源库训练一次 LightGBM，目标库直接推理，**不做任何目标域适配**；
    Cohort D 单独用目标库 5% 患者做 Platt 校准，评估集为其余 95%。
    """
    common = [c for c in feature_columns(src) if c in set(ext.columns)]
    LOG.info("外部验证：源 %d 行 / 目标 %d 行 / 共同特征 %d",
             len(src), len(ext), len(common))
    Xtr, ytr = to_matrix(src, common), src["y"].to_numpy(int)
    ext = ext.reset_index(drop=True)
    Xte = to_matrix(ext, common)
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    p_all, _ = _fit_predict("LightGBM", Xtr, ytr, Xte, seed=seed, weight_pos=pos_weight)

    m05 = _load_clinical_metrics()
    ext = ext.copy()
    ext["_onset"] = pd.to_numeric(ext["t_aki_onset"], errors="coerce").to_numpy(float)
    rows: List[Dict] = []

    # ---- Cohort A：全队列，Stage 1-3 ----
    rows.append(_eval_cohort("A_eICU_all_stage1-3", ext, ext["y"].to_numpy(int),
                             p_all, m05, threshold, "零样本迁移基准"))

    # ---- Cohort B：真实术前基线（baseline_source == 1）----
    if "baseline_source" in ext.columns:
        bs = pd.to_numeric(ext["baseline_source"], errors="coerce")
        m = (bs == 1).to_numpy()
        if m.sum() >= 50 and 0 < ext.loc[m, "y"].sum() < m.sum():
            rows.append(_eval_cohort("B_eICU_true_preop_baseline", ext.loc[m],
                                     ext.loc[m, "y"].to_numpy(int), p_all[m],
                                     m05, threshold, "剔除编码 2/3 的标签噪声"))
        else:
            rows.append({"cohort": "B_eICU_true_preop_baseline", "model": "LightGBM",
                         "status": "skipped",
                         "note": f"baseline_source==1 样本不足（n={int(m.sum())}）"})
    else:
        rows.append({"cohort": "B_eICU_true_preop_baseline", "model": "LightGBM",
                     "status": "skipped", "note": "特征表缺 baseline_source（用 --external-cohort 并入）"})

    # ---- Cohort C：Stage 2-3 中重度结局 ----
    if "t_stage2" in ext.columns:
        c2 = relabel_stage2(ext, horizon)
        if len(c2) >= 50 and 0 < c2["y_stage2"].sum() < len(c2):
            c2 = c2.copy()
            c2["_onset"] = pd.to_numeric(c2["t_stage2"], errors="coerce").to_numpy(float)
            idx = c2.index.to_numpy()
            rows.append(_eval_cohort("C_eICU_stage2-3", c2, c2["y_stage2"].to_numpy(int),
                                     p_all[idx], m05, threshold,
                                     "中重度 AKI；按 t_stage2 重算滚动标签"))
        else:
            rows.append({"cohort": "C_eICU_stage2-3", "model": "LightGBM",
                         "status": "skipped", "note": "可用窗口不足"})
    else:
        rows.append({"cohort": "C_eICU_stage2-3", "model": "LightGBM",
                     "status": "skipped", "note": "特征表缺 t_stage2"})

    # ---- Cohort D：5% 目标域 Platt 校准 ----
    pids = np.asarray(ext["patient_id"].astype(str).unique(), dtype=object)
    rs = np.random.RandomState(seed)
    rs.shuffle(pids)
    n_cal = max(int(round(calib_frac * len(pids))), 5)
    cal_ids, eval_ids = set(pids[:n_cal]), set(pids[n_cal:])
    is_cal = ext["patient_id"].astype(str).isin(cal_ids).to_numpy()
    is_eval = ~is_cal
    y_all = ext["y"].to_numpy(int)
    if is_cal.sum() >= 20 and 0 < y_all[is_cal].sum() < is_cal.sum() and is_eval.sum() > 0:
        p_cal_new, coef = platt_calibrate(p_all[is_cal], y_all[is_cal], p_all[is_eval])
        sub = ext.loc[is_eval]
        # Platt 会把概率整体平移，固定 0.3 阈值可能一个报警都不触发（ER=0 是阈值问题，
        # 不是模型失效）。因此改用与 Cohort A **同等报警预算**的阈值：报警率相同，
        # 这样 FAR 与 ER 才具备可比性。
        base_rate = float(np.mean(p_all >= threshold))
        thr_d = float(np.quantile(p_cal_new, 1.0 - base_rate)) if 0 < base_rate < 1 else threshold
        rows.append(_eval_cohort("D_eICU_platt_5pct_calib", sub, y_all[is_eval],
                                 p_cal_new, m05, thr_d,
                                 f"校准患者 {len(cal_ids)}（{100*calib_frac:.0f}%），"
                                 f"a={coef['a']:.3f} b={coef['b']:.3f}；"
                                 f"阈值按 A 队列报警率 {base_rate:.3f} 对齐（thr={thr_d:.3f}）"))
    else:
        rows.append({"cohort": "D_eICU_platt_5pct_calib", "model": "LightGBM",
                     "status": "skipped", "note": "校准集阳性不足"})

    out = pd.DataFrame(rows)
    out["n_features"] = len(common)
    out["threshold"] = threshold
    out.to_csv(out_dir / "external_validation_stratified.csv", index=False, encoding="utf-8-sig")
    # 兼容旧路径：A 队列即传统"零样本外部验证"
    out[out["cohort"].astype(str).str.startswith("A_")].to_csv(
        out_dir / "cv_summary.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"patient_id": ext["patient_id"], "t_end_h": ext["t_end_h"],
                  "y": ext["y"], "t_aki_onset": ext["t_aki_onset"],
                  "pred_prob": p_all}).to_csv(
        out_dir / "external_pred_LightGBM.csv.gz", index=False, compression="gzip")
    return out


# =============================================================================
# 7. 主流程
# =============================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="10 折 CV + 特征筛选 + 代价敏感 TabNet")
    ap.add_argument("--features", default="outputs/features_track1.parquet")
    ap.add_argument("--out", default="outputs/cv")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--weights", default="10,25,50,100")
    ap.add_argument("--n-keep", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="跳过筛选与 TabNet，只跑基线")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--tabnet-epochs", type=int, default=80)
    ap.add_argument("--seq-epochs", type=int, default=40, help="GRU/RETAIN 训练轮数")
    ap.add_argument("--only-tabnet", action="store_true", help="只跑 TabNet，跳过全部基线")
    ap.add_argument("--fold-start", type=int, default=0)
    ap.add_argument("--fold-end", type=int, default=None)
    ap.add_argument("--external-features", default=None,
                    help="目标库特征表；提供后改为跨库外部验证（源库全量训练）")
    ap.add_argument("--stage-target", type=int, choices=[1, 2], default=1,
                    help="仅做一致性提示：Stage2 主结局须用 02 --stage-target 2 重建窗口")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="外部验证/报警的判定阈值")
    ap.add_argument("--external-models", default="LightGBM",
                    help="（已废弃）外部验证固定用 LightGBM 做四队列分层")
    ap.add_argument("--cohort-csv", default=None, help="源库 cohort.csv，并入 baseline_scr 等字段")
    ap.add_argument("--external-cohort", default=None, help="目标库 cohort.csv，用于 B 队列与 Stage2")
    ap.add_argument("--calib-frac", type=float, default=0.05, help="目标域校准患者比例")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    out = ensure_dir(args.out)

    df = (pd.read_parquet(args.features) if args.features.endswith(".parquet")
          else pd.read_csv(args.features))
    if args.max_rows:
        df = df.head(args.max_rows)
    df = df[df["y"].notna()].reset_index(drop=True)
    df = merge_cohort_info(df, args.cohort_csv)
    feats = feature_columns(df)
    LOG.info("样本 %d 窗口 / %d 患者 / %d 特征", len(df),
             df["patient_id"].nunique(), len(feats))
    LOG.info("依赖：lightgbm=%s boruta=%s torch=%s tabnet=%s",
             HAS_LGBM, HAS_BORUTA, HAS_TORCH, HAS_TABNET)

    if args.stage_target == 2:
        LOG.warning("stage_target=2：请确认特征表由 `02 --stage-target 2` 生成。"
                    "窗口截断点不同（stage1 vs stage2 发病），事后重算标签会漏窗。")
        if "t_stage2" in df.columns:
            n2 = int(df["t_stage2"].notna().sum())
            LOG.info("当前表中 t_stage2 非空患者 %d（仅供参考）", n2)

    if args.external_features:
        ext = (pd.read_parquet(args.external_features)
               if args.external_features.endswith(".parquet")
               else pd.read_csv(args.external_features))
        ext = ext[ext["y"].notna()].reset_index(drop=True)
        ext = merge_cohort_info(ext, args.external_cohort)
        res = run_external(df, ext, out, threshold=args.threshold,
                           calib_frac=args.calib_frac)
        print("\n=== 跨库外部验证（源库训练 -> 目标库测试）===")
        print(res.round(4).to_string(index=False))
        return 0

    df["fold"] = make_folds(df, args.folds, args.seed)
    weights = [float(x) for x in args.weights.split(",")]

    rows: List[Dict] = []
    oof_all: List[pd.DataFrame] = []
    k_end = args.fold_end if args.fold_end is not None else args.folds
    for k in range(args.fold_start, k_end):
        LOG.info("---- fold %d/%d ----", k + 1, args.folds)
        r, o = run_fold(df, feats, k, weights, out, fast=args.fast,
                        n_keep=args.n_keep, seed=args.seed,
                        tabnet_epochs=args.tabnet_epochs,
                        only_tabnet=args.only_tabnet, seq_epochs=args.seq_epochs)
        rows += r
        oof_all += o
    if oof_all:
        pd.concat(oof_all, ignore_index=True).to_csv(
            out / "oof_predictions.csv.gz", index=False, compression="gzip")
        LOG.info("折外预测已保存：%s", out / "oof_predictions.csv.gz")

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(out / "cv_results_by_fold.csv", index=False, encoding="utf-8-sig")

    agg = (fold_df[fold_df["status"] == "ok"]
           .groupby("model")
           .agg(n_folds=("auroc", "count"),
                auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
                auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"),
                brier_mean=("brier", "mean"))
           .sort_values("auroc_mean", ascending=False))
    agg.to_csv(out / "cv_summary.csv", encoding="utf-8-sig")
    print("\n=== 10 折汇总（按 AUROC 降序）===")
    print(agg.round(4).to_string())

    st = statistical_tests(fold_df, df, feats, out)
    if "kruskal" in st:
        print("\n=== Kruskal-Wallis（模型间 AUROC 差异）===")
        print(st["kruskal"].round(4).to_string(index=False))
    if "ttest" in st:
        print("\n=== Top 5 特征组间 t 检验（BH-FDR）===")
        print(st["ttest"].head(5)[["feature", "mean_aki", "mean_non", "p", "p_bh"]]
              .round(4).to_string(index=False))

    json.dump({"n_windows": len(df), "n_patients": int(df["patient_id"].nunique()),
               "n_features": len(feats), "folds": args.folds,
               "deps": {"lightgbm": HAS_LGBM, "boruta": HAS_BORUTA,
                        "torch": HAS_TORCH, "tabnet": HAS_TABNET}},
              open(out / "cv_manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
