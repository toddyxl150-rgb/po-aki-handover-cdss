"""npj Digital Medicine 要求的精确统计量：Dunn 事后检验 + 患者级 Bootstrap 95% CI。

输出：
    outputs/paper_assets/stat_tests_npj.csv
    outputs/paper_assets/stat_tests_npj.md

方法要点
--------
* 模型两两比较：10 折 AUROC 的 Dunn 检验（Holm 校正，双侧）。
* Track 1 vs Track 2：同一批患者、每例一个预测 → 成对患者级 Bootstrap（1000 次，seed=42）；
  另附 DeLong 配对检验作为交叉验证。
* 外部队列 A–D：窗口级指标但按 `patient_id` 整群重抽样（cluster bootstrap），避免同一患者
  的多个窗口被当成独立样本而低估方差。
* 差值 P 值：双侧，取 Bootstrap 分布中跨越 0 的两尾比例（含 +1 校正）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path("outputs")
OUT = ROOT / "paper_assets"
N_BOOT = 1000
SEED = 42


# ---------------------------------------------------------------- 基础指标
def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    """秩和法 AUROC（O(n log n)）。概率连续，忽略并列。"""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    y_sorted = y[order]
    ranks = np.arange(1, len(y) + 1, dtype=float)
    return float((ranks[y_sorted == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fast_auprc(y: np.ndarray, p: np.ndarray) -> float:
    """AUPRC 一律用 sklearn 的 average_precision_score。

    早先这里用的是"每个阳性点的精度求和"的手写式，在**存在并列分数**时会把同一
    阈值组内的阳性重复计入，与 07/08 使用的 sklearn AP 产生 7e-5 量级偏差
    （Track 1: 0.47419 vs 0.47426，4285 名患者中有 123 个并列概率）。
    统计表与正文必须同口径，因此统一到 sklearn。
    """
    n1 = int(y.sum())
    if n1 == 0:
        return float("nan")
    return float(average_precision_score(y, p))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def recall_at_budget(y: np.ndarray, p: np.ndarray, thr: float,
                     t_end: np.ndarray, onset: np.ndarray,
                     horizon: float = 24.0) -> float:
    """患者级事件召回，与 `05_clinical_metrics.event_recall` 完全同口径。

    分母 = **全部发生 PO-AKI 的患者**（`t_aki_onset` 非空），而不是 `y==1` 的窗口。
    两者差异很大：某患者发病于 40 h，其 t_end=12 的窗口标签是 y=0（不在 24 h 视界内），
    但该患者仍应计入分母。用 y==1 作分母会把 ER 高估约 44%（实测 0.3645 vs 0.2535）。
    分子 = 在 [onset - horizon, onset) 内至少一次报警。
    """
    aki = np.isfinite(onset)
    n_aki = int(aki.sum())
    if n_aki == 0:
        return float("nan")
    captured = (p >= thr) & aki & (t_end >= onset - horizon) & (t_end < onset)
    return float(captured.sum() / n_aki)


def _ci(vals: Sequence[float]) -> Tuple[float, float]:
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _p_two_sided(deltas: Sequence[float]) -> float:
    d = np.asarray([x for x in deltas if np.isfinite(x)], dtype=float)
    if d.size == 0:
        return float("nan")
    le = int((d <= 0).sum())
    ge = int((d >= 0).sum())
    return float(min(1.0, 2 * (min(le, ge) + 1) / (d.size + 1)))


# ---------------------------------------------------------------- DeLong
def _delong_components(y: np.ndarray, p: np.ndarray) -> Tuple[float, np.ndarray]:
    """返回 (AUC, 每例的 placement value)，用于配对 DeLong 检验。"""
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    order = np.argsort(np.concatenate([pos, neg]))
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv), dtype=float)
    i = 0
    while i < len(allv):                      # 处理并列
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    rank_pos = ranks[order < m]
    rank_neg = ranks[order >= m]
    auc = (rank_pos.sum() - m * (m + 1) / 2) / (m * n)
    v10 = (rank_pos - np.array([(np.sum(neg < x) + 0.5 * np.sum(neg == x)) for x in pos])) / n
    v01 = 1 - (rank_neg - np.array([(np.sum(pos < x) + 0.5 * np.sum(pos == x)) for x in neg])) / m
    return float(auc), np.concatenate([v10, v01])


def delong_paired(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float]:
    a1, v1 = _delong_components(y, p1)
    a2, v2 = _delong_components(y, p2)
    diff = v1 - v2
    s = np.var(diff, ddof=1) / len(diff)
    if s <= 0:
        return float(a1 - a2), float("nan")
    z = (a1 - a2) / np.sqrt(s)
    from scipy import stats as st
    return float(a1 - a2), float(2 * (1 - st.norm.cdf(abs(z))))


# ---------------------------------------------------------------- 1) 模型比较
def model_comparison() -> pd.DataFrame:
    f = pd.read_csv(ROOT / "cv_clean/cv_results_by_fold.csv")
    ok = f[(f.status == "ok") & f.auroc.notna()]
    groups = {k: v["auroc"].to_numpy(float) for k, v in ok.groupby("model") if len(v) >= 3}

    spec = importlib.util.spec_from_file_location("m05", "05_clinical_metrics.py")
    m05 = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("m05", m05)
    spec.loader.exec_module(m05)  # type: ignore[union-attr]
    dunn = m05.dunn_test(groups)

    ref = "LightGBM"
    rows = []
    for model, arr in groups.items():
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        p_holm = np.nan
        if model != ref:
            m = dunn[((dunn.group1 == ref) & (dunn.group2 == model)) |
                     ((dunn.group1 == model) & (dunn.group2 == ref))]
            p_holm = float(m.p_holm.iloc[0]) if len(m) else np.nan
        rows.append({
            "comparison": f"{ref} vs {model}" if model != ref else f"{model} (reference)",
            "model": model, "n_folds": len(arr),
            "auroc_median": med, "auroc_q1": q1, "auroc_q3": q3,
            "auroc_mean": float(arr.mean()), "auroc_sd": float(arr.std(ddof=1)),
            "dunn_p_holm": p_holm,
        })
    return pd.DataFrame(rows).sort_values("auroc_median", ascending=False)


# ---------------------------------------------------------------- 2) Track 1 vs 2
def track_comparison() -> Tuple[pd.DataFrame, pd.DataFrame]:
    t2 = pd.read_csv(ROOT / "track_compare/oof_Track2_handover_-4-8h.csv.gz")
    t1 = pd.read_csv(ROOT / "track_compare/oof_Track1_ICU_0-12h.csv.gz")
    common = set(t2["patient_id"]) & set(t1["patient_id"])
    t1 = (t1[t1["patient_id"].isin(common)]
          .sort_values(["patient_id", "t_end_h"]).groupby("patient_id").head(1))
    t2 = t2[t2["patient_id"].isin(common)]
    key = ["patient_id"]
    m = t1[key + ["y", "pred_prob", "t_end_h", "t_aki_onset"]].merge(
        t2[key + ["y", "pred_prob", "t_end_h", "t_aki_onset"]], on=key, suffixes=("_1", "_2"))
    y1, p1 = m["y_1"].to_numpy(int), m["pred_prob_1"].to_numpy(float)
    y2, p2 = m["y_2"].to_numpy(int), m["pred_prob_2"].to_numpy(float)
    te1 = m["t_end_h_1"].to_numpy(float); on1 = m["t_aki_onset_1"].to_numpy(float)
    te2 = m["t_end_h_2"].to_numpy(float); on2 = m["t_aki_onset_2"].to_numpy(float)

    thr1 = float(np.quantile(p1, 0.90))
    thr2 = float(np.quantile(p2, 0.90))

    metrics = {
        "AUROC": (lambda y, p: fast_auc(y, p)),
        "AUPRC": (lambda y, p: fast_auprc(y, p)),
        "Recall@10%": None,
    }

    def compute(y, p, thr, name, te, on):
        return (recall_at_budget(y, p, thr, te, on) if name == "Recall@10%"
                else metrics[name](y, p))

    rng = np.random.RandomState(SEED)
    n = len(m)
    boot = {k: {"t1": [], "t2": [], "delta": []} for k in metrics}
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        for name in metrics:
            v1 = compute(y1[idx], p1[idx], thr1, name, te1[idx], on1[idx])
            v2 = compute(y2[idx], p2[idx], thr2, name, te2[idx], on2[idx])
            boot[name]["t1"].append(v1)
            boot[name]["t2"].append(v2)
            boot[name]["delta"].append(v2 - v1)

    rows = []
    for name in metrics:
        v1 = compute(y1, p1, thr1, name, te1, on1)
        v2 = compute(y2, p2, thr2, name, te2, on2)
        lo1, hi1 = _ci(boot[name]["t1"])
        lo2, hi2 = _ci(boot[name]["t2"])
        dlo, dhi = _ci(boot[name]["delta"])
        p_boot = _p_two_sided(boot[name]["delta"])
        rows.append({
            "metric": name, "track1": v1, "track1_lo": lo1, "track1_hi": hi1,
            "track2": v2, "track2_lo": lo2, "track2_hi": hi2,
            "delta": v2 - v1, "delta_lo": dlo, "delta_hi": dhi,
            "p_bootstrap": p_boot, "p_delong": np.nan, "n_patients": n,
        })
    tab = pd.DataFrame(rows)

    d_auc, p_delong = delong_paired(y1, p1, p2)
    tab.loc[tab.metric == "AUROC", "p_delong"] = p_delong
    return tab, pd.DataFrame({"delta_auroc_delong": [d_auc], "p": [p_delong]})


# ---------------------------------------------------------------- 3) 外部 A–D
def _external_frames() -> Dict[str, pd.DataFrame]:
    ext = pd.read_csv(ROOT / "cv_external_clean/external_pred_LightGBM.csv.gz")
    ext["patient_id"] = ext["patient_id"].astype(str)
    coh = pd.read_csv(ROOT / "cohort_eicu.csv",
                      usecols=["patient_id", "baseline_source", "followup_h", "t_stage2"])
    coh["patient_id"] = coh["patient_id"].astype(str)
    ext = ext.merge(coh, on="patient_id", how="left", suffixes=("", "_c"))
    if "t_stage2" not in ext.columns:
        ext["t_stage2"] = np.nan

    out: Dict[str, pd.DataFrame] = {}
    out["A"] = ext.copy()
    out["B"] = ext[ext["baseline_source"] == 1].copy()

    # Cohort C：按 t_stage2 重算 24h 滚动标签
    horizon = 24.0
    t_end = ext["t_end_h"].to_numpy(float)
    fu = ext["followup_h"].to_numpy(float)
    on = pd.to_numeric(ext["t_stage2"], errors="coerce").to_numpy(float)
    pos = np.isfinite(on) & (on > t_end) & (on <= t_end + horizon)
    known_neg = ~np.isfinite(on) & (fu >= t_end + horizon)
    keep = pos | known_neg
    c = ext.loc[keep].copy()
    c["y"] = pos[keep].astype(int)
    out["C"] = c

    # Cohort D：5% 患者 Platt 校准，评估其余 95%
    pids = np.asarray(ext["patient_id"].unique(), dtype=object)
    rs = np.random.RandomState(SEED)
    rs.shuffle(pids)
    n_cal = max(int(round(0.05 * len(pids))), 5)
    cal = set(pids[:n_cal])
    is_cal = ext["patient_id"].isin(cal).to_numpy()
    z = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))  # noqa: E731
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(
        z(ext["pred_prob"].to_numpy(float)[is_cal]).reshape(-1, 1),
        ext["y"].to_numpy(int)[is_cal])
    d = ext.loc[~is_cal].copy()
    d["pred_prob"] = lr.predict_proba(z(d["pred_prob"].to_numpy(float)).reshape(-1, 1))[:, 1]
    out["D"] = d
    out["_D_cal_ids"] = ext.loc[is_cal, ["patient_id"]].drop_duplicates()
    return out


def external_bootstrap() -> pd.DataFrame:
    fr = _external_frames()
    rng = np.random.RandomState(SEED)
    rows = []
    store = {}
    for name in ("A", "B", "C", "D"):
        d = fr[name]
        y = d["y"].to_numpy(int)
        p = d["pred_prob"].to_numpy(float)
        pid = d["patient_id"].to_numpy()
        groups = pd.Series(np.arange(len(d))).groupby(pid).apply(np.asarray).to_dict()
        keys = list(groups.keys())
        point = {"auroc": fast_auc(y, p), "auprc": fast_auprc(y, p), "brier": brier(y, p)}
        boot = {"auroc": [], "auprc": [], "brier": []}
        for _ in range(N_BOOT):
            pick = rng.choice(len(keys), len(keys), replace=True)
            idx = np.concatenate([groups[keys[k]] for k in pick])
            yy, pp = y[idx], p[idx]
            boot["auroc"].append(fast_auc(yy, pp))
            boot["auprc"].append(fast_auprc(yy, pp))
            boot["brier"].append(brier(yy, pp))
        store[name] = {"y": y, "p": p, "pid": pid, "groups": groups, "keys": keys}
        for metric in ("auroc", "auprc", "brier"):
            lo, hi = _ci(boot[metric])
            rows.append({"cohort": name, "metric": metric, "value": point[metric],
                         "ci_lo": lo, "ci_hi": hi,
                         "n_windows": len(d), "n_patients": d["patient_id"].nunique()})
    tab = pd.DataFrame(rows)

    # ---- 差值：A vs B（独立重抽样）、A vs D（同一批患者的配对比较）----
    def cluster_resample(name, rng):
        s = store[name]
        pick = rng.choice(len(s["keys"]), len(s["keys"]), replace=True)
        idx = np.concatenate([s["groups"][s["keys"][k]] for k in pick])
        return s["y"][idx], s["p"][idx]

    diffs = []
    for metric in ("auroc", "auprc"):
        deltas = []
        for _ in range(N_BOOT):
            ya, pa = cluster_resample("A", rng)
            yb, pb = cluster_resample("B", rng)
            deltas.append(fast_auc(ya, pa) - fast_auc(yb, pb) if metric == "auroc"
                          else fast_auprc(ya, pa) - fast_auprc(yb, pb))
        va = tab[(tab.cohort == "A") & (tab.metric == metric)].value.iloc[0]
        vb = tab[(tab.cohort == "B") & (tab.metric == metric)].value.iloc[0]
        lo, hi = _ci(deltas)
        diffs.append({"contrast": "A vs B", "metric": metric, "delta": va - vb,
                      "ci_lo": lo, "ci_hi": hi, "p_bootstrap": _p_two_sided(deltas)})

    # A vs D：D 是 A 的子集，用 D 的评估患者做配对重抽样
    sA, sD = store["A"], store["D"]
    d_pids = set(sD["keys"])
    maskA = np.isin(sA["pid"], list(d_pids))
    yA, pA, pidA = sA["y"][maskA], sA["p"][maskA], sA["pid"][maskA]
    gA = pd.Series(np.arange(len(yA))).groupby(pidA).apply(np.asarray).to_dict()
    keys = list(gA.keys())
    deltas = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idxA = np.concatenate([gA[keys[k]] for k in pick])
        idxD = np.concatenate([sD["groups"][keys[k]] for k in pick])
        deltas.append(brier(sD["y"][idxD], sD["p"][idxD]) - brier(yA[idxA], pA[idxA]))
    bA = brier(yA, pA)
    bD = brier(sD["y"], sD["p"])
    lo, hi = _ci(deltas)
    diffs.append({"contrast": "A vs D (Brier)", "metric": "brier", "delta": bD - bA,
                  "ci_lo": lo, "ci_hi": hi, "p_bootstrap": _p_two_sided(deltas)})
    return tab, pd.DataFrame(diffs)


# ---------------------------------------------------------------- 输出
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    models = model_comparison()
    track, delong = track_comparison()
    ext, diffs = external_bootstrap()

    models = models.assign(analysis="model_comparison_10fold")
    track = track.assign(analysis="track1_vs_track2_paired_bootstrap")
    ext = ext.assign(analysis="external_cohorts_cluster_bootstrap")
    diffs = diffs.assign(analysis="external_contrasts_cluster_bootstrap")

    allrows = pd.concat([models, track, ext, diffs], ignore_index=True, sort=False)
    allrows.to_csv(OUT / "stat_tests_npj.csv", index=False, encoding="utf-8-sig")

    md = ["# npj Digital Medicine 统计检验补充表", "",
          f"> Bootstrap 次数 {N_BOOT}，seed = {SEED}；差值 P 值为双侧 Bootstrap 尾概率，"
          "AUROC 另附 DeLong 配对检验作交叉验证。", ""]
    md += ["## 1. 内部 10 折模型比较（Dunn, Holm 校正）", "",
           "| Model | AUROC median [IQR] | Mean ± SD | Dunn P vs LightGBM (Holm) |",
           "|---|---|---:|---:|"]
    for _, r in models.iterrows():
        p = "—" if pd.isna(r["dunn_p_holm"]) else f"{r['dunn_p_holm']:.3g}"
        md.append(f"| {r['model']} | {r['auroc_median']:.4f} "
                  f"[{r['auroc_q1']:.4f}–{r['auroc_q3']:.4f}] | "
                  f"{r['auroc_mean']:.4f} ± {r['auroc_sd']:.4f} | {p} |")
    md += ["", "## 2. Track 1 vs Track 2（成对患者级 Bootstrap）", "",
           "| Metric | Track 1 (95% CI) | Track 2 (95% CI) | Δ (95% CI) | P (bootstrap) | P (DeLong) |",
           "|---|---|---|---|---:|---:|"]
    for _, r in track.iterrows():
        dl = "—" if pd.isna(r["p_delong"]) else f"{r['p_delong']:.3g}"
        md.append(f"| {r['metric']} | {r['track1']:.4f} ({r['track1_lo']:.4f}–{r['track1_hi']:.4f}) "
                  f"| {r['track2']:.4f} ({r['track2_lo']:.4f}–{r['track2_hi']:.4f}) "
                  f"| {r['delta']:+.4f} ({r['delta_lo']:+.4f}–{r['delta_hi']:+.4f}) "
                  f"| {r['p_bootstrap']:.3g} | {dl} |")
    md += ["", "## 3. 外部验证队列 A–D（患者级整群 Bootstrap）", "",
           "| Cohort | Metric | Value (95% CI) | Windows / patients |", "|---|---|---|---:|"]
    for _, r in ext.iterrows():
        md.append(f"| {r['cohort']} | {r['metric']} | {r['value']:.4f} "
                  f"({r['ci_lo']:.4f}–{r['ci_hi']:.4f}) | {r['n_windows']:,} / {r['n_patients']:,} |")
    md += ["", "### 队列间对比", "", "| Contrast | Metric | Δ (95% CI) | P |", "|---|---|---|---:|"]
    for _, r in diffs.iterrows():
        md.append(f"| {r['contrast']} | {r['metric']} | {r['delta']:+.4f} "
                  f"({r['ci_lo']:+.4f}–{r['ci_hi']:+.4f}) | {r['p_bootstrap']:.3g} |")
    (OUT / "stat_tests_npj.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(models.round(4).to_string(index=False))
    print()
    print(track.round(4).to_string(index=False))
    print()
    print(ext.round(4).to_string(index=False))
    print()
    print(diffs.round(4).to_string(index=False))
    print(f"\n-> {OUT/'stat_tests_npj.csv'}\n-> {OUT/'stat_tests_npj.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
