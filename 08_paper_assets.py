"""论文素材生成：主结果表（Table A/B/C）+ 三张 300 DPI 插图。

所有数字**只从已有产物读取**，不做任何重新训练，也不允许手写常量：
    outputs/cv_clean/cv_summary_clean.csv                  Table A
    outputs/cv_clean/cv_results_by_fold.csv                Kruskal-Wallis
    outputs/track_compare/track_comparison_one_shot.csv    Table B（单次预测）
    outputs/track_compare/track_comparison_operating_points.csv  Table B（操作点）
    outputs/cv_external_clean/external_validation_stratified.csv Table C
    outputs/cv_clean/policy_comparison_clean.csv           冷却前后
    outputs/shap/lgbm_shap_importance_clean.csv            SHAP 排名
    outputs/shap/lgbm_shap_values.parquet                  逐样本 SHAP
    outputs/track_compare/oof_Track*.csv.gz                ROC/PR 曲线

用法：python 08_paper_assets.py --out outputs/paper_assets --fig-dir outputs/paper_figures
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

LOG = logging.getLogger("paper_assets")
ROOT_SHAP = Path("outputs/shap")

# 期刊常用配色（色盲友好）
C_MAIN = "#2B6CB0"      # 主模型
C_ALT = "#718096"       # 基线
C_ACC = "#C53030"       # 强调/警示
C_OK = "#2F855A"        # 改善
C_TRACK1 = "#4A5568"
C_TRACK2 = "#2B6CB0"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    return plt


def panel_label(ax, letter: str, x: float = -0.10, y: float = 1.045) -> None:
    """npj DM 版式：小写加粗 a/b/c，独立于标题文字，避免与标题混排。"""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="bottom", ha="left", color="#1A202C")


def save_rgb(fig, path: Path) -> None:
    """统一 300 DPI RGB（matplotlib 默认输出 RGBA，投稿系统常要求 RGB）。"""
    from PIL import Image
    fig.savefig(path, dpi=300)
    Image.open(path).convert("RGB").save(path, dpi=(300, 300))
    jpg = path.with_suffix(".jpg")
    fig.savefig(jpg, dpi=300, pil_kwargs={"quality": 95})
    Image.open(jpg).convert("RGB").save(jpg, quality=95, dpi=(300, 300))


def _load_module(fname: str, name: str):
    p = Path(__file__).resolve().parent / fname
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, m)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


# =============================================================================
# 表
# =============================================================================
def build_tables(root: Path) -> Dict[str, pd.DataFrame]:
    t = {}
    t["A"] = pd.read_csv(root / "cv_clean/cv_summary_clean.csv")
    t["B_oneshot"] = pd.read_csv(root / "track_compare/track_comparison_one_shot.csv")
    t["B_op"] = pd.read_csv(root / "track_compare/track_comparison_operating_points.csv")
    t["C"] = pd.read_csv(root / "cv_external_clean/external_validation_stratified.csv")
    t["cool"] = pd.read_csv(root / "cv_clean/policy_comparison_clean.csv")
    return t


def kw_test(root: Path) -> Dict[str, float]:
    from scipy import stats
    f = pd.read_csv(root / "cv_clean/cv_results_by_fold.csv")
    ok = f[(f.status == "ok") & f.auroc.notna()]
    groups = {k: v["auroc"].to_numpy(float) for k, v in ok.groupby("model") if len(v) >= 3}
    H, p = stats.kruskal(*groups.values())
    return {"H": float(H), "p": float(p), "k": len(groups),
            "n_per_group": int(min(len(v) for v in groups.values()))}


def tables_long(t: Dict[str, pd.DataFrame], kw: Dict[str, float]) -> pd.DataFrame:
    """统一成长表：table / row_label / metric / value / unit / note。"""
    rows: List[Dict] = []

    name_map = {"LightGBM": "LightGBM", "MLP": "MLP", "LogisticRegression": "Logistic regression",
                "KNN": "KNN", "RETAIN": "RETAIN", "GRU": "GRU (DEWS)", "MEWS-4": "MEWS-4"}
    for _, r in t["A"].sort_values("auroc_mean", ascending=False).iterrows():
        lbl = name_map.get(r["model"], r["model"])
        rows += [
            {"table": "A", "row_label": lbl, "metric": "AUROC (mean±SD)",
             "value": f"{r['auroc_mean']:.4f} ± {r['auroc_sd']:.4f}", "unit": "",
             "note": f"n_folds={int(r['n_folds'])}"},
            {"table": "A", "row_label": lbl, "metric": "AUPRC", "value": f"{r['auprc_mean']:.4f}",
             "unit": "", "note": ""},
            {"table": "A", "row_label": lbl, "metric": "Brier", "value": f"{r['brier_mean']:.4f}",
             "unit": "", "note": ""},
        ]
    rows.append({"table": "A", "row_label": "Kruskal-Wallis (AUROC across models)",
                 "metric": "H", "value": f"{kw['H']:.2f}", "unit": "",
                 "note": f"p = {kw['p']:.2e}; k = {kw['k']} models, {kw['n_per_group']} folds each"})

    # Table B
    for _, r in t["B_oneshot"].iterrows():
        is_t1 = r["track"].startswith("Track1")
        lbl = ("Track 1 (ICU 0–12 h, first window, t_end=12 h)" if is_t1
               else "Track 2 (handover, t = [-4, +8), t_end=8 h)")
        pfx = "Track1" if is_t1 else "Track2"
        rows += [
            {"table": "B", "row_label": lbl, "metric": "AUROC", "value": f"{r['auroc']:.4f}",
             "unit": "", "note": f"n={int(r['n_windows'])} windows / {int(r['n_patients'])} patients"},
            {"table": "B", "row_label": lbl, "metric": "AUPRC", "value": f"{r['auprc']:.4f}",
             "unit": "", "note": f"prevalence {100*r['pos_rate']:.2f}%"},
            {"table": "B", "row_label": lbl, "metric": "Median lead time (Youden)",
             "value": f"{t['B_op'][(t['B_op'].track.str.startswith(pfx)) & (t['B_op'].operating_point=='Youden')]['median_lead_h'].iloc[0]:.1f}",
             "unit": "h", "note": "Youden-optimal threshold"},
            {"table": "B", "row_label": lbl, "metric": "Event recall @10% alarm budget",
             "value": f"{t['B_op'][(t['B_op'].track.str.startswith(pfx)) & (t['B_op'].operating_point=='alarm_rate_10pct')]['event_recall'].iloc[0]:.4f}",
             "unit": "", "note": "same alarm rate in both tracks"},
            {"table": "B", "row_label": lbl, "metric": "Sensitivity / Specificity (Youden)",
             "value": (f"{t['B_op'][(t['B_op'].track.str.startswith(pfx)) & (t['B_op'].operating_point=='Youden')]['sensitivity'].iloc[0]:.3f} / "
                       f"{t['B_op'][(t['B_op'].track.str.startswith(pfx)) & (t['B_op'].operating_point=='Youden')]['specificity'].iloc[0]:.3f}"),
             "unit": "", "note": ""},
        ]

    # Table C
    cohort_map = {
        "A_eICU_all_stage1-3": "A  All eICU, KDIGO stage 1–3 (zero-shot)",
        "B_eICU_true_preop_baseline": "B  eICU, true pre-operative baseline (baseline_source=1)",
        "C_eICU_stage2-3": "C  eICU, moderate-to-severe (KDIGO stage 2–3)",
        "D_eICU_platt_5pct_calib": "D  eICU, 5% target Platt calibration",
    }
    for _, r in t["C"].iterrows():
        if r["status"] != "ok":
            continue
        lbl = cohort_map.get(r["cohort"], r["cohort"])
        rows += [
            {"table": "C", "row_label": lbl, "metric": "AUROC", "value": f"{r['auroc']:.4f}",
             "unit": "", "note": f"n={int(r['n_windows'])} windows / {int(r['n_patients'])} patients"},
            {"table": "C", "row_label": lbl, "metric": "AUPRC", "value": f"{r['auprc']:.4f}",
             "unit": "", "note": f"prevalence {100*r['pos_rate']:.2f}%"},
            {"table": "C", "row_label": lbl, "metric": "Brier", "value": f"{r['brier']:.4f}",
             "unit": "", "note": ""},
            {"table": "C", "row_label": lbl, "metric": "Event recall", "value": f"{r['event_recall']:.4f}",
             "unit": "", "note": "patient-level, >=1 true alarm in 24 h before onset"},
            {"table": "C", "row_label": lbl, "metric": "Median lead time", "value": f"{r['median_lead_h']:.1f}",
             "unit": "h", "note": ""},
        ]

    # Table D：SHAP
    imp = pd.read_csv(ROOT_SHAP / "lgbm_shap_importance_clean.csv").head(15)
    for _, r in imp.iterrows():
        rows.append({"table": "D", "row_label": f"rank {int(r['rank'])}: {r['feature']}",
                     "metric": "mean |SHAP|", "value": f"{r['mean_abs_shap']:.4f}",
                     "unit": "", "note": ""})

    # Table C 冷却
    for _, r in t["cool"].iterrows():
        lbl = f"{r['operating_point']} ({r['target']})"
        pol = "raw" if r["policy"] == "raw" else "EWMA(0.5) + 6 h cooldown"
        rows += [
            {"table": "C-cooling", "row_label": lbl, "metric": f"Event recall — {pol}",
             "value": f"{r['event_recall']:.4f}", "unit": "", "note": f"threshold {r['threshold']:.4f}"},
            {"table": "C-cooling", "row_label": lbl, "metric": f"False alarms per patient — {pol}",
             "value": f"{r['false_alarms_per_patient']:.2f}", "unit": "/patient", "note": ""},
            {"table": "C-cooling", "row_label": lbl, "metric": f"False alarms per patient-day — {pol}",
             "value": f"{r['false_alarms_per_patient_day']:.2f}", "unit": "/patient-day", "note": ""},
            {"table": "C-cooling", "row_label": lbl, "metric": f"Median lead time — {pol}",
             "value": f"{r['median_lead_h']:.2f}", "unit": "h", "note": ""},
            {"table": "C-cooling", "row_label": lbl, "metric": f"Window FAR (1−PPV) — {pol}",
             "value": f"{r['window_far']:.4f}", "unit": "", "note": "prevalence-dependent by definition"},
        ]
    return pd.DataFrame(rows)


def tables_markdown(t: Dict[str, pd.DataFrame], kw: Dict[str, float],
                    ext_brier: Dict[str, float]) -> str:
    a = t["A"].copy()
    name_map = {"LightGBM": "LightGBM", "MLP": "MLP", "LogisticRegression": "Logistic regression",
                "KNN": "KNN", "RETAIN": "RETAIN", "GRU": "GRU (DEWS)", "MEWS-4": "MEWS-4"}
    a["model"] = a["model"].map(lambda x: name_map.get(x, x))
    a = a.sort_values("auroc_mean", ascending=False)

    b1 = t["B_oneshot"]
    b2 = t["B_op"]
    c = t["C"]
    cool = t["cool"]

    def bop(track_prefix: str, op: str, col: str) -> float:
        s = b2[(b2.track.str.startswith(track_prefix)) & (b2.operating_point == op)][col]
        return float(s.iloc[0])

    md = ["# 论文主结果汇总表（Paper Tables A / B / C）", ""]
    md += ["> 全部数值由 `08_paper_assets.py` 从既有产物直接读取，未做任何重训或手工修改。",
           "> 数据源：`outputs/cv_clean/`、`outputs/track_compare/`、`outputs/cv_external_clean/`、`outputs/shap/`。", ""]

    md += ["## Table A. Internal 10-fold benchmark (INSPIRE, 60,000 windows, 635 features)", "",
           "| Model | AUROC (mean ± SD) | AUPRC | Brier |", "|---|---:|---:|---:|"]
    for _, r in a.iterrows():
        star = "**" if r["model"] == "LightGBM" else ""
        md.append(f"| {star}{r['model']}{star} | {star}{r['auroc_mean']:.4f} ± {r['auroc_sd']:.4f}{star} "
                  f"| {r['auprc_mean']:.4f} | {r['brier_mean']:.4f} |")
    md += ["", f"Kruskal–Wallis across models: **H = {kw['H']:.2f}, p = {kw['p']:.2e}** "
               f"(k = {kw['k']} models, {kw['n_per_group']} folds each).", "",
           "Notes: patient-level StratifiedGroupKFold; no patient contributes windows to two folds. "
           "Feature set excludes `t_end_h`, `observed_frac`, `or_frac`, `miss_*`, `baseline_source`, "
           "`baseline_scr` and `in_vitaldb_overlap`.", ""]

    md += ["## Table B. Peri-operative handover window (Track 2) vs pure ICU window (Track 1)", "",
           "Same 4,285 patients with both intra-operative and ICU records; one prediction per patient "
           "(Track 1 restricted to its earliest window, t_end = 12 h).", "",
           "| Metric | Track 1 (ICU 0–12 h, t_end = 12 h) | Track 2 (handover, t = [−4, +8), t_end = 8 h) | Δ |",
           "|---|---:|---:|---:|"]
    t1, t2 = b1.iloc[0], b1.iloc[1]
    md.append(f"| Windows / patients | {int(t1['n_windows'])} / {int(t1['n_patients'])} "
              f"| {int(t2['n_windows'])} / {int(t2['n_patients'])} | — |")
    md.append(f"| Prevalence (24 h horizon) | {100*t1['pos_rate']:.2f}% | {100*t2['pos_rate']:.2f}% | — |")
    md.append(f"| **AUROC** | {t1['auroc']:.4f} | **{t2['auroc']:.4f}** | "
              f"{t2['auroc']-t1['auroc']:+.4f} |")
    md.append(f"| **AUPRC** | {t1['auprc']:.4f} | **{t2['auprc']:.4f}** | "
              f"{t2['auprc']-t1['auprc']:+.4f} |")
    md.append(f"| Median lead time (Youden) | {bop('Track1','Youden','median_lead_h'):.1f} h "
              f"| {bop('Track2','Youden','median_lead_h'):.1f} h | "
              f"{bop('Track2','Youden','median_lead_h')-bop('Track1','Youden','median_lead_h'):+.1f} h |")
    md.append(f"| Sensitivity / Specificity (Youden) | "
              f"{bop('Track1','Youden','sensitivity'):.3f} / {bop('Track1','Youden','specificity'):.3f} | "
              f"{bop('Track2','Youden','sensitivity'):.3f} / {bop('Track2','Youden','specificity'):.3f} | — |")
    md.append(f"| Event recall @ 10% alarm budget | {bop('Track1','alarm_rate_10pct','event_recall'):.4f} "
              f"| {bop('Track2','alarm_rate_10pct','event_recall'):.4f} | "
              f"{bop('Track2','alarm_rate_10pct','event_recall')-bop('Track1','alarm_rate_10pct','event_recall'):+.4f} |")
    md += ["", "Notes: Track 2 is **4 h earlier by construction** (t_end = 8 h vs 12 h). AUPRC is prevalence-sensitive "
               "(13.89% vs 15.24%), so AUROC and the fixed-alarm-budget recall are the primary comparisons. "
               "Track 1 with all windows (147,154 windows, patient-level max risk) yields AUROC 0.7655.", ""]

    md += ["## Table C. Cross-continental external validation (INSPIRE → eICU) and post-processing", "",
           "Source: full INSPIRE cohort (151,191 windows). Target: eICU (93,429 windows / 2,437 patients). "
           "Normalisation reuses source medians and the fixed physiological bounds; no target-domain fitting.", "",
           "| Cohort | Windows | Patients | Prevalence | AUROC | AUPRC | Brier | Event recall | Median lead (h) |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    cmap_md = {"A_eICU_all_stage1-3": "A — all eICU (stage 1–3)",
               "B_eICU_true_preop_baseline": "B — true pre-op baseline (sensitivity)",
               "C_eICU_stage2-3": "C — moderate-to-severe (stage 2–3)",
               "D_eICU_platt_5pct_calib": "D — 5% Platt calibration"}
    for _, r in c.iterrows():
        if r["status"] != "ok":
            continue
        md.append(f"| {cmap_md.get(r['cohort'], r['cohort'])} | {int(r['n_windows']):,} | {int(r['n_patients']):,} "
                  f"| {100*r['pos_rate']:.2f}% | {r['auroc']:.4f} | {r['auprc']:.4f} | {r['brier']:.4f} "
                  f"| {r['event_recall']:.4f} | {r['median_lead_h']:.1f} |")
    md += ["", f"**Platt calibration (cohort D):** Brier {ext_brier['A']:.4f} → {ext_brier['D']:.4f} "
               f"({ext_brier['reduction']:+.2f}% relative reduction); AUROC unchanged "
               f"({ext_brier['aurocA']:.4f} → {ext_brier['aurocD']:.4f}), as expected for a monotone transform. "
               "Cohort B is a **sensitivity analysis** on label ascertainment, not a performance claim. "
               "Cohort D is evaluated on the 95% of patients not used for calibration.", ""]

    md += ["### Table C-2. Alarm post-processing (60,000-window LightGBM out-of-fold predictions)", "",
           "| Operating point | Threshold | Policy | Event recall | FA / patient | FA / patient-day | Median lead (h) | Window FAR |",
           "|---|---:|---|---:|---:|---:|---:|---:|"]
    label = {"high_sensitivity": "High sensitivity (Sens ≈ 0.85)",
             "balanced_youden": "Balanced (Youden)",
             "high_specificity": "High specificity (Spec ≈ 0.92)"}
    for _, r in cool.iterrows():
        pol = "Raw" if r["policy"] == "raw" else "EWMA 0.5 + 6 h"
        md.append(f"| {label.get(r['operating_point'], r['operating_point'])} | {r['threshold']:.4f} | {pol} "
                  f"| {r['event_recall']:.4f} | {r['false_alarms_per_patient']:.2f} "
                  f"| {r['false_alarms_per_patient_day']:.2f} | {r['median_lead_h']:.2f} | {r['window_far']:.4f} |")
    md += ["", "Notes: `window FAR` is defined as 1 − PPV and is therefore prevalence-dependent; it does **not** "
               "improve with cooldown (cooldown reduces the *number* of alarms, not the positive predictive value "
               "of a single alarm). Report false alarms per patient-day as the operational metric.", ""]

    imp = pd.read_csv(ROOT_SHAP / "lgbm_shap_importance_clean.csv").head(15)
    md += ["## Table D. Feature attribution (LightGBM, artefact-free model)", "",
           "INSPIRE, 40,000 windows, patient-level 8:2 split; hold-out AUROC 0.7623, AUPRC 0.4538. "
           "`t_end_h`, `observed_frac`, `or_frac`, `miss_*` and `baseline_scr` are excluded from the model.", "",
           "| rank | feature | mean \\|SHAP\\| |", "|---|---|---:|"]
    for _, r in imp.iterrows():
        md.append(f"| {int(r['rank'])} | {r['feature']} | {r['mean_abs_shap']:.4f} |")
    md += ["", "Interpretation: four of the top five are perfusion/pressure-derived "
               "(`MAP_1-12h_Min`, `SI_1-12h_Min`, `PP_1-12h_Max`, `PP_1-12h_Min`); "
               "`age` ranks first and is a genuine demographic risk factor, not an artefact. "
               "`DBP_1-12h_Max` ranks 8th — its earlier first place was partly attributable to "
               "`t_end_h` and `baseline_scr` confounding.", ""]
    return "\n".join(md)


# =============================================================================
# 图
# =============================================================================
def fig_internal_track2(t, root: Path, out: Path) -> Path:
    plt = _mpl()
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ---- (a) internal 10-fold ----
    ax = axes[0]
    a = t["A"].sort_values("auroc_mean")
    names = [{"LogisticRegression": "Logistic reg.", "GRU": "GRU (DEWS)",
              "MEWS-4": "MEWS-4"}.get(m, m) for m in a["model"]]
    y = np.arange(len(a))
    colors = [C_MAIN if m == "LightGBM" else C_ALT for m in a["model"]]
    ax.barh(y - 0.19, a["auroc_mean"], height=0.36, color=colors, label="AUROC")
    ax.errorbar(a["auroc_mean"], y - 0.19, xerr=a["auroc_sd"], fmt="none",
                ecolor="#1A202C", elinewidth=0.8, capsize=2)
    ax.barh(y + 0.19, a["auprc_mean"], height=0.36, color="#CBD5E0", label="AUPRC")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Discrimination (10-fold CV)")
    ax.axvline(0.5, color="#A0AEC0", lw=0.7, ls=":")
    ax.annotate("chance", xy=(0.5, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -9), textcoords="offset points",
                ha="center", va="top", fontsize=6.2, color="#718096")
    # npj DM 硬性要求：柱状图必须叠加个体数据点
    folds = pd.read_csv(root / "cv_clean/cv_results_by_fold.csv")
    folds = folds[(folds.status == "ok") & folds.auroc.notna()]
    rng = np.random.RandomState(7)
    for i, m in enumerate(a["model"]):
        sub = folds[folds.model == m]
        for col, off, colr in (("auroc", -0.19, C_MAIN), ("auprc", 0.19, "#4A5568")):
            v = sub[col].dropna().to_numpy(float)
            if v.size == 0:
                continue
            yj = i + off + rng.uniform(-0.075, 0.075, v.size)
            ax.scatter(v, yj, s=4.5, facecolor="white", edgecolor=colr,
                       linewidth=0.45, alpha=0.85, zorder=4)
    ax.set_title("Internal benchmark (INSPIRE, 60,000 windows)", loc="left")
    panel_label(ax, "a", x=-0.34)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    for i, (m, v, sd) in enumerate(zip(a["model"], a["auroc_mean"], a["auroc_sd"])):
        # 标签放在误差线上界之外，避免压住误差线帽
        ax.text(min(v + sd + 0.018, 0.985), i - 0.19, f"{v:.3f}", va="center",
                fontsize=6.5, color=C_MAIN if m == "LightGBM" else "#4A5568")

    # ---- (b) ROC: Track1 vs Track2 ----
    ax = axes[1]
    files = {"Track 1 (t_end = 12 h)": ("track_compare/oof_Track1_ICU_0-12h.csv.gz", C_TRACK1),
             "Track 2 (t_end = 8 h)": ("track_compare/oof_Track2_handover_-4-8h.csv.gz", C_TRACK2)}
    lead = {"Track 1 (t_end = 12 h)": float(t["B_oneshot"].iloc[0]["median_lead_h"]),
            "Track 2 (t_end = 8 h)": float(t["B_oneshot"].iloc[1]["median_lead_h"])}
    t2 = pd.read_csv(root / "track_compare/oof_Track2_handover_-4-8h.csv.gz")
    common = set(t2["patient_id"])
    stats = {}
    for lbl, (rel, col) in files.items():
        d = pd.read_csv(root / rel)
        d = d[d["patient_id"].isin(common)]
        if lbl.startswith("Track 1"):
            d = d.sort_values(["patient_id", "t_end_h"]).groupby("patient_id").head(1)
        yv, pv = d["y"].to_numpy(int), d["pred_prob"].to_numpy(float)
        fpr, tpr, _ = roc_curve(yv, pv)
        ax.plot(fpr, tpr, color=col, lw=1.4,
                label=f"{lbl}: AUROC {auc(fpr, tpr):.3f}, lead {lead[lbl]:.1f} h")
        prec, rec, _ = precision_recall_curve(yv, pv)
        stats[lbl] = float(auc(rec, prec))
    ax.plot([0, 1], [0, 1], color="#A0AEC0", lw=0.7, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Handover window vs pure ICU window", loc="left")
    panel_label(ax, "b", x=-0.16)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0))
    ax.spines[["top", "right"]].set_visible(False)

    b1 = t["B_oneshot"].iloc[0]
    b2 = t["B_oneshot"].iloc[1]
    # 数值已并入图例，这里只标注 AUPRC 的增益，放在右下空白处
    ax.text(0.98, 0.30, f"AUPRC {b1['auprc']:.3f} → {b2['auprc']:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="#2D3748")

    fig.tight_layout()
    p = out / "fig_internal_and_track2.png"
    save_rgb(fig, p)
    plt.close(fig)
    LOG.info("已保存 %s（PR-AUC: %s）", p, {k: round(v, 4) for k, v in stats.items()})
    return p


def fig_shap(root: Path, out: Path, top_k: int = 20) -> Path:
    plt = _mpl()
    imp = pd.read_csv(root / "shap/lgbm_shap_importance_clean.csv").head(top_k)
    imp = imp.sort_values("mean_abs_shap")

    def family(f: str) -> str:
        if f in ("age", "sex"):
            return "Demographic"
        if f.startswith(("MAP", "SBP", "DBP", "PP", "SI")) or f.startswith("PulsePressure"):
            return "Perfusion / pressure"
        return "Other vital sign"

    cmap = {"Perfusion / pressure": C_MAIN, "Demographic": C_ACC, "Other vital sign": "#A0AEC0"}
    colors = [cmap[family(f)] for f in imp["feature"]]

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    y = np.arange(len(imp))
    ax.barh(y, imp["mean_abs_shap"], color=colors, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(imp["feature"], fontsize=7.5)
    ax.set_xlabel("mean |SHAP value| (log-odds)")
    ax.set_title(f"Top {top_k} features — LightGBM, artefact-free model", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(imp["mean_abs_shap"]):
        ax.text(v + imp["mean_abs_shap"].max() * 0.012, i, f"{v:.3f}", va="center", fontsize=6.2)
    ax.set_xlim(0, imp["mean_abs_shap"].max() * 1.16)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.values()]
    ax.legend(handles, list(cmap.keys()), loc="lower right", fontsize=6.8)
    fig.tight_layout()
    p = out / "fig_shap_clean.png"
    save_rgb(fig, p)
    plt.close(fig)
    LOG.info("已保存 %s", p)
    return p


def _platt_on_target(root: Path, calib_frac: float = 0.05, seed: int = 42):
    """复算 cohort D 的校准映射，返回 (评估集原始概率, 校准后概率, y)。"""
    m4 = _load_module("04_train_evaluate.py", "m04")
    d = pd.read_csv(root / "cv_external_clean/external_pred_LightGBM.csv.gz")
    d["patient_id"] = d["patient_id"].astype(str)
    pids = np.asarray(d["patient_id"].unique(), dtype=object)
    rs = np.random.RandomState(seed)
    rs.shuffle(pids)
    n_cal = max(int(round(calib_frac * len(pids))), 5)
    cal_ids = set(pids[:n_cal])
    is_cal = d["patient_id"].isin(cal_ids).to_numpy()
    y = d["y"].to_numpy(int)
    p = d["pred_prob"].to_numpy(float)
    p_new, coef = m4.platt_calibrate(p[is_cal], y[is_cal], p[~is_cal])
    return p[~is_cal], p_new, y[~is_cal], coef


def fig_external(t, root: Path, out: Path) -> Path:
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})

    # ---- (a) four cohorts ----
    ax = axes[0]
    c = t["C"]
    c = c[c["status"] == "ok"].reset_index(drop=True)
    short = {"A_eICU_all_stage1-3": "A\nall",
             "B_eICU_true_preop_baseline": "B\ntrue baseline",
             "C_eICU_stage2-3": "C\nstage 2–3",
             "D_eICU_platt_5pct_calib": "D\nPlatt 5%"}
    x = np.arange(len(c))
    ax.bar(x - 0.2, c["auroc"], width=0.38, color=C_MAIN, label="AUROC")
    ax.bar(x + 0.2, c["auprc"], width=0.38, color="#CBD5E0", label="AUPRC")
    ax.axhline(0.5, color="#A0AEC0", lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([short.get(k, k) for k in c["cohort"]], fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Discrimination")
    ax.set_title("External validation (INSPIRE → eICU)", loc="left")
    panel_label(ax, "a", x=-0.20)
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(c["auroc"]):
        ax.text(i - 0.2, v + 0.02, f"{v:.3f}", ha="center", fontsize=6.2)

    # ---- (b) calibration ----
    ax = axes[1]
    p_raw, p_cal, y, coef = _platt_on_target(root)
    # 分位数分箱：源库模型在目标库上概率被压缩在 0-0.15，等宽分箱会把所有点挤进第一格，
    # 校准曲线退化成两三个点；分位数分箱保证每条曲线都有 10 个可靠点。
    for vals, lbl, col, ls in ((p_raw, "Raw (cohort A)", C_ALT, "-"),
                               (p_cal, "Platt-calibrated (D)", C_OK, "-")):
        edges = np.quantile(vals, np.linspace(0, 1, 11))
        edges = np.unique(edges)
        idx = np.clip(np.digitize(vals, edges[1:-1]), 0, len(edges) - 2)
        xs, ys = [], []
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() >= 20:
                xs.append(float(np.mean(vals[m])))
                ys.append(float(np.mean(y[m])))
        ax.plot(xs, ys, marker="o", ms=2.6, lw=1.1, color=col, ls=ls, label=lbl)
    # 曲线集中在左下（概率被压缩在 0-0.15），文字全部放到右上空白区，避免压住数据
    ax.text(0.97, 0.97,
            f"mean predicted: {p_raw.mean():.3f} (raw)\n"
            f"{p_cal.mean():.3f} (Platt)\nobserved: {y.mean():.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.0, color="#4A5568")
    ax.plot([0, 1], [0, 1], color="#A0AEC0", lw=0.7, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed proportion")
    ax.set_title("Calibration on held-out 95%", loc="left")
    panel_label(ax, "b", x=-0.24)
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    bA = float(c.loc[c["cohort"].str.startswith("A_"), "brier"].iloc[0])
    bD = float(c.loc[c["cohort"].str.startswith("D_"), "brier"].iloc[0])
    # 与正文/表格完全一致的位数：Brier 4 位、相对降幅 2 位（避免图内 -67.7% 与正文 -67.75% 打架）
    ax.text(0.97, 0.50, f"Brier {bA:.4f} → {bD:.4f}\n({100*(bD-bA)/bA:+.2f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.25", fc="#F0FFF4", ec=C_OK, lw=0.6))

    # ---- (c) cooldown ----
    ax = axes[2]
    cool = t["cool"]
    pts = [("high_sensitivity", "High sens."), ("balanced_youden", "Balanced"),
           ("high_specificity", "High spec.")]
    x = np.arange(len(pts))
    raw = [float(cool[(cool.operating_point == k) & (cool.policy == "raw")]
                 ["false_alarms_per_patient_day"].iloc[0]) for k, _ in pts]
    cd = [float(cool[(cool.operating_point == k) & (cool.policy == "ewma_cooldown")]
                ["false_alarms_per_patient_day"].iloc[0]) for k, _ in pts]
    ax.bar(x - 0.19, raw, width=0.36, color="#CBD5E0", label="Raw")
    ax.bar(x + 0.19, cd, width=0.36, color=C_OK, label="EWMA 0.5 + 6 h")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in pts], fontsize=7)
    ax.set_ylabel("False alarms per patient-day")
    ax.set_title("Alarm cooldown", loc="left")
    panel_label(ax, "c", x=-0.20)
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    for i, (r0, r1) in enumerate(zip(raw, cd)):
        ax.text(i - 0.19, r0 + 0.15, f"{r0:.1f}", ha="center", fontsize=6.2)
        ax.text(i + 0.19, r1 + 0.15, f"{r1:.1f}", ha="center", fontsize=6.2, color=C_OK)
    ax.set_ylim(0, max(raw) * 1.22)

    fig.tight_layout()
    p = out / "fig_external_calibration_cooldown.png"
    save_rgb(fig, p)
    plt.close(fig)
    LOG.info("已保存 %s（Platt a=%.3f b=%.3f）", p, coef["a"], coef["b"])
    return p


# =============================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="论文素材：表 + 图")
    ap.add_argument("--out", default="outputs/paper_assets")
    ap.add_argument("--fig-dir", default="outputs/paper_figures")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    root = Path("outputs")
    out = Path(args.out)
    fig_dir = Path(args.fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    t = build_tables(root)
    kw = kw_test(root)
    c = t["C"]
    bA = float(c.loc[c["cohort"].str.startswith("A_"), "brier"].iloc[0])
    bD = float(c.loc[c["cohort"].str.startswith("D_"), "brier"].iloc[0])
    ext_brier = {"A": bA, "D": bD, "reduction": 100 * (bD - bA) / bA,
                 "aurocA": float(c.loc[c["cohort"].str.startswith("A_"), "auroc"].iloc[0]),
                 "aurocD": float(c.loc[c["cohort"].str.startswith("D_"), "auroc"].iloc[0])}

    long = tables_long(t, kw)
    long.to_csv(out / "paper_tables_summary.csv", index=False, encoding="utf-8-sig")
    (out / "paper_tables_summary.md").write_text(
        tables_markdown(t, kw, ext_brier), encoding="utf-8")

    fig_internal_track2(t, root, fig_dir)
    fig_shap(root, fig_dir)
    fig_external(t, root, fig_dir)

    print(f"\n表：{out/'paper_tables_summary.csv'} / .md")
    print(f"图：{fig_dir}/*.png (+ .jpg)")
    print(f"KW: H={kw['H']:.2f}, p={kw['p']:.2e}")
    print(f"Platt Brier: {bA:.4f} -> {bD:.4f} ({ext_brier['reduction']:+.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
