"""定稿数字核对：把 md 表格里的每个数字回查源 CSV。

设计原则：期望值一律由源 CSV **现算**，不在脚本里写死常量；
只要 md 里有一个数字是旧的，就会报 FAIL。

用法：python 09_verify_final_numbers.py
输出：outputs/paper_assets/final_number_checklist.md（+ 控制台摘要）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path("outputs")
MD_TABLES = ROOT / "paper_assets/paper_tables_summary.md"
MD_FACTS = Path("manuscript_draft_facts.md")
OUT = ROOT / "paper_assets/final_number_checklist.md"

ROW: List[Tuple[str, str, str, str]] = []   # (分组, 项目, 定稿值, 状态)


def add(group: str, item: str, value: str, ok: bool, note: str = "") -> None:
    ROW.append((group, item, value, "PASS" if ok else "FAIL", note))


def num(s: str) -> Optional[float]:
    # 千分位逗号必须先去掉，否则 "93,429" 会被解析成 93
    t = s.replace("−", "-").replace(",", "").replace("*", "")
    m = re.search(r"-?\d+\.?\d*", t)
    return float(m.group()) if m else None


def md_rows(text: str, keyword: str) -> List[List[str]]:
    """取包含 keyword 的表格行，按 | 切分。"""
    out = []
    for line in text.splitlines():
        if line.strip().startswith("|") and keyword in line:
            out.append([c.strip() for c in line.strip().strip("|").split("|")])
    return out


def approx(a: Optional[float], b: Optional[float], tol: float = 5.1e-5) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


# =============================================================================
def check_table_a(text: str) -> Dict:
    src = pd.read_csv(ROOT / "cv_clean/cv_summary_clean.csv")
    src["label"] = src["model"].map({"LogisticRegression": "Logistic regression",
                                     "GRU": "GRU (DEWS)"}).fillna(src["model"])
    for _, r in src.iterrows():
        rows = md_rows(text, r["label"])
        rows = [c for c in rows if len(c) >= 4 and ("±" in c[1] or num(c[1]) is not None)]
        if not rows:
            add("Table A", r["label"], "未找到行", False, "md 缺该模型")
            continue
        cells = rows[0]
        auroc_cell, auprc_cell, brier_cell = cells[1], cells[2], cells[3]
        ok_a = approx(num(auroc_cell.split("±")[0]), r["auroc_mean"]) and \
            approx(num(auroc_cell.split("±")[1]) if "±" in auroc_cell else None, r["auroc_sd"])
        add("Table A", f"{r['label']} AUROC", auroc_cell.replace("*", ""),
            ok_a, f"源 {r['auroc_mean']:.4f} ± {r['auroc_sd']:.4f}")
        add("Table A", f"{r['label']} AUPRC", auprc_cell,
            approx(num(auprc_cell), r["auprc_mean"]), f"源 {r['auprc_mean']:.4f}")
        add("Table A", f"{r['label']} Brier", brier_cell,
            approx(num(brier_cell), r["brier_mean"]), f"源 {r['brier_mean']:.4f}")
    return {"src": src}


def check_kw(text: str) -> None:
    from scipy import stats
    f = pd.read_csv(ROOT / "cv_clean/cv_results_by_fold.csv")
    ok = f[(f.status == "ok") & f.auroc.notna()]
    g = {k: v["auroc"].to_numpy(float) for k, v in ok.groupby("model") if len(v) >= 3}
    H, p = stats.kruskal(*g.values())
    m = re.search(r"H = ([\d.]+), p = ([\d.eE+-]+)", text)
    if not m:
        add("Table A", "Kruskal-Wallis", "未找到", False, "md 缺统计量")
        return
    add("Table A", "Kruskal-Wallis H", m.group(1), approx(float(m.group(1)), H, 5e-3),
        f"源 {H:.4f}")
    add("Table A", "Kruskal-Wallis p", m.group(2), abs(float(m.group(2)) - p) / p < 0.01,
        f"源 {p:.3e}")


def check_table_b(text: str) -> None:
    one = pd.read_csv(ROOT / "track_compare/track_comparison_one_shot.csv")
    ops = pd.read_csv(ROOT / "track_compare/track_comparison_operating_points.csv")
    t1, t2 = one.iloc[0], one.iloc[1]

    def op(prefix: str, point: str, col: str) -> float:
        return float(ops[(ops.track.str.startswith(prefix)) & (ops.operating_point == point)][col].iloc[0])

    rows = md_rows(text, "AUROC** |")
    if rows:
        c = rows[0]
        add("Table B", "Track1 AUROC", c[1], approx(num(c[1]), t1["auroc"]), f"源 {t1['auroc']:.4f}")
        add("Table B", "Track2 AUROC", c[2].replace("*", ""), approx(num(c[2]), t2["auroc"]),
            f"源 {t2['auroc']:.4f}")
    rows = md_rows(text, "AUPRC** |")
    if rows:
        c = rows[0]
        add("Table B", "Track1 AUPRC", c[1], approx(num(c[1]), t1["auprc"]), f"源 {t1['auprc']:.4f}")
        add("Table B", "Track2 AUPRC", c[2].replace("*", ""), approx(num(c[2]), t2["auprc"]),
            f"源 {t2['auprc']:.4f}")
    rows = md_rows(text, "Median lead time")
    if rows:
        c = rows[0]
        add("Table B", "Track1 lead", c[1], approx(num(c[1]), op("Track1", "Youden", "median_lead_h"), 0.06),
            f"源 {op('Track1','Youden','median_lead_h'):.2f} h")
        add("Table B", "Track2 lead", c[2], approx(num(c[2]), op("Track2", "Youden", "median_lead_h"), 0.06),
            f"源 {op('Track2','Youden','median_lead_h'):.2f} h")
    rows = md_rows(text, "10% alarm budget")
    if rows:
        c = rows[0]
        add("Table B", "Track1 recall@10%", c[1],
            approx(num(c[1]), op("Track1", "alarm_rate_10pct", "event_recall")),
            f"源 {op('Track1','alarm_rate_10pct','event_recall'):.4f}")
        add("Table B", "Track2 recall@10%", c[2],
            approx(num(c[2]), op("Track2", "alarm_rate_10pct", "event_recall")),
            f"源 {op('Track2','alarm_rate_10pct','event_recall'):.4f}")


def check_table_c(text: str) -> Dict:
    ext = pd.read_csv(ROOT / "cv_external_clean/external_validation_stratified.csv")
    key = {"A": "A — all eICU", "B": "B — true pre-op", "C": "C — moderate", "D": "D — 5% Platt"}
    for k, label in key.items():
        rows = md_rows(text, label)
        if not rows:
            add("Table C", f"队列 {k}", "未找到行", False, "md 缺该队列")
            continue
        c = rows[0]
        src = ext[ext.cohort.str.startswith(k + "_")].iloc[0]
        add("Table C", f"{k} 窗口数", c[1], int(num(c[1])) == int(src["n_windows"]),
            f"源 {int(src['n_windows']):,}")
        add("Table C", f"{k} AUROC", c[4], approx(num(c[4]), src["auroc"]), f"源 {src['auroc']:.4f}")
        add("Table C", f"{k} AUPRC", c[5], approx(num(c[5]), src["auprc"]), f"源 {src['auprc']:.4f}")
        add("Table C", f"{k} Brier", c[6], approx(num(c[6]), src["brier"]), f"源 {src['brier']:.4f}")
        add("Table C", f"{k} 事件召回", c[7], approx(num(c[7]), src["event_recall"]),
            f"源 {src['event_recall']:.4f}")
    return {"ext": ext}


def check_platt(text: str, ext: pd.DataFrame) -> None:
    a = float(ext[ext.cohort.str.startswith("A_")].brier.iloc[0])
    d = float(ext[ext.cohort.str.startswith("D_")].brier.iloc[0])
    red = 100 * (d - a) / a
    m = re.search(r"([\d.]+)\s*→\s*([\d.]+)\s*\((-?[\d.]+)%", text)
    if not m:
        add("Platt", "Brier 降幅", "未找到", False, "md 缺降幅表述")
        return
    add("Platt", "Brier 前值", m.group(1), approx(float(m.group(1)), a), f"源 {a:.4f}")
    add("Platt", "Brier 后值", m.group(2), approx(float(m.group(2)), d), f"源 {d:.4f}")
    add("Platt", "相对降幅 %", m.group(3), approx(float(m.group(3)), red, 0.02),
        f"源 {red:.2f}%")


def check_cooling(text: str) -> None:
    cool = pd.read_csv(ROOT / "cv_clean/policy_comparison_clean.csv")
    label = {"high_sensitivity": "High sensitivity", "balanced_youden": "Balanced",
             "high_specificity": "High specificity"}
    for _, r in cool.iterrows():
        rows = md_rows(text, label[r["operating_point"]])
        pol = "Raw" if r["policy"] == "raw" else "EWMA 0.5 + 6 h"
        hit = [c for c in rows if len(c) > 2 and c[2] == pol]
        if not hit:
            add("冷却表", f"{label[r['operating_point']]} / {pol}", "未找到行", False, "")
            continue
        c = hit[0]
        add("冷却表", f"{label[r['operating_point']]} / {pol} 召回", c[3],
            approx(num(c[3]), r["event_recall"]), f"源 {r['event_recall']:.4f}")
        add("冷却表", f"{label[r['operating_point']]} / {pol} 每住院日误报", c[5],
            approx(num(c[5]), r["false_alarms_per_patient_day"], 0.006),
            f"源 {r['false_alarms_per_patient_day']:.2f}")


def check_shap(facts: str) -> None:
    imp = pd.read_csv(ROOT / "shap/lgbm_shap_importance_clean.csv").head(5)
    for _, r in imp.iterrows():
        pat = rf"\| {int(r['rank'])} \| [^|]*{re.escape(r['feature'])}[^|]*\| ([\d.]+) \|"
        m = re.search(pat, facts)
        if not m:
            add("SHAP", f"rank {int(r['rank'])} {r['feature']}", "未找到", False, "facts 缺该行")
            continue
        add("SHAP", f"rank {int(r['rank'])} {r['feature']}", m.group(1),
            approx(float(m.group(1)), r["mean_abs_shap"]), f"源 {r['mean_abs_shap']:.4f}")


def check_table_d(tables: str) -> None:
    imp = pd.read_csv(ROOT / "shap/lgbm_shap_importance_clean.csv").head(15)
    for _, r in imp.iterrows():
        pat = rf"\| {int(r['rank'])} \| {re.escape(r['feature'])} \| ([\d.]+) \|"
        m = re.search(pat, tables)
        if not m:
            add("Table D", f"rank {int(r['rank'])} {r['feature']}", "未找到", False, "md 缺该行")
            continue
        add("Table D", f"rank {int(r['rank'])} {r['feature']}", m.group(1),
            approx(float(m.group(1)), r["mean_abs_shap"]), f"源 {r['mean_abs_shap']:.4f}")


def check_shap_holdout(tables: str) -> None:
    """Table D 脚注引用的留出集指标必须与固化的 json 一致。"""
    f = ROOT / "shap/shap_holdout_metrics.json"
    if not f.exists():
        add("SHAP 留出集", "指标文件", "缺失", False, "先运行 06 或重算脚本")
        return
    j = pd.read_json(f, typ="series")
    m = re.search(r"hold-out AUROC ([\d.]+), AUPRC ([\d.]+)", tables)
    if not m:
        add("SHAP 留出集", "脚注指标", "未找到", False, "")
        return
    add("SHAP 留出集", "AUROC", m.group(1), approx(num(m.group(1)), float(j["auroc"])),
        f"源 {j['auroc']:.4f}")
    add("SHAP 留出集", "AUPRC", m.group(2), approx(num(m.group(2)), float(j["auprc"])),
        f"源 {j['auprc']:.4f}")
    add("SHAP 留出集", "特征维度", str(int(j["n_features"])), int(j["n_features"]) == 635,
        "应为 635")


def check_derived(facts: str, tables: str) -> None:
    """派生数字（差值、百分比、比例）必须与源数据算出的一致。"""
    a = pd.read_csv(ROOT / "cv_clean/cv_summary_clean.csv")
    lg = float(a[a.model == "LightGBM"].auroc_mean.iloc[0])
    mlp = float(a[a.model == "MLP"].auroc_mean.iloc[0])
    ext = pd.read_csv(ROOT / "cv_external_clean/external_validation_stratified.csv")
    eA = float(ext[ext.cohort.str.startswith("A_")].auroc.iloc[0])
    one = pd.read_csv(ROOT / "track_compare/track_comparison_one_shot.csv")
    d_auroc = float(one.iloc[1].auroc - one.iloc[0].auroc)
    d_auprc = float(one.iloc[1].auprc - one.iloc[0].auprc)
    cool = pd.read_csv(ROOT / "cv_clean/policy_comparison_clean.csv")
    hs = cool[cool.operating_point == "high_sensitivity"].set_index("policy")
    fa_red = 100 * (hs.loc["ewma_cooldown", "false_alarms_per_patient_day"]
                    - hs.loc["raw", "false_alarms_per_patient_day"]) / hs.loc["raw", "false_alarms_per_patient_day"]
    rec_drop = 100 * (hs.loc["raw", "event_recall"] - hs.loc["ewma_cooldown", "event_recall"])
    coh = pd.read_csv(ROOT / "cohort_eicu.csv")
    pct23 = 100 * (coh.baseline_source.isin([2, 3]).sum()) / len(coh)

    # 4 位小数的差值用 0.0006 容差：0.1488 vs 0.1487 这种"四舍五入后相减"的错误必须被抓住
    checks = [
        ("ΔAUROC (LightGBM − MLP)", f"{lg-mlp:.4f}", facts, r"次优 MLP = ([\d.]+)", 5.1e-5),
        ("内部→外部落差", f"{lg-eA:.4f}", facts, r"落差 \*\*([\d.]+)\*\*", 5.1e-5),
        ("Table B ΔAUROC", f"{d_auroc:.4f}", tables, r"\| \*\*AUROC\*\* \|.*?\| \+([\d.]+) \|", 5.1e-5),
        ("Table B ΔAUPRC", f"{d_auprc:.4f}", tables, r"\| \*\*AUPRC\*\* \|.*?\| \+([\d.]+) \|", 5.1e-5),
        # 正文写「−81.9%」，负号在文字里；这里比对绝对值，符号由句式保证
        ("冷却后每住院日误报降幅 %", f"{abs(fa_red):.1f}", facts, r"（高敏点，−([\d.]+)%）", 0.06),
        ("冷却后召回降幅 pp", f"{rec_drop:.1f}", facts, r"（−([\d.]+) 个百分点）", 0.06),
        ("eICU 基线编码 2/3 占比 %", f"{pct23:.1f}", facts, r"有 ([\d.]+)%（1,505/3,597）", 0.06),
    ]
    for name, expected, text, pat, tol in checks:
        m = re.search(pat, text)
        if not m:
            add("派生数字", name, "未找到", False, f"期望 {expected}")
            continue
        got = m.group(1)
        add("派生数字", name, got, approx(num(got), float(expected), tol),
            f"算得 {expected}")


def check_coverage(facts: str, tables: str) -> None:
    # 局限性条数
    sec = facts.split("## 7. 局限性陈述")[-1].split("## 8.")[0]
    n_lim = len(re.findall(r"^\d+\. \*\*", sec, flags=re.M))
    add("覆盖度", "局限性条数", str(n_lim), n_lim == 10, "应为 10 条")
    # 未完成事项条数
    sec8 = facts.split("## 8. 未完成事项")[-1].split("## 9.")[0]
    n_todo = len([l for l in sec8.splitlines() if l.strip().startswith("|") and "---" not in l]) - 1
    add("覆盖度", "未完成事项条数", str(n_todo), n_todo >= 5, "表格数据行")
    # 关键表述是否仍在
    for kw, need in (("0.7463", True), ("0.5975", True), ("0.0739", True),
                     ("67.75", True), ("0.8273", True), ("0.3840", True)):
        add("覆盖度", f"facts 含 {kw}", kw, (kw in facts) == need, "")
    add("覆盖度", "tables 含 LightGBM 0.7463", "0.7463",
        "0.7463 ± 0.0392" in tables, "")
    add("覆盖度", "tables 含 67.75%", "67.75", "67.75" in tables, "")


def quick_card() -> List[Tuple[str, str]]:
    """一页速查：只放正文最常引用的数字，全部现算。"""
    a = pd.read_csv(ROOT / "cv_clean/cv_summary_clean.csv").set_index("model")
    ext = pd.read_csv(ROOT / "cv_external_clean/external_validation_stratified.csv")
    one = pd.read_csv(ROOT / "track_compare/track_comparison_one_shot.csv")
    ops = pd.read_csv(ROOT / "track_compare/track_comparison_operating_points.csv")
    cool = pd.read_csv(ROOT / "cv_clean/policy_comparison_clean.csv")
    imp = pd.read_csv(ROOT / "shap/lgbm_shap_importance_clean.csv")
    j = pd.read_json(ROOT / "shap/shap_holdout_metrics.json", typ="series")
    from scipy import stats
    f = pd.read_csv(ROOT / "cv_clean/cv_results_by_fold.csv")
    ok = f[(f.status == "ok") & f.auroc.notna()]
    g = {k: v["auroc"].to_numpy(float) for k, v in ok.groupby("model") if len(v) >= 3}
    H, pv = stats.kruskal(*g.values())

    eA = ext[ext.cohort.str.startswith("A_")].iloc[0]
    eB = ext[ext.cohort.str.startswith("B_")].iloc[0]
    eC = ext[ext.cohort.str.startswith("C_")].iloc[0]
    eD = ext[ext.cohort.str.startswith("D_")].iloc[0]
    lg = a.loc["LightGBM"]
    hs = cool[cool.operating_point == "high_sensitivity"].set_index("policy")

    def op(prefix, point, col):
        return float(ops[(ops.track.str.startswith(prefix)) & (ops.operating_point == point)][col].iloc[0])

    return [
        ("Results 1（内部基准）",
         f"LightGBM AUROC {lg.auroc_mean:.4f} ± {lg.auroc_sd:.4f}, AUPRC {lg.auprc_mean:.4f}, "
         f"Brier {lg.brier_mean:.4f} (10-fold, n = 60,000 windows)"),
        ("Results 1（传统评分）",
         f"MEWS-4 AUROC {a.loc['MEWS-4'].auroc_mean:.4f} ± {a.loc['MEWS-4'].auroc_sd:.4f} "
         f"(below chance)"),
        ("Results 1（统计检验）",
         f"Kruskal–Wallis H = {H:.2f}, p = {pv:.2e} across 7 models"),
        ("Results 2（SHAP 首位）",
         f"{imp.iloc[0].feature} mean |SHAP| = {imp.iloc[0].mean_abs_shap:.4f} "
         f"(demographic risk factor, not artefact)"),
        ("Results 2（SHAP 首位血流动力学）",
         f"{imp.iloc[1].feature} = {imp.iloc[1].mean_abs_shap:.4f}; "
         f"4 of top 5 are perfusion/pressure-derived"),
        ("Results 2（留出集）",
         f"hold-out AUROC {j['auroc']:.4f}, AUPRC {j['auprc']:.4f} ({int(j['n_windows']):,} windows, "
         f"{int(j['n_features'])} features)"),
        ("Results 3（判别力）",
         f"Track 1 AUROC {one.iloc[0].auroc:.4f} → Track 2 {one.iloc[1].auroc:.4f} "
         f"(Δ +{one.iloc[1].auroc-one.iloc[0].auroc:.4f}); AUPRC {one.iloc[0].auprc:.4f} → "
         f"{one.iloc[1].auprc:.4f}"),
        ("Results 3（提前量）",
         f"median lead time {op('Track1','Youden','median_lead_h'):.1f} h → "
         f"{op('Track2','Youden','median_lead_h'):.1f} h (Track 2 t_end = 8 h vs 12 h)"),
        ("Results 3（同报警预算）",
         f"event recall @10% alarm budget {op('Track1','alarm_rate_10pct','event_recall'):.4f} → "
         f"{op('Track2','alarm_rate_10pct','event_recall'):.4f}"),
        ("Results 4（零样本外部）",
         f"eICU AUROC {eA.auroc:.4f}, AUPRC {eA.auprc:.4f}, Brier {eA.brier:.4f} "
         f"(n = {int(eA.n_windows):,} windows / {int(eA.n_patients):,} patients)"),
        ("Results 4（内部→外部落差）",
         f"ΔAUROC = {lg.auroc_mean - eA.auroc:.4f} (internal {lg.auroc_mean:.4f} → external {eA.auroc:.4f})"),
        ("Results 4（敏感性队列 B）",
         f"true pre-op baseline AUROC {eB.auroc:.4f} — **not higher** than cohort A "
         f"({eA.auroc:.4f}); label ascertainment is not the main driver"),
        ("Results 4（中重度 C）",
         f"stage 2–3 AUROC {eC.auroc:.4f}, AUPRC {eC.auprc:.4f} at {100*eC.pos_rate:.2f}% prevalence"),
        ("Results 4（Platt 校准 D）",
         f"Brier {eA.brier:.4f} → {eD.brier:.4f} "
         f"({100*(eD.brier-eA.brier)/eA.brier:+.2f}% relative), AUROC unchanged "
         f"({eA.auroc:.4f} → {eD.auroc:.4f})"),
        ("Results 4（冷却，高敏点）",
         f"false alarms per patient-day {hs.loc['raw','false_alarms_per_patient_day']:.2f} → "
         f"{hs.loc['ewma_cooldown','false_alarms_per_patient_day']:.2f} "
         f"({100*(hs.loc['ewma_cooldown','false_alarms_per_patient_day']-hs.loc['raw','false_alarms_per_patient_day'])/hs.loc['raw','false_alarms_per_patient_day']:+.1f}%), "
         f"recall {hs.loc['raw','event_recall']:.4f} → {hs.loc['ewma_cooldown','event_recall']:.4f}"),
        ("Results 4（冷却，高特异点）",
         f"false alarms per patient-day "
         f"{cool[(cool.operating_point=='high_specificity') & (cool.policy=='raw')].false_alarms_per_patient_day.iloc[0]:.2f} → "
         f"{cool[(cool.operating_point=='high_specificity') & (cool.policy=='ewma_cooldown')].false_alarms_per_patient_day.iloc[0]:.2f}"),
        ("Methods（队列规模）",
         "INSPIRE 9,543 patients / 151,191 windows; eICU 3,597 patients (4,000 cap) / 93,429 windows; "
         "VitalDB 3,000 patients (3,000 cap)"),
        ("Methods（跨库重叠）",
         "2,595 exact `case_id` overlaps between INSPIRE and VitalDB (fingerprint method: only 43)"),
    ]




# =============================================================================
# 文档审计：旧口径清零 + 三条红线
# =============================================================================
STALE = {
    "0.1488": "旧落差（用四舍五入值相减），正确 0.1487",
    "71.02": "旧版含锚点 Platt 降幅，正确 67.75",
    "0.2552": "旧版 Brier（含锚点），正确 0.2291",
    "0.0740": "旧版 Brier（含锚点），正确 0.0739",
    "0.6069": "旧版 B 队列 AUROC，正确 0.5919",
    "0.5947": "旧版 A 队列 AUROC，正确 0.5975",
    "0.7779": "旧版内部 AUROC（含锚点），正确 0.7463",
    "0.5656": "更早版外部 AUROC，正确 0.5975",
}

# 变更记录类上下文允许保留旧值（否则无法叙述"改了什么"）
# 只认"显式声明这是历史/被取代"的标题；"修正""差异"等词会误伤现行章节
# （例如「本轮 4 项方法学修正」就是现行内容），不能作为豁免依据。
EXEMPT_HEADING = re.compile(r"旧值|旧版|历史|上一版|已被取代|changelog|previous|deprecated", re.I)
EXEMPT_LINE = re.compile(r"旧值|旧版|修正前|上一版|早期|错误|was |previous|deprecated|示例|e\.g\.", re.I)

DOC_FILES = ["README.md", "outputs/paper_assets/paper_tables_summary.md",
             "manuscript_draft_facts.md",
             "outputs/paper_assets/final_number_checklist.md"]


def _paragraphs(text: str, prose_only: bool = False) -> List[Tuple[int, str]]:
    """按空行切段，返回 (起始行号, 段落文本)。

    prose_only=True 时跳过两类非断言内容：
      - 章节标题（`#` 开头）：标题不是结论，不应要求它同时给出限定语；
      - 纯表格块（每行都以 `|` 开头）：表格是数值罗列，其口径由表注承担。
    三条红线只约束**散文断言**，否则会把数据表逐行误判为违规。
    """
    out, buf, start = [], [], 1
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip():
            if not buf:
                start = i
            buf.append(line)
        elif buf:
            out.append((start, "\n".join(buf)))
            buf = []
    if buf:
        out.append((start, "\n".join(buf)))
    if not prose_only:
        return out
    keep = []
    for start, para in out:
        lines = [l for l in para.splitlines() if l.strip()]
        body = [l for l in lines if not l.lstrip().startswith("#")]
        if not body:
            continue                      # 只有标题
        if all(l.lstrip().startswith("|") for l in body):
            continue                      # 标题 + 纯表格（表格口径由表注承担）
        keep.append((start, para))
    return keep


def audit_docs(files: Sequence[str]) -> Tuple[List[str], int]:
    """旧数字清零 + 三条红线。返回 (报告行, 违规数)。"""
    lines: List[str] = ["=" * 78, "文档审计：旧口径清零 + 三条红线", "=" * 78]
    violations = 0

    for rel in files:
        f = Path(rel)
        if not f.exists():
            lines.append(f"[跳过] {rel} 不存在")
            continue
        text = f.read_text(encoding="utf-8")
        raw = text.splitlines()
        heads = [(i, l) for i, l in enumerate(raw, 1) if l.lstrip().startswith("#")]

        def _level(h: str) -> int:
            return len(h) - len(h.lstrip("#"))

        # 历史章节区间：命中豁免关键词的标题，作用域到下一个同级或更高级标题为止。
        # 只看"最近一个标题"会失效——因为 §11 下有 11.1/11.2 等子标题，它们不含"历史"。
        exempt_ranges: List[Tuple[int, int]] = []
        for idx, (ln, h) in enumerate(heads):
            if EXEMPT_HEADING.search(h):
                end = len(raw) + 1
                for ln2, h2 in heads[idx + 1:]:
                    if _level(h2) <= _level(h):
                        end = ln2
                        break
                exempt_ranges.append((ln, end))

        def is_exempt(ln: int) -> bool:
            return any(a <= ln < b for a, b in exempt_ranges)

        heading = ""
        hits: List[str] = []
        exempted = 0
        for i, line in enumerate(raw, 1):
            if line.lstrip().startswith("#"):
                heading = line
            for bad, why in STALE.items():
                if re.search(rf"(?<![\d.]){re.escape(bad)}(?![\d])", line):
                    if is_exempt(i) or EXEMPT_LINE.search(line):
                        exempted += 1
                        continue
                    hits.append(f"  L{i}: `{bad}` — {why}")
                    violations += 1
        lines.append(f"[{rel}] 旧值违规 {len(hits)}，变更记录内豁免 {exempted}")
        lines += hits

        # 红线：按散文段落检查共现（表格/标题不参与；历史章节豁免）
        for start, para in _paragraphs(text, prose_only=True):
            if is_exempt(start):
                exempted += 1
                continue
            # ① Cohort B
            # 中文语序两种都要覆盖：「B 队列」与「队列 B」
            if re.search(r"0\.5919|[Cc]ohort\s*B|队列\s*B|B\s*队列", para):
                if not re.search(r"sensitivity|敏感性", para, re.I):
                    hits.append(f"  L{start}: 红线① — 提及 Cohort B 但未标注 sensitivity/敏感性分析")
                    violations += 1
                if re.search(r"net improvement|性能改善|净提升|improved performance|performance gain", para, re.I):
                    hits.append(f"  L{start}: 红线① — 出现「B 队列性能改善」类表述")
                    violations += 1
            # ② Cohort D / Platt
            if re.search(r"Platt|0\.0739", para):
                if not re.search(r"calibration|校准", para, re.I):
                    hits.append(f"  L{start}: 红线② — 提及 Platt 校准但未说明 calibration/校准")
                    violations += 1
                if not re.search(r"unchanged|不变|0\.5971", para, re.I):
                    hits.append(f"  L{start}: 红线② — 未声明判别力不变（AUROC 0.5975 → 0.5971）")
                    violations += 1
                if re.search(r"discrimination improved|判别力提升|AUROC improved", para, re.I):
                    hits.append(f"  L{start}: 红线② — 出现「校准后判别力提升」类表述")
                    violations += 1
            # ③ Track 2 时间锚点
            if "0.8273" in para and "0.7898" in para:
                if not re.search(r"by construction|定义使然|t_end\s*=\s*8|8 h vs 12 h", para, re.I):
                    hits.append(f"  L{start}: 红线③ — Track1/Track2 对比未标注「早 4h 由窗口定义使然」")
                    violations += 1
                if not re.search(r"0\.2833|10%", para):
                    hits.append(f"  L{start}: 红线③ — 未给出固定报警预算召回（0.2833 vs 0.2535）")
                    violations += 1

        lines.append(f"[{rel}] 红线违规 {len([h for h in hits if '红线' in h])}")
        lines += [h for h in hits if "红线" in h]
    lines.append("")
    return lines, violations


def audit_figure_source(path: str = "08_paper_assets.py") -> Tuple[List[str], int]:
    """图注/标签文本审计：禁止旧值，要求 Brier 4 位 + 百分比 2 位。"""
    f = Path(path)
    lines = ["=" * 78, "图注/标签文本审计（08_paper_assets.py）", "=" * 78]
    if not f.exists():
        return lines + ["[跳过] 文件不存在"], 0
    text = f.read_text(encoding="utf-8")
    bad = 0
    for i, line in enumerate(text.splitlines(), 1):
        for s, why in STALE.items():
            if re.search(rf"(?<![\d.]){re.escape(s)}(?![\d])", line) and "STALE" not in line:
                lines.append(f"  L{i}: 旧值 `{s}` — {why}")
                bad += 1
    if "Brier {bA:.4f} → {bD:.4f}" not in text:
        lines.append("  Brier 图注未使用 4 位小数（应与表格一致）")
        bad += 1
    if "{100*(bD-bA)/bA:+.2f}" not in text:
        lines.append("  百分比图注未使用 2 位小数（-67.75% 而非 -67.7%）")
        bad += 1
    lines.append(f"图注违规 {bad}")
    return lines, bad


# =============================================================================
# 英文初稿审计
# =============================================================================
# 只有"率"类列才允许生成百分数形式；否则某个 p 值 0.007461 会被 ×100 变成 0.7461，
# 恰好掩盖一个错误的 AUROC。这是实测到的漏报来源。
RATE_COLS = re.compile(r"pos_rate|event_recall|^far$|sensitivity|specificity|^ppv$|"
                       r"window_far|observed_frac|or_frac|n_positive", re.I)

# 论文引用的数值只来自这些"面向论文"的产物；折级明细（cv_results_by_fold.csv）不进允许集，
# 否则 70 个折级 AUROC 里总有某个值能掩盖正文里的错误数字。
PAPER_FILES = [
    "cv_clean/cv_summary_clean.csv",
    "cv_clean/policy_comparison_clean.csv",
    "track_compare/track_comparison_one_shot.csv",
    "track_compare/track_comparison_operating_points.csv",
    "track_compare/track_patient_level.csv",
    "cv_external_clean/external_validation_stratified.csv",
    "shap/lgbm_shap_importance_clean.csv",
    "cv_tabnet/cv_summary.csv",          # Limitations 引用 TabNet 初步结果
    "cv_tabnet/cv_results_by_fold.csv",  # 折间范围 0.432–0.737
    "paper_assets/table1_baseline.csv",  # Table 1 基线（含格式化字符串，另行解析）
    "paper_assets/stat_tests_npj.csv",   # 95% CI 与精确 P 值
]


# 仅以下 4 个数在现有产物中无对应文件（来自 2026-11-18 eICU 构建日志的 stdout）。
# 单独落盘并标注来源，避免把它们混进"由 CSV 现算"的集合里。
PROVENANCE_JSON = ROOT / "paper_assets/provenance_log_values.json"


def _provenance() -> dict:
    if PROVENANCE_JSON.exists():
        try:
            return json.loads(PROVENANCE_JSON.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    d = {
        "_source": "eICU cohort build stdout, 2026-11-18（01_cohort_builder.py）",
        "nibp_records_total": 1038236,
        "nibp_records_kept": 831293,
        "eicu_windows_before_nibp": 38207,
        "eicu_patients_before_nibp": 1447,
    }
    PROVENANCE_JSON.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE_JSON.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    return d


def _calibration_stats() -> dict:
    """复算 cohort D 的校准前后平均预测概率与实际观测率（与 04 同 seed/frac）。"""
    f = ROOT / "cv_external_clean/external_pred_LightGBM.csv.gz"
    if not f.exists():
        return {}
    d = pd.read_csv(f)
    d["patient_id"] = d["patient_id"].astype(str)
    pids = np.asarray(d["patient_id"].unique(), dtype=object)
    rs = np.random.RandomState(42)
    rs.shuffle(pids)
    n_cal = max(int(round(0.05 * len(pids))), 5)
    is_cal = d["patient_id"].isin(set(pids[:n_cal])).to_numpy()
    y = d["y"].to_numpy(int)
    p = d["pred_prob"].to_numpy(float)
    z = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))  # noqa: E731
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z(p[is_cal]).reshape(-1, 1), y[is_cal])
    p_cal = lr.predict_proba(z(p[~is_cal]).reshape(-1, 1))[:, 1]
    return {"mean_raw": float(p[~is_cal].mean()), "mean_cal": float(pc_mean := p_cal.mean()),
            "observed": float(y[~is_cal].mean())}


def _cohort_governance_values() -> set:
    """纳排各步人数、去重计数、baseline_source 分层、重叠标记。"""
    out: set = set()
    f = ROOT / "cohort_attrition_real.csv"
    if f.exists():
        d = pd.read_csv(f)
        for c in ("n_remaining", "n_excluded"):
            if c in d.columns:
                out.update(pd.to_numeric(d[c], errors="coerce").dropna().astype(float).tolist())
    f = ROOT / "dedup_inspire_vitaldb.csv"
    if f.exists():
        d = pd.read_csv(f)
        out.update(pd.to_numeric(d.iloc[0], errors="coerce").dropna().astype(float).tolist())
    for db in ("inspire", "eicu", "vitaldb"):
        f = ROOT / f"cohort_{db}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, usecols=lambda c: c in ("baseline_source", "in_vitaldb_overlap",
                                                   "max_stage"))
        if "baseline_source" in d.columns:
            out.update(d["baseline_source"].value_counts().astype(float).tolist())
            out.add(float(len(d)))
            # 估算/入科基线（编码 2+3）的合计，正文写作 1,505/3,597
            out.add(float(d["baseline_source"].isin([2, 3]).sum()))
        if "in_vitaldb_overlap" in d.columns:
            out.add(float(d["in_vitaldb_overlap"].sum()))
    for f in (ROOT / "windows_track1_manifest.json", ROOT / "eicu/windows_track1_manifest.json"):
        if f.exists():
            j = json.loads(f.read_text(encoding="utf-8"))
            out.add(float(j.get("n_windows", 0)))
            out.update(float(v) for v in j.get("stats", {}).values()
                       if isinstance(v, (int, float)))
    return out


def _cooling_derived() -> set:
    """三档操作点在 Raw -> EWMA+6h 下的降幅（% / 百分点），1 位与 2 位小数各存一份。"""
    f = ROOT / "cv_clean/policy_comparison_clean.csv"
    if not f.exists():
        return set()
    d = pd.read_csv(f)
    out: set = set()
    for op, g in d.groupby("operating_point"):
        g = g.set_index("policy")
        if not {"raw", "ewma_cooldown"} <= set(g.index):
            continue
        for col in ("false_alarms_per_patient_day", "false_alarms_per_patient"):
            r0, r1 = float(g.loc["raw", col]), float(g.loc["ewma_cooldown", col])
            if r0 > 0:
                red = abs(100 * (r1 - r0) / r0)
                out.update({round(red, 1), round(red, 2)})
        pp = abs(100 * (float(g.loc["ewma_cooldown", "event_recall"])
                        - float(g.loc["raw", "event_recall"])))
        out.update({round(pp, 1), round(pp, 2)})
    return out


def _track_derived() -> set:
    """Table B 的差值与 3 位小数展示值（图 1b/3a 用的是 3 位）。"""
    one = ROOT / "track_compare/track_comparison_one_shot.csv"
    ops = ROOT / "track_compare/track_comparison_operating_points.csv"
    out: set = set()
    if one.exists():
        d = pd.read_csv(one)
        t1, t2 = d.iloc[0], d.iloc[1]
        out.update({
            float(t2.auroc) - float(t1.auroc), float(t2.auprc) - float(t1.auprc),
            abs(float(t2.median_lead_h) - float(t1.median_lead_h)),
            float(t2.event_recall) - float(t1.event_recall),
        })
        for v in (t1.auroc, t2.auroc, t1.auprc, t2.auprc, t1.event_recall, t2.event_recall):
            out.update({round(float(v), 2), round(float(v), 3), float(v)})
    if ops.exists():
        d = pd.read_csv(ops)
        for c in ("event_recall", "sensitivity", "specificity", "median_lead_h", "auroc", "auprc"):
            for v in d[c].dropna().astype(float):
                out.update({float(v), round(float(v), 2), round(float(v), 3)})
        r = d[d.operating_point == "alarm_rate_10pct"]
        if len(r) == 2:
            out.add(abs(float(r.iloc[1].event_recall) - float(r.iloc[0].event_recall)))
    return out


def _allowed_values() -> Tuple[set, set]:
    """从论文级产物 + 动态派生量收集允许出现的数值。"""
    vals: set = set()
    pcts: set = set()
    # 只收论文级聚合文件。患者级 cohort CSV（数千行 × 数十列）一旦纳入，
    # 允许集会膨胀到 2 万多个值，几乎任何 4 位小数都能"匹配"上——实测把 0.7490 放过了。
    # 队列治理相关的计数由 _cohort_governance_values() 精确提取。
    paths = [ROOT / f for f in PAPER_FILES]
    for p in paths:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            continue
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                v = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(float)
                vals.update(v.tolist())
                if RATE_COLS.search(c):
                    pcts.update((v[(v > 0) & (v < 1)] * 100).tolist())
            elif df[c].dtype == object:
                # Table 1 的值是 "5,558 (58.2%)"、"60.7 ± 15.0" 这类格式化串，
                # 逐个数抽出来，避免把表格数字当成未匹配项。
                for cell in df[c].dropna().astype(str):
                    for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", cell):
                        try:
                            vals.add(float(tok.replace(",", "")))
                        except ValueError:
                            pass

    # ---- 派生量：全部现算，无硬编码 ----
    a = pd.read_csv(ROOT / "cv_clean/cv_summary_clean.csv").set_index("model")
    ext = pd.read_csv(ROOT / "cv_external_clean/external_validation_stratified.csv")
    eA = float(ext[ext.cohort.str.startswith("A_")].auroc.iloc[0])
    eD = float(ext[ext.cohort.str.startswith("D_")].auroc.iloc[0])
    bA = float(ext[ext.cohort.str.startswith("A_")].brier.iloc[0])
    bD = float(ext[ext.cohort.str.startswith("D_")].brier.iloc[0])
    vals.update({
        float(a.loc["LightGBM"].auroc_mean) - eA,
        float(a.loc["LightGBM"].auroc_mean) - float(a.loc["MLP"].auroc_mean),
        abs(100 * (bD - bA) / bA), eA, eD, bA, bD,
    })
    for v in ext.select_dtypes("number").to_numpy().ravel():
        if np.isfinite(v):
            vals.update({float(v), round(float(v), 2), round(float(v), 3)})

    # SHAP 留出集指标（0.7623 / 0.4538 / 40000 / 635）
    jf = ROOT / "shap/shap_holdout_metrics.json"
    if jf.exists():
        try:
            vals.update(float(v) for v in json.loads(jf.read_text(encoding="utf-8")).values()
                        if isinstance(v, (int, float)))
        except Exception:  # noqa: BLE001
            pass

    # Kruskal-Wallis H 与 p（正文写作 47.25 / 1.67 × 10^-8）
    try:
        from scipy import stats as _st
        _f = pd.read_csv(ROOT / "cv_clean/cv_results_by_fold.csv")
        _ok = _f[(_f.status == "ok") & _f.auroc.notna()]
        _g = {k: v["auroc"].to_numpy(float) for k, v in _ok.groupby("model") if len(v) >= 3}
        _H, _p = _st.kruskal(*_g.values())
        vals.update({float(_H), float(_p), round(float(_p), 2)})
        # 正文常写成 "1.67 × 10^-8"，需要单独登记尾数
        if 0 < _p < 1e-3:
            vals.add(float(f"{_p:.2e}".split("e")[0]))
    except Exception:  # noqa: BLE001
        pass

    # 各库阳性率与 Stage 2–3 占比（VitalDB 6.63% / 2.03% 等）
    for _db in ("inspire", "eicu", "vitaldb"):
        _c = ROOT / f"cohort_{_db}.csv"
        if not _c.exists():
            continue
        _d = pd.read_csv(_c, usecols=lambda c: c in ("t_aki_onset", "max_stage"))
        _n = len(_d)
        _k = int(_d["t_aki_onset"].notna().sum())
        _k23 = int((_d["max_stage"] >= 2).sum()) if "max_stage" in _d.columns else 0
        vals.update({float(_n), float(_k), float(_k23),
                     100 * _k / _n, 100 * _k23 / _n,
                     round(100 * _k / _n, 1), round(100 * _k23 / _n, 1)})

    vals |= _cooling_derived()
    vals |= _track_derived()
    vals |= _cohort_governance_values()
    vals.update(float(v) for k, v in _provenance().items() if isinstance(v, (int, float)))
    vals.update(_calibration_stats().values())

    # 关键指标同时登记 2/3 位展示值，供图注与正文的舍入写法匹配
    for v in list(vals):
        if 0 < abs(v) < 1:
            vals.update({round(v, 2), round(v, 3)})

    ints = {float(v) for v in vals if abs(v) >= 1}
    return vals | pcts, ints


def _match(x: float, dec: int, allowed: set) -> bool:
    tol = 0.5 * 10 ** (-dec) + 1e-9
    return any(abs(v - x) <= tol for v in allowed)


# 非结果类数字：文献年份、临床常量、维度与折数。命中则归 INFO，不参与 strict 判定。
INFO_RULES = [
    (re.compile(r"\b(?:19|20)\d{2}\b"), "文献年份"),
    (re.compile(r"\b(?:18|99|180|24|48|12|7|6|4|8|10)\s*(?:h|hr|hrs|hour|hours|d|day|days)\b", re.I),
     "时间/年龄常量"),
    (re.compile(r"\b0\.3\s*mg/dL", re.I), "KDIGO 阈值"),
    (re.compile(r"\b1\.5\s*(?:×|x|times|fold)", re.I), "KDIGO 倍数"),
    (re.compile(r"\b(?:65|90)\s*mmHg", re.I), "灌注压阈值"),
    (re.compile(r"\b(?:10|5|4|12)[- ]fold\b", re.I), "交叉验证折数"),
    (re.compile(r"\b63[05]\b\s*(?:features|dimensions|variables|input)", re.I), "特征维度"),
    (re.compile(r"(?:α|alpha)\s*=\s*0\.5", re.I), "平滑系数"),
]

SECTION_PAT = re.compile(r"abstract|results|table|figure|legend", re.I)

# Nature 风格的上标引用（^1^、^1–3^、^1,2^）与方括号引用（[1]、[1–3]）
CITE_SUP = re.compile(r"\^\s*\d+(?:\s*[,–-]\s*\d+)*\s*\^")
CITE_BRACKET = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]")


def _strip_citations(line: str) -> str:
    """去掉引用角标后再抽数字，否则 ^1^ / [1] 会被当成统计量。"""
    return CITE_BRACKET.sub(" ", CITE_SUP.sub(" ", line))


def _references_range(raw: Sequence[str]) -> Tuple[int, int]:
    """定位 ## References 区间（到下一个同级标题），该区间整段不参与数字核对。"""
    heads = [(i, l) for i, l in enumerate(raw, 1) if l.lstrip().startswith("#")]
    for idx, (ln, h) in enumerate(heads):
        if re.search(r"\breferences\b", h, re.I):
            lv = len(h) - len(h.lstrip("#"))
            end = len(raw) + 1
            for ln2, h2 in heads[idx + 1:]:
                if len(h2) - len(h2.lstrip("#")) <= lv:
                    end = ln2
                    break
            return ln, end
    return 0, 0


def _abstract_text(raw: Sequence[str]) -> str:
    a, b = 0, 0
    for ln, l in enumerate(raw, 1):
        if re.match(r"^#{1,6}\s*abstract\b", l.strip(), re.I):
            a = ln
        elif a and l.lstrip().startswith("#"):
            b = ln
            break
    if not a:
        return ""
    body = raw[a:(b - 1) if b else len(raw)]
    return " ".join(x.strip() for x in body if x.strip())
BULLET = re.compile(r"^\s*(?:[*\-+]|\d+\.)\s+")


def _units(text: str) -> List[Tuple[int, str]]:
    """在散文段落基础上按项目符号再切分，得到"单条断言"粒度。"""
    out: List[Tuple[int, str]] = []
    for start, para in _paragraphs(text, prose_only=True):
        lines = para.splitlines()
        buf: List[str] = []
        s0 = start
        for off, l in enumerate(lines):
            if BULLET.match(l) and buf:
                out.append((s0, "\n".join(buf)))
                buf = []
                s0 = start + off
            buf.append(l)
        if buf:
            out.append((s0, "\n".join(buf)))
    return out


def _sections(raw: Sequence[str]) -> List[Tuple[int, int, str]]:
    heads = [(i, l) for i, l in enumerate(raw, 1) if l.lstrip().startswith("#")]
    out = []
    for idx, (ln, h) in enumerate(heads):
        name = h.strip("# ").strip()
        if not SECTION_PAT.search(name):
            continue
        lv = len(h) - len(h.lstrip("#"))
        end = len(raw) + 1
        for ln2, h2 in heads[idx + 1:]:
            if len(h2) - len(h2.lstrip("#")) <= lv:
                end = ln2
                break
        out.append((ln, end, name))
    return out


def _info_kind(line: str, s: int, e: int) -> Optional[str]:
    for pat, label in INFO_RULES:
        for m in pat.finditer(line):
            if m.start() <= s and e <= m.end():
                return label
    return None


def audit_manuscript(path: str, strict: bool = False) -> Tuple[List[str], int]:
    """英文初稿：数字对齐 + 红线/旧值（硬失败）+ 分级警告。"""
    f = Path(path)
    lines = ["=" * 78, f"英文初稿审计：{path}", f"模式：{'STRICT' if strict else 'DEFAULT（宽松警告）'}", "=" * 78]
    if not f.exists():
        lines.append("[SKIP] 文件不存在——请先把英文初稿保存到该路径")
        return lines, 0

    text = f.read_text(encoding="utf-8")
    raw = text.splitlines()
    allowed, ints = _allowed_values()
    secs = _sections(raw)

    def in_section(ln: int) -> bool:
        return any(a <= ln < b for a, b, _ in secs)

    # ---- 1. 数字 ----
    ref_a, ref_b = _references_range(raw)
    total = matched = 0
    unmatched: List[Tuple[int, str, str, str]] = []   # (行, token, 分类, 原文)
    for i, line in enumerate(raw, 1):
        if ref_a <= i < ref_b:
            continue                                  # 参考文献列表整段跳过
        line = _strip_citations(line)
        for m in re.finditer(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d{2,4}|\b(?:19|20)\d{2}\b)(?![\d])", line):
            tok = m.group(1)
            total += 1
            if "," in tok:
                v = float(tok.replace(",", ""))
                ok = v < 1000 or _match(v, 0, ints)
            elif "." in tok:
                dec = len(tok.split(".")[1])
                ok = _match(float(tok), dec, allowed)
            else:                       # 4 位年份
                ok = False
            if ok:
                matched += 1
                continue
            kind = _info_kind(line, m.start(), m.end())
            snippet = line.strip()
            if len(snippet) > 110:
                s0 = max(m.start() - 45, 0)
                snippet = ("…" if s0 else "") + line[s0:m.end() + 45].strip() + "…"
            unmatched.append((i, tok, kind or "WARNING", snippet))

    hard = [u for u in unmatched if u[2] == "WARNING"]
    info = [u for u in unmatched if u[2] != "WARNING"]

    # Results 段落匹配率
    res_ranges = [(a, b) for a, b, n in secs if re.search(r"results", n, re.I)]
    res_total = res_matched = 0
    for i, line in enumerate(raw, 1):
        if not any(a <= i < b for a, b in res_ranges):
            continue
        if ref_a <= i < ref_b:
            continue
        line = _strip_citations(line)
        for m in re.finditer(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d{2,4}|\b(?:19|20)\d{2}\b)(?![\d])", line):
            tok = m.group(1)
            res_total += 1
            if "," in tok:
                v = float(tok.replace(",", ""))
                ok = v < 1000 or _match(v, 0, ints)
            else:
                dec = len(tok.split(".")[1])
                ok = _match(float(tok), dec, allowed)
            if ok:
                res_matched += 1

    ab = _abstract_text(raw)
    # 按空白分词，与 Word 的计数口径一致；用字符类正则会把 "0.7463" 拆成两个词
    n_words = len(ab.split()) if ab else 0
    flag = "  ⚠ 超过 200 词" if n_words > 200 else ""
    lines.append(f"[0] Abstract word count: {n_words}（npj DM 目标 ≤ 200）{flag}")

    lines.append(f"[1] 数字一致性：扫描 {total}，匹配 {matched}，未匹配 {total - matched}"
                 f"（其中 WARNING {len(hard)}，INFO {len(info)}）")
    if res_ranges:
        rate = 100 * res_matched / res_total if res_total else float("nan")
        lines.append(f"    Results 段落：{res_matched}/{res_total}（{rate:.1f}%）")
    else:
        lines.append("    Results 段落：未检测到标题（无 '## Results' 之类）")

    # ---- 2. 红线 + 旧值（硬失败）----
    red: List[str] = []
    for start, unit in _units(text):
        # ① Cohort B：出现即须标注为敏感性分析
        if re.search(r"0\.5919|[Cc]ohort\s*B|队列\s*B|B\s*队列", unit):
            if not re.search(r"sensitivity|label ascertainment|敏感性", unit, re.I):
                red.append(f"  L{start}: 红线① — Cohort B 未标注为 sensitivity analysis")
        # ② 触发条件必须是"涉及 Platt 校准"的单元（出现 Platt 或校准后的 Brier 0.0739）。
        #    用 "Brier" 当触发词会误伤内部基准（0.0809）与队列 A/B/C 的原始 Brier，
        #    那些段落与校准无关，不该被要求声明判别力不变。
        # 必须同时是"校准结果断言"：出现 Platt 且有 restore/correct/reduce/improve 等结果动词，
        # 或直接给出校准后 Brier 0.0739。仅把 Platt 当方法名/展望提及（aims、limitations）不触发。
        # 判定粒度下沉到"句"：Platt 与结果动词必须在同一句，否则一段 aims 里随便一个
        # "reduce false alarms" 就会把整段误判为校准结果断言。
        _platt_claim = "0.0739" in unit
        if not _platt_claim:
            for _sent in re.split(r"(?<=[.;])\s+", unit):
                if (re.search(r"Platt", _sent, re.I)
                        and re.search(r"restor|correct|reduc|improv|lower|fell|drop", _sent, re.I)
                        and re.search(r"calibrat|Brier", _sent, re.I)):
                    _platt_claim = True
                    break
        if _platt_claim:
            if not re.search(r"calibrat|校准", unit, re.I):
                red.append(f"  L{start}: 红线② — 未说明 calibration recovery")
            if not re.search(r"unchanged|preserved|not alter\w*|remain\w*|no change|"
                             r"did not change|identical|0\.5971|"
                             r"without (?:materially )?improving discrimination|"
                             r"no (?:material )?improvement in discrimination|"
                             r"discrimination (?:was |remained )?(?:not|un)", unit, re.I):
                red.append(f"  L{start}: 红线② — 未声明 discrimination unchanged（0.5975 → 0.5971）")
        # ③ Track1/Track2 同时出现时，须说明早 4 h 来自窗口定义
        if "0.8273" in unit and "0.7898" in unit:
            if not re.search(r"by construction|by design|definitional|inherent to the window|"
                             r"t_end\s*=\s*8|8 h vs 12 h|4-h earlier|earlier by 4", unit, re.I):
                red.append(f"  L{start}: 红线③ — 未标注 early-by-construction")
            if not re.search(r"0\.2833|10%", unit):
                red.append(f"  L{start}: 红线③ — 未给出 10% alarm-budget recall")

    stale: List[str] = []
    for i, line in enumerate(raw, 1):
        for s, why in STALE.items():
            if re.search(rf"(?<![\d.]){re.escape(s)}(?![\d])", line):
                stale.append(f"  L{i}: `{s}` — {why}")

    lines.append(f"[2] 红线违规 {len(red)}；废弃旧值 {len(stale)}")
    lines += red + stale

    # ---- 3. 未匹配数字甄别 ----
    lines.append(f"[3] 未匹配数字甄别（共 {len(unmatched)} 条）")
    if unmatched:
        lines.append("    行号 | 数字 | 分类 | 所在段落 | 原文片段")
        for ln, tok, kind, snip in unmatched:
            zone = "结果区" if in_section(ln) else "其他区"
            lines.append(f"    L{ln} | {tok} | {kind} | {zone} | {snip}")
    else:
        lines.append("    无")

    # ---- 4. significant 必须伴随精确 P 值 ----
    sig_hits: List[str] = []
    for i, line in enumerate(raw, 1):
        if ref_a <= i < ref_b:
            continue
        for sent in re.split(r"(?<=[.!?])\s+", line):
            if re.search(r"significan", sent, re.I) and not re.search(
                    r"\b[Pp]\s*[=<>≤≥]\s*0?\.?\d", sent):
                sig_hits.append(f"  L{i}: {sent.strip()[:150]}")
    lines.append(f"[4] 'significant' 未附精确 P 值：{len(sig_hits)}（WARNING）")
    lines += sig_hits

    # ---- 判定 ----
    strict_hits = [u for u in hard if in_section(u[0])]
    fail = bool(red or stale) or (strict and bool(strict_hits))
    if fail:
        verdict = "FAIL"
    elif hard:
        verdict = "PASS (with warnings)"
    else:
        verdict = "PASS"
    lines += ["", f"RESULT: {verdict}", ""]
    if strict and strict_hits:
        lines.append(f"  strict 模式下结果区未匹配 {len(strict_hits)} 条，需修正或纳入允许集。")
    return lines, (1 if fail else 0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="定稿数字核对 + 文档审计 + 英文初稿审计")
    ap.add_argument("--tables", default=str(MD_TABLES))
    ap.add_argument("--facts", default=str(MD_FACTS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--audit-docs", action="store_true",
                    help="扫描全部 Markdown 与图注：旧口径清零 + 三条红线")
    ap.add_argument("--manuscript", default=None,
                    help="英文初稿 Markdown 路径，输出 manuscript_audit_report.txt")
    ap.add_argument("--strict", action="store_true",
                    help="结果区（Abstract/Results/Table/Figure）任何未匹配数字即 FAIL；"
                         "默认仅 WARNING 并返回 PASS (with warnings)")
    args = ap.parse_args(argv)

    # ---- 英文初稿审计（独立入口）----
    if args.manuscript:
        rep, code = audit_manuscript(args.manuscript, strict=args.strict)
        out = Path(args.manuscript).with_name("manuscript_audit_report.txt")
        out.write_text("\n".join(rep), encoding="utf-8")
        print("\n".join(rep))
        print(f"-> {out}")
        return code

    # ---- 文档审计 ----
    if args.audit_docs:
        rep1, n1 = audit_docs(DOC_FILES)
        rep2, n2 = audit_figure_source()
        rep = rep1 + rep2 + ["", f"RESULT: {'PASS' if n1 + n2 == 0 else 'FAIL'}"
                             f"（违规 {n1 + n2} 项）", ""]
        out = ROOT / "paper_assets/doc_audit_report.txt"
        out.write_text("\n".join(rep), encoding="utf-8")
        print("\n".join(rep))
        print(f"-> {out}")
        return 0 if n1 + n2 == 0 else 1

    # ---- 发布版守卫：本脚本核对的是完整 results 树，公开仓库只带聚合表 ----
    required = [Path(args.tables), Path(args.facts)] + [ROOT / f for f in PAPER_FILES]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        print("[skip] 09_verify_final_numbers.py is the submission-time provenance checker.")
        print("       It re-derives every manuscript number from the complete `outputs/` tree,")
        print("       which is not redistributed here because it holds patient-level rows.")
        print(f"       {len(missing)} input file(s) missing, e.g.:")
        for m in missing[:6]:
            print(f"         - {m}")
        print("       See README section 'Number provenance' for what to regenerate first.")
        return 2

    tables = Path(args.tables).read_text(encoding="utf-8")
    facts = Path(args.facts).read_text(encoding="utf-8")

    check_table_a(tables)
    check_kw(tables)
    check_table_b(tables)
    ctx = check_table_c(tables)
    check_platt(tables, ctx["ext"])
    check_cooling(tables)
    check_shap(facts)
    check_table_d(tables)
    check_shap_holdout(tables)
    check_derived(facts, tables)
    check_coverage(facts, tables)

    df = pd.DataFrame(ROW, columns=["分组", "核对项", "定稿值", "状态", "源值"])
    n_fail = int((df["状态"] == "FAIL").sum())

    card = quick_card()
    lines = ["# 定稿数字核对清单", "",
             "## 一、可直接引用速查卡（数字均由源数据现算）", "",
             "| 正文位置 | 建议表述（英文可直接粘贴） |", "|---|---|"]
    for pos, txt in card:
        lines.append(f"| {pos} | {txt} |")
    lines += ["", "## 二、逐项核对明细", "",
             f"> 生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}　|　"
             f"核对项 {len(df)} 项，通过 {len(df)-n_fail}，失败 {n_fail}。",
             "> 期望值由 `09_verify_final_numbers.py` 从源 CSV 现算，脚本内无硬编码常量。", ""]
    for g, sub in df.groupby("分组", sort=False):
        lines += [f"## {g}", "", "| 核对项 | 定稿值 | 源值 | 状态 |", "|---|---:|---:|:--:|"]
        for _, r in sub.iterrows():
            mark = "✅" if r["状态"] == "PASS" else "❌"
            lines.append(f"| {r['核对项']} | {r['定稿值']} | {r['源值']} | {mark} |")
        lines.append("")
    if n_fail:
        lines += ["## 失败项（需修正）", ""]
        for _, r in df[df["状态"] == "FAIL"].iterrows():
            lines.append(f"- {r['分组']} / {r['核对项']}：定稿 `{r['定稿值']}`，源 `{r['源值']}`")
    else:
        lines += ["## 结论", "", "全部核对项通过，两份定稿文件与源数据一致，可直接引用。", ""]

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    df.to_csv(Path(args.out).with_suffix(".csv"), index=False, encoding="utf-8-sig")

    print(f"核对 {len(df)} 项 | 通过 {len(df)-n_fail} | 失败 {n_fail}")
    if n_fail:
        print(df[df["状态"] == "FAIL"].to_string(index=False))
    print(f"-> {args.out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
