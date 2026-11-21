"""05_clinical_metrics.py — 顶刊临床评价指标计算器

状态
----
核心指标已实现并可自检（`--self-test`）。输入是模块 4 的预测输出：
每行一个窗口，至少包含 `patient_id, t_end_h, y, t_aki_onset, pred_prob`。

指标分三类
----------
A. 判别与校准（窗口级）
   AUROC、AUPRC、Brier、Sensitivity、Specificity

B. 临床报警（这是"早期预警"论文的真正卖点，必须报）
   ER  = 被捕获的 AKI 患者数 / AKI 患者总数
         「捕获」定义：发病前 24h 内至少有一次真阳性报警
   FAR = 1 - 真报警数 / 总报警数 = 误报 / (真报+误报)      （窗口级）
   Median Lead-time = median(T_AKI - T_first_true_alarm)   （小时）
         只在被捕获的患者上计算；未捕获者没有提前量，不能按 0 计入

C. 临床效用
   DCA 净获益 NB(pt) = TP/n - FP/n × pt/(1-pt)

三个容易算错的地方
----------------
1. **Lead-time 只在被捕获患者上算中位数**。把未捕获患者记为 0 会把提前量严重拉低。
2. **FAR 是窗口级**，一位长住院患者会产生几十个窗口，误报会被放大；
   报告时必须同时给"每患者每日误报数"，否则读者无法判断临床负担。
3. **AUROC 的 CI 要按患者重抽样**（bootstrap by patient_id），
   按窗口重抽样会把 CI 压得过窄，因为同一患者的窗口高度相关。

用法
----
    python 05_clinical_metrics.py --self-test
    python 05_clinical_metrics.py --pred preds.csv --threshold 0.35
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from stats_utils import benjamini_hochberg, holm, tukey_hsd


# =============================================================================
# A. 判别与校准
# =============================================================================
def discrimination(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    out = {"n": int(y.size), "n_pos": int(y.sum()),
           "prevalence": float(y.mean()) if y.size else np.nan}
    if 0 < y.sum() < y.size:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    else:
        out["auroc"] = np.nan
        out["auprc"] = np.nan
    out["brier"] = float(brier_score_loss(y, p)) if y.size else np.nan
    return out


def confusion(y: np.ndarray, alarm: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    a = np.asarray(alarm, dtype=bool)
    tp = int((a & (y == 1)).sum())
    fp = int((a & (y == 0)).sum())
    fn = int((~a & (y == 1)).sum())
    tn = int((~a & (y == 0)).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
    }


# =============================================================================
# B. 临床报警
# =============================================================================
def event_recall(df: pd.DataFrame, alarm_col: str = "alarm",
                 horizon_h: float = 24.0) -> Dict[str, float]:
    """患者级事件召回率 ER。

    df 需含 patient_id, t_end_h, t_aki_onset, alarm。
    捕获 = 该患者在 [onset-horizon, onset) 内至少有一次 alarm=True。
    """
    aki = df[df["t_aki_onset"].notna()]
    if aki.empty:
        return {"n_aki_patients": 0, "n_captured": 0, "event_recall": np.nan,
                "median_lead_h": np.nan}

    captured, leads = 0, []
    for pid, g in aki.groupby("patient_id"):
        onset = float(g["t_aki_onset"].iloc[0])
        pre = g[(g["t_end_h"] >= onset - horizon_h) & (g["t_end_h"] < onset) & g[alarm_col]]
        if len(pre):
            captured += 1
            leads.append(onset - float(pre["t_end_h"].min()))
    return {
        "n_aki_patients": int(aki["patient_id"].nunique()),
        "n_captured": captured,
        "event_recall": captured / aki["patient_id"].nunique(),
        "median_lead_h": float(np.median(leads)) if leads else np.nan,
        "iqr_lead_h": (float(np.percentile(leads, 25)), float(np.percentile(leads, 75)))
                      if leads else (np.nan, np.nan),
    }


def false_alarm_rate(df: pd.DataFrame, alarm_col: str = "alarm") -> Dict[str, float]:
    """窗口级误报率 FAR = FP / (TP + FP)，并给出每患者每日误报数。"""
    a = df[alarm_col].to_numpy(bool)
    y = df["y"].to_numpy(int)
    tp = int((a & (y == 1)).sum())
    fp = int((a & (y == 0)).sum())
    total = tp + fp
    n_pat = max(df["patient_id"].nunique(), 1)
    # 每患者每日误报：分母用**真实监护时长**（每患者首末窗口跨度，步长 1h），
    # 不能拿患者数当分母——那样"每患者误报"会被误读成"每日误报"。
    span = (df.groupby("patient_id")["t_end_h"].agg(lambda x: x.max() - x.min() + 1.0))
    pat_days = max(float(span.sum()) / 24.0, 1e-6)
    return {
        "n_alarms": total, "true_alarms": tp, "false_alarms": fp,
        "far": (fp / total) if total else np.nan,
        "false_alarms_per_patient": fp / n_pat,
        "false_alarms_per_patient_day": fp / pat_days,
        "patient_days": pat_days,
    }


# =============================================================================
# C. 决策曲线
# =============================================================================
def decision_curve(y: np.ndarray, p: np.ndarray,
                   thresholds: Optional[Sequence[float]] = None) -> pd.DataFrame:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if thresholds is None:
        thresholds = np.arange(0.01, 0.51, 0.01)
    n = y.size
    prev = y.mean() if n else np.nan
    rows = []
    for pt in thresholds:
        a = p >= pt
        tp = int((a & (y == 1)).sum())
        fp = int((a & (y == 0)).sum())
        nb = tp / n - fp / n * (pt / (1 - pt))
        rows.append({"threshold": float(pt), "net_benefit": nb,
                     "treat_all": prev - (1 - prev) * (pt / (1 - pt)),
                     "treat_none": 0.0, "tp": tp, "fp": fp})
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame, threshold: float = 0.5,
             prob_col: str = "pred_prob") -> Dict[str, float]:
    """一次算全。df 需含 patient_id, t_end_h, y, t_aki_onset, pred_prob。"""
    need = {"patient_id", "t_end_h", "y", "t_aki_onset", prob_col}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"缺少列 {sorted(missing)}")

    out = discrimination(df["y"], df[prob_col])
    df = df.assign(alarm=df[prob_col].to_numpy(float) >= threshold)
    out.update(confusion(df["y"], df["alarm"]))
    out.update(event_recall(df))
    out.update(false_alarm_rate(df))
    out["threshold"] = threshold
    return out




# =============================================================================
# D. 额外指标与统计检验
# =============================================================================
def nne(ppv: float) -> float:
    """Number Needed to Evaluate = 1/PPV：每检出 1 例真阳性需要评估多少例报警。"""
    return float("inf") if not ppv or ppv <= 0 else 1.0 / ppv


def patient_level(df: pd.DataFrame, prob_col: str = "pred_prob") -> Dict[str, float]:
    """患者级判别：取每位患者的**最大**预测风险，与"该患者是否发生 AKI"比较。

    窗口级 AUROC 会因为同一患者贡献几十个相关窗口而虚高；患者级指标更贴近临床决策
    （"这个患者要不要加强监护"），两者必须同时报告。
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    g = df.groupby("patient_id").agg(pred=(prob_col, "max"), y=("y", "max"))
    y = g["y"].astype(int).to_numpy()
    p = g["pred"].to_numpy(float)
    out = {"n_patients": int(len(g)), "n_aki_patients": int(y.sum())}
    if 0 < y.sum() < y.size:
        out["patient_auroc"] = float(roc_auc_score(y, p))
        out["patient_auprc"] = float(average_precision_score(y, p))
    else:
        out["patient_auroc"] = np.nan
        out["patient_auprc"] = np.nan
    return out


def dunn_test(samples: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Dunn 事后检验（Kruskal-Wallis 显著后使用），Holm 校正。"""
    from scipy import stats
    vals = np.concatenate(list(samples.values()))
    N = vals.size
    ranks = stats.rankdata(vals)
    R, pos = {}, 0
    for k, v in samples.items():
        R[k] = ranks[pos:pos + v.size].mean()
        pos += v.size
    _, cnt = np.unique(vals, return_counts=True)
    tie = float(np.sum(cnt ** 3 - cnt))
    sigma2 = N * (N + 1) / 12.0 - tie / (12.0 * (N - 1))
    keys = list(samples)
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            z = (R[a] - R[b]) / np.sqrt(sigma2 * (1 / samples[a].size + 1 / samples[b].size))
            rows.append({"group1": a, "group2": b, "z": float(z),
                         "p_raw": float(2 * (1 - stats.norm.cdf(abs(z))))})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_raw"].to_numpy())
    return out


def compare_models(metrics_df: pd.DataFrame, metric: str = "auroc",
                   group_col: str = "model") -> Dict[str, pd.DataFrame]:
    """Kruskal-Wallis + Dunn(Holm) + Tukey HSD 三件套。

    metrics_df 需含 group_col 与 metric 两列（每行 = 一个折的指标）。
    """
    from scipy import stats
    d = metrics_df[[group_col, metric]].dropna()
    groups = {k: g[metric].to_numpy(float) for k, g in d.groupby(group_col) if len(g) >= 3}
    out: Dict[str, pd.DataFrame] = {}
    if len(groups) < 2:
        return out
    H, p = stats.kruskal(*groups.values())
    out["kruskal"] = pd.DataFrame([{"test": "Kruskal-Wallis", "H": float(H), "p": float(p),
                                    "n_groups": len(groups)}])
    out["dunn"] = dunn_test(groups)
    tk = tukey_hsd(d[metric].to_numpy(float), d[group_col].to_numpy())
    if tk is not None:
        out["tukey"] = tk
    else:
        out["tukey"] = pd.DataFrame([{"note": "statsmodels 不可用，Tukey HSD 已跳过；"
                                              "Dunn(Holm) 已给出等价的事后两两比较"}])
    return out


def feature_group_test(df: pd.DataFrame, feats: Sequence[str],
                       out_path: Optional[str] = None) -> pd.DataFrame:
    """AKI vs 非 AKI 的特征组间 t 检验 + Benjamini-Hochberg FDR。"""
    from scipy import stats
    pos, neg = df[df["y"] == 1], df[df["y"] == 0]
    rows = []
    for f in feats:
        a, b = pos[f].dropna().to_numpy(float), neg[f].dropna().to_numpy(float)
        if a.size < 10 or b.size < 10:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"feature": f, "mean_aki": a.mean(), "mean_non": b.mean(),
                     "diff": a.mean() - b.mean(), "t": t, "p": p})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["p_bh"] = benjamini_hochberg(out["p"].to_numpy())
    out["significant_bh"] = out["p_bh"] < 0.05
    out = out.sort_values("p").reset_index(drop=True)
    if out_path:
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out




# =============================================================================
# E. 时序平滑与报警冷却期（降低 FAR 的临床后处理）
# =============================================================================
def smooth_probs(df: pd.DataFrame, alpha: float = 0.5,
                 prob_col: str = "pred_prob") -> pd.DataFrame:
    """按患者时间序做 EWMA：p_s[t] = α·p[t] + (1-α)·p_s[t-1]。

    目的：单窗口的尖峰噪声不应触发报警。α=0.5 相当于半衰期 1 个窗口。
    """
    d = df.sort_values(["patient_id", "t_end_h"]).copy()
    out = []
    for _, g in d.groupby("patient_id", sort=False):
        p = g[prob_col].to_numpy(float)
        s = np.empty_like(p)
        acc = p[0]
        for i, v in enumerate(p):
            acc = v if i == 0 else alpha * v + (1 - alpha) * acc
            s[i] = acc
        out.append(g.assign(p_smooth=s))
    return pd.concat(out, ignore_index=True)


def apply_cooldown(df: pd.DataFrame, threshold: float = 0.3,
                   cooldown_h: float = 6.0, prob_col: str = "p_smooth") -> pd.DataFrame:
    """报警冷却：一次触发后 (t0, t0+cooldown_h] 内不再产生新报警。

    只抑制"新报警"，不改变概率；被抑制的窗口按非报警处理。
    这是后处理，必须在论文里与模型性能分开报告。
    """
    d = df.sort_values(["patient_id", "t_end_h"]).copy()
    alarms = np.zeros(len(d), dtype=bool)
    pos = 0
    for _, g in d.groupby("patient_id", sort=False):
        last = -np.inf
        for t, p in zip(g["t_end_h"].to_numpy(float), g[prob_col].to_numpy(float)):
            if p >= threshold and t > last + cooldown_h:
                alarms[pos] = True
                last = t
            pos += 1
    d["alarm"] = alarms
    return d


def evaluate_policy(df: pd.DataFrame, threshold: float = 0.3, alpha: Optional[float] = None,
                    cooldown_h: float = 0.0, prob_col: str = "pred_prob") -> Dict[str, float]:
    """在给定（平滑, 冷却）策略下算全套临床指标。"""
    d = smooth_probs(df, alpha, prob_col) if alpha else df.copy()
    if alpha:
        prob_col = "p_smooth"
    d = apply_cooldown(d, threshold, cooldown_h, prob_col)
    out = confusion(d["y"], d["alarm"])
    out.update(event_recall(d))
    out.update(false_alarm_rate(d))
    out["n_alarms"] = int(d["alarm"].sum())
    return out


def compare_policies(df: pd.DataFrame, threshold: float = 0.3,
                     alphas: Sequence[Optional[float]] = (None, 0.5),
                     cooldowns: Sequence[float] = (0.0, 6.0)) -> pd.DataFrame:
    """平滑 × 冷却 的对照表。"""
    rows = []
    for a in alphas:
        for cd in cooldowns:
            r = evaluate_policy(df, threshold, a, cd)
            rows.append({
                "alpha": "raw" if a is None else a,
                "cooldown_h": cd,
                "event_recall": r["event_recall"],
                "window_far": r["far"],
                "false_alarms_per_patient": r["false_alarms_per_patient"],
                "median_lead_h": r["median_lead_h"],
                "sensitivity": r["sensitivity"],
                "specificity": r["specificity"],
                "nne": nne(r["ppv"]),
                "n_alarms": r["n_alarms"],
            })
    return pd.DataFrame(rows)




# =============================================================================
# F. 多阈值 × 冷却 对比（论文主表）
# =============================================================================
def _threshold_for_sens(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """取灵敏度 ≈ target 的阈值。"""
    pos = np.sort(p[np.asarray(y) == 1])[::-1]
    if pos.size == 0:
        return float("nan")
    k = int(np.clip(round(target * pos.size), 1, pos.size)) - 1
    return float(pos[k])


def _threshold_for_spec(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """取特异度 ≈ target 的阈值。"""
    neg = np.sort(p[np.asarray(y) == 0])
    if neg.size == 0:
        return float("nan")
    k = int(np.clip(round(target * neg.size), 1, neg.size)) - 1
    return float(neg[k])


def _threshold_youden(y: np.ndarray, p: np.ndarray) -> float:
    """Youden's J = Sens + Spec - 1 最大处的阈值。"""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(np.asarray(y, dtype=int), np.asarray(p, float))
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def compare_policies_multi(df: pd.DataFrame,
                           points: Sequence[Tuple[str, str, float]] = (
                               ("high_sensitivity", "sens", 0.85),
                               ("balanced_youden", "youden", 0.0),
                               ("high_specificity", "spec", 0.92)),
                           alpha: float = 0.5, cooldown_h: float = 6.0,
                           prob_col: str = "pred_prob") -> pd.DataFrame:
    """三个临床操作点 × {Raw, EWMA+冷却} 的对比表。

    阈值在**原始概率**上按目标灵敏度/特异度确定，再原样用于平滑后概率
    （若对平滑序列重新取阈值，等于换了操作点，两行就不可比）。
    """
    y = df["y"].to_numpy(int)
    p = df[prob_col].to_numpy(float)
    rows = []
    for name, kind, target in points:
        if kind == "sens":
            thr = _threshold_for_sens(y, p, target)
        elif kind == "spec":
            thr = _threshold_for_spec(y, p, target)
        else:
            thr = _threshold_youden(y, p)
        if not np.isfinite(thr):
            continue

        for policy in ("raw", "ewma_cooldown"):
            if policy == "raw":
                d = df.copy()
                d["alarm"] = d[prob_col].to_numpy(float) >= thr
            else:
                d = smooth_probs(df, alpha, prob_col)
                d = apply_cooldown(d, thr, cooldown_h, "p_smooth")
            conf = confusion(d["y"], d["alarm"])
            er = event_recall(d)
            fa = false_alarm_rate(d)
            rows.append({
                "operating_point": name, "target": f"{kind}={target:g}",
                "threshold": thr, "policy": policy,
                "cooldown_h": 0.0 if policy == "raw" else cooldown_h,
                "alpha": "" if policy == "raw" else alpha,
                "event_recall": er["event_recall"],
                "window_far": fa["far"],
                "false_alarms_per_patient": fa["false_alarms_per_patient"],
                "false_alarms_per_patient_day": fa["false_alarms_per_patient_day"],
                "median_lead_h": er["median_lead_h"],
                "sensitivity": conf["sensitivity"], "specificity": conf["specificity"],
                "ppv": conf["ppv"], "nne": nne(conf["ppv"]),
                "n_alarms": fa["n_alarms"],
            })
    return pd.DataFrame(rows)


# =============================================================================
# 自检
# =============================================================================
def _toy() -> pd.DataFrame:
    rows = []
    # A: onset 30，窗口 12-29 全为阳性；仅在 t_end=20 报警 -> 捕获，提前 10h
    for t in range(12, 30):
        rows.append({"patient_id": "A", "t_end_h": t, "y": 1, "t_aki_onset": 30.0,
                     "pred_prob": 0.9 if t == 20 else 0.1})
    # B: onset 40；仅 t_end=30 报警 -> 捕获，提前 10h
    for t in range(12, 40):
        rows.append({"patient_id": "B", "t_end_h": t,
                     "y": 1 if t > 16 else 0, "t_aki_onset": 40.0,
                     "pred_prob": 0.8 if t == 30 else 0.1})
    # C: 无 AKI；3 次误报
    for t in range(12, 51):
        rows.append({"patient_id": "C", "t_end_h": t, "y": 0, "t_aki_onset": np.nan,
                     "pred_prob": 0.7 if t in (20, 21, 22) else 0.05})
    # D: onset 50；从不报警 -> 未捕获
    for t in range(12, 50):
        rows.append({"patient_id": "D", "t_end_h": t,
                     "y": 1 if t > 26 else 0, "t_aki_onset": 50.0,
                     "pred_prob": 0.05})
    return pd.DataFrame(rows)


def run_self_test() -> int:
    fails: List[str] = []
    df = _toy()
    r = evaluate(df, threshold=0.5)

    if r["n_aki_patients"] != 3:
        fails.append(f"AKI 患者数应为 3，实得 {r['n_aki_patients']}")
    if r["n_captured"] != 2:
        fails.append(f"捕获患者数应为 2，实得 {r['n_captured']}")
    if not np.isclose(r["event_recall"], 2 / 3):
        fails.append(f"ER 应为 2/3，实得 {r['event_recall']}")
    if r["n_alarms"] != 5 or r["true_alarms"] != 2:
        fails.append(f"报警数应 5（真 2），实得 {r['n_alarms']}（真 {r['true_alarms']}）")
    if not np.isclose(r["far"], 0.6):
        fails.append(f"FAR 应为 0.6，实得 {r['far']}")
    if not np.isclose(r["median_lead_h"], 10.0):
        fails.append(f"中位提前量应为 10h，实得 {r['median_lead_h']}")

    # 漏报患者不能按 0 提前量计入中位数
    if r["n_captured"] == 3:
        fails.append("未捕获患者被错误计入提前量")

    if not np.isclose(nne(r["ppv"]), 1 / r["ppv"]):
        fails.append("NNE 应等于 1/PPV")

    pl = patient_level(df)
    if pl["n_patients"] != 4 or pl["n_aki_patients"] != 3:
        fails.append(f"患者级样本数应为 4/3，实得 {pl['n_patients']}/{pl['n_aki_patients']}")

    # 冷却期：C 在 t=20,21,22 连续报警，6h 冷却后只应保留 t=20
    dd = apply_cooldown(df.assign(p_smooth=df["pred_prob"]), 0.5, 6.0)
    c = dd[dd["patient_id"] == "C"]
    if int(c["alarm"].sum()) != 1 or float(c.loc[c["alarm"], "t_end_h"].iloc[0]) != 20:
        fails.append(f"冷却期失效：C 的报警数 {int(c['alarm'].sum())}，"
                     f"首个 {c.loc[c['alarm'],'t_end_h'].tolist()}")
    # EWMA：首值等于原值
    sm = smooth_probs(df, 0.5)
    if not np.isclose(sm.loc[sm["patient_id"] == "A", "p_smooth"].iloc[0], 0.1):
        fails.append("EWMA 首值应等于原始概率")

    # DCA 端点：阈值极低时净获益接近 treat_all
    dca = decision_curve(df["y"].to_numpy(int), df["pred_prob"].to_numpy(float),
                         thresholds=[0.05, 0.5, 0.9])
    if not (dca.iloc[0]["net_benefit"] > dca.iloc[1]["net_benefit"]):
        fails.append("DCA: 阈值 0.05 的净获益应高于 0.5")

    print("=" * 70)
    print("模块5 自检：ER / FAR / Lead-time / DCA / 判别指标")
    print("=" * 70)
    for f in fails:
        print("  [FAIL]", f)
    print(f"  ER={r['event_recall']:.3f}  FAR={r['far']:.3f}  "
          f"Lead={r['median_lead_h']:.1f}h  AUROC={r['auroc']:.3f}")
    print(f"  失败 {len(fails)} 项")
    return 1 if fails else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="临床评价指标")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pred", help="含 patient_id/t_end_h/y/t_aki_onset/pred_prob 的 CSV")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--compare", help="cv_results_by_fold.csv，做模型间统计检验")
    ap.add_argument("--metric", default="auroc")
    ap.add_argument("--alpha", type=float, default=0.5, help="EWMA 平滑系数")
    ap.add_argument("--cooldown", type=float, default=6.0, help="报警冷却小时数")
    ap.add_argument("--policy-table", action="store_true", help="输出平滑×冷却对照表")
    ap.add_argument("--policy-multi", action="store_true",
                    help="输出三操作点（高敏/Youden/高特异）× 冷却 对比表")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if args.compare:
        d = pd.read_csv(args.compare)
        d = d[d["status"] == "ok"]
        res = compare_models(d, args.metric)
        for k, v in res.items():
            print(f"\n=== {k} ===")
            print(v.round(4).to_string(index=False))
        return 0

    if not args.pred:
        ap.error("需要 --pred 或 --compare（或 --self-test）")

    df = pd.read_csv(args.pred)
    if args.policy_multi:
        tab = compare_policies_multi(df, alpha=args.alpha, cooldown_h=args.cooldown)
        print("\n=== 多阈值 × 冷却 对比 ===")
        print(tab.round(4).to_string(index=False))
        out = Path(args.pred).with_name("policy_comparison_clean.csv")
        tab.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n已写出 {out}")
        return 0

    if args.policy_table:
        tab = compare_policies(df, args.threshold, (None, args.alpha), (0.0, args.cooldown))
        print("\n=== 平滑 × 冷却 策略对照 ===")
        print(tab.round(4).to_string(index=False))
        out = Path(args.pred).with_name("policy_comparison.csv")
        tab.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n已写出 {out}")
        return 0

    r = evaluate(df, args.threshold)
    r.update(patient_level(df))
    r["nne"] = nne(r.get("ppv", np.nan))
    for k, v in r.items():
        print(f"{k:>26}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
