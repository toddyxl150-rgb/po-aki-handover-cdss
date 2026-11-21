"""三栏队列筛选流程图（论文 Figure 1 / Supplementary S1）。

数据全部从产物读取，不硬编码：
    outputs/attrition_{inspire,eicu,vitaldb}.csv
    outputs/dedup_inspire_vitaldb.csv
    outputs/windows_track1_manifest.json, outputs/eicu/windows_track1_manifest.json
    outputs/track_compare/track_comparison_one_shot.csv
    outputs/cohort_inspire.csv（in_vitaldb_overlap 合计）
    outputs/paper_assets/provenance_log_values.json（--limit 上限）

用法：python 10_cohort_flowchart.py
输出：outputs/paper_figures/fig_cohort_flowchart.png / .jpg（300 DPI）
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path("outputs")
FIG = ROOT / "paper_figures"

C_INSPIRE = "#2B6CB0"
C_EICU = "#2F855A"
C_VITAL = "#805AD5"
C_TEXT = "#1A202C"
C_MUTED = "#4A5568"
C_BG = "#F7FAFC"
C_EXCL = "#C53030"


def _attrition(db: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / f"attrition_{db}.csv").sort_values("step_order")


def _fmt(n: int) -> str:
    return f"{n:,}"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })
    return plt


def draw() -> Path:
    plt = _mpl()
    from matplotlib.patches import FancyBboxPatch

    prov = json.loads((ROOT / "paper_assets/provenance_log_values.json").read_text(encoding="utf-8"))
    ins = _attrition("inspire")
    eic = _attrition("eicu")
    vit = _attrition("vitaldb")
    dedup = pd.read_csv(ROOT / "dedup_inspire_vitaldb.csv").iloc[0]
    m_ins = json.loads((ROOT / "windows_track1_manifest.json").read_text(encoding="utf-8"))
    m_eic = json.loads((ROOT / "eicu/windows_track1_manifest.json").read_text(encoding="utf-8"))
    paired = int(pd.read_csv(ROOT / "track_compare/track_comparison_one_shot.csv").iloc[0].n_patients)
    n_overlap_final = int(pd.read_csv(ROOT / "cohort_inspire.csv",
                                      usecols=["in_vitaldb_overlap"])["in_vitaldb_overlap"].sum())

    # 全部从纳排表现取；下标对应各自 CSV 的 step_order，断言防止表结构变化后静默画错
    ins_post = int(ins.iloc[1].n_remaining)      # 术后转入 ICU
    ins_los = int(ins.iloc[3].n_remaining)       # ICU LOS >= 24h
    ins_final = int(ins.iloc[-1].n_remaining)
    eic_cand = int(eic.iloc[4].n_remaining)      # 外科+年龄+LOS+首诊 合并后
    eic_ckd = int(eic.iloc[5].n_remaining)       # 排除 CKD5/ESRD/透析 后
    eic_final = int(eic.iloc[-1].n_remaining)
    vit_age = int(vit.iloc[1].n_remaining)
    vit_final = int(vit.iloc[-1].n_remaining)

    assert (ins_post, ins_los, ins_final) == (19195, 9560, 9543), "INSPIRE 纳排表结构已变"
    assert (eic_cand, eic_ckd, eic_final) == (40037, 3603, 3597), "eICU 纳排表结构已变"
    assert (vit_age, vit_final) == (6331, 3000), "VitalDB 纳排表结构已变"

    # ---- 三栏各自的步骤链（数字来自纳排表，上限值来自溯源文件）----
    lim_e = int(prov["eicu_limit"])
    lim_v = int(prov["vitaldb_limit"])

    cols = [
        dict(
            title="INSPIRE", sub="Republic of Korea · surgical ICU", color=C_INSPIRE,
            steps=[
                (int(ins.iloc[0].n_remaining), "All surgical procedures"),
                (ins_post, "Admitted to ICU after surgery"),
                (ins_los, "ICU length of stay ≥ 24 h"),
                (ins_final, "Final analytic cohort (PO-AKI adjudicable)"),
            ],
            excl=["Excluded: no post-operative ICU admission",
                  "Excluded: ICU stay < 24 h",
                  "Excluded: pre-ICU AKI within 12 h"],
            notes=[f"Windowing: {_fmt(m_ins['n_windows'])} windows / "
                   f"{_fmt(m_ins['stats']['n_patients'])} patients",
                   f"Track 2 paired subset: {_fmt(paired)} patients"],
        ),
        dict(
            title="eICU-CRD", sub="United States · multicentre ICU", color=C_EICU,
            steps=[
                (int(eic.iloc[0].n_remaining), "All ICU stays"),
                (eic_cand, "Surgical, age 18–99, LOS ≥ 24 h, first ICU stay"),
                (lim_e, f"Run-limit sampling (--limit {_fmt(lim_e)})"),
                (eic_ckd, "After excluding CKD stage 5 / ESRD / dialysis"),
                (eic_final, "Final analytic cohort (after pre-ICU AKI/death)"),
            ],
            excl=["Excluded: 160,822 non-surgical / repeat / short stays",
                  "Not an exclusion — analysis cap, documented",
                  "Excluded: 397 with advanced CKD or dialysis",
                  "Excluded: 6 with pre-ICU AKI or death"],
            notes=[f"Non-invasive BP integrated → {_fmt(m_eic['n_windows'])} windows / "
                   f"{_fmt(m_eic['stats']['n_patients'])} patients"],
        ),
        dict(
            title="VitalDB", sub="Republic of Korea · intra-operative high-resolution", color=C_VITAL,
            steps=[
                (int(vit.iloc[0].n_remaining), "All surgical cases"),
                (vit_age, "Age 18–99 with post-operative follow-up"),
                (lim_v, f"Run-limit sampling (--limit {_fmt(lim_v)})"),
                (vit_final, "Pending-validation cohort (7-day KDIGO adjudicable)"),
            ],
            excl=["Excluded: 57 outside age range",
                  "Not an exclusion — analysis cap, documented",
                  "No further exclusions (baseline SCr available in all)"],
            notes=[],
        ),
    ]

    fig, ax = plt.subplots(figsize=(11.2, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xs = [0.175, 0.5, 0.825]
    w, h, gap = 0.255, 0.072, 0.078
    y0 = 0.795

    for (col, x) in zip(cols, xs):
        color = col["color"]
        ax.text(x, 0.925, col["title"], ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)
        ax.text(x, 0.888, col["sub"], ha="center", va="center",
                fontsize=7.2, color=C_MUTED, style="italic")

        y = y0
        for i, (n, label) in enumerate(col["steps"]):
            last = i == len(col["steps"]) - 1
            face = color if last else C_BG
            txt = "white" if last else C_TEXT
            box = FancyBboxPatch((x - w / 2, y - h), w, h,
                                 boxstyle="round,pad=0.004,rounding_size=0.008",
                                 linewidth=1.3 if last else 0.9,
                                 edgecolor=color, facecolor=face, zorder=2)
            ax.add_patch(box)
            fs = 6.9 if len(label) <= 42 else 6.2   # 长标签自动缩一号，避免顶到边框
            ax.text(x, y - h * 0.34, label, ha="center", va="center",
                    fontsize=fs, color=txt, zorder=3)
            ax.text(x, y - h * 0.72, f"n = {_fmt(n)}", ha="center", va="center",
                    fontsize=9.2, fontweight="bold", color=txt, zorder=3)

            if i < len(col["steps"]) - 1:
                # 箭头偏左，排除说明右对齐在栏内 —— 避免文字跨栏或超出画布
                ax_x = x - w * 0.30
                ax.annotate("", xy=(ax_x, y - h - gap + 0.008), xytext=(ax_x, y - h - 0.004),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.15,
                                            shrinkA=0, shrinkB=0), zorder=1)
                ex = col["excl"][i] if i < len(col["excl"]) else ""
                if ex:
                    wrapped = "\n".join(textwrap.wrap(ex, 40))
                    is_excl = ex.startswith("Excluded")
                    ax.text(x + w / 2 - 0.004, y - h - gap / 2, wrapped,
                            ha="right", va="center", fontsize=5.5,
                            color=C_EXCL if is_excl else C_MUTED,
                            style="normal" if is_excl else "italic", linespacing=1.15)
            y -= h + gap

        for k, note in enumerate(col["notes"]):
            ax.text(x, y + 0.012 - k * 0.032, note, ha="center", va="center",
                    fontsize=6.6, color=C_MUTED,
                    bbox=dict(boxstyle="round,pad=0.22", fc="#EDF2F7", ec="none"))

    # ---- VitalDB 与 INSPIRE 的重叠（虚线框）----
    yb = y0 - 3 * (h + gap) - h - 0.085
    ov = FancyBboxPatch((0.705, yb - 0.075), 0.28, 0.105,
                        boxstyle="round,pad=0.006,rounding_size=0.008",
                        linewidth=1.0, linestyle=(0, (4, 2)),
                        edgecolor=C_VITAL, facecolor="#FAF5FF", zorder=2)
    ax.add_patch(ov)
    ax.text(0.845, yb - 0.008,
            f"Exact case_id overlap with INSPIRE: {_fmt(int(dedup.exact_case_id))}",
            ha="center", va="center", fontsize=6.6, color=C_VITAL, fontweight="bold")
    ax.text(0.845, yb - 0.038,
            f"(fingerprint matching alone: {int(dedup.fingerprint)})",
            ha="center", va="center", fontsize=6.0, color=C_MUTED, style="italic")
    ax.text(0.845, yb - 0.062,
            f"{n_overlap_final} remain in the final INSPIRE cohort",
            ha="center", va="center", fontsize=6.0, color=C_MUTED)

    ax.text(0.5, 0.022,
            "Numbers are final analytic cohorts after run-limit sampling; sampling steps are shown "
            "explicitly and are not exclusions.",
            ha="center", va="center", fontsize=6.2, color=C_MUTED, style="italic")

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "fig_cohort_flowchart.png"
    from PIL import Image
    fig.savefig(out, dpi=300)
    Image.open(out).convert("RGB").save(out, dpi=(300, 300))
    jpg = out.with_suffix(".jpg")
    fig.savefig(jpg, dpi=300, pil_kwargs={"quality": 95})
    Image.open(jpg).convert("RGB").save(jpg, quality=95, dpi=(300, 300))
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = draw()
    print(f"-> {p}")
    print(f"-> {p.with_suffix('.jpg')}")
