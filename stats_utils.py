#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats_utils.py — 多重比较校正的本地实现
=====================================

本机 `statsmodels` 无法安装（安装进程被系统反复终止），而它在本项目里只用到两个函数：
`multipletests` 与 `pairwise_tukeyhsd`。为避免整条流水线被一个可选依赖卡死，
这里用纯 numpy 实现等价功能，并保留 statsmodels 可用时的优先调用。

实现内容
--------
* benjamini_hochberg(p)  BH-FDR，等价 statsmodels multipletests(method='fdr_bh')
* holm(p)                Holm 逐步法，等价 method='holm'
* tukey_hsd(...)         statsmodels 可用时直接调用；否则返回 None（由调用方显式标注跳过）
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _as_array(p: Sequence[float]) -> np.ndarray:
    return np.asarray(p, dtype=float)


def benjamini_hochberg(p: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR 校正，返回与输入同序的 q 值。

    步骤：升序排序 -> q_(i) = p_(i) * m / i -> 从大到小取累积最小值 -> 截断到 1。
    """
    p = _as_array(p)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m, dtype=float)
    out[order] = np.clip(ranked, 0, 1)
    return out


def holm(p: Sequence[float]) -> np.ndarray:
    """Holm 逐步校正（控制 FWER），返回与输入同序的校正 p 值。"""
    p = _as_array(p)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * (m - np.arange(m))
    ranked = np.maximum.accumulate(ranked)   # Holm 需单调不减
    out = np.empty(m, dtype=float)
    out[order] = np.clip(ranked, 0, 1)
    return out


def tukey_hsd(values: Sequence[float], groups: Sequence[str]
              ) -> Optional[pd.DataFrame]:
    """Tukey HSD。statsmodels 不可用时返回 None，调用方需显式标注"已跳过"。"""
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    tk = pairwise_tukeyhsd(np.asarray(values, float), np.asarray(groups))
    return pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])


def self_test() -> int:
    fails = []
    # BH：教科书例子 p=[0.01,0.02,0.03,0.04,0.05]，m=5
    q = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
    expect = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
    if not np.allclose(q, expect):
        fails.append(f"BH 期望 {expect}，实得 {np.round(q,4)}")
    # 单调性：q 不应小于原始 p
    if (q < np.array([0.01, 0.02, 0.03, 0.04, 0.05]) - 1e-12).any():
        fails.append("BH: q 小于原始 p")
    # Holm：p=[0.01,0.04,0.03] -> 排序 0.01,0.03,0.04 -> *3,*2,*1 = 0.03,0.06,0.04
    # 单调不减累积最大 -> 0.03,0.06,0.06 -> 映射回原序 [0.03,0.06,0.06]
    h = holm([0.01, 0.04, 0.03])
    if not np.allclose(h, [0.03, 0.06, 0.06]):
        fails.append(f"Holm 期望 [0.03,0.06,0.06]，实得 {np.round(h,4)}")
    # 单元素与空
    if not np.allclose(benjamini_hochberg([0.02]), [0.02]):
        fails.append("BH 单元素异常")
    if benjamini_hochberg([]).size != 0:
        fails.append("BH 空输入异常")

    print("=" * 60)
    print("stats_utils 自检")
    print("=" * 60)
    for f in fails:
        print("  [FAIL]", f)
    print(f"  失败 {len(fails)} 项")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
