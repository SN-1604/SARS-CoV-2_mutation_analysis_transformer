# -*- coding: utf-8 -*-
"""smooth_entropy_monthly.py — 月度熵平滑 -> entropy_monthly_smoothed/
方法与旧管线 weekly_entropy_smoothed 完全一致 (已在 90/90 个周文件上精确复现验证):
  5 点居中滑动平均, 边缘截断 (pandas rolling(5, center=True, min_periods=1).mean()),
  按月份位置取窗, 不插值缺失月份。
输入:  entropy_monthly/{国家}_monthly_entropy.npy  (dict {"YYYY-MM": float})
输出:  entropy_monthly_smoothed/{国家}_monthly_entropy.npy (同构 dict)
"""
import os

import numpy as np
import pandas as pd

SRC = "entropy_monthly"
DST = "entropy_monthly_smoothed"


def main():
    os.makedirs(DST, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(SRC)):
        if not f.endswith("_monthly_entropy.npy"):
            continue
        d = np.load(os.path.join(SRC, f), allow_pickle=True).item()
        ks = sorted(d.keys())
        sm = pd.Series([d[k] for k in ks]).rolling(5, center=True, min_periods=1).mean()
        out = {k: float(v) for k, v in zip(ks, sm)}
        np.save(os.path.join(DST, f), out)
        n += 1
    print(f"smoothed {n} countries -> {DST}/")


if __name__ == "__main__":
    main()
