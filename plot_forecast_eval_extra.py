# -*- coding: utf-8 -*-
"""plot_forecast_eval_extra.py — 月度熵预测模型效果评价补充图（新形式）

输出(300 dpi):
  figure_monthly_revised/supplementary/figS28_forecast_residuals.png
    a 残差分布（对数密度阶梯直方图 + 正态参考）: 右偏重尾
    b 残差正态 QQ 图: 右尾显著重于正态
  figure_monthly_revised/supplementary/figS29_monthly_error_box.png
    逐验证月残差箱线图: 10月/12月误差抬升(熵突增期低估)
与既有 figS1(条形)/figS2(直方+散点)/figS3(分布带)/figS4(散点+跟踪线) 形式不重复。

数据: val_pred_monthly_v2.npz (静态校准口径, n=477, pooled R²=0.826)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, r"C:/Users/DFD/.agents/skills/figure-style")
import kernel as fsk

fsk.apply_figure_style(font="Microsoft YaHei", sizes=(8, 7, 6))
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "axes.labelpad": 6,
})

BLUE, ORANGE, RED, GREY = "#0072B2", "#E69F00", "#D55E00", "#666666"
OUT = "figure_monthly_revised/supplementary"
os.makedirs(OUT, exist_ok=True)

VP = np.load("val_pred_monthly_v2.npz", allow_pickle=True)
YT, YP = VP["y_true"], VP["y_pred"]
RES = YT - YP
MONTHS = VP["months"].astype(str)
N = len(RES)
VMONTHS = sorted(set(MONTHS))


def bbox_check(fig, tag):
    import matplotlib.text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = fig.get_tightbbox(r).get_points() * fig.dpi
    x0, y0 = float(pts[0][0]) - 6, float(pts[0][1]) - 6
    x1, y1 = float(pts[1][0]) + 6, float(pts[1][1]) + 6

    def in_box(bt):
        return bt.x0 >= x0 and bt.y0 >= y0 and bt.x1 <= x1 and bt.y1 <= y1

    stale = set()
    spines = []
    for ax in fig.axes:
        if not ax.axison:
            stale.update(ax.get_xticklabels(which="both"))
            stale.update(ax.get_yticklabels(which="both"))
            continue
        spines += [(s, s.get_window_extent(r)) for s in ax.spines.values()
                   if s.get_visible()]
        xlo, xhi = sorted(ax.get_xlim())
        ylo, yhi = sorted(ax.get_ylim())
        for t in ax.get_xticklabels(which="both"):
            if not (xlo <= t.get_position()[0] <= xhi):
                stale.add(t)
        for t in ax.get_yticklabels(which="both"):
            if not (ylo <= t.get_position()[1] <= yhi):
                stale.add(t)
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(matplotlib.text.Text)
             if t.get_text().strip() and t.get_visible() and t not in stale]
    ticklabels = {ax: set(ax.get_xticklabels(which="both")
                          + ax.get_yticklabels(which="both"))
                  for ax in fig.axes}
    overlaps = [(a.get_text()[:20], b.get_text()[:20])
                for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:]
                if ba.overlaps(bb)]
    overlaps += [(t.get_text()[:20], "spine")
                 for t, bt in texts for s, bs in spines
                 if bt.overlaps(bs) and t not in ticklabels[s.axes]]
    outside = [t.get_text()[:20] for t, bt in texts if not in_box(bt)]
    if overlaps or outside:
        print("[%s] bbox冲突:" % tag, overlaps, "越界:", outside)
    else:
        print("[%s] bbox检查通过" % tag)


# ---------------- figS28 残差结构 ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

# a) 残差分布(对数密度) + 正态参考
XLO, XHI = -20, 40
n_out = int((RES > XHI).sum())
bins = np.linspace(XLO, XHI, 61)
ax1.hist(RES, bins=bins, density=True, histtype="stepfilled", alpha=0.25,
         color=BLUE, lw=0)
ax1.hist(RES, bins=bins, density=True, histtype="step", lw=1.3, color=BLUE)
xs = np.linspace(XLO, XHI, 300)
ax1.plot(xs, stats.norm.pdf(xs, RES.mean(), RES.std()), color=GREY, lw=1.0,
         ls=(0, (4, 3)))
ax1.axvline(0, color="#333333", lw=0.8)
ax1.set_yscale("log")
ax1.set_ylim(5e-5, 0.3)
ax1.set_xlim(XLO, XHI)
ax1.set_xlabel("残差（观测−预测，比特/月）")
ax1.set_ylabel("密度（对数刻度）")
ax1.text(0.03, 0.97, "均值%+.1f，中位%+.1f\n正态参考为虚线"
         % (RES.mean(), np.median(RES)), transform=ax1.transAxes, fontsize=7,
         color=GREY, va="top")
ax1.text(0.97, 0.03, "%d个残差>%d未绘出（最大%.0f）" % (n_out, XHI, RES.max()),
         transform=ax1.transAxes, fontsize=6, color=GREY, ha="right",
         va="bottom")
ax1.set_title("残差右偏、重尾（偏度%.1f）" % stats.skew(RES))
fsk.panel_letter(ax1, "a")

# b) 正态 QQ 图
osm = stats.norm.ppf((np.arange(1, N + 1) - 0.5) / N)
osr = np.sort(RES)
q1, q3 = np.percentile(RES, [25, 75])
t1, t3 = stats.norm.ppf([0.25, 0.75])
slope = (q3 - q1) / (t3 - t1)
ref = lambda x: q1 + slope * (x - t1)
ax2.plot(osm, osr, "o", ms=3, color=BLUE, mew=0, alpha=0.5)
xl = [osm[0], osm[-1]]
ax2.plot(xl, [ref(xl[0]), ref(xl[1])], color=GREY, lw=1.0, ls=(0, (4, 3)))
ax2.set_xlabel("正态理论分位数")
ax2.set_ylabel("残差分位数（比特/月）")
ax2.text(0.03, 0.97, "参考线过第1/3四分位\n超额峰度%.0f" % stats.kurtosis(RES),
         transform=ax2.transAxes, fontsize=7, color=GREY, va="top")
ax2.set_title("右尾显著重于正态分布")
ax2.margins(0.04)
fsk.panel_letter(ax2, "b")
fig.tight_layout(w_pad=1.8)
fig.savefig(os.path.join(OUT, "figS28_forecast_residuals.png"))
bbox_check(fig, "figS28")
plt.close(fig)

# ---------------- figS29 逐月残差箱线 ----------------
fig, ax = plt.subplots(figsize=(4.6, 2.9))
data = [RES[MONTHS == m] for m in VMONTHS]
bp = ax.boxplot(data, positions=np.arange(len(VMONTHS)), widths=0.55,
                showfliers=True, patch_artist=True,
                boxprops=dict(facecolor=BLUE, alpha=0.45, lw=0.6),
                medianprops=dict(color="#08306B", lw=1.4),
                whiskerprops=dict(color=BLUE, lw=0.7),
                capprops=dict(color=BLUE, lw=0.7),
                flierprops=dict(marker="o", ms=2, mfc=GREY, mec="none",
                                alpha=0.5))
ax.axhline(0, color="#333333", lw=0.8, ls=(0, (4, 3)))
ax.set_ylim(-20, 44)
n_fly = int(sum((d > 44).sum() for d in data))
for i, (m, d) in enumerate(zip(VMONTHS, data)):
    hi = np.percentile(d, 75) + 1.5 * (np.percentile(d, 75)
                                       - np.percentile(d, 25))
    mae = np.abs(d).mean()
    col = RED if mae > 3.5 else GREY
    ax.text(i, min(max(d.max(), hi) + 2.0, 41), "MAE=%.1f" % mae,
            fontsize=6, color=col, ha="center", va="bottom")
ax.set_xticks(np.arange(len(VMONTHS)))
ax.set_xticklabels(["%s\nn=%d" % (m, len(d)) for m, d in zip(VMONTHS, data)],
                   fontsize=6)
ax.set_ylabel("残差（观测−预测，比特/月）")
ax.set_xlabel("验证月份")
if n_fly:
    ax.text(0.98, 0.03, "%d个残差>44未绘出" % n_fly, transform=ax.transAxes,
            fontsize=6, color=GREY, ha="right", va="bottom")
ax.set_title("10月、12月误差抬升（熵突增期低估）")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figS29_monthly_error_box.png"))
bbox_check(fig, "figS29")
plt.close(fig)

print("saved figS28_forecast_residuals.png, figS29_monthly_error_box.png")
