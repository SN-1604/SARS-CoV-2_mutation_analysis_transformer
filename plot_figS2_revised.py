# -*- coding: utf-8 -*-
"""plot_figS2_revised.py — 分国家验证 R² 分布图 (figS2)
输出: figure_monthly_revised/supplementary/figS2_per_country_r2_v2.png  (106 国口径)
直方图 + 累积分布, 增加 R²=0.8 达标参考线。运行: python plot_figS2_revised.py
"""
import csv, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.linewidth": 0.6, "axes.labelpad": 4,
})

OUT = "figure_monthly_revised/supplementary"
os.makedirs(OUT, exist_ok=True)
GOOD, BAD, NEU = "#0072B2", "#D55E00", "#666666"


def load_r2(path, only_kept):
    r2, n_excl = [], 0
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if only_kept and r.get("excluded_eval") == "True":
            n_excl += 1
            continue
        if r["r2"]:
            r2.append(float(r["r2"]))
    return np.array(r2), n_excl


def figS2(r2, name, title_left, note=None):
    n = len(r2)
    med = float(np.median(r2))
    q25, q75 = np.percentile(r2, [25, 75])
    ge08 = float((r2 >= 0.8).mean()) * 100
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]
    ax.hist(r2, bins=np.linspace(-1, 1, 41), color=GOOD, edgecolor="white",
            lw=0.4)
    ax.axvline(med, color=BAD, lw=0.9, ls="--")
    ax.axvline(0.8, color=NEU, lw=0.7, ls=":")
    ax.text(med + 0.02, ax.get_ylim()[1] * 0.92, f"中位数 {med:.2f}",
            fontsize=6, color=BAD)
    ax.text(0.03, 0.72, f"R²≥0.8 达标: {ge08:.0f}%",
            transform=ax.transAxes, fontsize=6, color=NEU, va="top")
    ax.set_xlabel("分国家验证 R²")
    ax.set_ylabel("国家数")
    ax.set_title(title_left, loc="left")
    ax = axes[1]
    xs = np.sort(r2)
    n_clip = int((xs < -1).sum())
    ax.plot(np.clip(xs, -1, None), np.arange(1, n + 1) / n * 100,
            color=GOOD, lw=1.2)
    for q, v in [(25, q25), (50, med), (75, q75)]:
        vc = max(v, -1)
        ax.plot([vc, vc, -1], [0, q, q], color=NEU, lw=0.6, ls=":")
        ax.plot([vc], [q], marker="o", ms=3, color=BAD)
        ax.annotate(f"Q{q}={v:.2f}", (vc, q), textcoords="offset points",
                    xytext=(6, -8), fontsize=6, color=NEU)
    ax.axvline(0.8, color=NEU, lw=0.7, ls=":")
    ax.set_xlim(-1.05, 1.05)
    note_lines = []
    if n_clip:
        note_lines.append(f"{n_clip} 国 R²<-1 未显示")
    if note:
        note_lines.append(note)
    if note_lines:
        ax.text(0.02, 0.97, "\n".join(note_lines), transform=ax.transAxes,
                va="top", fontsize=6, color=NEU)
    ax.set_xlabel("分国家验证 R²")
    ax.set_ylabel("累积国家比例 (%)")
    ax.set_ylim(0, 102)
    ax.set_title("国家间预测精度差异大", loc="left")
    for a in axes:
        a.tick_params(direction="out")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name, flush=True)


def main():
    r2_v2, _ = load_r2("entropy_forecast_monthly_v2_per_country.csv",
                       only_kept=False)
    med2, q752 = np.median(r2_v2), np.percentile(r2_v2, 75)
    figS2(r2_v2, "figS2_per_country_r2_v2.png",
          f"半数国家 R²>{med2:.2f}, 1/4 国家 R²>{q752:.2f}"
          f"（n={len(r2_v2)}）")


if __name__ == "__main__":
    main()
