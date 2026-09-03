# -*- coding: utf-8 -*-
"""plot_country_panels_v2.py — 代表国家分析图 (v2 口径)
输入: country_scenarios_v2.npz + country_scenarios_v2_meta.json
输出 (300 dpi, 新增不覆盖):
  figure_monthly_revised/supplementary/figS11_country_trajectories_v2.png
      六国验证期: 观测熵 / 基线预测 / 众数组合情景预测 三线对比
  figure_monthly_revised/supplementary/figS12_country_dose_response_v2.png
      六国单因素 (覆盖率/速度/管控) 剂量-响应曲线
"""
import csv, json, os

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

Z = np.load("country_scenarios_v2.npz", allow_pickle=True)
META = {m["country"]: m for m in
        json.load(open("country_scenarios_v2_meta.json", encoding="utf-8"))}
R2U = {r["country"]: float(r["r2_uncalibrated"]) for r in csv.DictReader(
    open("entropy_forecast_monthly_v2_per_country.csv", encoding="utf-8"))}

COUNTRIES = [str(c) for c in Z["countries6"]]
CN = {c: str(n) for c, n in zip(Z["countries6"], Z["cn_names"])}
ORDER = sorted(range(6), key=lambda k: META[COUNTRIES[k]]["reduction"])

C_OBS, C_BASE, C_BEST = "#333333", "#0072B2", "#009E73"
COL = {"vacc_coverage": "#0072B2", "vacc_speed": "#56B4E9",
       "stringency": "#009E73"}
GN = {"vacc_coverage": "疫苗覆盖率", "vacc_speed": "疫苗接种速度",
      "stringency": "管控严格度"}


def combo_str(cb):
    parts = []
    if cb[0] != 1.0:
        parts.append(f"覆盖率×{cb[0]:g}")
    if cb[1] != 1.0:
        parts.append(f"速度×{cb[1]:g}")
    if cb[2] != 1.0:
        parts.append(f"管控×{cb[2]:g}")
    return " · ".join(parts) if parts else "均不变"


# ---------------------------------------------------------------- figS11
def figS11():
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
    for ax, k in zip(axes.flat, ORDER):
        c = COUNTRIES[k]
        months = [str(m) for m in Z[f"months_{k}"]]
        x = np.arange(len(months))
        yo, yb, ybest = Z[f"yobs_{k}"], Z[f"ybase_{k}"], Z[f"ybest_{k}"]
        ax.fill_between(x, yb, ybest, color=C_BEST, alpha=0.15, lw=0,
                        zorder=1)
        ax.plot(x, yo, color=C_OBS, lw=1.3, marker="o", ms=2.5, zorder=3,
                label="观测平滑熵")
        ax.plot(x, yb, color=C_BASE, lw=1.3, marker="s", ms=2.3, zorder=3,
                label="基线情景预测")
        ax.plot(x, ybest, color=C_BEST, lw=1.3, marker="^", ms=2.5, zorder=3,
                label="最优组合情景预测")
        m = META[c]
        ax.set_title(f"{CN[c]}：月均降 {m['reduction']*100:.1f}%\n"
                     f"{combo_str(m['combo'])}", loc="left", fontsize=7,
                     linespacing=1.4)
        ax.text(0.98, 0.04, f"验证 R²={R2U[c]:.2f}*",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6, color="#666666")
        ax.set_xticks(x)
        ax.set_xticklabels([mm[2:] for mm in months], rotation=45, ha="right")
        ax.margins(y=0.14)
        ax.tick_params(direction="out")
    for ax in axes[:, 0]:
        ax.set_ylabel("月度序列熵")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.text(0.01, 0.978,
             "六个代表国家验证期（2022-08~12）：观测 vs 基线 vs 最优组合情景",
             ha="left", va="top", fontsize=8)
    fig.text(0.99, 0.962,
             "* 基线/情景为未校准集成预测；组合为该国验证期众数组合",
             ha="right", va="top", fontsize=6, color="#666666")
    fig.tight_layout(rect=[0, 0.05, 1, 0.935])
    fig.savefig(os.path.join(OUT, "figS11_country_trajectories_v2.png"),
                dpi=300)
    plt.close(fig)
    print("figS11 done")


# ---------------------------------------------------------------- figS12
def figS12():
    deltas = Z["deltas"] * 100
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
    for ax, k in zip(axes.flat, ORDER):
        c = COUNTRIES[k]
        for g in ("vacc_coverage", "vacc_speed", "stringency"):
            ax.plot(deltas, Z[f"dose_{g}"][:, k] * 100, color=COL[g],
                    lw=1.3, marker="o", ms=2.0, label=GN[g])
        ax.axhline(0, color="#666666", lw=0.8, zorder=1)
        ax.axvline(0, color="#666666", lw=0.5, ls=":", zorder=1)
        ax.set_title(CN[c], loc="left", fontsize=7)
        ax.margins(y=0.14)
        ax.tick_params(direction="out")
    for ax in axes[-1]:
        ax.set_xlabel("因素变化幅度（%）")
    for ax in axes[:, 0]:
        ax.set_ylabel("预测熵相对变化（%）")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("六个代表国家的单因素剂量-响应（该国验证样本均值）",
                 x=0.01, ha="left", fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(os.path.join(OUT, "figS12_country_dose_response_v2.png"),
                dpi=300)
    plt.close(fig)
    print("figS12 done")


# ---------------------------------------------------------------- figS13
def figS13():
    import datetime as dt
    weeks = np.array([dt.date.fromisoformat(str(w)) for w in Z["weeks"]])
    x = np.arange(len(weeks))
    levers = [("coverage", "疫苗覆盖率（每百人总剂次）"),
              ("speed", "疫苗接种速度（每百人日增首剂）"),
              ("stringency", "管控严格度（0–100）")]
    v0 = (dt.date(2022, 8, 1) - weeks[0]).days // 7     # 验证期起
    fig, axes = plt.subplots(6, 3, figsize=(7.2, 9.2), sharex=True)
    for row, k in enumerate(ORDER):
        c = COUNTRIES[k]
        for col, (lk, lname) in enumerate(levers):
            ax = axes[row, col]
            ax.axvspan(v0, len(weeks) - 1, color="#999999", alpha=0.10,
                       lw=0, zorder=1)
            ax.plot(x, Z[f"actual_{k}_{lk}"], color="#333333", lw=1.2,
                    zorder=3, label="实际观测")
            ax.plot(x, Z[f"cf_{k}_{lk}"], color=COL[{
                        "coverage": "vacc_coverage", "speed": "vacc_speed",
                        "stringency": "stringency"}[lk]],
                    lw=1.2, ls="--", zorder=3, label="最优组合反事实")
            ax.margins(y=0.12)
            ax.tick_params(direction="out")
            if row == 0:
                ax.set_title(lname, loc="left", fontsize=7)
            if col == 0:
                ax.set_ylabel(CN[c], fontsize=7)
    for ax in axes[-1]:
        ax.set_xticks(x[::26])
        ax.set_xticklabels([str(weeks[i])[:7] for i in x[::26]],
                           rotation=45, ha="right")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.text(0.01, 0.985,
             "六国三因素实际轨迹 vs 最优组合反事实轨迹（灰带=验证期 2022-08~12）",
             ha="left", va="top", fontsize=8)
    fig.text(0.99, 0.968,
             "反事实：全时段施加该国众数组合扰动（含约束钳制与速度→覆盖率联动）",
             ha="right", va="top", fontsize=6, color="#666666")
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    fig.savefig(os.path.join(OUT, "figS13_country_levers_v2.png"), dpi=300)
    plt.close(fig)
    print("figS13 done")


if __name__ == "__main__":
    figS11()
    figS12()
    figS13()
    print("DONE ->", OUT)
