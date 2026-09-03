# -*- coding: utf-8 -*-
"""plot_supplementary_figures_v2.py — 最终模型与事后解释的补充可视化
输出: figure_monthly_revised/supplementary/figS1..figS8 (300 dpi)
全部图表基于全样本/全国家统计, 不挑选个别代表国家。
遵循 figure-style 规范, 视觉语言与 plot_explanation_figures_v2.py 一致。
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

MET = json.load(open("entropy_forecast_monthly_v2_metrics.json", encoding="utf-8"))
FE = json.load(open("factor_effects_monthly_v2.json", encoding="utf-8"))
PC = json.load(open("policy_combination_results_v2.json", encoding="utf-8"))
RECS = list(csv.DictReader(open("policy_recommendations_v2.csv", encoding="utf-8")))
VP = np.load("val_pred_monthly_v2.npz", allow_pickle=True)

CN = {"vacc_coverage": "疫苗覆盖率", "vacc_speed": "疫苗接种速度",
      "stringency": "管控严格度", "cases": "每日净增病例数", "deaths": "死亡数",
      "reproduction_rate": "传播率Rt*"}
RT_NOTE = "* 传播率Rt为领先病例水平 2–3 周的估计值"
COL = {"vacc_coverage": "#0072B2", "vacc_speed": "#56B4E9",
       "stringency": "#009E73", "cases": "#E69F00", "deaths": "#D55E00",
       "reproduction_rate": "#CC79A7"}
GOOD, BAD, NEU = "#0072B2", "#D55E00", "#666666"

MONTHS = [f"{y:04d}-{m:02d}" for y in (2020, 2021, 2022)
          for m in range(1, 13) if (y, m) >= (2020, 1) and (y, m) <= (2022, 12)]
SPLIT = "2022-08"                     # 训练/验证分界 (目标月 >= 2022-08 为验证)


def split_vline(ax):
    x = MONTHS.index(SPLIT) - 0.5
    ax.axvline(x, color=NEU, lw=0.8, ls="--", zorder=1)
    ax.text(x, 1.0, " 训练 | 验证 ", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=6, color=NEU)


def month_ticks(ax, step=6):
    idx = list(range(0, len(MONTHS), step))
    ax.set_xticks(idx)
    ax.set_xticklabels([MONTHS[i] for i in idx], rotation=45, ha="right")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name, flush=True)


def load_entropy_all(dir_):
    """返回 {month: [各国熵值]} —— 全国家全月份"""
    by_month = {m: [] for m in MONTHS}
    n_c = 0
    for f in sorted(os.listdir(dir_)):
        if not f.endswith("_monthly_entropy.npy"):
            continue
        ent = np.load(os.path.join(dir_, f), allow_pickle=True).item()
        n_c += 1
        for m in MONTHS:
            if m in ent:
                by_month[m].append(float(ent[m]))
    return by_month, n_c


# ---------------------------------------------------------------- figS1
def figS1():
    """v2 模型性能: [全模型未校准, 全模型静态校准, 纯熵消融, 持续性基线]
    左 R² / 右 MAE (106 国口径)"""
    import train_entropy_forecast_monthly as T1
    rows = [("全模型\n未校准", MET["val_full_ensemble"]["r2"],
             MET["val_full_ensemble"]["mae"]),
            ("全模型\n静态校准", MET["val_full_ensemble_calibrated"]["r2"],
             MET["val_full_ensemble_calibrated"]["mae"]),
            ("纯熵消融\n未校准", MET["val_entonly_ensemble"]["r2"],
             MET["val_entonly_ensemble"]["mae"])]
    T1.ENT_DIR = "entropy_monthly_smoothed"
    Xe_, Xh_, Xm_, y_, cs_, ts_, countries_ = T1.build_samples(3)
    va_ = ts_ > T1.TRAIN_END
    base = []
    for i in np.nonzero(va_)[0]:
        v = 0.0
        for j in range(Xm_.shape[1] - 1, -1, -1):
            if Xm_[i, j] > 0:
                v = Xh_[i, j]
                break
        base.append(v)
    base = np.array(base)
    yv = y_[va_]
    r2b = 1 - ((base - yv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    maeb = float(np.abs(base - yv).mean())
    rows.append(("持续性基线", float(r2b), maeb))

    labels = [r[0] for r in rows]
    r2 = [r[1] for r in rows]
    mae = [r[2] for r in rows]
    x = np.arange(len(rows))
    colors1 = ["#56B4E9", "#0072B2", "#E69F00", "#999999"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    ax.bar(x, r2, color=colors1, width=0.62)
    for i, v in enumerate(r2):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("验证集 R²")
    ax.set_title(f"集成模型 R²={MET['val_full_ensemble_calibrated']['r2']:.3f}, "
                 f"表征贡献 {MET['emb_contribution']*100:.1f}%"
                 f"（n={MET['n_val']}, 106 国）", loc="left")
    ax.margins(y=0.15)
    ax = axes[1]
    ax.bar(x, mae, color=colors1, width=0.62)
    for i, v in enumerate(mae):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("验证集 MAE")
    ax.set_title("静态校准与表征引入同步降低误差", loc="left")
    ax.margins(y=0.15)
    for a in axes:
        a.tick_params(direction="out")
    fig.tight_layout()
    save(fig, "figS1_model_performance.png")


# ---------------------------------------------------------------- figS2
def figS2():
    r2 = np.array([float(r["r2"]) for r in PC_R2])
    n = len(r2)
    med = float(np.median(r2))
    q25, q75 = np.percentile(r2, [25, 75])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]
    ax.hist(r2, bins=np.linspace(-1, 1, 41), color="#0072B2", edgecolor="white",
            lw=0.4)
    ax.axvline(med, color=BAD, lw=0.9, ls="--")
    ax.text(med + 0.02, ax.get_ylim()[1] * 0.92, f"中位数 {med:.2f}",
            fontsize=6, color=BAD)
    ax.set_xlabel("分国家验证 R²")
    ax.set_ylabel("国家数")
    ax.set_title(f"半数国家 R²>0.4, 1/4 国家 R²>{q75:.2f}（n={n}）", loc="left")
    ax = axes[1]
    xs = np.sort(r2)
    n_clip = int((xs < -1).sum())
    ax.plot(np.clip(xs, -1, None), np.arange(1, n + 1) / n * 100,
            color="#0072B2", lw=1.2)
    for q, v in [(25, q25), (50, med), (75, q75)]:
        vc = max(v, -1)
        ax.plot([vc, vc, -1], [0, q, q], color=NEU, lw=0.6, ls=":")
        ax.plot([vc], [q], marker="o", ms=3, color=BAD)
        ax.annotate(f"Q{q}={v:.2f}", (vc, q), textcoords="offset points",
                    xytext=(6, -8), fontsize=6, color=NEU)
    ax.set_xlim(-1.05, 1.05)
    if n_clip:
        ax.text(0.02, 0.97, f"{n_clip} 国 R²<-1 未显示", transform=ax.transAxes,
                va="top", fontsize=6, color=NEU)
    ax.set_xlabel("分国家验证 R²")
    ax.set_ylabel("累积国家比例 (%)")
    ax.set_ylim(0, 102)
    ax.set_title("国家间预测精度差异大", loc="left")
    for a in axes:
        a.tick_params(direction="out")
    fig.tight_layout()
    save(fig, "figS2_per_country_r2.png")


# ---------------------------------------------------------------- figS3
def figS3():
    raw_d, n_raw = load_entropy_all("entropy_monthly")
    smo_d, n_smo = load_entropy_all("entropy_monthly_smoothed")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, d, n_c, tag in [(axes[0], raw_d, n_raw, "原始月度熵"),
                            (axes[1], smo_d, n_smo, "平滑月度熵")]:
        med, lo, hi, lo2, hi2 = [], [], [], [], []
        for m in MONTHS:
            v = np.array(d[m]) if d[m] else np.array([np.nan])
            med.append(np.nanmedian(v))
            q = np.nanpercentile(v, [5, 25, 75, 95])
            lo2.append(q[0]); lo.append(q[1]); hi.append(q[2]); hi2.append(q[3])
        x = np.arange(len(MONTHS))
        ax.fill_between(x, lo2, hi2, color="#0072B2", alpha=0.12, lw=0,
                        label="5–95 百分位")
        ax.fill_between(x, lo, hi, color="#0072B2", alpha=0.28, lw=0,
                        label="25–75 百分位")
        ax.plot(x, med, color="#0072B2", lw=1.3, label="中位数")
        split_vline(ax)
        month_ticks(ax)
        ax.set_title(f"{tag}的跨国分布演变（每月 n≤{n_c} 国）", loc="left")
        ax.set_xlabel("月份")
        ax.margins(y=0.10)
    axes[0].set_ylabel("月度香农熵")
    axes[1].legend(frameon=False, loc="upper left", fontsize=6)
    for a in axes:
        a.tick_params(direction="out")
    fig.tight_layout()
    save(fig, "figS3_entropy_evolution.png")


# ---------------------------------------------------------------- figS4
def figS4():
    yt, yp = VP["y_true"], np.clip(VP["y_pred"], 0, None)
    months = VP["months"].astype(str)
    n = len(yt)
    ss_res = float(((yp - yt) ** 2).sum())
    r2 = 1 - ss_res / float(((yt - yt.mean()) ** 2).sum())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    ax.scatter(yt, yp, s=5, color="#0072B2", alpha=0.45, lw=0)
    lim = [0, max(yt.max(), yp.max()) * 1.03]
    ax.plot(lim, lim, color=NEU, lw=0.8, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("观测熵（平滑）")
    ax.set_ylabel("预测熵")
    ax.text(0.03, 0.97, f"R²={r2:.3f}\nn={n}", transform=ax.transAxes,
            va="top", fontsize=6, color=NEU)
    ax.set_title("验证集预测与观测总体一致", loc="left")
    ax = axes[1]
    vmonths = [m for m in MONTHS if m >= SPLIT]
    x = np.arange(len(vmonths))
    med_t, med_p, lo_t, hi_t = [], [], [], []
    for m in vmonths:
        sel = months == m
        med_t.append(np.median(yt[sel])); med_p.append(np.median(yp[sel]))
        q = np.percentile(yt[sel], [25, 75])
        lo_t.append(q[0]); hi_t.append(q[1])
    ax.fill_between(x, lo_t, hi_t, color="#0072B2", alpha=0.2, lw=0,
                    label="观测 25–75 百分位")
    ax.plot(x, med_t, color="#0072B2", lw=1.3, marker="o", ms=2.5,
            label="观测中位数")
    ax.plot(x, med_p, color="#D55E00", lw=1.3, marker="s", ms=2.5,
            label="预测中位数")
    ax.set_xticks(x)
    ax.set_xticklabels(vmonths, rotation=45, ha="right")
    ax.set_ylabel("月度熵（跨国中位）")
    ax.set_title("逐月跨国中位水平的预测跟踪", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    ax.margins(y=0.10)
    for a in axes:
        a.tick_params(direction="out")
    fig.tight_layout()
    save(fig, "figS4_obs_vs_pred.png")


# ---------------------------------------------------------------- figS5
def figS5():
    deltas = np.array(FE["deltas"], dtype=float) * 100
    order = ["vacc_coverage", "vacc_speed", "stringency",
             "cases", "deaths", "reproduction_rate"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True)
    for ax, g in zip(axes.flat, order):
        raw = FE["raw"][g]
        y = np.array([raw[f"{d:.1f}"]["mean_rel"] for d in FE["deltas"]]) * 100
        ax.axhline(0, color=NEU, lw=0.8, zorder=1)
        ax.axvline(0, color=NEU, lw=0.5, ls=":", zorder=1)
        ax.plot(deltas, y, color=COL[g], lw=1.3, marker="o", ms=2.2)
        ax.set_title(CN[g], loc="left", color=COL[g])
        ax.margins(y=0.12)
        ax.tick_params(direction="out")
    for ax in axes[-1]:
        ax.set_xlabel("因素变化幅度 (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("预测熵相对变化 (%)")
    fig.suptitle("各因素全档位剂量-响应曲线（n=477 验证样本均值）",
                 x=0.01, ha="left", fontsize=8)
    fig.text(0.99, 0.005, RT_NOTE, ha="right", fontsize=6, color=NEU)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save(fig, "figS5_full_dose_response.png")


# ---------------------------------------------------------------- figS6
def figS6():
    nr = PC["speed_reachability"]["n_reachable"]
    tiers = sorted(nr, key=float)
    counts = [nr[t] for t in tiers]
    total = len(PC["speed_reachability"]["highest_reachable"])
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    bars = ax.bar([f"×{t}" for t in tiers], counts, color="#56B4E9", width=0.6)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 2,
                f"{c} ({c/total*100:.0f}%)", ha="center", va="bottom",
                fontsize=6)
    ax.set_ylabel("可达国家数")
    ax.set_xlabel("疫苗接种速度档位（每日净增 ≤1 剂/百人约束下）")
    ax.set_title(f"速度档位越高可达国家越少（共 {total} 国）", loc="left",
                 pad=10)
    ax.set_ylim(0, max(counts) * 1.22)
    ax.tick_params(direction="out")
    fig.tight_layout()
    save(fig, "figS6_speed_reachability.png")


# ---------------------------------------------------------------- figS7
def figS7():
    cov = np.array([float(r["best_coverage"]) for r in RECS])
    spd = np.array([float(r["best_speed"]) for r in RECS])
    stg = np.array([float(r["best_stringency"]) for r in RECS])
    red = np.array([float(r["mean_reduction"]) for r in RECS]) * 100
    n = len(RECS)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    panels = [(cov, PC["grid"]["coverage"], "最优疫苗覆盖率倍数", COL["vacc_coverage"]),
              (spd, PC["grid"]["speed"], "最优疫苗接种速度倍数", COL["vacc_speed"]),
              (stg, PC["grid"]["stringency"], "最优管控严格度倍数", COL["stringency"])]
    for ax, (v, grid, lab, c) in zip(axes.flat[:3], panels):
        cnt = [int((v == g).sum()) for g in grid]
        ax.bar([f"×{g:g}" for g in grid], cnt, color=c, width=0.6)
        for i, cc in enumerate(cnt):
            if cc:
                ax.text(i, cc + 0.5, str(cc), ha="center", va="bottom",
                        fontsize=6)
        ax.set_ylabel("国家数")
        ax.set_xlabel(lab)
        ax.margins(y=0.15)
    mode_cov = PC["grid"]["coverage"][int(np.argmax([(cov == g).sum() for g in PC["grid"]["coverage"]]))]
    mode_spd = PC["grid"]["speed"][int(np.argmax([(spd == g).sum() for g in PC["grid"]["speed"]]))]
    mode_stg = PC["grid"]["stringency"][int(np.argmax([(stg == g).sum() for g in PC["grid"]["stringency"]]))]
    axes.flat[0].set_title(f"最优覆盖率以 ×{mode_cov:g} 档最多（n={n} 国）", loc="left")
    axes.flat[1].set_title(f"最优速度以 ×{mode_spd:g} 档最多", loc="left")
    axes.flat[2].set_title(f"最优管控以 ×{mode_stg:g} 档最多", loc="left")
    ax = axes.flat[3]
    ax.hist(red, bins=24, color="#0072B2", edgecolor="white", lw=0.4)
    med = float(np.median(red))
    ax.axvline(med, color=BAD, lw=0.9, ls="--")
    ax.text(med + 0.2, ax.get_ylim()[1] * 0.9, f"中位数 {med:.1f}%",
            fontsize=6, color=BAD)
    ax.set_xlabel("最优组合下预测熵降幅 (%)")
    ax.set_ylabel("国家数")
    ax.set_title("多数国家可降低预测熵", loc="left")
    for a in axes.flat:
        a.tick_params(direction="out")
    fig.tight_layout()
    save(fig, "figS7_combo_distribution.png")


# ---------------------------------------------------------------- figS8
def figS8():
    base = np.array([float(r["baseline_pred"]) for r in RECS])
    red = np.array([float(r["mean_reduction"]) for r in RECS]) * 100
    names = [r["country"] for r in RECS]
    n = len(RECS)
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.scatter(base, red, s=8, color="#0072B2", alpha=0.55, lw=0)
    idx = {lab: i for lab, i in
           [("降幅最大", int(np.argmax(red))), ("降幅最小", int(np.argmin(red))),
            ("基线最高", int(np.argmax(base))), ("基线最低", int(np.argmin(base)))]}
    seen = set()
    x_hi = np.percentile(base, 90)
    y_mid = np.median(red)
    for lab, i in idx.items():
        if i in seen:
            continue
        seen.add(i)
        right = base[i] > x_hi          # 靠右的点文字放左侧
        val = f"{red[i]:.1f}%" if "降幅" in lab else f"{base[i]:.1f}"
        ax.annotate(f"{names[i]}\n({lab}: {val})",
                    (base[i], red[i]), textcoords="offset points",
                    xytext=(-8 if right else 10,
                            -16 if red[i] > y_mid else 10),
                    ha="right" if right else "left",
                    va="top" if red[i] > y_mid else "bottom",
                    fontsize=6, color=NEU,
                    arrowprops=dict(arrowstyle="-", color=NEU, lw=0.5))
    ax.set_xlabel("基线预测熵（真实策略下）")
    ax.set_ylabel("最优组合熵降幅 (%)")
    corr = float(np.corrcoef(base, red)[0, 1])
    trend = "基线熵越高, 可挖潜的降幅越大" if corr > 0.1 else         ("基线熵越高, 可挖潜空间越小" if corr < -0.1 else "降幅与基线熵无明显单调关系")
    ax.set_title(f"{trend}（n={n} 国, r={corr:.2f}）", loc="left")
    ax.margins(0.06)
    ax.tick_params(direction="out")
    fig.tight_layout()
    save(fig, "figS8_reduction_vs_baseline.png")


if __name__ == "__main__":
    figS1(); figS3(); figS4(); figS5(); figS6(); figS7(); figS8()  # figS2 见 plot_figS2_revised.py
